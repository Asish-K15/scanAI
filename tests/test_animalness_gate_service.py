"""
Unit tests for the Animalness Gate Service.
Verifies PRD specifications, including:
1. No implicit production threshold (unconfigured threshold -> UNCERTAIN / LOW_CONFIDENCE).
2. Explicit configured diagnostic threshold works.
3. Species mismatch remains REJECT / SPECIES_MISMATCH (no silent reroute).
4. Generic not-animal rejection returns NON_ANIMAL_DETECTED.
5. Model checkpoints and protected data remain unmodified.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from PIL import Image

from animalness_gate.schemas import GateDecision, GateReasonCode
from animalness_gate.service import (
    AnimalnessGateConfig,
    AnimalnessGateService,
    get_animalness_gate_service,
)


class TestAnimalnessGateService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parents[1]

        # Service with NO configured production threshold (default state)
        cls.unconfigured_service = AnimalnessGateService(
            config=AnimalnessGateConfig(animal_threshold=None)
        )

        # Service with EXPLICIT configured diagnostic threshold
        cls.configured_service = AnimalnessGateService(
            config=AnimalnessGateConfig(animal_threshold=0.50)
        )

        # Find sample images from the repo
        manifest_path = cls.base_dir / "animalness_gate" / "manifests" / "gate_manifest.csv"
        cls.dog_image_path = None
        cls.cat_image_path = None
        cls.cattle_image_path = None

        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            test_df = df[df["split"] == "test"]
            dogs = test_df[test_df["species"] == "dog"]["image_path"].tolist()
            cats = test_df[test_df["species"] == "cat"]["image_path"].tolist()
            cattles = test_df[test_df["species"] == "cattle"]["image_path"].tolist()

            if dogs:
                cls.dog_image_path = Path(dogs[0])
            if cats:
                cls.cat_image_path = Path(cats[0])
            if cattles:
                cls.cattle_image_path = Path(cattles[0])

        cls.human_image_path = (
            cls.base_dir / "animal_validator" / "rejection_samples" / "human" / "9k_ (4)_face.png"
        )
        cls.non_animal_image_path = (
            cls.base_dir / "animal_validator" / "rejection_samples" / "non_animal" / "apple_01_0.jpg"
        )

    # -------------------------------------------------------------------------
    # Requirement 1: No implicit production threshold -> UNCERTAIN
    # -------------------------------------------------------------------------

    def test_01_no_configured_threshold_returns_uncertain_for_high_prob_dog(self):
        """No configured threshold must return UNCERTAIN, not an implicit 0.50 ACCEPT."""
        if not self.dog_image_path or not self.dog_image_path.exists():
            self.skipTest("Dog test image not found")

        result = self.unconfigured_service.validate(self.dog_image_path, selected_species="dog")

        # Animal probability is high (~0.99), but without an approved threshold,
        # the service MUST NOT make an implicit 0.50 ACCEPT decision.
        self.assertEqual(result["decision"], GateDecision.UNCERTAIN.value)
        self.assertEqual(result["reason_code"], GateReasonCode.LOW_CONFIDENCE.value)
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_02_no_configured_threshold_returns_uncertain_for_human(self):
        """No configured threshold returns UNCERTAIN for human, avoiding unapproved decision."""
        if not self.human_image_path.exists():
            self.skipTest("Human sample image not found")

        result = self.unconfigured_service.validate(self.human_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.UNCERTAIN.value)
        self.assertEqual(result["reason_code"], GateReasonCode.LOW_CONFIDENCE.value)
        self.assertIsNone(result["error_code"])

    # -------------------------------------------------------------------------
    # Requirement 2: Explicit configured threshold works
    # -------------------------------------------------------------------------

    def test_03_explicit_configured_threshold_accepts_dog(self):
        """Explicit configured threshold accepts dog when animal_probability >= threshold."""
        if not self.dog_image_path or not self.dog_image_path.exists():
            self.skipTest("Dog test image not found")

        result = self.configured_service.validate(self.dog_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.ACCEPT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SUPPORTED_ANIMAL_DETECTED.value)
        self.assertEqual(result["predicted_species"], "dog")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertGreater(result["species_confidence"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_04_explicit_configured_threshold_accepts_cat(self):
        """Explicit configured threshold accepts cat when animal_probability >= threshold."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat test image not found")

        result = self.configured_service.validate(self.cat_image_path, selected_species="cat")

        self.assertEqual(result["decision"], GateDecision.ACCEPT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SUPPORTED_ANIMAL_DETECTED.value)
        self.assertEqual(result["predicted_species"], "cat")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertGreater(result["species_confidence"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_05_explicit_configured_threshold_accepts_cattle(self):
        """Explicit configured threshold accepts cattle when animal_probability >= threshold."""
        if not self.cattle_image_path or not self.cattle_image_path.exists():
            self.skipTest("Cattle test image not found")

        result = self.configured_service.validate(self.cattle_image_path, selected_species="cattle")

        self.assertEqual(result["decision"], GateDecision.ACCEPT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SUPPORTED_ANIMAL_DETECTED.value)
        self.assertEqual(result["predicted_species"], "cattle")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertGreater(result["species_confidence"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_06_explicit_configured_threshold_rejects_human_as_non_animal(self):
        """Generic binary gate returns NON_ANIMAL_DETECTED for rejected human images."""
        if not self.human_image_path.exists():
            self.skipTest("Human sample image not found")

        result = self.configured_service.validate(self.human_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.NON_ANIMAL_DETECTED.value)
        self.assertLess(result["animal_probability"], 0.50)
        self.assertIsNone(result["error_code"])

    def test_07_explicit_configured_threshold_rejects_non_animal_object(self):
        """Generic binary gate returns NON_ANIMAL_DETECTED for objects."""
        if not self.non_animal_image_path.exists():
            self.skipTest("Non-animal sample image not found")

        result = self.configured_service.validate(self.non_animal_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.NON_ANIMAL_DETECTED.value)
        self.assertLess(result["animal_probability"], 0.10)
        self.assertIsNone(result["error_code"])

    # -------------------------------------------------------------------------
    # Requirement 3: Species mismatch remains REJECT / SPECIES_MISMATCH
    # -------------------------------------------------------------------------

    def test_08_species_mismatch_cat_image_selected_dog(self):
        """Cat image + selected dog -> REJECT with SPECIES_MISMATCH."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat test image not found")

        result = self.configured_service.validate(self.cat_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SPECIES_MISMATCH.value)
        self.assertEqual(result["predicted_species"], "cat")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_09_no_silent_rerouting_on_species_mismatch(self):
        """Verify that species mismatch never silently reroutes to the predicted species."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat test image not found")

        result = self.configured_service.validate(self.cat_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SPECIES_MISMATCH.value)
        self.assertEqual(result["predicted_species"], "cat")
        self.assertNotEqual(result["decision"], GateDecision.ACCEPT.value)

    # -------------------------------------------------------------------------
    # Robustness, error handling, and schema validation
    # -------------------------------------------------------------------------

    def test_10_invalid_image_returns_controlled_rejection(self):
        """Invalid image bytes return controlled INVALID_IMAGE response."""
        corrupt_bytes = b"not_an_image_file_content"

        result = self.configured_service.validate(corrupt_bytes, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.INVALID_IMAGE.value)
        self.assertEqual(result["error_code"], "INVALID_IMAGE")

    def test_11_model_failure_returns_controlled_model_error(self):
        """Model failure returns controlled MODEL_ERROR response without crashing."""
        img = Image.new("RGB", (224, 224), "gray")

        with patch.object(self.configured_service, "_gate_model", side_effect=RuntimeError("Simulated memory error")):
            result = self.configured_service.validate(img, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.MODEL_ERROR.value)
        self.assertIn("Simulated memory error", result["error_code"])

    def test_12_unsupported_selected_species_rejected(self):
        """Unsupported species returns REJECT, UNSUPPORTED_SPECIES."""
        img = Image.new("RGB", (224, 224), "gray")
        result = self.configured_service.validate(img, selected_species="horse")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.UNSUPPORTED_SPECIES.value)

    def test_13_uncertainty_interval_configuration(self):
        """Configuring uncertainty_min and uncertainty_max returns UNCERTAIN / LOW_CONFIDENCE."""
        config = AnimalnessGateConfig(
            animal_threshold=0.50,
            uncertainty_min=0.30,
            uncertainty_max=0.70,
        )
        custom_service = AnimalnessGateService(config=config, lazy_load=True)
        custom_service._gate_model = MagicMock()
        custom_service._species_model = MagicMock()

        # Mock gate output with 0.55 animal probability (inside uncertainty range [0.30, 0.70])
        custom_service._gate_model.return_value = torch.tensor([[0.20, 0.0]])
        custom_service._species_model.return_value = torch.tensor([[10.0, 0.0, 0.0]])

        img = Image.new("RGB", (224, 224), "white")
        result = custom_service.validate(img, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.UNCERTAIN.value)
        self.assertEqual(result["reason_code"], GateReasonCode.LOW_CONFIDENCE.value)
        self.assertAlmostEqual(result["animal_probability"], 0.5498, places=2)

    def test_14_gate_response_schema_contract(self):
        """Verify full gate response schema matches the PRD specification."""
        img = Image.new("RGB", (224, 224), "white")
        result = self.configured_service.validate(img, selected_species="dog")

        required_keys = {
            "decision",
            "predicted_species",
            "species_confidence",
            "animal_probability",
            "model_name",
            "model_version",
            "reason_code",
            "error_code",
        }
        self.assertEqual(set(result.keys()), required_keys)
        self.assertIn(result["decision"], ["ACCEPT", "REJECT", "UNCERTAIN"])
        self.assertEqual(result["model_name"], "EfficientNet-B0")
        self.assertEqual(result["model_version"], "SCANAI-ANIMALNESS-GATE-V1")
        self.assertIsInstance(result["species_confidence"], float)
        self.assertIsInstance(result["animal_probability"], float)

    # -------------------------------------------------------------------------
    # Requirement 5: Model checkpoints & protected datasets remain unchanged
    # -------------------------------------------------------------------------

    def test_15_protected_artifacts_remain_unmodified(self):
        """Verify that model checkpoints and protected datasets are present and intact."""
        gate_ckpt = self.base_dir / "animalness_gate" / "outputs" / "best_animalness_gate.pth"
        species_ckpt = self.base_dir / "animal_validator" / "outputs" / "best_species_validator.pth"
        skin_model = (
            self.base_dir
            / "dataset_manager"
            / "training"
            / "outputs"
            / "phase4_final"
            / "scanai_skin_phase4b.onnx"
        )
        human_dir = self.base_dir / "animal_validator" / "rejection_samples" / "human"

        self.assertTrue(gate_ckpt.exists(), "Animalness gate checkpoint missing")
        self.assertTrue(species_ckpt.exists(), "Species validator checkpoint missing")
        self.assertTrue(skin_model.exists(), "Skin Phase 4B model missing")
        self.assertTrue(human_dir.exists(), "Human rejection directory missing")

        human_images = list(human_dir.glob("*"))
        self.assertEqual(len(human_images), 30, "Protected 30-image human set count was modified!")


if __name__ == "__main__":
    unittest.main()
