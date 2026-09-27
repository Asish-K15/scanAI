"""
Unit tests for the Animalness Gate Service.
Verifies all contract specifications and test requirements from the PRD.
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
        cls.service = get_animalness_gate_service()

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

        cls.human_image_path = cls.base_dir / "animal_validator" / "rejection_samples" / "human" / "9k_ (4)_face.png"
        cls.non_animal_image_path = cls.base_dir / "animal_validator" / "rejection_samples" / "non_animal" / "apple_01_0.jpg"

    def test_01_human_image_rejected(self):
        """1. Human image -> REJECT."""
        if not self.human_image_path.exists():
            self.skipTest("Human sample image not found")

        result = self.service.validate(self.human_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertIn(
            result["reason_code"],
            [GateReasonCode.NON_ANIMAL_DETECTED.value, GateReasonCode.HUMAN_DETECTED.value],
        )
        self.assertLess(result["animal_probability"], 0.50)
        self.assertIsNone(result["error_code"])

    def test_02_non_animal_image_rejected(self):
        """2. Non-animal image -> REJECT."""
        if not self.non_animal_image_path.exists():
            self.skipTest("Non-animal sample image not found")

        result = self.service.validate(self.non_animal_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.NON_ANIMAL_DETECTED.value)
        self.assertLess(result["animal_probability"], 0.10)
        self.assertIsNone(result["error_code"])

    def test_03_dog_plus_selected_dog_accepted(self):
        """3. Dog + selected dog -> ACCEPT."""
        if not self.dog_image_path or not self.dog_image_path.exists():
            self.skipTest("Dog test image not found")

        result = self.service.validate(self.dog_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.ACCEPT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SUPPORTED_ANIMAL_DETECTED.value)
        self.assertEqual(result["predicted_species"], "dog")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertGreater(result["species_confidence"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_04_cat_plus_selected_cat_accepted(self):
        """4. Cat + selected cat -> ACCEPT."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat test image not found")

        result = self.service.validate(self.cat_image_path, selected_species="cat")

        self.assertEqual(result["decision"], GateDecision.ACCEPT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SUPPORTED_ANIMAL_DETECTED.value)
        self.assertEqual(result["predicted_species"], "cat")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertGreater(result["species_confidence"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_05_cattle_plus_selected_cattle_accepted(self):
        """5. Cattle + selected cattle -> ACCEPT."""
        if not self.cattle_image_path or not self.cattle_image_path.exists():
            self.skipTest("Cattle test image not found")

        result = self.service.validate(self.cattle_image_path, selected_species="cattle")

        self.assertEqual(result["decision"], GateDecision.ACCEPT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SUPPORTED_ANIMAL_DETECTED.value)
        self.assertEqual(result["predicted_species"], "cattle")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertGreater(result["species_confidence"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_06_cat_image_plus_selected_dog_rejected_species_mismatch(self):
        """6. Cat image + selected dog -> REJECT, SPECIES_MISMATCH."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat test image not found")

        result = self.service.validate(self.cat_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SPECIES_MISMATCH.value)
        self.assertEqual(result["predicted_species"], "cat")
        self.assertGreater(result["animal_probability"], 0.90)
        self.assertIsNone(result["error_code"])

    def test_07_invalid_image_returns_controlled_rejection(self):
        """7. Invalid image -> controlled rejection/error."""
        corrupt_bytes = b"not_an_image_file_content"

        result = self.service.validate(corrupt_bytes, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.INVALID_IMAGE.value)
        self.assertEqual(result["error_code"], "INVALID_IMAGE")

    def test_08_model_failure_returns_controlled_model_error(self):
        """8. Model failure -> controlled MODEL_ERROR response."""
        img = Image.new("RGB", (224, 224), "gray")

        with patch.object(self.service, "_gate_model", side_effect=RuntimeError("CUDA OOM or memory error")):
            result = self.service.validate(img, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.MODEL_ERROR.value)
        self.assertIn("CUDA OOM", result["error_code"])

    def test_09_no_silent_rerouting_on_species_mismatch(self):
        """9. Verify that species mismatch never silently reroutes."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat test image not found")

        result = self.service.validate(self.cat_image_path, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SPECIES_MISMATCH.value)
        self.assertEqual(result["predicted_species"], "cat")
        self.assertNotEqual(result["decision"], GateDecision.ACCEPT.value)

    def test_10_unsupported_selected_species_rejected(self):
        """Unsupported species -> REJECT, UNSUPPORTED_SPECIES."""
        img = Image.new("RGB", (224, 224), "gray")
        result = self.service.validate(img, selected_species="horse")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.UNSUPPORTED_SPECIES.value)

    def test_11_human_with_hint_returns_human_detected(self):
        """Validate HUMAN_DETECTED reason code when hint is present."""
        if not self.human_image_path.exists():
            self.skipTest("Human sample image not found")

        result = self.service.validate(self.human_image_path, selected_species="dog", hint="human")

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.HUMAN_DETECTED.value)

    def test_12_uncertainty_handling_preserves_concept_of_uncertainty(self):
        """Preserve concept of uncertainty when uncertainty bounds are configured."""
        config = AnimalnessGateConfig(
            uncertainty_min=0.30,
            uncertainty_max=0.70,
        )
        custom_service = AnimalnessGateService(config=config, lazy_load=True)
        custom_service._gate_model = MagicMock()
        custom_service._species_model = MagicMock()

        # Mock gate output with 0.55 animal probability (inside uncertainty range)
        # Logits that yield approx [0.55, 0.45]
        custom_service._gate_model.return_value = torch.tensor([[0.20, 0.0]])
        # Mock species output
        custom_service._species_model.return_value = torch.tensor([[10.0, 0.0, 0.0]])

        img = Image.new("RGB", (224, 224), "white")
        result = custom_service.validate(img, selected_species="dog")

        self.assertEqual(result["decision"], GateDecision.UNCERTAIN.value)
        self.assertEqual(result["reason_code"], GateReasonCode.LOW_CONFIDENCE.value)
        self.assertAlmostEqual(result["animal_probability"], 0.5498, places=2)

    def test_13_gate_response_schema_contract(self):
        """Verify full gate response schema matches the PRD specification."""
        img = Image.new("RGB", (224, 224), "white")
        result = self.service.validate(img, selected_species="dog")

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


if __name__ == "__main__":
    unittest.main()
