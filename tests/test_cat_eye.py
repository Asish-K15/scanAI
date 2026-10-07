import hashlib
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from app.routers.predict import predict
from app.services.cat_eye import (
    CatEyeInvalidCropError,
    CatEyeModel,
    CatEyeModelNotAvailableError,
    CLASS_INDEX_TO_NAME,
    CLASS_NAMES,
    compute_file_sha256,
    EXPECTED_SHA256,
    get_cat_eye_model,
    reset_cat_eye_model,
    UNCERTAINTY_THRESHOLD,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "dataset_manager"
    / "training"
    / "outputs"
    / "cat_eye"
    / "efficientnet_b0"
    / "best_checkpoint.pth"
)


class FakeUploadFile:
    def __init__(self, image: Image.Image, content_type: str = "image/jpeg"):
        self.image = image
        self.content_type = content_type

    async def read(self) -> bytes:
        buf = BytesIO()
        self.image.save(buf, format="JPEG")
        return buf.getvalue()


class FakeGateService:
    def __init__(self, decision: str = "ACCEPT"):
        self.decision = decision

    def validate(self, image, selected_species, **kwargs):
        return {
            "decision": self.decision,
            "predicted_species": selected_species,
            "species_confidence": 0.99,
            "animal_probability": 0.99,
            "model_name": "EfficientNet-B0",
            "model_version": "SCANAI-ANIMALNESS-GATE-V1",
            "reason_code": "SUPPORTED_ANIMAL_DETECTED"
            if self.decision == "ACCEPT"
            else "NON_ANIMAL_DETECTED",
            "error_code": None,
        }


class TestCatEyeModel(unittest.TestCase):
    """
    Focused unit tests for the frozen Cat Eye model and inference service.
    Covers contract requirements 1 through 13.
    """

    @classmethod
    def setUpClass(cls):
        cls.initial_sha256 = compute_file_sha256(CHECKPOINT_PATH)
        cls.initial_size = CHECKPOINT_PATH.stat().st_size

    def setUp(self):
        reset_cat_eye_model()

    def tearDown(self):
        reset_cat_eye_model()

    # ---------------------------------------------------------
    # 1. Checkpoint exists
    # ---------------------------------------------------------
    def test_01_checkpoint_exists(self):
        self.assertTrue(
            CHECKPOINT_PATH.exists(),
            f"Cat Eye checkpoint not found at: {CHECKPOINT_PATH}",
        )
        self.assertTrue(CHECKPOINT_PATH.is_file())

    # ---------------------------------------------------------
    # 2. Checkpoint SHA-256 exact match
    # ---------------------------------------------------------
    def test_02_checkpoint_sha256_exact_match(self):
        actual_sha = compute_file_sha256(CHECKPOINT_PATH)
        self.assertEqual(
            actual_sha,
            EXPECTED_SHA256,
            f"Checkpoint SHA-256 mismatch! Got {actual_sha}, expected {EXPECTED_SHA256}",
        )
        self.assertEqual(actual_sha, "997576aa344eeec8f03c30d4b1d501850a89921e204a6b16455e147d5bd8ab7b")

    # ---------------------------------------------------------
    # 3. Model loads successfully
    # ---------------------------------------------------------
    def test_03_efficientnet_b0_model_loads(self):
        model = get_cat_eye_model()
        self.assertIsInstance(model, CatEyeModel)
        self.assertIsInstance(model.model, nn.Module)
        self.assertEqual(model.model.classifier[1].out_features, 6)

    # ---------------------------------------------------------
    # 4. Exact six-class mapping
    # ---------------------------------------------------------
    def test_04_exact_six_class_mapping(self):
        expected_classes = [
            "cataracts",
            "conjunctivitis",
            "cornealulcer",
            "glaucoma",
            "healthy",
            "uveitis",
        ]
        self.assertEqual(CLASS_NAMES, expected_classes)
        self.assertEqual(len(CLASS_NAMES), 6)

        expected_mapping = {
            0: "cataracts",
            1: "conjunctivitis",
            2: "cornealulcer",
            3: "glaucoma",
            4: "healthy",
            5: "uveitis",
        }
        self.assertEqual(CLASS_INDEX_TO_NAME, expected_mapping)

        # Also verify against checkpoint internal metadata
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint.get("classes"), expected_classes)

    # ---------------------------------------------------------
    # 5. Preprocessing: RGB, 224x224, ImageNet normalization
    # ---------------------------------------------------------
    def test_05_preprocessing_rgb_and_normalization(self):
        model = get_cat_eye_model()

        # Test with RGBA image
        rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        tensor = model.preprocess(rgba_img)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

        # Test with Grayscale L image
        gray_img = Image.new("L", (80, 80), 128)
        gray_tensor = model.preprocess(gray_img)
        self.assertEqual(gray_tensor.shape, (1, 3, 224, 224))

    # ---------------------------------------------------------
    # 6. Tensor dimensions (Input [1, 3, 224, 224] -> Output [1, 6])
    # ---------------------------------------------------------
    def test_06_tensor_dimensions(self):
        model = get_cat_eye_model()
        dummy_input = torch.randn(1, 3, 224, 224, device=model.device)

        with torch.no_grad():
            output = model.model(dummy_input)

        self.assertEqual(output.shape, (1, 6))

    # ---------------------------------------------------------
    # 7. Output contains exactly 6 class probabilities
    # ---------------------------------------------------------
    def test_07_output_contains_six_class_probabilities(self):
        model = get_cat_eye_model()
        img = Image.new("RGB", (64, 64), color="blue")
        result = model.predict(img)

        probabilities = result["probabilities"]
        self.assertEqual(len(probabilities), 6)
        self.assertEqual(set(probabilities.keys()), set(CLASS_NAMES))

        # Check values sum to ~1.0
        prob_sum = sum(probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

        for prob in probabilities.values():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    # ---------------------------------------------------------
    # 8. Predicted class mapping is correct (controlled logits)
    # ---------------------------------------------------------
    def test_08_predicted_class_mapping_controlled_logits(self):
        model = get_cat_eye_model()
        img = Image.new("RGB", (64, 64), color="white")

        for target_idx, expected_class in CLASS_INDEX_TO_NAME.items():
            # Build controlled logits where target_idx has the highest logit
            logits = torch.zeros(1, 6)
            logits[0, target_idx] = 10.0

            with patch.object(model, "model", return_value=logits):
                result = model.predict(img)
                self.assertEqual(result["predicted_class_index"], target_idx)
                self.assertEqual(result["predicted_class"], expected_class)
                self.assertEqual(result["condition"], expected_class)

    # ---------------------------------------------------------
    # 9. Confidence is returned
    # ---------------------------------------------------------
    def test_09_confidence_returned(self):
        model = get_cat_eye_model()
        img = Image.new("RGB", (64, 64), color="green")
        result = model.predict(img)

        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(
            result["confidence"],
            result["probabilities"][result["predicted_class"]],
        )

    # ---------------------------------------------------------
    # 10. confidence < 0.40 -> is_uncertain = True (deterministic)
    # ---------------------------------------------------------
    def test_10_uncertainty_below_0_40(self):
        model = get_cat_eye_model()
        img = Image.new("RGB", (64, 64), color="white")

        # Flat logits -> softmax gives 1/6 = 0.1667 < 0.40
        flat_logits = torch.ones(1, 6)

        with patch.object(model, "model", return_value=flat_logits):
            result = model.predict(img)
            self.assertAlmostEqual(result["confidence"], 1.0 / 6.0, places=4)
            self.assertTrue(result["is_uncertain"])
            self.assertTrue(result["uncertain"])

    # ---------------------------------------------------------
    # 11. confidence >= 0.40 -> is_uncertain = False (deterministic)
    # ---------------------------------------------------------
    def test_11_uncertainty_at_or_above_0_40(self):
        model = get_cat_eye_model()
        img = Image.new("RGB", (64, 64), color="white")

        # Logits producing ~0.60 confidence
        logits = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        with patch.object(model, "model", return_value=logits):
            result = model.predict(img)
            self.assertGreaterEqual(result["confidence"], UNCERTAINTY_THRESHOLD)
            self.assertFalse(result["is_uncertain"])
            self.assertFalse(result["uncertain"])

    # ---------------------------------------------------------
    # 12. Input dimensions < 32x32 rejected BEFORE inference
    # ---------------------------------------------------------
    def test_12_crop_protection_rejects_under_32x32_before_inference(self):
        model = get_cat_eye_model()

        small_cases = [
            (31, 31),
            (16, 16),
            (31, 64),
            (64, 31),
            (10, 10),
            (1, 1),
        ]

        for width, height in small_cases:
            img = Image.new("RGB", (width, height), color="white")
            mock_model = MagicMock()

            with patch.object(model, "model", mock_model):
                with self.assertRaises(CatEyeInvalidCropError):
                    model.predict(img)

                # Ensure model inference was NEVER called
                mock_model.assert_not_called()

    # ---------------------------------------------------------
    # 13. Checkpoint immutability
    # ---------------------------------------------------------
    def test_13_checkpoint_immutability(self):
        current_sha = compute_file_sha256(CHECKPOINT_PATH)
        current_size = CHECKPOINT_PATH.stat().st_size

        self.assertEqual(
            current_sha,
            self.initial_sha256,
            "Checkpoint SHA-256 was altered during testing!",
        )
        self.assertEqual(
            current_sha,
            "997576aa344eeec8f03c30d4b1d501850a89921e204a6b16455e147d5bd8ab7b",
        )
        self.assertEqual(
            current_size,
            self.initial_size,
            "Checkpoint file size was altered during testing!",
        )


class TestCatEyeAPIIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests for Cat Eye routing via /api/predict.
    Covers Animalness Gate ordering and API response contract.
    """

    # ---------------------------------------------------------
    # 14. Animalness Gate ordering before Cat Eye
    # ---------------------------------------------------------
    async def test_14_animalness_gate_reject_halts_before_cat_eye_inference(self):
        valid_image = Image.new("RGB", (64, 64), "white")
        upload_file = FakeUploadFile(valid_image)

        with patch(
            "app.routers.predict.get_animalness_gate_service",
            return_value=FakeGateService(decision="REJECT"),
        ):
            with patch("app.routers.predict.get_cat_eye_model") as mock_cat_eye:
                result = await predict(
                    image=upload_file,
                    species="cat",
                    body_area="eye",
                )

                # Gate returns REJECT response immediately
                self.assertEqual(result["decision"], "REJECT")
                # Cat Eye model was never called
                mock_cat_eye.assert_not_called()

    async def test_14_animalness_gate_uncertain_halts_before_cat_eye_inference(self):
        valid_image = Image.new("RGB", (64, 64), "white")
        upload_file = FakeUploadFile(valid_image)

        with patch(
            "app.routers.predict.get_animalness_gate_service",
            return_value=FakeGateService(decision="UNCERTAIN"),
        ):
            with patch("app.routers.predict.get_cat_eye_model") as mock_cat_eye:
                result = await predict(
                    image=upload_file,
                    species="cat",
                    body_area="eye",
                )

                self.assertEqual(result["decision"], "UNCERTAIN")
                mock_cat_eye.assert_not_called()

    # ---------------------------------------------------------
    # 15. Cat Eye response contract through API
    # ---------------------------------------------------------
    async def test_15_cat_eye_api_response_contract(self):
        valid_image = Image.new("RGB", (64, 64), "white")
        upload_file = FakeUploadFile(valid_image)

        with patch(
            "app.routers.predict.get_animalness_gate_service",
            return_value=FakeGateService(decision="ACCEPT"),
        ):
            result = await predict(
                image=upload_file,
                species="cat",
                body_area="eye",
            )

        expected_top_level_keys = {
            "species",
            "body_area",
            "condition",
            "confidence",
            "confidence_level",
            "uncertain",
            "severity",
            "urgency",
            "evidence_status",
            "evidence",
            "recommendation",
        }
        self.assertEqual(set(result.keys()), expected_top_level_keys)

        self.assertEqual(result["species"], "cat")
        self.assertEqual(result["body_area"], "eye")
        self.assertIn(result["condition"], CLASS_NAMES)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["uncertain"], bool)

        # Evidence sub-schema
        evidence = result["evidence"]
        self.assertEqual(
            set(evidence["probabilities"].keys()),
            set(CLASS_NAMES),
        )
        self.assertEqual(evidence["model_version"], "v1.0.0")
        self.assertEqual(evidence["engine"], "PyTorch")
        self.assertTrue(evidence["screening_only"])

        # Urgency/severity clinical boundary check
        self.assertIsNone(result["severity"])
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["recommendation"], "insufficient evidence / urgency undefined")

    # ---------------------------------------------------------
    # 16. Crop protection via API (<32x32 rejected before inference)
    # ---------------------------------------------------------
    async def test_16_cat_eye_crop_under_32x32_via_api_returns_400(self):
        small_image = Image.new("RGB", (24, 24), "white")
        upload_file = FakeUploadFile(small_image)

        with patch(
            "app.routers.predict.get_animalness_gate_service",
            return_value=FakeGateService(decision="ACCEPT"),
        ):
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as ctx:
                await predict(
                    image=upload_file,
                    species="cat",
                    body_area="eye",
                )
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("32x32", str(ctx.exception.detail))

    # ---------------------------------------------------------
    # 17. Species mismatch (dog selected with cat image) never routes to Cat Eye
    # ---------------------------------------------------------
    async def test_17_selected_species_dog_with_cat_image_rejects_species_mismatch_and_never_routes_to_cat_eye(self):
        cat_image = Image.new("RGB", (64, 64), "white")
        upload_file = FakeUploadFile(cat_image)

        mismatch_gate_response = {
            "decision": "REJECT",
            "predicted_species": "cat",
            "species_confidence": 0.98,
            "animal_probability": 0.99,
            "model_name": "EfficientNet-B0",
            "model_version": "SCANAI-ANIMALNESS-GATE-V1",
            "reason_code": "SPECIES_MISMATCH",
            "error_code": None,
        }

        mock_gate = MagicMock()
        mock_gate.validate.return_value = mismatch_gate_response

        with patch("app.routers.predict.get_animalness_gate_service", return_value=mock_gate):
            with patch("app.routers.predict.get_cat_eye_model") as mock_cat_eye:
                result = await predict(
                    image=upload_file,
                    species="dog",
                    body_area="eye",
                )
                self.assertEqual(result["decision"], "REJECT")
                self.assertEqual(result["reason_code"], "SPECIES_MISMATCH")
                # Never silently reroute to Cat Eye
                mock_cat_eye.assert_not_called()

    # ---------------------------------------------------------
    # 18. Clinical boundaries (high confidence never creates severity or urgency)
    # ---------------------------------------------------------
    async def test_18_cat_eye_clinical_boundaries_never_infer_urgency_or_severity(self):
        valid_image = Image.new("RGB", (64, 64), "white")
        upload_file = FakeUploadFile(valid_image)

        fake_high_conf_result = {
            "condition": "cornealulcer",
            "confidence": 0.995,
            "confidence_level": "high",
            "uncertain": False,
            "probabilities": {name: (0.995 if name == "cornealulcer" else 0.001) for name in CLASS_NAMES},
            "model": "scanai_cat_eye_efficientnet_b0_v1",
            "model_version": "v1.0.0",
            "engine": "PyTorch",
            "screening_only": True,
        }

        mock_model = MagicMock()
        mock_model.predict.return_value = fake_high_conf_result

        with patch(
            "app.routers.predict.get_animalness_gate_service",
            return_value=FakeGateService(decision="ACCEPT"),
        ):
            with patch("app.routers.predict.get_cat_eye_model", return_value=mock_model):
                result = await predict(
                    image=upload_file,
                    species="cat",
                    body_area="eye",
                )

        self.assertEqual(result["confidence"], 0.995)
        self.assertEqual(result["confidence_level"], "high")
        self.assertIsNone(result["severity"])
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["recommendation"], "insufficient evidence / urgency undefined")


if __name__ == "__main__":
    unittest.main()
