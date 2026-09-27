"""
Integration tests for Animalness Gate in /api/predict.
Verifies the complete flow, disease model protection, and non-rerouting requirements.
"""

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from PIL import Image

from animalness_gate.schemas import GateDecision, GateReasonCode
from app.routers.predict import predict


class FakeUploadImage:
    def __init__(self, pil_image: Image.Image, filename: str = "test.jpg", content_type: str = "image/jpeg"):
        self.filename = filename
        self.content_type = content_type
        buffer = BytesIO()
        fmt = "PNG" if filename.lower().endswith(".png") else "JPEG"
        pil_image.save(buffer, format=fmt)
        self._bytes = buffer.getvalue()

    async def read(self):
        return self._bytes


class TestAnimalnessGateIntegration(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parents[1]

        # Locate sample images
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

    async def test_01_human_image_rejected_at_predict_api(self):
        """Human image -> REJECT and disease models are never called."""
        if not self.human_image_path.exists():
            self.skipTest("Human image not found")

        img = Image.open(self.human_image_path)
        upload = FakeUploadImage(img, filename="human.png", content_type="image/png")

        with patch("app.routers.predict.get_dog_eye_model") as mock_dog_eye, \
             patch("app.routers.predict.get_skin_model") as mock_skin:

            result = await predict(
                image=upload,
                species="dog",
                body_area="skin",
            )

            mock_dog_eye.assert_not_called()
            mock_skin.assert_not_called()

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertIn(
            result["reason_code"],
            [GateReasonCode.NON_ANIMAL_DETECTED.value, GateReasonCode.HUMAN_DETECTED.value],
        )
        self.assertLess(result["animal_probability"], 0.50)

    async def test_02_non_animal_image_rejected_at_predict_api(self):
        """Non-animal image -> REJECT and disease models are never called."""
        if not self.non_animal_image_path.exists():
            self.skipTest("Non-animal image not found")

        img = Image.open(self.non_animal_image_path)
        upload = FakeUploadImage(img, filename="apple.jpg", content_type="image/jpeg")

        with patch("app.routers.predict.get_dog_eye_model") as mock_dog_eye, \
             patch("app.routers.predict.get_skin_model") as mock_skin:

            result = await predict(
                image=upload,
                species="dog",
                body_area="skin",
            )

            mock_dog_eye.assert_not_called()
            mock_skin.assert_not_called()

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.NON_ANIMAL_DETECTED.value)

    async def test_03_dog_plus_selected_dog_accepted_routes_to_dog_eye(self):
        """Dog image + selected dog -> ACCEPT and routes to Dog Eye model."""
        if not self.dog_image_path or not self.dog_image_path.exists():
            self.skipTest("Dog image not found")

        img = Image.open(self.dog_image_path)
        upload = FakeUploadImage(img, filename="dog.jpg", content_type="image/jpeg")

        fake_dog_eye = MagicMock()
        fake_dog_eye.predict.return_value = {
            "condition": "conjunctivitis",
            "confidence": 0.85,
            "confidence_level": "high",
            "uncertain": False,
            "probabilities": {"conjunctivitis": 0.85, "entropion": 0.15},
            "model": "EfficientNet-B0",
            "model_version": "dog-eye-v1",
            "engine": "ONNX Runtime",
            "screening_only": True,
        }

        with patch("app.routers.predict.get_dog_eye_model", return_value=fake_dog_eye):
            result = await predict(
                image=upload,
                species="dog",
                body_area="eye",
            )

            fake_dog_eye.predict.assert_called_once()

        self.assertEqual(result["species"], "dog")
        self.assertEqual(result["body_area"], "eye")
        self.assertEqual(result["condition"], "conjunctivitis")
        self.assertIn("recommendation", result)

    async def test_04_cat_plus_selected_cat_accepted_routes_to_skin(self):
        """Cat image + selected cat -> ACCEPT and routes to Skin model."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat image not found")

        img = Image.open(self.cat_image_path)
        upload = FakeUploadImage(img, filename="cat.png", content_type="image/png")

        fake_skin = MagicMock()
        fake_skin.predict.return_value = {
            "condition": "skin__ringworm",
            "confidence": 0.90,
            "confidence_level": "high",
            "uncertain": False,
            "probabilities": {"skin__ringworm": 0.90, "skin__healthy_skin": 0.10},
            "model": "EfficientNet-B0",
            "model_version": "SCANAI-SKIN-PHASE4B",
            "engine": "ONNX Runtime",
            "screening_only": True,
        }

        with patch("app.routers.predict.get_skin_model", return_value=fake_skin):
            result = await predict(
                image=upload,
                species="cat",
                body_area="skin",
            )

            fake_skin.predict.assert_called_once()

        self.assertEqual(result["species"], "cat")
        self.assertEqual(result["body_area"], "skin")
        self.assertEqual(result["condition"], "skin__ringworm")

    async def test_05_cattle_plus_selected_cattle_accepted_routes_to_skin(self):
        """Cattle image + selected cattle -> ACCEPT and routes to Skin model."""
        if not self.cattle_image_path or not self.cattle_image_path.exists():
            self.skipTest("Cattle image not found")

        img = Image.open(self.cattle_image_path)
        upload = FakeUploadImage(img, filename="cattle.jpg", content_type="image/jpeg")

        fake_skin = MagicMock()
        fake_skin.predict.return_value = {
            "condition": "skin__lumpy_skin_disease",
            "confidence": 0.92,
            "confidence_level": "high",
            "uncertain": False,
            "probabilities": {"skin__lumpy_skin_disease": 0.92, "skin__healthy_skin": 0.08},
            "model": "EfficientNet-B0",
            "model_version": "SCANAI-SKIN-PHASE4B",
            "engine": "ONNX Runtime",
            "screening_only": True,
        }

        with patch("app.routers.predict.get_skin_model", return_value=fake_skin):
            result = await predict(
                image=upload,
                species="cattle",
                body_area="skin",
            )

            fake_skin.predict.assert_called_once()

        self.assertEqual(result["species"], "cattle")
        self.assertEqual(result["body_area"], "skin")
        self.assertEqual(result["condition"], "skin__lumpy_skin_disease")

    async def test_06_cat_image_selected_dog_rejected_species_mismatch(self):
        """Cat image + selected dog -> REJECT with SPECIES_MISMATCH and no silent rerouting."""
        if not self.cat_image_path or not self.cat_image_path.exists():
            self.skipTest("Cat image not found")

        img = Image.open(self.cat_image_path)
        upload = FakeUploadImage(img, filename="cat.png", content_type="image/png")

        with patch("app.routers.predict.get_dog_eye_model") as mock_dog_eye, \
             patch("app.routers.predict.get_skin_model") as mock_skin:

            result = await predict(
                image=upload,
                species="dog",
                body_area="eye",
            )

            mock_dog_eye.assert_not_called()
            mock_skin.assert_not_called()

        self.assertEqual(result["decision"], GateDecision.REJECT.value)
        self.assertEqual(result["reason_code"], GateReasonCode.SPECIES_MISMATCH.value)
        self.assertEqual(result["predicted_species"], "cat")

    async def test_07_gate_uncertain_decision_stops_flow_and_returns_gate_response(self):
        """Gate UNCERTAIN -> returns gate response and disease model is not called."""
        img = Image.new("RGB", (224, 224), "gray")
        upload = FakeUploadImage(img, filename="test.jpg")

        fake_gate_service = MagicMock()
        fake_gate_service.validate.return_value = {
            "decision": "UNCERTAIN",
            "predicted_species": "dog",
            "species_confidence": 0.51,
            "animal_probability": 0.52,
            "model_name": "EfficientNet-B0",
            "model_version": "SCANAI-ANIMALNESS-GATE-V1",
            "reason_code": "LOW_CONFIDENCE",
            "error_code": None,
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=fake_gate_service), \
             patch("app.routers.predict.get_dog_eye_model") as mock_dog_eye:

            result = await predict(
                image=upload,
                species="dog",
                body_area="eye",
            )

            mock_dog_eye.assert_not_called()

        self.assertEqual(result["decision"], "UNCERTAIN")
        self.assertEqual(result["reason_code"], "LOW_CONFIDENCE")

    async def test_08_gate_model_error_returns_gate_response(self):
        """Gate MODEL_ERROR -> returns controlled gate response without crashing."""
        img = Image.new("RGB", (224, 224), "gray")
        upload = FakeUploadImage(img, filename="test.jpg")

        fake_gate_service = MagicMock()
        fake_gate_service.validate.return_value = {
            "decision": "REJECT",
            "predicted_species": None,
            "species_confidence": 0.0,
            "animal_probability": 0.0,
            "model_name": "EfficientNet-B0",
            "model_version": "SCANAI-ANIMALNESS-GATE-V1",
            "reason_code": "MODEL_ERROR",
            "error_code": "Simulated hardware fault",
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=fake_gate_service), \
             patch("app.routers.predict.get_dog_eye_model") as mock_dog_eye:

            result = await predict(
                image=upload,
                species="dog",
                body_area="eye",
            )

            mock_dog_eye.assert_not_called()

        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(result["reason_code"], "MODEL_ERROR")
        self.assertEqual(result["error_code"], "Simulated hardware fault")


if __name__ == "__main__":
    unittest.main()
