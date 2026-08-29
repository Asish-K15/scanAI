import unittest

from io import BytesIO
from unittest.mock import patch

from PIL import Image

from app.routers.predict import predict


class FakeUploadFile:
    content_type = "image/jpeg"

    async def read(self):
        image = Image.new("RGB", (10, 10), "white")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()


class FakeModel:
    def predict(self, image):
        return {
            "condition": "conjunctivitis",
            "confidence": 0.7150973081588745,
            "confidence_level": "moderate",
            "uncertain": False,
            "probabilities": {
                "conjunctivitis": 0.7150973081588745,
                "entropion": 0.2849027216434479,
            },
            "model": "EfficientNet-B0",
            "model_version": "dog-eye-v1",
            "engine": "ONNX Runtime",
            "screening_only": True,
        }


class FakeSkinModel:
    def predict(self, image):
        return {
            "condition": "skin__mange",
            "confidence": 0.91,
            "confidence_level": "high",
            "uncertain": False,
            "probabilities": {
                "skin__allergic_dermatitis": 0.01,
                "skin__foot_and_mouth_disease": 0.01,
                "skin__fungal_infection": 0.01,
                "skin__healthy_skin": 0.01,
                "skin__hotspot": 0.01,
                "skin__lumpy_skin_disease": 0.01,
                "skin__mange": 0.91,
                "skin__pyoderma": 0.01,
                "skin__ringworm": 0.01,
                "skin__scabies": 0.01,
            },
            "model": "EfficientNet-B0",
            "model_version": "SCANAI-SKIN-PHASE4B",
            "engine": "ONNX Runtime",
            "screening_only": True,
        }


class TestPredictRoute(unittest.IsolatedAsyncioTestCase):

    async def test_dog_eye_route_preserves_insufficient_evidence_behavior(self):
        image = FakeUploadFile()

        with patch(
            "app.routers.predict.get_dog_eye_model",
            return_value=FakeModel(),
        ):
            result = await predict(
                image=image,
                species="dog",
                body_area="eye",
            )

        self.assertEqual(result["species"], "dog")
        self.assertEqual(result["body_area"], "eye")
        self.assertEqual(result["condition"], "conjunctivitis")
        self.assertEqual(result["confidence_level"], "moderate")

        self.assertIsNone(result["severity"])
        self.assertIsNone(result["urgency"])

        self.assertEqual(
            result["evidence_status"],
            "insufficient_evidence",
        )
        self.assertEqual(
            result["recommendation"],
            "insufficient evidence / urgency undefined",
        )

        self.assertEqual(
            result["evidence"]["model"],
            "EfficientNet-B0",
        )
        self.assertEqual(
            result["evidence"]["model_version"],
            "dog-eye-v1",
        )
        self.assertEqual(
            result["evidence"]["engine"],
            "ONNX Runtime",
        )
        self.assertTrue(result["evidence"]["screening_only"])

    async def test_skin_route_uses_skin_model(self):
        image = FakeUploadFile()

        with patch(
            "app.routers.predict.get_skin_model",
            return_value=FakeSkinModel(),
        ) as mock_skin_model:
            result = await predict(
                image=image,
                species="dog",
                body_area="skin",
            )

        mock_skin_model.assert_called_once()

        self.assertEqual(result["species"], "dog")
        self.assertEqual(result["body_area"], "skin")
        self.assertEqual(result["condition"], "skin__mange")
        self.assertEqual(result["confidence"], 0.91)
        self.assertEqual(result["confidence_level"], "high")
        self.assertFalse(result["uncertain"])

        self.assertEqual(
            result["evidence"]["model_version"],
            "SCANAI-SKIN-PHASE4B",
        )
        self.assertEqual(
            result["evidence"]["engine"],
            "ONNX Runtime",
        )
        self.assertTrue(
            result["evidence"]["screening_only"]
        )

    async def test_skin_route_does_not_create_urgency_from_confidence(self):
        image = FakeUploadFile()

        with patch(
            "app.routers.predict.get_skin_model",
            return_value=FakeSkinModel(),
        ):
            result = await predict(
                image=image,
                species="dog",
                body_area="skin",
            )

        self.assertEqual(result["confidence_level"], "high")
        self.assertIsNone(result["severity"])
        self.assertIsNone(result["urgency"])

    async def test_cat_skin_route_uses_skin_model(self):
        image = FakeUploadFile()

        with patch(
            "app.routers.predict.get_skin_model",
            return_value=FakeSkinModel(),
        ):
            result = await predict(
                image=image,
                species="cat",
                body_area="skin",
            )

        self.assertEqual(result["species"], "cat")
        self.assertEqual(result["body_area"], "skin")
        self.assertEqual(result["condition"], "skin__mange")
        self.assertEqual(
            result["model_version"]
            if "model_version" in result
            else result["evidence"]["model_version"],
            "SCANAI-SKIN-PHASE4B",
        )

    async def test_skin_inference_failure_returns_500(self):
        class FailingSkinModel:
            def predict(self, image):
                raise RuntimeError("test skin inference failure")

        image = FakeUploadFile()

        with patch(
            "app.routers.predict.get_skin_model",
            return_value=FailingSkinModel(),
        ):
            with self.assertRaises(Exception) as context:
                await predict(
                    image=image,
                    species="dog",
                    body_area="skin",
                )

        self.assertEqual(
            context.exception.status_code,
            500,
        )

    async def test_invalid_species_returns_400(self):
        image = FakeUploadFile()

        with self.assertRaises(Exception) as context:
            await predict(
                image=image,
                species="horse",
                body_area="eye",
            )

        self.assertEqual(context.exception.status_code, 400)

    async def test_invalid_body_area_returns_400(self):
        image = FakeUploadFile()

        with self.assertRaises(Exception) as context:
            await predict(
                image=image,
                species="dog",
                body_area="heart",
            )

        self.assertEqual(context.exception.status_code, 400)

    async def test_non_image_upload_returns_400(self):
        class FakeTextUpload:
            content_type = "text/plain"

            async def read(self):
                return b"not an image"

        image = FakeTextUpload()

        with self.assertRaises(Exception) as context:
            await predict(
                image=image,
                species="dog",
                body_area="eye",
            )

        self.assertEqual(context.exception.status_code, 400)

    async def test_corrupted_image_returns_400(self):
        class FakeCorruptUpload:
            content_type = "image/jpeg"

            async def read(self):
                return b"this is not a valid image"

        image = FakeCorruptUpload()

        with self.assertRaises(Exception) as context:
            await predict(
                image=image,
                species="dog",
                body_area="eye",
            )

        self.assertEqual(context.exception.status_code, 400)

    async def test_unsupported_model_route_returns_501(self):
        image = FakeUploadFile()

        with self.assertRaises(Exception) as context:
            await predict(
                image=image,
                species="cat",
                body_area="eye",
            )

        self.assertEqual(context.exception.status_code, 501)

    async def test_dog_eye_inference_failure_returns_500(self):
        class FailingModel:
            def predict(self, image):
                raise RuntimeError("test inference failure")

        image = FakeUploadFile()

        with patch(
            "app.routers.predict.get_dog_eye_model",
            return_value=FailingModel(),
        ):
            with self.assertRaises(Exception) as context:
                await predict(
                    image=image,
                    species="dog",
                    body_area="eye",
                )

        self.assertEqual(context.exception.status_code, 500)

    async def test_dog_eye_response_matches_expected_contract(self):
        image = FakeUploadFile()

        with patch(
            "app.routers.predict.get_dog_eye_model",
            return_value=FakeModel(),
        ):
            result = await predict(
                image=image,
                species="dog",
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

        self.assertEqual(
            set(result.keys()),
            expected_top_level_keys,
        )

        self.assertIsInstance(result["species"], str)
        self.assertIsInstance(result["body_area"], str)
        self.assertIsInstance(result["condition"], str)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["confidence_level"], str)
        self.assertIsInstance(result["uncertain"], bool)
        self.assertIsNone(result["severity"])
        self.assertIsNone(result["urgency"])
        self.assertIsInstance(result["evidence_status"], str)
        self.assertIsInstance(result["evidence"], dict)
        self.assertIsInstance(result["recommendation"], str)

        expected_evidence_keys = {
            "condition",
            "confidence",
            "confidence_level",
            "uncertain",
            "probabilities",
            "model",
            "model_version",
            "engine",
            "screening_only",
        }

        self.assertEqual(
            set(result["evidence"].keys()),
            expected_evidence_keys,
        )

        self.assertEqual(
            set(result["evidence"]["probabilities"].keys()),
            {
                "conjunctivitis",
                "entropion",
            },
        )

        self.assertIsInstance(
            result["evidence"]["probabilities"]["conjunctivitis"],
            float,
        )
        self.assertIsInstance(
            result["evidence"]["probabilities"]["entropion"],
            float,
        )

        self.assertEqual(
            result["evidence_status"],
            "insufficient_evidence",
        )
        self.assertEqual(
            result["recommendation"],
            "insufficient evidence / urgency undefined",
        )


if __name__ == "__main__":
    unittest.main()