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


if __name__ == "__main__":
    unittest.main()
