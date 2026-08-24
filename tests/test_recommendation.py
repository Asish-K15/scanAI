import unittest

from app.services.recommendation import build_recommendation


class TestRecommendation(unittest.TestCase):

    def test_insufficient_evidence_preserves_model_output(self):
        model_result = {
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

        result = build_recommendation("dog", "eye", model_result)

        self.assertEqual(result["species"], "dog")
        self.assertEqual(result["body_area"], "eye")
        self.assertEqual(result["condition"], "conjunctivitis")
        self.assertEqual(result["confidence"], 0.7150973081588745)
        self.assertEqual(result["confidence_level"], "moderate")
        self.assertFalse(result["uncertain"])

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
            result["evidence"]["probabilities"],
            model_result["probabilities"],
        )
        self.assertEqual(result["evidence"]["model"], "EfficientNet-B0")
        self.assertEqual(
            result["evidence"]["model_version"],
            "dog-eye-v1",
        )
        self.assertEqual(result["evidence"]["engine"], "ONNX Runtime")
        self.assertTrue(result["evidence"]["screening_only"])


if __name__ == "__main__":
    unittest.main()
