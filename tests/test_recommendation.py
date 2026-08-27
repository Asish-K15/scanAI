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

        result = build_recommendation(
            "dog",
            "eye",
            model_result,
        )

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

        self.assertTrue(
            result["evidence"]["screening_only"],
        )

    def test_low_confidence_does_not_create_urgency(self):
        model_result = {
            "condition": "conjunctivitis",
            "confidence": 0.42,
            "confidence_level": "low",
            "uncertain": True,
            "probabilities": {
                "conjunctivitis": 0.42,
                "entropion": 0.58,
            },
            "model": "EfficientNet-B0",
            "model_version": "dog-eye-v1",
            "engine": "ONNX Runtime",
            "screening_only": True,
        }

        result = build_recommendation(
            "dog",
            "eye",
            model_result,
        )

        self.assertTrue(result["uncertain"])
        self.assertEqual(
            result["confidence_level"],
            "low",
        )

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

    def test_low_risk_evidence_creates_routine(self):
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

        result = build_recommendation(
            "dog",
            "eye",
            model_result,
            low_risk_evidence=True,
        )

        self.assertEqual(
            result["urgency"],
            "Routine",
        )

        self.assertEqual(
            result["evidence_status"],
            "approved_urgency_evidence",
        )

        self.assertEqual(
            result["recommendation"],
            "routine / non-urgent monitoring",
        )

        self.assertIsNone(
            result["severity"],
        )

    def test_emergency_is_preserved(self):
        model_result = {
            "condition": "deep-tissue laceration",
            "confidence": 0.42,
            "confidence_level": "low",
            "uncertain": True,
            "probabilities": {},
            "model": "test-model",
            "model_version": "test-v1",
            "engine": "test",
            "screening_only": True,
        }

        result = build_recommendation(
            "dog",
            "skin",
            model_result,
            severity="severe",
            active_hemorrhage=True,
            low_risk_evidence=True,
        )

        self.assertEqual(
            result["urgency"],
            "Emergency",
        )

        self.assertEqual(
            result["evidence_status"],
            "approved_urgency_evidence",
        )

        self.assertEqual(
            result["recommendation"],
            "emergency / immediate veterinary attention",
        )

        self.assertEqual(
            result["severity"],
            "severe",
        )


if __name__ == "__main__":
    unittest.main()