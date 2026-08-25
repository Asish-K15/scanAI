import unittest

from app.services.fusion_urgency import (
    determine_urgency,
)


class TestFusionUrgency(unittest.TestCase):

    def test_severe_deep_tissue_laceration_with_active_hemorrhage_is_emergency(self):
        urgency = determine_urgency(
            severity="severe",
            condition="deep-tissue laceration",
            active_hemorrhage=True,
        )

        self.assertEqual(
            urgency,
            "Emergency",
        )

    def test_low_confidence_does_not_create_urgency(self):
        urgency = determine_urgency(
            confidence_level="low",
        )

        self.assertIsNone(
            urgency,
        )

    def test_moderate_confidence_does_not_create_urgency(self):
        urgency = determine_urgency(
            confidence_level="moderate",
        )

        self.assertIsNone(
            urgency,
        )

    def test_high_confidence_does_not_create_urgency(self):
        urgency = determine_urgency(
            confidence_level="high",
        )

        self.assertIsNone(
            urgency,
        )

    def test_dog_eye_conjunctivitis_does_not_create_urgency(self):
        urgency = determine_urgency(
            body_area="eye",
            condition="conjunctivitis",
            confidence_level="high",
        )

        self.assertIsNone(
            urgency,
        )

    def test_dog_eye_entropion_does_not_create_urgency(self):
        urgency = determine_urgency(
            body_area="eye",
            condition="entropion",
            confidence_level="high",
        )

        self.assertIsNone(
            urgency,
        )

    def test_insufficient_evidence_returns_none(self):
        urgency = determine_urgency(
            evidence_status="insufficient_evidence",
        )

        self.assertIsNone(
            urgency,
        )

    def test_emergency_is_not_removed_by_low_confidence(self):
        urgency = determine_urgency(
            severity="severe",
            condition="deep-tissue laceration",
            active_hemorrhage=True,
            confidence_level="low",
        )

        self.assertEqual(
            urgency,
            "Emergency",
        )

    def test_severe_without_active_hemorrhage_does_not_create_emergency(self):
        urgency = determine_urgency(
            severity="severe",
            condition="deep-tissue laceration",
            active_hemorrhage=False,
        )

        self.assertIsNone(
            urgency,
        )

    def test_active_hemorrhage_without_severe_laceration_does_not_create_emergency(self):
        urgency = determine_urgency(
            severity="moderate",
            condition="deep-tissue laceration",
            active_hemorrhage=True,
        )

        self.assertIsNone(
            urgency,
        )

    def test_no_approved_rule_returns_none(self):
        urgency = determine_urgency(
            severity="mild",
            condition="unknown",
            body_area="skin",
        )

        self.assertIsNone(
            urgency,
        )
    def test_low_risk_evidence_creates_routine(self):
        urgency = determine_urgency(
            low_risk_evidence=True,
        )

        self.assertEqual(
            urgency,
            "Routine",
        )

    def test_low_risk_evidence_false_does_not_create_routine(self):
        urgency = determine_urgency(
            low_risk_evidence=False,
        )

        self.assertIsNone(
            urgency,
        )

    def test_low_risk_evidence_absent_does_not_create_routine(self):
        urgency = determine_urgency()

        self.assertIsNone(
            urgency,
        )

    def test_low_risk_evidence_does_not_override_emergency(self):
        urgency = determine_urgency(
            severity="severe",
            condition="deep-tissue laceration",
            active_hemorrhage=True,
            low_risk_evidence=True,
        )

        self.assertEqual(
            urgency,
            "Emergency",
        )

if __name__ == "__main__":
    unittest.main()