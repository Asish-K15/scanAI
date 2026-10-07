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

    def test_approved_evidence_statuses(self):
        from app.services.fusion_urgency import APPROVED_EVIDENCE_STATUSES

        self.assertEqual(
            APPROVED_EVIDENCE_STATUSES,
            {
                "approved_urgency_evidence",
                "insufficient_evidence",
            },
        )

    def test_conflicts_leave_urgency_undefined(self):
        urgency = determine_urgency(
            severity="severe",
            condition="deep-tissue laceration",
            active_hemorrhage=True,
            conflicts=["conflicting report"],
        )

        self.assertIsNone(urgency)

    def test_string_active_hemorrhage_does_not_trigger_emergency(self):
        for bad_val in ("false", "true", "yes", 1, [True], {"active": True}):
            urgency = determine_urgency(
                severity="severe",
                condition="deep-tissue laceration",
                active_hemorrhage=bad_val,
            )
            self.assertIsNone(
                urgency,
                f"Non-boolean active_hemorrhage {bad_val!r} must not trigger Emergency.",
            )

    def test_string_low_risk_evidence_does_not_trigger_routine(self):
        for bad_val in ("false", "true", "yes", 1, [True]):
            urgency = determine_urgency(
                low_risk_evidence=bad_val,
            )
            self.assertIsNone(
                urgency,
                f"Non-boolean low_risk_evidence {bad_val!r} must not trigger Routine.",
            )

    def test_invalid_severity_does_not_trigger_emergency(self):
        for bad_sev in ("critical", "high", "unknown", 1, None):
            urgency = determine_urgency(
                severity=bad_sev,
                condition="deep-tissue laceration",
                active_hemorrhage=True,
            )
            self.assertIsNone(
                urgency,
                f"Invalid severity {bad_sev!r} must not trigger Emergency.",
            )


if __name__ == "__main__":
    unittest.main()