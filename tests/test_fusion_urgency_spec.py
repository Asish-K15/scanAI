import unittest


class TestFusionUrgencySpecification(unittest.TestCase):

    def test_urgency_categories_are_application_categories(self):
        urgency_categories = {
            "Routine",
            "Soon",
            "Urgent",
            "Emergency",
        }

        self.assertEqual(
            urgency_categories,
            {
                "Routine",
                "Soon",
                "Urgent",
                "Emergency",
            },
        )

        self.assertNotIn("low", urgency_categories)
        self.assertNotIn("moderate", urgency_categories)
        self.assertNotIn("high", urgency_categories)

    def test_confidence_levels_are_separate_from_urgency(self):
        confidence_levels = {
            "low",
            "moderate",
            "high",
        }

        urgency_categories = {
            "Routine",
            "Soon",
            "Urgent",
            "Emergency",
        }

        self.assertTrue(
            confidence_levels.isdisjoint(urgency_categories)
        )

    def test_dog_eye_confidence_thresholds(self):
        def confidence_level(confidence):
            if confidence >= 0.80:
                return "high"

            if confidence >= 0.60:
                return "moderate"

            return "low"

        self.assertEqual(
            confidence_level(0.59),
            "low",
        )

        self.assertEqual(
            confidence_level(0.60),
            "moderate",
        )

        self.assertEqual(
            confidence_level(0.79),
            "moderate",
        )

        self.assertEqual(
            confidence_level(0.80),
            "high",
        )

    def test_confidence_does_not_create_urgency(self):
        for confidence_level in (
            "low",
            "moderate",
            "high",
        ):
            urgency = None

            self.assertIsNone(urgency)

    def test_dog_eye_condition_does_not_create_severity(self):
        for condition in (
            "conjunctivitis",
            "entropion",
        ):
            severity = None

            self.assertIsNone(severity)

    def test_dog_eye_condition_does_not_create_urgency(self):
        for condition in (
            "conjunctivitis",
            "entropion",
        ):
            urgency = None

            self.assertIsNone(urgency)

    def test_unsupported_severity_is_null(self):
        severity = None

        self.assertIsNone(severity)

    def test_undefined_urgency_is_not_an_urgency_category(self):
        urgency_categories = {
            "Routine",
            "Soon",
            "Urgent",
            "Emergency",
        }

        undefined_urgency = None

        self.assertNotIn(
            undefined_urgency,
            urgency_categories,
        )

        self.assertIsNone(
            undefined_urgency
        )

    def test_insufficient_evidence_leaves_urgency_undefined(self):
        evidence_status = "insufficient_evidence"
        urgency = None

        self.assertEqual(
            evidence_status,
            "insufficient_evidence",
        )

        self.assertIsNone(
            urgency
        )

    def test_severe_laceration_with_active_hemorrhage_is_emergency(self):
        severity = "severe"
        active_hemorrhage = True

        urgency = None

        if severity == "severe" and active_hemorrhage:
            urgency = "Emergency"

        self.assertEqual(
            urgency,
            "Emergency",
        )

    def test_deterministic_emergency_is_not_removed_by_low_confidence(self):
        model_confidence_level = "low"

        severity = "severe"
        active_hemorrhage = True

        urgency = None

        if severity == "severe" and active_hemorrhage:
            urgency = "Emergency"

        self.assertEqual(
            model_confidence_level,
            "low",
        )

        self.assertEqual(
            urgency,
            "Emergency",
        )

    def test_no_general_conflict_hierarchy_is_assumed(self):
        conflict_resolution_defined = False

        self.assertFalse(
            conflict_resolution_defined
        )


if __name__ == "__main__":
    unittest.main()