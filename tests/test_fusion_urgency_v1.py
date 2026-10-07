"""
ScanAI Fusion / Urgency v1 Test Suite.

Implements and verifies the 15 approved Fusion/Urgency v1 test cases
(TC-FUS-01 through TC-FUS-15) and input contract validations, conforming
strictly to docs/FUSION_URGENCY_SPEC.md (at commit 294b587) and the
Authoritative Architecture Decision v1:
- Fusion / Urgency is an internal service boundary (fuse_triage, determine_urgency).
- Public integration point is POST /api/predict.
- Production recommendation and evidence dictionary representation are preserved.
- Exactly 11 top-level response fields.
- APPROVED_EVIDENCE_STATUSES = {"approved_urgency_evidence", "insufficient_evidence"}.
"""

import json
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.fusion_urgency import (
    APPROVED_EVIDENCE_STATUSES,
    determine_urgency,
    fuse_triage,
)
from app.services.recommendation import build_recommendation


class FakeGateService:
    def validate(self, image, selected_species, **kwargs):
        return {
            "decision": "ACCEPT",
            "predicted_species": selected_species,
            "species_confidence": 0.99,
            "animal_probability": 0.99,
            "model_name": "EfficientNet-B0",
            "model_version": "SCANAI-ANIMALNESS-GATE-V1",
            "reason_code": "SUPPORTED_ANIMAL_DETECTED",
            "error_code": None,
        }


class FakeSkinModel:
    def predict(self, image):
        return {
            "condition": "skin__hotspot",
            "confidence": 0.50,
            "confidence_level": "low",
            "uncertain": True,
            "probabilities": {
                "skin__hotspot": 0.50,
                "skin__pyoderma": 0.50,
            },
            "model": "EfficientNet-B0",
            "model_version": "SCANAI-SKIN-PHASE4B",
            "engine": "PyTorch",
            "screening_only": True,
        }


class FakeDogEyeModel:
    def predict(self, image):
        return {
            "condition": "conjunctivitis",
            "confidence": 0.72,
            "confidence_level": "moderate",
            "uncertain": False,
            "probabilities": {
                "conjunctivitis": 0.72,
                "entropion": 0.28,
            },
            "model": "EfficientNet-B0",
            "model_version": "dog-eye-v1",
            "engine": "ONNX Runtime",
            "screening_only": True,
        }


def make_test_image_bytes():
    image = Image.new("RGB", (10, 10), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestFusionUrgencyV1(unittest.TestCase):
    """
    15 Approved Fusion / Urgency v1 Tests (TC-FUS-01 through TC-FUS-15)
    and internal service / API integration contract tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.approved_keys = {
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

    def test_tc_fus_01_emergency_endpoint(self):
        """
        TC-FUS-01: Severe deep-tissue laceration + active hemorrhage -> Emergency.
        Approved recommendation: 'emergency / immediate veterinary attention'.
        Full emergency evidence preserved; severity preserved as 'severe'.
        """
        payload = {
            "species": "dog",
            "body_area": "skin",
            "model_prediction": {
                "condition": "skin__hotspot",
                "confidence": 0.45,
                "confidence_level": "low",
                "uncertain": True,
                "model": "EfficientNet-B0",
                "model_version": "SCANAI-SKIN-PHASE4B",
                "screening_only": True,
            },
            "clinical_evidence": {
                "severity": "severe",
                "observed_condition": "deep-tissue laceration",
                "active_hemorrhage": True,
            },
        }

        result = fuse_triage(payload)
        self.assertEqual(result["urgency"], "Emergency")
        self.assertEqual(result["evidence_status"], "approved_urgency_evidence")
        self.assertEqual(
            result["recommendation"],
            "emergency / immediate veterinary attention",
        )
        self.assertEqual(result["severity"], "severe")
        self.assertIsInstance(result["evidence"], dict)
        self.assertEqual(result["evidence"]["condition"], "skin__hotspot")
        self.assertIn("clinical_evidence", result["evidence"])
        self.assertEqual(
            result["evidence"]["clinical_evidence"]["observed_condition"],
            "deep-tissue laceration",
        )

    def test_tc_fus_02_routine_endpoint(self):
        """
        TC-FUS-02: Independently established low_risk_evidence = true -> Routine.
        Approved recommendation: 'routine / non-urgent monitoring'.
        Model prediction preserved in output; severity remains null.
        """
        payload = {
            "species": "cat",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.88,
                "confidence_level": "high",
                "uncertain": False,
                "model": "EfficientNet-B0",
                "model_version": "scanai_cat_eye_efficientnet_b0_v1",
                "screening_only": True,
            },
            "clinical_evidence": {
                "low_risk_evidence": True,
            },
        }

        result = fuse_triage(payload)
        self.assertEqual(result["urgency"], "Routine")
        self.assertEqual(result["evidence_status"], "approved_urgency_evidence")
        self.assertEqual(
            result["recommendation"],
            "routine / non-urgent monitoring",
        )
        self.assertIsNone(result["severity"])
        self.assertIsInstance(result["evidence"], dict)

    def test_tc_fus_03_routine_exception_emergency_precedence(self):
        """
        TC-FUS-03: Emergency signals present AND low_risk_evidence = true -> Emergency.
        Routine evidence must NOT override Emergency evidence.
        """
        payload = {
            "species": "dog",
            "body_area": "skin",
            "model_prediction": {
                "condition": "skin__hotspot",
                "confidence": 0.50,
                "confidence_level": "low",
                "uncertain": True,
                "model": "EfficientNet-B0",
                "model_version": "SCANAI-SKIN-PHASE4B",
                "screening_only": True,
            },
            "clinical_evidence": {
                "severity": "severe",
                "observed_condition": "deep-tissue laceration",
                "active_hemorrhage": True,
                "low_risk_evidence": True,
            },
        }

        result = fuse_triage(payload)
        self.assertEqual(result["urgency"], "Emergency")
        self.assertEqual(result["evidence_status"], "approved_urgency_evidence")
        self.assertEqual(
            result["recommendation"],
            "emergency / immediate veterinary attention",
        )

    def test_tc_fus_04_insufficient_evidence_missing_clinical_evidence_normalized(self):
        """
        TC-FUS-04: If clinical_evidence object is absent entirely or empty, normalize to {}
        rather than rejecting the request.
        Treat as containing no independently established urgency evidence:
        urgency = null, evidence_status = 'insufficient_evidence',
        recommendation = 'insufficient evidence / urgency undefined'.
        """
        payload_omitted = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.72,
                "confidence_level": "moderate",
                "uncertain": False,
                "model": "EfficientNet-B0",
                "model_version": "dog-eye-v1",
                "screening_only": True,
            },
        }

        payload_empty = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.72,
                "confidence_level": "moderate",
                "uncertain": False,
                "model": "EfficientNet-B0",
                "model_version": "dog-eye-v1",
                "screening_only": True,
            },
            "clinical_evidence": {},
        }

        for payload in (payload_omitted, payload_empty):
            result = fuse_triage(payload)
            self.assertIsNone(result["urgency"])
            self.assertEqual(result["evidence_status"], "insufficient_evidence")
            self.assertEqual(
                result["recommendation"],
                "insufficient evidence / urgency undefined",
            )
            self.assertIsInstance(result["evidence"], dict)
            self.assertIsNone(result["severity"])

    def test_tc_fus_05_soon_undefined(self):
        """
        TC-FUS-05: Soon has no approved trigger in v1.
        A suspicious sign or high probability condition must NOT trigger Soon.
        Urgency remains null.
        """
        payload = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.85,
                "confidence_level": "high",
                "uncertain": False,
            },
            "clinical_evidence": {
                "suspicious_sign": True,
            },
        }

        result = fuse_triage(payload)
        self.assertIsNone(result["urgency"])
        self.assertNotEqual(result["urgency"], "Soon")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["recommendation"],
            "insufficient evidence / urgency undefined",
        )

    def test_tc_fus_06_urgent_undefined(self):
        """
        TC-FUS-06: Urgent has no approved trigger in v1.
        Severe sign alone (without deep-tissue laceration and active hemorrhage)
        must NOT trigger Urgent or Emergency.
        Urgency remains null, severity is preserved.
        """
        payload = {
            "species": "dog",
            "body_area": "skin",
            "model_prediction": {
                "condition": "skin__hotspot",
                "confidence": 0.80,
                "confidence_level": "high",
                "uncertain": False,
            },
            "clinical_evidence": {
                "severity": "severe",
            },
        }

        result = fuse_triage(payload)
        self.assertEqual(result["severity"], "severe")
        self.assertIsNone(result["urgency"])
        self.assertNotEqual(result["urgency"], "Urgent")
        self.assertNotEqual(result["urgency"], "Emergency")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["recommendation"],
            "insufficient evidence / urgency undefined",
        )

    def test_tc_fus_07_confidence_independence(self):
        """
        TC-FUS-07: Numeric confidence (e.g. 0.9999) must not determine urgency or severity.
        """
        payload = {
            "species": "cat",
            "body_area": "eye",
            "model_prediction": {
                "condition": "blepharitis",
                "confidence": 0.9999,
                "confidence_level": "high",
                "uncertain": False,
            },
            "clinical_evidence": {},
        }

        result = fuse_triage(payload)
        self.assertEqual(result["confidence"], 0.9999)
        self.assertIsNone(result["urgency"])
        self.assertIsNone(result["severity"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")

    def test_tc_fus_08_confidence_thresholds_independence(self):
        """
        TC-FUS-08: Engineering confidence thresholds (high, moderate, low)
        must NOT determine clinical urgency.
        """
        for level in ("low", "moderate", "high"):
            payload = {
                "species": "dog",
                "body_area": "eye",
                "model_prediction": {
                    "condition": "conjunctivitis",
                    "confidence": 0.85 if level == "high" else (0.65 if level == "moderate" else 0.40),
                    "confidence_level": level,
                    "uncertain": False,
                },
                "clinical_evidence": {},
            }
            result = fuse_triage(payload)
            self.assertEqual(result["confidence_level"], level)
            self.assertIsNone(result["urgency"])
            self.assertEqual(result["evidence_status"], "insufficient_evidence")

    def test_tc_fus_09_dog_eye_condition_independence(self):
        """
        TC-FUS-09: Dog Eye condition (e.g. entropion) does not imply severity or urgency.
        Both severity and urgency remain null.
        """
        payload = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "entropion",
                "confidence": 0.92,
                "confidence_level": "high",
                "uncertain": False,
            },
            "clinical_evidence": {},
        }

        result = fuse_triage(payload)
        self.assertIsNone(result["severity"])
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")

    def test_tc_fus_10_skin_screening_boundary(self):
        """
        TC-FUS-10: Skin model remains screening-only.
        No severity or urgency mappings allowed.
        """
        payload = {
            "species": "dog",
            "body_area": "skin",
            "model_prediction": {
                "condition": "skin__pyoderma",
                "confidence": 0.91,
                "confidence_level": "high",
                "uncertain": False,
            },
            "clinical_evidence": {},
        }

        result = fuse_triage(payload)
        self.assertIsNone(result["severity"])
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")

    def test_tc_fus_11_unresolved_conflict_handling(self):
        """
        TC-FUS-11: Conflicting evidence preserved in evidence dictionary,
        urgency = null, evidence_status = 'insufficient_evidence',
        recommendation = 'conflicting evidence / urgency undefined'.
        """
        conflicts = [
            "deep-tissue laceration reported by owner",
            "superficial abrasion observed by clinician",
        ]
        payload = {
            "species": "dog",
            "body_area": "skin",
            "model_prediction": {
                "condition": "skin__hotspot",
                "confidence": 0.70,
                "confidence_level": "moderate",
                "uncertain": False,
            },
            "clinical_evidence": {
                "conflicts": conflicts,
            },
        }

        result = fuse_triage(payload)
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["recommendation"],
            "conflicting evidence / urgency undefined",
        )
        self.assertIsInstance(result["evidence"], dict)
        self.assertIn("conflicts", result["evidence"])
        self.assertEqual(result["evidence"]["conflicts"], conflicts)

    def test_tc_fus_12_no_highest_confidence_conflict_resolution(self):
        """
        TC-FUS-12: Contradictory signals with differing confidence scores (0.95 vs 0.50).
        Prohibits selecting the winning signal by confidence.
        Urgency must remain null, recommendation must state conflicting evidence.
        """
        conflicts = [
            "model prediction conjunctivitis (confidence: 0.95)",
            "clinical observation foreign body (confidence: 0.50)",
        ]
        payload = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.95,
                "confidence_level": "high",
                "uncertain": False,
            },
            "clinical_evidence": {
                "conflicts": conflicts,
            },
        }

        result = fuse_triage(payload)
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["recommendation"],
            "conflicting evidence / urgency undefined",
        )
        self.assertEqual(result["evidence"]["conflicts"], conflicts)

    def test_tc_fus_13_no_mathematical_fusion(self):
        """
        TC-FUS-13: Mathematical fusion formulas (weighted_score, risk_score, etc.)
        must not determine urgency or severity.
        """
        payload = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.85,
                "confidence_level": "high",
                "uncertain": False,
                "weighted_score": 0.90,
                "urgency_score": 0.88,
            },
            "clinical_evidence": {
                "risk_score": 0.92,
            },
        }

        result = fuse_triage(payload)
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["recommendation"],
            "insufficient evidence / urgency undefined",
        )

    def test_tc_fus_14_prohibited_assumptions_low_confidence_uncertainty(self):
        """
        TC-FUS-14: confidence = 0.20, uncertain = True.
        Low confidence / uncertainty does not create Routine or high urgency.
        """
        payload = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.20,
                "confidence_level": "low",
                "uncertain": True,
            },
            "clinical_evidence": {},
        }

        result = fuse_triage(payload)
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["confidence"], 0.20)
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["recommendation"],
            "insufficient evidence / urgency undefined",
        )

    def test_tc_fus_15_exact_11_top_level_output_keys_schema(self):
        """
        TC-FUS-15: Final Recommendation Schema verification.
        Response must contain exactly the 11 approved keys, with no extra keys,
        and evidence_status restricted to APPROVED_EVIDENCE_STATUSES.
        Evidence must be a dictionary preserving the production representation.
        """
        payload = {
            "species": "dog",
            "body_area": "skin",
            "model_prediction": {
                "condition": "skin__hotspot",
                "confidence": 0.75,
                "confidence_level": "moderate",
                "uncertain": False,
            },
            "clinical_evidence": {
                "severity": "moderate",
            },
        }

        result = fuse_triage(payload)
        self.assertEqual(set(result.keys()), self.approved_keys)
        self.assertEqual(len(result.keys()), 11)
        self.assertIsInstance(result["evidence"], dict)
        self.assertIn(result["evidence_status"], APPROVED_EVIDENCE_STATUSES)

    # -------------------------------------------------------------------------
    # Internal Service Input Contract Tests
    # -------------------------------------------------------------------------

    def test_service_01_missing_model_prediction_raises_value_error(self):
        """
        Missing model_prediction raises ValueError.
        """
        payload = {
            "species": "dog",
            "body_area": "eye",
            "clinical_evidence": {},
        }
        with self.assertRaises(ValueError) as ctx:
            fuse_triage(payload)
        self.assertIn("model_prediction is required", str(ctx.exception))

    def test_service_02_null_model_prediction_raises_value_error(self):
        """
        Null model_prediction raises ValueError.
        """
        payload = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": None,
            "clinical_evidence": {},
        }
        with self.assertRaises(ValueError) as ctx:
            fuse_triage(payload)
        self.assertIn("model_prediction is required", str(ctx.exception))

    def test_service_03_non_object_model_prediction_raises_value_error(self):
        """
        Non-object model_prediction raises ValueError.
        """
        for bad_mp in ("not-a-dict", [1, 2, 3], 42):
            payload = {
                "species": "dog",
                "body_area": "eye",
                "model_prediction": bad_mp,
                "clinical_evidence": {},
            }
            with self.assertRaises(ValueError) as ctx:
                fuse_triage(payload)
            self.assertIn("model_prediction must be a dictionary", str(ctx.exception))

    def test_service_04_null_clinical_evidence_accepted_and_normalized(self):
        """
        Explicit null clinical_evidence is accepted and normalized to {}.
        """
        payload = {
            "species": "dog",
            "body_area": "eye",
            "model_prediction": {
                "condition": "conjunctivitis",
                "confidence": 0.72,
                "confidence_level": "moderate",
                "uncertain": False,
            },
            "clinical_evidence": None,
        }
        result = fuse_triage(payload)
        self.assertIsNone(result["urgency"])
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["recommendation"],
            "insufficient evidence / urgency undefined",
        )
        self.assertIsInstance(result["evidence"], dict)
        self.assertIsNone(result["severity"])

    def test_service_05_malformed_clinical_evidence_raises_value_error(self):
        """
        Non-dict present clinical_evidence raises ValueError.
        """
        for bad_ce in ("severe", ["deep-tissue laceration"], 123):
            payload = {
                "species": "dog",
                "body_area": "eye",
                "model_prediction": {
                    "condition": "conjunctivitis",
                    "confidence": 0.85,
                    "confidence_level": "high",
                    "uncertain": False,
                },
                "clinical_evidence": bad_ce,
            }
            with self.assertRaises(ValueError) as ctx:
                fuse_triage(payload)
            self.assertIn("clinical_evidence must be a dictionary", str(ctx.exception))

    def test_service_06_top_level_non_dict_raises_value_error(self):
        """
        Top-level non-dict payload raises ValueError.
        """
        for bad_payload in ("invalid", [1, 2], 42, True):
            with self.assertRaises(ValueError) as ctx:
                fuse_triage(bad_payload)
            self.assertIn("Request payload must be a JSON object", str(ctx.exception))

    def test_service_07_invalid_active_hemorrhage_raises_value_error(self):
        """
        Non-boolean active_hemorrhage raises ValueError and does not coerce strings.
        """
        for bad_ah in ("false", "true", "yes", 1, 0, [True]):
            payload = {
                "species": "dog",
                "body_area": "skin",
                "model_prediction": {
                    "condition": "skin__hotspot",
                    "confidence": 0.50,
                    "confidence_level": "low",
                    "uncertain": True,
                },
                "clinical_evidence": {
                    "active_hemorrhage": bad_ah,
                },
            }
            with self.assertRaises(ValueError) as ctx:
                fuse_triage(payload)
            self.assertIn("active_hemorrhage must be a boolean", str(ctx.exception))

    def test_service_08_invalid_low_risk_evidence_raises_value_error(self):
        """
        Non-boolean low_risk_evidence raises ValueError and does not coerce strings.
        """
        for bad_lre in ("false", "true", "yes", 1, 0, [True]):
            payload = {
                "species": "dog",
                "body_area": "eye",
                "model_prediction": {
                    "condition": "conjunctivitis",
                    "confidence": 0.70,
                    "confidence_level": "moderate",
                    "uncertain": False,
                },
                "clinical_evidence": {
                    "low_risk_evidence": bad_lre,
                },
            }
            with self.assertRaises(ValueError) as ctx:
                fuse_triage(payload)
            self.assertIn("low_risk_evidence must be a boolean", str(ctx.exception))

    def test_service_09_invalid_severity_raises_value_error(self):
        """
        Severity must only be one of the approved values: mild, moderate, severe.
        """
        for bad_sev in ("critical", "high", "unknown", 123):
            payload = {
                "species": "dog",
                "body_area": "skin",
                "model_prediction": {
                    "condition": "skin__hotspot",
                    "confidence": 0.70,
                    "confidence_level": "moderate",
                    "uncertain": False,
                },
                "clinical_evidence": {
                    "severity": bad_sev,
                },
            }
            with self.assertRaises(ValueError) as ctx:
                fuse_triage(payload)
            self.assertIn("severity must be one of: mild, moderate, severe", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Public Integration Point Tests: POST /api/predict
    # -------------------------------------------------------------------------

    def test_api_predict_emergency_integration(self):
        """
        POST /api/predict correctly integrates clinical evidence inputs:
        severe + deep-tissue laceration + active hemorrhage -> Emergency.
        """
        image_bytes = make_test_image_bytes()
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {
            "species": "dog",
            "body_area": "skin",
            "severity": "severe",
            "observed_condition": "deep-tissue laceration",
            "active_hemorrhage": "true",
            "low_risk_evidence": "false",
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=FakeGateService()):
            with patch("app.routers.predict.get_skin_model", return_value=FakeSkinModel()):
                response = self.client.post("/api/predict", files=files, data=data)

        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["urgency"], "Emergency")
        self.assertEqual(res["severity"], "severe")
        self.assertEqual(res["evidence_status"], "approved_urgency_evidence")
        self.assertEqual(res["recommendation"], "emergency / immediate veterinary attention")
        self.assertEqual(set(res.keys()), self.approved_keys)
        self.assertIsInstance(res["evidence"], dict)

    def test_api_predict_routine_integration(self):
        """
        POST /api/predict correctly integrates low_risk_evidence=true -> Routine.
        """
        image_bytes = make_test_image_bytes()
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {
            "species": "dog",
            "body_area": "eye",
            "low_risk_evidence": "true",
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=FakeGateService()):
            with patch("app.routers.predict.get_dog_eye_model", return_value=FakeDogEyeModel()):
                response = self.client.post("/api/predict", files=files, data=data)

        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["urgency"], "Routine")
        self.assertEqual(res["evidence_status"], "approved_urgency_evidence")
        self.assertEqual(res["recommendation"], "routine / non-urgent monitoring")
        self.assertEqual(set(res.keys()), self.approved_keys)
        self.assertIsInstance(res["evidence"], dict)

    def test_api_predict_conflict_integration(self):
        """
        POST /api/predict correctly preserves conflicts and leaves urgency undefined.
        """
        image_bytes = make_test_image_bytes()
        conflicts = ["owner reports laceration", "clinician observes abrasion"]
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {
            "species": "dog",
            "body_area": "skin",
            "conflicts": json.dumps(conflicts),
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=FakeGateService()):
            with patch("app.routers.predict.get_skin_model", return_value=FakeSkinModel()):
                response = self.client.post("/api/predict", files=files, data=data)

        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertIsNone(res["urgency"])
        self.assertEqual(res["evidence_status"], "insufficient_evidence")
        self.assertEqual(res["recommendation"], "conflicting evidence / urgency undefined")
        self.assertEqual(set(res.keys()), self.approved_keys)
        self.assertIsInstance(res["evidence"], dict)
        self.assertEqual(res["evidence"]["conflicts"], conflicts)

    def test_api_predict_rejects_invalid_active_hemorrhage_string(self):
        """
        POST /api/predict rejects invalid non-boolean active_hemorrhage string with 400.
        """
        image_bytes = make_test_image_bytes()
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {
            "species": "dog",
            "body_area": "skin",
            "active_hemorrhage": "garbage",
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=FakeGateService()):
            with patch("app.routers.predict.get_skin_model", return_value=FakeSkinModel()):
                response = self.client.post("/api/predict", files=files, data=data)

        self.assertEqual(response.status_code, 400)
        self.assertIn("active_hemorrhage", response.json()["detail"])

    def test_api_predict_rejects_invalid_low_risk_evidence_string(self):
        """
        POST /api/predict rejects invalid non-boolean low_risk_evidence string with 400.
        """
        image_bytes = make_test_image_bytes()
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {
            "species": "dog",
            "body_area": "eye",
            "low_risk_evidence": "garbage",
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=FakeGateService()):
            with patch("app.routers.predict.get_dog_eye_model", return_value=FakeDogEyeModel()):
                response = self.client.post("/api/predict", files=files, data=data)

        self.assertEqual(response.status_code, 400)
        self.assertIn("low_risk_evidence", response.json()["detail"])

    def test_api_predict_rejects_invalid_severity_string(self):
        """
        POST /api/predict rejects unapproved severity value with 400.
        """
        image_bytes = make_test_image_bytes()
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        data = {
            "species": "dog",
            "body_area": "skin",
            "severity": "critical",
        }

        with patch("app.routers.predict.get_animalness_gate_service", return_value=FakeGateService()):
            with patch("app.routers.predict.get_skin_model", return_value=FakeSkinModel()):
                response = self.client.post("/api/predict", files=files, data=data)

        self.assertEqual(response.status_code, 400)
        self.assertIn("severity", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
