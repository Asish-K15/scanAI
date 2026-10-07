"""
ScanAI Fusion / Urgency Engine.

This module implements only the explicitly approved v1
Fusion/Urgency rules.

Important:
- Model confidence is NOT urgency.
- Model confidence is NOT severity.
- Dog Eye conditions do not imply severity or urgency.
- None urgency means insufficient approved evidence.
- No general conflict hierarchy is implemented.
- low_risk_evidence is an independently established upstream
  technical input; this engine does not derive it.
"""


URGENCY_CATEGORIES = {
    "Routine",
    "Soon",
    "Urgent",
    "Emergency",
}

APPROVED_EVIDENCE_STATUSES = {
    "approved_urgency_evidence",
    "insufficient_evidence",
}


def determine_urgency(
    *,
    severity: str | None = None,
    condition: str | None = None,
    body_area: str | None = None,
    active_hemorrhage: bool = False,
    confidence_level: str | None = None,
    evidence_status: str | None = None,
    low_risk_evidence: bool | None = None,
    conflicts: list | None = None,
) -> str | None:
    """
    Determine application urgency using only approved v1 rules.

    Returns:
        "Routine", "Soon", "Urgent", "Emergency", or None.

    None means that there is insufficient approved evidence to assign
    one of the four application urgency categories.

    low_risk_evidence:
        An independently established upstream/deterministic technical
        input. The Fusion/Urgency engine does not derive this value
        from Dog Eye condition, model confidence, confidence level,
        or severity.

    The following are deliberately NOT implemented:
        - confidence -> urgency conversion
        - Dog Eye condition -> urgency
        - severity -> urgency mappings other than the approved rule
        - general conflict hierarchy
        - mathematical fusion of evidence
        - Soon assignment
        - Urgent assignment
    """

    # ---------------------------------------------------------
    # Unresolved conflicting signals leave urgency undefined.
    # ---------------------------------------------------------
    if conflicts:
        return None

    # ---------------------------------------------------------
    # Approved deterministic Emergency rule
    #
    # severe + deep-tissue laceration + active hemorrhage
    # -> Emergency
    #
    # Emergency is evaluated before Routine so that
    # low_risk_evidence cannot override independently
    # established Emergency evidence.
    # ---------------------------------------------------------

    if (
        severity == "severe"
        and condition == "deep-tissue laceration"
        and active_hemorrhage is True
    ):
        return "Emergency"

    # ---------------------------------------------------------
    # Approved Routine rule
    #
    # low_risk_evidence=True means that an independent
    # upstream/deterministic assessment has already established
    # that the case meets the project's minor/low-risk criterion.
    #
    # This engine does not derive low_risk_evidence.
    # ---------------------------------------------------------

    if low_risk_evidence is True:
        return "Routine"

    # ---------------------------------------------------------
    # Dog Eye v1
    #
    # conjunctivitis / entropion do not provide severity or
    # urgency by themselves.
    # ---------------------------------------------------------

    if (
        body_area == "eye"
        and condition in {
            "conjunctivitis",
            "entropion",
        }
    ):
        return None

    # ---------------------------------------------------------
    # Confidence is never converted into urgency.
    # ---------------------------------------------------------

    if confidence_level in {
        "low",
        "moderate",
        "high",
    }:
        return None

    # ---------------------------------------------------------
    # Insufficient evidence means urgency remains undefined.
    # ---------------------------------------------------------

    if evidence_status == "insufficient_evidence":
        return None

    # ---------------------------------------------------------
    # No additional v1 urgency mappings are approved yet.
    #
    # Soon and Urgent remain intentionally undefined.
    # ---------------------------------------------------------

    return None


def fuse_triage(payload: dict) -> dict:
    """
    Internal service function for Fusion / Urgency v1.

    Consumes internal structured prediction and clinical evidence payloads,
    validates client inputs, applies approved v1 fusion rules, and returns
    the 11-field recommendation dictionary preserving the production
    evidence data structure.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request payload must be a JSON object.")

    if "model_prediction" not in payload:
        raise ValueError("model_prediction is required.")

    model_pred = payload["model_prediction"]
    if model_pred is None:
        raise ValueError("model_prediction is required.")

    if not isinstance(model_pred, dict):
        raise ValueError("model_prediction must be a dictionary.")

    clinical_ev = payload.get("clinical_evidence")
    if clinical_ev is not None and not isinstance(clinical_ev, dict):
        raise ValueError("clinical_evidence must be a dictionary.")

    clinical_ev = clinical_ev or {}

    species = payload.get("species")
    body_area = payload.get("body_area")

    severity = clinical_ev.get("severity")
    if severity is not None and severity not in {"mild", "moderate", "severe"}:
        raise ValueError("severity must be one of: mild, moderate, severe.")

    raw_active_hemorrhage = clinical_ev.get("active_hemorrhage")
    if raw_active_hemorrhage is None:
        active_hemorrhage = False
    elif isinstance(raw_active_hemorrhage, bool):
        active_hemorrhage = raw_active_hemorrhage
    else:
        raise ValueError("active_hemorrhage must be a boolean.")

    raw_low_risk = clinical_ev.get("low_risk_evidence")
    if raw_low_risk is None:
        low_risk_evidence = None
    elif isinstance(raw_low_risk, bool):
        low_risk_evidence = raw_low_risk
    else:
        raise ValueError("low_risk_evidence must be a boolean.")

    observed_condition = clinical_ev.get("observed_condition")
    conflicts = clinical_ev.get("conflicts")
    if conflicts is not None and not isinstance(conflicts, list):
        raise ValueError("conflicts must be a list.")

    from app.services.recommendation import build_recommendation

    return build_recommendation(
        species=species,
        body_area=body_area,
        model_result=model_pred,
        severity=severity,
        active_hemorrhage=active_hemorrhage,
        low_risk_evidence=low_risk_evidence,
        observed_condition=observed_condition,
        conflicts=conflicts,
        clinical_evidence=clinical_ev if clinical_ev else None,
    )