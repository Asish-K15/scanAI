from app.services.fusion_urgency import APPROVED_EVIDENCE_STATUSES, determine_urgency


def build_recommendation(
    species,
    body_area,
    model_result,
    *,
    severity=None,
    active_hemorrhage=False,
    low_risk_evidence=None,
    observed_condition=None,
    conflicts=None,
    clinical_evidence=None,
):
    """
    Transform model output into the approved v1
    Fusion / Urgency recommendation schema.

    low_risk_evidence is an independently established upstream
    technical input. This layer does not derive it from:
    - Dog Eye condition
    - model confidence
    - confidence level
    - severity

    severity and active_hemorrhage are independently established
    evidence inputs for approved urgency rules.

    The frozen Dog Eye model output is preserved.
    """

    condition = model_result.get("condition")
    confidence = model_result.get("confidence")
    confidence_level = model_result.get("confidence_level")
    uncertain = model_result.get("uncertain")

    # If observed_condition is provided from clinical evidence, use it for urgency rule evaluation
    eval_condition = observed_condition or condition

    urgency = determine_urgency(
        severity=severity,
        condition=eval_condition,
        body_area=body_area,
        active_hemorrhage=active_hemorrhage,
        confidence_level=confidence_level,
        evidence_status=None,
        low_risk_evidence=low_risk_evidence,
        conflicts=conflicts,
    )

    if conflicts is not None and len(conflicts) > 0:
        urgency = None
        evidence_status = "insufficient_evidence"
        recommendation_text = "conflicting evidence / urgency undefined"

    elif urgency == "Emergency":
        evidence_status = "approved_urgency_evidence"
        recommendation_text = "emergency / immediate veterinary attention"

    elif urgency == "Routine":
        evidence_status = "approved_urgency_evidence"
        recommendation_text = "routine / non-urgent monitoring"

    else:
        urgency = None
        evidence_status = "insufficient_evidence"
        recommendation_text = (
            "insufficient evidence / urgency undefined"
        )

    evidence = {
        "condition": condition,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "uncertain": uncertain,
        "probabilities": model_result.get("probabilities"),
        "model": model_result.get("model"),
        "model_version": model_result.get("model_version"),
        "engine": model_result.get("engine"),
        "screening_only": model_result.get("screening_only"),
    }

    if conflicts is not None and len(conflicts) > 0:
        evidence["conflicts"] = list(conflicts)

    if clinical_evidence is not None:
        evidence["clinical_evidence"] = clinical_evidence

    return {
        "species": species,
        "body_area": body_area,
        "condition": condition,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "uncertain": uncertain,
        "severity": severity,
        "urgency": urgency,
        "evidence_status": evidence_status,
        "evidence": evidence,
        "recommendation": recommendation_text,
    }