def build_recommendation(species, body_area, model_result):
    """
    Transform existing model output into the approved v1
    Fusion / Urgency recommendation schema.

    This is a schema transformation layer only.
    It does not perform clinical reasoning, mathematical
    fusion, confidence-to-urgency conversion, or disease-specific
    severity/urgency mapping.
    """

    evidence = {
        "condition": model_result.get("condition"),
        "confidence": model_result.get("confidence"),
        "confidence_level": model_result.get("confidence_level"),
        "uncertain": model_result.get("uncertain"),
    }

    return {
        "species": species,
        "body_area": body_area,
        "condition": model_result.get("condition"),
        "confidence": model_result.get("confidence"),
        "confidence_level": model_result.get("confidence_level"),
        "uncertain": model_result.get("uncertain"),
        "severity": None,
        "urgency": None,
        "evidence_status": "insufficient_evidence",
        "evidence": evidence,
        "recommendation": "insufficient evidence / urgency undefined",
    }