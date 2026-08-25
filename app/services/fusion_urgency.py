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
"""




URGENCY_CATEGORIES = {
    "Routine",
    "Soon",
    "Urgent",
    "Emergency",
}


def determine_urgency(
    *,
    severity: str | None = None,
    condition: str | None = None,
    body_area: str | None = None,
    active_hemorrhage: bool = False,
    confidence_level: str | None = None,
    evidence_status: str | None = None,
) -> str | None:
    """
    Determine application urgency using only approved v1 rules.

    Returns:
        "Routine", "Soon", "Urgent", "Emergency", or None.

    None means that there is insufficient approved evidence to assign
    one of the four application urgency categories.

    The following are deliberately NOT implemented:
        - confidence -> urgency conversion
        - Dog Eye condition -> urgency
        - severity -> urgency mappings other than the approved rule
        - general conflict hierarchy
        - mathematical fusion of evidence
    """

    # ---------------------------------------------------------
    # Approved deterministic Emergency rule
    #
    # severe + deep-tissue laceration + active hemorrhage
    # -> Emergency
    # ---------------------------------------------------------

    if (
        severity == "severe"
        and condition == "deep-tissue laceration"
        and active_hemorrhage
    ):
        return "Emergency"

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
    # ---------------------------------------------------------

    return None