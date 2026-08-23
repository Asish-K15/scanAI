SUPPORTED_BODY_AREAS = {
    "eye",
    "skin",
    "ear",
    "wound",
    "body_condition",
}


def route_body_area(body_area: str) -> str:
    body_area = body_area.strip().lower()

    if body_area not in SUPPORTED_BODY_AREAS:
        raise ValueError(
            f"Unsupported body area: {body_area}. "
            f"Supported areas: {sorted(SUPPORTED_BODY_AREAS)}"
        )

    return body_area
