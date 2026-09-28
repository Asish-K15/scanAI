"""
Schemas and types for the Animalness Gate service.
Adheres strictly to the ScanAI Animalness Gate PRD contract.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GateDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    UNCERTAIN = "UNCERTAIN"


class GateReasonCode(str, Enum):
    SUPPORTED_ANIMAL_DETECTED = "SUPPORTED_ANIMAL_DETECTED"
    HUMAN_DETECTED = "HUMAN_DETECTED"
    NON_ANIMAL_DETECTED = "NON_ANIMAL_DETECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    SPECIES_MISMATCH = "SPECIES_MISMATCH"
    UNSUPPORTED_SPECIES = "UNSUPPORTED_SPECIES"
    INVALID_IMAGE = "INVALID_IMAGE"
    MODEL_ERROR = "MODEL_ERROR"


class AnimalnessGateResponse(BaseModel):
    """
    Structured response contract for Animalness Gate.
    """
    decision: GateDecision = Field(
        ...,
        description="Gate decision: ACCEPT, REJECT, or UNCERTAIN",
    )
    predicted_species: Optional[str] = Field(
        None,
        description="Predicted species from the validator ('dog', 'cat', 'cattle', or None)",
    )
    species_confidence: float = Field(
        0.0,
        description="Confidence score for predicted species [0.0 - 1.0]",
    )
    animal_probability: float = Field(
        0.0,
        description="Probability that the image contains an animal [0.0 - 1.0]",
    )
    model_name: str = Field(
        "EfficientNet-B0",
        description="Name of the underlying animalness backbone architecture",
    )
    model_version: str = Field(
        "SCANAI-ANIMALNESS-GATE-V1",
        description="Model checkpoint version",
    )
    reason_code: GateReasonCode = Field(
        ...,
        description="Standardized reason code explaining the decision",
    )
    error_code: Optional[str] = Field(
        None,
        description="Controlled error details if an error occurred, otherwise null",
    )

    model_config = {
        "use_enum_values": True,
    }
