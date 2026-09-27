"""
Animalness Gate Service.

Provides animalness and species validation prior to routing to disease models.
Follows the ScanAI / PetVision AI Animalness Gate PRD requirements.
"""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from animalness_gate.schemas import (
    AnimalnessGateResponse,
    GateDecision,
    GateReasonCode,
)


class AnimalnessGateConfig:
    """
    Configuration for Animalness Gate decision boundaries.

    IMPORTANT: Per PRD Section 9, threshold analysis is currently diagnostic only.
    No production threshold has been approved. The gate is structured so thresholds
    and uncertainty boundaries can be configured without rewriting the service.
    """

    def __init__(
        self,
        animal_threshold: Optional[float] = None,
        uncertainty_min: Optional[float] = None,
        uncertainty_max: Optional[float] = None,
        species_confidence_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.animal_threshold = animal_threshold
        self.uncertainty_min = uncertainty_min
        self.uncertainty_max = uncertainty_max
        self.species_confidence_threshold = species_confidence_threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def is_threshold_configured(self) -> bool:
        """Returns True only when an explicit animalness threshold has been configured."""
        return self.animal_threshold is not None


class AnimalnessGateService:
    """
    Animalness Gate validator combining animalness detection (binary)
    and species classification (dog, cat, cattle).
    """

    MODEL_NAME = "EfficientNet-B0"
    MODEL_VERSION = "SCANAI-ANIMALNESS-GATE-V1"

    SUPPORTED_SPECIES: Set[str] = {"dog", "cat", "cattle"}
    GATE_CLASSES: List[str] = ["animal", "not_animal"]
    SPECIES_CLASSES: List[str] = ["dog", "cat", "cattle"]

    IMAGE_SIZE: int = 224

    def __init__(
        self,
        config: Optional[AnimalnessGateConfig] = None,
        gate_checkpoint_path: Optional[Union[str, Path]] = None,
        species_checkpoint_path: Optional[Union[str, Path]] = None,
        lazy_load: bool = False,
    ):
        self.config = config or AnimalnessGateConfig()
        self.device = self.config.device

        base_dir = Path(__file__).resolve().parent.parent

        self.gate_checkpoint_path = Path(
            gate_checkpoint_path
            or (base_dir / "animalness_gate" / "outputs" / "best_animalness_gate.pth")
        )
        self.species_checkpoint_path = Path(
            species_checkpoint_path
            or (base_dir / "animal_validator" / "outputs" / "best_species_validator.pth")
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._gate_model: Optional[nn.Module] = None
        self._species_model: Optional[nn.Module] = None

        if not lazy_load:
            self.load_models()

    def load_models(self) -> None:
        """Load both the animalness gate model and the species validator."""
        self._load_gate_model()
        self._load_species_model()

    def _load_gate_model(self) -> None:
        if not self.gate_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Animalness gate checkpoint not found: {self.gate_checkpoint_path}"
            )

        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(self.GATE_CLASSES))

        checkpoint = torch.load(
            self.gate_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self._gate_model = model

    def _load_species_model(self) -> None:
        if not self.species_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Species validator checkpoint not found: {self.species_checkpoint_path}"
            )

        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(self.SPECIES_CLASSES))

        checkpoint = torch.load(
            self.species_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self._species_model = model

    def _ensure_models_loaded(self) -> None:
        if self._gate_model is None:
            self._load_gate_model()
        if self._species_model is None:
            self._load_species_model()

    def _prepare_image(
        self,
        image_input: Union[Image.Image, bytes, BinaryIO, str, Path],
    ) -> Image.Image:
        """Convert input to a loaded RGB PIL Image."""
        if isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            pil_image.load()
            return pil_image

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            pil_image = Image.open(path).convert("RGB")
            pil_image.load()
            return pil_image

        if isinstance(image_input, bytes):
            pil_image = Image.open(BytesIO(image_input)).convert("RGB")
            pil_image.load()
            return pil_image

        if hasattr(image_input, "read"):
            data = image_input.read()
            pil_image = Image.open(BytesIO(data)).convert("RGB")
            pil_image.load()
            return pil_image

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def validate(
        self,
        image: Union[Image.Image, bytes, BinaryIO, str, Path],
        selected_species: str,
    ) -> Dict:
        """
        Validate that an uploaded image contains a supported animal and that
        the predicted species matches the user's selected species.

        Returns a dictionary adhering to the AnimalnessGateResponse contract.
        """
        # 1. Validate selected species
        clean_selected = str(selected_species).strip().lower() if selected_species else ""
        if clean_selected not in self.SUPPORTED_SPECIES:
            return AnimalnessGateResponse(
                decision=GateDecision.REJECT,
                predicted_species=None,
                species_confidence=0.0,
                animal_probability=0.0,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                reason_code=GateReasonCode.UNSUPPORTED_SPECIES,
                error_code=None,
            ).model_dump()

        # 2. Validate and load image
        try:
            pil_image = self._prepare_image(image)
        except (UnidentifiedImageError, OSError, ValueError, FileNotFoundError) as exc:
            return AnimalnessGateResponse(
                decision=GateDecision.REJECT,
                predicted_species=None,
                species_confidence=0.0,
                animal_probability=0.0,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                reason_code=GateReasonCode.INVALID_IMAGE,
                error_code="INVALID_IMAGE",
            ).model_dump()

        # 3. Model inference with controlled error handling
        try:
            self._ensure_models_loaded()
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Animalness gate model
                gate_logits = self._gate_model(tensor)
                gate_probs = torch.softmax(gate_logits, dim=1)[0]
                animal_probability = float(gate_probs[0].item())

                # Species validator model
                species_logits = self._species_model(tensor)
                species_probs = torch.softmax(species_logits, dim=1)[0]
                species_conf, species_idx = torch.max(species_probs, dim=0)
                predicted_species = self.SPECIES_CLASSES[species_idx.item()]
                species_confidence = float(species_conf.item())

        except Exception as exc:
            return AnimalnessGateResponse(
                decision=GateDecision.REJECT,
                predicted_species=None,
                species_confidence=0.0,
                animal_probability=0.0,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                reason_code=GateReasonCode.MODEL_ERROR,
                error_code=str(exc),
            ).model_dump()

        # 4. Enforce PRD requirement: No implicit fallback threshold.
        # If no approved production animalness threshold is configured, return UNCERTAIN with LOW_CONFIDENCE.
        if not self.config.is_threshold_configured:
            return AnimalnessGateResponse(
                decision=GateDecision.UNCERTAIN,
                predicted_species=predicted_species,
                species_confidence=species_confidence,
                animal_probability=animal_probability,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                reason_code=GateReasonCode.LOW_CONFIDENCE,
                error_code=None,
            ).model_dump()

        # 5. Check uncertainty bounds or species confidence threshold if configured
        is_uncertain = False
        if (
            self.config.uncertainty_min is not None
            and self.config.uncertainty_max is not None
            and self.config.uncertainty_min <= animal_probability <= self.config.uncertainty_max
        ):
            is_uncertain = True

        if (
            self.config.species_confidence_threshold is not None
            and species_confidence < self.config.species_confidence_threshold
        ):
            is_uncertain = True

        if is_uncertain:
            return AnimalnessGateResponse(
                decision=GateDecision.UNCERTAIN,
                predicted_species=predicted_species,
                species_confidence=species_confidence,
                animal_probability=animal_probability,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                reason_code=GateReasonCode.LOW_CONFIDENCE,
                error_code=None,
            ).model_dump()

        # 6. Evaluate animalness against the explicitly configured threshold.
        # The binary model separates animals from non-animals (including humans and objects).
        # In the absence of an independent human detector, generic not-animal rejections return NON_ANIMAL_DETECTED.
        if animal_probability < self.config.animal_threshold:
            return AnimalnessGateResponse(
                decision=GateDecision.REJECT,
                predicted_species=predicted_species,
                species_confidence=species_confidence,
                animal_probability=animal_probability,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                reason_code=GateReasonCode.NON_ANIMAL_DETECTED,
                error_code=None,
            ).model_dump()

        # 7. Animal confirmed -> Verify species match (NEVER silently reroute)
        if predicted_species.lower() != clean_selected:
            return AnimalnessGateResponse(
                decision=GateDecision.REJECT,
                predicted_species=predicted_species,
                species_confidence=species_confidence,
                animal_probability=animal_probability,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                reason_code=GateReasonCode.SPECIES_MISMATCH,
                error_code=None,
            ).model_dump()

        # 8. Supported animal and species matched -> ACCEPT
        return AnimalnessGateResponse(
            decision=GateDecision.ACCEPT,
            predicted_species=predicted_species,
            species_confidence=species_confidence,
            animal_probability=animal_probability,
            model_name=self.MODEL_NAME,
            model_version=self.MODEL_VERSION,
            reason_code=GateReasonCode.SUPPORTED_ANIMAL_DETECTED,
            error_code=None,
        ).model_dump()


_animalness_gate_service: Optional[AnimalnessGateService] = None


def get_animalness_gate_service(
    config: Optional[AnimalnessGateConfig] = None,
    force_reload: bool = False,
) -> AnimalnessGateService:
    """Get or create singleton AnimalnessGateService instance."""
    global _animalness_gate_service

    if _animalness_gate_service is None or force_reload or config is not None:
        _animalness_gate_service = AnimalnessGateService(config=config)

    return _animalness_gate_service
