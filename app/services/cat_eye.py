import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class CatEyeModelNotAvailableError(RuntimeError):
    """Raised when the Cat Eye model checkpoint cannot be loaded."""


class CatEyeInvalidCropError(ValueError):
    """Raised when the input image or crop is smaller than the minimum required dimensions."""


# ============================================================
# FROZEN CAT EYE CONTRACT CONSTANTS
# ============================================================

MODEL_NAME = "scanai_cat_eye_efficientnet_b0_v1"
MODEL_VERSION = "v1.0.0"
ARCHITECTURE = "EfficientNet-B0"

EXPECTED_SHA256 = "997576aa344eeec8f03c30d4b1d501850a89921e204a6b16455e147d5bd8ab7b"

# Exact frozen six-class mapping (MUST NOT BE REORDERED OR RENAMED)
CLASS_NAMES: List[str] = [
    "cataracts",      # 0
    "conjunctivitis",  # 1
    "cornealulcer",   # 2
    "glaucoma",       # 3
    "healthy",        # 4
    "uveitis",        # 5
]

CLASS_INDEX_TO_NAME: Dict[int, str] = {
    0: "cataracts",
    1: "conjunctivitis",
    2: "cornealulcer",
    3: "glaucoma",
    4: "healthy",
    5: "uveitis",
}

CLASS_NAME_TO_INDEX: Dict[str, int] = {
    name: idx for idx, name in CLASS_INDEX_TO_NAME.items()
}

INPUT_SIZE: Tuple[int, int] = (224, 224)
MIN_IMAGE_DIM: int = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

UNCERTAINTY_THRESHOLD: float = 0.40
HIGH_CONFIDENCE_THRESHOLD: float = 0.80
MODERATE_CONFIDENCE_THRESHOLD: float = 0.60


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Calculate the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class CatEyeModel:
    """
    Inference service for the frozen ScanAI Cat Eye EfficientNet-B0 model.
    Follows the exact frozen contract v1.0.0.
    """

    CLASS_NAMES = CLASS_NAMES
    CLASS_INDEX_TO_NAME = CLASS_INDEX_TO_NAME
    CLASS_NAME_TO_INDEX = CLASS_NAME_TO_INDEX
    MODEL_NAME = MODEL_NAME
    MODEL_VERSION = MODEL_VERSION
    ARCHITECTURE = ARCHITECTURE
    EXPECTED_SHA256 = EXPECTED_SHA256
    UNCERTAINTY_THRESHOLD = UNCERTAINTY_THRESHOLD
    MIN_IMAGE_DIM = MIN_IMAGE_DIM
    INPUT_SIZE = INPUT_SIZE

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        verify_sha: bool = True,
    ):
        if checkpoint_path is None:
            project_root = Path(__file__).resolve().parents[2]
            self.checkpoint_path = (
                project_root
                / "dataset_manager"
                / "training"
                / "outputs"
                / "cat_eye"
                / "efficientnet_b0"
                / "best_checkpoint.pth"
            )
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise CatEyeModelNotAvailableError(
                f"Cat Eye checkpoint not found: {self.checkpoint_path}"
            )

        if verify_sha:
            actual_sha = compute_file_sha256(self.checkpoint_path)
            if actual_sha != self.EXPECTED_SHA256:
                raise CatEyeModelNotAvailableError(
                    f"Cat Eye checkpoint SHA-256 mismatch! "
                    f"Expected: {self.EXPECTED_SHA256}, Got: {actual_sha}"
                )

        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """Instantiate EfficientNet-B0 with 6 classes and load frozen weights."""
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(self.CLASS_NAMES))

        checkpoint = torch.load(
            self.checkpoint_path,
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
        return model

    def check_crop_protection(self, image: Image.Image) -> None:
        """
        Verify image dimensions are at least 32x32.
        Images smaller than 32x32 must be rejected before inference.
        """
        width, height = image.size
        if width < self.MIN_IMAGE_DIM or height < self.MIN_IMAGE_DIM:
            raise CatEyeInvalidCropError(
                f"Input image/crop dimensions ({width}x{height}) are smaller than "
                f"the minimum required {self.MIN_IMAGE_DIM}x{self.MIN_IMAGE_DIM} threshold."
            )

    def preprocess(self, image: Union[Image.Image, str, Path]) -> torch.Tensor:
        """
        Crop protection check followed by RGB conversion, resize to 224x224,
        and ImageNet normalization.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        self.check_crop_protection(image)

        image_rgb = image.convert("RGB")
        tensor = self.transform(image_rgb)
        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def get_confidence_level(confidence: float) -> str:
        """Categorize confidence level."""
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        if confidence >= MODERATE_CONFIDENCE_THRESHOLD:
            return "moderate"
        return "low"

    def predict(self, image: Union[Image.Image, str, Path]) -> Dict:
        """
        Run Cat Eye model inference.

        1. Rejects dimensions below 32x32 BEFORE inference.
        2. Applies RGB -> 224x224 -> ImageNet normalization.
        3. Runs EfficientNet-B0 inference.
        4. Converts logits to probabilities via softmax.
        5. Computes confidence and predicted class.
        6. Sets is_uncertain = True if confidence < 0.40, else False.
        """
        input_tensor = self.preprocess(image)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities_tensor = torch.softmax(outputs, dim=1)[0]
            probabilities_np = probabilities_tensor.cpu().numpy()
            logits_np = outputs[0].cpu().numpy()

        predicted_index = int(np.argmax(probabilities_np))
        predicted_class = self.CLASS_NAMES[predicted_index]
        confidence = float(probabilities_np[predicted_index])

        is_uncertain = bool(confidence < self.UNCERTAINTY_THRESHOLD)
        confidence_level = self.get_confidence_level(confidence)

        probabilities_dict = {
            class_name: float(probabilities_np[i])
            for i, class_name in enumerate(self.CLASS_NAMES)
        }
        logits_dict = {
            class_name: float(logits_np[i])
            for i, class_name in enumerate(self.CLASS_NAMES)
        }

        return {
            "condition": predicted_class,
            "predicted_class": predicted_class,
            "predicted_class_index": predicted_index,
            "class_index": predicted_index,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "is_uncertain": is_uncertain,
            "uncertain": is_uncertain,
            "probabilities": probabilities_dict,
            "logits": logits_dict,
            "model": self.MODEL_NAME,
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "architecture": self.ARCHITECTURE,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.EXPECTED_SHA256,
            "engine": "PyTorch",
            "screening_only": True,
        }


# ============================================================
# SINGLETON SERVICE ACCESS
# ============================================================

_cat_eye_model: Optional[CatEyeModel] = None


def get_cat_eye_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    verify_sha: bool = True,
) -> CatEyeModel:
    """Retrieve or instantiate singleton CatEyeModel."""
    global _cat_eye_model

    if _cat_eye_model is None or checkpoint_path is not None:
        model = CatEyeModel(
            checkpoint_path=checkpoint_path,
            device=device,
            verify_sha=verify_sha,
        )
        if checkpoint_path is None:
            _cat_eye_model = model
        return model

    return _cat_eye_model


def reset_cat_eye_model() -> None:
    """Reset the singleton instance (for testing)."""
    global _cat_eye_model
    _cat_eye_model = None
