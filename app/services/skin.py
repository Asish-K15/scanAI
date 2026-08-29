from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class SkinModelNotAvailableError(RuntimeError):
    """Raised when the Skin model cannot be loaded."""


class SkinModel:
    CLASS_NAMES = [
        "skin__allergic_dermatitis",
        "skin__foot_and_mouth_disease",
        "skin__fungal_infection",
        "skin__healthy_skin",
        "skin__hotspot",
        "skin__lumpy_skin_disease",
        "skin__mange",
        "skin__pyoderma",
        "skin__ringworm",
        "skin__scabies",
    ]

    MODEL_VERSION = "SCANAI-SKIN-PHASE4B"

    MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            project_root
            / "dataset_manager"
            / "training"
            / "outputs"
            / "phase4_final"
            / "scanai_skin_phase4b.onnx"
        )

        if not self.model_path.exists():
            raise SkinModelNotAvailableError(
                f"Skin model artifact not found: {self.model_path}"
            )

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise SkinModelNotAvailableError(
                f"Unable to load Skin ONNX model: {exc}"
            ) from exc

        inputs = self.session.get_inputs()

        if not inputs:
            raise SkinModelNotAvailableError(
                "Skin ONNX model has no inputs."
            )

        self.input_name = inputs[0].name

    @staticmethod
    def _softmax(logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(
            exp_logits,
            axis=1,
            keepdims=True,
        )

    @staticmethod
    def get_confidence_level(confidence):
        if confidence >= 0.80:
            return "high"
        if confidence >= 0.60:
            return "moderate"
        return "low"

    def _preprocess(self, image):
        image = image.convert("RGB")
        image = image.resize((224, 224))

        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        array = (array - self.MEAN) / self.STD

        array = np.transpose(
            array,
            (2, 0, 1),
        )

        return np.expand_dims(
            array.astype(np.float32),
            axis=0,
        )

    def predict(self, image):
        input_tensor = self._preprocess(image)

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor,
            },
        )

        logits = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        if logits.shape != (1, len(self.CLASS_NAMES)):
            raise RuntimeError(
                "Unexpected Skin model output shape: "
                f"{logits.shape}"
            )

        probabilities = self._softmax(logits)[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_index]
        )

        condition = self.CLASS_NAMES[predicted_index]

        confidence_level = self.get_confidence_level(
            confidence
        )

        return {
            "condition": condition,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "uncertain": confidence_level == "low",
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(
                    self.CLASS_NAMES,
                    probabilities,
                )
            },
            "model": "EfficientNet-B0",
            "model_version": self.MODEL_VERSION,
            "engine": "ONNX Runtime",
            "screening_only": True,
        }


_model = None


def get_skin_model():
    global _model

    if _model is None:
        _model = SkinModel()

    return _model