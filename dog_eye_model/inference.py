from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "onnx"
    / "scanai_dog_eye_efficientnet_b0.onnx"
)

CLASS_NAMES = {
    0: "conjunctivitis",
    1: "entropion",
}

IMAGE_SIZE = (224, 224)

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

# ============================================================
# SCANAI MODEL METADATA
# ============================================================

MODEL_NAME = "EfficientNet-B0"
MODEL_VERSION = "dog-eye-v1"
INFERENCE_ENGINE = "ONNX Runtime"

# Engineering confidence policy.
# These are NOT clinically validated thresholds.
HIGH_CONFIDENCE_THRESHOLD = 0.80
MODERATE_CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# MODEL
# ============================================================

class DogEyeModel:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found:\n{self.model_path}"
            )

        print("Loading ONNX model...")
        print("Model:", self.model_path)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

        print("Input :", self.input_name)
        print("Output:", self.output_name)


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(self, image):

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        image = image.convert("RGB")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = (
            image - MEAN
        ) / STD

        # HWC → CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # CHW → NCHW
        image = np.expand_dims(
            image,
            axis=0
        )

        return image.astype(np.float32)


    # ========================================================
    # SOFTMAX
    # ========================================================

    @staticmethod
    def softmax(logits):

        logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )

        exp_logits = np.exp(logits)

        return (
            exp_logits /
            np.sum(
                exp_logits,
                axis=1,
                keepdims=True
            )
        )


    # ========================================================
    # CONFIDENCE POLICY
    # ========================================================

    @staticmethod
    def get_confidence_level(confidence):

        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return "high"

        if confidence >= MODERATE_CONFIDENCE_THRESHOLD:
            return "moderate"

        return "low"


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, image):

        input_tensor = self.preprocess(image)

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor
            }
        )

        logits = outputs[0]

        probabilities = self.softmax(logits)[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        confidence_level = (
            self.get_confidence_level(
                confidence
            )
        )

        # Low-confidence predictions should
        # be treated as uncertain by the platform.
        uncertain = (
            confidence < MODERATE_CONFIDENCE_THRESHOLD
        )

        return {
            "condition": predicted_class,

            "confidence": confidence,

            "confidence_level": confidence_level,

            "uncertain": uncertain,

            "probabilities": {
                "conjunctivitis": float(
                    probabilities[0]
                ),
                "entropion": float(
                    probabilities[1]
                ),
            },

            "model": MODEL_NAME,

            "model_version": MODEL_VERSION,

            "engine": INFERENCE_ENGINE,

            "screening_only": True,
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SCANAI DOG EYE DISEASE")
    print("ONNX INFERENCE TEST")
    print("=" * 60)

    model = DogEyeModel()

    print()
    print("Model loaded successfully.")

    print()
    print("Confidence policy:")
    print(
        f"  High     : >= {HIGH_CONFIDENCE_THRESHOLD:.2f}"
    )
    print(
        f"  Moderate : >= {MODERATE_CONFIDENCE_THRESHOLD:.2f}"
    )
    print(
        f"  Low      : <  {MODERATE_CONFIDENCE_THRESHOLD:.2f}"
    )

    print()
    print("IMPORTANT:")
    print("These thresholds are engineering")
    print("confidence thresholds, NOT clinically")
    print("validated diagnostic thresholds.")

    print()
    print("Now test with:")
    print()
    print("model.predict(<image_path>)")