from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_finetuned_best.pth"
)

ONNX_FILE = (
    PROJECT_DIR
    / "outputs"
    / "onnx"
    / "scanai_dog_eye_efficientnet_b0.onnx"
)

IMAGE = (
    PROJECT_DIR.parent
    / "dataset_manager"
    / "expansion"
    / "dog_conjunctivitis"
    / "test_D0_0e49d71d-60a5-11ec-8402-0a7404972c70_jpg.rf.WV8mVlnd5VD3jq4BEaiG.jpg"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD IMAGE
# ============================================================

print("=" * 60)
print("SCANAI DOG EYE DISEASE")
print("PYTORCH CPU vs ONNX CPU")
print("=" * 60)

print()
print("Image:")
print(IMAGE)

image = Image.open(IMAGE).convert("RGB")

tensor = transform(image).unsqueeze(0)


# ============================================================
# PYTORCH CPU
# ============================================================

print()
print("Loading PyTorch model on CPU...")

model = efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    2
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

with torch.no_grad():

    pytorch_logits = model(
        tensor
    ).numpy()


# ============================================================
# PYTORCH SOFTMAX
# ============================================================

def softmax(x):

    x = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    e = np.exp(x)

    return e / np.sum(
        e,
        axis=1,
        keepdims=True
    )


pytorch_prob = softmax(
    pytorch_logits
)[0]

pytorch_prediction = int(
    np.argmax(pytorch_prob)
)


# ============================================================
# ONNX CPU
# ============================================================

print()
print("Loading ONNX Runtime on CPU...")

session = ort.InferenceSession(
    str(ONNX_FILE),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

onnx_logits = session.run(
    [output_name],
    {
        input_name:
        tensor.numpy()
    }
)[0]

onnx_prob = softmax(
    onnx_logits
)[0]

onnx_prediction = int(
    np.argmax(onnx_prob)
)


# ============================================================
# RESULTS
# ============================================================

class_names = [
    "conjunctivitis",
    "entropion"
]

print()
print("=" * 60)
print("CPU COMPARISON")
print("=" * 60)

print()

print("PyTorch logits:")
print(pytorch_logits[0])

print()
print("ONNX logits:")
print(onnx_logits[0])

print()

print("PyTorch probabilities:")
print(pytorch_prob)

print()
print("ONNX probabilities:")
print(onnx_prob)

print()

print(
    "PyTorch prediction:",
    class_names[pytorch_prediction]
)

print(
    "ONNX prediction:",
    class_names[onnx_prediction]
)

print()

print(
    "Maximum logit difference:",
    np.max(
        np.abs(
            pytorch_logits -
            onnx_logits
        )
    )
)

print(
    "Maximum probability difference:",
    np.max(
        np.abs(
            pytorch_prob -
            onnx_prob
        )
    )
)


# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 60)

if pytorch_prediction == onnx_prediction:

    print("CPU CONSISTENCY: PASS")

else:

    print("CPU CONSISTENCY: MISMATCH")

print("=" * 60)
