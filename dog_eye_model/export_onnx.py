from pathlib import Path

import torch
import torch.nn as nn
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

OUTPUT_DIR = PROJECT_DIR / "outputs" / "onnx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_FILE = OUTPUT_DIR / "scanai_dog_eye_efficientnet_b0.onnx"

DEVICE = torch.device("cpu")


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("SCANAI DOG EYE DISEASE")
print("PYTORCH -> ONNX EXPORT")
print("=" * 60)

print()
print("Checkpoint:")
print(CHECKPOINT)

print()
print("Output:")
print(ONNX_FILE)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading EfficientNet-B0...")

model = efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    2
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# EXPORT
# ============================================================

print()
print("Exporting to ONNX...")

dummy_input = torch.randn(
    1,
    3,
    224,
    224,
    device=DEVICE
)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILE,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {
            0: "batch"
        },
        "logits": {
            0: "batch"
        }
    },
    opset_version=18,
    dynamo=False,
)

print()
print("=" * 60)
print("ONNX EXPORT COMPLETE")
print("=" * 60)

print()
print("ONNX file:")
print(ONNX_FILE)

print()
print("Checkpoint epoch:", checkpoint["epoch"])
print("Validation accuracy:", checkpoint["valid_accuracy"])