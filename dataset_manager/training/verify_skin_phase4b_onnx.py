from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from torchvision import models
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    BASE_DIR
    / "training"
    / "outputs"
    / "phase4b"
    / "checkpoints"
    / "efficientnet_b0_phase4b_best.pth"
)

ONNX_PATH = (
    BASE_DIR
    / "training"
    / "outputs"
    / "phase4_final"
    / "scanai_skin_phase4b.onnx"
)

NUM_CLASSES = 10


print("=" * 70)
print("SCANAI ANIMAL SKIN")
print("PHASE 4E - ONNX ARTIFACT VERIFICATION")
print("=" * 70)


# ------------------------------------------------------------
# FILE CHECK
# ------------------------------------------------------------

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

if not ONNX_PATH.exists():
    raise FileNotFoundError(ONNX_PATH)


print()
print("PyTorch checkpoint:")
print(CHECKPOINT)

print()
print("ONNX artifact:")
print(ONNX_PATH)

print()
print("ONNX size:")
print(f"{ONNX_PATH.stat().st_size / (1024 * 1024):.2f} MB")


# ------------------------------------------------------------
# ONNX STRUCTURE CHECK
# ------------------------------------------------------------

print()
print("=" * 70)
print("CHECKING ONNX STRUCTURE")
print("=" * 70)

onnx_model = onnx.load(str(ONNX_PATH))

onnx.checker.check_model(onnx_model)

print("ONNX checker: PASS")

print()
print("IR version:")
print(onnx_model.ir_version)

print()
print("Opsets:")

for opset in onnx_model.opset_import:
    print(
        f"  domain={opset.domain or 'ai.onnx'} "
        f"version={opset.version}"
    )


# ------------------------------------------------------------
# ONNX RUNTIME
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING ONNX RUNTIME")
print("=" * 70)

providers = ort.get_available_providers()

print("Available providers:")
for provider in providers:
    print(f"  {provider}")

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

print()
print("ONNX Runtime load: PASS")


input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print()
print("Input:")
print("  name :", input_info.name)
print("  shape:", input_info.shape)
print("  type :", input_info.type)

print()
print("Output:")
print("  name :", output_info.name)
print("  shape:", output_info.shape)
print("  type :", output_info.type)


# ------------------------------------------------------------
# BUILD PYTORCH MODEL
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING PYTORCH MODEL")
print("=" * 70)

model = models.efficientnet_b0(weights=None)

for param in model.features.parameters():
    param.requires_grad = False

for block_index in [7, 8]:
    for param in model.features[block_index].parameters():
        param.requires_grad = True

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("PyTorch model load: PASS")


# ------------------------------------------------------------
# SAME INPUT COMPARISON
# ------------------------------------------------------------

print()
print("=" * 70)
print("PYTORCH vs ONNX NUMERICAL COMPARISON")
print("=" * 70)

torch.manual_seed(12345)

test_input = torch.randn(
    1,
    3,
    224,
    224,
    dtype=torch.float32,
)

with torch.no_grad():
    pytorch_output = model(test_input).numpy()

onnx_output = session.run(
    [output_info.name],
    {
        input_info.name: test_input.numpy()
    },
)[0]


print()
print("PyTorch output shape:", pytorch_output.shape)
print("ONNX output shape   :", onnx_output.shape)


if pytorch_output.shape != onnx_output.shape:
    raise RuntimeError(
        "OUTPUT SHAPE MISMATCH"
    )


absolute_difference = np.abs(
    pytorch_output - onnx_output
)

max_difference = float(
    absolute_difference.max()
)

mean_difference = float(
    absolute_difference.mean()
)


print()
print("Maximum absolute difference:")
print(f"  {max_difference:.8f}")

print()
print("Mean absolute difference:")
print(f"  {mean_difference:.8f}")


# ------------------------------------------------------------
# PREDICTION COMPARISON
# ------------------------------------------------------------

pytorch_prediction = int(
    np.argmax(pytorch_output[0])
)

onnx_prediction = int(
    np.argmax(onnx_output[0])
)

print()
print("PyTorch predicted class:", pytorch_prediction)
print("ONNX predicted class   :", onnx_prediction)


if pytorch_prediction != onnx_prediction:
    raise RuntimeError(
        "PREDICTION MISMATCH"
    )


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 4E VERIFICATION RESULT")
print("=" * 70)

print()
print("ONNX structure        : PASS")
print("ONNX Runtime loading  : PASS")
print("Output shape          : PASS")
print("Prediction agreement  : PASS")

print()
print(
    f"Max numerical diff    : {max_difference:.8f}"
)

print(
    f"Mean numerical diff   : {mean_difference:.8f}"
)

print()
print("FINAL RESULT: ONNX ARTIFACT VERIFIED")
print("=" * 70)