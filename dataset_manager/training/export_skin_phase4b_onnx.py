from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# SCANAI ANIMAL SKIN
# FINAL MODEL EXPORT
# PHASE 4B -> ONNX
# ============================================================

print("=" * 70)
print("SCANAI ANIMAL SKIN")
print("FINAL PHASE 4B -> ONNX EXPORT")
print("=" * 70)


BASE_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    BASE_DIR
    / "training"
    / "outputs"
    / "phase4b"
    / "checkpoints"
    / "efficientnet_b0_phase4b_best.pth"
)

OUTPUT_DIR = (
    BASE_DIR
    / "training"
    / "outputs"
    / "phase4_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ONNX_PATH = (
    OUTPUT_DIR
    / "scanai_skin_phase4b.onnx"
)


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

NUM_CLASSES = len(CLASS_NAMES)


if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


print()
print("Checkpoint:")
print(CHECKPOINT)

print()
print("Output:")
print(ONNX_PATH)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print()
print("=" * 70)
print("BUILDING MODEL")
print("=" * 70)

model = models.efficientnet_b0(
    weights=None
)


# Match Phase 4B architecture exactly.

for param in model.features.parameters():
    param.requires_grad = False


for block_index in [7, 8]:

    for param in model.features[
        block_index
    ].parameters():

        param.requires_grad = True


in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES,
)


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.eval()


print()
print("Checkpoint loaded successfully.")

print()
print("Model:")
print("  EfficientNet-B0")
print("  ImageNet pretrained")
print("  Last 2 feature blocks fine-tuned")
print("  Classifier trainable")
print("  Best epoch: 9")


# ------------------------------------------------------------
# DUMMY INPUT
# ------------------------------------------------------------

dummy_input = torch.randn(
    1,
    3,
    224,
    224,
)


# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

print()
print("=" * 70)
print("EXPORTING ONNX")
print("=" * 70)

torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {
            0: "batch"
        },
        "logits": {
            0: "batch"
        },
    },
)


print()
print("ONNX export complete.")

print()
print("Artifact:")
print(ONNX_PATH)

print()
print("Size:")
print(
    f"{ONNX_PATH.stat().st_size / (1024 * 1024):.2f} MB"
)


# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------

metadata_path = (
    OUTPUT_DIR
    / "skin_model_metadata.json"
)

metadata = {
    "model": "EfficientNet-B0",
    "model_version": "SCANAI-SKIN-PHASE4B",
    "best_epoch": 9,
    "input_name": "image",
    "output_name": "logits",
    "input_shape": [
        "batch",
        3,
        224,
        224
    ],
    "num_classes": NUM_CLASSES,
    "classes": CLASS_NAMES,
    "preprocessing": {
        "color": "RGB",
        "resize": [224, 224],
        "mean": [
            0.485,
            0.456,
            0.406
        ],
        "std": [
            0.229,
            0.224,
            0.225
        ]
    },
    "architecture": {
        "backbone": "EfficientNet-B0",
        "early_blocks": "frozen",
        "last_two_blocks": "trainable",
        "classifier": "trainable"
    },
    "validation": {
        "accuracy": 0.8732,
        "macro_f1": 0.8432
    },
    "test": {
        "images": 1107,
        "accuracy": 0.9097,
        "macro_f1": 0.8971,
        "macro_precision": 0.8882,
        "macro_recall": 0.9107,
        "weighted_f1": 0.9111
    },
    "screening_only": True
}


import json

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print()
print("Metadata:")
print(metadata_path)


print()
print("=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)