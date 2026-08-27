from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


# ============================================================
# SCANAI ANIMAL SKIN
# PHASE 4F - FINAL MODEL MANIFEST
# ============================================================

print("=" * 70)
print("SCANAI ANIMAL SKIN")
print("PHASE 4F - FINAL MODEL MANIFEST")
print("=" * 70)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "training"
    / "outputs"
    / "phase4_final"
    / "scanai_skin_phase4b.onnx"
)

METADATA_PATH = (
    BASE_DIR
    / "training"
    / "outputs"
    / "phase4_final"
    / "skin_model_metadata.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "training"
    / "outputs"
    / "phase4_final"
    / "SCANAI-SKIN-PHASE4B-MANIFEST.json"
)


# ------------------------------------------------------------
# CLASS ORDER
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# FILE CHECKS
# ------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Final ONNX model not found:\n{MODEL_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Model metadata not found:\n{METADATA_PATH}"
    )


# ------------------------------------------------------------
# SHA-256
# ------------------------------------------------------------

print()
print("=" * 70)
print("CALCULATING SHA-256")
print("=" * 70)

sha256 = hashlib.sha256()

with open(MODEL_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha256.update(chunk)

model_sha256 = sha256.hexdigest()

print()
print("SHA-256:")
print(model_sha256)


# ------------------------------------------------------------
# MODEL SIZE
# ------------------------------------------------------------

model_size_bytes = MODEL_PATH.stat().st_size
model_size_mb = model_size_bytes / (1024 * 1024)


# ------------------------------------------------------------
# FINAL MANIFEST
# ------------------------------------------------------------

manifest = {
    "project": "SCANAI ANIMAL SKIN",

    "model": {
        "name": "EfficientNet-B0",
        "version": "SCANAI-SKIN-PHASE4B",
        "framework": "PyTorch -> ONNX",
        "runtime": "ONNX Runtime",
        "artifact": "scanai_skin_phase4b.onnx",
        "artifact_size_bytes": model_size_bytes,
        "artifact_size_mb": round(model_size_mb, 4),
        "sha256": model_sha256,
    },

    "training": {
        "phase": "4B",
        "best_epoch": 9,
        "backbone": "EfficientNet-B0",
        "pretrained": "ImageNet",
        "early_blocks": "frozen",
        "last_two_blocks": "trainable",
        "classifier": "trainable",
        "loss": "class-weighted CrossEntropy",
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "epochs": 10,
        "batch_size": 32,
        "image_size": 224,
        "seed": 42,
    },

    "input_contract": {
        "input_name": "image",
        "type": "float32",
        "shape": [
            "batch",
            3,
            224,
            224
        ],
        "color": "RGB",
        "normalization": {
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
        "resize": [
            224,
            224
        ],
    },

    "output_contract": {
        "output_name": "logits",
        "type": "float32",
        "shape": [
            "batch",
            10
        ],
        "prediction": "argmax(logits)",
        "confidence": "softmax(logits)[predicted_class]",
        "class_order": CLASS_NAMES,
    },

    "classes": CLASS_NAMES,

    "validation": {
        "images": 1088,
        "accuracy": 0.8732,
        "macro_precision": 0.8432,
        "macro_recall": 0.8483,
        "macro_f1": 0.8432,
        "weighted_f1": 0.8742,
    },

    "final_test": {
        "images": 1107,
        "accuracy": 0.9097,
        "macro_precision": 0.8882,
        "macro_recall": 0.9107,
        "macro_f1": 0.8971,
        "weighted_f1": 0.9111,
    },

    "onnx_verification": {
        "status": "PASS",
        "onnx_checker": "PASS",
        "onnx_runtime_load": "PASS",
        "output_shape_check": "PASS",
        "prediction_agreement": "PASS",
        "max_absolute_difference": 0.00000262,
        "mean_absolute_difference": 0.00000089,
        "opset": 18,
    },

    "data_integrity": {
        "test_set_used_for_training": False,
        "test_set_used_for_model_selection": False,
        "test_images_modified": False,
        "test_images_deleted": False,
        "baseline_modified": False,
        "reviewed_quality_clean_modified": False,
    },

    "deployment_policy": {
        "screening_only": True,
        "clinical_diagnosis": False,
        "severity_inference": False,
        "urgency_inference": False,
        "treatment_recommendation": False,
        "clinical_mapping_from_confidence": False,
    },

    "handoff": {
        "status": "READY_FOR_PARTNER_A_INTEGRATION",
        "integration_target": "app/services/skin.py",
        "api_target": "/api/predict",
        "model_source_of_truth": "scanai_skin_phase4b.onnx",
        "class_order_source_of_truth": CLASS_NAMES,
    },

    "generated_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}


# ------------------------------------------------------------
# WRITE MANIFEST
# ------------------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL MODEL MANIFEST CREATED")
print("=" * 70)

print()
print("Model:")
print("  SCANAI-SKIN-PHASE4B")

print()
print("Artifact:")
print(MODEL_PATH)

print()
print("SHA-256:")
print(model_sha256)

print()
print("Validation Macro F1:")
print("  0.8432")

print()
print("Final Test Macro F1:")
print("  0.8971")

print()
print("Final Test Accuracy:")
print("  0.9097")

print()
print("ONNX verification:")
print("  PASS")

print()
print("Manifest:")
print(OUTPUT_PATH)

print()
print("=" * 70)
print("PHASE 4F COMPLETE")
print("=" * 70)

print()
print("FINAL STATUS:")
print("READY FOR PARTNER A INTEGRATION")