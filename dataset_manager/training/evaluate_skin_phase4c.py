import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# SCANAI ANIMAL SKIN
# PHASE 4C - DETAILED VALIDATION EVALUATION
# ============================================================

print("=" * 70)
print("SCANAI ANIMAL SKIN")
print("PHASE 4C - DETAILED VALIDATION EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_ROOT = BASE_DIR / "splits_phase4"
VALID_DIR = DATASET_ROOT / "valid"

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
    / "phase4c"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("Validation dataset:")
print(VALID_DIR)

print()
print("Checkpoint:")
print(CHECKPOINT)


# ------------------------------------------------------------
# SAFETY CHECKS
# ------------------------------------------------------------

if not VALID_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found:\n{VALID_DIR}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


# ------------------------------------------------------------
# CLASS ORDER
# ------------------------------------------------------------

CLASS_NAMES = [
    "allergic_dermatitis",
    "foot_and_mouth_disease",
    "fungal_infection",
    "healthy_skin",
    "hotspot",
    "lumpy_skin_disease",
    "mange",
    "pyoderma",
    "ringworm",
    "scabies",
]


# ------------------------------------------------------------
# TRANSFORMS
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING VALIDATION DATASET")
print("=" * 70)

dataset = datasets.ImageFolder(
    VALID_DIR,
    transform=transform,
)

print()
print("Validation images:", len(dataset))

print()
print("Detected classes:")

for idx, name in enumerate(dataset.classes):
    print(f"  {idx}: {name}")


# ------------------------------------------------------------
# VERIFY CLASS ORDER
# ------------------------------------------------------------

if dataset.classes != CLASS_NAMES:
    raise RuntimeError(
        "\nClass order mismatch!\n\n"
        f"Expected:\n{CLASS_NAMES}\n\n"
        f"Found:\n{dataset.classes}\n"
    )


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING PHASE 4B MODEL")
print("=" * 70)

print()
print("EfficientNet-B0")
print("Fine-tuning: last 2 blocks + classifier")
print("Checkpoint: BEST validation Macro F1")


model = models.efficientnet_b0(weights=None)

# Match Phase 4B architecture
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

for param in model.classifier.parameters():
    param.requires_grad = True


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

model = model.to(device)
model.eval()


print()
print("Model loaded successfully.")


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

print()
print("=" * 70)
print("RUNNING VALIDATION INFERENCE")
print("=" * 70)

all_targets = []
all_predictions = []
all_confidences = []

with torch.no_grad():

    for batch_index, (images, targets) in enumerate(loader):

        images = images.to(device, non_blocking=True)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = torch.max(
            probabilities,
            dim=1,
        )

        all_targets.extend(
            targets.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_confidences.extend(
            confidences.cpu().numpy().tolist()
        )

        processed = min(
            (batch_index + 1) * BATCH_SIZE,
            len(dataset),
        )

        if processed % 500 < BATCH_SIZE or processed == len(dataset):
            print(
                f"Processed {processed}/{len(dataset)}"
            )


y_true = np.array(all_targets)
y_pred = np.array(all_predictions)
confidences = np.array(all_confidences)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0,
)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
)


# ------------------------------------------------------------
# MOST CONFUSED PAIRS
# ------------------------------------------------------------

confusion_pairs = []

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        if i == j:
            continue

        count = int(cm[i, j])

        if count > 0:
            confusion_pairs.append({
                "actual": CLASS_NAMES[i],
                "predicted": CLASS_NAMES[j],
                "count": count,
            })


confusion_pairs = sorted(
    confusion_pairs,
    key=lambda x: x["count"],
    reverse=True,
)


# ------------------------------------------------------------
# PER-CLASS RESULTS
# ------------------------------------------------------------

per_class_rows = []

for class_index, class_name in enumerate(CLASS_NAMES):

    row = report_dict[class_name]

    per_class_rows.append({
        "class": class_name,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1-score"],
        "support": int(row["support"]),
    })


per_class_df = pd.DataFrame(
    per_class_rows
)


# ------------------------------------------------------------
# SAVE CONFUSION MATRIX CSV
# ------------------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

cm_df.index.name = "actual"

cm_path = OUTPUT_DIR / "confusion_matrix.csv"

cm_df.to_csv(cm_path)


# ------------------------------------------------------------
# SAVE PER-CLASS CSV
# ------------------------------------------------------------

per_class_path = OUTPUT_DIR / "per_class_metrics.csv"

per_class_df.to_csv(
    per_class_path,
    index=False,
)


# ------------------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------------------

prediction_rows = []

for index, (target, prediction, confidence) in enumerate(
    zip(y_true, y_pred, confidences)
):

    image_path = dataset.samples[index][0]

    prediction_rows.append({
        "path": image_path,
        "actual": CLASS_NAMES[target],
        "predicted": CLASS_NAMES[prediction],
        "confidence": float(confidence),
        "correct": bool(target == prediction),
    })


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_path = OUTPUT_DIR / "validation_predictions.csv"

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

report_path = OUTPUT_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SCANAI ANIMAL SKIN\n"
        "PHASE 4C - DETAILED VALIDATION EVALUATION\n"
        "\n"
    )

    f.write(
        f"Validation images: {len(dataset)}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Macro Precision: {macro_precision:.6f}\n"
    )

    f.write(
        f"Macro Recall: {macro_recall:.6f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.6f}\n"
    )

    f.write(
        f"Weighted F1: {weighted_f1:.6f}\n"
    )

    f.write("\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=====================\n\n")
    f.write(report_text)

    f.write("\n\n")
    f.write("MOST CONFUSED PAIRS\n")
    f.write("===================\n\n")

    for pair in confusion_pairs[:20]:

        f.write(
            f"{pair['actual']} -> "
            f"{pair['predicted']} : "
            f"{pair['count']}\n"
        )


# ------------------------------------------------------------
# SAVE SUMMARY JSON
# ------------------------------------------------------------

summary = {
    "phase": "4C",
    "model": "EfficientNet-B0",
    "checkpoint": str(CHECKPOINT),
    "validation_images": int(len(dataset)),
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "best_phase4b_macro_f1": 0.8432,
    "phase2_macro_f1": 0.8135,
    "phase4a_macro_f1": 0.7951,
    "improvement_over_phase2": float(
        macro_f1 - 0.8135
    ),
    "improvement_over_phase4a": float(
        macro_f1 - 0.7951
    ),
    "most_confused_pairs": confusion_pairs[:20],
    "per_class": per_class_rows,
}


summary_path = OUTPUT_DIR / "phase4c_evaluation_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 4C RESULTS")
print("=" * 70)

print()
print(f"Validation images : {len(dataset)}")
print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {macro_precision:.4f}")
print(f"Macro Recall      : {macro_recall:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")

print()
print("PER-CLASS METRICS")
print("=" * 70)

print(
    per_class_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
        },
    )
)

print()
print("=" * 70)
print("TOP CONFUSION PAIRS")
print("=" * 70)

for pair in confusion_pairs[:15]:

    print(
        f"{pair['actual']:<30} -> "
        f"{pair['predicted']:<30} "
        f"{pair['count']}"
    )


print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print()
print(cm_path)
print(per_class_path)
print(predictions_path)
print(report_path)
print(summary_path)

print()
print("=" * 70)
print("PHASE 4C COMPLETE")
print("=" * 70)

print()
print("IMPORTANT:")
print("  Original test set was NOT used.")
print("  splits/test was NOT modified.")
print("  baseline_clean was NOT modified.")
print("  reviewed_quality_clean was NOT modified.")

print()
print("Next decision:")
print("  Analyze per-class weaknesses and confusion pairs")
print("  before deciding on another training experiment.")