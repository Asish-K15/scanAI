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
# PHASE 4D - FINAL UNTOUCHED TEST EVALUATION
# ============================================================

print("=" * 70)
print("SCANAI ANIMAL SKIN")
print("PHASE 4D - FINAL UNTOUCHED TEST EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = BASE_DIR / "splits" / "test"

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
    / "phase4d_test"
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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("TEST DATASET:")
print(TEST_DIR)

print()
print("SELECTED CHECKPOINT:")
print(CHECKPOINT)


# ------------------------------------------------------------
# SAFETY CHECKS
# ------------------------------------------------------------

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
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
# TRANSFORM
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
# LOAD TEST DATASET
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING ORIGINAL TEST DATASET")
print("=" * 70)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform,
)

print()
print("Test images:", len(test_dataset))

print()
print("Detected classes:")

for idx, name in enumerate(test_dataset.classes):
    print(f"  {idx}: {name}")


# ------------------------------------------------------------
# VERIFY CLASS ORDER
# ------------------------------------------------------------

if test_dataset.classes != CLASS_NAMES:
    raise RuntimeError(
        "\nCLASS ORDER MISMATCH!\n\n"
        f"Expected:\n{CLASS_NAMES}\n\n"
        f"Found:\n{test_dataset.classes}\n"
    )


# ------------------------------------------------------------
# TEST CLASS COUNTS
# ------------------------------------------------------------

test_counts = {}

for class_name in CLASS_NAMES:
    test_counts[class_name] = 0

for _, label in test_dataset.samples:
    test_counts[CLASS_NAMES[label]] += 1

print()
print("Test class counts:")

for class_name in CLASS_NAMES:
    print(
        f"  {class_name:<32}"
        f"{test_counts[class_name]:>5}"
    )


test_loader = DataLoader(
    test_dataset,
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
print("LOADING SELECTED PHASE 4B MODEL")
print("=" * 70)

model = models.efficientnet_b0(
    weights=None
)


# Match Phase 4B architecture:
# early blocks frozen
# last two blocks trainable
# classifier trainable

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


for param in model.classifier.parameters():
    param.requires_grad = True


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


model.load_state_dict(state_dict)

model = model.to(device)

model.eval()


print()
print("Model loaded successfully.")

print()
print("Selected model:")
print("  EfficientNet-B0")
print("  Phase 4B")
print("  Best epoch: 9")
print("  Validation Macro F1: 0.8432")


# ------------------------------------------------------------
# RUN TEST INFERENCE
# ------------------------------------------------------------

print()
print("=" * 70)
print("RUNNING FINAL TEST INFERENCE")
print("=" * 70)

all_targets = []
all_predictions = []
all_confidences = []

with torch.no_grad():

    for batch_index, (
        images,
        targets
    ) in enumerate(test_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
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
            len(test_dataset),
        )

        if (
            processed % 500 < BATCH_SIZE
            or processed == len(test_dataset)
        ):

            print(
                f"Processed "
                f"{processed}/{len(test_dataset)}"
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


confusion_pairs.sort(
    key=lambda x: x["count"],
    reverse=True
)


# ------------------------------------------------------------
# PER-CLASS METRICS
# ------------------------------------------------------------

per_class_rows = []

for class_index, class_name in enumerate(
    CLASS_NAMES
):

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
# CONFIDENCE STATISTICS
# ------------------------------------------------------------

confidence_stats = {
    "mean": float(np.mean(confidences)),
    "median": float(np.median(confidences)),
    "min": float(np.min(confidences)),
    "max": float(np.max(confidences)),
    "below_50_percent": int(
        np.sum(confidences < 0.50)
    ),
    "50_to_70_percent": int(
        np.sum(
            (confidences >= 0.50)
            & (confidences < 0.70)
        )
    ),
    "70_to_80_percent": int(
        np.sum(
            (confidences >= 0.70)
            & (confidences < 0.80)
        )
    ),
    "80_to_90_percent": int(
        np.sum(
            (confidences >= 0.80)
            & (confidences < 0.90)
        )
    ),
    "90_percent_or_higher": int(
        np.sum(confidences >= 0.90)
    ),
}


# ------------------------------------------------------------
# SAVE CONFUSION MATRIX
# ------------------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

cm_df.index.name = "actual"

cm_path = (
    OUTPUT_DIR
    / "test_confusion_matrix.csv"
)

cm_df.to_csv(cm_path)


# ------------------------------------------------------------
# SAVE PER-CLASS METRICS
# ------------------------------------------------------------

per_class_path = (
    OUTPUT_DIR
    / "test_per_class_metrics.csv"
)

per_class_df.to_csv(
    per_class_path,
    index=False,
)


# ------------------------------------------------------------
# SAVE TEST PREDICTIONS
# ------------------------------------------------------------

prediction_rows = []

for index, (
    target,
    prediction,
    confidence
) in enumerate(
    zip(
        y_true,
        y_pred,
        confidences,
    )
):

    image_path = test_dataset.samples[index][0]

    prediction_rows.append({
        "path": image_path,
        "actual": CLASS_NAMES[target],
        "predicted": CLASS_NAMES[prediction],
        "confidence": float(confidence),
        "correct": bool(
            target == prediction
        ),
    })


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_path = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ------------------------------------------------------------
# SAVE CLASSIFICATION REPORT
# ------------------------------------------------------------

report_path = (
    OUTPUT_DIR
    / "test_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SCANAI ANIMAL SKIN\n"
        "PHASE 4D - FINAL TEST EVALUATION\n\n"
    )

    f.write(
        f"Test images: {len(test_dataset)}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Macro Precision: "
        f"{macro_precision:.6f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{macro_recall:.6f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{macro_f1:.6f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{weighted_f1:.6f}\n"
    )

    f.write("\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=====================\n\n")
    f.write(report_text)

    f.write("\n\n")
    f.write("TOP CONFUSION PAIRS\n")
    f.write("===================\n\n")

    for pair in confusion_pairs[:20]:

        f.write(
            f"{pair['actual']} -> "
            f"{pair['predicted']} : "
            f"{pair['count']}\n"
        )


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

summary = {

    "phase": "4D",

    "evaluation": "final_untouched_test",

    "model": "EfficientNet-B0",

    "model_version": "SCANAI-SKIN-PHASE4B",

    "checkpoint": str(
        CHECKPOINT
    ),

    "best_epoch": 9,

    "validation_macro_f1": 0.8432,

    "previous_phase2_macro_f1": 0.8135,

    "test_images": int(
        len(test_dataset)
    ),

    "test_class_counts": test_counts,

    "accuracy": float(
        accuracy
    ),

    "macro_precision": float(
        macro_precision
    ),

    "macro_recall": float(
        macro_recall
    ),

    "macro_f1": float(
        macro_f1
    ),

    "weighted_f1": float(
        weighted_f1
    ),

    "improvement_over_phase2_test_not_comparable": True,

    "confidence_statistics":
        confidence_stats,

    "most_confused_pairs":
        confusion_pairs[:20],

    "per_class":
        per_class_rows,
}


summary_path = (
    OUTPUT_DIR
    / "phase4d_final_test_summary.json"
)

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
# PRINT FINAL RESULTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 4D - FINAL TEST RESULTS")
print("=" * 70)

print()
print(
    f"Test images       : "
    f"{len(test_dataset)}"
)

print(
    f"Accuracy          : "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision   : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1       : "
    f"{weighted_f1:.4f}"
)


print()
print("=" * 70)
print("PER-CLASS TEST METRICS")
print("=" * 70)

print(
    per_class_df.to_string(
        index=False,
        formatters={
            "precision":
                "{:.4f}".format,
            "recall":
                "{:.4f}".format,
            "f1":
                "{:.4f}".format,
        },
    )
)


print()
print("=" * 70)
print("TOP TEST CONFUSION PAIRS")
print("=" * 70)

for pair in confusion_pairs[:15]:

    print(
        f"{pair['actual']:<30} -> "
        f"{pair['predicted']:<30} "
        f"{pair['count']}"
    )


print()
print("=" * 70)
print("TEST CONFIDENCE STATISTICS")
print("=" * 70)

print(
    f"Mean confidence       : "
    f"{confidence_stats['mean']:.4f}"
)

print(
    f"Median confidence     : "
    f"{confidence_stats['median']:.4f}"
)

print(
    f"Below 50%             : "
    f"{confidence_stats['below_50_percent']}"
)

print(
    f"50-70%                : "
    f"{confidence_stats['50_to_70_percent']}"
)

print(
    f"70-80%                : "
    f"{confidence_stats['70_to_80_percent']}"
)

print(
    f"80-90%                : "
    f"{confidence_stats['80_to_90_percent']}"
)

print(
    f"90%+                  : "
    f"{confidence_stats['90_percent_or_higher']}"
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
print("PHASE 4D COMPLETE")
print("=" * 70)

print()
print("SAFETY:")
print("  Original test images were READ ONLY.")
print("  splits/test was NOT modified.")
print("  baseline_clean was NOT modified.")
print("  reviewed_quality_clean was NOT modified.")
print("  No training was performed on test images.")

print()
print("Selected candidate:")
print("  SCANAI-SKIN-PHASE4B")
print("  EfficientNet-B0")
print("  Best epoch: 9")
print("  Validation Macro F1: 0.8432")