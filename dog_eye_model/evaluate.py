from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from dataset import load_datasets


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_DIR = Path(__file__).resolve().parent

# Fine-tuned checkpoint
CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_finetuned_best.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
)

CM_DIR = (
    PROJECT_DIR
    / "outputs"
    / "confusion_matrix"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("SCANAI DOG EYE DISEASE")
print("EFFICIENTNET-B0 FINE-TUNED TEST EVALUATION")
print("=" * 60)

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print()
print("Checkpoint:")
print(CHECKPOINT)


# ============================================================
# LOAD DATA
# ============================================================

_, _, test_dataset = load_datasets()

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

print()
print("Test samples:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading fine-tuned model...")

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
# INFERENCE
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

print()
print("Running independent test inference...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob[:, 1]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("FINE-TUNED TEST RESULTS")
print("=" * 60)

print()
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = [
    "conjunctivitis",
    "entropion"
]

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()
print("                 Predicted")
print("              Conj.  Entropion")
print(
    f"Actual Conj.   {cm[0,0]:4d}     {cm[0,1]:4d}"
)
print(
    f"Actual Ent.    {cm[1,0]:4d}     {cm[1,1]:4d}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

df = test_dataset.df.copy()

df["true_label"] = y_true
df["predicted_label"] = y_pred

df["true_class"] = df["true_label"].map({
    0: "conjunctivitis",
    1: "entropion"
})

df["predicted_class"] = df["predicted_label"].map({
    0: "conjunctivitis",
    1: "entropion"
})

df["prob_conjunctivitis"] = y_prob[:, 0]
df["prob_entropion"] = y_prob[:, 1]

df["correct"] = (
    df["true_label"] ==
    df["predicted_label"]
)

prediction_file = (
    OUTPUT_DIR
    / "efficientnet_b0_finetuned_test_predictions.csv"
)

df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_file = (
    CM_DIR
    / "efficientnet_b0_finetuned_confusion_matrix.csv"
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(cm_file)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "model": "EfficientNet-B0 fine-tuned",
    "test_samples": len(y_true),
    "accuracy": accuracy,
    "precision_entropion": precision,
    "recall_entropion": recall,
    "f1_entropion": f1,
    "roc_auc": roc_auc,
    "best_validation_accuracy": checkpoint["valid_accuracy"],
    "best_epoch": checkpoint["epoch"],
}])

summary_file = (
    OUTPUT_DIR
    / "efficientnet_b0_finetuned_test_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("FINE-TUNED EVALUATION COMPLETE")
print("=" * 60)

print()
print("Predictions:")
print(prediction_file)

print()
print("Confusion matrix:")
print(cm_file)

print()
print("Summary:")
print(summary_file)