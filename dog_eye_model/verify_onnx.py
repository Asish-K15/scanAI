from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort

from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader

from dataset import load_datasets


# ============================================================
# PATHS
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


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 60)
print("SCANAI DOG EYE DISEASE")
print("PYTORCH vs ONNX CONSISTENCY TEST")
print("=" * 60)

print()
print("Checkpoint:")
print(CHECKPOINT)

print()
print("ONNX:")
print(ONNX_FILE)


# ============================================================
# LOAD TEST DATA
# ============================================================

_, _, test_dataset = load_datasets()

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


print()
print("Test samples:", len(test_dataset))


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================

print()
print("Loading PyTorch model...")

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
# PYTORCH INFERENCE
# ============================================================

print()
print("Running PyTorch inference...")

pytorch_predictions = []
pytorch_probabilities = []
true_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(dim=1)

        pytorch_predictions.extend(
            predictions.cpu().numpy()
        )

        pytorch_probabilities.extend(
            probabilities.cpu().numpy()
        )

        true_labels.extend(
            labels.numpy()
        )


pytorch_predictions = np.array(
    pytorch_predictions
)

pytorch_probabilities = np.array(
    pytorch_probabilities
)

true_labels = np.array(
    true_labels
)


# ============================================================
# LOAD ONNX
# ============================================================

print()
print("Loading ONNX Runtime...")

session = ort.InferenceSession(
    str(ONNX_FILE),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Input :", input_name)
print("Output:", output_name)


# ============================================================
# ONNX INFERENCE
# ============================================================

print()
print("Running ONNX inference...")

onnx_predictions = []
onnx_probabilities = []


for images, labels in test_loader:

    # PyTorch tensor -> NumPy
    input_data = images.numpy()

    outputs = session.run(
        [output_name],
        {
            input_name: input_data
        }
    )[0]

    # Softmax
    exp_outputs = np.exp(
        outputs - np.max(
            outputs,
            axis=1,
            keepdims=True
        )
    )

    probabilities = (
        exp_outputs /
        np.sum(
            exp_outputs,
            axis=1,
            keepdims=True
        )
    )

    predictions = np.argmax(
        outputs,
        axis=1
    )

    onnx_predictions.extend(
        predictions
    )

    onnx_probabilities.extend(
        probabilities
    )


onnx_predictions = np.array(
    onnx_predictions
)

onnx_probabilities = np.array(
    onnx_probabilities
)


# ============================================================
# CONSISTENCY
# ============================================================

same_predictions = (
    pytorch_predictions ==
    onnx_predictions
)

prediction_match = same_predictions.mean()


probability_difference = np.max(
    np.abs(
        pytorch_probabilities -
        onnx_probabilities
    )
)


# ============================================================
# ACCURACY
# ============================================================

pytorch_accuracy = (
    pytorch_predictions ==
    true_labels
).mean()

onnx_accuracy = (
    onnx_predictions ==
    true_labels
).mean()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("ONNX CONSISTENCY RESULTS")
print("=" * 60)

print()
print("Test samples:", len(true_labels))

print()
print(
    f"PyTorch accuracy : {pytorch_accuracy:.4f}"
)

print(
    f"ONNX accuracy    : {onnx_accuracy:.4f}"
)

print()
print(
    f"Prediction matches: "
    f"{same_predictions.sum()} / {len(true_labels)}"
)

print(
    f"Prediction match rate: "
    f"{prediction_match:.4f}"
)

print()
print(
    f"Maximum probability difference: "
    f"{probability_difference:.8f}"
)


# ============================================================
# DIFFERENCES
# ============================================================

different_indices = np.where(
    ~same_predictions
)[0]


print()
print("=" * 60)
print("PREDICTION DIFFERENCES")
print("=" * 60)

print()
print("Different predictions:", len(different_indices))

if len(different_indices) > 0:

    for index in different_indices:

        print()
        print("Index:", index)
        print("True:", true_labels[index])
        print("PyTorch:", pytorch_predictions[index])
        print("ONNX:", onnx_predictions[index])

        print(
            "PyTorch probabilities:",
            pytorch_probabilities[index]
        )

        print(
            "ONNX probabilities:",
            onnx_probabilities[index]
        )


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print("=" * 60)

if (
    np.array_equal(
        pytorch_predictions,
        onnx_predictions
    )
):

    print("ONNX VERIFICATION: PASS")

else:

    print("ONNX VERIFICATION: REVIEW REQUIRED")

print("=" * 60)