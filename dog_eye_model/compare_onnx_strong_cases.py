from pathlib import Path

import pandas as pd

from inference import DogEyeModel


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
SCANAI_ROOT = PROJECT_DIR.parent

PREDICTIONS = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "efficientnet_b0_finetuned_test_predictions.csv"
)


# ============================================================
# LOAD TEST RESULTS
# ============================================================

df = pd.read_csv(PREDICTIONS)

df["confidence"] = df[
    [
        "prob_conjunctivitis",
        "prob_entropion",
    ]
].max(axis=1)

strong_cases = (
    df
    .sort_values(
        "confidence",
        ascending=False
    )
    .head(5)
)


# ============================================================
# LOAD ONNX
# ============================================================

model = DogEyeModel()


# ============================================================
# COMPARE
# ============================================================

print()
print("=" * 70)
print("PYTORCH vs ONNX — STRONG TEST CASES")
print("=" * 70)

print()

matches = 0

for _, row in strong_cases.iterrows():

    image_path = (
        SCANAI_ROOT
        / "dataset_manager"
        / row["path"]
    )

    result = model.predict(
        image_path
    )

    pytorch_prediction = row[
        "predicted_class"
    ]

    onnx_prediction = result[
        "condition"
    ]

    pytorch_confidence = float(
        row["confidence"]
    )

    onnx_confidence = result[
        "confidence"
    ]

    same_prediction = (
        pytorch_prediction ==
        onnx_prediction
    )

    if same_prediction:
        matches += 1

    print(
        "TRUE             :",
        row["true_class"]
    )

    print(
        "PyTorch          :",
        pytorch_prediction
    )

    print(
        "PyTorch confidence:",
        f"{pytorch_confidence:.6f}"
    )

    print(
        "ONNX             :",
        onnx_prediction
    )

    print(
        "ONNX confidence :",
        f"{onnx_confidence:.6f}"
    )

    print(
        "MATCH            :",
        "YES" if same_prediction else "NO"
    )

    print(
        "IMAGE            :",
        row["path"]
    )

    print("-" * 70)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print()
print(
    "Cases tested :",
    len(strong_cases)
)

print(
    "Prediction matches:",
    f"{matches}/{len(strong_cases)}"
)

print(
    "Match rate:",
    f"{matches / len(strong_cases):.2%}"
)