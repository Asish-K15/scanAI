from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent

PREDICTIONS = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "efficientnet_b0_test_predictions.csv"
)


df = pd.read_csv(PREDICTIONS)


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors = df[df["correct"] == False].copy()

errors["confidence"] = errors[
    ["prob_conjunctivitis", "prob_entropion"]
].max(axis=1)


errors = errors.sort_values(
    "confidence",
    ascending=False
)


print("=" * 60)
print("SCANAI DOG EYE DISEASE")
print("BASELINE ERROR ANALYSIS")
print("=" * 60)

print()
print("Total test images:", len(df))
print("Correct:", int(df["correct"].sum()))
print("Incorrect:", len(errors))

print()
print("=" * 60)
print("ERROR DIRECTION")
print("=" * 60)

print(
    pd.crosstab(
        errors["true_class"],
        errors["predicted_class"]
    )
)

print()
print("=" * 60)
print("MISCLASSIFIED IMAGES")
print("=" * 60)

columns = [
    "path",
    "true_class",
    "predicted_class",
    "prob_conjunctivitis",
    "prob_entropion",
    "confidence",
]

print(
    errors[columns].to_string(
        index=False
    )
)


# ============================================================
# SAVE ERRORS
# ============================================================

output = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "efficientnet_b0_errors.csv"
)

errors.to_csv(
    output,
    index=False
)

print()
print("=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)

print()
print("Saved:")
print(output)