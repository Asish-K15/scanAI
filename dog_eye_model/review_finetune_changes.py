from pathlib import Path
import pandas as pd
import shutil


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_DIR.parent / "dataset_manager"

BASELINE = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "efficientnet_b0_test_predictions.csv"
)

FINETUNED = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "efficientnet_b0_finetuned_test_predictions.csv"
)

OUT = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "finetune_change_review"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

b = pd.read_csv(BASELINE)

f = pd.read_csv(FINETUNED)

x = b[
    [
        "path",
        "true_class",
        "predicted_class",
        "correct",
    ]
].merge(
    f[
        [
            "path",
            "predicted_class",
            "correct",
        ]
    ],
    on="path",
    suffixes=("_baseline", "_finetuned")
)


# ============================================================
# GROUPS
# ============================================================

groups = {

    "fixed": x[
        (~x["correct_baseline"]) &
        (x["correct_finetuned"])
    ],

    "broken": x[
        (x["correct_baseline"]) &
        (~x["correct_finetuned"])
    ],

    "still_wrong": x[
        (~x["correct_baseline"]) &
        (~x["correct_finetuned"])
    ],
}


# ============================================================
# COPY IMAGES
# ============================================================

for name, df in groups.items():

    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(name.upper())
    print("=" * 60)
    print("Images:", len(df))

    copied = 0

    for _, row in df.iterrows():

        raw_path = Path(row["path"])

        # CSV paths are relative to ScanAI project root
        if raw_path.is_absolute():
             source = raw_path
        else:
            source = DATASET_ROOT/ raw_path
 
        destination = folder / source.name

        print()
        print("TRUE       :", row["true_class"])
        print("BASELINE   :", row["predicted_class_baseline"])
        print("FINE-TUNED :", row["predicted_class_finetuned"])
        print("SOURCE     :", source)

        if source.exists():

            shutil.copy2(
                source,
                destination
            )

            copied += 1
            print("COPIED     :", destination)

        else:

            print("!!! NOT FOUND !!!")

    print()
    print("Copied:", copied)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("REVIEW DATASET CREATED")
print("=" * 60)

print("Location:")
print(OUT)

for name in groups:

    folder = OUT / name

    count = len([
        p for p in folder.iterdir()
        if p.is_file()
    ])

    print(f"{name:12}: {count} images")