"""
Stage: Stratified Train / Validation / Test Split
---------------------------------------------------
Splits baseline_clean/ into:

    splits/train/
    splits/val/
    splits/test/

Default ratio:
    70% train
    15% validation
    15% test

The baseline_clean dataset has already been cleaned for:
    - safe single-class duplicates
    - cross-class duplicate contamination
    - excluded/low-volume classes

Therefore this script does NOT use the old duplicates.csv.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).parent

# IMPORTANT: use the verified baseline dataset
CLEAN_DIR = BASE_DIR / "baseline_clean"

SPLITS_DIR = BASE_DIR / "splits"
METADATA_DIR = BASE_DIR / "metadata"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42


# ============================================================
# SPLIT ONE CLASS
# ============================================================

def split_class(image_paths, ratios, rng):
    """
    Split one class into train / val / test.

    Uses a fixed random seed so the split is reproducible.
    """

    images = list(image_paths)

    # Reproducible shuffle
    rng.shuffle(images)

    n_total = len(images)

    train_count = int(n_total * ratios["train"])
    val_count = int(n_total * ratios["val"])

    train = images[:train_count]

    val = images[
        train_count:
        train_count + val_count
    ]

    test = images[
        train_count + val_count:
    ]

    return {
        "train": train,
        "val": val,
        "test": test
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test split from baseline_clean/"
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.70,
        help="Training ratio (default: 0.70)"
    )

    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation ratio (default: 0.15)"
    )

    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test ratio (default: 0.15)"
    )

    args = parser.parse_args()

    ratios = {
        "train": args.train,
        "val": args.val,
        "test": args.test
    }

    # --------------------------------------------------------
    # Validate ratios
    # --------------------------------------------------------

    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(
            "train + val + test ratios must equal 1.0"
        )

    # --------------------------------------------------------
    # Check baseline_clean
    # --------------------------------------------------------

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Baseline dataset not found: {CLEAN_DIR}"
        )

    print(f"Input dataset: {CLEAN_DIR}")
    print(f"Split ratios: {ratios}")
    print(f"Random seed: {SEED}")
    print()

    # --------------------------------------------------------
    # Remove old splits if they exist
    # --------------------------------------------------------

    if SPLITS_DIR.exists():
        print("Removing existing splits/ directory...")
        shutil.rmtree(SPLITS_DIR)

    # Create split directories
    for split in ("train", "val", "test"):
        (SPLITS_DIR / split).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Reproducible random generator
    # --------------------------------------------------------

    rng = random.Random(SEED)

    manifest_rows = []
    class_summary = []

    total_images = 0

    # --------------------------------------------------------
    # Process each top-level group
    # --------------------------------------------------------

    for group_dir in sorted(CLEAN_DIR.iterdir()):

        if not group_dir.is_dir():
            continue

        # Ignore unmapped data
        if group_dir.name == "_UNMAPPED":
            continue

        # ----------------------------------------------------
        # Process each class
        # ----------------------------------------------------

        for class_dir in sorted(group_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            images = [
                p for p in class_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTS
            ]

            if not images:
                continue

            unified_class = (
                f"{group_dir.name}__{class_dir.name}"
            )

            # ------------------------------------------------
            # Split
            # ------------------------------------------------

            split_result = split_class(
                images,
                ratios,
                rng
            )

            # ------------------------------------------------
            # Copy images
            # ------------------------------------------------

            for split_name, split_images in split_result.items():

                destination_dir = (
                    SPLITS_DIR
                    / split_name
                    / unified_class
                )

                destination_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                for img_path in split_images:

                    destination = (
                        destination_dir
                        / img_path.name
                    )

                    shutil.copy2(
                        img_path,
                        destination
                    )

                    manifest_rows.append([
                        str(img_path),
                        unified_class,
                        split_name
                    ])

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            train_count = len(
                split_result["train"]
            )

            val_count = len(
                split_result["val"]
            )

            test_count = len(
                split_result["test"]
            )

            class_total = (
                train_count
                + val_count
                + test_count
            )

            total_images += class_total

            class_summary.append([
                unified_class,
                class_total,
                train_count,
                val_count,
                test_count
            ])

    # ========================================================
    # WRITE MANIFEST
    # ========================================================

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = (
        METADATA_DIR
        / "split_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "original_path",
            "class",
            "split"
        ])

        writer.writerows(
            manifest_rows
        )

    # ========================================================
    # WRITE SUMMARY
    # ========================================================

    summary_path = (
        METADATA_DIR
        / "split_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class",
            "total",
            "train",
            "val",
            "test"
        ])

        writer.writerows(
            class_summary
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    train_total = sum(
        row[2] for row in class_summary
    )

    val_total = sum(
        row[3] for row in class_summary
    )

    test_total = sum(
        row[4] for row in class_summary
    )

    print()
    print("=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)

    print(f"Total images : {total_images}")
    print(f"Train        : {train_total}")
    print(f"Validation   : {val_total}")
    print(f"Test         : {test_total}")

    print()
    print(f"Train ratio  : {train_total / total_images:.3f}")
    print(f"Val ratio    : {val_total / total_images:.3f}")
    print(f"Test ratio   : {test_total / total_images:.3f}")

    print()
    print(f"Manifest : {manifest_path}")
    print(f"Summary  : {summary_path}")
    print(f"Splits   : {SPLITS_DIR}")

    print()
    print("Baseline split is ready.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()