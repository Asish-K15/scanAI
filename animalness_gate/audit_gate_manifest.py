from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    BASE_DIR
    / "animalness_gate"
    / "manifests"
    / "gate_manifest.csv"
)

EXTERNAL_HUMAN_DIR = (
    BASE_DIR
    / "animal_validator"
    / "rejection_samples"
    / "human"
)

EXTERNAL_NON_ANIMAL_DIR = (
    BASE_DIR
    / "animal_validator"
    / "rejection_samples"
    / "non_animal"
)


# ------------------------------------------------------------
# Expected counts
# ------------------------------------------------------------

EXPECTED_COUNTS = {
    ("train", "animal"): 3405,
    ("train", "not_animal"): 1444,
    ("validation", "animal"): 727,
    ("validation", "not_animal"): 356,
    ("test", "animal"): 728,
}


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def count_images(directory: Path) -> int:

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():

    # ------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------

    print("=" * 70)
    print("ANIMALNESS GATE MANIFEST AUDIT")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )


    df = pd.read_csv(MANIFEST_PATH)

    print()
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Total rows: {len(df)}")


    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------

    required_columns = {
        "image_path",
        "gate_label",
        "split",
        "source",
        "source_class",
        "species",
        "group_id",
        "phash",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print()
    print("Required columns: PASS")


    # ------------------------------------------------------------
    # Expected split counts
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING SPLIT COUNTS")
    print("=" * 70)

    actual_counts = (
        df.groupby(
            ["split", "gate_label"]
        )
        .size()
        .to_dict()
    )


    for key, expected in EXPECTED_COUNTS.items():

        actual = actual_counts.get(
            key,
            0,
        )

        print(
            f"{key}: "
            f"expected={expected}, "
            f"actual={actual}"
        )

        if actual != expected:
            raise RuntimeError(
                f"Count mismatch for {key}: "
                f"expected {expected}, "
                f"got {actual}"
            )


    print("Split counts: PASS")


    # ------------------------------------------------------------
    # Validate allowed values
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING LABELS AND SPLITS")
    print("=" * 70)


    allowed_labels = {
        "animal",
        "not_animal",
    }

    allowed_splits = {
        "train",
        "validation",
        "test",
    }


    unexpected_labels = (
        set(df["gate_label"].dropna().unique())
        - allowed_labels
    )

    unexpected_splits = (
        set(df["split"].dropna().unique())
        - allowed_splits
    )


    if unexpected_labels:
        raise RuntimeError(
            f"Unexpected gate labels: "
            f"{sorted(unexpected_labels)}"
        )

    if unexpected_splits:
        raise RuntimeError(
            f"Unexpected splits: "
            f"{sorted(unexpected_splits)}"
        )


    print("Gate labels: PASS")
    print("Splits: PASS")


    # ------------------------------------------------------------
    # Test-set protection
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING TEST-SET PROTECTION")
    print("=" * 70)


    test_df = df[
        df["split"] == "test"
    ]

    if len(test_df) != 728:
        raise RuntimeError(
            f"Expected exactly 728 test records, "
            f"found {len(test_df)}"
        )


    if not all(
        test_df["gate_label"] == "animal"
    ):
        raise RuntimeError(
            "Test split contains non-animal records."
        )


    print(
        "Test records: 728"
    )

    print(
        "Test label: animal only"
    )

    print("Test-set protection: PASS")


    # ------------------------------------------------------------
    # Duplicate image-path check
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING DUPLICATE IMAGE PATHS")
    print("=" * 70)


    duplicate_paths = (
        df[
            df["image_path"].duplicated(
                keep=False
            )
        ]
    )

    if not duplicate_paths.empty:
        print(
            duplicate_paths[
                [
                    "image_path",
                    "split",
                    "gate_label",
                ]
            ].to_string(index=False)
        )

        raise RuntimeError(
            f"Found {len(duplicate_paths)} "
            "rows belonging to duplicate image paths."
        )


    print("Duplicate image paths: 0")
    print("Duplicate check: PASS")


    # ------------------------------------------------------------
    # Split-overlap check
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING SPLIT OVERLAP")
    print("=" * 70)


    train_paths = set(
        df.loc[
            df["split"] == "train",
            "image_path",
        ]
    )

    validation_paths = set(
        df.loc[
            df["split"] == "validation",
            "image_path",
        ]
    )

    test_paths = set(
        df.loc[
            df["split"] == "test",
            "image_path",
        ]
    )


    train_validation = (
        train_paths & validation_paths
    )

    train_test = (
        train_paths & test_paths
    )

    validation_test = (
        validation_paths & test_paths
    )


    print(
        f"Train <-> Validation: "
        f"{len(train_validation)}"
    )

    print(
        f"Train <-> Test: "
        f"{len(train_test)}"
    )

    print(
        f"Validation <-> Test: "
        f"{len(validation_test)}"
    )


    if train_validation:
        raise RuntimeError(
            "Train/validation overlap detected."
        )

    if train_test:
        raise RuntimeError(
            "Train/test overlap detected."
        )

    if validation_test:
        raise RuntimeError(
            "Validation/test overlap detected."
        )


    print("Split overlap: PASS")


    # ------------------------------------------------------------
    # Verify every manifest path exists
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING IMAGE PATHS")
    print("=" * 70)


    missing_paths = []

    for image_path in df["image_path"]:

        absolute_path = (
            BASE_DIR / image_path
        )

        if not absolute_path.exists():
            missing_paths.append(
                image_path
            )


    print(
        f"Missing image files: "
        f"{len(missing_paths)}"
    )


    if missing_paths:

        print()
        print("First missing paths:")

        for path in missing_paths[:20]:
            print(f"  {path}")

        raise RuntimeError(
            "Some manifest image paths do not exist."
        )


    print("Image path existence: PASS")


    # ------------------------------------------------------------
    # Verify external rejection sets
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING EXTERNAL REJECTION SETS")
    print("=" * 70)


    human_count = count_images(
        EXTERNAL_HUMAN_DIR
    )

    external_non_animal_count = count_images(
        EXTERNAL_NON_ANIMAL_DIR
    )


    print(
        f"Human rejection images: "
        f"{human_count}"
    )

    print(
        f"External non-animal rejection images: "
        f"{external_non_animal_count}"
    )


    if human_count != 30:
        raise RuntimeError(
            "Expected 30 human rejection images."
        )


    if external_non_animal_count != 30:
        raise RuntimeError(
            "Expected 30 external non-animal "
            "rejection images."
        )


    print(
        "External rejection set counts: PASS"
    )


    # ------------------------------------------------------------
    # Ensure external rejection images are NOT in manifest
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING EXTERNAL REJECTION ISOLATION")
    print("=" * 70)


    manifest_paths = set(
        df["image_path"]
    )


    def relative_paths(directory: Path):

        result = set()

        for path in directory.rglob("*"):

            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                result.add(
                    str(
                        path.relative_to(BASE_DIR)
                    )
                )

        return result


    human_paths = relative_paths(
        EXTERNAL_HUMAN_DIR
    )

    external_non_animal_paths = relative_paths(
        EXTERNAL_NON_ANIMAL_DIR
    )


    human_overlap = (
        manifest_paths & human_paths
    )

    external_non_animal_overlap = (
        manifest_paths
        & external_non_animal_paths
    )


    print(
        f"Human rejection <-> manifest: "
        f"{len(human_overlap)}"
    )

    print(
        f"External non-animal <-> manifest: "
        f"{len(external_non_animal_overlap)}"
    )


    if human_overlap:
        raise RuntimeError(
            "Human rejection images leaked "
            "into the gate manifest."
        )


    if external_non_animal_overlap:
        raise RuntimeError(
            "External non-animal rejection images "
            "leaked into the gate manifest."
        )


    print(
        "External rejection isolation: PASS"
    )


    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print()
    print("ALL MANIFEST CHECKS PASSED.")

    print()
    print("Final dataset:")
    print(
        "  Train      : 4593"
    )

    print(
        "  Validation : 1039"
    )

    print(
        "  Test       : 728"
    )

    print(
        "  External rejection: 60"
    )

    print()
    print(
        "The 728-image animal test set and "
        "60-image external rejection set remain "
        "outside the training/validation data."
    )


if __name__ == "__main__":
    main()
