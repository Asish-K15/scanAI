import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    BASE_DIR
    / "animalness_gate"
    / "manifests"
    / "gate_manifest.csv"
)

CHECKPOINT_PATH = (
    BASE_DIR
    / "animalness_gate"
    / "outputs"
    / "best_animalness_gate.pth"
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

OUTPUT_DIR = (
    BASE_DIR
    / "animalness_gate"
    / "outputs"
    / "independent_evaluation"
)

# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "animal",
    "not_animal",
]

CLASS_TO_INDEX = {
    "animal": 0,
    "not_animal": 1,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# Dataset
# ============================================================

class ImageDataset(Dataset):

    def __init__(
        self,
        records,
    ):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = Path(
            record["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(
            image
        )

        label = record.get(
            "label"
        )

        return (
            image,
            label,
            str(image_path),
        )




def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Load model
    # ============================================================

    print("=" * 70)
    print("ANIMALNESS GATE INDEPENDENT EVALUATION")
    print("=" * 70)


    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print()
    print(
        f"Device: {device}"
    )


    if torch.cuda.is_available():

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
    )

    model = models.efficientnet_b0(
        weights=None
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        2,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)

    model.eval()


    print()
    print(
        "Model loaded:"
    )

    print(
        f"  Model: "
        f"{checkpoint.get('model', 'EfficientNet-B0')}"
    )

    print(
        f"  Version: "
        f"{checkpoint.get('model_version', 'unknown')}"
    )

    print(
        f"  Best epoch: "
        f"{checkpoint.get('best_epoch', 'unknown')}"
    )


    # ============================================================
    # Prediction helper
    # ============================================================

    def predict_records(records):

        dataset = ImageDataset(
            records
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        results = []

        with torch.no_grad():

            for images, labels, paths in loader:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                outputs = model(
                    images
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1,
                )

                predictions = torch.argmax(
                    probabilities,
                    dim=1,
                )

                for i in range(
                    len(paths)
                ):

                    animal_probability = (
                        float(
                            probabilities[
                                i, 0
                            ].item()
                        )
                    )

                    not_animal_probability = (
                        float(
                            probabilities[
                                i, 1
                            ].item()
                        )
                    )

                    predicted_index = int(
                        predictions[
                            i
                        ].item()
                    )

                    predicted_class = (
                        CLASS_NAMES[
                            predicted_index
                        ]
                    )

                    record = {
                        "image_path": paths[i],
                        "predicted_class":
                            predicted_class,
                        "animal_probability":
                            animal_probability,
                        "not_animal_probability":
                            not_animal_probability,
                        "confidence":
                            max(
                                animal_probability,
                                not_animal_probability,
                            ),
                    }

                    if labels[i] is not None:

                        actual_index = int(
                            labels[i]
                        )

                        actual_class = (
                            CLASS_NAMES[
                                actual_index
                            ]
                        )

                        record[
                            "actual_class"
                        ] = actual_class

                        record[
                            "correct"
                        ] = (
                            predicted_class
                            == actual_class
                        )

                    results.append(
                        record
                    )

        return results


    # ============================================================
    # 1. Untouched animal test set
    # ============================================================

    print()
    print("=" * 70)
    print("1. UNTOUCHED ANIMAL TEST SET")
    print("=" * 70)


    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()


    if len(test_df) != 728:
        raise RuntimeError(
            f"Expected 728 animal test images, "
            f"found {len(test_df)}."
        )


    animal_records = []

    for image_path in test_df[
        "image_path"
    ]:

        animal_records.append(
            {
                "image_path":
                    str(
                        BASE_DIR
                        / image_path
                    ),
                "label": 0,
            }
        )


    animal_results = predict_records(
        animal_records
    )


    animal_predictions_df = pd.DataFrame(
        animal_results
    )


    animal_predictions_path = (
        OUTPUT_DIR
        / "animal_test_predictions.csv"
    )

    animal_predictions_df.to_csv(
        animal_predictions_path,
        index=False,
    )


    animal_accuracy = accuracy_score(
        animal_predictions_df[
            "actual_class"
        ],
        animal_predictions_df[
            "predicted_class"
        ],
    )


    animal_macro_precision = (
        precision_score(
            animal_predictions_df[
                "actual_class"
            ],
            animal_predictions_df[
                "predicted_class"
            ],
            average="macro",
            zero_division=0,
        )
    )


    animal_macro_recall = (
        recall_score(
            animal_predictions_df[
                "actual_class"
            ],
            animal_predictions_df[
                "predicted_class"
            ],
            average="macro",
            zero_division=0,
        )
    )


    animal_macro_f1 = f1_score(
        animal_predictions_df[
            "actual_class"
        ],
        animal_predictions_df[
            "predicted_class"
        ],
        average="macro",
        zero_division=0,
    )


    print(
        f"Animal test images: "
        f"{len(animal_results)}"
    )

    print(
        f"Animal accuracy: "
        f"{animal_accuracy:.4f}"
    )

    print(
        f"Animal Macro F1: "
        f"{animal_macro_f1:.4f}"
    )


    # ============================================================
    # 2. External human rejection set
    # ============================================================

    print()
    print("=" * 70)
    print("2. EXTERNAL HUMAN REJECTION SET")
    print("=" * 70)


    human_paths = sorted(
        [
            path
            for path in EXTERNAL_HUMAN_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


    if len(human_paths) != 30:
        raise RuntimeError(
            f"Expected 30 human images, "
            f"found {len(human_paths)}."
        )


    human_records = [
        {
            "image_path": str(path),
            "label": 1,
        }
        for path in human_paths
    ]


    human_results = predict_records(
        human_records
    )


    human_df = pd.DataFrame(
        human_results
    )


    human_rejection_rate = (
        (
            human_df[
                "predicted_class"
            ]
            == "not_animal"
        )
        .mean()
    )


    print(
        f"Human images: "
        f"{len(human_df)}"
    )

    print(
        f"Rejected as not_animal: "
        f"{int((human_df['predicted_class'] == 'not_animal').sum())}"
    )

    print(
        f"Human rejection rate: "
        f"{human_rejection_rate:.4f}"
    )


    # ============================================================
    # 3. External non-animal rejection set
    # ============================================================

    print()
    print("=" * 70)
    print("3. EXTERNAL NON-ANIMAL REJECTION SET")
    print("=" * 70)


    non_animal_paths = sorted(
        [
            path
            for path in EXTERNAL_NON_ANIMAL_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


    if len(non_animal_paths) != 30:
        raise RuntimeError(
            f"Expected 30 external non-animal "
            f"images, found {len(non_animal_paths)}."
        )


    non_animal_records = [
        {
            "image_path": str(path),
            "label": 1,
        }
        for path in non_animal_paths
    ]


    non_animal_results = predict_records(
        non_animal_records
    )


    non_animal_df = pd.DataFrame(
        non_animal_results
    )


    non_animal_rejection_rate = (
        (
            non_animal_df[
                "predicted_class"
            ]
            == "not_animal"
        )
        .mean()
    )


    print(
        f"Non-animal images: "
        f"{len(non_animal_df)}"
    )

    print(
        f"Rejected as not_animal: "
        f"{int((non_animal_df['predicted_class'] == 'not_animal').sum())}"
    )

    print(
        f"Non-animal rejection rate: "
        f"{non_animal_rejection_rate:.4f}"
    )


    # ============================================================
    # Save external predictions
    # ============================================================

    human_path = (
        OUTPUT_DIR
        / "human_predictions.csv"
    )

    human_df.to_csv(
        human_path,
        index=False,
    )


    non_animal_path = (
        OUTPUT_DIR
        / "external_non_animal_predictions.csv"
    )

    non_animal_df.to_csv(
        non_animal_path,
        index=False,
    )


    # ============================================================
    # Combined rejection summary
    # ============================================================

    combined_rejection_df = pd.concat(
        [
            human_df.assign(
                rejection_source="human"
            ),
            non_animal_df.assign(
                rejection_source="non_animal"
            ),
        ],
        ignore_index=True,
    )


    combined_path = (
        OUTPUT_DIR
        / "rejection_predictions.csv"
    )

    combined_rejection_df.to_csv(
        combined_path,
        index=False,
    )


    # ============================================================
    # Confidence summaries
    # ============================================================

    def confidence_summary(
        dataframe,
    ):

        return {
            "count": int(
                len(dataframe)
            ),
            "mean": float(
                dataframe[
                    "confidence"
                ].mean()
            ),
            "median": float(
                dataframe[
                    "confidence"
                ].median()
            ),
            "minimum": float(
                dataframe[
                    "confidence"
                ].min()
            ),
            "maximum": float(
                dataframe[
                    "confidence"
                ].max()
            ),
        }


    # ============================================================
    # Threshold analysis
    # ============================================================

    print()
    print("=" * 70)
    print("4. CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)


    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]


    threshold_records = []


    for threshold in thresholds:

        # Reject if NOT_ANIMAL probability is at least
        # the threshold.
        #
        # This is only an analysis rule.
        # It is NOT a production decision rule.

        human_rejected = (
            human_df[
                "not_animal_probability"
            ]
            >= threshold
        )

        non_animal_rejected = (
            non_animal_df[
                "not_animal_probability"
            ]
            >= threshold
        )

        animal_accepted = (
            animal_predictions_df[
                "animal_probability"
            ]
            >= threshold
        )

        threshold_records.append(
            {
                "threshold":
                    threshold,

                "human_total":
                    len(human_df),

                "human_rejected":
                    int(
                        human_rejected.sum()
                    ),

                "human_rejection_rate":
                    float(
                        human_rejected.mean()
                    ),

                "non_animal_total":
                    len(non_animal_df),

                "non_animal_rejected":
                    int(
                        non_animal_rejected.sum()
                    ),

                "non_animal_rejection_rate":
                    float(
                        non_animal_rejected.mean()
                    ),

                "animal_total":
                    len(
                        animal_predictions_df
                    ),

                "animal_accepted":
                    int(
                        animal_accepted.sum()
                    ),

                "animal_acceptance_rate":
                    float(
                        animal_accepted.mean()
                    ),
            }
        )


    threshold_df = pd.DataFrame(
        threshold_records
    )


    threshold_path = (
        OUTPUT_DIR
        / "threshold_analysis.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )


    print(
        threshold_df.to_string(
            index=False
        )
    )


    # ============================================================
    # Failure cases
    # ============================================================

    print()
    print("=" * 70)
    print("5. FAILURE CASE SUMMARY")
    print("=" * 70)


    human_false_accepts = human_df[
        human_df[
            "predicted_class"
        ] == "animal"
    ].sort_values(
        "animal_probability",
        ascending=False,
    )


    non_animal_false_accepts = (
        non_animal_df[
            non_animal_df[
                "predicted_class"
            ] == "animal"
        ]
        .sort_values(
            "animal_probability",
            ascending=False,
        )
    )


    animal_false_rejects = (
        animal_predictions_df[
            animal_predictions_df[
                "predicted_class"
            ] == "not_animal"
        ]
        .sort_values(
            "not_animal_probability",
            ascending=False,
        )
    )


    print()
    print(
        f"Human false accepts: "
        f"{len(human_false_accepts)}"
    )

    print(
        f"Non-animal false accepts: "
        f"{len(non_animal_false_accepts)}"
    )

    print(
        f"Animal false rejects: "
        f"{len(animal_false_rejects)}"
    )


    print()
    print("Top human false accepts:")

    if human_false_accepts.empty:

        print("  None")

    else:

        print(
            human_false_accepts[
                [
                    "image_path",
                    "animal_probability",
                    "not_animal_probability",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


    print()
    print("Top non-animal false accepts:")

    if non_animal_false_accepts.empty:

        print("  None")

    else:

        print(
            non_animal_false_accepts[
                [
                    "image_path",
                    "animal_probability",
                    "not_animal_probability",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


    print()
    print("Top animal false rejects:")

    if animal_false_rejects.empty:

        print("  None")

    else:

        print(
            animal_false_rejects[
                [
                    "image_path",
                    "animal_probability",
                    "not_animal_probability",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


    # ============================================================
    # Save final evaluation report
    # ============================================================

    report = {
        "model": checkpoint.get(
            "model",
            "EfficientNet-B0",
        ),

        "model_version": checkpoint.get(
            "model_version",
            "SCANAI-ANIMALNESS-GATE-V1",
        ),

        "best_epoch": checkpoint.get(
            "best_epoch"
        ),

        "validation_macro_f1": checkpoint.get(
            "validation_macro_f1"
        ),

        "animal_test": {
            "count":
                len(animal_predictions_df),
            "accuracy":
                float(animal_accuracy),
            "macro_precision":
                float(animal_macro_precision),
            "macro_recall":
                float(animal_macro_recall),
            "macro_f1":
                float(animal_macro_f1),
            "confidence":
                confidence_summary(
                    animal_predictions_df
                ),
        },

        "human_rejection": {
            "count":
                len(human_df),
            "rejection_count":
                int(
                    (
                        human_df[
                            "predicted_class"
                        ]
                        == "not_animal"
                    ).sum()
                ),
            "rejection_rate":
                float(
                    human_rejection_rate
                ),
            "confidence":
                confidence_summary(
                    human_df
                ),
        },

        "external_non_animal_rejection": {
            "count":
                len(non_animal_df),
            "rejection_count":
                int(
                    (
                        non_animal_df[
                            "predicted_class"
                        ]
                        == "not_animal"
                    ).sum()
                ),
            "rejection_rate":
                float(
                    non_animal_rejection_rate
                ),
            "confidence":
                confidence_summary(
                    non_animal_df
                ),
        },

        "failure_cases": {
            "human_false_accepts":
                len(
                    human_false_accepts
                ),
            "non_animal_false_accepts":
                len(
                    non_animal_false_accepts
                ),
            "animal_false_rejects":
                len(
                    animal_false_rejects
                ),
        },

        "artifacts": {
            "animal_test_predictions":
                str(
                    animal_predictions_path
                ),
            "human_predictions":
                str(
                    human_path
                ),
            "external_non_animal_predictions":
                str(
                    non_animal_path
                ),
            "rejection_predictions":
                str(
                    combined_path
                ),
            "threshold_analysis":
                str(
                    threshold_path
                ),
        },
    }


    report_path = (
        OUTPUT_DIR
        / "independent_evaluation_report.json"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )


    # ============================================================
    # Final summary
    # ============================================================

    print()
    print("=" * 70)
    print("INDEPENDENT EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Animal test accuracy: "
        f"{animal_accuracy:.4f}"
    )

    print(
        "Animal test Macro F1: "
        f"{animal_macro_f1:.4f}"
    )

    print(
        "Human rejection rate: "
        f"{human_rejection_rate:.4f}"
    )

    print(
        "Non-animal rejection rate: "
        f"{non_animal_rejection_rate:.4f}"
    )

    print()
    print(
        f"Report: {report_path}"
    )

    print()
    print(
        "IMPORTANT: No production threshold "
        "has been selected."
    )

if __name__ == "__main__":
    main()
