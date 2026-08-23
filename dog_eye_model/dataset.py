from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

SCANAI_ROOT = Path(__file__).resolve().parents[1]

DATASET_MANAGER = SCANAI_ROOT / "dataset_manager"

MANIFEST = (
    DATASET_MANAGER
    / "metadata"
    / "dog_eye_disease_master_manifest.csv"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_TO_IDX = {
    "conjunctivitis": 0,
    "entropion": 1,
}


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# EXPERIMENT 3
# TARGETED TRAINING AUGMENTATION
# ============================================================

TRAIN_TRANSFORM = transforms.Compose([

    # Preserve most of the eye while allowing
    # small framing variations.
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.75, 1.0),
        ratio=(0.90, 1.10),
    ),

    # Eye images can reasonably be viewed from
    # either horizontal orientation.
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # Small camera/viewpoint variation.
    transforms.RandomRotation(
        degrees=8
    ),

    # Mild geometric variation.
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    ),

    # Mild illumination/camera variation.
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# VALIDATION / TEST
# ============================================================

# IMPORTANT:
# No random augmentation here.
#
# Validation and test must remain deterministic so that
# comparisons with the previous experiments are fair.

EVAL_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# DATASET
# ============================================================

class DogEyeDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(
            row["path"]
        )

        # Manifest paths are relative to
        # dataset_manager.
        if not image_path.is_absolute():

            image_path = (
                DATASET_MANAGER
                / image_path
            )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = CLASS_TO_IDX[
            row["class"]
        ]

        if self.transform:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    df = pd.read_csv(
        MANIFEST
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    valid_df = df[
        df["split"] == "valid"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    train_dataset = DogEyeDataset(
        train_df,
        transform=TRAIN_TRANSFORM,
    )

    valid_dataset = DogEyeDataset(
        valid_df,
        transform=EVAL_TRANSFORM,
    )

    test_dataset = DogEyeDataset(
        test_df,
        transform=EVAL_TRANSFORM,
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
    )


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DOG EYE DATASET LOADER TEST")
    print("EXPERIMENT 3 - TARGETED AUGMENTATION")
    print("=" * 60)

    print()
    print("Manifest:")
    print(MANIFEST)

    train_dataset, valid_dataset, test_dataset = (
        load_datasets()
    )

    print()
    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Valid samples: {len(valid_dataset)}"
    )

    print(
        f"Test samples:  {len(test_dataset)}"
    )

    image, label = train_dataset[0]

    print()
    print("Sample:")

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Label:",
        label
    )

    print(
        "Image dtype:",
        image.dtype
    )

    print(
        "Image min:",
        round(
            image.min().item(),
            4
        )
    )

    print(
        "Image max:",
        round(
            image.max().item(),
            4
        )
    )

    print()
    print("CLASS MAPPING:")
    print(CLASS_TO_IDX)

    print()
    print("DATA AUGMENTATION:")
    print("  RandomResizedCrop : 0.75 - 1.00")
    print("  HorizontalFlip    : p=0.50")
    print("  Rotation          : +/- 8 degrees")
    print("  Translation       : +/- 5%")
    print("  Scale             : 0.95 - 1.05")
    print("  ColorJitter       : mild")
    print("  VerticalFlip      : NO")
    print("  Perspective       : NO")

    print()
    print("Dataset loader test: PASS")