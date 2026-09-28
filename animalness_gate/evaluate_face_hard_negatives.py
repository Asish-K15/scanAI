from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    BASE_DIR
    / "animalness_gate"
    / "outputs"
    / "best_animalness_gate.pth"
)

HARD_NEGATIVE_DIR = (
    BASE_DIR
    / "animalness_gate"
    / "data_source"
    / "face_hard_negative"
)

CLASS_NAMES = [
    "animal",
    "not_animal",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 70)
print("FACE HARD-NEGATIVE DIAGNOSTIC")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

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
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print()
print("Model loaded:")
print(
    f"  Version: "
    f"{checkpoint.get('model_version', 'unknown')}"
)
print(
    f"  Best epoch: "
    f"{checkpoint.get('best_epoch', 'unknown')}"
)


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
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
])


# ------------------------------------------------------------
# Find hard negatives
# ------------------------------------------------------------

paths = sorted([
    p
    for p in HARD_NEGATIVE_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower()
    in IMAGE_EXTENSIONS
])


print()
print(
    f"Hard-negative images: {len(paths)}"
)

if len(paths) != 300:
    raise RuntimeError(
        f"Expected 300 hard-negative images, "
        f"found {len(paths)}"
    )


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

results = []

with torch.no_grad():

    for path in paths:

        image = Image.open(
            path
        ).convert("RGB")

        tensor = transform(
            image
        ).unsqueeze(0).to(device)

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

        animal_probability = float(
            probabilities[0].item()
        )

        not_animal_probability = float(
            probabilities[1].item()
        )

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_class = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        results.append({
            "image_path": str(path),
            "predicted_class":
                predicted_class,
            "animal_probability":
                animal_probability,
            "not_animal_probability":
                not_animal_probability,
        })


df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

not_animal_count = int(
    (
        df["predicted_class"]
        == "not_animal"
    ).sum()
)

animal_count = int(
    (
        df["predicted_class"]
        == "animal"
    ).sum()
)

rejection_rate = (
    not_animal_count
    / len(df)
)


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Total images: "
    f"{len(df)}"
)

print(
    f"Rejected as not_animal: "
    f"{not_animal_count}"
)

print(
    f"Accepted as animal: "
    f"{animal_count}"
)

print(
    f"Rejection rate: "
    f"{rejection_rate:.4f}"
)


# ------------------------------------------------------------
# Probability summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("PROBABILITY SUMMARY")
print("=" * 70)

print(
    df[
        "not_animal_probability"
    ].describe().to_string()
)


# ------------------------------------------------------------
# Top false accepts
# ------------------------------------------------------------

false_accepts = (
    df[
        df["predicted_class"]
        == "animal"
    ]
    .sort_values(
        "animal_probability",
        ascending=False,
    )
)

print()
print("=" * 70)
print("HARD-NEGATIVE FALSE ACCEPTS")
print("=" * 70)

print(
    f"Count: {len(false_accepts)}"
)

if false_accepts.empty:

    print("None")

else:

    print(
        false_accepts[
            [
                "image_path",
                "animal_probability",
                "not_animal_probability",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)