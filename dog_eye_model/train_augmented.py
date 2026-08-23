import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

from dataset import load_datasets


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "checkpoints"
)

PLOT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "plots"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("SCANAI DOG EYE DISEASE")
print("EFFICIENTNET-B0 TARGETED AUGMENTATION")
print("EXPERIMENT 3")
print("=" * 60)

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

print()


# ============================================================
# DATASET
# ============================================================

train_dataset, valid_dataset, test_dataset = (
    load_datasets()
)

print("Dataset:")
print("  Train:", len(train_dataset))
print("  Valid:", len(valid_dataset))
print("  Test :", len(test_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)


# Freeze pretrained backbone
for parameter in model.features.parameters():
    parameter.requires_grad = False


# Replace classifier
in_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    model.train()

    # Frozen backbone stays in evaluation mode.
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        correct / total,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in valid_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        correct / total,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

history = {
    "experiment": "efficientnet_b0_targeted_augmentation",
    "train_loss": [],
    "train_accuracy": [],
    "valid_loss": [],
    "valid_accuracy": [],
}

best_valid_accuracy = 0.0

best_checkpoint = (
    CHECKPOINT_DIR
    / "efficientnet_b0_augmented_best.pth"
)


print()
print("=" * 60)
print("EXPERIMENT 3 TRAINING")
print("=" * 60)

print()
print("Augmentation:")
print("  RandomResizedCrop : 0.75 - 1.00")
print("  HorizontalFlip    : 0.50")
print("  Rotation          : +/- 8 degrees")
print("  Translation       : +/- 5%")
print("  Scale             : 0.95 - 1.05")
print("  ColorJitter       : mild")

print()


for epoch in range(EPOCHS):

    start_time = time.time()

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    valid_loss, valid_accuracy = (
        validate()
    )

    elapsed = (
        time.time()
        - start_time
    )

    history[
        "train_loss"
    ].append(train_loss)

    history[
        "train_accuracy"
    ].append(train_accuracy)

    history[
        "valid_loss"
    ].append(valid_loss)

    history[
        "valid_accuracy"
    ].append(valid_accuracy)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {valid_loss:.4f} "
        f"| Val Acc: {valid_accuracy:.4f} "
        f"| Time: {elapsed:.1f}s"
    )

    if valid_accuracy > best_valid_accuracy:

        best_valid_accuracy = (
            valid_accuracy
        )

        torch.save(
            {
                "experiment":
                    "efficientnet_b0_targeted_augmentation",

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "valid_accuracy":
                    valid_accuracy,

                "class_to_idx": {
                    "conjunctivitis": 0,
                    "entropion": 1,
                },
            },
            best_checkpoint,
        )

        print(
            f"  BEST MODEL SAVED "
            f"({valid_accuracy:.4f})"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    PLOT_DIR
    / "efficientnet_b0_augmented_history.json"
)

with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("EXPERIMENT 3 COMPLETE")
print("=" * 60)

print()
print(
    f"Best validation accuracy: "
    f"{best_valid_accuracy:.4f}"
)

print()
print("Checkpoint:")
print(best_checkpoint)

print()
print("History:")
print(history_path)

print()
print("=" * 60)
print("IMPORTANT")
print("=" * 60)
print(
    "Test set was NOT used during training."
)
print(
    "Run independent test evaluation next."
)