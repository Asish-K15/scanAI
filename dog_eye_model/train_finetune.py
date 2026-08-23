import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from dataset import load_datasets


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
EPOCHS = 8

CLASSIFIER_LR = 1e-4
BACKBONE_LR = 1e-5

WEIGHT_DECAY = 1e-4

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

BASELINE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "efficientnet_b0_baseline_best.pth"
)

FINETUNE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "efficientnet_b0_finetuned_best.pth"
)

HISTORY_PATH = (
    PLOT_DIR
    / "efficientnet_b0_finetuned_history.json"
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
print("EFFICIENTNET-B0 FINE-TUNING")
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
print("Starting from baseline checkpoint:")
print(BASELINE_CHECKPOINT)


# ============================================================
# DATASETS
# ============================================================

train_dataset, valid_dataset, test_dataset = load_datasets()

print()
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

test_loader = DataLoader(
    test_dataset,
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

model = efficientnet_b0(
    weights=None
)

in_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)


# ============================================================
# LOAD BASELINE CHECKPOINT
# ============================================================

print()
print("Loading baseline weights...")

checkpoint = torch.load(
    BASELINE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Baseline epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Baseline validation accuracy: "
    f"{checkpoint['valid_accuracy']:.4f}"
)


# ============================================================
# FREEZE EVERYTHING
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# UNFREEZE UPPER FEATURE BLOCKS
# ============================================================

# EfficientNet-B0 torchvision features:
#
# 0 = stem
# 1-6 = progressively deeper blocks
# 7 = final convolution
# 8 = final normalization
#
# We fine-tune only the upper portion.

for parameter in model.features[6:].parameters():
    parameter.requires_grad = True


# Classifier must remain trainable

for parameter in model.classifier.parameters():
    parameter.requires_grad = True


model = model.to(DEVICE)


# ============================================================
# SHOW TRAINABLE PARAMETERS
# ============================================================

print()
print("=" * 60)
print("TRAINABLE PARAMETERS")
print("=" * 60)

trainable_parameters = 0
total_parameters = 0

for name, parameter in model.named_parameters():

    total_parameters += parameter.numel()

    if parameter.requires_grad:

        trainable_parameters += parameter.numel()

        print(name)


print()
print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)

print(
    "Total parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable percentage:",
    f"{100 * trainable_parameters / total_parameters:.2f}%"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# DISCRIMINATIVE LEARNING RATES
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.features[6:].parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch():

    model.train()

    # Keep the frozen early backbone in evaluation mode.
    # This prevents frozen BatchNorm statistics from changing.

    model.features[:6].eval()

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

        optimizer.zero_grad(
            set_to_none=True
        )

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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# TRAINING LOOP
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "valid_loss": [],
    "valid_accuracy": [],
    "learning_rate_backbone": [],
    "learning_rate_classifier": [],
}


best_valid_accuracy = (
    checkpoint["valid_accuracy"]
)


# Start with the baseline as the current best.
#
# This is important:
# fine-tuning must actually beat the baseline
# before we consider it an improvement.

best_epoch = checkpoint["epoch"]


print()
print("=" * 60)
print("FINE-TUNING")
print("=" * 60)

print()
print(
    f"Starting validation accuracy: "
    f"{best_valid_accuracy:.4f}"
)


for epoch in range(EPOCHS):

    start_time = time.time()

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    valid_loss, valid_accuracy = (
        validate()
    )

    scheduler.step()

    elapsed = (
        time.time()
        - start_time
    )

    backbone_lr = (
        optimizer.param_groups[0]["lr"]
    )

    classifier_lr = (
        optimizer.param_groups[1]["lr"]
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["valid_loss"].append(
        valid_loss
    )

    history["valid_accuracy"].append(
        valid_accuracy
    )

    history["learning_rate_backbone"].append(
        backbone_lr
    )

    history["learning_rate_classifier"].append(
        classifier_lr
    )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"| "
        f"Train Loss: {train_loss:.4f} "
        f"| "
        f"Train Acc: {train_accuracy:.4f} "
        f"| "
        f"Val Loss: {valid_loss:.4f} "
        f"| "
        f"Val Acc: {valid_accuracy:.4f} "
        f"| "
        f"Backbone LR: {backbone_lr:.2e} "
        f"| "
        f"Time: {elapsed:.1f}s"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if valid_accuracy > best_valid_accuracy:

        best_valid_accuracy = (
            valid_accuracy
        )

        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": epoch + 1,
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
                "base_model":
                    "efficientnet_b0_baseline",
                "fine_tuned":
                    True,
            },
            FINETUNE_CHECKPOINT,
        )

        print(
            f"  BEST FINE-TUNED MODEL SAVED "
            f"({valid_accuracy:.4f})"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    HISTORY_PATH,
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
print("FINE-TUNING COMPLETE")
print("=" * 60)

print()
print(
    f"Best validation accuracy: "
    f"{best_valid_accuracy:.4f}"
)

print(
    f"Best epoch: {best_epoch}"
)

print()
print("Fine-tuned checkpoint:")

if FINETUNE_CHECKPOINT.exists():

    print(FINETUNE_CHECKPOINT)

else:

    print(
        "No fine-tuned checkpoint was created."
    )

    print(
        "The fine-tuned model did not "
        "beat the baseline validation accuracy."
    )

print()
print("History:")
print(HISTORY_PATH)

print()
print("=" * 60)
print("IMPORTANT")
print("=" * 60)

print()
print(
    "The test set was NOT used during fine-tuning."
)

print(
    "Run the independent test evaluation "
    "only after fine-tuning is complete."
)