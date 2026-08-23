import torch
from torch.utils.data import DataLoader

from dataset import load_datasets


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DOG EYE DATALOADER + GPU TEST")
print("=" * 60)

print()
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset, valid_dataset, test_dataset = load_datasets()


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)


# ============================================================
# GET ONE BATCH
# ============================================================

images, labels = next(iter(train_loader))

print()
print("CPU BATCH")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Images dtype:", images.dtype)
print("Labels dtype:", labels.dtype)


# ============================================================
# MOVE TO GPU
# ============================================================

images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

print()
print("GPU BATCH")
print("Images device:", images.device)
print("Labels device:", labels.device)


# ============================================================
# GPU SANITY COMPUTATION
# ============================================================

dummy = images.mean()

torch.cuda.synchronize() if device.type == "cuda" else None

print()
print("GPU computation successful.")
print("Dummy tensor device:", dummy.device)

if device.type == "cuda":
    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        "GB"
    )

print()
print("DATALOADER + GPU TEST: PASS")