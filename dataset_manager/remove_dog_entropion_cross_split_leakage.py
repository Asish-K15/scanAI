from pathlib import Path
import pandas as pd
import shutil

CSV = Path("metadata/expansion_checks/dog_entropion_split_leakage.csv")

SOURCE = Path("expansion/dog_entropion")

BACKUP = Path(
    "metadata/expansion_backup/"
    "dog_entropion_cross_split_leakage"
)

BACKUP.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Only the visually confirmed high-risk matches
df = df[df["hamming_distance"] <= 4].copy()

print("=" * 60)
print("DOG ENTROPION CROSS-SPLIT LEAKAGE CLEANUP")
print("=" * 60)
print(f"Confirmed suspicious pairs : {len(df)}")
print()

to_remove = set()

for _, row in df.iterrows():

    split_a = row["split_a"]
    split_b = row["split_b"]

    path_a = Path(row["image_a"])
    path_b = Path(row["image_b"])

    # We always keep TRAIN.
    if split_a == "train" and split_b in {"valid", "test"}:
        to_remove.add(path_b)

    elif split_b == "train" and split_a in {"valid", "test"}:
        to_remove.add(path_a)

    else:
        print("WARNING: unexpected pair:")
        print(row)

print(f"Files to remove : {len(to_remove)}")
print()

print("FILES TO REMOVE:")
for p in sorted(to_remove, key=str):
    print(" ", p)

print()
print("=" * 60)
print("BACKUP + DELETE")
print("=" * 60)

for p in sorted(to_remove, key=str):

    if not p.exists():
        print("MISSING:", p)
        continue

    destination = BACKUP / p.name

    shutil.copy2(p, destination)
    print("Backup:", destination)

    p.unlink()
    print("Deleted:", p)

print()
print("=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)

print(f"Backed up : {len(list(BACKUP.iterdir()))}")
print(f"Removed   : {len(to_remove)}")