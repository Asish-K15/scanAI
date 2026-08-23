from pathlib import Path
import pandas as pd
import shutil

CSV = Path("metadata/expansion_checks/dog_entropion_internal_duplicates.csv")
SOURCE = Path("expansion/dog_entropion")
BACKUP = Path("metadata/expansion_backup/dog_entropion_exact_duplicates")

BACKUP.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df = df[df["category"] == "exact_duplicate"].copy()

# ---------------------------------------------------------
# Build duplicate groups using union-find
# ---------------------------------------------------------
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra = find(a)
    rb = find(b)
    if ra != rb:
        parent[rb] = ra

for _, row in df.iterrows():
    union(row["image"], row["best_match"])

groups = {}

for path in parent:
    root = find(path)
    groups.setdefault(root, []).append(path)

# ---------------------------------------------------------
# Decide which file to keep
# Preference:
#   1. train
#   2. valid
#   3. test
# ---------------------------------------------------------
to_delete = []
to_keep = []

def split_priority(path):
    name = Path(path).name.lower()

    if name.startswith("train_"):
        return 0
    if name.startswith("valid_"):
        return 1
    if name.startswith("test_"):
        return 2

    return 3

for group in groups.values():

    group = sorted(group)

    # Prefer train copy
    keep = sorted(group, key=lambda x: (split_priority(x), x))[0]

    to_keep.append(keep)

    for path in group:
        if path != keep:
            to_delete.append(path)

# ---------------------------------------------------------
# Print decision BEFORE changing anything
# ---------------------------------------------------------
print("=" * 60)
print("DOG ENTROPION EXACT DUPLICATE CLEANUP")
print("=" * 60)

print(f"Duplicate groups : {len(groups)}")
print(f"Files involved   : {sum(len(g) for g in groups.values())}")
print(f"Keep             : {len(to_keep)}")
print(f"Delete           : {len(to_delete)}")
print()

print("KEEP:")
for p in sorted(to_keep):
    print("  ", p)

print()
print("DELETE:")
for p in sorted(to_delete):
    print("  ", p)

# ---------------------------------------------------------
# Backup first
# ---------------------------------------------------------
print()
print("Backing up files...")

for path in to_delete:
    src = Path(path)

    if not src.exists():
        print("WARNING - missing:", src)
        continue

    shutil.copy2(src, BACKUP / src.name)

print(f"Backup complete: {len(to_delete)} files")

# ---------------------------------------------------------
# Delete
# ---------------------------------------------------------
print()
print("Deleting duplicate copies...")

deleted = 0

for path in to_delete:
    src = Path(path)

    if src.exists():
        src.unlink()
        deleted += 1

print(f"Deleted: {deleted}")

print()
print("=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)