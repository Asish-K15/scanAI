from pathlib import Path
import pandas as pd
import shutil

CSV = Path("metadata/expansion_checks/dog_entropion_split_leakage.csv")

BACKUP = Path(
    "metadata/expansion_backup/dog_entropion_distance6_candidates"
)

REPORT = Path(
    "metadata/expansion_checks/dog_entropion_distance6_cleanup.csv"
)

BACKUP.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Only distance-6 cross-split matches
df = df[df["hamming_distance"] == 6].copy()

delete = {}

for _, row in df.iterrows():

    a = row["image_a"]
    b = row["image_b"]

    split_a = row["split_a"]
    split_b = row["split_b"]

    # Prefer TRAIN over VALID/TEST
    if split_a == "train" and split_b in {"valid", "test"}:
        keep = a
        remove = b

    elif split_b == "train" and split_a in {"valid", "test"}:
        keep = b
        remove = a

    # Between VALID and TEST, prefer VALID
    elif split_a == "valid" and split_b == "test":
        keep = a
        remove = b

    elif split_b == "valid" and split_a == "test":
        keep = b
        remove = a

    else:
        continue

    delete[remove] = {
        "remove": remove,
        "keep": keep,
        "split_remove": split_a if remove == a else split_b,
        "split_keep": split_a if keep == a else split_b,
        "distance": int(row["hamming_distance"])
    }

rows = list(delete.values())

out = pd.DataFrame(rows)

out.to_csv(REPORT, index=False)

print("=" * 60)
print("DOG ENTROPION DISTANCE-6 CLEANUP PREVIEW")
print("=" * 60)

print(f"Distance-6 pairs       : {len(df)}")
print(f"Unique deletion files  : {len(rows)}")
print()

print("DELETION COUNTS:")
if len(out):
    print(out["split_remove"].value_counts().to_string())

print()
print("FILES THAT WOULD BE REMOVED:")
print()

for r in rows:
    print(
        f"REMOVE [{r['split_remove']}] {r['remove']}"
    )
    print(
        f"KEEP   [{r['split_keep']}] {r['keep']}"
    )
    print("-" * 60)

print()
print(f"Report: {REPORT}")