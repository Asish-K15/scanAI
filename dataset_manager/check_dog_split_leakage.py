from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd

SOURCE = Path("expansion/dog_entropion")
OUT = Path("metadata/expansion_checks/dog_entropion_split_leakage.csv")

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_images(split):
    return [
        p for p in SOURCE.glob(f"{split}_*")
        if p.is_file() and p.suffix.lower() in EXTS
    ]


def get_hash(path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


splits = {
    "train": get_images("train"),
    "valid": get_images("valid"),
    "test": get_images("test"),
}

print("=" * 60)
print("DOG ENTROPION CROSS-SPLIT LEAKAGE CHECK")
print("=" * 60)

for name, files in splits.items():
    print(f"{name:>6}: {len(files)}")

print()
print("Hashing images...")

hashes = {}

for split, files in splits.items():
    hashes[split] = {}

    for p in files:
        h = get_hash(p)
        if h is not None:
            hashes[split][p] = h

rows = []

comparisons = [
    ("train", "valid"),
    ("train", "test"),
    ("valid", "test"),
]

for split_a, split_b in comparisons:

    print(f"\nComparing {split_a} <-> {split_b}...")

    for path_a, hash_a in hashes[split_a].items():

        best_distance = 999
        best_match = None

        for path_b, hash_b in hashes[split_b].items():

            distance = hash_a - hash_b

            if distance < best_distance:
                best_distance = distance
                best_match = path_b

        if best_distance == 0:
            category = "exact_duplicate"
        elif best_distance <= 4:
            category = "very_close"
        elif best_distance <= 8:
            category = "near_duplicate"
        else:
            category = "different"

        rows.append({
            "split_a": split_a,
            "image_a": str(path_a),
            "split_b": split_b,
            "best_match": str(best_match),
            "hamming_distance": best_distance,
            "category": category
        })


df = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

for pair in comparisons:
    a, b = pair

    subset = df[
        (df.split_a == a) &
        (df.split_b == b)
    ]

    print(f"\n{a.upper()} <-> {b.upper()}")
    print(subset["category"].value_counts().to_string())

print()
print("=" * 60)
print("ALL CROSS-SPLIT SUSPICIOUS MATCHES")
print("=" * 60)

suspicious = df[df.category != "different"].sort_values(
    ["category", "hamming_distance"]
)

if len(suspicious) == 0:
    print("NONE")
else:
    print(
        suspicious[
            [
                "split_a",
                "image_a",
                "split_b",
                "best_match",
                "hamming_distance",
                "category"
            ]
        ].to_string(index=False)
    )

print()
print(f"Report: {OUT}")