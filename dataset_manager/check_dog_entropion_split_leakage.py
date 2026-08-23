from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd

SOURCE = Path("expansion/dog_entropion")
OUT = Path("metadata/expansion_checks/dog_entropion_split_leakage.csv")

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_images():
    return [
        p for p in SOURCE.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    ]


def phash(path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


images = get_images()

splits = {
    "train": [p for p in images if p.name.startswith("train_")],
    "valid": [p for p in images if p.name.startswith("valid_")],
    "test": [p for p in images if p.name.startswith("test_")]
}

print("=" * 60)
print("DOG ENTROPION CROSS-SPLIT LEAKAGE CHECK")
print("=" * 60)

for split, files in splits.items():
    print(f"{split:>6}: {len(files)}")

print()
print("Hashing images...")

hashes = {}

for p in images:
    h = phash(p)
    if h is not None:
        hashes[p] = h

rows = []

pairs = [
    ("train", "valid"),
    ("train", "test"),
    ("valid", "test")
]

for split_a, split_b in pairs:

    print()
    print(f"Comparing {split_a} <-> {split_b}...")

    for a in splits[split_a]:

        if a not in hashes:
            continue

        best_distance = 999
        best_match = None

        for b in splits[split_b]:

            if b not in hashes:
                continue

            distance = hashes[a] - hashes[b]

            if distance < best_distance:
                best_distance = distance
                best_match = b

        if best_distance <= 8:

            if best_distance == 0:
                category = "exact_duplicate"
            elif best_distance <= 4:
                category = "very_close"
            else:
                category = "near_duplicate"

            rows.append({
                "split_a": split_a,
                "image_a": str(a),
                "split_b": split_b,
                "image_b": str(best_match),
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

if len(df) == 0:
    print("NONE")
else:
    print(df.groupby(
        ["split_a", "split_b", "category"]
    ).size().to_string())

    print()
    print("SUSPICIOUS MATCHES")
    print("-" * 60)

    print(df.sort_values(
        "hamming_distance"
    ).to_string(index=False))

print()
print(f"Report: {OUT}")