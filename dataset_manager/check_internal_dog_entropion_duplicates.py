from pathlib import Path
from PIL import Image
import pandas as pd
import imagehash

SOURCE = Path("expansion/dog_entropion")
OUT = Path("metadata/expansion_checks/dog_entropion_internal_duplicates.csv")

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    ]


def phash(path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


images = get_images(SOURCE)

print("=" * 60)
print("DOG ENTROPION INTERNAL DUPLICATE CHECK")
print("=" * 60)
print(f"Images: {len(images)}")
print()

hashes = {}

print("Hashing images...")

for i, p in enumerate(images, 1):
    h = phash(p)

    if h is not None:
        hashes[p] = h

    if i % 100 == 0:
        print(f"  {i}/{len(images)}")


rows = []

paths = list(hashes.keys())

print()
print("Comparing images...")

for i in range(len(paths)):

    best_distance = 999
    best_match = None

    for j in range(len(paths)):

        if i == j:
            continue

        distance = hashes[paths[i]] - hashes[paths[j]]

        if distance < best_distance:
            best_distance = distance
            best_match = paths[j]

    if best_distance == 0:
        category = "exact_duplicate"
    elif best_distance <= 4:
        category = "very_close"
    elif best_distance <= 8:
        category = "near_duplicate"
    else:
        category = "different"

    rows.append({
        "image": str(paths[i]),
        "best_match": str(best_match),
        "hamming_distance": best_distance,
        "category": category
    })


df = pd.DataFrame(rows)

df.to_csv(OUT, index=False)

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(df["category"].value_counts().to_string())

print()
print(f"Report: {OUT}")
print()
print("NO FILES WERE DELETED OR MOVED.")