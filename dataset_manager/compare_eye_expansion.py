from pathlib import Path
from PIL import Image
import pandas as pd
import imagehash

EXISTING = Path("clean/eye/eye_infection")
NEW = Path("expansion/eye_infection_cat")
OUT = Path("metadata/expansion_checks/eye_infection_comparison.csv")

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
        print(f"Could not read: {path} -> {e}")
        return None


existing = get_images(EXISTING)
new = get_images(NEW)

print("=" * 60)
print("EYE INFECTION EXPANSION DUPLICATE CHECK")
print("=" * 60)
print(f"Existing images : {len(existing)}")
print(f"New images      : {len(new)}")
print()

existing_hashes = {}

print("Hashing existing images...")
for p in existing:
    h = phash(p)
    if h is not None:
        existing_hashes[p] = h

print("Hashing new images...")
new_hashes = {}

for p in new:
    h = phash(p)
    if h is not None:
        new_hashes[p] = h


rows = []

for new_path, new_hash in new_hashes.items():

    best_distance = 999
    best_match = None

    for old_path, old_hash in existing_hashes.items():

        distance = new_hash - old_hash

        if distance < best_distance:
            best_distance = distance
            best_match = old_path

    if best_distance == 0:
        category = "exact_duplicate"

    elif best_distance <= 4:
        category = "very_close"

    elif best_distance <= 8:
        category = "near_duplicate"

    else:
        category = "different"

    rows.append({
        "new_image": str(new_path),
        "best_existing_match": str(best_match),
        "hamming_distance": best_distance,
        "category": category
    })


df = pd.DataFrame(rows)

df = df.sort_values(
    ["category", "hamming_distance", "new_image"]
)

OUT.parent.mkdir(parents=True, exist_ok=True)
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