from pathlib import Path
from PIL import Image
import imagehash
import pandas as pd

CONJ = Path("expansion/dog_conjunctivitis")
ENTR = Path("expansion/dog_entropion")

OUT = Path(
    "metadata/expansion_checks/dog_cross_class_duplicates.csv"
)

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    ]


def get_hash(path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None


conj_images = get_images(CONJ)
entr_images = get_images(ENTR)

print("=" * 60)
print("DOG CROSS-CLASS DUPLICATE CHECK")
print("=" * 60)

print(f"Conjunctivitis: {len(conj_images)}")
print(f"Entropion:      {len(entr_images)}")
print()

print("Hashing Conjunctivitis...")
conj_hashes = {}

for i, p in enumerate(conj_images, 1):
    h = get_hash(p)

    if h is not None:
        conj_hashes[p] = h

    if i % 100 == 0:
        print(f"  {i}/{len(conj_images)}")

print()
print("Hashing Entropion...")
entr_hashes = {}

for i, p in enumerate(entr_images, 1):
    h = get_hash(p)

    if h is not None:
        entr_hashes[p] = h

    if i % 100 == 0:
        print(f"  {i}/{len(entr_images)}")

print()
print("Comparing classes...")

rows = []

for conj_path, conj_hash in conj_hashes.items():

    best_distance = 999
    best_match = None

    for entr_path, entr_hash in entr_hashes.items():

        distance = conj_hash - entr_hash

        if distance < best_distance:
            best_distance = distance
            best_match = entr_path

    if best_distance == 0:
        category = "exact_duplicate"
    elif best_distance <= 4:
        category = "very_close"
    elif best_distance <= 8:
        category = "near_duplicate"
    else:
        category = "different"

    rows.append({
        "conjunctivitis": str(conj_path),
        "entropion": str(best_match),
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

print(df["category"].value_counts().to_string())

print()
print("=" * 60)
print("CROSS-CLASS SUSPICIOUS MATCHES")
print("=" * 60)

suspicious = df[
    df["category"] != "different"
].sort_values("hamming_distance")

if suspicious.empty:
    print("NONE")
else:
    print(
        suspicious[
            [
                "conjunctivitis",
                "entropion",
                "hamming_distance",
                "category"
            ]
        ].to_string(index=False)
    )

print()
print(f"Report: {OUT}")