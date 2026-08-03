"""
Stage 4: Perceptual Duplicate Detection
-----------------------------------------
Scans every image in clean/, computes a perceptual hash (pHash) for each,
and flags near-duplicates — including across different source datasets,
which is the case that matters most (many Roboflow/Kaggle sets scrape the
same original web images).

Outputs:
  metadata/duplicates.csv   - every duplicate pair found, with similarity distance
  metadata/dedup_report.txt - summary counts

Run AFTER taxonomy_mapper.py. This does NOT delete anything automatically —
review duplicates.csv first, then run with --remove to actually delete them.
"""

import argparse
import csv
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

CLEAN_DIR = Path(__file__).parent / "clean"
METADATA_DIR = Path(__file__).parent / "metadata"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAMMING_THRESHOLD = 5  # lower = stricter match; 5 is a reasonable near-duplicate cutoff


def compute_hashes():
    hashes = {}
    all_images = [p for p in CLEAN_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    print(f"Hashing {len(all_images)} images...")
    for img_path in tqdm(all_images):
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img)
                hashes[img_path] = h
        except Exception as e:
            print(f"  !! couldn't read {img_path.name}: {e}")
    return hashes


def find_duplicates(hashes: dict):
    """O(n^2) comparison — fine for a few thousand images per class; if you
    have tens of thousands total, bucket by hash prefix first to speed this up."""
    items = list(hashes.items())
    duplicates = []
    for i in tqdm(range(len(items)), desc="comparing"):
        path_a, hash_a = items[i]
        for j in range(i + 1, len(items)):
            path_b, hash_b = items[j]
            distance = hash_a - hash_b
            if distance <= HAMMING_THRESHOLD:
                duplicates.append((str(path_a), str(path_b), distance))
    return duplicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true",
                         help="Actually delete the second image in each duplicate pair")
    args = parser.parse_args()

    METADATA_DIR.mkdir(exist_ok=True)

    hashes = compute_hashes()
    duplicates = find_duplicates(hashes)

    csv_path = METADATA_DIR / "duplicates.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_a", "image_b", "hamming_distance"])
        writer.writerows(duplicates)

    report_path = METADATA_DIR / "dedup_report.txt"
    total_images = len(hashes)
    dup_pct = (len(duplicates) / total_images * 100) if total_images else 0
    with open(report_path, "w") as f:
        f.write(f"Total images scanned: {total_images}\n")
        f.write(f"Duplicate pairs found: {len(duplicates)}\n")
        f.write(f"Approx duplicate rate: {dup_pct:.1f}%\n")

    print(f"\n{len(duplicates)} duplicate pairs found ({dup_pct:.1f}% of {total_images} images)")
    print(f"Details written to {csv_path}")

    if args.remove:
        removed = 0
        for _, path_b, _ in duplicates:
            p = Path(path_b)
            if p.exists():
                p.unlink()
                removed += 1
        print(f"Removed {removed} duplicate images.")
    else:
        print("Review duplicates.csv, then re-run with --remove to actually delete them.")


if __name__ == "__main__":
    main()
