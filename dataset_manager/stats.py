"""
Stage 6: Dataset Statistics
-----------------------------
Walks clean/ and produces:
  metadata/dataset_inventory.csv - image count per unified class
  A console summary flagging classes below the 200-image minimum-viable
  baseline from your target-count plan, so you know exactly where to focus
  self-collection effort next.
"""

import csv
from pathlib import Path

CLEAN_DIR = Path(__file__).parent / "clean"
METADATA_DIR = Path(__file__).parent / "metadata"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_VIABLE = 200
GOOD = 500
EXCELLENT = 800


def tier(count: int) -> str:
    if count >= EXCELLENT:
        return "excellent"
    if count >= GOOD:
        return "good"
    if count >= MIN_VIABLE:
        return "minimum_viable"
    return "BELOW_MINIMUM"


def main():
    METADATA_DIR.mkdir(exist_ok=True)

    rows = []
    for group_dir in sorted(CLEAN_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        for class_dir in sorted(group_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            unified_class = f"{group_dir.name}/{class_dir.name}"
            rows.append([unified_class, count, tier(count)])

    rows.sort(key=lambda r: r[1])  # weakest classes first

    inventory_path = METADATA_DIR / "dataset_inventory.csv"
    with open(inventory_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "image_count", "tier"])
        writer.writerows(rows)

    total = sum(r[1] for r in rows)
    below_min = [r for r in rows if r[2] == "BELOW_MINIMUM"]

    print(f"Total images across {len(rows)} classes: {total}\n")
    print(f"{len(below_min)} classes below minimum-viable (200 images) — prioritize these:")
    for cls, count, _ in below_min:
        print(f"  {cls}: {count}")

    print(f"\nFull breakdown written to {inventory_path}")


if __name__ == "__main__":
    main()
