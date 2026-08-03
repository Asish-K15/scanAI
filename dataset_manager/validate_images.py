"""
Stage 5: Image Validation
---------------------------
Flags images that are corrupt, too small, or too blurry to be useful for
training. Does NOT delete anything automatically — writes a report so you
can eyeball a sample before removing.

Blur is estimated via the variance of the Laplacian (a standard, simple
blur-detection heuristic) - lower variance means blurrier.
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

CLEAN_DIR = Path(__file__).parent / "clean"
METADATA_DIR = Path(__file__).parent / "metadata"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_DIMENSION = 100          # px, below this an image is too small to be useful
BLUR_VARIANCE_THRESHOLD = 30  # below this, flag as likely too blurry


def check_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return {"status": "corrupt", "reason": "unreadable"}

    h, w = img.shape[:2]
    if min(h, w) < MIN_DIMENSION:
        return {"status": "flagged", "reason": f"too_small ({w}x{h})"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < BLUR_VARIANCE_THRESHOLD:
        return {"status": "flagged", "reason": f"likely_blurry (var={variance:.1f})"}

    return {"status": "ok", "reason": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-corrupt", action="store_true",
                         help="Delete only unreadable/corrupt files (safe to automate)")
    args = parser.parse_args()

    METADATA_DIR.mkdir(exist_ok=True)
    all_images = [p for p in CLEAN_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    rows = []
    corrupt_count = 0
    flagged_count = 0

    for img_path in tqdm(all_images, desc="validating"):
        result = check_image(img_path)
        rows.append([str(img_path), result["status"], result["reason"]])
        if result["status"] == "corrupt":
            corrupt_count += 1
            if args.remove_corrupt:
                img_path.unlink()
        elif result["status"] == "flagged":
            flagged_count += 1

    report_path = METADATA_DIR / "image_quality_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "status", "reason"])
        writer.writerows(rows)

    print(f"\nScanned {len(all_images)} images")
    print(f"  corrupt/unreadable: {corrupt_count}" + (" (removed)" if args.remove_corrupt else ""))
    print(f"  flagged (small/blurry): {flagged_count} — review these manually before deleting")
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
