"""
Validate a specific staging dataset without touching clean/.

Uses the same validation rules as validate_images.py.
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_DIMENSION = 100
BLUR_VARIANCE_THRESHOLD = 30


def check_image(path: Path):
    img = cv2.imread(str(path))

    if img is None:
        return {
            "status": "corrupt",
            "reason": "unreadable"
        }

    h, w = img.shape[:2]

    if min(h, w) < MIN_DIMENSION:
        return {
            "status": "flagged",
            "reason": f"too_small ({w}x{h})"
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < BLUR_VARIANCE_THRESHOLD:
        return {
            "status": "flagged",
            "reason": f"likely_blurry (var={variance:.1f})"
        }

    return {
        "status": "ok",
        "reason": ""
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Folder containing images to validate"
    )

    parser.add_argument(
        "--out",
        default="metadata/subset_quality_report.csv",
        help="Output CSV report"
    )

    args = parser.parse_args()

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    images = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    rows = []

    corrupt_count = 0
    flagged_count = 0
    ok_count = 0

    for img_path in tqdm(images, desc="validating"):

        result = check_image(img_path)

        rows.append([
            str(img_path),
            result["status"],
            result["reason"]
        ])

        if result["status"] == "corrupt":
            corrupt_count += 1

        elif result["status"] == "flagged":
            flagged_count += 1

        else:
            ok_count += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "path",
            "status",
            "reason"
        ])

        writer.writerows(rows)

    print()
    print("=" * 60)
    print("SUBSET IMAGE VALIDATION")
    print("=" * 60)

    print(f"Source:              {source}")
    print(f"Images scanned:      {len(images)}")
    print(f"OK:                  {ok_count}")
    print(f"Flagged:             {flagged_count}")
    print(f"Corrupt/unreadable:  {corrupt_count}")

    print("-" * 60)
    print(f"Report: {out}")
    print("=" * 60)

    print()
    print("IMPORTANT:")
    print("No images were deleted.")


if __name__ == "__main__":
    main()