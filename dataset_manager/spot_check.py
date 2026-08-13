"""
spot_check.py — Build side-by-side contact sheets for manual taxonomy review.

Use this to visually verify ambiguous or newly-classified classes BEFORE the
dataset freeze. The taxonomy mapper leans on domain decisions in taxonomy.json,
but you (and your partner, independently) should eyeball a sample of images to
confirm a class is genuinely photo-diagnosable and correctly named.

Example:
    python spot_check.py --source "raw\\kaggle__devang03mgr__cattle-diseases-datasets\\foot-and-mouth"
                          --source "raw\\roboflow__cattle-diseases\\(BRD) Disease Ecthym"
                          --out outputs\\spot_check --samples 15

Flags:
  --source PATH   (repeatable) a raw (or clean) class folder to sample
  --samples N     number of images per class to show (default 15)
  --out DIR       output directory for the contact-sheet PNGs (default outputs/spot_check)
  --cols N        columns in the grid (default 5)

Each source produces one PNG: <out>/<classname>.png
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMB = 256


def build_contact_sheet(source: Path, samples: int, cols: int, out_dir: Path,
                        randomize: bool, seed: int):
    images = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"  !! no images in {source}")
        return

    if randomize:
        rng = random.Random(seed)
        images = rng.sample(images, min(samples, len(images)))
        choice = f"random sample (seed={seed})"
    else:
        images = images[:samples]
        choice = f"first {samples}"

    rows = (len(images) + cols - 1) // cols

    pad = 10
    label = 22
    sheet_w = cols * THUMB + (cols + 1) * pad
    sheet_h = rows * (THUMB + label) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, sheet_w - 1, sheet_h - 1], outline="black")

    for i, img_path in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (THUMB + pad)
        y = pad + r * (THUMB + label + pad)
        try:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((THUMB, THUMB))
            offset = ((THUMB - im.width) // 2, (THUMB - im.height) // 2)
            sheet.paste(im, (x + offset[0], y + offset[1]))
            draw.text((x, y + THUMB), img_path.name[:34], fill="black")
        except Exception as exc:  # corrupt/unsupported image
            draw.text((x + 4, y + 4), f"ERR: {exc}", fill="red")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.name}.png"
    sheet.save(out_path)
    print(f"  saved {out_path}  ({choice}, {len(images)} images, {cols}x{rows})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="class folder to sample (repeatable)")
    ap.add_argument("--samples", type=int, default=16,
                    help="images per class to show (16-25 recommended)")
    ap.add_argument("--out", default="outputs/spot_check")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--random", action="store_true",
                    help="randomly sample instead of taking the first N")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds the random sampler for reproducibility")
    args = ap.parse_args()

    base = Path(__file__).parent
    out_dir = base / args.out
    for src in args.source:
        build_contact_sheet(base / src, args.samples, args.cols, out_dir,
                            args.random, args.seed)


if __name__ == "__main__":
    main()
