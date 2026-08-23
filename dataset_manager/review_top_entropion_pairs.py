from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV = Path("metadata/expansion_checks/dog_entropion_very_close_analysis.csv")
OUT = Path("outputs/spot_check/entropion_top_pairs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Strongest structural matches
df = df[df["ssim"] >= 0.85].copy()
df = df.sort_values("ssim", ascending=False)

print("=" * 60)
print("TOP ENTROPION VERY-CLOSE PAIRS")
print("=" * 60)
print("Pairs:", len(df))
print()

for i, row in df.iterrows():
    print(
        f"{row['ssim']:.4f} | "
        f"{row['image']} <-> {row['best_match']}"
    )

# Create individual side-by-side images
for n, (_, row) in enumerate(df.iterrows(), 1):

    p1 = Path(row["image"])
    p2 = Path(row["best_match"])

    try:
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        size = (500, 500)

        img1.thumbnail(size)
        img2.thumbnail(size)

        canvas = Image.new("RGB", (1050, 620), "white")

        x1 = (500 - img1.width) // 2
        x2 = 550 + (500 - img2.width) // 2

        canvas.paste(img1, (x1, 20))
        canvas.paste(img2, (x2, 20))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 535),
            f"LEFT: {p1.name}",
            fill="black"
        )

        draw.text(
            (10, 560),
            f"RIGHT: {p2.name}",
            fill="black"
        )

        draw.text(
            (10, 590),
            f"SSIM={row['ssim']:.4f} | "
            f"pHash={int(row['hamming_distance'])} | "
            f"pixel_diff={row['pixel_difference']:.4f}",
            fill="black"
        )

        canvas.save(
            OUT / f"pair_{n:02d}_ssim_{row['ssim']:.4f}.jpg"
        )

    except Exception as e:
        print("ERROR:", e)

print()
print("Saved comparisons to:")
print(OUT)