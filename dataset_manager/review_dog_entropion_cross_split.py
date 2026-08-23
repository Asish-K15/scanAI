from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV = Path("metadata/expansion_checks/dog_entropion_split_leakage.csv")
OUT = Path("outputs/spot_check/entropion_cross_split_review")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Only strongest suspicious matches first
df = df[df["hamming_distance"] <= 4].copy()

print("=" * 60)
print("DOG ENTROPION CROSS-SPLIT HIGH-RISK REVIEW")
print("=" * 60)
print(f"Pairs: {len(df)}")
print()

try:
    font = ImageFont.truetype("arial.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()
    small = ImageFont.load_default()

for i, row in enumerate(df.itertuples(index=False), 1):

    p1 = Path(row.image_a)
    p2 = Path(row.image_b)

    try:
        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")

        width = 500
        height = 400

        im1.thumbnail((width, height))
        im2.thumbnail((width, height))

        canvas = Image.new(
            "RGB",
            (width * 2 + 40, height + 120),
            "white"
        )

        x1 = (width - im1.width) // 2
        x2 = width + 40 + (width - im2.width) // 2

        canvas.paste(im1, (x1, 50))
        canvas.paste(im2, (x2, 50))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 10),
            f"PAIR {i} | DISTANCE {row.hamming_distance}",
            fill="black",
            font=font
        )

        draw.text(
            (10, height + 60),
            f"LEFT [{row.split_a}]: {p1.name[:60]}",
            fill="black",
            font=small
        )

        draw.text(
            (width + 40, height + 60),
            f"RIGHT [{row.split_b}]: {p2.name[:60]}",
            fill="black",
            font=small
        )

        canvas.save(
            OUT / f"pair_{i:02d}_distance_{row.hamming_distance}.jpg",
            quality=95
        )

    except Exception as e:
        print("ERROR:", e)

print()
print(f"Saved {len(df)} comparisons to:")
print(OUT)