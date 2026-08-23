from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV = Path("metadata/expansion_checks/dog_entropion_split_leakage.csv")
OUT = Path("outputs/spot_check/entropion_cross_split_distance8")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Only cross-split distance-8 matches
df = df[
    (df["category"] == "near_duplicate") &
    (df["hamming_distance"] == 8)
].copy()

print("=" * 60)
print("DOG ENTROPION CROSS-SPLIT DISTANCE-8 REVIEW")
print("=" * 60)
print(f"Pairs: {len(df)}")
print()

try:
    font = ImageFont.truetype("arial.ttf", 20)
    small_font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

for i, row in enumerate(df.itertuples(index=False), 1):

    p1 = Path(row.image_a)
    p2 = Path(row.best_match)

    try:
        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")

        width = 500
        height = 400

        im1.thumbnail((width, height))
        im2.thumbnail((width, height))

        canvas = Image.new(
            "RGB",
            (width * 2 + 40, height + 150),
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
            f"{row.split_a.upper()}: {p1.name[:70]}",
            fill="black",
            font=small_font
        )

        draw.text(
            (width + 40, height + 60),
            f"{row.split_b.upper()}: {p2.name[:70]}",
            fill="black",
            font=small_font
        )

        canvas.save(
            OUT / f"pair_{i:02d}_distance_8.jpg",
            quality=95
        )

    except Exception as e:
        print("ERROR:", e)

print()
print(f"Saved comparisons to:")
print(OUT)