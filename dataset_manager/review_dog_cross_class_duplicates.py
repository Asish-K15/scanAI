from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV = Path("metadata/expansion_checks/dog_cross_class_duplicates.csv")
OUT = Path("outputs/spot_check/dog_cross_class_review")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df = df[df["category"] != "different"].sort_values(
    "hamming_distance"
)

print("=" * 60)
print("DOG CROSS-CLASS DUPLICATE REVIEW")
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

    p1 = Path(row.conjunctivitis)
    p2 = Path(row.entropion)

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

        canvas.paste(im1, (x1, 55))
        canvas.paste(im2, (x2, 55))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 10),
            f"PAIR {i} | DISTANCE {row.hamming_distance} | "
            f"{row.category}",
            fill="black",
            font=font
        )

        draw.text(
            (10, height + 65),
            "CONJ: " + p1.name[:65],
            fill="black",
            font=small_font
        )

        draw.text(
            (width + 40, height + 65),
            "ENTR: " + p2.name[:65],
            fill="black",
            font=small_font
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