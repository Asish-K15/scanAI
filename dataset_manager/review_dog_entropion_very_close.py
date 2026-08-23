from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

CSV = Path("metadata/expansion_checks/dog_entropion_internal_duplicates.csv")
OUT = Path("outputs/spot_check/entropion_very_close_pairs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df = df[df["category"] == "very_close"].copy()

seen = set()
pairs = []

for _, row in df.iterrows():
    a = row["image"]
    b = row["best_match"]

    key = tuple(sorted([a, b]))

    if key in seen:
        continue

    seen.add(key)

    pairs.append({
        "image": a,
        "match": b,
        "distance": int(row["hamming_distance"])
    })

print("=" * 60)
print("DOG ENTROPION VERY-CLOSE PAIR REVIEW")
print("=" * 60)
print(f"Unique pairs: {len(pairs)}")
print()

# Font
try:
    font = ImageFont.truetype("arial.ttf", 18)
    small_font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

for i, pair in enumerate(pairs, 1):

    p1 = Path(pair["image"])
    p2 = Path(pair["match"])

    try:
        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")

        width = 500
        height = 400

        im1.thumbnail((width, height))
        im2.thumbnail((width, height))

        canvas = Image.new(
            "RGB",
            (width * 2 + 40, height + 100),
            "white"
        )

        x1 = (width - im1.width) // 2
        x2 = width + 40 + (width - im2.width) // 2

        canvas.paste(im1, (x1, 50))
        canvas.paste(im2, (x2, 50))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 10),
            f"PAIR {i}    HAMMING DISTANCE = {pair['distance']}",
            fill="black",
            font=font
        )

        draw.text(
            (10, height + 55),
            "LEFT: " + p1.name[:65],
            fill="black",
            font=small_font
        )

        draw.text(
            (width + 40, height + 55),
            "RIGHT: " + p2.name[:65],
            fill="black",
            font=small_font
        )

        canvas.save(
            OUT / f"pair_{i:02d}_distance_{pair['distance']}.jpg",
            quality=95
        )

    except Exception as e:
        print("ERROR:", pair, e)

print()
print(f"Saved {len(pairs)} pair images to:")
print(OUT)