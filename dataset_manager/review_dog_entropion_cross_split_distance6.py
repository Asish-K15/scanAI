from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CSV = Path("metadata/expansion_checks/dog_entropion_split_leakage.csv")
OUT = Path("outputs/spot_check/entropion_cross_split_distance6")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Only highest-risk remaining cross-split matches
df = df[df["hamming_distance"] == 6].copy()

# Remove duplicate pair representations
seen = set()
pairs = []

for _, row in df.iterrows():
    a = row["image_a"]
    b = row["image_b"]

    key = tuple(sorted([a, b]))

    if key in seen:
        continue

    seen.add(key)

    pairs.append({
        "image_a": a,
        "image_b": b,
        "split_a": row["split_a"],
        "split_b": row["split_b"],
        "distance": int(row["hamming_distance"])
    })

print("=" * 60)
print("DOG ENTROPION CROSS-SPLIT DISTANCE-6 REVIEW")
print("=" * 60)
print(f"Unique pairs: {len(pairs)}")
print()

try:
    font = ImageFont.truetype("arial.ttf", 22)
    small_font = ImageFont.truetype("arial.ttf", 15)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

for i, pair in enumerate(pairs, 1):

    p1 = Path(pair["image_a"])
    p2 = Path(pair["image_b"])

    try:
        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")

        width = 500
        height = 400

        im1.thumbnail((width, height))
        im2.thumbnail((width, height))

        canvas = Image.new(
            "RGB",
            (width * 2 + 40, height + 130),
            "white"
        )

        x1 = (width - im1.width) // 2
        x2 = width + 40 + (width - im2.width) // 2

        canvas.paste(im1, (x1, 50))
        canvas.paste(im2, (x2, 50))

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 10),
            f"PAIR {i} | DISTANCE = 6",
            fill="black",
            font=font
        )

        draw.text(
            (10, height + 55),
            f"LEFT [{pair['split_a']}]: {p1.name[:60]}",
            fill="black",
            font=small_font
        )

        draw.text(
            (width + 40, height + 55),
            f"RIGHT [{pair['split_b']}]: {p2.name[:60]}",
            fill="black",
            font=small_font
        )

        canvas.save(
            OUT / f"pair_{i:02d}_distance_6.jpg",
            quality=95
        )

    except Exception as e:
        print("ERROR:", pair, e)

print()
print(f"Saved {len(pairs)} comparisons to:")
print(OUT)