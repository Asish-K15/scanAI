from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import math

REPORT = Path("metadata/expansion_checks/dog_internal_duplicates.csv")
OUT = Path("outputs/spot_check/dog_internal_duplicates_review.png")

df = pd.read_csv(REPORT)

df = df[df["category"].isin(["very_close", "near_duplicate"])].copy()

# Remove duplicate pair views
seen = set()
pairs = []

for _, row in df.sort_values("hamming_distance").iterrows():

    a = row["image"]
    b = row["best_match"]

    key = tuple(sorted([a, b]))

    if key in seen:
        continue

    seen.add(key)

    pairs.append({
        "image": a,
        "match": b,
        "distance": row["hamming_distance"],
        "category": row["category"]
    })

print("Suspicious unique pairs:", len(pairs))

THUMB_W = 450
THUMB_H = 300
LABEL_H = 65
PAIR_H = THUMB_H + LABEL_H

canvas = Image.new(
    "RGB",
    (THUMB_W * 2, PAIR_H * len(pairs)),
    "white"
)

draw = ImageDraw.Draw(canvas)


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((THUMB_W - 20, THUMB_H - 20))
        return img
    except Exception:
        return None


for i, pair in enumerate(pairs):

    y = i * PAIR_H

    img1 = load_image(pair["image"])
    img2 = load_image(pair["match"])

    if img1:
        x = (THUMB_W - img1.width) // 2
        canvas.paste(img1, (x, y + 5))

    if img2:
        x = THUMB_W + (THUMB_W - img2.width) // 2
        canvas.paste(img2, (x, y + 5))

    name1 = Path(pair["image"]).name[:55]
    name2 = Path(pair["match"]).name[:55]

    draw.text(
        (10, y + THUMB_H + 5),
        f"NEW: {name1}",
        fill="black"
    )

    draw.text(
        (THUMB_W + 10, y + THUMB_H + 5),
        f"MATCH: {name2}",
        fill="black"
    )

    draw.text(
        (10, y + THUMB_H + 35),
        f"{pair['category']} | distance={pair['distance']}",
        fill="red"
    )


OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT)

print(f"Saved: {OUT}")