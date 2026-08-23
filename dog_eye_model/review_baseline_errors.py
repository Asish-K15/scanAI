from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent

CSV = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "efficientnet_b0_errors.csv"
)

OUT = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
    / "baseline_error_review"
)

OUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(CSV)


try:
    font = ImageFont.truetype("arial.ttf", 22)
    small_font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()


for i, row in df.iterrows():

    image_path = Path(row["path"])

    if not image_path.is_absolute():
        image_path = (
            PROJECT_DIR.parent
            / "dataset_manager"
            / image_path
        )

    try:
        image = Image.open(image_path).convert("RGB")

        image.thumbnail((700, 550))

        canvas = Image.new(
            "RGB",
            (760, 700),
            "white"
        )

        x = (760 - image.width) // 2

        canvas.paste(
            image,
            (x, 20)
        )

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (20, 590),
            f"ERROR {i + 1}",
            fill="black",
            font=font
        )

        draw.text(
            (20, 625),
            f"TRUE: {row['true_class']}    "
            f"PREDICTED: {row['predicted_class']}",
            fill="black",
            font=small_font
        )

        draw.text(
            (20, 650),
            f"Conjunctivitis: {row['prob_conjunctivitis']:.3f}    "
            f"Entropion: {row['prob_entropion']:.3f}",
            fill="black",
            font=small_font
        )

        output = (
            OUT
            / f"error_{i + 1:02d}.jpg"
        )

        canvas.save(
            output,
            quality=95
        )

    except Exception as e:
        print("ERROR:", image_path)
        print(e)


print("=" * 60)
print("BASELINE ERROR REVIEW")
print("=" * 60)
print()
print("Images:", len(df))
print()
print("Saved to:")
print(OUT)