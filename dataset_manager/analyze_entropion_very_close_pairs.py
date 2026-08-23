from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

CSV = Path("metadata/expansion_checks/dog_entropion_internal_duplicates.csv")
OUT = Path("metadata/expansion_checks/dog_entropion_very_close_analysis.csv")

df = pd.read_csv(CSV)
df = df[df["category"] == "very_close"].copy()

seen = set()
rows = []

for _, r in df.iterrows():

    a = r["image"]
    b = r["best_match"]

    key = tuple(sorted([a, b]))

    if key in seen:
        continue

    seen.add(key)

    try:
        img1 = Image.open(a).convert("L").resize((256, 256))
        img2 = Image.open(b).convert("L").resize((256, 256))

        arr1 = np.asarray(img1, dtype=np.float32)
        arr2 = np.asarray(img2, dtype=np.float32)

        # Normalized mean absolute pixel difference
        mad = np.mean(np.abs(arr1 - arr2)) / 255.0

        # Structural similarity
        similarity = ssim(
            arr1,
            arr2,
            data_range=255
        )

        rows.append({
            "image": a,
            "best_match": b,
            "hamming_distance": int(r["hamming_distance"]),
            "pixel_difference": round(float(mad), 4),
            "ssim": round(float(similarity), 4)
        })

    except Exception as e:
        print("ERROR:", a, b, e)

out = pd.DataFrame(rows)

out = out.sort_values(
    ["ssim", "pixel_difference"],
    ascending=[False, True]
)

out.to_csv(OUT, index=False)

print("=" * 60)
print("VERY-CLOSE DUPLICATE SECONDARY ANALYSIS")
print("=" * 60)

print("Pairs analyzed:", len(out))
print()

print(out.to_string(index=False))

print()
print("Saved:", OUT)
