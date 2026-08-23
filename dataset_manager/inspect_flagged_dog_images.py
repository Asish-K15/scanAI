import pandas as pd
import cv2
from pathlib import Path

report = pd.read_csv("metadata/dog_conjunctivitis_quality.csv")

flagged = report[report["status"] == "flagged"]

rows = []

for path in flagged["path"]:

    p = Path(path)
    img = cv2.imread(str(p))

    if img is None:
        continue

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    rows.append({
        "path": str(p),
        "width": w,
        "height": h,
        "min_dimension": min(w, h),
        "blur_variance": round(variance, 2)
    })

df = pd.DataFrame(rows)

print()
print("=" * 70)
print("FLAGGED DOG CONJUNCTIVITIS ANALYSIS")
print("=" * 70)

print()
print("Dimension statistics:")
print(df["min_dimension"].describe())

print()
print("Blur variance statistics:")
print(df["blur_variance"].describe())

print()
print("Images below 100 px:")
print((df["min_dimension"] < 100).sum())

print()
print("Images below blur threshold (30):")
print((df["blur_variance"] < 30).sum())

print()
print("Sample:")
print(df.sort_values("blur_variance").head(30).to_string(index=False))

df.to_csv(
    "metadata/dog_conjunctivitis_flagged_analysis.csv",
    index=False
)

print()
print("Report saved:")
print("metadata/dog_conjunctivitis_flagged_analysis.csv")