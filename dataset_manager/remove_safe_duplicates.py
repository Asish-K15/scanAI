import pandas as pd
from pathlib import Path

csv_path = Path("metadata/safe_duplicate_removals.csv")
df = pd.read_csv(csv_path)

removed = 0
missing = 0

for image_path in df["image"]:
    path = Path(image_path)

    if path.exists():
        path.unlink()
        removed += 1
    else:
        missing += 1

print(f"Removed: {removed}")
print(f"Missing: {missing}")