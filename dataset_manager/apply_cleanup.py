import pandas as pd
from pathlib import Path
import shutil

MANIFEST = Path("metadata/cleanup_manifest.csv")
CLEAN = Path("clean")
BACKUP = Path("metadata/cleanup_backup")

df = pd.read_csv(MANIFEST)

if len(df) != df["path"].nunique():
    raise RuntimeError("Duplicate paths in manifest.")

BACKUP.mkdir(parents=True, exist_ok=True)

moved = 0
missing = 0

for raw_path in df["path"]:
    src = Path(raw_path).resolve()

    if not src.exists():
        missing += 1
        continue

    rel = src.relative_to(CLEAN.resolve())
    dst = BACKUP / rel

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise RuntimeError(f"Backup already exists: {dst}")

    shutil.move(str(src), str(dst))
    moved += 1

print("=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)
print(f"Expected: {len(df)}")
print(f"Moved:    {moved}")
print(f"Missing:  {missing}")
print("=" * 60)

if moved != len(df) or missing != 0:
    raise RuntimeError("Cleanup count mismatch!")

print("SUCCESS: Manifest applied exactly.")