import pandas as pd
from pathlib import Path

manifest = pd.read_csv("metadata/cleanup_manifest.csv")

clean = Path("clean")
backup = Path("metadata/cleanup_backup")

found_backup = 0
still_clean = 0
missing = 0

for p in manifest["path"]:
    p = Path(p)

    try:
        rel = p.relative_to(clean)
    except ValueError:
        text = str(p).replace("\\", "/")
        if text.startswith("clean/"):
            rel = Path(text[6:])
        else:
            print("Unexpected path:", p)
            missing += 1
            continue

    clean_path = clean / rel
    backup_path = backup / rel

    if backup_path.exists():
        found_backup += 1
    elif clean_path.exists():
        still_clean += 1
    else:
        missing += 1

print("=" * 50)
print("CLEANUP STATE")
print("=" * 50)
print("Manifest entries :", len(manifest))
print("Found in backup  :", found_backup)
print("Still in clean   :", still_clean)
print("Missing entirely :", missing)
print("=" * 50)