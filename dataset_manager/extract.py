"""
Stage 2: Extract
-----------------
Walks downloads/, unzips anything not already extracted, and copies the
resulting image folders into raw/<source>__<dataset_name>/ so every dataset
sits at a consistent path before taxonomy mapping.
"""

import shutil
import zipfile
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).parent / "downloads"
RAW_DIR = Path(__file__).parent / "raw"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_zips(root: Path):
    for zip_path in root.rglob("*.zip"):
        target = zip_path.parent / zip_path.stem
        if target.exists():
            continue
        print(f"  extracting {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)
        except zipfile.BadZipFile:
            print(f"  !! bad zip, skipping: {zip_path}")


def copy_to_raw(dataset_dir: Path, dest_name: str):
    dest = RAW_DIR / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in dataset_dir.rglob("*"):
        if img.suffix.lower() in IMAGE_EXTS:
            # preserve the immediate parent folder name as the raw class label
            class_name = img.parent.name
            class_dir = dest / class_name
            class_dir.mkdir(exist_ok=True)
            shutil.copy2(img, class_dir / img.name)
            count += 1
    print(f"  -> {dest_name}: {count} images copied to raw/")


def main():
    RAW_DIR.mkdir(exist_ok=True)
    if not DOWNLOAD_DIR.exists():
        print("No downloads/ folder found — run download.py first.")
        return

    print("Extracting archives...")
    extract_zips(DOWNLOAD_DIR)

    print("Copying to raw/ ...")
    for source_dir in DOWNLOAD_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        for dataset_dir in source_dir.iterdir():
            if dataset_dir.is_dir():
                dest_name = f"{source_dir.name}__{dataset_dir.name}"
                copy_to_raw(dataset_dir, dest_name)

    print("\nDone. raw/ now contains one folder per dataset, each with class subfolders.")


if __name__ == "__main__":
    main()
