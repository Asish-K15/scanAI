"""
Stage 1: Download
------------------
Pulls datasets from Kaggle and Roboflow into downloads/<source>/<dataset_name>/

Setup required before running:
  1. Kaggle: pla    ce your kaggle.json API token at ~/.kaggle/kaggle.json
     (get it from kaggle.com -> Account -> Create New API Token)
  2. Roboflow: set the ROBOFLOW_API_KEY environment variable
     (get it from your Roboflow workspace settings)

Usage:
  python download.py
  (edit the KAGGLE_DATASETS and ROBOFLOW_DATASETS lists below first)

Note: some Kaggle dataset pages block automated browsing but the Kaggle API
still works fine for actual downloads once your token is set up.
"""

import os
import subprocess
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).parent / "downloads"

# ---- Fill these in with the datasets you've actually vetted (class list,
# license, sample-checked) using the checklist from your dataset research ----

KAGGLE_DATASETS = [
    # "owner/dataset-slug",
    # e.g. "atharvaingle/crop-recommendation-dataset"
]
ROBOFLOW_DATASETS = [
    {
        "workspace": "kendys-workspace",
        "project": "dog-and-cat-skin-disease-identification-2",
        "version": 2
    }
]


def download_kaggle(dataset_slug: str):
    """Downloads and unzips a Kaggle dataset into downloads/kaggle/<slug>/"""
    out_dir = DOWNLOAD_DIR / "kaggle" / dataset_slug.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {dataset_slug} -> {out_dir}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(out_dir), "--unzip"],
        check=True,
    )


def download_roboflow(workspace: str, project: str, version: int, fmt: str = "folder"):
    """Downloads a Roboflow Universe dataset into downloads/roboflow/<project>/"""
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the ROBOFLOW_API_KEY environment variable first.")

    out_dir = DOWNLOAD_DIR / "roboflow" / project
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[roboflow] downloading {workspace}/{project} v{version} -> {out_dir}")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)

    ver = proj.version(version)

    print("\n=== VERSION INFO ===")
    print(ver)
    print("====================\n")

    dataset = ver.download(fmt, location=str(out_dir))

    print("\n=== DOWNLOAD RESULT ===")
    print(dataset)
    print("=======================\n")

    print("\n===== DOWNLOAD RESULT =====")
    print("\n=== DATASET INFO ===")
    print("Location:", dataset.location)
    print("Name:", dataset.name)
    print("====================")
    print("===========================\n")

    return dataset
def main():
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    for slug in KAGGLE_DATASETS:
        try:
            download_kaggle(slug)
        except Exception as e:
            print(f"  !! failed: {slug} -> {e}")

    for ds in ROBOFLOW_DATASETS:
        try:
            download_roboflow(ds["workspace"], ds["project"], ds["version"])
        except Exception as e:
            print(f"  !! failed: {ds['project']} -> {e}")

    print("\nDone. Check downloads/ for results, then run extract.py if any zips remain.")


if __name__ == "__main__":
    main()
