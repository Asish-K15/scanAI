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
    "devang03mgr/cattle-diseases-datasets",
    "smadive/pet-disease-images",
]
ROBOFLOW_DATASETS = [
    #Skin Disease
    {
        "workspace": "kendys-workspace",
        "project": "dog-and-cat-skin-disease-identification-2",
        "version": 2
    }
    #Cattle Disease
    {
        
        "workspace": "sliit-workspace",
        "project": "cattle-diseases",
        "version": 1
    }
    #Lumpy skin disease
    {
        "workspace": "qq-mgfrz",
        "project": "lumpy-skin-disease",
        "version": 1
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
    # IMPORTANT: do NOT pre-create out_dir here. Some Roboflow SDK versions treat
    # an already-existing target folder as "already downloaded" and silently skip
    # the actual file transfer. Let the SDK create the folder itself.
    print(f"[roboflow] downloading {workspace}/{project} v{version} -> {out_dir}")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    try:
        dataset = ver.download(fmt, location=str(out_dir), overwrite=True)
    except TypeError:
        # older roboflow SDK versions don't accept the overwrite= kwarg
        dataset = ver.download(fmt, location=str(out_dir))

    # Verify something actually landed on disk instead of trusting the returned object
    image_exts = {".jpg", ".jpeg", ".png"}
    found_images = [f for f in Path(dataset.location).rglob("*") if f.suffix.lower() in image_exts]

    if len(found_images) == 0:
        print(f"  !! WARNING: 0 images found at {dataset.location} after download.")
        print(f"     dataset object: {vars(dataset)}")
        print(f"     Manual fallback: open https://universe.roboflow.com/{workspace}/{project}")
        print(f"     click 'Download Dataset', choose the Folder Structure / classification export,")
        print(f"     and extract the zip into: {out_dir}")
    else:
        print(f"  -> {len(found_images)} images downloaded to {dataset.location}")
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
