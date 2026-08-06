"""
Stage 1: Download
------------------
Pulls datasets from Kaggle and Roboflow into downloads/<source>/<dataset_name>/

Datasets are read from docs/dataset_sources.csv — add a new row there to
download a new dataset, no code changes needed.

CSV columns:
  dataset_name, source, url, license, status, task_type, version, format,
  species, num_images, num_classes, notes

  - source:   "Kaggle" or "Roboflow"
  - url:      the dataset page URL (workspace/project or owner/slug is parsed out)
  - status:   "active" = download and use in pipeline; "pending" = candidate
              dataset; "rejected" = not suitable; "archived" = previously
              used but no longer active. Only "active" rows are downloaded.
  - task_type: "classification" or "object_detection" (used later in the pipeline)
  - version:  (Roboflow only) the dataset version number
  - format:   (Roboflow only) export format, e.g. "folder" or "yolov8"

Only rows with status "active" are processed. Adding a dataset only requires
a new row in the CSV — no Python changes. Datasets that already exist on disk
are skipped so re-runs are fast.

Setup required before running:
  1. Kaggle: place your kaggle.json API token at ~/.kaggle/kaggle.json
     (get it from kaggle.com -> Account -> Create New API Token)
  2. Roboflow: set the ROBOFLOW_API_KEY environment variable
     (get it from your Roboflow workspace settings)

Usage:
  python download.py

Note: some Kaggle dataset pages block automated browsing but the Kaggle API
still works fine for actual downloads once your token is set up.
"""

import csv
import os
import subprocess
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).parent / "downloads"
SOURCES_CSV = Path(__file__).parent.parent / "docs" / "dataset_sources.csv"

# Only datasets explicitly approved for the pipeline are downloaded.
#   active   = download and use in pipeline
#   pending  = candidate dataset (not yet approved)
#   rejected = not suitable
#   archived = previously used, no longer active
ACTIVE_STATUSES = {"active"}


def load_sources(csv_path: Path) -> list:
    """Reads dataset sources from the CSV, returning a list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_roboflow_url(url: str):
    """Extracts (workspace, project) from a Roboflow Universe URL."""
    # e.g. https://universe.roboflow.com/<workspace>/<project>
    parts = [p for p in url.rstrip("/").split("/") if p]
    # find the "roboflow.com" host segment, then take the next two parts
    try:
        idx = next(i for i, p in enumerate(parts) if "roboflow.com" in p)
        workspace = parts[idx + 1]
        project = parts[idx + 2]
    except (StopIteration, IndexError):
        raise ValueError(f"Could not parse Roboflow URL: {url}")
    return workspace, project


def parse_kaggle_slug(url: str) -> str:
    """Extracts 'owner/dataset-slug' from a Kaggle dataset URL."""
    # e.g. https://www.kaggle.com/datasets/<owner>/<slug>
    parts = [p for p in url.rstrip("/").split("/") if p]
    try:
        idx = next(i for i, p in enumerate(parts) if p == "datasets")
        return f"{parts[idx + 1]}/{parts[idx + 2]}"
    except (StopIteration, IndexError):
        raise ValueError(f"Could not parse Kaggle URL: {url}")


def already_downloaded(out_dir: Path) -> bool:
    """True if the target folder exists and contains at least one file."""
    return out_dir.exists() and any(out_dir.iterdir())


def download_kaggle(dataset_slug: str):
    """Downloads and unzips a Kaggle dataset into downloads/kaggle/<slug>/"""
    out_dir = DOWNLOAD_DIR / "kaggle" / dataset_slug.replace("/", "__")
    if already_downloaded(out_dir):
        print(f"[kaggle] already downloaded: {dataset_slug} -> {out_dir}")
        return
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
    if already_downloaded(out_dir):
        print(f"[roboflow] already downloaded: {workspace}/{project} -> {out_dir}")
        return
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

    sources = load_sources(SOURCES_CSV)
    if not sources:
        print(f"No dataset sources found in {SOURCES_CSV}")
        return

    for row in sources:
        name = row.get("dataset_name", "").strip()
        source = row.get("source", "").strip().lower()
        url = row.get("url", "").strip()
        status = row.get("status", "").strip().lower()

        if not name or not source or not url:
            print(f"  !! skipped row with missing name/source/url: {row}")
            continue

        # Only download datasets explicitly approved for the pipeline
        if status not in ACTIVE_STATUSES:
            print(f"  -- skipped {name} (status='{status}')")
            continue

        try:
            if source == "kaggle":
                slug = parse_kaggle_slug(url)
                download_kaggle(slug)
            elif source == "roboflow":
                workspace, project = parse_roboflow_url(url)
                version = int(row.get("version") or 1)
                fmt = (row.get("format") or "folder").strip()
                download_roboflow(workspace, project, version, fmt)
            else:
                print(f"  !! unknown source '{source}' for {name} — skipping")
        except Exception as e:
            print(f"  !! failed: {name} -> {e}")

    print("\nDone. Check downloads/ for results, then run extract.py if any zips remain.")


if __name__ == "__main__":
    main()