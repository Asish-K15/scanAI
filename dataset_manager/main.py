"""
Runs the full Dataset Manager pipeline in order:
  1. extract   - unzip downloads/, copy into raw/
  2. map       - taxonomy_mapper.py, raw/ -> clean/
  3. validate  - flag corrupt/blurry images (removes corrupt only)
  4. dedup     - find and report duplicates (does NOT auto-remove)
  5. stats     - dataset_inventory.csv

Note: download.py is run separately since it needs your API keys configured
first (see download.py docstring). Run it manually before this script.

Usage:
  python main.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script_name: str, args=None):
    args = args or []
    print(f"\n{'='*60}\nRunning {script_name}\n{'='*60}")
    result = subprocess.run([sys.executable, str(HERE / script_name)] + args)
    if result.returncode != 0:
        print(f"!! {script_name} exited with an error — check output above before continuing.")
        sys.exit(1)


def main():
    if not (HERE / "downloads").exists() or not any((HERE / "downloads").iterdir()):
        print("downloads/ is empty. Run download.py first (after setting up your API keys).")
        sys.exit(1)

    run("extract.py")
    run("taxonomy_mapper.py")
    run("validate_images.py", ["--remove-corrupt"])
    run("dedup.py")  # review metadata/duplicates.csv, then run dedup.py --remove manually
    run("split_dataset.py")  # 70/15/15 stratified, duplicate-cluster-aware
    run("stats.py")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("Next steps:")
    print("  1. Review clean/_UNMAPPED/ — add missing aliases to taxonomy.json and re-run taxonomy_mapper.py")
    print("  2. Review metadata/duplicates.csv, then run: python dedup.py --remove")
    print("  3. Review metadata/dataset_inventory.csv for weak classes needing self-collection")
    print("  4. splits/train, splits/val, splits/test are now your training-ready dataset")
    print("     (leakage-safe: near-duplicate clusters never span multiple splits)")
    print("=" * 60)


if __name__ == "__main__":
    main()
