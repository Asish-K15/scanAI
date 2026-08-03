"""
Stage 3: Taxonomy Mapper
-------------------------
Reads every raw/<dataset>/<raw_class>/ folder, matches the raw_class name
against taxonomy.json aliases, and copies images into
clean/<unified_class>/ using the unified name.

Anything that doesn't match any alias goes into clean/_UNMAPPED/<raw_name>/
so you can review it and add a new alias to taxonomy.json rather than
silently losing data.

Run this AFTER extract.py and BEFORE dedup.py.
"""

import json
import re
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
CLEAN_DIR = Path(__file__).parent / "clean"
TAXONOMY_PATH = Path(__file__).parent / "taxonomy.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(name: str) -> str:
    """Lowercase, strip punctuation/underscores/spaces for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_alias_lookup(taxonomy: dict) -> dict:
    """Flattens taxonomy.json into {normalized_alias: 'group/unified_class'}"""
    lookup = {}
    for group, classes in taxonomy.items():
        if group.startswith("_") or group == "excluded_not_photo_diagnosable":
            continue
        if not isinstance(classes, dict):
            continue
        for unified_class, aliases in classes.items():
            for alias in aliases:
                lookup[normalize(alias)] = f"{group}/{unified_class}"
    return lookup


def main():
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    lookup = build_alias_lookup(taxonomy)
    excluded = {normalize(x) for x in taxonomy.get("excluded_not_photo_diagnosable", [])}

    CLEAN_DIR.mkdir(exist_ok=True)
    unmapped_log = []
    excluded_log = []
    mapped_count = 0

    for dataset_dir in RAW_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir():
                continue

            norm = normalize(class_dir.name)

            if norm in excluded:
                excluded_log.append((dataset_dir.name, class_dir.name))
                continue

            unified = lookup.get(norm)
            if unified is None:
                # try partial match as a fallback
                for alias_norm, target in lookup.items():
                    if alias_norm in norm or norm in alias_norm:
                        unified = target
                        break

            if unified is None:
                unmapped_log.append((dataset_dir.name, class_dir.name))
                dest = CLEAN_DIR / "_UNMAPPED" / class_dir.name
            else:
                dest = CLEAN_DIR / unified

            dest.mkdir(parents=True, exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    # prefix filename with source dataset to avoid collisions
                    new_name = f"{dataset_dir.name}__{img.name}"
                    shutil.copy2(img, dest / new_name)
                    mapped_count += 1

    print(f"Mapped {mapped_count} images into clean/")
    if excluded_log:
        print(f"\nExcluded (not photo-diagnosable) — {len(excluded_log)} class folders skipped:")
        for ds, cls in excluded_log:
            print(f"  {ds} / {cls}")
    if unmapped_log:
        print(f"\n!! {len(unmapped_log)} class folders had no taxonomy match — check clean/_UNMAPPED/")
        print("   Add missing aliases to taxonomy.json and re-run this script.")
        for ds, cls in unmapped_log:
            print(f"  {ds} / {cls}")


if __name__ == "__main__":
    main()
