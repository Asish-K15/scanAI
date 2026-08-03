"""
Stage: Stratified Split with Leakage Prevention
--------------------------------------------------
Splits clean/ into splits/train, splits/val, splits/test (default 70/15/15,
stratified per class) while guaranteeing that near-duplicate images
(as found by dedup.py) never end up split across train/val/test.

Why this matters: if the same source photo (or a near-duplicate of it)
appears in both train and test, your reported test accuracy is
inflated and meaningless. This groups near-duplicates into clusters
using Union-Find, then splits at the CLUSTER level, not the image level.

Run AFTER dedup.py has produced metadata/duplicates.csv (even if you
haven't removed duplicates yet — this script uses that file to keep
near-duplicate clusters together regardless).

Usage:
  python split_dataset.py                  # default 70/15/15
  python split_dataset.py --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

CLEAN_DIR = Path(__file__).parent / "clean"
SPLITS_DIR = Path(__file__).parent / "splits"
METADATA_DIR = Path(__file__).parent / "metadata"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


class UnionFind:
    """Groups near-duplicate images into clusters so a cluster is never split
    across train/val/test."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def clusters(self):
        groups = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return list(groups.values())


def load_duplicate_pairs():
    dup_csv = METADATA_DIR / "duplicates.csv"
    pairs = []
    if dup_csv.exists():
        with open(dup_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["image_a"], row["image_b"]))
    return pairs


def split_class(image_paths, duplicate_pairs, ratios, rng):
    """Returns {split_name: [image_paths]} for one class, clustering
    near-duplicates first so they land in the same split."""
    image_set = set(str(p) for p in image_paths)

    uf = UnionFind(image_set)
    for a, b in duplicate_pairs:
        if a in image_set and b in image_set:
            uf.union(a, b)

    clusters = uf.clusters()
    rng.shuffle(clusters)

    n_total = sum(len(c) for c in clusters)
    train_target = n_total * ratios["train"]
    val_target = n_total * ratios["val"]

    train, val, test = [], [], []
    running = 0
    for cluster in clusters:
        if running < train_target:
            train.extend(cluster)
        elif running < train_target + val_target:
            val.extend(cluster)
        else:
            test.extend(cluster)
        running += len(cluster)

    return {"train": train, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    ratios = {"train": args.train, "val": args.val, "test": args.test}
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "train+val+test ratios must sum to 1.0"

    rng = random.Random(SEED)
    duplicate_pairs = load_duplicate_pairs()

    for split in ("train", "val", "test"):
        (SPLITS_DIR / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    class_summary = []

    for group_dir in sorted(CLEAN_DIR.iterdir()):
        if not group_dir.is_dir() or group_dir.name == "_UNMAPPED":
            continue
        for class_dir in sorted(group_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            unified_class = f"{group_dir.name}__{class_dir.name}"
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            if not images:
                continue

            split_result = split_class(images, duplicate_pairs, ratios, rng)

            for split_name, split_images in split_result.items():
                dest_dir = SPLITS_DIR / split_name / unified_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img_path_str in split_images:
                    img_path = Path(img_path_str)
                    if img_path.exists():
                        shutil.copy2(img_path, dest_dir / img_path.name)
                        manifest_rows.append([str(img_path), unified_class, split_name])

            class_summary.append([
                unified_class,
                len(split_result["train"]),
                len(split_result["val"]),
                len(split_result["test"]),
            ])

    METADATA_DIR.mkdir(exist_ok=True)
    manifest_path = METADATA_DIR / "split_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_path", "class", "split"])
        writer.writerows(manifest_rows)

    summary_path = METADATA_DIR / "split_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "train", "val", "test"])
        writer.writerows(class_summary)

    print(f"Split complete. {len(manifest_rows)} images assigned across train/val/test.")
    print(f"Manifest: {manifest_path}")
    print(f"Per-class summary: {summary_path}")
    print("\nsplits/ is now ready for train_baseline.ipynb")


if __name__ == "__main__":
    main()
