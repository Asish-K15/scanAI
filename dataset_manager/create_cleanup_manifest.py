import pandas as pd
from pathlib import Path
from collections import defaultdict

DUP_FILE = Path("metadata/duplicates.csv")
OUT_FILE = Path("metadata/cleanup_manifest.csv")
CLEAN_DIR = Path("clean")


# ============================================================
# PATH NORMALIZATION
# ============================================================

def normalize_path(path):
    """
    Convert every image path to a single clean/<relative> form.
    This prevents absolute + relative representations of the
    same physical file from appearing as separate manifest rows.
    """
    p = Path(str(path))

    try:
        rel = p.relative_to(CLEAN_DIR)
    except ValueError:
        text = str(p).replace("\\", "/")

        marker = "clean/"

        if marker in text:
            rel = Path(text.split(marker, 1)[1])
        else:
            raise ValueError(f"Cannot normalize path: {path}")

    return (CLEAN_DIR / rel).resolve()


# ============================================================
# LOAD DUPLICATE DATA
# ============================================================

d = pd.read_csv(DUP_FILE)

# Normalize duplicate-pair paths
d["image_a"] = d["image_a"].map(normalize_path)
d["image_b"] = d["image_b"].map(normalize_path)


# ============================================================
# UNION-FIND
# ============================================================

parent = {}


def find(x):
    if x not in parent:
        parent[x] = x

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):
    ra = find(a)
    rb = find(b)

    if ra != rb:
        parent[rb] = ra


for a, b in zip(d["image_a"], d["image_b"]):
    union(a, b)


# ============================================================
# BUILD CONNECTED COMPONENTS
# ============================================================

clusters = defaultdict(list)

for image in parent:
    clusters[find(image)].append(image)


# ============================================================
# GENERATE REMOVAL CANDIDATES
# ============================================================

removals = []

same_class_removed = 0
exact_cross_removed = 0


for images in clusters.values():

    images = sorted(set(images))

    classes = {
        image.parent.name
        for image in images
    }

    # --------------------------------------------------------
    # SAME-CLASS CLUSTER
    # --------------------------------------------------------

    if len(classes) == 1:

        # Keep deterministic first representative
        for image in images[1:]:

            removals.append({
                "path": str(image),
                "reason": "same_class_duplicate",
                "cluster_size": len(images),
                "classes": "|".join(sorted(classes)),
            })

            same_class_removed += 1

    # --------------------------------------------------------
    # CROSS-CLASS CLUSTER
    # --------------------------------------------------------

    else:

        cluster_set = set(images)

        cluster_pairs = d[
            d["image_a"].isin(cluster_set)
            & d["image_b"].isin(cluster_set)
        ]

        has_exact_conflict = (
            cluster_pairs["hamming_distance"] == 0
        ).any()

        # Remove exact cross-class clusters.
        # Preserve near-only cross-class clusters.
        if has_exact_conflict:

            for image in images:

                removals.append({
                    "path": str(image),
                    "reason": "exact_cross_class_conflict",
                    "cluster_size": len(images),
                    "classes": "|".join(sorted(classes)),
                })

                exact_cross_removed += 1


# ============================================================
# UNMAPPED FILES
# ============================================================

for image in CLEAN_DIR.joinpath("_UNMAPPED").rglob("*"):

    if image.is_file():

        image = image.resolve()

        removals.append({
            "path": str(image),
            "reason": "unmapped",
            "cluster_size": 1,
            "classes": "_UNMAPPED",
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

manifest = pd.DataFrame(removals)


# ============================================================
# CRITICAL SAFETY CHECK
# ============================================================

if manifest.empty:
    raise RuntimeError("Cleanup manifest is empty.")


# Normalize one final time
manifest["path"] = manifest["path"].map(normalize_path)

# Remove duplicate physical paths
before = len(manifest)

manifest = (
    manifest
    .drop_duplicates(subset=["path"], keep="first")
    .reset_index(drop=True)
)

duplicates_removed = before - len(manifest)


# ============================================================
# VERIFY ALL FILES EXIST
# ============================================================

missing = [
    str(p)
    for p in manifest["path"]
    if not p.exists()
]

if missing:

    print("ERROR: Manifest contains missing files:")
    for p in missing[:20]:
        print(p)

    raise RuntimeError(
        f"{len(missing)} manifest files do not exist. "
        "Nothing was written."
    )


# ============================================================
# SAFETY: NEVER REMOVE THE SAME IMAGE TWICE
# ============================================================

if manifest["path"].duplicated().any():

    raise RuntimeError(
        "Duplicate paths remain in cleanup manifest."
    )


# ============================================================
# SAVE
# ============================================================

manifest.to_csv(OUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

same_count = (
    manifest["reason"] == "same_class_duplicate"
).sum()

cross_count = (
    manifest["reason"] == "exact_cross_class_conflict"
).sum()

unmapped_count = (
    manifest["reason"] == "unmapped"
).sum()


print()
print("=" * 60)
print("CLEANUP MANIFEST GENERATED")
print("=" * 60)

print(f"Duplicate manifest entries removed: {duplicates_removed}")

print()
print(f"Same-class removals:       {same_count}")
print(f"Exact cross-class removal: {cross_count}")
print(f"Unmapped removals:         {unmapped_count}")

print("-" * 60)

print(f"TOTAL FILES TO REMOVE:     {len(manifest)}")

print("=" * 60)

print()
print(f"Manifest: {OUT_FILE}")

print()
print("Safety checks:")
print(f"Unique paths:              {manifest.path.nunique()}")
print(f"Missing files:             {len(missing)}")

print()
print("IMPORTANT:")
print("No files were deleted.")
print("Near-only cross-class clusters were retained.")