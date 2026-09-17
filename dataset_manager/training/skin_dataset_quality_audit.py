from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

from PIL import Image, ImageStat


# ============================================================
# SCANAI ANIMAL SKIN
# PHASE 3D - AUTOMATED DATASET QUALITY AUDIT
# ============================================================

print("=" * 70)
print("SCANAI ANIMAL SKIN")
print("PHASE 3D - AUTOMATED DATASET QUALITY AUDIT")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

BASELINE_DIR = PROJECT_DIR / "baseline_clean"

OUTPUT_DIR = (
    PROJECT_DIR
    / "training"
    / "outputs"
    / "dataset_quality_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# Images smaller than this are suspicious.
MIN_WIDTH = 160
MIN_HEIGHT = 160

# Extremely small images.
VERY_LOW_WIDTH = 96
VERY_LOW_HEIGHT = 96

# Very dark / nearly blank image threshold.
DARK_MEAN_THRESHOLD = 18.0

# Very bright / nearly blank image threshold.
BRIGHT_MEAN_THRESHOLD = 245.0

# Low contrast threshold.
LOW_CONTRAST_THRESHOLD = 12.0

# Number of words/characters in filename that can indicate
# contextual or disease-name contamination.
SUSPICIOUS_FILENAME_TERMS = [
    "hotspot",
    "hot-spot",
    "ringworm",
    "fungal",
    "fungus",
    "pyoderma",
    "mange",
    "scabies",
    "allergy",
    "allergic",
    "dermatitis",
    "lumpy",
    "lumpy-skin",
    "lumpy_skin",
    "foot-mouth",
    "foot_and_mouth",
    "healthy",
    "disease",
    "symptom",
    "treatment",
    "remedy",
    "medicine",
    "cure",
    "veterinary",
    "vet",
    "explained",
    "how-to",
    "how_to",
    "youtube",
    "facebook",
    "instagram",
    "poster",
    "infographic",
    "advertisement",
    "advert",
    "thumbnail",
]

CONTEXT_FILENAME_TERMS = [
    "herd",
    "crowd",
    "market",
    "farm",
    "farmer",
    "cattle",
    "cow",
    "cows",
    "goat",
    "goats",
    "sheep",
    "animal",
    "animals",
]


# ============================================================
# INFORMATION
# ============================================================

print()
print("Baseline dataset:")
print(BASELINE_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)

if not BASELINE_DIR.exists():
    raise FileNotFoundError(
        f"\nBaseline directory not found:\n{BASELINE_DIR}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def file_hash(path):
    """
    Exact file hash.
    Used only to identify exact duplicate files.
    """

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def get_class_from_path(path):
    """
    Expected structure:

    baseline_clean/
        skin/
            skin__class/
                image.jpg
    """

    try:
        relative = path.relative_to(BASELINE_DIR)

        parts = relative.parts

        if len(parts) >= 2:
            return parts[1]

    except Exception:
        pass

    return ""


def filename_flags(path):
    """
    Detect suspicious words in filenames.

    This is a FLAG only.
    It does NOT automatically mean the image is bad.
    """

    name = path.name.lower()

    suspicious = []
    context = []

    for term in SUSPICIOUS_FILENAME_TERMS:

        if term in name:
            suspicious.append(term)

    for term in CONTEXT_FILENAME_TERMS:

        if term in name:
            context.append(term)

    return suspicious, context


def analyze_image(path):
    """
    Analyze image quality.

    Returns:
        width
        height
        mean_brightness
        contrast
        aspect_ratio
        flags
    """

    flags = []

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            width, height = image.size

            # ------------------------------------------------
            # Basic dimensions
            # ------------------------------------------------

            if (
                width < VERY_LOW_WIDTH
                or height < VERY_LOW_HEIGHT
            ):

                flags.append(
                    "very_low_resolution"
                )

            elif (
                width < MIN_WIDTH
                or height < MIN_HEIGHT
            ):

                flags.append(
                    "low_resolution"
                )

            # ------------------------------------------------
            # Aspect ratio
            # ------------------------------------------------

            if height == 0:

                aspect_ratio = None

            else:

                aspect_ratio = width / height

                if aspect_ratio > 5.0:

                    flags.append(
                        "extreme_aspect_ratio"
                    )

                if aspect_ratio < 0.20:

                    flags.append(
                        "extreme_aspect_ratio"
                    )

            # ------------------------------------------------
            # Brightness / contrast
            # ------------------------------------------------

            stat = ImageStat.Stat(image)

            mean_rgb = stat.mean

            mean_brightness = (
                mean_rgb[0]
                + mean_rgb[1]
                + mean_rgb[2]
            ) / 3.0

            # Convert to grayscale for contrast.
            gray = image.convert("L")

            gray_stat = ImageStat.Stat(gray)

            contrast = gray_stat.stddev[0]

            if mean_brightness <= DARK_MEAN_THRESHOLD:

                flags.append(
                    "very_dark"
                )

            if mean_brightness >= BRIGHT_MEAN_THRESHOLD:

                flags.append(
                    "very_bright"
                )

            if contrast <= LOW_CONTRAST_THRESHOLD:

                flags.append(
                    "very_low_contrast"
                )

            return {
                "width": width,
                "height": height,
                "mean_brightness": round(
                    mean_brightness,
                    3
                ),
                "contrast": round(
                    contrast,
                    3
                ),
                "aspect_ratio": (
                    round(
                        aspect_ratio,
                        3
                    )
                    if aspect_ratio is not None
                    else ""
                ),
                "flags": flags,
                "error": "",
            }

    except Exception as exc:

        return {
            "width": "",
            "height": "",
            "mean_brightness": "",
            "contrast": "",
            "aspect_ratio": "",
            "flags": [
                "unreadable_image"
            ],
            "error": str(exc),
        }


# ============================================================
# FIND IMAGES
# ============================================================

print()
print("=" * 70)
print("SCANNING IMAGE FILES")
print("=" * 70)

image_paths = []

for path in BASELINE_DIR.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    image_paths.append(path)

image_paths.sort()

print()
print(
    f"Images found: {len(image_paths)}"
)


# ============================================================
# AUDIT
# ============================================================

print()
print("=" * 70)
print("RUNNING QUALITY AUDIT")
print("=" * 70)

records = []

hash_map = {}

class_counts = Counter()

flag_counts = Counter()

filename_flag_counts = Counter()

unreadable_count = 0


for number, path in enumerate(
    image_paths,
    start=1
):

    relative_path = path.relative_to(
        BASELINE_DIR
    )

    class_name = get_class_from_path(
        path
    )

    class_counts[class_name] += 1

    # --------------------------------------------------------
    # Filename analysis
    # --------------------------------------------------------

    suspicious_terms, context_terms = (
        filename_flags(path)
    )

    filename_flags_list = []

    if suspicious_terms:

        filename_flags_list.append(
            "suspicious_filename_terms"
        )

        filename_flag_counts[
            "suspicious_filename_terms"
        ] += 1

    if context_terms:

        filename_flags_list.append(
            "context_filename_terms"
        )

        filename_flag_counts[
            "context_filename_terms"
        ] += 1

    # --------------------------------------------------------
    # Image analysis
    # --------------------------------------------------------

    image_info = analyze_image(
        path
    )

    quality_flags = list(
        image_info["flags"]
    )

    for flag in quality_flags:

        flag_counts[flag] += 1

    if "unreadable_image" in quality_flags:

        unreadable_count += 1

    # --------------------------------------------------------
    # Exact duplicate analysis
    # --------------------------------------------------------

    duplicate_of = ""

    try:

        image_hash = file_hash(path)

        if image_hash in hash_map:

            duplicate_of = str(
                hash_map[image_hash]
            )

            if (
                "exact_duplicate"
                not in quality_flags
            ):

                quality_flags.append(
                    "exact_duplicate"
                )

                flag_counts[
                    "exact_duplicate"
                ] += 1

        else:

            hash_map[image_hash] = (
                relative_path
            )

    except Exception:

        image_hash = ""

    # --------------------------------------------------------
    # Overall suspicious status
    # --------------------------------------------------------

    # Hard quality flags determine suspicious status.
all_flags = list(quality_flags)

if duplicate_of:
    all_flags.append("exact_duplicate")

    # Filename terms are metadata clues only.
# They must NOT automatically make an image suspicious.

hard_quality_flags = list(quality_flags)

if duplicate_of:
    hard_quality_flags.append("exact_duplicate")

suspicious = len(hard_quality_flags) > 0

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

# ------------------------------------------------------------
# Priority
# ------------------------------------------------------------

if "unreadable_image" in all_flags:
    score += 100

if "very_dark" in all_flags:
    score += 80

if "very_low_contrast" in all_flags:
    score += 60

if "low_resolution" in all_flags:
    score += 40
    if "unreadable_image" in all_flags:
        score += 100

    if "very_low_resolution" in all_flags:
        score += 80

    if "exact_duplicate" in all_flags:
        score += 70

    if "very_dark" in all_flags:
        score += 50

    if "very_bright" in all_flags:
        score += 50

    if "very_low_contrast" in all_flags:
        score += 40

    if "extreme_aspect_ratio" in all_flags:
        score += 40

    if "suspicious_filename_terms" in all_flags:
        score += 30

    if "context_filename_terms" in all_flags:
        score += 20

    if score >= 80:
        priority = "P0"

    elif score >= 50:
        priority = "P1"

    elif score >= 20:
        priority = "P2"

    elif score > 0:
        priority = "P3"

    else:
        priority = "NONE"

    record = {
        "path": str(path),
        "relative_path": str(
            relative_path
        ),
        "class": class_name,
        "filename": path.name,
        "width": image_info["width"],
        "height": image_info["height"],
        "mean_brightness": image_info[
            "mean_brightness"
        ],
        "contrast": image_info[
            "contrast"
        ],
        "aspect_ratio": image_info[
            "aspect_ratio"
        ],
        "filename_suspicious_terms": (
            "|".join(
                suspicious_terms
            )
        ),
        "filename_context_terms": (
            "|".join(
                context_terms
            )
        ),
        "quality_flags": "|".join(
            quality_flags
        ),
        "duplicate_of": duplicate_of,
        "suspicious": suspicious,
        "priority": priority,
        "audit_score": score,
        "error": image_info["error"],
    }

    records.append(record)

    if (
        number % 500 == 0
        or number == len(image_paths)
    ):

        print(
            f"Processed "
            f"{number}/{len(image_paths)}"
        )


# ============================================================
# WRITE CSV HELPER
# ============================================================

def write_csv(
    output_path,
    rows,
):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# ALL AUDIT RESULTS
# ============================================================

audit_csv = (
    OUTPUT_DIR
    / "quality_audit.csv"
)

write_csv(
    audit_csv,
    records,
)

print()
print("Created:")
print(audit_csv)


# ============================================================
# SUSPICIOUS IMAGES
# ============================================================

suspicious_records = [
    row
    for row in records
    if row["suspicious"]
]

suspicious_records.sort(
    key=lambda row: (
        -int(row["audit_score"]),
        row["relative_path"],
    )
)

suspicious_csv = (
    OUTPUT_DIR
    / "suspicious_images.csv"
)

write_csv(
    suspicious_csv,
    suspicious_records,
)

print()
print("Created:")
print(suspicious_csv)


# ============================================================
# FILENAME FLAGS
# ============================================================

filename_records = [
    row
    for row in records
    if (
        row["filename_suspicious_terms"]
        or row["filename_context_terms"]
    )
]

filename_records.sort(
    key=lambda row: (
        -int(row["audit_score"]),
        row["relative_path"],
    )
)

filename_csv = (
    OUTPUT_DIR
    / "filename_flags.csv"
)

write_csv(
    filename_csv,
    filename_records,
)

print()
print("Created:")
print(filename_csv)


# ============================================================
# IMAGE QUALITY FLAGS
# ============================================================

quality_records = [
    row
    for row in records
    if row["quality_flags"]
]

quality_records.sort(
    key=lambda row: (
        -int(row["audit_score"]),
        row["relative_path"],
    )
)

quality_csv = (
    OUTPUT_DIR
    / "image_quality_flags.csv"
)

write_csv(
    quality_csv,
    quality_records,
)

print()
print("Created:")
print(quality_csv)


# ============================================================
# EXACT DUPLICATES
# ============================================================

duplicate_records = [
    row
    for row in records
    if "exact_duplicate"
    in row["quality_flags"]
]

duplicate_records.sort(
    key=lambda row: row[
        "relative_path"
    ]
)

duplicate_csv = (
    OUTPUT_DIR
    / "duplicate_candidates.csv"
)

write_csv(
    duplicate_csv,
    duplicate_records,
)

print()
print("Created:")
print(duplicate_csv)


# ============================================================
# REVIEW QUEUE
# ============================================================

review_queue = [
    row
    for row in records
    if row["priority"] != "NONE"
]

review_queue.sort(
    key=lambda row: (
        {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        .get(row["priority"], 9),
        -int(row["audit_score"]),
        row["relative_path"],
    )
)

review_queue_csv = (
    OUTPUT_DIR
    / "REVIEW_QUEUE.csv"
)

write_csv(
    review_queue_csv,
    review_queue,
)

print()
print("Created:")
print(review_queue_csv)


# ============================================================
# SUMMARY
# ============================================================

priority_counts = Counter(
    row["priority"]
    for row in records
)

class_suspicious_counts = Counter(
    row["class"]
    for row in suspicious_records
)


summary = {
    "dataset": str(
        BASELINE_DIR
    ),
    "total_images": len(records),
    "suspicious_images": len(
        suspicious_records
    ),
    "clean_by_automated_checks": (
        len(records)
        - len(suspicious_records)
    ),
    "unreadable_images": unreadable_count,
    "priority_counts": dict(
        priority_counts
    ),
    "quality_flag_counts": dict(
        flag_counts
    ),
    "filename_flag_counts": dict(
        filename_flag_counts
    ),
    "suspicious_images_by_class": dict(
        class_suspicious_counts
    ),
    "class_counts": dict(
        class_counts
    ),
    "exact_duplicate_candidates": len(
        duplicate_records
    ),
    "important_note": (
        "Automated flags are candidates only. "
        "No images were deleted or modified."
    ),
}


summary_json = (
    OUTPUT_DIR
    / "audit_summary.json"
)

with open(
    summary_json,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("PHASE 3D COMPLETE")
print("=" * 70)

print()
print(
    f"Total images              : "
    f"{len(records)}"
)

print(
    f"Suspicious candidates     : "
    f"{len(suspicious_records)}"
)

print(
    f"No automated flags        : "
    f"{len(records) - len(suspicious_records)}"
)

print()
print("Priority:")

print(
    f"  P0: {priority_counts.get('P0', 0)}"
)

print(
    f"  P1: {priority_counts.get('P1', 0)}"
)

print(
    f"  P2: {priority_counts.get('P2', 0)}"
)

print(
    f"  P3: {priority_counts.get('P3', 0)}"
)

print()
print("Quality flags:")

for flag, count in sorted(
    flag_counts.items(),
    key=lambda x: -x[1],
):

    print(
        f"  {flag}: {count}"
    )

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Generated:")

print(
    "  quality_audit.csv"
)

print(
    "  suspicious_images.csv"
)

print(
    "  filename_flags.csv"
)

print(
    "  image_quality_flags.csv"
)

print(
    "  duplicate_candidates.csv"
)

print(
    "  REVIEW_QUEUE.csv"
)

print(
    "  audit_summary.json"
)

print()
print("=" * 70)
print("SAFETY")
print("=" * 70)

print()
print("NO IMAGES WERE DELETED.")
print("NO IMAGES WERE MODIFIED.")
print("baseline_clean/ was NOT modified.")
print("validation/test were NOT modified.")
print("Only audit reports were created.")

print()