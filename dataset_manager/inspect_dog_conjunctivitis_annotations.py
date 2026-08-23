from pathlib import Path
import json
from collections import Counter, defaultdict

ZIP_JSONS = {
    "train": Path("downloads"),
}

# We will locate the original annotation files automatically
ROOT = Path(".")

annotation_files = list(
    ROOT.rglob("_annotations.coco.json")
)

print("=" * 60)
print("DOG CONJUNCTIVITIS ANNOTATION ANALYSIS")
print("=" * 60)

if not annotation_files:
    print("No _annotations.coco.json files found.")
    raise SystemExit

for path in annotation_files:

    print()
    print("=" * 60)
    print(path)
    print("=" * 60)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {
        c["id"]: c["name"]
        for c in data["categories"]
    }

    image_names = {
        img["id"]: img["file_name"]
        for img in data["images"]
    }

    image_classes = defaultdict(set)

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]

        image_classes[image_id].add(
            categories.get(category_id, str(category_id))
        )

    class_counts = Counter()

    for classes in image_classes.values():
        for c in classes:
            class_counts[c] += 1

    print("Images:", len(data["images"]))
    print("Annotations:", len(data["annotations"]))

    print()
    print("Images containing each class:")

    for cls, count in class_counts.most_common():
        print(f"{cls:20} {count}")

    print()
    print("Images containing conjunctivitis + another class:")

    mixed = []

    for image_id, classes in image_classes.items():

        if "conjunctivitis" in classes and len(classes) > 1:

            mixed.append(
                (
                    image_names.get(image_id, "UNKNOWN"),
                    sorted(classes)
                )
            )

    print("Mixed-label images:", len(mixed))

    for name, classes in mixed[:100]:
        print(f"{name} -> {', '.join(classes)}")