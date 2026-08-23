import zipfile
import json
from collections import Counter, defaultdict
from pathlib import Path

ZIP_PATH = Path(
    r"C:\Users\ashis\Downloads\Dog Eye Problems Detection-Forked on 8-22-2026.coco (1).zip"
)

print("=" * 60)
print("DOG ORIGINAL COCO ANNOTATION ANALYSIS")
print("=" * 60)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    print("ZIP:", ZIP_PATH)
    print()

    for split in ["train", "valid", "test"]:

        annotation_path = f"{split}/_annotations.coco.json"

        print("=" * 60)
        print(split.upper())
        print("=" * 60)

        data = json.loads(z.read(annotation_path))

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
            for cls in classes:
                class_counts[cls] += 1

        print("Images:", len(data["images"]))
        print("Annotations:", len(data["annotations"]))

        print()
        print("Categories:")

        for category_id, name in categories.items():
            print(f"{category_id}: {name}")

        print()
        print("Images containing each class:")

        for cls, count in class_counts.most_common():
            print(f"{cls:20} {count}")

        # Specifically inspect conjunctivitis images
        conjunctivitis_images = []

        for image_id, classes in image_classes.items():

            if "conjunctivitis" in classes:
                conjunctivitis_images.append(
                    (
                        image_names.get(image_id, "UNKNOWN"),
                        sorted(classes)
                    )
                )

        print()
        print(
            "Images containing conjunctivitis:",
            len(conjunctivitis_images)
        )

        # Mixed labels
        mixed = [
            (name, classes)
            for name, classes in conjunctivitis_images
            if len(classes) > 1
        ]

        print()
        print(
            "Conjunctivitis images containing another class:",
            len(mixed)
        )

        if mixed:

            print()
            print("Mixed-label examples:")

            for name, classes in mixed[:100]:
                print(
                    f"{name} -> {', '.join(classes)}"
                )

        else:
            print("NONE")

        print()