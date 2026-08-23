import zipfile
import json
from collections import Counter

ZIP_PATH = r"C:\Users\ashis\Downloads\Dog Eye Problems Detection-Forked on 8-22-2026.coco (1).zip"

with zipfile.ZipFile(ZIP_PATH) as z:

    for split in ["train", "valid", "test"]:

        annotation_file = f"{split}/_annotations.coco.json"
        data = json.loads(z.read(annotation_file))

        categories = {
            c["id"]: c["name"]
            for c in data["categories"]
        }

        image_labels = {}

        for ann in data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]

            image_labels.setdefault(image_id, set()).add(
                categories[category_id]
            )

        counts = Counter()

        for labels in image_labels.values():
            for label in labels:
                counts[label] += 1

        print()
        print("=" * 60)
        print(split.upper())
        print("=" * 60)

        print("Images:", len(data["images"]))
        print()

        for label, count in counts.most_common():
            print(f"{label:<20} {count}")

        print()
        print("Images containing conjunctivitis:",
              counts["conjunctivitis"])