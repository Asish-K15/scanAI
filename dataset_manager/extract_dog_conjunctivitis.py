import zipfile
import json
import shutil
from pathlib import Path

ZIP_PATH = Path(
    r"C:\Users\ashis\Downloads\Dog Eye Problems Detection-Forked on 8-22-2026.coco (1).zip"
)

OUT_DIR = Path("expansion/dog_conjunctivitis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

total = 0

with zipfile.ZipFile(ZIP_PATH) as z:

    for split in ["train", "valid", "test"]:

        annotation_file = f"{split}/_annotations.coco.json"

        data = json.loads(z.read(annotation_file))

        categories = {
            c["id"]: c["name"]
            for c in data["categories"]
        }

        conjunctivitis_id = next(
            cid for cid, name in categories.items()
            if name == "conjunctivitis"
        )

        conjunctivitis_images = set()

        for ann in data["annotations"]:

            if ann["category_id"] == conjunctivitis_id:
                conjunctivitis_images.add(ann["image_id"])

        images_by_id = {
            image["id"]: image["file_name"]
            for image in data["images"]
        }

        split_count = 0

        for image_id in conjunctivitis_images:

            filename = images_by_id[image_id]

            zip_path = f"{split}/{filename}"

            destination = OUT_DIR / f"{split}_{Path(filename).name}"

            with z.open(zip_path) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)

            split_count += 1
            total += 1

        print(
            f"{split}: extracted {split_count} conjunctivitis images"
        )

print()
print("=" * 60)
print("DOG CONJUNCTIVITIS EXTRACTION COMPLETE")
print("=" * 60)
print(f"Total extracted: {total}")
print(f"Output: {OUT_DIR}")