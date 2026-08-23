import zipfile
import json
import shutil
from pathlib import Path

ZIP_PATH = Path(
    r"C:\Users\ashis\Downloads\Dog Eye Problems Detection-Forked on 8-22-2026.coco (1).zip"
)

OUTPUT = Path("expansion/dog_entropion")

TARGET_CLASS = "entropion"

OUTPUT.mkdir(parents=True, exist_ok=True)

total = 0

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    for split in ["train", "valid", "test"]:

        annotation_file = f"{split}/_annotations.coco.json"

        data = json.loads(z.read(annotation_file))

        categories = {
            c["id"]: c["name"]
            for c in data["categories"]
        }

        target_ids = {
            cid
            for cid, name in categories.items()
            if name == TARGET_CLASS
        }

        image_ids = set()

        for ann in data["annotations"]:
            if ann["category_id"] in target_ids:
                image_ids.add(ann["image_id"])

        images = {
            img["id"]: img["file_name"]
            for img in data["images"]
            if img["id"] in image_ids
        }

        extracted = 0

        for image_id, filename in images.items():

            source = f"{split}/{filename}"

            if source not in z.namelist():
                print("Missing:", source)
                continue

            suffix = Path(filename).suffix

            output_name = (
                f"{split}_{Path(filename).stem}{suffix}"
            )

            output_path = OUTPUT / output_name

            with z.open(source) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted += 1
            total += 1

        print(f"{split}: extracted {extracted} {TARGET_CLASS} images")

print()
print("=" * 60)
print("DOG ENTROPION EXTRACTION COMPLETE")
print("=" * 60)
print("Total extracted:", total)
print("Output:", OUTPUT)