import zipfile
import json

zip_path = r"C:\Users\ashis\Downloads\Dog Eye Problems Detection-Forked on 8-22-2026.coco (1).zip"

with zipfile.ZipFile(zip_path) as z:

    for split in ["train", "valid", "test"]:

        annotation_file = f"{split}/_annotations.coco.json"

        data = json.loads(z.read(annotation_file))

        print()
        print("=" * 60)
        print(split.upper())
        print("=" * 60)

        print("Images:", len(data.get("images", [])))
        print("Annotations:", len(data.get("annotations", [])))

        print()
        print("Categories:")

        for category in data.get("categories", []):
            print(f"{category['id']}: {category['name']}")