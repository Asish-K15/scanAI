import pandas as pd
from pathlib import Path

csv_path = Path("metadata/duplicates.csv")
out_path = Path("metadata/duplicate_clusters.csv")

df = pd.read_csv(csv_path)

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

for _, row in df.iterrows():
    union(row["image_a"], row["image_b"])

clusters = {}

for image in parent:
    root = find(image)
    clusters.setdefault(root, []).append(image)

rows = []

for cluster_id, images in enumerate(clusters.values(), start=1):
    for image in images:
        rows.append({
            "cluster_id": cluster_id,
            "cluster_size": len(images),
            "image": image
        })

out = pd.DataFrame(rows)
out.to_csv(out_path, index=False)

print(f"Duplicate pairs: {len(df)}")
print(f"Unique duplicate images: {len(parent)}")
print(f"Duplicate clusters: {len(clusters)}")
print(f"Largest cluster: {max(len(x) for x in clusters.values())}")
print(f"Written to: {out_path}")