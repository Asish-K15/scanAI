import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

DUP_FILE = "metadata/duplicates.csv"
OUT_FILE = "metadata/duplicate_clusters.csv"

df = pd.read_csv(DUP_FILE)

# --------------------------------------------------
# Union-Find
# --------------------------------------------------

parent = {}
rank = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    if a not in parent:
        parent[a] = a
        rank[a] = 0
    if b not in parent:
        parent[b] = b
        rank[b] = 0

    ra = find(a)
    rb = find(b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


# Build graph from every duplicate pair
for _, row in df.iterrows():
    union(row["image_a"], row["image_b"])

# --------------------------------------------------
# Group images into connected components
# --------------------------------------------------

clusters = defaultdict(list)

for image in parent:
    clusters[find(image)].append(image)

print("Total duplicate pairs:", len(df))
print("Total duplicate clusters:", len(clusters))

# --------------------------------------------------
# Analyze each cluster
# --------------------------------------------------

rows = []

for cluster_id, images in enumerate(clusters.values(), start=1):

    classes = sorted(set(Path(x).parent.name for x in images))

    # Pairs belonging to this cluster
    image_set = set(images)
    mask = df["image_a"].isin(image_set) & df["image_b"].isin(image_set)
    pairs = df[mask]

    exact_pairs = int((pairs["hamming_distance"] == 0).sum())
    near_pairs = int((pairs["hamming_distance"] > 0).sum())

    rows.append({
        "cluster_id": cluster_id,
        "image_count": len(images),
        "class_count": len(classes),
        "classes": "|".join(classes),
        "pair_count": len(pairs),
        "exact_pairs": exact_pairs,
        "near_pairs": near_pairs,
        "cross_class": len(classes) > 1
    })

out = pd.DataFrame(rows)

out = out.sort_values(
    ["cross_class", "image_count"],
    ascending=[False, False]
)

out.to_csv(OUT_FILE, index=False)

print()
print("Clusters:", len(out))
print("Same-class clusters:", int((out.class_count == 1).sum()))
print("Cross-class clusters:", int((out.class_count > 1).sum()))
print()
print("Cross-class images:",
      int(out.loc[out.cross_class, "image_count"].sum()))
print()
print("Largest cross-class clusters:")
print(
    out[out.cross_class]
    .head(30)
    .to_string(index=False)
)

print()
print("Saved:", OUT_FILE)