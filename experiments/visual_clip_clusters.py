import numpy as np
import matplotlib.pyplot as plt
import umap
from datasets import load_dataset
from src.clip_utils import embed_image
import seaborn as sns

sns.set_theme(style="whitegrid")


dataset = load_dataset("cifar10", split="train[:2000]")

embeddings = []
labels = []

for item in dataset:
    emb = embed_image(item["img"])
    embeddings.append(emb)
    labels.append(item["label"])

embeddings = np.array(embeddings)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")

projection = reducer.fit_transform(embeddings)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=6)

plt.title("Semantic Clusters in CLIP Embedding Space")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.colorbar(scatter)

plt.tight_layout()
plt.savefig("figures/clip_clusters.png", dpi=300, bbox_inches="tight")
plt.show()
