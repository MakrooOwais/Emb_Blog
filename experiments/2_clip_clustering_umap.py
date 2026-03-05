from datasets import load_dataset
from src.clip_utils import embed_image
import numpy as np
import umap
import matplotlib.pyplot as plt
from src.utils import save_fig

dataset = load_dataset("cifar10", split="train[:2000]")

embeddings = []
labels = []

for item in dataset:

    img = item["img"]

    emb = embed_image(img)

    embeddings.append(emb)
    labels.append(item["label"])

embeddings = np.array(embeddings)

reducer = umap.UMAP()

proj = reducer.fit_transform(embeddings)

plt.figure(figsize=(8, 6))

plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=5)

plt.title("CLIP embedding clusters")

save_fig("clip_clusters.png")

plt.show()
