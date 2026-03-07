import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from src.text_utils import embed_sentences
import seaborn as sns
from datasets import load_dataset

sns.set_theme(style="whitegrid")

dataset = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(1000))
sentences = dataset["text"]

emb = embed_sentences(sentences)

nbrs = NearestNeighbors(n_neighbors=10, metric="cosine").fit(emb)

distances, indices = nbrs.kneighbors(emb)

hub_count = {}

for row in indices:
    for idx in row:
        hub_count[idx] = hub_count.get(idx, 0) + 1

counts = list(hub_count.values())

plt.figure(figsize=(6, 4))

plt.hist(counts, bins=30)

plt.title("Hubness in Embedding Space")
plt.xlabel("Number of times appearing as nearest neighbor")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("figures/hubness_histogram.png", dpi=300, bbox_inches="tight")

plt.show()
