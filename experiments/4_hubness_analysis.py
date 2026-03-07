from sklearn.neighbors import NearestNeighbors
import numpy as np
from src.text_utils import embed_sentences
from datasets import load_dataset

dataset = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(1000))
sentences = dataset["text"]

emb = embed_sentences(sentences)

nbrs = NearestNeighbors(n_neighbors=10, metric="cosine")
nbrs.fit(emb)

distances, indices = nbrs.kneighbors(emb)

hub_count = {}

for row in indices:

    for idx in row:

        hub_count[idx] = hub_count.get(idx, 0) + 1

sorted_hubs = sorted(hub_count.items(), key=lambda x: x[1], reverse=True)

print("Top hub sentences:\n")

for idx, count in sorted_hubs[:10]:
    print(f"Count: {count} | Sentence: {sentences[idx]}")
