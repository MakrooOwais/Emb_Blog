from sklearn.neighbors import NearestNeighbors
import numpy as np
from src.text_utils import embed_sentences

sentences = [f"This is sentence {i}" for i in range(500)]

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
    print(idx, count)
