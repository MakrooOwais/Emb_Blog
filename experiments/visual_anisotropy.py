import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from src.text_utils import embed_sentences
import seaborn as sns
from datasets import load_dataset

sns.set_theme(style="whitegrid")

# Load 1000 random samples from a real corpus
dataset = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(1000))
sentences = dataset["text"]

emb = embed_sentences(sentences)

sim = cosine_similarity(emb)

values = sim.flatten()

plt.figure(figsize=(6, 4))

plt.hist(values, bins=50)

plt.title("Distribution of Pairwise Cosine Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("figures/anisotropy_distribution.png", dpi=300, bbox_inches="tight")

plt.show()
