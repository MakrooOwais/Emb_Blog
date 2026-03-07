import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.text_utils import embed_sentences
import seaborn as sns
from datasets import load_dataset

sns.set_theme(style="whitegrid")

dataset = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(1000))
sentences = dataset["text"]

emb = embed_sentences(sentences)

pca = PCA()
pca.fit(emb)

variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(7, 4))

plt.plot(variance)

plt.title("Intrinsic Dimensionality of Embedding Space")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")

plt.axhline(0.9, linestyle="--")

plt.tight_layout()
plt.savefig("figures/intrinsic_dimension.png", dpi=300, bbox_inches="tight")

plt.show()
