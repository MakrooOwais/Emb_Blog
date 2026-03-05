import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.text_utils import embed_sentences
import seaborn as sns

sns.set_theme(style="whitegrid")

sentences = [f"This is sentence {i}" for i in range(1000)]

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
