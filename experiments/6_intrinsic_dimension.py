from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from src.text_utils import embed_sentences
from src.utils import save_fig
from datasets import load_dataset

dataset = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(1000))
sentences = dataset["text"]

emb = embed_sentences(sentences)

pca = PCA()

pca.fit(emb)

var = np.cumsum(pca.explained_variance_ratio_)

plt.plot(var)

plt.xlabel("Components")
plt.ylabel("Variance explained")

plt.title("Intrinsic dimensionality of embedding space")

save_fig("intrinsic_dimension.png")

plt.show()
