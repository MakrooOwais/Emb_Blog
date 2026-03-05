# Geometry of Embedding Spaces

This repository contains reproducible experiments exploring the **geometry and behavior of embedding spaces** in modern machine learning models.

Embeddings are vector representations of data (text, images, audio, etc.) that capture semantic relationships. Models such as **CLIP, BERT, and sentence transformers** map inputs into high-dimensional vector spaces where **distance and direction correspond to meaning**.

The goal of this project is to demonstrate several surprising and useful properties of these spaces through small, reproducible experiments.

The experiments in this repository generate both **numerical results and visualizations** that illustrate how semantic structure emerges in learned representations.

---

# Experiments Included

The repository contains six experiments that highlight different geometric properties of embedding spaces.

## 1. Semantic Vector Arithmetic (CLIP)

Many embedding spaces exhibit **linear semantic structure**. Differences between embeddings often correspond to meaningful semantic directions.

Example intuition:

```
embedding(person_with_glasses) - embedding(person_without_glasses)
≈ embedding("glasses")
```

This experiment demonstrates that the **difference between two image embeddings aligns with the text embedding of the corresponding concept**.

Output:

* Cosine similarity scores between vector differences and text embeddings
* A bar chart showing concept alignment

Generated figure:

```
figures/vector_arithmetic.png
```

*Image credit (Glasses vs No Glasses):* [OpticsTown](https://www.opticstown.com/a/blog/media/rvroptics.myshopify.com/Post/featured_img/1-3-5.jpg)

---

## 2. Semantic Clustering in Embedding Space

Even when models are not explicitly trained for classification, embeddings often **cluster by semantic category**.

In this experiment we embed images from CIFAR-10 using CLIP and project the embeddings into two dimensions using **UMAP**.

Result:

Objects with similar semantic meaning cluster together:

* airplanes
* dogs
* trucks
* cats

Generated figure:

```
figures/clip_clusters.png
```

This demonstrates that the embedding space captures **high-level semantic structure**.

---

## 3. Embedding Space Anisotropy

Ideally, embedding vectors would be **uniformly distributed across space**. In practice, many embedding spaces are **anisotropic** — vectors occupy a narrow cone rather than spreading uniformly.

This experiment computes cosine similarities between many random sentence embeddings.

Observation:

Instead of being centered around zero, the cosine similarities are **shifted toward positive values**.

Generated figure:

```
figures/anisotropy_distribution.png
```

This phenomenon has been observed in many models including BERT and sentence transformers.

---

## 4. Hubness in High-Dimensional Spaces

High-dimensional vector spaces exhibit a phenomenon called **hubness**.

Certain vectors become **nearest neighbors for an unusually large number of other vectors**, acting as hubs in the space.

This experiment:

1. Embeds many sentences
2. Computes nearest neighbors
3. Counts how often each vector appears as a neighbor

Generated figure:

```
figures/hubness_histogram.png
```

Hubness can negatively impact retrieval systems and similarity search.

---

## 5. Concept Directions

Concepts often appear as **directions in embedding space**.

For example, sentiment can be approximated by the vector:

```
mean(positive_embeddings) - mean(negative_embeddings)
```

This experiment constructs a **sentiment direction** and measures how strongly new sentences align with it.

This demonstrates how embeddings can be used for:

* concept probing
* interpretability
* representation analysis

---

## 6. Intrinsic Dimensionality

Although embeddings may be high dimensional (e.g., 768 or 1024 dimensions), the **effective dimensionality is often much lower**.

This experiment uses PCA to measure how much variance is captured by increasing numbers of components.

Typical result:

```
~90% of variance captured by < 100 dimensions
```

Generated figure:

```
figures/intrinsic_dimension.png
```

This suggests embeddings lie on **low-dimensional manifolds within high-dimensional space**.

---

# Repository Structure

```
embedding-space-blog/
│
├─ README.md
├─ requirements.txt
│
├─ data/
│   ├─ images/
│   │   ├─ glasses/
│   │   └─ no_glasses/
│
├─ src/
│   ├─ config.py
│   ├─ utils.py
│   ├─ clip_utils.py
│   └─ text_utils.py
│
├─ experiments/
│   ├─ 1_clip_vector_arithmetic.py
│   ├─ 2_clip_clustering_umap.py
│   ├─ 3_embedding_anisotropy.py
│   ├─ 4_hubness_analysis.py
│   ├─ 5_concept_direction.py
│   ├─ 6_intrinsic_dimension.py
│
└─ figures/
```

---

# Installation

Clone the repository:

```
git clone <repo-url>
cd embedding-space-blog
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Experiments

Each experiment is independent and can be run directly.

Example:

```
python experiments/1_clip_vector_arithmetic.py
```

Other experiments:

```
python experiments/2_clip_clustering_umap.py
python experiments/3_embedding_anisotropy.py
python experiments/4_hubness_analysis.py
python experiments/5_concept_direction.py
python experiments/6_intrinsic_dimension.py
```

Generated figures will appear in:

```
figures/
```

---

# Requirements

Key libraries used in this project:

* PyTorch
* OpenCLIP
* HuggingFace Transformers
* scikit-learn
* UMAP
* matplotlib
* seaborn

---

# Why These Experiments Matter

Embedding spaces are central to modern machine learning systems:

* Large Language Models
* Vision models
* Multimodal systems
* Retrieval and recommendation systems

Understanding the **geometry of embeddings** helps explain why these models behave the way they do.

Key insights demonstrated in this repository:

* Meaning emerges as **directions in vector space**
* Semantic structure forms **clusters**
* Embeddings exhibit **anisotropy and hubness**
* Concepts can be approximated as **linear directions**
* High-dimensional embeddings often lie on **low-dimensional manifolds**

---

# References

Relevant research and models:

CLIP
Learning Transferable Visual Models From Natural Language Supervision

BERT
Pre-training of Deep Bidirectional Transformers for Language Understanding

Concept Activation Vectors (TCAV)
Interpretability Beyond Feature Attribution

Hubness in High-Dimensional Spaces
Radovanović et al.

---

# Reproducibility

All experiments in this repository are designed to be:

* small
* fast
* reproducible

They are suitable for exploration, teaching, and demonstration of embedding space behavior.

---

# License

MIT License
