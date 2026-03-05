# The Hidden Geometry of Embedding Spaces

When we talk about modern AI—like Large Language Models (LLMs), multimodal systems like CLIP, and search engines—we constantly hear the term **"embedding."** We know that embeddings are vectors (lists of numbers) representing data like text or images. But what actually happens when we map words, sentences, or images into these high-dimensional mathematical spaces?

It turns out that embeddings don't just act as unique identifiers. The space they inhabit has a rich, complex, and sometimes surprising geometry where distance and direction correspond to **meaning**. 

In this post, we’ll explore six fascinating properties of embedding spaces through hands-on experiments using open-source models like CLIP and BERT.

---

## 1. Semantic Vector Arithmetic: Math with Meanings

One of the most famous properties of word embeddings is that you can do math with them. The classic example is: `King - Man + Woman ≈ Queen`. 

But this linear semantic structure extends beyond single words and even across modalities like vision and text. In our first experiment using the **CLIP model** (which maps images and text into the same space), we wanted to see if the *difference* between two image embeddings could represent a tangible concept.

If we take the embedding of an image of a person *with glasses* and subtract the embedding of a person *without glasses*, the resulting vector direction strongly aligns with the text embedding for the word **"glasses"**. It shows that concepts in multimodal embedding spaces form distinct linear pathways.

*(See `figures/vector_arithmetic.png` for our cosine similarity alignments!)*

---

## 2. Semantic Clustering: Birds of a Feather

What happens if you embed thousands of images without giving the model any labels or categories, and then map those high-dimensional vectors down to a 2D plot?

Using images from the CIFAR-10 dataset embedded via CLIP and projected with **UMAP** (a dimensionality reduction technique), we observe distinct semantic clustering. 

Even though the model wasn't explicitly trained to classify these exact images into discrete buckets, the geometric space naturally groups similar concepts together. Dogs, trucks, airplanes, and cats all form their own distinct islands in the embedding space. This proves that distance in embedding space correlates deeply with semantic similarity.

*(See `figures/clip_clusters.png`)*

---

## 3. Anisotropy: The "Narrow Cone" Problem

Ideally, we might imagine embedding vectors spread out uniformly in all directions, like stars filling the night sky. In reality, many models—especially text models like BERT—produce **anisotropic** embeddings. 

This means the vectors don't use the whole space. Instead, they occupy a narrow cone. In our experiment with a standard BERT model, we computed the pairwise cosine similarity of hundreds of random sentences. Instead of the similarities centering around zero (which would happen if they were uniformly distributed and mostly orthogonal), the average cosine similarity was **~0.65**. 

Because vectors are clumped together in a specific direction, differentiating between concepts requires looking at minor variations *within* that cone.

*(See `figures/anisotropy_distribution.png`)*

---

## 4. Hubness: The "Popular Kid" Phenomenon

High-dimensional spaces do weird things. One of the strangest is the **hubness problem**. 

In a high-dimensional vector space, a small number of points tend to become the "nearest neighbors" to a disproportionately large number of other points. These vectors act as "hubs."

In our sentence embedding experiment, out of 500 sentences, a handful of specific embeddings appeared as a top-10 nearest neighbor for over 30 different sentences! 

This is a notorious problem for retrieval systems and search engines. If certain embeddings act as universal hubs, they will constantly show up in search results regardless of the query, degrading the system's accuracy.

*(See `figures/hubness_histogram.png`)*

---

## 5. Concepts as Directions

If difference vectors can represent objects like glasses, can they also represent abstract concepts like sentiment? Yes.

We can define a **sentiment direction** by taking the mean embedding of positive sentences ("This movie is fantastic", "I loved the film") and subtracting the mean embedding of negative sentences ("This movie is terrible", "Awful acting"). 

When we test a new sentence ("The movie was wonderful") against this established sentiment direction vector, we get a solid positive cosine similarity (around **0.226** in our test). This property is foundational for interpretability and probing—we can understand what a model is "thinking" by finding the directions that correlate with human-understandable concepts.

---

## 6. Intrinsic Dimensionality: Less Than Meets the Eye

Modern embeddings are incredibly wide. A standard BERT embedding has 768 dimensions; OpenAI's and CLIP's can have over 1000. 

But do they actually need all that room? We ran a **Principal Component Analysis (PCA)** on a set of 1,000 sentence embeddings to measure their variance. 

The results show that the *effective* or **intrinsic dimensionality** of the space is much lower than the actual number of dimensions. A huge percentage of the variance is captured by just the first 50 to 100 principal components. This suggests that the embeddings actually lie on a lower-dimensional manifold twisted within the high-dimensional space.

*(See `figures/intrinsic_dimension.png`)*

---

### Conclusion

Embeddings are not just random coordinates; they are structured, geometric representations of human meaning. By understanding phenomena like anisotropy, hubness, and intrinsic dimensionality, we can build better retrieval systems, probe model behavior, and ultimately understand how neural networks conceptualize our world.

*All code to reproduce these experiments and generate the figures is available in this repository. Feel free to run them yourself!*