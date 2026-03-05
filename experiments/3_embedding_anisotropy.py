from src.text_utils import embed_sentences
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "The cat sat on the mat",
    "Machine learning is fascinating",
    "Deep learning models learn representations",
    "The weather is beautiful today",
    "I enjoy reading research papers",
    "Neural networks can approximate functions",
    "Transformers changed natural language processing",
]

emb = embed_sentences(sentences)

sim = cosine_similarity(emb)

print("Average cosine similarity:", sim.mean())
