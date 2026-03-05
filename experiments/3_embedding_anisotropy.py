from datasets import load_dataset
from src.text_utils import embed_sentences
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load 1000 random samples from a real corpus (AG News)
dataset = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(1000))
sentences = dataset["text"]

emb = embed_sentences(sentences)

sim = cosine_similarity(emb)

print("Average cosine similarity:", sim.mean())
