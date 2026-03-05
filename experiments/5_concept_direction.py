from src.text_utils import embed_sentences
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

positive = [
    "This movie is fantastic",
    "I loved the film",
    "Amazing acting",
]

negative = [
    "This movie is terrible",
    "I hated the film",
    "Awful acting",
]

pos = embed_sentences(positive)
neg = embed_sentences(negative)

direction = pos.mean(axis=0) - neg.mean(axis=0)

test = embed_sentences(["The movie was wonderful"])

score = cosine_similarity([direction], test)

print("Sentiment direction similarity:", score)
