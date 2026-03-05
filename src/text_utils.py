from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from .config import TEXT_MODEL

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
model = AutoModel.from_pretrained(TEXT_MODEL)

model.eval()


def embed_sentence(sentence):

    tokens = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        output = model(**tokens)

    emb = output.last_hidden_state.mean(dim=1)

    return emb.numpy()[0]


def embed_sentences(sentences):

    return np.array([embed_sentence(s) for s in sentences])
