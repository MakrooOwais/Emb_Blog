import torch
import open_clip
from PIL import Image
import numpy as np
from .config import DEVICE, CLIP_MODEL, CLIP_PRETRAIN

model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAIN
)

tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

model = model.to(DEVICE)
model.eval()


def embed_image(path):
    if isinstance(path, str):
        img_obj = Image.open(path)
    else:
        img_obj = path
    img = preprocess(img_obj).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_image(img)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def embed_text(text):
    tokens = tokenizer([text]).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_text(tokens)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]
