import numpy as np
from src.clip_utils import embed_image, embed_text
from src.utils import cosine

img_glasses = embed_image("data/images/glasses/1.jpg")
img_no_glasses = embed_image("data/images/no_glasses/1.jpg")

delta = img_glasses - img_no_glasses

concepts = ["glasses", "wearing glasses", "spectacles", "person", "eyes"]

print("\nVector arithmetic results:\n")

for c in concepts:

    emb = embed_text(c)

    sim = cosine(delta, emb)

    print(f"{c:15s} : {sim:.3f}")
