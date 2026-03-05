import matplotlib.pyplot as plt
from src.clip_utils import embed_image, embed_text
from src.utils import cosine
import seaborn as sns

sns.set_theme(style="whitegrid")

img_glasses = embed_image("data/images/glasses/1.jpg")
img_no_glasses = embed_image("data/images/no_glasses/1.jpg")

delta = img_glasses - img_no_glasses

concepts = ["glasses", "wearing glasses", "spectacles", "person", "eyes", "face"]

scores = []

for c in concepts:
    scores.append(cosine(delta, embed_text(c)))

plt.figure(figsize=(7, 4))

plt.bar(concepts, scores)

plt.title("Alignment of Image Difference with Text Concepts")
plt.ylabel("Cosine Similarity")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("figures/vector_arithmetic.png", dpi=300, bbox_inches="tight")
plt.show()
