import numpy as np
import matplotlib.pyplot as plt
import os


def normalize(v):
    return v / np.linalg.norm(v)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def save_fig(name):
    path = f"figures/{name}"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
