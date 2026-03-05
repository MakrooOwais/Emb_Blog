import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAIN = "laion2b_s34b_b79k"

TEXT_MODEL = "bert-base-uncased"

FIGURE_DIR = "figures/"
