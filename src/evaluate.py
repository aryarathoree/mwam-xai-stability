import torch
import numpy as np

def compute_attribution(model, x_tensor):
    """Computes gradient-based feature importance vectors."""
    x_tensor = x_tensor.clone().detach().requires_grad_(True)
    logits = model(x_tensor)
    target_score = logits[:, 1].sum()
    target_score.backward()
    attr = (x_tensor * x_tensor.grad).abs().detach().cpu().numpy()
    return attr.reshape(len(x_tensor), -1)

def compute_jaccard(a1, a2, k=15):
    """Calculates Top-k Jaccard Similarity (J_k)."""
    top1 = set(np.argsort(a1)[-k:])
    top2 = set(np.argsort(a2)[-k:])
    return len(top1.intersection(top2)) / float(len(top1.union(top2)))

def compute_esens(a1, a2):
    """Calculates Relative Explanation Sensitivity (E_sens)."""
    diff = np.linalg.norm(a1 - a2, axis=1)
    norm = np.linalg.norm(a1, axis=1) + 1e-8
    return np.mean(diff / norm)