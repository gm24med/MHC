import torch

def ensure_list(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, list):
        return x
    return list(x)

def get_last_k(history, k):
    if not history:
        return []
    return history[-k:]
