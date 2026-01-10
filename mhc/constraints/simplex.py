import torch
import torch.nn.functional as F

def project_simplex(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Ensures mixing weights are non-negative and sum to 1 using Softmax.
    
    Args:
        logits: Input weights before normalization (mixing logits).
        temperature: Controls the sharpness of the distribution.
        
    Returns:
        alphas: Normalized mixing weights.
    """
    return F.softmax(logits / temperature, dim=-1)
