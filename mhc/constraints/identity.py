import torch
import torch.nn.functional as F

def project_identity_preserving(logits: torch.Tensor, epsilon: float = 0.1, temperature: float = 1.0) -> torch.Tensor:
    """
    Guarantees that the last state (the most recent one) has at least `epsilon` weight.
    
    The weights are calculated as:
    alpha_last = epsilon + (1 - epsilon) * softmax(logits)_last
    alpha_others = (1 - epsilon) * softmax(logits)_others
    
    Args:
        logits: Input weights before normalization.
        epsilon: Minimum weight for the latest state (0 <= epsilon < 1).
        temperature: Controls the sharpness.
        
    Returns:
        alphas: Normalized weights with identity preservation guarantee.
    """
    probs = F.softmax(logits / temperature, dim=-1)
    
    # Scale all probabilities by (1 - epsilon)
    alphas = probs * (1.0 - epsilon)
    
    # Add epsilon to the last weight
    alphas[..., -1] += epsilon
    
    return alphas
