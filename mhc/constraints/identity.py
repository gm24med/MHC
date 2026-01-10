import torch
import torch.nn.functional as F

def project_identity_preserving(
    logits: torch.Tensor, 
    epsilon: float = 0.1, 
    temperature: float = 1.0
) -> torch.Tensor:
    """Projects logits onto the simplex while guaranteeing identity preservation.

    This constraint ensures that the most recent state (the last element in 
    the history window) always receives at least a minimum weight `epsilon`.
    The remaining `1 - epsilon` weight is distributed among all states 
    (including the latest) via softmax.

    Args:
        logits (torch.Tensor): Input weights before normalization.
        epsilon (float): Minimum guaranteed weight for the latest state. 
            Must be in the range [0, 1). Defaults to 0.1.
        temperature (float): Scaling factor for the underlying softmax. 
            Defaults to 1.0.

    Returns:
        torch.Tensor: Normalized weights with identity-preservation guarantees.
    """
    probs = F.softmax(logits / temperature, dim=-1)
    
    # Scale all probabilities by (1 - epsilon)
    alphas = probs * (1.0 - epsilon)
    
    # Add epsilon to the last weight
    alphas[..., -1] += epsilon
    
    return alphas
