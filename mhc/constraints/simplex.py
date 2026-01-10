import torch
import torch.nn.functional as F

def project_simplex(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Projects logits onto the probability simplex using Softmax.

    This ensures that the resulting mixing weights are non-negative and
    sum to 1, forming a valid convex combination.

    Args:
        logits (torch.Tensor): Input weights before normalization (arbitrary real values).
        temperature (float): Scaling factor for the softmax. Lower values make
            the distribution "sharper" (closer to a one-hot vector), while
            higher values make it more uniform. Defaults to 1.0.

    Returns:
        torch.Tensor: Normalized mixing weights (alphas) on the simplex.
    """
    return F.softmax(logits / temperature, dim=-1)
