import torch

def project_doubly_stochastic(
    logits: torch.Tensor,
    iterations: int = 10,
    temperature: float = 1.0
) -> torch.Tensor:
    """Projects a matrix onto the set of doubly stochastic matrices.

    Uses the Sinkhorn-Knopp algorithm to ensure that each row and each column
    of the resulting matrix sums to 1. This is useful for "Matrix Mixing"
    where we mix across both history depth and feature dimensions.

    Args:
        logits (torch.Tensor): Input matrix (or batch of matrices).
        iterations (int): Number of Sinkhorn iterations. Defaults to 10.
        temperature (float): Softmax temperature for initial normalization.

    Returns:
        torch.Tensor: A doubly stochastic matrix.
    """
    # Stabilize logits before exp to reduce overflow risk
    scaled = logits / temperature
    scaled = scaled - scaled.amax(dim=-1, keepdim=True)
    M = torch.exp(scaled)

    for _ in range(iterations):
        # Row normalization
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-9)
        # Column normalization
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-9)

    return M
