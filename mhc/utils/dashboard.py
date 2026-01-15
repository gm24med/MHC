from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Optional
from .visualization import extract_mixing_weights

def log_to_wandb(model: nn.Module, step: Optional[int] = None, prefix: str = "mhc"):
    """Logs mHC mixing weights to Weights & Biases.

    Args:
        model: The PyTorch model with mHC layers.
        step: Optional training step.
        prefix: Metric prefix.
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    weights = extract_mixing_weights(model)
    logs = {}

    for name, alphas in weights.items():
        # Log the full distribution as a histogram-like bar chart if possible,
        # but for standard WandB we log individual alpha values and entropy.
        for i, alpha in enumerate(alphas):
            label = f"alpha_{i}" if i < len(alphas) - 1 else "alpha_latest"
            logs[f"{prefix}/{name}/{label}"] = alpha.item()

        # Stability Metric: Entropy (lower is more concentrated/stable)
        entropy = -(alphas * torch.log(alphas + 1e-9)).sum().item()
        logs[f"{prefix}/{name}/entropy"] = entropy

    wandb.log(logs, step=step)

def log_to_tensorboard(writer: Any, model: nn.Module, step: int, prefix: str = "mhc"):
    """Logs mHC mixing weights to TensorBoard.

    Args:
        writer: SummaryWriter instance.
        model: The PyTorch model.
        step: Training step.
        prefix: Metric prefix.
    """
    weights = extract_mixing_weights(model)

    for name, alphas in weights.items():
        for i, alpha in enumerate(alphas):
            label = f"alpha_{i}" if i < len(alphas) - 1 else "alpha_latest"
            writer.add_scalar(f"{prefix}/{name}/{label}", alpha.item(), step)

        entropy = -(alphas * torch.log(alphas + 1e-9)).sum().item()
        writer.add_scalar(f"{prefix}/{name}/entropy", entropy, step)
