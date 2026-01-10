"""Visualization utilities for mHC models.

This module provides functions to visualize mixing weights, gradient flow,
and training dynamics for models using Manifold-Constrained Hyper-Connections.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from ..layers.mhc_skip import MHCSkip
from ..layers.managed import MHCSequential


def extract_mixing_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract mixing weights (alphas) from all MHCSkip layers in a model.

    Args:
        model: PyTorch model containing MHCSkip layers.

    Returns:
        Dictionary mapping layer names to their mixing weight tensors.
    """
    weights = {}

    for name, module in model.named_modules():
        if isinstance(module, MHCSkip):
            # Get the current mixing weights (after softmax/projection)
            with torch.no_grad():
                if module.mode == "residual":
                    continue

                logits = module.mixing_logits

                if module.mode == "hc":
                    alphas = torch.softmax(logits / module.temperature, dim=-1)
                elif module.mode == "mhc":
                    if module.constraint == "simplex":
                        from ..constraints import project_simplex
                        alphas = project_simplex(logits, temperature=module.temperature)
                    elif module.constraint == "identity":
                        from ..constraints import project_identity_preserving
                        alphas = project_identity_preserving(
                            logits, epsilon=module.epsilon, temperature=module.temperature
                        )
                    else:
                        continue
                else:
                    continue

                weights[name] = alphas.cpu()

    return weights


def plot_mixing_weights(
    model: nn.Module,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "viridis"
) -> plt.Figure:
    """Plot mixing weights as a heatmap across all MHCSkip layers.

    Args:
        model: PyTorch model containing MHCSkip layers.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height).
        cmap: Colormap for the heatmap.

    Returns:
        Matplotlib figure object.
    """
    weights_dict = extract_mixing_weights(model)

    if not weights_dict:
        raise ValueError("No MHCSkip layers found in the model")

    # Convert to numpy array for plotting
    layer_names = list(weights_dict.keys())
    weights_array = np.array([weights_dict[name].numpy() for name in layer_names])

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(weights_array, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_yticks(np.arange(len(layer_names)))
    ax.set_yticklabels([name.split('.')[-1] for name in layer_names])
    ax.set_xlabel('History Position (0=oldest, -1=latest)', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('mHC Mixing Weights Across Layers', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight (α)', rotation=270, labelpad=20, fontsize=12)

    # Add grid
    ax.set_xticks(np.arange(weights_array.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(layer_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_gradient_flow(
    model: nn.Module,
    input_data: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    loss_fn: Optional[nn.Module] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Visualize gradient flow through the network.

    Args:
        model: PyTorch model to analyze.
        input_data: Input tensor for forward pass.
        target: Optional target for loss computation.
        loss_fn: Optional loss function. If None, uses dummy loss.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure object.
    """
    model.train()
    model.zero_grad()

    # Forward pass
    output = model(input_data)

    # Compute loss
    if loss_fn is not None and target is not None:
        loss = loss_fn(output, target)
    else:
        # Dummy loss for visualization
        loss = output.mean()

    # Backward pass
    loss.backward()

    # Extract gradient norms
    layer_names = []
    grad_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            grad_norms.append(param.grad.norm().item())

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(layer_names))
    ax.bar(x, grad_norms, alpha=0.7, color='steelblue', edgecolor='black')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Flow Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at mean
    mean_grad = np.mean(grad_norms)
    ax.axhline(y=mean_grad, color='r', linestyle='--', label=f'Mean: {mean_grad:.4f}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_history_contribution(
    alphas: torch.Tensor,
    layer_name: str = "MHCSkip",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """Plot the contribution of each historical state for a single layer.

    Args:
        alphas: Mixing weights tensor of shape (max_history,).
        layer_name: Name of the layer for the title.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure object.
    """
    alphas_np = alphas.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(alphas_np))
    bars = ax.bar(x, alphas_np, alpha=0.7, edgecolor='black')

    # Color the latest state differently
    bars[-1].set_color('coral')
    bars[-1].set_label('Latest State')
    for i in range(len(bars) - 1):
        bars[i].set_color('steelblue')

    ax.set_xlabel('History Position (0=oldest)', fontsize=12)
    ax.set_ylabel('Mixing Weight (α)', fontsize=12)
    ax.set_title(f'Historical State Contributions - {layer_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f't-{len(alphas_np)-1-i}' for i in x])
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, alphas_np)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_training_dashboard(
    metrics_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Create a comprehensive training dashboard with multiple metrics.

    Args:
        metrics_dict: Dictionary containing training metrics.
            Expected keys: 'loss', 'accuracy', 'mixing_weights_history'
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure object.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Loss curve
    if 'loss' in metrics_dict:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(metrics_dict['loss'], linewidth=2, color='steelblue')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)

    # Plot 2: Accuracy curve
    if 'accuracy' in metrics_dict:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(metrics_dict['accuracy'], linewidth=2, color='forestgreen')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Training Accuracy', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

    # Plot 3: Mixing weights evolution
    if 'mixing_weights_history' in metrics_dict:
        ax3 = fig.add_subplot(gs[1, :])
        weights_history = np.array(metrics_dict['mixing_weights_history'])

        for i in range(weights_history.shape[1]):
            label = f'α_{i}' if i < weights_history.shape[1] - 1 else 'α_latest'
            ax3.plot(weights_history[:, i], label=label, linewidth=2)

        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Mixing Weight', fontsize=11)
        ax3.set_title('Mixing Weights Evolution', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(alpha=0.3)

    plt.suptitle('mHC Training Dashboard', fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
