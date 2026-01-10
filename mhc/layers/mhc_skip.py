import torch
import torch.nn as nn
from typing import List, Optional
from ..constraints import project_simplex, project_identity_preserving

class MHCSkip(nn.Module):
    """Manifold-Constrained Hyper-Connections (mHC) Skip Layer.

    This layer implements the core mixing logic for Hyper-Connections. It learns
    to mix a sliding window of previous states with the output of the current
    layer's transformation, such that:
    x_{l+1} = f(x_l) + sum(alpha_k * x_k)

    Mixing weights (alphas) are constrained according to the specified mode
    (e.g., simplex or identity-preserving) to ensure numerical stability.

    Attributes:
        mode (str): Mixing mode. One of:
            - "residual": Standard sum of current state and previous state.
            - "hc": Unconstrained Hyper-Connections (simple softmax).
            - "mhc": Manifold-Constrained Hyper-Connections (with constraints).
        max_history (int): Maximum number of previous states to mix.
        constraint (str): Geometric constraint for "mhc" mode. One of:
            - "simplex": sum(alpha) = 1, alpha >= 0.
            - "identity": Simplex + minimum weight epsilon on latest state.
        epsilon (float): Minimum weight for the latest state in "identity" mode.
        temperature (float): Softmax temperature for mixing weight sharpness.
        mixing_logits (nn.Parameter): Learnable parameters for mixing weights.
    """

    def __init__(
        self,
        mode: str = "mhc",
        max_history: int = 4,
        constraint: str = "simplex",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        init: str = "identity"
    ) -> None:
        """Initializes the MHCSkip layer.

        Args:
            mode: Mixing strategy choice. Defaults to "mhc".
            max_history: Window size for history mixing. Defaults to 4.
            constraint: Mathematical constraint for mHC mode. Defaults to "simplex".
            epsilon: Minimum identity weight for epsilon-bound constraints. Defaults to 0.1.
            temperature: Sharpness factor for softmax. Defaults to 1.0.
            init: Initialization strategy for mixing weights ("identity" or "uniform").
        """
        super().__init__()
        self.mode = mode
        self.max_history = max_history
        self.constraint = constraint
        self.epsilon = epsilon
        self.temperature = temperature
        
        self.mixing_logits = nn.Parameter(torch.zeros(max_history))
        self._reset_parameters(init)

    def _reset_parameters(self, init_type: str) -> None:
        """Initializes mixing logits based on the specified strategy."""
        if init_type == "identity":
            with torch.no_grad():
                self.mixing_logits.fill_(-10.0)
                self.mixing_logits[-1] = 0.0
        elif init_type == "uniform":
            nn.init.zeros_(self.mixing_logits)

    def forward(self, x: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        """Computes the skip mixing forward pass.

        Args:
            x: Output tensor from the current layer's core transformation (f(x_l)).
            history: List of historical states [x_0, x_1, ..., x_{l-1}].

        Returns:
            torch.Tensor: The mixed output x_{l+1}.
        """
        if self.mode == "residual" or not history:
            return x + (history[-1] if history else 0)

        hist_window = history[-self.max_history:]
        K = len(hist_window)
        logits = self.mixing_logits[-K:]
        
        if self.mode == "hc":
            alphas = torch.softmax(logits / self.temperature, dim=-1)
        elif self.mode == "mhc":
            if self.constraint == "simplex":
                alphas = project_simplex(logits, temperature=self.temperature)
            elif self.constraint == "identity":
                alphas = project_identity_preserving(
                    logits, epsilon=self.epsilon, temperature=self.temperature
                )
            else:
                raise ValueError(f"Unknown constraint: {self.constraint}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        history_mix = 0
        for i, alpha in enumerate(alphas):
            h_state = hist_window[i]
            if h_state.shape != x.shape:
                raise RuntimeError(
                    f"Shape mismatch in MHCSkip: current input shape {x.shape} "
                    f"does not match history state shape {h_state.shape}. "
                    "Ensure all layers in a Hyper-Connection block preserve dimensions."
                )
            history_mix = history_mix + alpha * h_state
            
        return x + history_mix
