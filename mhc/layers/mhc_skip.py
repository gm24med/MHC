import torch
import torch.nn as nn
from typing import List, Optional
from ..constraints import project_simplex, project_identity_preserving

class MHCSkip(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) Skip Layer.
    
    Mixes the current state with a sliding window of previous states from a history list.
    
    Args:
        mode: Mixing mode. One of "residual", "hc", "mhc".
        max_history: Maximum number of previous states to mix.
        constraint: Type of constraint to apply. One of "simplex", "identity".
        epsilon: Minimum weight for the latest state when using "identity" constraint.
        temperature: Softmax temperature for mixing weights.
        init: Initialization strategy. "identity" (recommends latest state) or "uniform".
    """
    def __init__(
        self,
        mode: str = "mhc",
        max_history: int = 4,
        constraint: str = "simplex",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        init: str = "identity"
    ):
        super().__init__()
        self.mode = mode
        self.max_history = max_history
        self.constraint = constraint
        self.epsilon = epsilon
        self.temperature = temperature
        
        # Learnable logits for mixing weights
        # We use max_history as the size for the logits
        self.mixing_logits = nn.Parameter(torch.zeros(max_history))
        
        self._reset_parameters(init)

    def _reset_parameters(self, init_type: str):
        if init_type == "identity":
            # Initialize so that the last state (index -1) is dominant
            with torch.no_grad():
                self.mixing_logits.fill_(-10.0) # Low values for others
                self.mixing_logits[-1] = 0.0     # Higher value for latest
        elif init_type == "uniform":
            nn.init.zeros_(self.mixing_logits)

    def forward(self, x: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Current tensor (output of the latest layer f(x)).
            history: List of previous states [x_0, x_1, ..., x_{l-1}].
        """
        if self.mode == "residual" or not history:
            # Standard residual connection logic or fallback
            return x + (history[-1] if history else 0)

        # 1. Select history window (at most max_history items)
        # We need to mix x AND elements from history.
        # Total items to mix = len(history_window) + 1 (for current state x placeholder)
        # Actually, in most HC formulations, x is the transformation f(x_l), 
        # and we mix it with a weighted sum of previous states x_k.
        # x_{l+1} = Transformation(x_l) + Mix(history)
        
        hist_window = history[-self.max_history:]
        K = len(hist_window)
        
        # 2. Get normalized alphas for the window
        # We only use the last K logits
        logits = self.mixing_logits[-K:]
        
        if self.mode == "hc":
            # Unconstrained normalized weights
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

        # 3. Compute weighted sum of history
        # hist_window tensors should have same shape as x
        # history_mix = sum(alpha_k * x_k)
        history_mix = 0
        for i, alpha in enumerate(alphas):
            history_mix = history_mix + alpha * hist_window[i]
            
        return x + history_mix
