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
        init: str = "identity",
        auto_project: bool = False
    ) -> None:
        """Initializes the MHCSkip layer.

        Args:
            mode: Mixing strategy choice. Defaults to "mhc".
            max_history: Window size for history mixing. Defaults to 4.
            constraint: Mathematical constraint for mHC mode. Defaults to "simplex".
            epsilon: Minimum identity weight for epsilon-bound constraints. Defaults to 0.1.
            temperature: Sharpness factor for softmax. Defaults to 1.0.
            init: Initialization strategy for mixing weights ("identity" or "uniform").
            auto_project: If True, project mismatched history shapes to match x.
        """
        super().__init__()
        self.mode = mode
        self.max_history = max_history
        self.constraint = constraint
        self.epsilon = epsilon
        self.temperature = temperature
        self.auto_project = auto_project
        self.projection: Optional[nn.Module] = None

        self.mixing_logits = nn.Parameter(torch.zeros(max_history))
        self._reset_parameters(init)

    def _build_projection(self, history: torch.Tensor, x: torch.Tensor) -> nn.Module:
        if history.dim() == 4 and x.dim() == 4:
            if history.shape[2:] != x.shape[2:]:
                raise RuntimeError(
                    "Auto projection only supports channel changes when spatial dims match."
                )
            projection = nn.Conv2d(
                in_channels=history.shape[1],
                out_channels=x.shape[1],
                kernel_size=1,
                bias=False
            )
        else:
            if history.shape[:-1] != x.shape[:-1]:
                raise RuntimeError(
                    "Auto projection only supports matching leading dimensions."
                )
            projection = nn.Linear(
                in_features=history.shape[-1],
                out_features=x.shape[-1],
                bias=False
            )
        return projection.to(device=x.device, dtype=x.dtype)

    def _project_history(self, history: List[torch.Tensor], x: torch.Tensor) -> List[torch.Tensor]:
        mismatched = [h for h in history if h.shape != x.shape]
        if not mismatched:
            return history
        base_shape = mismatched[0].shape
        if any(h.shape != base_shape for h in mismatched):
            raise RuntimeError(
                "Auto projection requires all mismatched history states to share a shape."
            )
        if self.projection is None:
            self.projection = self._build_projection(mismatched[0], x)
        return [self.projection(h) if h.shape != x.shape else h for h in history]

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
        if not history:
            return x
        if self.mode == "residual":
            h = history[-1]
            if h.shape != x.shape:
                if not self.auto_project:
                    raise RuntimeError(
                        f"{self.__class__.__name__} shape mismatch: current input shape "
                        f"{x.shape} does not match history state shape {h.shape}. "
                        "Enable auto_project or ensure dimensions match."
                    )
                h = self._project_history([h], x)[0]
            return x + h

        hist_window = history[-self.max_history:]
        if any(h.shape != x.shape for h in hist_window):
            if not self.auto_project:
                raise RuntimeError(
                    f"{self.__class__.__name__} shape mismatch: current input shape "
                    f"{x.shape} does not match history state shape. "
                    "Enable auto_project or ensure dimensions match."
                )
            hist_window = self._project_history(hist_window, x)
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
