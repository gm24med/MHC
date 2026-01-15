import torch
import torch.nn as nn
from typing import List, Optional
from ..constraints import project_simplex, project_identity_preserving
from ..config import resolve_default

class MHCSkip(nn.Module):
    """Manifold-Constrained Hyper-Connections (mHC) Skip Layer.

    This layer implements the core mixing logic for Hyper-Connections. It learns
    to mix a sliding window of previous states with the output of the current
    layer's transformation, such that:
    x_{l+1} = f(x_l) + sum(alpha_k * x_k)

    Mixing weights (alphas) are constrained according to the specified mode
    (e.g., simplex or identity-preserving) to ensure numerical stability.

    Attributes:
        mode (str): Mixing mode. One of: "residual", "hc", "mhc".
        max_history (int): Maximum number of previous states to mix.
        constraint (str): Geometric constraint for "mhc" mode.
        epsilon (float): Minimum weight for the latest state in "identity" mode.
        temperature (float): Softmax temperature for mixing weight sharpness.
        mixing_logits (nn.Parameter): Learnable parameters for mixing weights.
        use_gating (bool): If True, learns a global gate to modulate history contribution.
        prune_threshold (float): Weights below this value will be zeroed out.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        max_history: Optional[int] = None,
        constraint: Optional[str] = None,
        epsilon: Optional[float] = None,
        temperature: Optional[float] = None,
        init: Optional[str] = None,
        auto_project: Optional[bool] = None,
        use_gating: bool = False,
        prune_threshold: float = 0.0,
        stochastic: bool = False
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
            use_gating: If True, learns a global gate to modulate history contribution.
            prune_threshold: Weights below this value will be zeroed out.
            stochastic: If True, use stochastic mixing (Gumbel-Softmax).
        """
        super().__init__()
        self.mode = resolve_default(mode, "mode")
        self.max_history = resolve_default(max_history, "max_history")
        self.constraint = resolve_default(constraint, "constraint")
        self.epsilon = resolve_default(epsilon, "epsilon")
        self.temperature = resolve_default(temperature, "temperature")
        self.auto_project = resolve_default(auto_project, "auto_project")
        self.use_gating = use_gating
        self.prune_threshold = prune_threshold
        self.stochastic = stochastic
        self.projection: Optional[nn.Module] = None

        self.mixing_logits = nn.Parameter(torch.zeros(self.max_history))
        if self.use_gating:
            self.gate_logit = nn.Parameter(torch.tensor(2.0))

        self._reset_parameters(resolve_default(init, "init"))

    def _build_projection(self, history: torch.Tensor, x: torch.Tensor) -> nn.Module:
        if history.dim() == 4 and x.dim() == 4:
            if history.shape[2:] != x.shape[2:]:
                raise RuntimeError("Auto projection only supports channel changes when spatial dims match.")
            projection = nn.Conv2d(history.shape[1], x.shape[1], kernel_size=1, bias=False)
        else:
            if history.shape[:-1] != x.shape[:-1]:
                raise RuntimeError("Auto projection only supports matching leading dimensions.")
            projection = nn.Linear(history.shape[-1], x.shape[-1], bias=False)
        return projection.to(device=x.device, dtype=x.dtype)

    def _project_history(self, history: List[torch.Tensor], x: torch.Tensor) -> List[torch.Tensor]:
        mismatched = [h for h in history if h.shape != x.shape]
        if not mismatched:
            return history
        base_shape = mismatched[0].shape
        if any(h.shape != base_shape for h in mismatched):
            raise RuntimeError("Auto projection requires all mismatched history states to share a shape.")
        if self.projection is None:
            self.projection = self._build_projection(mismatched[0], x)
        return [self.projection(h) if h.shape != x.shape else h for h in history]

    def _reset_parameters(self, init_type: str) -> None:
        if init_type == "identity":
            with torch.no_grad():
                self.mixing_logits.fill_(-10.0)
                self.mixing_logits[-1] = 0.0
        elif init_type == "uniform":
            nn.init.zeros_(self.mixing_logits)

    def forward(self, x: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        if not history:
            return x
        if self.mode == "residual":
            h = history[-1]
            if h.shape != x.shape:
                if not self.auto_project:
                    raise RuntimeError("Shape mismatch in residual mode. Enable auto_project.")
                h = self._project_history([h], x)[0]
            return x + h

        hist_window = history[-self.max_history:]
        if any(h.shape != x.shape for h in hist_window):
            if not self.auto_project:
                raise RuntimeError("Shape mismatch in mHC mode. Enable auto_project.")
            hist_window = self._project_history(hist_window, x)

        K = len(hist_window)
        logits = self.mixing_logits[-K:]

        if self.stochastic and self.training:
            alphas = torch.nn.functional.gumbel_softmax(
                logits, tau=self.temperature, hard=False, dim=-1
            )
        elif self.mode == "hc":
            alphas = torch.softmax(logits / self.temperature, dim=-1)
        elif self.mode == "mhc":
            if self.constraint == "simplex":
                alphas = project_simplex(logits, temperature=self.temperature)
            elif self.constraint == "identity":
                alphas = project_identity_preserving(logits, epsilon=self.epsilon, temperature=self.temperature)
            else:
                raise ValueError(f"Unknown constraint: {self.constraint}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        history_mix = 0
        for i, alpha in enumerate(alphas):
            if alpha < self.prune_threshold:
                continue
            h_state = hist_window[i]
            if h_state.shape != x.shape:
                raise RuntimeError("Shape mismatch in history mix.")
            history_mix = history_mix + alpha * h_state

        if self.use_gating:
            history_mix = history_mix * torch.sigmoid(self.gate_logit)

        return x + history_mix
