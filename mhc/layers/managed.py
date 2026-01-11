import torch
import torch.nn as nn
from typing import Iterable, Optional

from .mhc_skip import MHCSkip
from .history_buffer import HistoryBuffer
from ..config import resolve_default

class MHCSequential(nn.Module):
    """A Sequential container that automatically manages Hyper-Connections.

    This class wraps a sequence of modules and handles the history management
    transparently. It inserts an `MHCSkip` layer after each wrapped module and
    updates a internal `HistoryBuffer` during the forward pass.

    This is the recommended way to use Hyper-Connections in a simple feedstock
    network (like an MLP or a Stack of Transformer layers) as it eliminates
    the need to manually manage buffers.

    Attributes:
        wrapped_modules (nn.ModuleList): The original sequential modules.
        skip_layers (nn.ModuleList): Corresponding MHCSkip layers for each module.
        history_buffer (HistoryBuffer): Shared buffer for historical states.
    """

    def __init__(
        self,
        modules: Iterable[nn.Module],
        max_history: Optional[int] = None,
        mode: Optional[str] = None,
        constraint: Optional[str] = None,
        epsilon: Optional[float] = None,
        detach_history: Optional[bool] = None,
        clear_history_each_forward: Optional[bool] = None
    ) -> None:
        """Initializes the MHCSequential container.

        Args:
            modules: An iterable of modules (e.g., layers) to be wrapped.
            max_history: Max history window size for the skip layers. Defaults to 4.
            mode: Mixing mode ("mhc", "hc", "residual"). Defaults to "mhc".
            constraint: Geometric constraint type. Defaults to "simplex".
            epsilon: Identity preservation epsilon. Defaults to 0.1.
            detach_history: Whether to detach history tensors. Recommended to be
                True for long sequential chains to avoid memory issues.
            clear_history_each_forward: Whether to reset history at each forward.
        """
        super().__init__()
        self.wrapped_modules = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.clear_history_each_forward = resolve_default(
            clear_history_each_forward, "clear_history_each_forward"
        )
        self.history_buffer = HistoryBuffer(
            max_history=resolve_default(max_history, "max_history"),
            detach_history=resolve_default(detach_history, "detach_history")
        )

        for module in modules:
            self.wrapped_modules.append(module)
            self.skip_layers.append(
                MHCSkip(
                    mode=resolve_default(mode, "mode"),
                    max_history=resolve_default(max_history, "max_history"),
                    constraint=resolve_default(constraint, "constraint"),
                    epsilon=resolve_default(epsilon, "epsilon")
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automated history management.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: The final output of the sequential stack.
        """
        if self.clear_history_each_forward:
            self.history_buffer.clear()

        # Initial state x_0
        self.history_buffer.append(x)

        for module, skip in zip(self.wrapped_modules, self.skip_layers):
            # Apply module transformation
            f_x = module(x)

            # Mix with history
            x = skip(f_x, self.history_buffer.get())

            # Update history with the mixed state
            self.history_buffer.append(x)

        return x
