from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback
except ImportError:
    try:
        import pytorch_lightning as L  # type: ignore
        from pytorch_lightning.callbacks import Callback  # type: ignore
    except ImportError:
        L = None
        Callback = object  # type: ignore

from ..layers.managed import MHCSequential
from ..layers.mhc_skip import MHCSkip

class MHCLightningCallback(Callback):
    """Automates mHC history management and logging for PyTorch Lightning.

    This callback handles:
    1. Clearing history at the start of training/validation/test steps.
    2. Logging mixing weights (alphas) to the configured logger.
    """

    def __init__(
        self,
        log_mixing_weights: bool = True,
        log_interval: int = 50,
        prefix: str = "mhc"
    ):
        """
        Args:
            log_mixing_weights: If True, logs mean mixing weights per layer.
            log_interval: Number of steps between weight logging.
            prefix: Prefix for logged metrics.
        """
        if L is None:
            raise ImportError("PyTorch Lightning is required for MHCLightningCallback.")
        super().__init__()
        self.log_mixing_weights = log_mixing_weights
        self.log_interval = log_interval
        self.prefix = prefix

    def _clear_all_history(self, model: nn.Module):
        # Handle MHCSequential
        for module in model.modules():
            if isinstance(module, MHCSequential):
                module.history_buffer.clear()

        # Handle models where inject_mhc was used
        if hasattr(model, "clear_mhc_history"):
            model.clear_mhc_history()

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int):
        self._clear_all_history(pl_module)

    def on_validation_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._clear_all_history(pl_module)

    def on_test_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._clear_all_history(pl_module)

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        if self.log_mixing_weights and (batch_idx + 1) % self.log_interval == 0:
            self._log_alphas(pl_module)

    def _log_alphas(self, pl_module: L.LightningModule):
        for name, module in pl_module.named_modules():
            if isinstance(module, MHCSkip):
                with torch.no_grad():
                    # This is slightly simplified; in a real scenario we'd use visualization helpers
                    # but we stay lightweight here.
                    if hasattr(module, "mixing_logits"):
                        weights = torch.softmax(module.mixing_logits / getattr(module, "temperature", 1.0), dim=-1)
                        # Log the latest skip weight as a proxy for stability
                        pl_module.log(f"{self.prefix}/{name}/alpha_latest", weights[-1], on_step=True)
                        # Log entropy of weights as a measure of mixing diversity
                        entropy = -(weights * torch.log(weights + 1e-9)).sum()
                        pl_module.log(f"{self.prefix}/{name}/entropy", entropy, on_step=True)
