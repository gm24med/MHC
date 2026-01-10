import torch
import torch.nn as nn
from typing import Type, List, Optional, Union
from ..layers.mhc_skip import MHCSkip
from ..layers.history_buffer import HistoryBuffer

class InjectedMHC(nn.Module):
    """A wrapper that pairs a module with an MHCSkip layer.
    
    This is used by the injection utility to replace existing modules.
    """
    def __init__(
        self, 
        base_module: nn.Module, 
        history_buffer: HistoryBuffer,
        **mhc_kwargs
    ):
        super().__init__()
        self.base_module = base_module
        self.history_buffer = history_buffer
        self.skip = MHCSkip(**mhc_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_module(x)
        out = self.skip(out, self.history_buffer.get())
        self.history_buffer.append(out)
        return out

def inject_mhc(
    model: nn.Module,
    target_types: Union[Type[nn.Module], List[Type[nn.Module]]],
    max_history: int = 4,
    **mhc_kwargs
) -> nn.Module:
    """Automatically injects Hyper-Connections into a model.
    
    Replaces all instances of `target_types` with a wrapper that adds mHC-skip
    and shares a single HistoryBuffer.
    
    Args:
        model: The PyTorch model to modify.
        target_types: The module class(es) to detect and wrap (e.g., nn.Linear).
        max_history: Max history for the injected skips.
        **mhc_kwargs: Arguments passed to MHCSkip.
        
    Returns:
        nn.Module: The modified model.
    """
    if isinstance(target_types, type):
        target_types = [target_types]
    
    history_buffer = HistoryBuffer(max_history=max_history, detach_history=True)
    
    def _replace_recursive(module: nn.Module):
        for name, child in module.named_children():
            if any(isinstance(child, t) for t in target_types):
                # Replace with wrapper
                wrapper = InjectedMHC(child, history_buffer, max_history=max_history, **mhc_kwargs)
                setattr(module, name, wrapper)
            else:
                _replace_recursive(child)
                
    _replace_recursive(model)
    
    # Attach clear_history method to the model for convenience
    def clear_history():
        history_buffer.clear()
        # Initial x_0 is usually needed, but in this automated case, 
        # we append the output of the first injected layer.
        # This is a bit tricky. Usually, you clear at the start of a sequence.
        
    model.clear_mhc_history = clear_history
    
    return model
