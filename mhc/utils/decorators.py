from __future__ import annotations

import functools
from typing import Any, Type, TypeVar

import torch.nn as nn

T = TypeVar("T", bound=Type[nn.Module])

def mhc_compatible(cls: T) -> T:
    """Decorator to mark a custom module as mHC-compatible.

    This decorator ensures that any instance of the class can be easily
    handled by mHC utilities and provides a clear signal to other developers
    that the module supports history-aware skip connections.
    """
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        # Custom flag for detection without isinstance checks across imports
        self._mhc_compatible = True

    cls.__init__ = __init__  # type: ignore
    return cls
