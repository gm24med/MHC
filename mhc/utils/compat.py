from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


def compile_model(model: nn.Module, **kwargs: Any) -> nn.Module:
    """Return a torch.compile'd model when available, otherwise the original."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    return compile_fn(model, **kwargs)


@contextmanager
def autocast_context(device_type: str = "cuda", dtype: torch.dtype = torch.float16):
    """AMP-friendly autocast context that no-ops when unavailable."""
    if torch.cuda.is_available() and device_type == "cuda":
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield
