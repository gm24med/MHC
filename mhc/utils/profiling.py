import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn


class ForwardProfiler:
    """Lightweight forward-pass profiler using module hooks."""

    def __init__(
        self,
        module: nn.Module,
        filter_fn: Optional[Callable[[str, nn.Module], bool]] = None,
        use_cuda_sync: bool = False
    ) -> None:
        self.module = module
        self.filter_fn = filter_fn
        self.use_cuda_sync = use_cuda_sync
        self.times_ms: Dict[str, List[float]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._starts: Dict[str, float] = {}

    def _sync(self) -> None:
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self) -> None:
        def _pre_hook(name: str):
            def _hook(_module, _inputs):
                self._sync()
                self._starts[name] = time.perf_counter()
            return _hook

        def _post_hook(name: str):
            def _hook(_module, _inputs, _output):
                self._sync()
                start = self._starts.get(name)
                if start is None:
                    return
                duration_ms = (time.perf_counter() - start) * 1000.0
                self.times_ms.setdefault(name, []).append(duration_ms)
            return _hook

        for name, module in self.module.named_modules():
            if self.filter_fn is not None and not self.filter_fn(name, module):
                continue
            self._handles.append(module.register_forward_pre_hook(_pre_hook(name)))
            self._handles.append(module.register_forward_hook(_post_hook(name)))

    def stop(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def summary(self) -> Dict[str, float]:
        return {
            name: sum(times) / len(times)
            for name, times in self.times_ms.items()
            if times
        }
