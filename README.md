# mhc

A reference-grade PyTorch package for Manifold-Constrained Hyper-Connections (mHC).

## Installation
```bash
pip install -e .
```

## Quickstart
```python
import torch
from mhc import MHCSkip, HistoryBuffer

mhc = MHCSkip(mode="mhc", max_history=4, constraint="simplex", init="identity")
buf = HistoryBuffer(max_history=4)

x = torch.randn(2, 16, 64)
for _ in range(6):
    x = mhc(x, buf.get())
    buf.append(x)
```
