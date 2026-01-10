# mhc: Manifold-Constrained Hyper-Connections

Welcome to the documentation for `mhc`, a reference-grade PyTorch implementation of Manifold-Constrained Hyper-Connections.

## Overview

Hyper-Connections (HC) and mHC provide a way to mix historical network states to improve feature representation and training stability.

- **Stable mixing**: Using geometric constraints like Simplex and Identity-preservation.
- **Easy integration**: Use `MHCSequential` or `inject_mhc` for transparent usage.
- **Advanced features**: Support for Matrix Mixing and Doubly Stochastic constraints.

## TensorFlow (Optional)

Install the extra and use the TensorFlow layers:

```bash
pip install "mhc[tf]"
```

```python
from mhc.tf import TFMHCSequential
```

## Visualization (Optional)

```bash
pip install "mhc[viz]"
```

## Quickstart

```python
from mhc import MHCSequential
import torch.nn as nn

# Wrap your layers
model = MHCSequential([
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64)
], max_history=4)

x = torch.randn(1, 64)
out = model(x)
```
