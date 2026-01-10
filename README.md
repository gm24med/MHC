# mhc — Manifold-Constrained Hyper-Connections

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Hyper-Connections (HC)** and **mHC** for stable, high-performance deep neural networks.

`mhc` is a reference-grade PyTorch implementation of Manifold-Constrained Hyper-Connections. It provides a drop-in replacement for standard residual connections, allowing for richer skip mixing across multiple previous states while preserving stability through geometric constraints.

---

## 🚀 Quickstart

### Installation

```bash
pip install -e .
# or with uv
uv add mhc
```

### High-Level Usage (Recommended)

Use `MHCSequential` to automatically manage history buffers without any manual loop logic:

```python
import torch.nn as nn
from mhc import MHCSequential

model = MHCSequential([
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
], max_history=4)

x = torch.randn(1, 64)
out = model(x) # Done! History is managed internally.
```

### Automatic Model Transformation

Inject Hyper-Connections into any existing model:

```python
from mhc import inject_mhc

model = my_existing_resnet()
inject_mhc(model, target_types=nn.Conv2d, max_history=4)
```

---

## 🧠 Core Concepts

### Why mHC?
Standard residual connections ($x_{l+1} = f(x_l) + x_l$) are a special case of Hyper-Connections where only the immediate previous state is used. `mHC` expands this by learning to mix a **sliding window** of past representations:

$$x_{l+1} = f_l(x_l) + \sum_{k=0}^{l} \alpha_{l,k} \, x_k$$

To prevent unstable amplification and preserve the identity-mapping property, `mhc` enforces constraints on the mixing weights $\alpha$:
- **Simplex**: Ensures weights are non-negative and sum to 1 (convex combination).
- **Identity-Preserving**: Guarantees a minimum weight on the most recent state.
- **Matrix Mixing**: Advanced historical feature mixing via doubly stochastic constraints.

---

## 🛠 Features

- **Transparent History**: Automated buffer management via `MHCSequential`.
- **Model Injection**: One-line transformation of existing models.
- **Stability First**: Built-in identity-preservation and norm-bounding.
- **Modern Tooling**: Native support for `uv`, `pytest`, `ruff`, and `mkdocs`.

---

## 🧪 Development & Testing

```bash
uv run pytest
```

---

## 📜 Roadmap

- [x] **v0.1**: Core `MHCSkip` & `HistoryBuffer`
- [x] **v0.2**: Simplex & Identity Constraints
- [x] **v0.3**: Matrix Mixing (Doubly Stochastic)
- [x] **v0.4**: Managed Layers & Model Injection
- [ ] **v0.5**: Deep Stability Benchmark Suite

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
