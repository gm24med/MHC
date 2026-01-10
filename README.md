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

### 3-Line Integration

```python
import torch
from mhc import MHCSkip, HistoryBuffer

# 1. Initialize the layer and a history buffer
mhc_skip = MHCSkip(mode="mhc", max_history=4, constraint="simplex")
buffer = HistoryBuffer(max_history=4)

# 2. In your model loop
for layer in layers:
    x = layer(x)
    x = mhc_skip(x, buffer.get()) # Mix current state with history
    buffer.append(x)              # Update history
```

---

## 🧠 Core Concepts

### Why mHC?
Standard residual connections ($x_{l+1} = f(x_l) + x_l$) are a special case of Hyper-Connections where only the immediate previous state is used. `mHC` expands this by learning to mix a **sliding window** of past representations:

$$x_{l+1} = f_l(x_l) + \sum_{k=0}^{l} \alpha_{l,k} \, x_k$$

To prevent unstable amplification and preserve the identity-mapping property, `mhc` enforces constraints on the mixing weights $\alpha$:
- **Simplex**: Ensures weights are non-negative and sum to 1 (convex combination).
- **Identity-Preserving**: Guarantees a minimum weight on the most recent state.

---

## 🛠 Features

- **Drop-in Compatibility**: Works with any PyTorch module (Transformers, CNNs, MLPs).
- **Stability First**: Built-in identity-preservation and norm-bounding.
- **Modern Tooling**: Native support for `uv`, `pytest`, and `ruff`.

---

## 🧪 Development & Testing

```bash
uv run pytest
```

---

## 📜 Roadmap

- [x] **v0.1**: Core `MHCSkip` & `HistoryBuffer`
- [x] **v0.2**: Simplex & Identity Constraints
- [ ] **v0.3**: Doubly Stochastic (Matrix Mixing)
- [ ] **v0.4**: Deep Stability Benchmark Suite

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
