<div align="center">


<img src="docs/images/logo.png" alt="mHC Logo" width="200"/>

# mhc — Manifold-Constrained Hyper-Connections

**Honey Badger Stability for Deeper, More Stable Neural Networks.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/gm24med/MHC/actions/workflows/ci.yml/badge.svg?style=for-the-badge)](https://github.com/gm24med/MHC/actions/workflows/ci.yml)

[Documentation](https://github.com/gm24med/MHC/tree/main/docs) • [Examples](https://github.com/gm24med/MHC/tree/main/examples) • [Paper](#citation) • [Contributing](#contributing)

</div>

---

## 🎯 What is mHC?

`mhc` is a next-generation PyTorch library that reimagines residual connections. Instead of simple one-to-one skips, mHC learns to dynamically mix multiple historical network states through geometrically constrained manifolds—bringing **Honey Badger toughness** to your gradients.

<div align="center">

| 🚀 **High Performance** | 🧠 **Smart Memory** | 🛠️ **Drop-in Ease** |
|:---:|:---:|:---:|
| Reach deeper than ever before with optimized gradient flow. | Dynamically mix past states for richer feature representation. | Transform any model to mHC with a single line of code. |

</div>

<br/>

<div align="center">
<img src="docs/images/architecture.png" alt="mHC Architecture" width="800"/>
</div>

---

## 🚀 Quick Start

### Installation

```bash
# With uv (recommended)
uv pip install -e .

# Or standard pip
pip install -e .
```

### 30-Second Example

```python
import torch
from mhc import MHCSequential

# Create a model with mHC skip connections
model = MHCSequential([
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
], max_history=4, mode="mhc", constraint="simplex")

# Use it like any PyTorch model
x = torch.randn(8, 64)
output = model(x)
```

### Inject into Existing Models

Transform any model to use mHC with one line:

```python
from mhc import inject_mhc
import torchvision.models as models

model = models.resnet50(pretrained=True)
inject_mhc(model, target_types=nn.Conv2d, max_history=4)
```

---

## 🤔 Why mHC?

### The Gradient Bottleneck
Standard residual connections only look one step back: $x_{l+1} = x_l + f(x_l)$. While revolutionary, this narrow window limits the network's ability to leverage long-range dependencies and can lead to diminishing returns in extremely deep architectures.

### The mHC Breakthrough
mHC implements a **history-aware manifold** that mixes a sliding window of $H$ past representations:

$$
x_{l+1} = f(x_l) + \sum_{k=l-H+1}^{l} \alpha_{l,k}\, x_k
$$

Where:
- **$\alpha_{l,k}$**: Learned mixing weights optimized for feature relevance.
- **Constraints**: Weights are projected onto stable manifolds (Simplex, Identity-preserving, or Doubly Stochastic) to ensure mathematical convergence.

### Key Advantages

| Benefit | Description |
|:---|:---|
| **Deep Stability** | Geometric constraints prevent gradient explosion even at 200+ layers. |
| **Feature Fusion** | Multi-history mixing allows layers to recover lost spatial or semantic info. |
| **Adaptive Flow** | The network learns *which* historical states are most important for the current layer. |

---

## 📊 Visualizing Results

<div align="center">

| **Mixing Weights** | **Gradient Flow** |
|:---:|:---:|
| ![Mixing Weights](mixing_weights.png) | ![Gradient Flow](gradient_flow.png) |
| *Learned coefficients over time* | *Improved backpropagation signal* |

</div>

---

## 🌳 Repository Structure

```text
MHC/
├── mhc/                     # Core package
│   ├── constraints/         # Mathematical projections
│   ├── layers/              # mHC Skip implementations
│   ├── tf/                  # TensorFlow compatibility
│   └── utils/               # Injection & logging tools
├── tests/                   # Robust PyTest suite
├── docs/                    # Documentation sources
└── examples/                # Quick-start notebooks
```

---

## 📜 Roadmap

- [x] **v0.4**: Managed Layers & Model Injection
- [ ] **v0.5**: Deep Stability Benchmark Suite
- [ ] **v0.6**: PyPI Release & Documentation Site
- [ ] **v0.7**: PyTorch Lightning & Hugging Face Integration
- [ ] **v0.8**: Pre-trained Model Zoo

---

<div align="center">

### **⭐ Star us on GitHub!**

[Report Bug](https://github.com/gm24med/MHC/issues) • [Request Feature](https://github.com/gm24med/MHC/issues) • [Discussions](https://github.com/gm24med/MHC/discussions)

</div>
