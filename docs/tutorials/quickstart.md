# 🚀 10-Minute Quickstart

Welcome to **mhc**! This guide will get you up and running with Manifold-Constrained Hyper-Connections in minutes. We'll cover installation, core concepts, and how to upgrade your existing models to use "Honey Badger" stability.

---

## 1. Installation

`mhc` is a lightweight library with zero heavy dependencies outside of PyTorch. We recommend using `uv` for speed, but `pip` works perfectly.

### Recommended (Advanced)
```bash
# Install with all developer extras (including visualization and dashboards)
uv pip install "mhc[all]"
```

### Standard
```bash
# Using uv
uv pip install mhc

# Using standard pip
pip install mhc
```

---

## 2. The Core Concept: Managed Sequential

The easiest way to use mHC is via `MHCSequential`. It’s a drop-in replacement for `nn.Sequential` that automatically manages your historical states, projections, and device placement.

```python
import torch
import torch.nn as nn
from mhc import MHCSequential

# 1. Define your standard layers
layers = [
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU()
] * 10 # Creates a 40-layer block

# 2. Wrap it in MHCSequential - Honey Badger Style
model = MHCSequential(
    layers,
    max_history=4,        # Each layer "sees" the last 4 network states
    mode="mhc",           # Enable Manifold Constraints
    constraint="identity" # Guarantee the core ResNet backbone
)

# 3. Use it like a normal PyTorch model
x = torch.randn(8, 128)
output = model(x) # History is managed automatically behind the scenes!
```

---

## 3. Injecting into Pre-trained Models

You don't need to rebuild your models from scratch. Use `inject_mhc` to surgically upgrade standard architectures (like those from `torchvision` or `transformers`) to manifold-aware versions.

```python
from mhc import inject_mhc
import torchvision.models as models

# Load a standard ResNet-50
model = models.resnet50(weights="DEFAULT")

# Target all Conv2d layers and add hyper-connections
# This transforms the ResNet-50 into a "Hyper-ResNet"
inject_mhc(model, target_types=torch.nn.Conv2d, max_history=4)

# Your model now learns weighted skips across multiple previous states
# while preserving its original pretrained weights!
```

---

## 4. Advanced Stability: Stochastic Mixing

For research and extreme architecture robustness, try **Variational mHC**. This uses the Gumbel-Softmax distribution to sample connections during training.

```python
model = MHCSequential(
    layers,
    stochastic=True,    # Enable Variational mixing
    temperature=0.5     # Control the "sharpness" of the choice
)

# During training: States are sampled stochastically
model.train()

# During evaluation: The pass becomes deterministic (using expected values)
model.eval()
```

---

## 5. Verification: The Stability Check

To ensure your mHC model is correctly configured, run the built-in sanity check:

```python
from mhc.utils import check_model_stability

# Verifies gradient flow and history window alignment
status = check_model_stability(model)
print(f"Model Integrity: {status}") # Should be 'EXCELLENT'
```

---

## 🚦 Next Steps

Ready for more? Explore the deeper mechanics:

*   🧗 **[Basic Usage Guide](basic_usage.md)**: Explore parameters like `auto_project` and `detach_history`.
*   🧠 **[Manifold Constraints](../concepts/manifold_constraints.md)**: Understand the math that prevents exploding activations.
*   ⚡ **[Lightning Integration](../guides/pytorch_lightning.md)**: Monitor your mixing weights in real-time.
*   🔭 **[Computer Vision Guide](../guides/vision.md)**: Specific tips for `MHCConv2d` and CNNs.
