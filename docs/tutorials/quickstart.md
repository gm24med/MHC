# 🚀 10-Minute Quickstart

Welcome to **mhc**! This guide will get you up and running with Manifold-Constrained Hyper-Connections in minutes.

## 1. Installation

Install via `uv` (recommended) or `pip`:

```bash
uv pip install mhc
# or
pip install mhc
```

## 2. The Core Concept: Managed Sequential

The easiest way to use mHC is via `MHCSequential`. It’s a drop-in replacement for `nn.Sequential` that automatically manages your historical states.

```python
import torch
import torch.nn as nn
from mhc import MHCSequential

# Create a deep network (e.g., 20 layers)
layers = []
for _ in range(20):
    layers.append(nn.Linear(128, 128))
    layers.append(nn.ReLU())

# Wrap it in MHCSequential
model = MHCSequential(
    layers,
    max_history=4,     # Look back 4 states
    mode="mhc",        # Use manifold constraints
    constraint="identity" # Ensure stability
)

x = torch.randn(8, 128)
output = model(x) # Done! History is managed behind the scenes.
```

## 3. Injecting into Existing Models

Already have a model (ResNet, Transformer, Bert)? Inject mHC with one line:

```python
from mhc import inject_mhc
import torchvision.models as models

model = models.resnet50()

# Targets all Conv2d layers and adds hyper-connections
inject_mhc(model, target_types=nn.Conv2d, max_history=4)

# Your model now learns weighted skips across multiple previous states!
```

## 4. Why Use It?

Standard ResNets only see the immediate previous state. **mHC** sees a sliding window of the past, allowing:
- **Resilient Gradients**: Geometric constraints prevent vanishing/exploding gradients in extremely deep models.
- **Feature Reuse**: Deep layers can directly access "younger" features from much earlier in the network.

---

### Next Steps
- Check out the [Framework Integrations](../guides/frameworks.md) for PyTorch Lightning support.
- Explore [Core Concepts](../concepts.md) for the math behind the manifolds.
