# Basic Usage Guide

This guide dives into the fundamental building blocks of `mhc` and explaining how to configure your skips for different scenarios. We cover both the automated and manual approaches to hyper-connectivity.

---

## The two ways to use mHC

### 1. The Managed Way (`MHCSequential`)

This is the recommended approach for 99% of use cases. It works as a drop-in replacement for `nn.Sequential` and handles the entire history lifecycle (retrieval, mixing, and eviction) automatically.

```python
from mhc import MHCSequential
import torch.nn as nn

# All layers between these Linear blocks will have hyper-connections
model = MHCSequential(
    [nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)],
    max_history=4,
    mode="mhc"
)
```

### 2. The Granular Way (`MHCSkip`)

Best for complex architectures (multiple branches, U-Nets, or loops) where you want to manually decide where history is injected.

```python
from mhc import MHCSkip

class MyCustomBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. Instantiate the atomic mixing node
        self.skip = MHCSkip(max_history=4)
        self.layer = nn.Linear(dim, dim)

    def forward(self, x, history=None):
        # 2. Run the main transformation
        out = self.layer(x)

        # 3. Mix current output with the provided list of history tensors
        # History must be a List[torch.Tensor]
        out = self.skip(out, history or [])

        return out
```

---

## Critical Parameters

### `max_history` (integer)

The size of the sliding window.

*   **1**: Behaves like a standard ResNet (only mixes the latest state).
*   **4-5**: **The Sweet Spot**. Provides significant stability gains with minimal VRAM overhead.
*   **8+**: For extremely deep research models (>500 layers) where long-range feature reuse is critical for avoiding signal "washing."

### `mode` (string)

This defines the mathematical engine used for mixing.

*   `"mhc"`: **Manifold-Constrained**. The primary mode. Uses geometric projections (Simplex) to ensure numerical stability at any depth.
*   `"hc"`: Standard **Hyper-Connections**. Uses Softmax for mixing. Highly flexible but can occasionally suffer from "tail noise" in ultra-deep networks.
*   `"residual"`: **Identity Skip**. Disable all learnable mixing and revert to a standard ResNet. Useful for A/B testing your architecture.

### `constraint` (string) - *only for mode="mhc"*

*   `"simplex"`: Weights are forced to sum to 1 and be non-negative. Guaranteed to conserve signal energy.
*   `"identity"`: Simplex + a minimum weight (`epsilon`) on the most recent state. This ensures the model never "disconnects" from the immediate sequential signal.

### `detach_history` (boolean)

A vital performance lever.

*   **`True`**: Treats historical tensors as constant inputs. Gathers gradients for mixing weights but **stops** gradients from flowing backwards into the historical window. Saves $\approx$ 50% VRAM.
*   **`False`**: Full end-to-end backpropagation through the entire history graph. Mathematically perfect but much more memory-intensive.

---

## Shape Matching with `auto_project`

In a standard ResNet, if you want a skip connection between a 64-channel layer and a 128-channel layer, you must manually add a 1x1 convolution.

**mHC handles this for you.** By setting `auto_project=True`, the mixing node will detect dimension mismatches (channel counts or spatial resolutions) and dynamically learn a small projection matrix to align them.

```python
# Automatically learns to project history tensors to match current layer
skip = MHCSkip(max_history=4, auto_project=True)
```

---

## 🧼 Best Practice: Manual Clearing

If you are using `MHCSkip` manually (not inside `MHCSequential`), you must remember to clear your history between training episodes:

```python
# Before starting a new sequence or batch
model.clear_history()
```

If you don't clear history, the gradients from "Sample A" might accidentally influence the history of "Sample B," leading to training divergence.
