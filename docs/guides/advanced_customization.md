# Advanced Customization & Research

`mhc` is not just a library of ready-made layers; it is a research platform. This guide explains how to extend `mhc` with custom manifold constraints, manual buffer management, and non-linear history mixing.

---

## 1. Implementing Custom Manifolds

While `simplex` and `identity` cover 90% of use cases, researchers may want to explore other geometries. For example, a **Hyperspherical Constraint** or a **Laplacian Sparsity** manifold.

To implement a custom constraint, follow this pattern:

```python
import torch

def my_custom_manifold(logits, temperature=1.0):
    # 1. Scale by temperature
    mu = logits / temperature

    # 2. Apply custom geometry (e.g. projection onto a unit sphere)
    norm = torch.norm(mu, p=2, dim=-1, keepdim=True)
    alpha = mu / (norm + 1e-8)

    return alpha

# Usage in mHC layers
layer = MHCSkip(max_history=4)
# In your training loop, manually apply the projection
alpha = my_custom_manifold(layer.logits)
```

---

## 2. Manual History Buffer Management

In certain architectures (like Recurrent Vision Transformers), you might need to manage history across disconnected modules. You can use the `HistoryBuffer` class directly:

```python
from mhc.layers.managed import HistoryBuffer

# Initialize a buffer for 512-channel feature maps
buffer = HistoryBuffer(max_history=8)

# During your forward pass
x_hidden = my_layer(x)
buffer.append(x_hidden)

# Pull the mixed state for another branch
mixed_history = buffer.get_mixed(weights=my_learned_weights)
```

---

## 3. Non-Linear History Mixing

Standard mHC uses weighted summation (Linear Mixing). High-end research often explores **Attention-based Mixing**.

Instead of a simple vector $\alpha$, you can treat history as a "Key-Value" store:
1.  **Keys**: Learned projections of historical states.
2.  **Values**: The historical states themselves.
3.  **Query**: The current state $x_l$.

This turns `mHC` into a **Local Temporal Attention** mechanism. While more compute-heavy, it allows the model to "attend" to specific past events in a context-dependent way.

---

## 4. Spectral Normalization for Manifolds

If you are training **Generative Adversarial Networks (GANs)** with mHC, we recommend applying **Spectral Normalization** to the `auto_project` matrices. This prevents the history-projection path from becoming a source of discriminator instability.

```python
from torch.nn.utils import spectral_norm
from mhc import MHCSequential

model = MHCSequential(..., auto_project=True)

# Apply SN to the internal projectors
for m in model.modules():
    if isinstance(m, nn.Linear) and hasattr(m, 'weight'):
        spectral_norm(m)
```

---

## 5. Information Theory: Measuring Feature Reuse

To quantify how much your model is benefiting from mHC, we recommend tracking the **Effective Signal Depth (ESD)**:

$$ESD = \sum_{k=1}^H \alpha_k \cdot (l - k)$$

Where $l$ is the current layer depth.
-   An $ESD$ close to 0 means the model is ignoring history.
-   A high $ESD$ means the model is effectively "re-shaping" itself into a much shallower, more robust ensemble.
