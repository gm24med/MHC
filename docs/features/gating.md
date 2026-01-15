# Dynamic History Gating

Dynamic Gating allows the network to automatically modulate the total contribution of the historical states.

While Manifold Constraints ensure the *ratio* between historical states is stable, **Gating** determines how much of that history should be added to the current transformation at all.

---

## How it Works

When `use_gating=True` is enabled in `MHCSkip` or `MHCSequential`, a learnable scalar parameter $g$ (gate logit) is added to the layer:

$$x_{l+1} = f(x_l) + \sigma(g) \cdot \sum \alpha_k x_k$$

The sigmoid function $\sigma(g)$ constrains the gate value between 0 and 1.

### Learnable Flexibility
*   **During Early Training**: The network might keep $\sigma(g)$ high to allow a lot of feature reuse and stabilize gradients.
*   **During Late Training**: As the current transformation $f(x_l)$ becomes more specialized, the network might choose to lower the gate to reduce noise from very old states.

---

## Scientific Insight

By monitoring the gate values across layers (using the **Stability Dashboard**), you can identify which parts of your network are relying heavily on long-range skip connections and which are behaving more like local transformations.

## Usage

```python
from mhc import MHCSequential
import torch.nn as nn

model = MHCSequential(
    [nn.Linear(64, 64) for _ in range(10)],
    use_gating=True  # Enables learnable gating for all layers
)
```
