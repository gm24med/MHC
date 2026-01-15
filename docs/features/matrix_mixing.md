# Matrix mHC (Experimental)

While standard mHC uses **scalar** mixing weights ($\alpha_k \in \mathbb{R}$), **Matrix mHC** uses full **mixing matrices** ($W_k \in \mathbb{R}^{C \times C}$). This allows the model to mix history not just across steps, but selectively across specific feature dimensions.

---

## 🏗️ How it Works

Instead of the scalar-vector multiplication:
$$x_{l+1} = \hat{x}_l + \sum \alpha_k \cdot x_k$$

Matrix mHC performs a matrix-vector product for each historical state:
$$x_{l+1} = \hat{x}_l + \sum W_k \cdot x_k$$

### Doubly Stochastic Constraint
To maintain the same stability guarantees as standard mHC, the mixing matrix $W$ is projected onto the set of **Doubly Stochastic Matrices**.

A matrix $W$ is doubly stochastic if:
1.  All elements are non-negative: $W_{ij} \geq 0$
2.  All rows sum to 1: $\sum_{j} W_{ij} = 1$
3.  All columns sum to 1: $\sum_{i} W_{ij} = 1$

This is enforced using the **Sinkhorn-Knopp algorithm**, which iteratively normalizes the rows and columns until convergence.

---

## 🔬 Scientific Benefit

Matrix Mixing is particularly powerful when:
- **Feature Channels are Heterogeneous**: Different channels in your network carry fundamentally different information (e.g. low-frequency vs high-frequency).
- **Complex Feature Reuse**: A layer at depth 50 might need "Channel 5" from Layer 10 but "Channel 12" from Layer 20. Matrix mHC can learn this mapping, whereas scalar mHC is forced to take all channels or none.

---

## Usage

Matrix Mixing is currently available via the `MatrixMHCSkip` layer.

```python
from mhc.layers import MatrixMHCSkip
import torch.nn as nn

# dim=64 channels
skip = MatrixMHCSkip(dim=64, max_history=4)

# In the forward pass
output = layer(input)
mixed = skip(output, history_list)
```

> [!WARNING]
> Matrix mHC has significantly higher computational and parameter overhead ($O(H \times C^2)$ parameters). It is recommended for research into small, high-capacity models.
