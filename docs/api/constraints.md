# API Reference: Geometric Constraints

The `mhc.constraints` module provides the mathematical functions that ensure the structural stability of the Honey Badger architecture. These functions are the core implementation of mHC's manifold projections and are exposed publicly for researchers to building custom skip-connection architectures.

---

## `project_simplex()`

Projects an arbitrary vector of logits onto the $(H-1)$ unit simplex. This is the bedrock of the "Manifold-Constrained" part of our library.

### Function Signature:

```python
def project_simplex(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
```

### Arguments:

-   **`logits`**:
    A 1D or 2D tensor of raw unnormalized scores. If 2D, the projection is applied across the last dimension (useful for batch-wise manifold processing).
-   **`temperature`**:
    A scaling factor applied to the logits before projection. A temperature near 0 makes the resulting weights almost one-hot (maximum sparsity), while a high temperature (e.g., 5.0) makes the distribution uniform ($1/H$).

### Mathematical Execution Trace:

Given internal scores $\mu$, we solve for the optimally sparse mixing vector $\mathbf{\alpha}$:

$$\min_{\alpha} \sum_{i=1}^n (\alpha_i - \mu_i)^2 \quad \text{subject to} \quad \sum \alpha_i = 1, \alpha_i \ge 0$$

This is computed using a high-performance sorting algorithm in $O(H \log H)$ time. Unlike Softmax, this projection guarantees that weights hit the boundary, allowing for **Exact Zero** historical contributions.

---

## `project_identity_preserving()`

The "Stability Backbone" constraint. It guarantees that the forward signal is never lost by enforcing a minimum weight "floor" on the most recent history state.

### Function Signature:

```python
def project_identity_preserving(
    logits: torch.Tensor,
    epsilon: float = 0.1,
    temperature: float = 1.0
) -> torch.Tensor:
```

### Arguments:

-   **`logits`**:
    Input scores for the historical window.
-   **`epsilon`**:
    The stability floor. A value of `0.1` ensures that even if all other history is preferred, the latest state still carries at least 10% of the signal energy.

### Internal Logic:

1.  **Isolation**: The projection isolates the last element (the most recent state $x_l$).
2.  **Clamping**: This element is clamped to the range $[\epsilon, 1.0]$.
3.  **Budgeting**: The remaining energy budget ($1 - \alpha_{latest}$) is then calculated.
4.  **Distribution**: The older historical logits are projected onto a simplex scaled to fit into the remaining budget.

---

## `project_doubly_stochastic()`

The engine for **Matrix mHC**. This constraint is used when mixing across history **and** channels simultaneously.

### Function Signature:

```python
def project_doubly_stochastic(
    logits: torch.Tensor,
    iterations: int = 10,
    temperature: float = 1.0
) -> torch.Tensor:
```

### The Sinkhorn-Knopp Algorithm:

Matrix mixing replaces scalar $\alpha$ values with a learnable mixing matrix $W \in \mathbb{R}^{C \times C \times H}$. To preserve stability, this matrix must be **Doubly Stochastic** (all rows and columns sum to 1).

**Why this matters**:
-   Ensures every output channel receives a normalized amount of historical energy.
-   Ensures every input channel contributes a normalized amount of importance.

We achieve this via the iterative **Sinkhorn-Knopp** procedure.
-   **`iterations`**: Reaching a tolerance of $1e-6$ usually takes 5-10 iterations.
-   **Auto-Grad**: The entire process is fully differentiable using the bi-diagonal gradient properties of the Sinkhorn operator.
