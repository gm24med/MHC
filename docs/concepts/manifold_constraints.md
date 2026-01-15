# Manifold Constraints: The Math of Stability

Unconstrained hyper-connectivity is mathematically unstable. Without geometric boundaries, the mixing weights $\alpha$ can grow arbitrarily, leading to activation values that hit the floating-point limit within a few layers (Exploding Gradients).

`mhc` solves this by introducing **Manifold Projections**. This guide explains the exact algorithms and why they are the "secret sauce" for Honey Badger stability.

---

## 1. Softmax vs. Euclidean Projection

Most historical attempts at "learned skips" use the **Softmax** function to normalize weights:
$$\alpha_i = \frac{e^{\mu_i / \tau}}{\sum e^{\mu_j / \tau}}$$

### Why Softmax Fails at Scale:

1.  **Infinite Tail**: Softmax can never produce a value of **exactly zero**. Every layer in history will always contribute *something*, even if it's just noise. In deep networks, this "noise accumulation" eventually washes out the signal.
2.  **Gradient Scaling**: The gradient of Softmax vanishes when the input logits are very different, making it nearly impossible for the model to "recover" or re-open a connection that was previously suppressed.

**The mHC Solution**: We use **Euclidean Projection** onto the $(H-1)$-simplex.

---

## 2. The Simplex Projection Algorithm

Given a vector of learned logits $\mu \in \mathbb{R}^H$, we find the vector $\alpha \in \mathbb{R}^H$ that is geometrically closest to $\mu$ while staying on the simplex:

$$\min_{\alpha} \frac{1}{2} \|\alpha - \mu\|^2 \quad \text{subject to} \quad \sum \alpha_i = 1, \alpha_i \geq 0$$

### The Internal "Solve" Trace:

`mhc` implements this using a high-performance $O(H \log H)$ optimization:

1.  **Sort** the logits $\mu$ in descending order: $u_1 \geq u_2 \geq \dots \geq u_H$.
2.  **Calculate** the cumulative sum of these sorted values.
3.  **Find** the optimal threshold $\rho$:
    $$\rho = \max \{ j \in [H] : u_j + \frac{1}{j} (1 - \sum_{i=1}^j u_i) > 0 \}$$
4.  **Compute** the Lagrange multiplier $\theta$:
    $$\theta = \frac{1}{\rho} (\sum_{i=1}^\rho u_i - 1)$$
5.  **Project**: $\alpha_i = \max(0, \mu_i - \theta)$.

### The Power of Structural Sparsity

Because of the `max(0, ...)` step, weights often hit the boundary of the manifold. This results in **Exact Zeros**.
- A weight of $0.05$ in Softmax is still a computational burden and a source of noise.
- A weight of $0.00$ in mHC is a **Physical Pruning**. The model effectively re-wires itself during training to find the most efficient information path.

---

## 3. The Identity-Preserving Manifold

When training networks with hundreds of layers, we must guarantee an "Identity backbone." We define a specialized manifold $\mathcal{M}_{\epsilon}$:

$$\mathcal{M}_{\epsilon} = \{ \alpha \in \Delta^{H-1} : \alpha_{latest} \geq \epsilon \}$$

### Multi-Stage Projection Logic:

1.  **Clamping**: We ensure the latest state weight is at least $\epsilon$ (default 0.1).
2.  **Budgeting**: The remaining $1 - \epsilon$ probability mass is distributed across the other $H-1$ historical states.
3.  **Stability**: This ensures that even if the model is exploring wildly, it is always at least "10% ResNet," preventing total signal loss during initial epochs.

---

## 4. Temperature & Entropy Dynamics

The `temperature` ($\tau$) parameter modifies the sharpness of the projection.

| Regime | $\tau$ Value | Resulting Manifold State |
| :--- | :--- | :--- |
| **Sharp** | $0.1 - 0.5$ | One history state dominates; others are strictly 0. |
| **Balanced** | $1.0$ | Distributed mixing with natural sparsity. |
| **Uniform** | $5.0+$ | All previous layers contribute equally ($1/H$). |

> [!TIP]
> Use a high temperature ($2.0$) for the first 5 epochs to encourage exploration, then decay to $1.0$ for stable convergence.

---

## 5. Differentiability: The Implicit Function Theorem

How do we backpropagate through a sorting algorithm and a max operator?

`mhc` projections are **piecewise differentiable**. Although the sorting operation itself has no gradient, the resulting indices are fixed in the locally linear region of the projection. We utilize the **Implicit Function Theorem** to compute the exact analytical gradient from $\frac{\partial \mathcal{L}}{\partial \alpha}$ to $\frac{\partial \mathcal{L}}{\partial \mu}$.

This ensures that your optimizer (Adam, SGD, etc.) sees a smooth, "Honey Badger tough" loss landscape, allowing for incredibly fast convergence.
