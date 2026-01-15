# Mathematical Derivation of mHC Stability

This section is for researchers who want to see the formal proofs of how mHC conserves energy and allows for nearly infinite depth without explosion.

---

## 1. Energy Conservation on the Simplex

Let $x_l$ be the activation vector at layer $l$. In a naive additive skip connection:
$$x_{l+1} = \mathcal{F}(x_l) + \sum_{k=0}^l x_k$$
The variance $\text{Var}(x_{l+1})$ grows linearly with depth $l$, eventually leading to activation drift.

**mHC proof of boundedness:**
If we enforce the Simplex constraint $\sum \alpha_k = 1, \alpha_k \ge 0$, the historical contribution is a **Convex Combination**:
$$h_l = \sum \alpha_k x_k$$
By Jensen's Inequality, if $\|x_k\| \le M$ for all previous states, then:
$$\|h_l\| = \|\sum \alpha_k x_k\| \le \sum \alpha_k \|x_k\| \le \sum \alpha_k M = M$$
Thus, mHC ensures that the "Skip Signal" **never exceeds the maximum energy of its constituents**.

---

## 2. Gradient Path Analysis

In a ResNet, the gradient $\frac{\partial \mathcal{L}}{\partial x_0}$ is computed through a single chain:
$$\frac{\partial \mathcal{L}}{\partial x_0} = \prod_{l=0}^D \left( \mathbf{I} + \frac{\partial \mathcal{F}_l}{\partial x_l} \right) \frac{\partial \mathcal{L}}{\partial x_D}$$

In mHC, the gradient is the sum of all possible paths through the history buffer. For a history window $H$, the number of directed acyclic paths grows by a factor related to the window size, creating a **dense multi-path highway**.

This effectively "averages out" the noise of any single unstable transformation $\mathcal{F}_q$, because the gradient can still reach the weights via $H-1$ other stable paths in the manifold.

---

## 3. The Implicit Function Theorem in Training

The projection $P: \mathbb{R}^H \to \Delta^{H-1}$ is a non-linear but continuous and piecewise-differentiable operator. During backpropagation, we must compute:
$$\frac{\partial P(\mu)}{\partial \mu}$$

Since $P$ is a projection onto a convex set, its Jacobian is well-defined almost everywhere.
- When a connection is active ($\alpha_k > 0$), the gradient acts like a standard identity.
- When a connection is pruned ($\alpha_k = 0$), the manifold logic prevents "ghost updates" to that connection until the gradient on its logit $\mu_k$ is strong enough to push it back onto the manifold boundary.

This leads to **Structural Sparsity Stability**: The network doesn't just learn weights; it learns to evolve its own connectivity graph on the fly.
