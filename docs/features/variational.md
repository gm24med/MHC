# Variational mHC

Variational mHC introduces stochasticity into the mixing weights, turning the hyper-connections into a **Bayesian-inspired ensemble** of historical states.

---

## Stochastic Mixing Math

When `stochastic=True` is enabled, the model samples mixing weights from a categorical distribution using the **Gumbel-Softmax trick**.

Sampling from a discrete categorical distribution is normally non-differentiable. Gumbel-Softmax provides a continuous, differentiable approximation:

$$\alpha_i = \frac{\exp((\log(\pi_i) + g_i) / \tau)}{\sum_{j=1}^H \exp((\log(\pi_j) + g_j) / \tau)}$$

Where:
- $\pi_i$ are the manifold-constrained mixing weights (probabilities).
- $g_i$ are independent samples drawn from the **Gumbel(0, 1)** distribution: $g = -\log(-\log(u))$ where $u \sim \text{Uniform}(0, 1)$.
- $\tau$ is the `temperature`.

### Sampling regimes:
- **Low Temperature ($\tau \to 0$)**: Samples become nearly one-hot (categorical). The model picks exactly one historical state to follow.
- **High Temperature ($\tau \to \dots$)**: Samples become uniform. The model averages history.

---

## Why use Variational Mixing?

1.  **Ensemble Robustness**: By sampling different history paths during training, the model acts as an ensemble. This prevents overfitting to specific layer connections.
2.  **Manifold Exploration**: Stochasticity forces the gradient to explore alternative historical connections that might be ignored by a greedy deterministic optimizer.
3.  **Uncertainty Quantification**: In inference, multiple stochastic passes can be used to measure the variance (uncertainty) of the manifold connections.

---

## Technical Behavior

### Training vs. Evaluation
Variational mixing respects the PyTorch module state:
- `model.train()`: **Stochastic**. Gumbel-Softmax is active.
- `model.eval()`: **Deterministic**. It uses the exact manifold projection $\pi_i$ without Gumbel noise.

### Parameters
Set these in `MHCSequential` or `MHCSkip`:
`stochastic=True`
- `temperature`: We recommend starting with `1.0` and annealing to `0.5` for sharper connections.
