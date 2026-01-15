# Auto-Sparsification (Pruning)

One of the unique advantages of Euclidean projections (Manifold Constraints) over standard Softmax is that they can push weights **exactly to zero**.

**Auto-Sparsification** leverages this to prune "useless" historical connections dynamically during training.

---

## The `prune_threshold`

By setting a `prune_threshold`, you can tell the model to ignore any historical contribution that is deemed insignificant by the learned manifold weights.

```python
model = MHCSequential(
    modules,
    prune_threshold=0.01  # Ignore any state with <1% weight
)
```

### Why use Pruning?

1.  **Computational Efficiency**: When a weight $\alpha_k$ is zero or very small, skipping the multiplication and addition $x + \alpha_k \cdot x_k$ saves FLOPs.
2.  **Noise Reduction**: By pruning very low weights, the network focuses only on the most informative historical states.
3.  **Model Compression**: In the future, this allows for permanent pruning of connections to create a sparse, high-speed inference model.

---

## Dynamic Density

Unlike static pruning, mHC pruning is **dynamic**. A connection that is pruned at epoch 10 might become relevant again at epoch 50 if its gradient signals start to rise. The manifold constraints will naturally "awake" the connection if it improves the objective.

> [!NOTE]
> Training with `prune_threshold > 0` and a low `temperature` is the most effective way to obtain sparse hyper-connectivity.
