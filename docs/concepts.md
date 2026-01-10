# Core Concepts

## Hyper-Connections (HC)

Standard residual connections are defined as:
$$x_{l+1} = f(x_l) + x_l$$

Hyper-Connections generalize this by mixing multiple previous states:
$$x_{l+1} = f_l(x_l) + \sum_{k=0}^{l} \alpha_{l,k} \, x_k$$

## Manifold-Constrained Hyper-Connections (mHC)

To ensure stability, `mHC` enforces geometric constraints on the mixing weights $\alpha$:

### 1. Simplex Constraint
Ensures that the weights form a convex combination:
- $\alpha_k \ge 0$
- $\sum \alpha_k = 1$

### 2. Identity-Preserving Constraint
Ensures that the latest state receives at least a minimum weight $\epsilon$:
- $\alpha_{last} \ge \epsilon$
- $\sum \alpha_k = 1, \alpha_k \ge 0$

### 3. Matrix Mixing
Uses a learnable matrix instead of scalar weights, allowing for more complex historical feature mixing.

## Shape Compatibility

All states in a mixing window must share compatible shapes. `MHCSkip` can optionally
project limited mismatches with `auto_project=True`, but it is not a general adapter.

## Shape Compatibility

All states in a mixing window must share compatible shapes. `MHCSkip` can optionally
project limited mismatches with `auto_project=True`, but it is not a general adapter.
