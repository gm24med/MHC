````markdown
# mhc — Technical Design Document (Reference Implementation Package)

> **Goal:** Build a *reference-grade*, **easy-to-use**, **well-tested**, and **research-faithful** PyTorch package implementing **Hyper-Connections (HC)** and **Manifold-Constrained Hyper-Connections (mHC)**, so that *anyone* can install, integrate, and reproduce meaningful experiments with minimal effort.

---

## Table of Contents
1. [Overview](#overview)
2. [Objectives and Non-Goals](#objectives-and-non-goals)
3. [User Personas and UX Requirements](#user-personas-and-ux-requirements)
4. [Core Concepts](#core-concepts)
5. [API Design](#api-design)
6. [Architecture and Modules](#architecture-and-modules)
7. [Constraints and Projections](#constraints-and-projections)
8. [Initialization Strategy](#initialization-strategy)
9. [Forward/Backward Behavior](#forwardbackward-behavior)
10. [Integration Patterns](#integration-patterns)
11. [Experiments and Reproducibility Suite](#experiments-and-reproducibility-suite)
12. [Testing and Quality Gates](#testing-and-quality-gates)
13. [Performance Considerations](#performance-considerations)
14. [Packaging and Distribution](#packaging-and-distribution)
15. [Documentation Standards](#documentation-standards)
16. [Roadmap](#roadmap)
17. [Risks and Mitigations](#risks-and-mitigations)
18. [Appendix: Suggested Repo Layout](#appendix-suggested-repo-layout)

---

## Overview

The `mhc` package provides a clean PyTorch implementation of:
- **Residual connections** (baseline)
- **Hyper-Connections (HC)**: richer skip mixing across multiple previous states
- **mHC**: HC **with manifold/geometry constraints** applied to the mixing weights to preserve stability/identity properties

The package is designed to be:
- **Drop-in**: integrate into existing models without rewriting them
- **Trustworthy**: tests validate identity-preservation and stability properties
- **Reproducible**: includes a minimal benchmark suite with fixed configs + seeds
- **Extensible**: add new constraints/manifolds without touching core layers

---

## Objectives and Non-Goals

### Objectives
1. **Simple install**
   - `pip install mhc` works without custom compilation.
2. **3-line integration**
   - A user can add mHC to a model with minimal code changes.
3. **Ablation switch**
   - Swap among `residual`, `hc`, `mhc` via a flag.
4. **Reference-grade correctness**
   - Clear math mapping + deterministic projection behavior + unit tests.
5. **Reproducible demo results**
   - At least one strong experiment where mHC has a clear effect.
6. **Good defaults**
   - Defaults should almost never destabilize training.

### Non-Goals
- Not a transformer framework
- Not a distributed training system
- Not a benchmark leaderboard product
- Not implementing every possible variant from future paper revisions

---

## User Personas and UX Requirements

### Persona A: Student / Beginner
- Wants to run an example and see the effect quickly.
- Needs simple docs and minimal knobs.

**UX needs**
- “Quickstart” runs in <10 minutes on CPU/GPU.
- Clear examples, small models, simple configs.

### Persona B: Researcher
- Wants to compare residual vs HC vs mHC fairly.
- Needs stability/identity tests and reproducible scripts.

**UX needs**
- Deterministic configs, seeds, clean ablations.
- Easy to plug into existing architectures.

### Persona C: Engineer
- Wants a clean module, minimal overhead.
- Needs stable API and consistent behavior.

**UX needs**
- Clear constraints on inputs/outputs, strong typing, tests.

---

## Core Concepts

### Notation
- Let `x_l` denote representation at layer `l`.
- Let `f_l(·)` denote the transformation of layer `l` (attention/MLP/conv/etc).
- Let `history = [x_0, x_1, ..., x_l]`.

### Baseline Residual
A standard residual block:
\[
x_{l+1} = f_l(x_l) + x_l
\]

### Hyper-Connections (HC)
Instead of adding only `x_l`, combine multiple previous states:
\[
x_{l+1} = f_l(x_l) + \sum_{k=0}^{l} \alpha_{l,k} \, x_k
\]
- `α_{l,k}` are learnable mixing weights.

### mHC (Manifold-Constrained HC)
mHC constrains the mixing weights to lie on a stable set/manifold `𝓜`:
\[
\alpha_l \in \mathcal{M}
\]
Examples of constraints:
- **Simplex**: `α ≥ 0` and `sum(α)=1`  (convex combination)
- **Doubly stochastic** (matrix mixing variant)
- **Identity-preserving**: weight on `x_l` is bounded below

**Key idea:** constraints preserve identity mapping behavior and prevent unstable amplification.

---

## API Design

### High-level design rules
- **One obvious entry point** for most users.
- Advanced features exist but are not required to start.
- Behavior is consistent across tensor shapes (token sequences, images, etc).

### Primary User-Facing API

#### `MHCSkip`
A module that combines current state with a bounded number of previous states.

```python
from mhc import MHCSkip

mhc = MHCSkip(
    mode="mhc",                 # "residual" | "hc" | "mhc"
    max_history=4,              # use last K states from history
    constraint="simplex",       # "simplex" | "identity" | "doubly_stochastic" (optional)
    temperature=1.0,            # softmax / normalization temperature
    init="identity",            # identity-friendly init
    projection="forward",       # "forward" | "post_step" | "none"
)
````

##### Forward signature

```python
y = mhc(x, history)
```

* `x`: current tensor (shape-agnostic; last dim is feature dim)
* `history`: list of prior tensors with same shape as `x`
* returns `y`: combined tensor with same shape as `x`

---

### Optional Convenience API

#### `HistoryBuffer`

Helps manage history in a safe, memory-bounded way.

```python
from mhc import HistoryBuffer

buf = HistoryBuffer(max_history=4, detach_history=False)
buf.append(x)
hist = buf.get()  # list of tensors
```

#### `wrap_model` (optional)

A wrapper utility to inject mHC into certain known architectures (best effort).
This is **not required** for v0.1, but can be included if stable.

---

## Architecture and Modules

### Design principles

* Separate **math** (constraints/projections) from **layers**.
* Make projection implementations independently testable.
* Avoid tight coupling with transformers/vision libraries.

### Key modules

1. `mhc/layers/`

   * `MHCSkip`: main mixing module
   * `ResidualSkip`: baseline identity behavior
   * Optional: `MHCBlock` (if we want block-level integration)

2. `mhc/constraints/`

   * Constraint definitions
   * Configuration parsing and validation

3. `mhc/manifolds/`

   * Projection algorithms (simplex, Sinkhorn, etc.)
   * Numerically stable implementations

4. `mhc/utils/`

   * Shape helpers (flatten/unflatten token dims)
   * Determinism helpers
   * Logging hooks

5. `mhc/experiments/`

   * Minimal reproducible scripts (one killer experiment + a tiny smoke test)

6. `mhc/tests/`

   * Unit tests
   * Property tests (identity, norm bounds, gradient sanity)

---

## Constraints and Projections

### Design choice: vector mixing vs matrix mixing

#### v0.1 recommendation: **vector mixing**

* `α` is a vector of size `K` (history length).
* Output is weighted sum of `K` previous states.

Pros:

* Simple
* Efficient
* Easy to test
* Good for adoption

#### Optional extension: **matrix mixing**

* Mix across features (more expressive, more complex).
* Can be added later once vector mixing is stable.

---

### Constraint: Simplex (recommended default)

* Enforce `α ≥ 0` and `sum(α)=1`.
* Ensures output is a convex combination of past states.
* Strong stability behavior.

Implementation options:

* Parameterize `α = softmax(z / T)` (simple, differentiable).
* Or hard projection onto simplex (more exact; can be optional).

**Recommended v0.1 default:** softmax parameterization (stable + simple).

---

### Constraint: Identity-preserving (recommended fallback)

Guarantee at least `ε` weight on current state:

* Let `α_last` correspond to weight for `x_l`.
* Ensure `α_last ≥ ε` by construction.

Implementation:

* Use a reparameterization:

  * set `α_last = ε + (1-ε)*softmax(z)_last`
  * scale other weights by `(1-ε)` accordingly

This makes it almost impossible to break identity mapping early in training.

---

### Constraint: Doubly stochastic (optional in v0.1)

Relevant if using matrix mixing. If kept in v0.1, implement for completeness but do not make it the default.

Projection algorithm:

* Sinkhorn-Knopp normalization with safe eps and max iters.

---

### Projection policy

`projection` controls *when* constraints are enforced:

* `"forward"`: enforce inside `forward()` (safe and easy)
* `"post_step"`: provide a helper to project after optimizer step (advanced)
* `"none"`: unconstrained HC (for ablation only)

**Recommended default:** `"forward"`

---

## Initialization Strategy

Initialization is critical; most instability shows up at step 0–100.

### Goals

* Start near residual behavior.
* Allow learning richer mixing gradually.

### Recommended init modes

#### `init="identity"`

Make the mixing weights strongly favor the latest state:

* For simplex: initialize logits so `α_last ≈ 1` and others ≈ 0.
* For identity-preserving: set `ε` moderately high (e.g. 0.5–0.9) at start.

#### `init="uniform"`

All history weights equal (only for analysis; not default).

#### `init="zeros"`

Equivalent to residual-like when combined with explicit residual addition.

**Default:** `identity`

---

## Forward/Backward Behavior

### Forward

Given `history` list:

1. select last `K` elements (bounded by `max_history`)
2. compute `α` (depending on `mode` and constraints)
3. compute `mix = Σ α_k * history_k`
4. output:

   * `mode="residual"`: `y = x + f(x)` (no history mixing)
   * `mode="hc"`: `y = f(x) + mix` (unconstrained)
   * `mode="mhc"`: `y = f(x) + mix` (constrained)

### Backward

* If using softmax parameterization, gradients flow naturally.
* If using hard projection, apply it in `no_grad()`; gradients flow through pre-projection values (common practical approach).

**v0.1 recommendation:** softmax parameterization for gradients.

---

## Integration Patterns

### Pattern 1: Manual integration inside a loop (most reliable)

Works with any stack of layers.

```python
mhc = MHCSkip(mode="mhc", max_history=4, constraint="simplex")
history = []

for layer in layers:
    x = layer(x)
    x = mhc(x, history)
    history.append(x)
```

### Pattern 2: Wrap an encoder block (optional helper)

If your model is block-based:

* `block(x)` returns `x`
* inject `mhc` after each block

### Pattern 3: Use `HistoryBuffer` (recommended for safety)

```python
buf = HistoryBuffer(max_history=4)

for layer in layers:
    x = layer(x)
    x = mhc(x, buf.get())
    buf.append(x)
```

---

## Experiments and Reproducibility Suite

### Philosophy

Do **one killer experiment** exceptionally well rather than many weak demos.

### Experiment 1 (Core): Deep Stability Stress Test

**Objective:** show that unconstrained HC diverges more often, residual trains but plateaus, mHC trains stably and/or improves.

Recommended variants:

* Deep MLP (fast, clean), and/or
* Deep Transformer encoder on a small dataset

**Deliverables**

* Config-driven training script
* Plots:

  * loss vs steps
  * gradient norm vs layer
  * divergence rate across seeds
  * activation norm statistics

**Ablations**

* `mode`: residual vs hc vs mhc
* `max_history`: 2, 4, 8
* constraint: simplex vs identity-preserving
* init: identity vs uniform

### Experiment 2 (Smoke): Tiny run (CI-friendly)

* 1–2 minutes runtime
* Confirms code doesn’t break on CPU
* Validates output shapes + no NaNs

### Reproducibility rules

* Fixed seeds (torch, numpy, python)
* Fixed configs saved to disk
* Logs include git commit hash

---

## Testing and Quality Gates

### Required test categories

#### Unit tests (fast)

* `test_shapes`: output shapes match inputs
* `test_simplex_sum1`: simplex α sums to 1 (within tolerance)
* `test_nonneg`: simplex α ≥ 0
* `test_identity_init`: with `init="identity"`, α_last is dominant

#### Property tests (stability)

* `test_norm_bound`: with simplex mixing, output norm does not explode under controlled inputs
* `test_no_nan`: forward/backward never produces NaNs for random inputs across seeds

#### Gradient sanity

* `test_backward_runs`: autograd works through MHCSkip
* `test_grad_magnitude_reasonable`: gradient norms within expected range on a toy network

### Quality gates (before release)

* `pytest` passes on CPU
* Lint: `ruff` or `flake8`
* Type hints: `mypy` optional but recommended
* Examples run end-to-end

---

## Performance Considerations

### Complexity

* Mixing cost is `O(K)` per layer for vector mixing.
* Keep default `max_history` small (4).

### Memory

* History can be expensive. Provide options:

  * `detach_history=True` (store detached tensors) for stability analysis
  * `store="fp16"` optional for large runs
  * `checkpointing` is out of scope for v0.1

### Recommended defaults

* `max_history=4`
* `constraint="simplex"`
* `projection="forward"`
* `init="identity"`

---

## Packaging and Distribution

### Package metadata

* Name: `mhc`
* Python: `>=3.9`
* PyTorch: `>=2.0`
* License: MIT or Apache-2.0 (choose one)

### Build system

* `pyproject.toml` (PEP 517)
* `setuptools` or `hatchling`

### Versioning strategy

* SemVer: `MAJOR.MINOR.PATCH`
* Tag releases tied to feature completeness:

  * `0.1.0`: vector-mixing + simplex + identity + experiments + tests
  * `0.2.0`: additional constraints + optional wrapper injection

---

## Documentation Standards

### README must include

1. What mHC is (short)
2. Installation
3. 3-line quickstart
4. One reproducible experiment command
5. How to run tests
6. API reference overview

### Docs site (optional but powerful)

* Use MkDocs or Sphinx
* Pages:

  * Concepts
  * API
  * Constraints
  * Reproducing experiments
  * FAQ

### Diagrams (recommended)

* A simple diagram showing:

  * residual uses only `x_l`
  * HC/mHC mix `[x_{l-K+1}, ..., x_l]`

---

## Roadmap

### v0.1 (must-have)

* `MHCSkip` with modes: residual/hc/mhc
* Constraints: simplex, identity-preserving
* `HistoryBuffer`
* One killer experiment + one smoke
* Tests (unit + property)

### v0.2 (nice-to-have)

* Optional wrapper integration for common patterns
* Better logging utilities
* Optional doubly stochastic (if matrix mixing is added)

### v0.3 (research polish)

* More benchmarks (ViT or encoder LM)
* Visualization scripts
* Paper-style report templates

---

## Risks and Mitigations

| Risk                        | Impact                    | Mitigation                                                            |
| --------------------------- | ------------------------- | --------------------------------------------------------------------- |
| Overengineering early       | delays release            | ship v0.1 with minimal constraints + one strong experiment            |
| Numerical instability       | loss of trust             | conservative defaults + identity init + no hard projection by default |
| Memory blow-up from history | unusable for large models | bounded history + HistoryBuffer + optional detach                     |
| Unclear benefit             | low adoption              | one clean, convincing experiment with plots + ablations               |
| API churn                   | user frustration          | freeze primary API (`MHCSkip`, `HistoryBuffer`) after v0.1            |

---

## Appendix: Suggested Repo Layout

```text
mhc/
├── mhc/
│   ├── __init__.py
│   ├── layers/
│   │   ├── mhc_skip.py
│   │   └── history_buffer.py
│   ├── constraints/
│   │   ├── simplex.py
│   │   └── identity.py
│   ├── manifolds/
│   │   └── sinkhorn.py          # optional
│   ├── utils/
│   │   ├── seed.py
│   │   └── tensor_ops.py
│   └── version.py
├── experiments/
│   ├── deep_mlp_stress_test.py
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── residual.yaml
│   │   ├── hc.yaml
│   │   └── mhc.yaml
│   └── README.md
├── tests/
│   ├── test_shapes.py
│   ├── test_simplex.py
│   ├── test_identity_init.py
│   ├── test_no_nan.py
│   └── test_backward.py
├── README.md
├── LICENSE
├── pyproject.toml
└── CITATION.cff
```

---

## Final Notes (Implementation Priorities)

If the goal is “so good anyone can use it”, prioritize in this order:

1. **Correctness + stability defaults**
2. **Quickstart + reproducible experiment**
3. **Tests that prove properties**
4. **Clean API**
5. Only then: extra constraints, wrappers, fancy benchmarks

---

### Minimal Quickstart Snippet (must work in v0.1)

```python
import torch
from mhc import MHCSkip, HistoryBuffer

mhc = MHCSkip(mode="mhc", max_history=4, constraint="simplex", init="identity")
buf = HistoryBuffer(max_history=4)

x = torch.randn(2, 16, 64)  # (batch, tokens, dim)

# pretend we have 6 layers
for _ in range(6):
    x = x + 0.01 * torch.randn_like(x)     # placeholder "layer"
    x = mhc(x, buf.get())
    buf.append(x)

print(x.shape)
```

---

**End of document.**

```
```
