# MHC Deep Review (critical pass)

## Scope
Reviewed PyTorch core layers, constraints, history buffer, injection, conv/ResNet blocks, matrix mixing, TF integration, tests, docs, and CI.

## Strengths
- Clean separation of constraints, layers, and utilities (`mhc/constraints/*`, `mhc/layers/*`, `mhc/utils/*`).
- Good test coverage for core PyTorch behavior (`tests/*`).
- Drop-in APIs are simple and ergonomic (`MHCSequential`, `inject_mhc`).
- Conv/ResNet blocks are practical for vision users (`mhc/layers/conv_mhc.py`).
- Visualization utilities help interpret mixing behavior (`mhc/utils/visualization.py`).
- TF integration is optional and minimally invasive (`mhc/tf/*`).

## Weaknesses / Risks

### 1) Injection safety and shared history
- `inject_mhc` previously used one global `HistoryBuffer` for all injected layers, which is unsafe for branched graphs.
- Branching can mix unrelated activations and corrupt skip behavior in non-linear architectures.
- **Status**: fixed by adding `history_scope` with default `"module"` (per-layer buffers). See `mhc/utils/injection.py`.

### 2) History management rigidity
- `HistoryBuffer.get()` returns the live list; external mutation can corrupt state (`mhc/layers/history_buffer.py`).
- `MHCSequential` always clears history on each forward, which blocks recurrent use without custom handling (`mhc/layers/managed.py`).
- `MHCConv2d` hard-codes `detach_history=True` without a parameter override (`mhc/layers/conv_mhc.py`).
  - **Status**: fixed by exposing `detach_history` in `MHCConv2d`.
  - **Status**: `HistoryBuffer.get()` now returns a copy to avoid external mutation.

### 3) Projection logic and shape mismatch handling
- `MHCSkip(auto_project)` only supports limited shape changes and is not documented in user-facing docs (`mhc/layers/mhc_skip.py`).
- `MHCConv2d` projects history on-the-fly but keeps original states in the buffer, causing repeated projection cost.
- Dead code exists: `if len(history) == 0` inside `if history` in `MHCConv2d`.

### 4) Matrix mixing semantics and stability
- `MatrixMHCSkip` assumes stackable history and flattens dimensions, which is not documented and may be surprising for conv tensors (`mhc/layers/matrix_skip.py`).
- `project_doubly_stochastic` uses `exp(logits / temperature)` directly; large logits can overflow (`mhc/constraints/matrix.py`, `mhc/tf/constraints.py`).
  - **Status**: fixed by stabilizing logits with max-subtraction before exp in both PyTorch and TF.

### 5) TF integration gaps
- `TFMHCSequential` stores layers in `self.layers` (conflicts with Keras conventions); should rename to `wrapped_layers`.
- TF history buffer uses Python lists, which is eager-friendly but not graph-safe.
- No TF docs in `README.md` or `docs/*`.

### 6) Packaging and dependency hygiene
- `pytest` and `matplotlib` are runtime deps; they should be optional extras (`pyproject.toml`).
- This inflates install size for end users not using tests/plots.

### 7) Docs/API mismatches
- README’s `MatrixMHCSkip` example uses `feature_dim` and `constraint`, but the class signature does not support these (`README.md`, `mhc/layers/matrix_skip.py`).
- README references `examples/tutorials/` which does not exist (`README.md`).
- API docs omit TF modules and recent utilities (`docs/api.md`).

### 8) Observability and error context
- Shape mismatch errors lack module path context in complex models.
- No built-in tracing for buffer sizes or mixing weights per module.

## Recommendations (prioritized)

### High
1) Injection safety (fixed) — default per-layer buffers, optional global scope for legacy behavior.
2) Expose `detach_history` in `MHCConv2d`.
3) Stabilize doubly-stochastic projection (log-sum-exp / subtract max before exp).
4) Document TF limitations and add TF usage docs.

### Medium
5) Return a copy in `HistoryBuffer.get()` to prevent external mutation.
6) Validate `MatrixMHCSkip` history shapes and document flattening semantics.
7) Improve shape mismatch errors with module path context.

### Low
8) Move `pytest`/`matplotlib` to optional extras (`dev`, `viz`).
9) Make `set_seed` logging optional.
10) Align docs with real examples and API.

## Next milestones
- Injection safety ✅ (done)
- TensorFlow docs + example
- Stability for matrix mixing
- Packaging cleanup (deps and extras)
