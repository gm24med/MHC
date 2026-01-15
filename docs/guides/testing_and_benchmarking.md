# Testing & Benchmarking

Quality is the foundation of the Honey Badger philosophy. This guide explains how to run the integrated test suite and how to reproduce our stability benchmarks.

---

## 1. Running the Unit Tests

`mhc` uses `pytest` for its test suite. We cover everything from manifold projection precision to multi-GPU device placement.

### Basic Test Pass
```bash
uv run pytest
```

### Coverage Report
To see which parts of the library are currently covered by tests:
```bash
uv run pytest --cov=mhc
```

### Key Test Suites:
- `tests/test_mhc_skip.py`: Unit tests for the core mixing logic and manifold projections.
- `tests/test_managed.py`: Integration tests for `MHCSequential`.
- `tests/test_injection.py`: Verifies that `inject_mhc` correctly modifies Hugging Face and Torchvision models.
- `tests/test_history_buffer.py`: Checks for memory leaks and correct sliding window behavior.

---

## 2. Reproducing Benchmarks

We provide standardized scripts to compare mHC against vanilla architectures.

### Stability Stress Test
This script trains a 50-layer MLP on a synthetic dataset to see how long it takes for the model to converge or explode.

```bash
uv run python experiments/benchmark_stability.py --layers 50 --mode mhc
```

**Parameters**:
- `--layers`: Total depth. Try increasing this to `100` or `200` to see mHC's superiority.
- `--mode`: `resnet` vs `mhc`.
- `--constraint`: `simplex` vs `identity`.

### Profiling Overhead
To measure the exact FLOPs and latency overhead of mHC compared to a standard ResNet:

```bash
uv run python tests/test_profiling.py
```

---

## 3. Contributing New Tests

If you are adding a new feature (like a new manifold constraint), please:
1.  **Add a unit test** in `tests/` that verifies the mathematical properties (e.g., "Do weights sum to 1?").
2.  **Add a stability test** to ensure the new feature doesn't cause NaNs in deep networks.
3.  **Run formatting**: We use `ruff` to keep the code clean.
    ```bash
    uv run ruff check mhc tests
    ```
