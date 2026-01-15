# Troubleshooting & Deep Architecture FAQ

mHC is a "Honey Badger"—it doesn't care about depth—but complex PyTorch integrations can still cause friction. This guide provides an exhaustive list of known edge cases, architectural "gotchas," and precision fixes.

---

## 🛑 Numerical Stability Issues

### 1. "My Loss is NaN from the first step"

-   **The Diagnosis**: Usually caused by using a high `learning_rate` with `mode="hc"` (Softmax). Softmax has a "long tail" that can sometimes accumulate gradients too aggressively.
-   **The Cure**:
    1.  Switch to `mode="mhc"`. The Euclidean Projection used in `mhc` mode is naturally more robust because it can zero-out noisy history.
    2.  Check for `input_normalization`. mHC amplifies existing signal; if your inputs are not zero-mean, they will quickly saturate the history mixing.

### 2. "Floating Point Underflow in Simplex Projection"

-   **The Case**: Logging shows that mixing weights $\alpha$ are extremely small ($1e-7$) but not exactly zero.
-   **The Cure**: Adjust the `temperature`. If $\tau$ is too high, the weights will stay small and uniform.
-   **Sweet Spot**: Maintain $\tau$ between $0.5$ and $2.0$.

---

## 🚀 Performance & Device Issues

### 3. "RuntimeError: Expected all tensors to be on the same device"

-   **The Diagnosis**: This happens if you instantiate layers manually and use `HistoryBuffer` without correctly moving it to the GPU.
-   **The Cure**: Use `MHCSequential`. It is "Device Aware" and will automatically move history buffers whenever the model moves (`model.to('cuda')`).

### 4. "OOM (Out of Memory) during Backprop"

-   **The Case**: Your model fits in VRAM during the forward pass but crashes during `loss.backward()`.
-   **The Diagnosis**: You are using `detach_history=False` (the default). PyTorch is trying to store the gradient graphs for $H$ previous layers.
-   **The Cure**: Set `detach_history=True`. This is the recommended mode for models with >100 layers. It cuts the memory footprint by up to 5x with minimal impact on accuracy.

---

## 🔍 Visual Debugging & Monitoring

### 5. Heatmap "Symmetry"
If you use visualization tools and see that the mixing weights $\alpha$ look identical for all layers, check your `clear_history` setting. If history is never cleared between batches, the manifold might "collapse" into a single global state.

### 6. Early Warning Signs of Divergence
-   **Entropy Spike**: If the entropy of your mixing weights suddenly jumps to maximum ($log(H)$), it means the model has lost its specialized paths and is defaulting to a noisy average.
-   **Weight Saturation**: If $\alpha_{latest}$ hits $0.99$ constantly, your network is ignoring history and behaving like a standard sequential model. Increase the `temperature` or decrease the `epsilon` to force more historical exploration.

---

## 🔌 Integration "Gotchas"

### 7. "My custom layer is being ignored by the Dashboard"

-   **The Cure**: You must use the `@mhc_compatible` decorator on your custom `nn.Module`. This adds a secret marker that the `MHCLightningCallback` uses to find your layers during model traversal.

```python
from mhc import mhc_compatible

@mhc_compatible
class MySuperLayer(nn.Module):
    # ... your logic here
```

### 8. "Mixing weights are all zeros"

-   **The Diagnosis**: Mathematically impossible if using `simplex` or `identity` constraints.
-   **Check**: Are you extracting parameters from the model *before* any data has been pushed? Buffers are initialized with zeros and only populate after the first `forward()` call.
