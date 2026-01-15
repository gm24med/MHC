# API Reference: Utilities

Power tools for integrating, monitoring, and visualizing mHC in existing production workflows. This module ensures that `mhc` is not just an architecture, but a complete research observability kit.

---

## 🏗️ Model Injection Utility: `inject_mhc`

The `inject_mhc` function is the fastest way to upgrade an existing architecture. It performs a "surgical traversal" of your model graph, identifying key nodes and wrapping them in hyper-connectivity.

### Functional Signature:

```python
def inject_mhc(
    module: nn.Module,
    target_types: Union[Type, Tuple[Type]] = None,
    target_class_name: str = None,
    max_history: int = 4,
    **mhc_kwargs
) -> nn.Module:
```

### Parameters:

-   **`module`** *(nn.Module)*:
    The target model to process (e.g., a pre-trained `resnet50` or `BertModel`).
-   **`target_types`**:
    A specific PyTorch layer type to search for. For example, `nn.Linear` or `nn.Conv2d`.
-   **`target_class_name`**:
    A string matching the class name exactly. Useful for third-party libraries where you want to target specific container blocks like `"BertLayer"`.
-   **`max_history`**:
    The window size for all newly created skip connections.
-   **`**mhc_kwargs`**:
    Any valid `MHCSkip` parameter (e.g., `mode="mhc"`, `detach_history=True`).

---

## 📊 Stability Dashboard & Visualization

mHC provides high-level plotting tools that extract data directly from the manifold projections to help you "debug" your network's connectivity.

### `extract_mixing_weights(model)`
Traverses the model and returns a dictionary of all $H$-dimensional mixing vectors $\alpha$.

### `plot_mixing_weights(model)`
Generates a heatmap of history usage across all layers.
-   **Horizontal Axis**: History Index ($0 \dots H-1$).
-   **Vertical Axis**: Layer Depth.

### `plot_gradient_flow(model)`
Visualizes the gradient variance at each layer. A "flat" flow indicates a healthy, stable mHC configuration.

---

## ⚡ PyTorch Lightning Callback

The `MHCLightningCallback` automates the "boring" parts of history management and integrates mHC metrics into your favorite logging backends.

### Features:

-   **Auto-Reset**: Automatically calls `model.clear_history()` at the start of every training and validation batch.
-   **Dynamic Logging**:
    -   **`log_weights`**: Periodic heatmap uploads.
    -   **`log_entropy`**: Plots the Shannon entropy $(\sum -\alpha \log \alpha)$ of the mixing manifolds. High entropy means "Exploration"; low entropy means "Specialization."
    -   **`logging_backend`**: Native support for `"wandb"` and `"tensorboard"`.

---

## 🧪 System & Integrity Helpers

### `check_model_stability(model)`
A diagnostic utility that pushes a dummy tensor through the model and verifies:
1.  **Buffer Growth**: Did history populate correctly?
2.  **Gradient Path**: Is the model differentiable from output back to input?
3.  **Spectral Drift**: Are the activations staying within standard deviation boundaries?

### `@mhc_compatible`
The mandatory decorator for custom modules. It explicitly opts-in a layer to the mHC monitoring ecosystem.
```python
@mhc_compatible
class MyLayer(nn.Module): ...
```
