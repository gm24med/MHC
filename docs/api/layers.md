# API Reference: Core Containers & Layers

This section provides the exhaustive technical specification for the primary building blocks of the `mhc` library. These are the components you will interact with most frequently when building Honey Badger models.

---

## MHCSequential

The `MHCSequential` class is a high-performance, manifold-aware container that extends `torch.nn.Module`. It is designed to be a direct, drop-in replacement for `nn.Sequential`.

### Initialization Parameters:

-   **`modules`** *(Iterable[nn.Module])*:
    The list of submodules to be wrapped. These can be any standard PyTorch layers (Linear, Conv2d, Transformers). `MHCSequential` will insert an `MHCSkip` node after each entry to manage the hyper-connectivity.

-   **`max_history`** *(int, default=4)*:
    The depth of the sliding window. A value of `4` means each layer draws from the outputs of up to 4 previous layers. Increasing this allows for longer-range feature reuse but increases VRAM consumption linearly.

-   **`mode`** *(str, default="mhc")*:
    The mixing algorithm used by the skips.
    -   `"mhc"`: Manifold-Constrained. The safest and most robust (Euclidean Projection).
    -   `"hc"`: Standard Softmax-based mixing (Can be unstable at extreme depths).
    -   `"residual"`: Identity skip (converts mHC into a standard ResNet for benchmarking).

-   **`constraint`** *(str, default="simplex")*:
    The geometric manifold to project onto.
    -   `"simplex"`: Convex combination (all weights $\ge 0$, $\sum = 1$).
    -   `"identity"`: Simplex with guaranteed center signal (Preserves the core "ResNet" backbone).

-   **`temperature`** *(float, default=1.0)*:
    Scalar that divides logits before projection. A lower temperature (e.g., 0.1) makes the manifold projection "sharper" (more zeros), while higher temperatures (e.g., 2.0) lead to more uniform mixing.

-   **`detach_history`** *(bool, default=False)*:
    If `True`, historical states are treated as constants during backpropagation. This is **crucial for saving memory** in deep networks, as it prevents the autograd engine from storing the entire history for gradient calculation.

-   **`clear_history_each_forward`** *(bool, default=True)*:
    Automated history flushing. Ensures that Batch A does not influence the history of Batch B. Disable this only if you are implementing custom recurrent logic.

-   **`auto_project`** *(bool, default=False)*:
    If `True`, the container will automatically resolve dimension mismatches (channel counts or spatial sizes) between historical states and current layers using learned $1 \times 1$ projections.

-   **`stochastic`** *(bool, default=False)*:
    Enables Gumbel-Softmax based Variational mixing, allowing the model to explore architecture configurations during training.

### Core Methods:

-   **`forward(x)`**:
    Executes the sequential pass. It manages the internal `HistoryBuffer` automatic updates and applies the manifold skips after each submodule.
-   **`clear_history()`**:
    Manually flushes the internal history. Use this if you are performing manual evaluation loops.

---

## MHCSkip

The atomic mixing engine. This layer contains the learnable parameters for the manifold and the projection logic.

### Parameters:

-   **`max_history`** *(int)*:
    The number of previous states this skip can "look at."
-   **`epsilon`** *(float, default=0.1)*:
    The minimum weight assigned to the latest state when using `constraint="identity"`.

### Internal Execution Logic:

When called with `forward(x, history)`, the layer performes the following:
1.  **Retrieve**: Pulls the raw $H$ tokens from the buffer.
2.  **Project**: Maps the internal `logits` to the simplex manifold $\Delta^{H-1}$ using Euclidean projection.
3.  **Mix**: Computes the weighted sum $\sum \alpha_k x_k$.
4.  **Accumulate**: Adds the mixture back to the current layer output $x$.

---

## HistoryBuffer

A specialized, device-aware sliding window container designed for high-throughput training.

### State Tracking:

-   **`max_history`** *(int)*:
    Maximum number of tensors stored before eviction starts.
-   **`containers`** *(List[Tensor])*:
    The internal double-ended queue.

### Critical Features:

-   **Device Awareness**:
    When you move an `MHCSequential` model from CPU to GPU (e.g., `model.to('cuda')`), the `HistoryBuffer` detects the change and automatically migrates all stored tensors to avoid device-mismatch errors.

-   **Reference Management**:
    `mhc` uses weak reference patterns and explicit popping to ensure that tensors are discarded correctly by PyTorch's memory manager, preventing hidden memory leaks often found in custom skip implementations.
