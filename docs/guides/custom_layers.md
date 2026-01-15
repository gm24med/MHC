# Custom Layers Integration

If you have a non-sequential model (like a multi-branch transformer or a graph neural network) but want it to benefit from automated history management and dashboard logging, research-grade mHC integration is still remarkably simple.

---

## 1. The `@mhc_compatible` Decorator

This decorator is a core part of the "Honey Badger" ecosystem. It marks a class as a first-class citizen, signaling to utilities like `inject_mhc` and `MHCLightningCallback` that this module handles its own hyper-connectivity.

### When to use it:
-   You are building a custom `nn.Module` that isn't a simple list of layers.
-   You want your custom layer's mixing weights to show up in the PyTorch Lightning monitoring dashboard automatically.

```python
from mhc import mhc_compatible, MHCSkip
import torch.nn as nn

@mhc_compatible
class MySuperLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. Add the mixing engine
        self.skip = MHCSkip(max_history=4)
        self.layer = nn.Linear(dim, dim)

    def forward(self, x, history=None):
        # 2. Main transformation
        residual = x
        x = self.layer(x)

        # 3. Apply mHC mix
        # Note: 'history' is typically passed down by the container
        x = self.skip(x, history or [])

        return x
```

---

## 2. Manual History Management

When NOT using `MHCSequential`, you are the "Governor" of the `HistoryBuffer`. You must decide precisely when a state is important enough to be remembered.

### Pattern: The Manual Buffer Loop

```python
from mhc import HistoryBuffer, MHCSkip

class MyNonSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a buffer for tracking the last 4 feature maps
        self.buffer = HistoryBuffer(max_history=4)
        self.skip = MHCSkip(max_history=4)
        self.layer = nn.Linear(64, 64)

    def forward(self, x):
        # 1. Fetch current history from the buffer
        hist = self.buffer.get()

        # 2. Transform input and apply the skip
        out = self.layer(x)
        out = self.skip(out, hist)

        # 3. Update the buffer with the new state
        # This will automatically evict the oldest state if count > max_history
        self.buffer.append(out)

        return out
```

---

## 3. Handling Resets in Training Loops

If you manage history manually via a `HistoryBuffer` attribute, remember that these buffers **persist** in the model's memory across batch iterations. In a standard training loop, you must clear the buffer between batches to prevent "Temporal Bleeding" (where features from batch 1 affect batch 2).

### The "Auto-Reset" Pattern:

```python
def training_step(self, batch, batch_idx):
    # CRITICAL: Always clear history at the start of a training sample
    self.model.clear_history()

    x, y = batch
    preds = self.model(x)
    loss = F.cross_entropy(preds, y)
    return loss
```

---

## 4. Why `@mhc_compatible` is Mandatory

Using the decorator ensures that the `mhc` ecosystem preserves your intentions:

1.  **Dashboarding**: The `MHCLightningCallback` performs a recursive search for all modules. It only inspects and logs the `logits` of layers that are explicitly declared `@mhc_compatible`.
2.  **Projection Safety**: If you use `inject_mhc` on a large model that contains some of your custom layers, the injector will see the decorator and **skip** those layers to avoid wrapping them twice.
3.  **Device-Awareness**: It allows standard mHC utilities to help move your nested buffers between CPU and GPU automatically.
