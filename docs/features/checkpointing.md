# Memory Checkpointing

Hyper-Connections significantly improve model performance but can increase memory consumption because they require keeping historical activations in memory for the backward pass.

**Skip Checkpointing** is a memory-optimization feature that trades compute for memory.

---

## The Memory-Compute Tradeoff

Standard PyTorch stores all intermediate activations during the forward pass. For a deep `MHCSequential` model, this includes the outputs of every layer + the historical buffers.

When `use_checkpointing=True` is enabled:
1.  **Forward Pass**: Intermediate activations are cleared from memory.
2.  **Backward Pass**: PyTorch **re-runs** the forward pass for specific segments (blocks) to re-calculate the missing activations.

This reduces the memory footprint from $O(Depth)$ to approximately $O(\sqrt{Depth})$, allowing you to train networks that are twice as deep on the same hardware.

---

## When to use Checkpointing?

*   **Deep Models**: If you are training models with >24 layers.
*   **Large History**: If you are using `max_history >= 8`.
*   **High Resolution**: For Vision models (Conv2D) where feature maps are large.

## Usage

```python
from mhc import MHCSequential
import torch.nn as nn

# Optimized for memory
model = MHCSequential(
    modules_list,
    use_checkpointing=True,
    detach_history=True  # Both recommended for extreme depth
)
```

> [!TIP]
> Gradient Checkpointing has a small (~20%) overhead in training time but can prevent **Out of Memory (OOM)** errors completely.
