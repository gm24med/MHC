# The History Lifecycle

Understanding exactly how a tensor transitions from "active output" to "historical state" is key to mastering mHC architectures. This guide traces a feature map through the sliding window lifecycle.

---

## 1. Birth: The Transformation Output

Every block in an mHC network (managed by `MHCSequential`) follows this logic:

1.  **Input $x_l$** enters the current module $f_l$.
2.  **Transformation**: $f_l(x_l) \to \hat{x}_{l}$ is computed. This is the raw output of your Linear or Conv layer.
3.  **Mixing**: The `MHCSkip` layer retrieves the history $[x_0, \dots, x_{l-1}]$ from the `HistoryBuffer`.
4.  **Consolidation**: The mixed state $x_{l+1} = \hat{x}_l + \text{Mix}(\text{History})$ is formed.

---

## 2. Retention: The History Buffer

As soon as $x_{l+1}$ is computed, it is stored in the `HistoryBuffer`.

### The Sliding Window Mechanics

Assume `max_history=3`. Let's look at the buffer state over steps:

- **Layer 0**: Buffer gets $[x_0]$ (The initial network input).
- **Layer 1**: Buffer gets $[x_0, x_1]$.
- **Layer 2**: Buffer gets $[x_0, x_1, x_2]$.
- **Layer 3**: Buffer gets $[x_1, x_2, x_3]$ (**$x_0$ is evicted** to maintain $H=3$).
- **Layer 4**: Buffer gets $[x_2, x_3, x_4]$.

### Memory Impact
Historical states are stored as **active pointers** in the buffer. If `detach_history=False`, these tensors hold their entire computation graphs, allowing gradients to flow back into the layers that originally produced them.

---

## 3. Eviction and Garbage Collection

When a state is "pushed out" of the window:
1.  It is removed from the `HistoryBuffer` list.
2.  If it is no longer being used by any other part of the graph (standard PyTorch autograd behavior), it becomes eligible for **Garbage Collection**.
3.  Memory is freed, keeping the total memory overhead of the skip connections bounded by the window size $H$, not the total depth $D$.

---

## 4. Resetting the Lifecycle

By default, the `MHCSequential` container clears the buffer at the start of every forward pass (`clear_history_each_forward=True`).

### Why reset?
- **Statelessness**: In supervised learning, the model should treat each batch as independent.
- **Safety**: Prevents information from Sample A leaking into the gradients of Sample B.

**When NOT to reset?**
In Recurrent-like settings or Time Series, you might set this to `False` to maintain long-range state across multiple forward calls, though this requires careful manual management.
