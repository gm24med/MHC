# Hugging Face Transformers Integration

Upgrading massive pretrained models like BERT, ViT, or GPT to use Hyper-Connections is a powerful way to stabilize fine-tuning or squeeze more accuracy out of a vanilla architecture. This guide explains how `inject_mhc` surgically modifies the Hugging Face transformer block.

---

## 1. How Injection Works Internally

Hugging Face models are hierarchically structured. A `BertModel` contains a `BertEncoder`, which contains a list of `BertLayer` modules. Each `BertLayer` contains a self-attention sub-block and a feed-forward sub-block.

The `inject_mhc` utility performs a **recursive traversal** of this tree:

1.  **Detection**: It finds modules that match the `target_class_name` (e.g., "BertLayer").
2.  **Wrapping**: It replaces the standard module with an `InjectedMHC` wrapper.
3.  **Hooking**: It adds `forward_pre_hook` and `forward_hook` to the model to handle the `HistoryBuffer` updates automatically.

> [!IMPORTANT]
> The injection is non-destructive. It preserves all original weights and only adds the minimal parameters needed for history mixing.

---

## 2. Targeting Specific Architectures

### A. BERT-like (Encoder ONLY)
Best for: Classification, NER, Summarization.

```python
from transformers import BertModel
from mhc import inject_mhc

model = BertModel.from_pretrained("bert-base-uncased")
# Target 'BertLayer' to wrap the entire self-attention + MLP block
model = inject_mhc(model, target_class_name="BertLayer", max_history=4)
```

### B. Vision Transformers (ViT)
Best for: Image Classification, Object Detection.

```python
from transformers import ViTModel
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
# ViT uses 'ViTLayer'
model = inject_mhc(model, target_class_name="ViTLayer", max_history=6)
```

### C. Generative Models (GPT / Llama)
Best for: Causal text generation.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
# GPT-2 uses 'GPT2Block'
model = inject_mhc(model, target_class_name="GPT2Block", max_history=4)
```

---

## 3. Parameter Efficiency & LoRA Comparison

mHC adds very few parameters to a Transformer. For a BERT-base model:
-   **LoRA**: Adds ~1M to 3M parameters in the attention matrices.
-   **mHC ($H=4$)**: Adds exactly $12 \times 4 = 48$ parameters (the mixing weights $\alpha$).

**mHC operates on the "Architecture Topology," while LoRA operates on the "Weight Values."**
We highly recommend using mHC **on top of** LoRA for the ultimate fine-tuning stability.

---

## 4. Best Practices for Transformers

### Constraint: "Identity" is King
Transformers are built on the premise that the residual signal carries the main representation. We highly recommend using `constraint="identity"` with `epsilon=0.1`.

### Detaching History
Transformers use a lot of memory for attention keys/values. Adding a history of 4 tensors can push you over the VRAM limit.
**Recommended Config:**

```python
inject_mhc(
    model,
    target_class_name="ViTLayer",
    detach_history=True,  # Crucial for Transformers
    mode="mhc"
)
```

---

## 5. Fine-Tuning Strategy: The "Warming" Phase

When you inject mHC into a pretrained model:
1.  **Initial State**: The `Identity` constraint starts the model with 90% weight on the current block and 10% on history.
2.  **Specialization**: As fine-tuning progresses, the model will learn which layers need more historical context.

**Tip**: Use a lower learning rate ($1 \times 10^{-5}$) for the first epoch to allow the mHC mixing weights to stabilize before starting heavy weight updates on the transformer layers.

---

## 6. Visualizing the Transformer Manifold

After injecting mHC, you can use the `extract_mixing_weights` utility to see which parts of the Transformer are relying most heavily on "Long-Range Skips." Usually, you will find that the **middle layers** of the Transformer show the most diverse use of history, while the early layers remain strictly sequential.
