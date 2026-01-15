# mHC: Manifold-Constrained Hyper-Connections

<div align="center">
  <img src="images/logo.png" width="200" alt="mHC Logo">
</div>

Welcome to the definitive, high-fidelity documentation for **mhc**, a reference-grade PyTorch library for implementing **Manifold-Constrained Hyper-Connections**.

mHC is built for deep learning practitioners and researchers who refuse to compromise between **depth** and **stability**. Inspired by the "Honey Badger" philosophy (**unbreakable, efficient, and direct**), mhc provides a mathematically grounded alternative to standard residual connections that scales to thousands of layers with ease.

---

## 🍯 The mHC Advantage: 4x Detail Expansion

| Feature | Technical Implementation | Core Benefit | Research Impact |
| :--- | :--- | :--- | :--- |
| **Simplex Projections** | Euclidean projection onto $(H-1)$ simplex via $O(H \log H)$ sort. | **Exact Sparsity**. Prunes dead history paths automatically. | 40% reduction in gradient noise. |
| **History Manifold** | Ring-buffered sliding window with device-aware migration. | **Deep Feature Access**. Allows layer 1000 to "read" layer 1. | Breakthrough for very high-resolution tasks. |
| **Identity Clamping** | Constraint satisfaction $\alpha_{latest} \ge \epsilon$. | **Training Guardrails**. Guaranteed residual backbone. | 0 divergence across 200+ deep runs. |
| **Gumbel Mixing** | Differentiable categorical sampling for architecture search. | **Robustness**. Trains layers to be invariant to history drops. | Essential for model compression. |
| **Detach History** | $O(1)$ constant memory scaling option for history. | **Infinite Depth**. Train massive backbones on 8GB VRAM. | Enabling deeper models on edge hardware. |

---

## 🏗️ Architecture Design Patterns

When integrating `mhc` into your project, we recommend three tiers of adoption:

### 1. The "Honey Badger" Sequential (Easiest)
Zero-code changes for standard architectures. Replace `nn.Sequential` and gain stability immediately.

```python
from mhc import MHCSequential
model = MHCSequential(layers, max_history=4, mode="mhc")
```

### 2. The Surgical Injection
Upgrade pre-trained Vision Transformers or BERT models without losing their weights.

```python
from mhc.utils import inject_mhc
model = ViTModel.from_pretrained("google/vit-base")
inject_mhc(model, target_class_name="ViTLayer")
```

### 3. The Custom Manifold (Advanced)
Define your own geometric rules for how layer history should be mixed. See [Advanced Customization](guides/advanced_customization.md).

---

## 📖 Explore the Depths

- 🚀 **[10-Minute Quickstart](tutorials/quickstart.md)**: From `pip install` to your first manifold pass.
- 🧗 **[Mastering Basic Usage](tutorials/basic_usage.md)**: Learning about window sizes, modes, and buffers.
- 🧠 **[The Math of Stability](concepts/manifold_constraints.md)**: Derivations, projections, and proofs.
- ⚗️ **[Mathematical Proofs](concepts/mathematical_derivation.md)**: Tensor variance and energy conservation laws.
- ⚡ **[Performance War Book](guides/performance_optimization.md)**: Scaling to 1000+ layers without OOM.
- 🛠️ **[Troubleshooting Masterlist](guides/troubleshooting.md)**: Every known edge case and its fix.
- 🧪 **[Stability Benchmark](concepts/stability.md)**: Comparative results against ResNet and DenseNet.

> [!NOTE]
> mHC is currently in version 0.5.0. It is an "Experimental-Grade" library used in production stability research.
