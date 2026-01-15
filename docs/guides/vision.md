# Computer Vision: mHC for Conv2D

While mHC was born in MLPs, its most dramatic impact is in **Convolutional Neural Networks (CNNs)**. In 2D space, hyper-connections allow for multi-scale feature reuse, acting as a "cross-layer attention" mechanism that is far more computationally efficient than standard attention.

---

## 1. The Dimensional Challenge

In a transformer or MLP, the tensor shape is usually constant $(B, L, C)$. In CNNs, the spatial dimensions $(H, W)$ change at every pooling layer or strided convolution.

### Dimensional Compatibility Rules:

1.  **Strict Mode (Default)**: A history state $x_k$ can only be mixed with the current state $x_l$ if their spatial dimensions match exactly. If dimensions differ, the older state is ignored.
2.  **Auto-Projection (`auto_project=True`)**: If spatial dimensions don't match, `mhc` detects the mismatch and applies a learnable downsampling:
    -   **Stride-Match**: A strided $1 \times 1$ conv is applied to the history to align it with the current smaller feature map.
    -   **Channel-Match**: Aligns $C_{in}$ to $C_{out}$.

---

## 2. Pre-configured Vision Blocks

`mhc` provides drop-in blocks modeled after the most successful CNN architectures.

### A. MHCConv2d
The atomic unit of vision. It bundles a standard `3x3` convolution with a manifold-constrained skip.

```python
from mhc.layers import MHCConv2d

layer = MHCConv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    max_history=4,
    mode="mhc"
)
```

### B. MHCBasicBlock & Bottleneck
Designed to upgrade ResNet-34/50/101 models. By replacing standard Residual Blocks with MHC Blocks, you transform the ResNet into a **Hyper-ResNet**.

| Block Type | Inner Structure | MHC Advantage |
| :--- | :--- | :--- |
| **Basic** | 2x (3x3 Conv) | Massive gradient flow across 4 blocks simultaneously. |
| **Bottleneck** | 1x1, 3x3, 1x1 | Reduces "semantic drift" in deep 100+ layer vision backbones. |

---

## 3. High-Fidelity Applications

### Medical Imaging (MRI/CT)
In medical imaging, small details (like tumors or micro-fractures) are often lost as tensors are downsampled through a deep backbone.
**Solution**: mHC allows deep layers to **directly sample** high-resolution history from the early layers, preserving micro-features while still gaining the semantic benefit of depth.

### Satellite & Aerial Data
Satellite imagery often has huge resolutions (4000x4000).
**Strategy**: Use mHC with a large `max_history` ($H=8$) in the early layers of the network to maintain a "Long-Range Spatial Memory" across the feature extraction phase.

---

## 4. Performance in Vision Models

Vision tensors are high-volume. Adding $H$ images together takes $O(B \times C \times H \times W \times H_{lookback})$.

### VRAM Optimization Checklist:

-   [ ]   **Lower `max_history`**: $H=3$ is often sufficient for Vision.
-   [ ]   **`detach_history=True`**: Saves ~40% VRAM in ResNet-50 upgrades.
-   [ ]   **Use `mhc_bottleneck`**: Only 1 in every 3 convs needs mHC to see 90% of the stability benefits.

---

## 5. Case Study: Segmentation & Hyper-UNets

In a UNet, standard skip connections concatenate encoder features to decoder features.
**The Hyper-UNet Approach**: Replace standard skips with `MHCSkip`. This allows the decoder to learn a **dynamic mixture** of multiple encoder resolutions, leading to significantly sharper boundaries in semantic segmentation maps.
