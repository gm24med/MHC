# TensorFlow / Keras Support

While `mhc` is primarily optimized for the PyTorch ecosystem, we believe in "Stability for Everyone." We provide high-quality, production-ready support for TensorFlow and Keras users through the `mhc.tf` module.

---

## 1. Atomic Mixing: `TFMHCSkip`

The `TFMHCSkip` Keras layer is a direct functional equivalent of the PyTorch version. It supports all manifold constraints (Simplex, Identity) and is fully compatible with Keras's `Functional` and `Subclassing` APIs.

```python
import tensorflow as tf
from mhc.tf import TFMHCSkip

# 1. Define your inputs
inputs = tf.keras.Input(shape=(128,))

# 2. Main transformation
dense_out = tf.keras.layers.Dense(128, activation="relu")(inputs)

# 3. Manifold Mix
# 'history' should be a list of past activation Tensors
x = TFMHCSkip(
    max_history=4,
    mode="mhc",
    constraint="identity"
)(dense_out, history=[inputs])
```

---

## 2. Managed Buffers: `TFMHCSequential`

For standard feed-forward networks, we provide a sequential container that handles history buffers automatically. This removes the "state-management" headache from your Keras models.

```python
from mhc.tf import TFMHCSequential

model = TFMHCSequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
], max_history=4)

# Forward pass automatically manages the internal HistoryBuffer
outputs = model(inputs)
```

---

## 3. The `tf.function` & Serialization Challenge

A major hurdle with historical mixing in TensorFlow is ensuring that the history buffer is part of the **Static Computational Graph**. Standard Python lists are "unrolled" during `tf.function` compilation, which can lead to bloated graph sizes or incorrect history state updates.

### The Solution: `TFMHCSequentialGraph`

This specialized container uses **TF TensorArrays** to store history.
-   **Graph-Safe**: The history is stored as a variable within the Keras model.
-   **Serializable**: You can save your model with `model.save('mhc_model')` and the history buffer configuration is preserved in the Protobuf metadata.

```python
from mhc.tf import TFMHCSequentialGraph

# Perfect for Production serving with TensorFlow Serving or TFLite
model = TFMHCSequentialGraph(
    layers=[...],
    max_history=4
)
```

---

## 4. Performance in TensorFlow

To match the speed of our PyTorch implementation, `mhc.tf` uses XLA-optimized projection kernels.

*   **XLA Acceleration**: If you wrap your training loop in `@tf.function(jit_compile=True)`, the manifold projection logic will be fused with the layer addition, resulting in near-zero overhead.
*   **Precision**: We recommend using `mixed_precision.set_global_policy('mixed_float16')` for large TensorFlow mHC models to keep memory usage identical to PyTorch.

---

## 5. Limitations

While we strive for parity, some advanced features of the PyTorch library are not yet available in TensorFlow:
-   **Variational Gumbel-Softmax**: Currently in experimental development for Keras.
-   **Automatic Model Injection**: Unlike PyTorch, Keras models are less malleable at runtime, so we don't currently support a `tf_inject_mhc` utility. You must build your model using `TFMHCSequential`.
