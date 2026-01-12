from typing import List, Optional

import tensorflow as tf

from .constraints import (
    project_simplex,
    project_identity_preserving,
    project_doubly_stochastic,
)
from .graph import TFHistoryBufferGraph


class TFHistoryBuffer:
    """History buffer for TensorFlow models (eager-friendly)."""

    def __init__(self, max_history: int = 4, detach_history: bool = False) -> None:
        self.max_history = max_history
        self.detach_history = detach_history
        self.buffer: List[tf.Tensor] = []

    def append(self, x: tf.Tensor) -> None:
        if self.detach_history:
            x = tf.stop_gradient(x)
        self.buffer.append(x)
        if len(self.buffer) > self.max_history:
            self.buffer.pop(0)

    def get(self) -> List[tf.Tensor]:
        return self.buffer

    def clear(self) -> None:
        self.buffer = []

    def __len__(self) -> int:
        return len(self.buffer)


class TFMHCSkip(tf.keras.layers.Layer):
    """TensorFlow implementation of Manifold-Constrained Hyper-Connections."""

    def __init__(
        self,
        mode: str = "mhc",
        max_history: int = 4,
        constraint: str = "simplex",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        init: str = "identity",
        auto_project: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        self.max_history = max_history
        self.constraint = constraint
        self.epsilon = epsilon
        self.temperature = temperature
        self.init = init
        self.auto_project = auto_project
        self.projection: Optional[tf.keras.layers.Layer] = None

    def build(self, input_shape):
        self.mixing_logits = self.add_weight(
            name="mixing_logits",
            shape=(self.max_history,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        if self.init == "identity":
            init = tf.concat(
                [tf.fill((self.max_history - 1,), -10.0), tf.zeros((1,))], axis=0
            )
            self.mixing_logits.assign(tf.cast(init, self.mixing_logits.dtype))
        super().build(input_shape)

    def _build_projection(self, history: tf.Tensor, x: tf.Tensor) -> tf.keras.layers.Layer:
        if len(history.shape) == 4 and len(x.shape) == 4:
            if history.shape[1:3] != x.shape[1:3]:
                raise RuntimeError(
                    "Auto projection only supports channel changes when spatial dims match."
                )
            return tf.keras.layers.Conv2D(
                filters=x.shape[-1], kernel_size=1, use_bias=False
            )
        if history.shape[:-1] != x.shape[:-1]:
            raise RuntimeError("Auto projection only supports matching leading dimensions.")
        return tf.keras.layers.Dense(units=x.shape[-1], use_bias=False)

    def _project_history(self, history: List[tf.Tensor], x: tf.Tensor) -> List[tf.Tensor]:
        mismatched = [h for h in history if h.shape != x.shape]
        if not mismatched:
            return history
        base_shape = mismatched[0].shape
        if any(h.shape != base_shape for h in mismatched):
            raise RuntimeError(
                "Auto projection requires all mismatched history states to share a shape."
            )
        if self.projection is None:
            self.projection = self._build_projection(mismatched[0], x)
        return [self.projection(h) if h.shape != x.shape else h for h in history]

    def call(self, x: tf.Tensor, history) -> tf.Tensor:
        if isinstance(history, tf.Tensor):
            if tf.shape(history)[0] == 0:
                return x
            hist_tensor = history
            hist_list = None
        else:
            if not history:
                return x
            hist_list = history
            hist_tensor = tf.stack(history, axis=0)
        if self.mode == "residual":
            h = hist_tensor[-1] if hist_list is None else hist_list[-1]
            if h.shape != x.shape:
                if not self.auto_project:
                    raise RuntimeError("Shape mismatch in TFMHCSkip. Enable auto_project.")
                h = self._project_history([h], x)[0]
            return x + h

        if hist_list is None:
            hist_window = hist_tensor
            if tf.reduce_any(tf.not_equal(tf.shape(hist_tensor)[1:], tf.shape(x))):
                if not self.auto_project:
                    raise RuntimeError("Shape mismatch in TFMHCSkip. Enable auto_project.")
                hist_list = self._project_history(tf.unstack(hist_tensor, axis=0), x)
                hist_window = tf.stack(hist_list, axis=0)
        else:
            hist_window = hist_list[-self.max_history :]
            if any(h.shape != x.shape for h in hist_window):
                if not self.auto_project:
                    raise RuntimeError("Shape mismatch in TFMHCSkip. Enable auto_project.")
                hist_window = self._project_history(hist_window, x)

        if hist_list is None:
            k = tf.shape(hist_window)[0]
            logits = self.mixing_logits[:k]
        else:
            k = len(hist_window)
            logits = self.mixing_logits[-k:]
        if self.mode == "hc":
            alphas = tf.nn.softmax(logits / self.temperature, axis=-1)
        elif self.mode == "mhc":
            if self.constraint == "simplex":
                alphas = project_simplex(logits, temperature=self.temperature)
            elif self.constraint == "identity":
                alphas = project_identity_preserving(
                    logits, epsilon=self.epsilon, temperature=self.temperature
                )
            else:
                raise ValueError(f"Unknown constraint: {self.constraint}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if hist_list is None:
            history_mix = tf.tensordot(alphas, hist_window, axes=[[0], [0]])
        else:
            history_mix = 0.0
            for idx, alpha in enumerate(tf.unstack(alphas)):
                history_mix = history_mix + alpha * hist_window[idx]

        return x + history_mix


class TFMatrixMHCSkip(tf.keras.layers.Layer):
    """Matrix mixing variant using doubly-stochastic constraints."""

    def __init__(self, max_history: int = 4, iterations: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_history = max_history
        self.iterations = iterations

    def build(self, input_shape):
        self.mixing_logits = self.add_weight(
            name="mixing_logits",
            shape=(self.max_history, self.max_history),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, history: List[tf.Tensor]) -> tf.Tensor:
        if not history:
            return x
        hist_window = history[-self.max_history :]
        k = len(hist_window)
        logits = self.mixing_logits[-k:, -k:]
        weights = project_doubly_stochastic(logits, iterations=self.iterations)
        h = tf.stack(hist_window, axis=1)
        flat = tf.reshape(h, (tf.shape(h)[0], k, -1))
        mixed = tf.matmul(weights, flat)
        mixed = tf.reshape(mixed, tf.shape(h))
        return x + mixed[:, -1]


class TFMHCSequential(tf.keras.layers.Layer):
    """Sequential container with automatic history management (eager-friendly).

    Note: This implementation is designed for eager execution and may not be
    graph-safe under tf.function due to Python-side history storage.
    """

    def __init__(
        self,
        layers: List[tf.keras.layers.Layer],
        max_history: int = 4,
        mode: str = "mhc",
        constraint: str = "simplex",
        epsilon: float = 0.1,
        detach_history: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.wrapped_layers = layers
        self.history = TFHistoryBuffer(max_history=max_history, detach_history=detach_history)
        self.skips = [
            TFMHCSkip(
                mode=mode,
                max_history=max_history,
                constraint=constraint,
                epsilon=epsilon,
            )
            for _ in layers
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if not tf.executing_eagerly():
            raise RuntimeError(
                "TFMHCSequential stores history in Python lists and is eager-only. "
                "Avoid wrapping it in tf.function."
            )
        self.history.clear()
        self.history.append(x)
        for layer, skip in zip(self.wrapped_layers, self.skips):
            out = layer(x)
            x = skip(out, self.history.get())
            self.history.append(x)
        return x

    def clear_history(self) -> None:
        self.history.clear()


class TFMHCSequentialGraph(tf.keras.layers.Layer):
    """Graph-safe sequential container using TensorArray history."""
    def __init__(
        self,
        layers: List[tf.keras.layers.Layer],
        max_history: int = 4,
        mode: str = "mhc",
        constraint: str = "simplex",
        epsilon: float = 0.1,
        detach_history: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.wrapped_layers = layers
        self.history = TFHistoryBufferGraph(
            max_history=max_history,
            detach_history=detach_history
        )
        self.skips = [
            TFMHCSkip(
                mode=mode,
                max_history=max_history,
                constraint=constraint,
                epsilon=epsilon,
            )
            for _ in layers
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        self.history.reset()
        self.history.append(x)
        for layer, skip in zip(self.wrapped_layers, self.skips):
            out = layer(x)
            hist = self.history.tail(skip.max_history)
            x = skip(out, tf.unstack(hist, axis=0))
            self.history.append(x)
        return x
