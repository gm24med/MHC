from typing import Optional

import tensorflow as tf


class TFHistoryBufferGraph(tf.keras.layers.Layer):
    """Graph-safe history buffer using TensorArray."""

    def __init__(self, max_history: int = 4, detach_history: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_history = max_history
        self.detach_history = detach_history
        self._array: Optional[tf.TensorArray] = None
        self._count = tf.Variable(0, trainable=False, dtype=tf.int32)

    def reset(self) -> None:
        self._array = None
        self._count.assign(0)

    def append(self, x: tf.Tensor) -> None:
        if self.detach_history:
            x = tf.stop_gradient(x)
        if self._array is None:
            self._array = tf.TensorArray(dtype=x.dtype, size=0, dynamic_size=True)
        self._array = self._array.write(self._count, x)
        self._count.assign_add(1)

    def get(self) -> tf.TensorArray:
        if self._array is None:
            return tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        return self._array

    def tail(self, k: int) -> tf.Tensor:
        """Return last k items as a stacked tensor."""
        array = self.get()
        count = self._count
        k = tf.minimum(tf.cast(k, tf.int32), count)
        start = tf.maximum(count - k, 0)
        indices = tf.range(start, count)
        return array.gather(indices)
