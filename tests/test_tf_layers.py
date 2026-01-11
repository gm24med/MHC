import pytest

tf = pytest.importorskip("tensorflow")

from mhc.tf import TFMHCSkip, TFMHCSequential  # noqa: E402


def test_tf_mhc_skip_basic():
    layer = TFMHCSkip(max_history=3, mode="mhc", constraint="simplex")
    x = tf.random.normal((2, 8))
    history = [tf.random.normal((2, 8)), tf.random.normal((2, 8))]
    out = layer(x, history)
    assert out.shape == x.shape


def test_tf_mhc_sequential():
    layers = [tf.keras.layers.Dense(8), tf.keras.layers.ReLU()]
    model = TFMHCSequential(layers, max_history=3)
    x = tf.random.normal((2, 8))
    out = model(x)
    assert out.shape == x.shape


def test_tf_mhc_sequential_no_tf_function():
    model = TFMHCSequential([tf.keras.layers.Dense(4)], max_history=2)

    @tf.function
    def wrapped(t):
        return model(t)

    x = tf.random.normal((2, 4))
    with pytest.raises(RuntimeError):
        _ = wrapped(x)
