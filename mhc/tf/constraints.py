import tensorflow as tf


def project_simplex(logits: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
    """Project logits to the simplex via softmax."""
    return tf.nn.softmax(logits / temperature, axis=-1)


def project_identity_preserving(
    logits: tf.Tensor,
    epsilon: float = 0.1,
    temperature: float = 1.0
) -> tf.Tensor:
    """Simplex projection with minimum weight on latest state."""
    probs = tf.nn.softmax(logits / temperature, axis=-1)
    alphas = probs * (1.0 - epsilon)
    last = tf.ones_like(alphas[..., -1:]) * epsilon
    alphas = tf.concat([alphas[..., :-1], alphas[..., -1:] + last], axis=-1)
    return alphas


def project_doubly_stochastic(
    logits: tf.Tensor,
    iterations: int = 10,
    temperature: float = 1.0
) -> tf.Tensor:
    """Sinkhorn projection to a doubly stochastic matrix."""
    m = tf.exp(logits / temperature)
    for _ in range(iterations):
        m = m / (tf.reduce_sum(m, axis=-1, keepdims=True) + 1e-9)
        m = m / (tf.reduce_sum(m, axis=-2, keepdims=True) + 1e-9)
    return m
