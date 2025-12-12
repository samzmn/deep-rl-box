"""Common functions implementing custom non-linear transformations.

This is a collection of element-wise non-linear transformations that may be
used to transform losses, value estimates, or other multidimensional data.

Code adapted from DeepMind's RLax to support TensorFlow.
"""

import tensorflow as tf

from deep_rl_box.utils import base


def identity(x: tf.Tensor) -> tf.Tensor:
    """Identity transform."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return x


def sigmoid(x: tf.Tensor) -> tf.Tensor:
    """Sigmoid transform."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return tf.sigmoid(x)


def logit(x: tf.Tensor) -> tf.Tensor:
    """Logit transform, inverse of sigmoid."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return -tf.math.log(1.0 / x - 1.0)


def signed_logp1(x: tf.Tensor) -> tf.Tensor:
    """Signed logarithm of x + 1."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return tf.sign(x) * tf.math.log1p(tf.abs(x))


def signed_expm1(x: tf.Tensor) -> tf.Tensor:
    """Signed exponential of x - 1, inverse of signed_logp1."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return tf.sign(x) * tf.math.expm1(tf.abs(x))


def signed_hyperbolic(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return tf.sign(x) * (tf.sqrt(tf.abs(x) + 1) - 1) + eps * x


def hyperbolic_sin(x: tf.Tensor) -> tf.Tensor:
    """Hyperbolic sinus transform."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return tf.sinh(x)


def hyperbolic_arcsin(x: tf.Tensor) -> tf.Tensor:
    """Hyperbolic arcsinus transform."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    return tf.asinh(x)


def signed_parabolic(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    z = tf.sqrt(1 + 4 * eps * (eps + 1 + tf.abs(x))) / 2 / eps - 1 / 2 / eps
    return tf.sign(x) * (tf.square(z) - 1)


def power(x: tf.Tensor, p: float) -> tf.Tensor:
    """Power transform; `power_tx(_, 1/p)` is the inverse of `power_tx(_, p)`."""
    base.assert_dtype(x, (tf.float16, tf.float32, tf.float64))
    q = tf.sqrt(tf.constant(p, dtype=x.dtype))
    return tf.sign(x) * (tf.pow(tf.abs(x) / q + 1.0, p) - 1) / q


def transform_to_2hot(scalar: tf.Tensor, min_value: float, max_value: float, num_bins: int) -> tf.Tensor:
    """Transforms a scalar tensor to a 2 hot representation."""
    scalar = tf.clip_by_value(scalar, min_value, max_value)
    scalar_bin = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower, upper = tf.floor(scalar_bin), tf.math.ceil(scalar_bin)
    lower_value = (lower / (num_bins - 1.0)) * (max_value - min_value) + min_value
    upper_value = (upper / (num_bins - 1.0)) * (max_value - min_value) + min_value
    p_lower = (upper_value - scalar) / (upper_value - lower_value + 1e-5)
    p_upper = 1 - p_lower
    lower_one_hot = tf.one_hot(tf.cast(lower, tf.int32), num_bins) * tf.expand_dims(p_lower, -1)
    upper_one_hot = tf.one_hot(tf.cast(upper, tf.int32), num_bins) * tf.expand_dims(p_upper, -1)
    return lower_one_hot + upper_one_hot


def transform_from_2hot(probs: tf.Tensor, min_value: float, max_value: float, num_bins: int) -> tf.Tensor:
    """Transforms from a categorical distribution to a scalar."""
    support_space = tf.linspace(min_value, max_value, num_bins)
    scalar = tf.reduce_sum(probs * tf.expand_dims(support_space, 0), -1)
    return scalar
