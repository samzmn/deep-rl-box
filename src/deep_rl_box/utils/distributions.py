"""Functions for working with probability distributions."""
import numpy as np
import tensorflow as tf

from deep_rl_box.utils import base


class CustomCategorical:
    def __init__(self, logits=None, probs=None):
        self.probs: tf.Tensor
        if logits is not None:
            self.probs = self._softmax(tf.convert_to_tensor(logits, dtype=tf.float32))
        elif probs is not None:
            probs_tensor = tf.convert_to_tensor(probs, dtype=tf.float32)
            self.probs = probs_tensor / tf.reduce_sum(probs_tensor)
        else:
            raise ValueError("Either logits or probs must be provided.")

    def _softmax(self, logits):
        return tf.nn.softmax(logits)

    def sample(self, num_samples=1):
        return tf.random.categorical(tf.math.log(self.probs), num_samples)[0]

    def log_prob(self, value):
        return tf.math.log(self.probs[value])

    def entropy(self):
        return -tf.reduce_sum(self.probs * tf.math.log(self.probs))


class CategoricalDistribution:
    def __init__(self, logits=None, probs=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if len(probs.shape) < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
            self.logits = tf.math.log(self.probs)
        else:
            if len(logits.shape) < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            self.logits = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
            self.probs = tf.nn.softmax(self.logits)
        self._num_events = self.probs.shape[-1]
        self.batch_shape = self.probs.shape[:-1]

    def sample(self, sample_shape=()):
        sample_shape = tf.TensorShape(sample_shape)
        probs_2d = tf.reshape(self.probs, [-1, self._num_events])
        samples_2d = tf.random.categorical(self.logits, sample_shape.num_elements(), dtype=tf.int32)
        return tf.reshape(samples_2d, self.batch_shape + sample_shape)

    def log_prob(self, value):
        value = tf.expand_dims(value, -1)
        value_one_hot = tf.one_hot(value, self._num_events)
        return tf.reduce_sum(value_one_hot * self.logits, axis=-1)

    def entropy(self):
        p_log_p = self.logits * self.probs
        return -tf.reduce_sum(p_log_p, axis=-1)


class NormalDistribution:
    def __init__(self, mean, stddev):
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.stddev = tf.convert_to_tensor(stddev, dtype=tf.float32)

    def sample(self, num_samples=1):
        return tf.random.normal(shape=(num_samples,), mean=self.mean, stddev=self.stddev)

    def entropy(self):
        return 0.5 * tf.math.log(2.0 * tf.constant(np.pi) * tf.square(self.stddev)) + 0.5

    def log_prob(self, value):
        return -0.5 * tf.square((value - self.mean) / self.stddev) - tf.math.log(self.stddev) - 0.5 * tf.math.log(2.0 * tf.constant(np.pi))


def categorical_distribution(logits: tf.Tensor) -> CustomCategorical:
    """Returns categorical distribution that supports sample(), entropy(), and log_prob()."""
    return CustomCategorical(logits=logits)


def normal_distribution(mu: tf.Tensor, sigma: tf.Tensor) -> NormalDistribution:
    """Returns normal distribution that supports sample(), entropy(), and log_prob()."""
    return NormalDistribution(mean=mu, stddev=sigma)


def categorical_importance_sampling_ratios(
    pi_logits_t: tf.Tensor, mu_logits_t: tf.Tensor, a_t: tf.Tensor
) -> tf.Tensor:
    """Compute importance sampling ratios from logits.

    Args:
        pi_logits_t: raw logits at time t for the target policy,
            shape [B, action_dim] or [T, B, action_dim].
        mu_logits_t: raw logits at time t for the behavior policy,
            shape [B, action_dim] or [T, B, action_dim].
        a_t: actions at time t, shape [B] or [T, B].

    Returns:
        importance sampling ratios, shape [B] or [T, B].
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(pi_logits_t, (2, 3), tf.float32)
    base.assert_rank_and_dtype(mu_logits_t, (2, 3), tf.float32)
    base.assert_rank_and_dtype(a_t, (1, 2), tf.int64)

    pi_logprob_a_t = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi_logits_t, labels=a_t)
    mu_logprob_a_t = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mu_logits_t, labels=a_t)

    # Alternativaly
    #pi_m = CategoricalDistribution(logits=pi_logits_t)
    #mu_m = CategoricalDistribution(logits=mu_logits_t)
    #pi_logprob_a_t = pi_m.log_prob(a_t)
    #mu_logprob_a_t = mu_m.log_prob(a_t)

    return tf.exp(pi_logprob_a_t - mu_logprob_a_t)
