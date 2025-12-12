"""Common ops for discrete-action Policy Gradient functions."""

from typing import NamedTuple, Optional
import tensorflow as tf

from deep_rl_box.utils import base

class EntropyExtra(NamedTuple):
    entropy: Optional[tf.Tensor]

def value_loss(target: tf.Tensor, predict: tf.Tensor) -> base.LossOutput:
    """Calculates the squared error loss.

    Args:
        target: the estimated target value, shape [B,] or [T, B].
        predict: the predicted value, shape [B,] or [T, B].

    Returns:
        A namedtuple with fields:
        * `loss`: Baseline 'loss', shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(target, (1, 2), tf.float32)
    base.assert_rank_and_dtype(predict, (1, 2), tf.float32)

    assert target.shape == predict.shape

    loss = 0.5 * tf.square(target - predict)

    if len(loss.shape) == 2:
        # Averaging over time dimension.
        loss = tf.reduce_mean(loss, axis=0)

    return base.LossOutput(loss, extra=None)

def entropy_loss(logits_t: tf.Tensor) -> base.LossOutput:
    """Calculates the entropy regularization loss.

    See "Function Optimization using Connectionist RL Algorithms" by Williams.
    (https://www.tandfonline.com/doi/abs/10.1080/09540099108946587)

    Args:
        logits_t: a sequence of raw action preferences, shape [B, action_dim] or [T, B, action_dim].

    Returns:
        A namedtuple with fields:
        * `loss`: Entropy 'loss', shape `[B]`.
        * `extra`: a namedtuple with fields:
            * `entropy`: Entropy of the policy, shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_t, (2, 3), tf.float32)

    m = tf.nn.softmax(logits_t)
    entropy = -tf.reduce_sum(m * tf.math.log(m + 1e-10), axis=-1)

    if len(entropy.shape) == 2:
        # Averaging over time dimension.
        entropy = tf.reduce_mean(entropy, axis=0)

    return base.LossOutput(entropy, None)

def policy_gradient_loss(
    logits_t: tf.Tensor,
    a_t: tf.Tensor,
    adv_t: tf.Tensor,
) -> base.LossOutput:
    """Calculates the policy gradient a.k.a. log-likelihood loss.

    See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
    (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

    Args:
        logits_t: a sequence of raw action preferences, shape [B, action_dim] or [T, B, action_dim].
        a_t: a sequence of actions sampled from the preferences `logits_t`, shape [B] or [T, B].
        adv_t: the observed or estimated advantages from executing actions `a_t`, shape [B] or [T, B].

    Returns:
        A namedtuple with fields:
        * `loss`: policy gradient 'loss', shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_t, (2, 3), tf.float32)
    base.assert_rank_and_dtype(a_t, (1, 2), tf.int64)
    base.assert_rank_and_dtype(adv_t, (1, 2), tf.float32)

    base.assert_batch_dimension(a_t, logits_t.shape[0])
    base.assert_batch_dimension(adv_t, logits_t.shape[0])
    # For rank 3, check [T, B].
    if len(logits_t.shape) == 3:
        base.assert_batch_dimension(a_t, logits_t.shape[1], 1)
        base.assert_batch_dimension(adv_t, logits_t.shape[1], 1)

    logprob_a_t = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_t, labels=a_t)
    loss = logprob_a_t * tf.stop_gradient(adv_t)

    if len(loss.shape) == 2:
        # Averaging over time dimension.
        loss = tf.reduce_mean(loss, axis=0)

    return base.LossOutput(loss, extra=None)

def clipped_surrogate_gradient_loss(
    prob_ratios_t: tf.Tensor,
    adv_t: tf.Tensor,
    epsilon: float,
) -> base.LossOutput:
    """Computes the clipped surrogate policy gradient loss for PPO algorithms.

    L_clipₜ(θ) = min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)

    Where rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ) and Âₜ are the advantages.

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Args:
        prob_ratios_t: Ratio of action probabilities for actions a_t:
            rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ), shape [B].
        adv_t: the observed or estimated advantages from executing actions a_t, shape [B].
        epsilon: Scalar value corresponding to how much to clip the objective.

    Returns:
        Loss whose gradient corresponds to a clipped surrogate policy gradient
            update, shape [B,].
    """
    base.assert_rank_and_dtype(prob_ratios_t, 1, tf.float32)
    base.assert_rank_and_dtype(adv_t, 1, tf.float32)

    clipped_ratios_t = tf.clip_by_value(prob_ratios_t, 1.0 - epsilon, 1.0 + epsilon)
    clipped_objective = tf.minimum(prob_ratios_t * tf.stop_gradient(adv_t), clipped_ratios_t * tf.stop_gradient(adv_t))

    return base.LossOutput(clipped_objective, extra=None)
