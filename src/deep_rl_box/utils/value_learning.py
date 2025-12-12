"""Functions for state value and action-value learning.

Value functions estimate the expected return (discounted sum of rewards) that
can be collected by an agent under a given policy of behavior. This subpackage
implements a number of functions for value learning in discrete scalar action
spaces. Actions are assumed to be represented as indices in the range `[0, A)`
where `A` is the number of distinct actions.
"""

from typing import NamedTuple, Optional
import tensorflow as tf

from deep_rl_box.utils import base
from deep_rl_box.utils import multistep


class QExtra(NamedTuple):
    target: Optional[tf.Tensor]
    td_error: Optional[tf.Tensor]


class DoubleQExtra(NamedTuple):
    target: tf.Tensor
    td_error: tf.Tensor
    best_action: tf.Tensor


class Extra(NamedTuple):
    target: Optional[tf.Tensor]


def qlearning(
    q_tm1: tf.Tensor,
    a_tm1: tf.Tensor,
    r_t: tf.Tensor,
    discount_t: tf.Tensor,
    q_t: tf.Tensor,
) -> base.LossOutput:
    r"""Implements the Q-learning loss.

    The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
    the target `r_t + discount_t * max q_t`.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/book/ebook/node65.html).

    Args:
      q_tm1: Tensor holding Q-values for first timestep in a batch of
        transitions, shape `[B x action_dim]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      q_t: Tensor holding Q-values for second timestep in a batch of
        transitions, shape `[B x action_dim]`.

    Returns:
      A namedtuple with fields:

      * `loss`: a tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
          * `td_error`: batch of temporal difference errors, shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(q_tm1, 2, tf.float32)
    base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
    base.assert_rank_and_dtype(r_t, 1, tf.float32)
    base.assert_rank_and_dtype(discount_t, 1, tf.float32)
    base.assert_rank_and_dtype(q_t, 2, tf.float32)

    base.assert_batch_dimension(a_tm1, q_tm1.shape[0])
    base.assert_batch_dimension(r_t, q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, q_tm1.shape[0])
    base.assert_batch_dimension(q_t, q_tm1.shape[0])

    # Q-learning op.
    # Build target and select head to update.
    target_tm1 = r_t + discount_t * tf.reduce_max(q_t, axis=1)
    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target_tm1 - qa_tm1
    loss = 0.5 * tf.square(td_error)

    return base.LossOutput(loss, QExtra(target_tm1, td_error))


def double_qlearning(
    q_tm1: tf.Tensor,
    a_tm1: tf.Tensor,
    r_t: tf.Tensor,
    discount_t: tf.Tensor,
    q_t_value: tf.Tensor,
    q_t_selector: tf.Tensor,
) -> base.LossOutput:
    r"""Implements the double Q-learning loss.

    The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
    the target `r_t + discount_t * q_t_value[argmax q_t_selector]`.

    See "Double Q-learning" by van Hasselt.
    (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

    Args:
      q_tm1: Tensor holding Q-values for first timestep in a batch of
        transitions, shape `[B x action_dim]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      q_t_value: Tensor of Q-values for second timestep in a batch of transitions,
        used to estimate the value of the best action, shape `[B x action_dim]`.
      q_t_selector: Tensor of Q-values for second timestep in a batch of
        transitions used to estimate the best action, shape `[B x action_dim]`.

    Returns:
      A namedtuple with fields:

      * `loss`: a tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`
          * `td_error`: batch of temporal difference errors, shape `[B]`
          * `best_action`: batch of greedy actions wrt `q_t_selector`, shape `[B]`
    """

    # Rank and compatibility checks.
    base.assert_rank_and_dtype(q_tm1, 2, tf.float32)
    base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
    base.assert_rank_and_dtype(r_t, 1, tf.float32)
    base.assert_rank_and_dtype(discount_t, 1, tf.float32)
    base.assert_rank_and_dtype(q_t_value, 2, tf.float32)
    base.assert_rank_and_dtype(q_t_selector, 2, tf.float32)

    base.assert_batch_dimension(a_tm1, q_tm1.shape[0])
    base.assert_batch_dimension(r_t, q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, q_tm1.shape[0])
    base.assert_batch_dimension(q_t_value, q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, q_tm1.shape[0])

    # double Q-learning op.
    # Build target and select head to update.

    best_action = tf.argmax(q_t_selector, axis=1)
    double_q_bootstrapped = base.batched_index(q_t_value, best_action)

    target_tm1 = r_t + discount_t * double_q_bootstrapped

    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target_tm1 - qa_tm1
    loss = 0.5 * tf.square(td_error)

    return base.LossOutput(loss, DoubleQExtra(target_tm1, td_error, best_action))


# Deprecated
# def _slice_with_actions(embeddings: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
#     """Slice a Tensor.

#     Take embeddings of the form [batch_size, action_dim, embed_dim]
#     and actions of the form [batch_size, 1], and return the sliced embeddings
#     like embeddings[:, actions, :].

#     Args:
#       embeddings: Tensor of embeddings to index.
#       actions: int Tensor to use as index into embeddings

#     Returns:
#       Tensor of embeddings indexed by actions
#     """
#     batch_size, action_dim = embeddings.shape[:2]

#     # Values are the 'values' in a sparse tensor we will be setting
#     act_idx = actions[:, None]

#     values = tf.reshape(tf.ones(actions.shape, dtype=tf.int8), [-1])

#     # Create a range for each index into the batch
#     act_range = tf.range(0, batch_size, dtype=tf.int64)[:, None]
#     # Combine this into coordinates with the action indices
#     indices = tf.concat([act_range, act_idx], 1)

#     # Needs transpose indices before adding to tf.sparse.SparseTensor.
#     actions_mask = tf.sparse.SparseTensor(indices, values, [batch_size, action_dim])
#     actions_mask = tf.sparse.to_dense(actions_mask, default_value=0)
#     actions_mask = tf.cast(actions_mask, tf.bool)

#     sliced_emb = tf.boolean_mask(embeddings, actions_mask)
#     # Make sure shape is the same as embeddings
#     sliced_emb = tf.reshape(sliced_emb, [embeddings.shape[0], -1])

#     return sliced_emb


def _slice_with_actions(embeddings: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    """Slice a Tensor using tf.gather.

    Takes embeddings of shape [batch_size, action_dim, embed_dim] and actions 
    of shape [batch_size, 1], returning sliced embeddings like embeddings[:, actions, :].

    Args:
        embeddings: Tensor of embeddings to index (shape `[batch_size, action_dim, embed_dim]`).
        actions: int Tensor to use as index into embeddings (shape `[batch_size, 1]`).

    Returns:
        Tensor of embeddings indexed by actions (shape `[batch_size, embed_dim]`).
    """
    # Squeeze actions to get shape `[batch_size]` for indexing
    actions = tf.squeeze(actions)

    # Gather embeddings using actions
    sliced_emb = tf.gather(embeddings, actions, axis=1, batch_dims=1)

    return sliced_emb  # Shape: [batch_size, embed_dim]


def l2_project(z_p: tf.Tensor, p: tf.Tensor, z_q: tf.Tensor) -> tf.Tensor:
    r"""Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.

    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).

    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.

    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.

    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p = tf.clip_by_value(z_p, clip_value_min=vmin, clip_value_max=vmax)[:, None, :]  # B x 1 x Kp

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = tf.where(d_neg > 0, 1.0 / d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = tf.where(d_pos > 0, 1.0 / d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = tf.cast(delta_qp >= 0.0, dtype=p.dtype)  # B x Kq x Kp

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1.0 - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return tf.reduce_sum(tf.clip_by_value(1.0 - delta_hat, clip_value_min=0.0, clip_value_max=1.0) * p, axis=2)


def categorical_dist_qlearning(
    atoms_tm1: tf.Tensor,
    logits_q_tm1: tf.Tensor,
    a_tm1: tf.Tensor,
    r_t: tf.Tensor,
    discount_t: tf.Tensor,
    atoms_t: tf.Tensor,
    logits_q_t: tf.Tensor,
) -> base.LossOutput:
    """Implements Distributional Q-learning as TensorFlow ops.

    The function assumes categorical value distributions parameterized by logits.

    See "A Distributional Perspective on Reinforcement Learning" by Bellemare,
    Dabney and Munos. (https://arxiv.org/abs/1707.06887).

    Args:
      atoms_tm1: 1-D tensor containing atom values for first timestep,
        shape `[num_atoms]`.
      logits_q_tm1: Tensor holding logits for first timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      atoms_t: 1-D tensor containing atom values for second timestep,
        shape `[num_atoms]`.
      logits_q_t: Tensor holding logits for second timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.

    Returns:
      A namedtuple with fields:

      * `loss`: a tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`: a tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_atoms]`.

    Raises:
      ValueError: If the tensors do not have the correct rank or compatibility.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_q_tm1, 3, tf.float32)
    base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
    base.assert_rank_and_dtype(r_t, 1, tf.float32)
    base.assert_rank_and_dtype(discount_t, 1, tf.float32)
    base.assert_rank_and_dtype(logits_q_t, 3, tf.float32)
    base.assert_rank_and_dtype(atoms_tm1, 1, tf.float32)
    base.assert_rank_and_dtype(atoms_t, 1, tf.float32)

    base.assert_batch_dimension(a_tm1, logits_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(logits_q_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(atoms_tm1, logits_q_tm1.shape[-1])
    base.assert_batch_dimension(atoms_t, logits_q_tm1.shape[-1])

    # Categorical distributional Q-learning op.
    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, None] + discount_t[:, None] * atoms_t[None, :]

    # Convert logits to distribution, then find greedy action in state s_t.
    q_t_probs = tf.nn.softmax(logits_q_t, axis=-1)
    q_t_mean = tf.reduce_sum(q_t_probs * atoms_t, axis=2)
    pi_t = tf.argmax(q_t_mean, axis=1)

    # Compute distribution for greedy action.
    p_target_z = _slice_with_actions(q_t_probs, pi_t)

    # Project using the Cramer distance
    target_tm1 = l2_project(target_z, p_target_z, atoms_tm1)

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_tm1, logits=logit_qa_tm1)

    return base.LossOutput(loss, Extra(target_tm1))


def categorical_dist_double_qlearning(
    atoms_tm1: tf.Tensor,
    logits_q_tm1: tf.Tensor,
    a_tm1: tf.Tensor,
    r_t: tf.Tensor,
    discount_t: tf.Tensor,
    atoms_t: tf.Tensor,
    logits_q_t: tf.Tensor,
    q_t_selector: tf.Tensor,
) -> base.LossOutput:
    """Implements Distributional Double Q-learning as TensorFlow ops.

    The function assumes categorical value distributions parameterized by logits,
    and combines distributional RL with double Q-learning.

    See "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
    Hessel, Modayil, van Hasselt, Schaul et al.
    (https://arxiv.org/abs/1710.02298).

    Args:
      atoms_tm1: 1-D tensor containing atom values for first timestep,
        shape `[num_atoms]`.
      logits_q_tm1: Tensor holding logits for first timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      atoms_t: 1-D tensor containing atom values for second timestep,
        shape `[num_atoms]`.
      logits_q_t: Tensor holding logits for second timestep in a batch of
        transitions, shape `[B, action_dim, num_atoms]`.
      q_t_selector: Tensor holding another set of Q-values for second timestep
        in a batch of transitions, shape `[B, action_dim]`.
        These values are used for estimating the best action. In Double DQN they
        come from the online network.

    Returns:
      A namedtuple with fields:

      * `loss`: Tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_atoms]` .

    Raises:
      ValueError: If the tensors do not have the correct rank or compatibility.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(logits_q_tm1, 3, tf.float32)
    base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
    base.assert_rank_and_dtype(r_t, 1, tf.float32)
    base.assert_rank_and_dtype(discount_t, 1, tf.float32)
    base.assert_rank_and_dtype(logits_q_t, 3, tf.float32)
    base.assert_rank_and_dtype(q_t_selector, 2, tf.float32)
    base.assert_rank_and_dtype(atoms_tm1, 1, tf.float32)
    base.assert_rank_and_dtype(atoms_t, 1, tf.float32)

    base.assert_batch_dimension(a_tm1, logits_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(logits_q_t, logits_q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, logits_q_tm1.shape[0])
    base.assert_batch_dimension(atoms_tm1, logits_q_tm1.shape[-1])
    base.assert_batch_dimension(atoms_t, logits_q_tm1.shape[-1])

    # Categorical distributional double Q-learning op.
    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, None] + discount_t[:, None] * atoms_t[None, :]

    # Convert logits to distribution, then find greedy policy action in
    # state s_t.
    q_t_probs = tf.nn.softmax(logits_q_t, axis=-1)
    pi_t = tf.argmax(q_t_selector, axis=1)
    # Compute distribution for greedy action.
    p_target_z = _slice_with_actions(q_t_probs, pi_t)

    # Project using the Cramer distance
    target_tm1 = l2_project(target_z, p_target_z, atoms_tm1)

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_tm1, logits=logit_qa_tm1)

    return base.LossOutput(loss, Extra(target_tm1))


def huber_loss(x: tf.Tensor, k: float = 1.0) -> tf.Tensor:
    """Returns huber-loss."""
    return tf.where(tf.abs(x) < k, 0.5 * tf.square(x), k * (tf.abs(x) - 0.5 * k))


def _quantile_regression_loss(
    dist_src: tf.Tensor,
    tau_src: tf.Tensor,
    dist_target: tf.Tensor,
    huber_param: float = 0.0,
) -> tf.Tensor:
    """Compute (Huber) QR loss between two discrete quantile-valued distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
      dist_src: source probability distribution, shape `[B, num_taus]`.
      tau_src: source distribution probability thresholds, shape `[B, num_taus]`.
      dist_target: target probability distribution, shape `[B, num_taus]`.
      huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

    Returns:
      Quantile regression loss.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(dist_src, 2, tf.float32)
    base.assert_rank_and_dtype(tau_src, 2, tf.float32)
    base.assert_rank_and_dtype(dist_target, 2, tf.float32)

    base.assert_batch_dimension(tau_src, dist_src.shape[0])
    base.assert_batch_dimension(dist_target, dist_src.shape[0])

    # Calculate quantile error.
    delta = tf.expand_dims(dist_target, axis=1) - tf.expand_dims(dist_src, axis=-1)

    delta_neg = tf.cast(delta < 0.0, dtype=tf.float32)
    weight = tf.abs(tf.expand_dims(tau_src, axis=-1) - delta_neg)

    # Calculate Huber loss.
    if huber_param > 0.0:
        loss = huber_loss(delta, huber_param)
    else:
        loss = tf.abs(delta)
    loss *= weight

    # Averaging over target-samples dimension, sum over src-samples dimension.
    return tf.reduce_sum(tf.reduce_mean(loss, axis=-1), axis=1)


def quantile_q_learning(
    dist_q_tm1: tf.Tensor,
    tau_q_tm1: tf.Tensor,
    a_tm1: tf.Tensor,
    r_t: tf.Tensor,
    discount_t: tf.Tensor,
    dist_q_t: tf.Tensor,
    huber_param: float = 0.0,
) -> base.LossOutput:
    """Implements Q-learning for quantile-valued Q distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
      dist_q_tm1: Tensor holding Q distribution at time t-1, shape `[B, num_taus, action_dim]`.
      tau_q_tm1: Q distribution probability thresholds, , shape `[B, num_taus]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      dist_q_t: Tensor holding target Q distribution at time t, shape `[B, num_taus, action_dim]`.
      huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

    Returns:
      A namedtuple with fields:

      * `loss`: Tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_taus]` .

    """

    # Rank and compatibility checks.
    base.assert_rank_and_dtype(dist_q_tm1, 3, tf.float32)
    base.assert_rank_and_dtype(tau_q_tm1, 2, tf.float32)
    base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
    base.assert_rank_and_dtype(r_t, 1, tf.float32)
    base.assert_rank_and_dtype(discount_t, 1, tf.float32)
    base.assert_rank_and_dtype(dist_q_t, 3, tf.float32)

    base.assert_batch_dimension(a_tm1, dist_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(dist_q_t, dist_q_tm1.shape[0])

    # Quantile Regression q learning op.
    # Only update the taken actions.
    dist_qa_tm1 = base.batched_index(dist_q_tm1, a_tm1, 2)  # [batch_size, num_taus]

    # Select target action according to greedy policy w.r.t. q_t_selector.
    q_t_selector = tf.reduce_mean(dist_q_t, axis=1)  # q_t_values
    a_t = tf.argmax(q_t_selector, axis=1)
    dist_qa_t = base.batched_index(dist_q_t, a_t, 2)  # [batch_size, num_taus]

    # Compute target, do not backpropagate into it.
    dist_target_tm1 = r_t[:, None] + discount_t[:, None] * dist_qa_t  # [batch_size, num_taus]

    loss = _quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target_tm1, huber_param)
    return base.LossOutput(loss, Extra(dist_target_tm1))


def quantile_double_q_learning(
    dist_q_tm1: tf.Tensor,
    tau_q_tm1: tf.Tensor,
    a_tm1: tf.Tensor,
    r_t: tf.Tensor,
    discount_t: tf.Tensor,
    dist_q_t: tf.Tensor,
    q_t_selector: tf.Tensor,
    huber_param: float = 0.0,
) -> base.LossOutput:
    """Implements Q-learning for quantile-valued Q distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
      dist_q_tm1: Tensor holding Q distribution at time t-1, shape `[B, num_taus, action_dim]`.
      tau_q_tm1: Q distribution probability thresholds, , shape `[B, num_taus]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      dist_q_t: Tensor holding target Q distribution at time t, shape `[B, num_taus, action_dim]`.
      q_t_selector: Tensor holding Q distribution at time t for selecting greedy action in
        target policy. This is separate from dist_q_t as in Double Q-Learning, but
        can be computed with the target network and a separate set of samples,
        shape `[B, num_taus, action_dim]`.
      huber_param: Huber loss parameter, defaults to 0 (no Huber loss).

    Returns:
      A namedtuple with fields:

      * `loss`: Tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[B, num_taus]` .

    """

    # Rank and compatibility checks.
    base.assert_rank_and_dtype(dist_q_tm1, 3, tf.float32)
    base.assert_rank_and_dtype(tau_q_tm1, 2, tf.float32)
    base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
    base.assert_rank_and_dtype(r_t, 1, tf.float32)
    base.assert_rank_and_dtype(discount_t, 1, tf.float32)
    base.assert_rank_and_dtype(dist_q_t, 3, tf.float32)
    base.assert_rank_and_dtype(q_t_selector, 3, tf.float32)

    base.assert_batch_dimension(a_tm1, dist_q_tm1.shape[0])
    base.assert_batch_dimension(r_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(dist_q_t, dist_q_tm1.shape[0])
    base.assert_batch_dimension(q_t_selector, dist_q_tm1.shape[0])

    # Quantile Regression double q learning op.
    # Only update the taken actions.
    dist_qa_tm1 = base.batched_index(dist_q_tm1, a_tm1, 2)  # [batch_size, num_taus]

    # Select target action according to greedy policy w.r.t. q_t_selector.
    q_t_selector = tf.reduce_mean(q_t_selector, axis=1)
    a_t = tf.argmax(q_t_selector, axis=1)
    dist_qa_t = base.batched_index(dist_q_t, a_t, 2)  # [batch_size, num_taus]

    # Compute target, do not backpropagate into it.
    dist_target_tm1 = r_t[:, None] + discount_t[:, None] * dist_qa_t  # [batch_size, num_taus]

    loss = _quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target_tm1, huber_param)
    return base.LossOutput(loss, Extra(dist_target_tm1))


def retrace(
    q_tm1: tf.Tensor,
    q_t: tf.Tensor,
    a_tm1: tf.Tensor,
    a_t: tf.Tensor,
    r_t: tf.Tensor,
    discount_t: tf.Tensor,
    pi_t: tf.Tensor,
    mu_t: tf.Tensor,
    lambda_: float,
    eps: float = 1e-8,
) -> base.LossOutput:
    """Calculates Retrace errors.

    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).

    Args:
      q_tm1: Q-values at time t-1, this is from the online Q network, shape [T, B, action_dim].
      q_t: Q-values at time t, this is often from the target Q network, shape [T, B, action_dim].
      a_tm1: action index at time t-1, the action the agent took in state s_tm1, shape [T, B].
      a_t: action index at time t, the action the agent took in state s_t, shape [T, B].
      r_t: reward at time t, for state-action pair (s_tm1, a_tm1), shape [T, B].
      discount_t: discount at time t, shape [T, B].
      pi_t: target policy probs at time t, shape [T, B, action_dim].
      mu_t: behavior policy probs at time t, shape [T, B, action_dim].
      lambda_: scalar mixing parameter lambda.
      eps: small value to add to mu_t for numerical stability.

    Returns:
      * `loss`: Tensor containing the batch of losses, shape `[T, B]`.
      * `extra`: a namedtuple with fields:
          * `target`:  Tensor containing the values that `q_tm1` at actions
          `a_tm1` are regressed towards, shape `[T, B]` .
          * `td_error`: batch of temporal difference errors, shape `[T, B]`
    """

    base.assert_rank_and_dtype(q_tm1, 3, tf.float32)
    base.assert_rank_and_dtype(q_t, 3, tf.float32)
    base.assert_rank_and_dtype(a_tm1, 2, tf.int64)
    base.assert_rank_and_dtype(a_t, 2, tf.int64)
    base.assert_rank_and_dtype(r_t, 2, tf.float32)
    base.assert_rank_and_dtype(discount_t, 2, tf.float32)
    base.assert_rank_and_dtype(pi_t, 3, tf.float32)
    base.assert_rank_and_dtype(mu_t, 2, tf.float32)

    pi_a_t = base.batched_index(pi_t, a_t)
    c_t = tf.minimum(tf.constant(1.0), pi_a_t / (mu_t + eps)) * lambda_

    target_tm1 = multistep.general_off_policy_returns_from_action_values(q_t, a_t, r_t, discount_t, c_t, pi_t)

    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    td_error = target_tm1 - qa_tm1
    loss = 0.5 * tf.square(td_error)

    return base.LossOutput(loss, QExtra(target=target_tm1, td_error=td_error))
