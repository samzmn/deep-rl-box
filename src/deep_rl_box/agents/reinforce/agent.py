"""REINFORCE agent class.

From the paper "Policy Gradient Methods for Reinforcement Learning with Function Approximation"
https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf.
"""
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras

# pylint: disable=import-error
import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
import deep_rl_box.utils.policy_gradient as rl
from deep_rl_box.utils import base
from deep_rl_box.utils import distributions

class Reinforce(types_lib.Agent):
    """Reinforce agent"""

    def __init__(
        self,
        policy_network: tf.keras.Model,
        policy_optimizer: tf.keras.optimizers.Optimizer,
        discount: float,
        transition_accumulator: replay_lib.TransitionAccumulator,
        normalize_returns: bool,
        clip_grad: bool,
        max_grad_norm: float,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            discount: the gamma discount for future rewards.
            transition_accumulator: external helper class to build n-step transition.
            normalize_returns: if True, normalize episode returns.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: TensorFlow runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')

        self.agent_name = 'REINFORCE'
        self._policy_network = policy_network
        self._policy_optimizer = policy_optimizer
        self._discount = discount

        self._transition_accumulator = transition_accumulator
        self._trajectory = collections.deque(maxlen=108000) # FILO
        self._normalize_returns = normalize_returns
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._loss_t = np.nan

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Agent take a step at timestep, return the action a_t,
        and record episode trajectory, learn after the episode terminated"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try to build transition and add into episodic replay
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._trajectory.append(transition)

        # Start to learn
        if timestep.done:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()
        self._trajectory.clear()

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        logits_t = self._choose_action(s_t)
        # Sample an action
        a_t = distributions.categorical_distribution(logits_t).sample()
        return np.array(a_t).item()

    @tf.function
    def _choose_action(self, s_t: tf.Tensor) -> tf.Tensor:
        """Given timestep, choose action a_t"""
        logits_t = self._policy_network(s_t).pi_logits
        return logits_t

    def _learn(self) -> None:
        # Turn entire episode trajectory into one Transition object
        transitions = replay_lib.np_stack_list_of_transitions(list(self._trajectory), replay_lib.TransitionStructure)
        self._update(transitions)

    def _update(self, transitions: replay_lib.Transition) -> None:
        with tf.GradientTape() as tape:
            loss = self._calc_loss(transitions)
        grads = tape.gradient(loss, self._policy_network.trainable_variables)

        if self._clip_grad:
            grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        self._policy_optimizer.apply_gradients(zip(grads, self._policy_network.trainable_variables))
        self._update_t += 1

        # For logging only.
        self._loss_t = loss.numpy()

    def _calc_loss(self, transitions: replay_lib.Transition) -> tf.Tensor:
        """Calculate loss summed over the trajectories of a single episode"""
        s_tm1 = tf.convert_to_tensor(transitions.s_tm1, dtype=tf.float32)  # [batch_size, state_shape]
        a_tm1 = tf.convert_to_tensor(transitions.a_tm1, dtype=tf.int64)  # [batch_size]
        r_t = tf.convert_to_tensor(transitions.r_t, dtype=tf.float32)  # [batch_size]
        done = tf.convert_to_tensor(transitions.done, dtype=tf.bool)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), tf.float32)
        base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
        base.assert_rank_and_dtype(r_t, 1, tf.float32)
        base.assert_rank_and_dtype(done, 1, tf.bool)

        # Compute episode returns.
        discount_t = tf.cast(~done, tf.float32) * self._discount # In Python, the ~ symbol is the bitwise NOT operator

        seq_len = len(r_t)
        returns = tf.TensorArray(dtype=tf.float32, size=seq_len)
        g = 0.0
        # Calculate returns from t=T-1, T-2, ..., 1, 0
        for t in reversed(range(0, seq_len)):
            g = r_t[t] + discount_t[t] * g
            returns = returns.write(t, g)
        returns = returns.stack()

        # Get policy action logits for s_tm1.
        logits_tm1 = self._policy_network(s_tm1).pi_logits

        # Compute policy gradient a.k.a. log-likelihood loss.
        loss = rl.policy_gradient_loss(logits_tm1, a_tm1, returns).loss

        # Averaging over batch dimension.
        # Negative sign to indicate we want to maximize the policy gradient objective function
        loss = -tf.reduce_mean(loss, axis=0)

        return loss

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            'policy_loss': self._loss_t,
            'updates': self._update_t,
        }
