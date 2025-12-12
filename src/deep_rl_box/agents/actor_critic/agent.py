"""Actor-Critic agent class.

From the paper "Actor-Critic Algorithms"
https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf.
"""
import collections
import numpy as np
import tensorflow as tf

import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
import deep_rl_box.utils.policy_gradient as rl
from deep_rl_box.utils import base
from deep_rl_box.utils import distributions


class ActorCritic(types_lib.Agent):
    """Actor-Critic agent"""

    def __init__(
        self,
        policy_network: tf.keras.Model,
        policy_optimizer: tf.keras.optimizers.Optimizer,
        transition_accumulator: replay_lib.TransitionAccumulator,
        discount: float,
        batch_size: int,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            transition_accumulator: external helper class to build n-step transition.
            discount: the gamma discount for future rewards.
            batch_size: sample batch_size of transitions.
            entropy_coef: the coefficient of entropy loss.
            value_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to be [0.0, 1.0], got {discount}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to be [1, 512], got {batch_size}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to be (0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= value_coef <= 1.0:
            raise ValueError(f'Expect value_coef to be (0.0, 1.0], got {value_coef}')

        self.agent_name = 'Actor-Critic'
        self._policy_network = policy_network
        self._policy_optimizer = policy_optimizer
        self._discount = discount

        self._transition_accumulator = transition_accumulator
        self._batch_size = batch_size

        self._storage = collections.deque(maxlen=1000)

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Agent take a step at timestep, return the action a_t,
        and record episode trajectory, start to learn when the replay is ready"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try to build transition
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._storage.append(transition)

        # Start learning when replay reach batch_size limit
        if len(self._storage) >= self._batch_size:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        s_t = tf.constant(timestep.observation[None, ...], dtype=tf.float32)
        a_t = self._choose_action(s_t)
        return np.array(a_t).item()

    @tf.function
    def _choose_action(self, s_t: tf.Tensor) -> tf.Tensor:
        """Given timestep, choose action a_t"""
        logits_t = self._policy_network(s_t).pi_logits
        # Sample an action
        a_t = distributions.categorical_distribution(logits_t).sample()
        return a_t

    def _learn(self) -> None:
        transitions = list(self._storage)
        transitions = replay_lib.np_stack_list_of_transitions(transitions, replay_lib.TransitionStructure, 0)
        self._update(transitions)
        self._storage.clear()  # discard old samples after using it

    def _update(self, transitions: replay_lib.Transition) -> None:
        s_tm1 = tf.constant(transitions.s_tm1, dtype=tf.float32)  # [batch_size, state_shape]
        a_tm1 = tf.constant(transitions.a_tm1, dtype=tf.int64)  # [batch_size]
        r_t = tf.constant(transitions.r_t, dtype=tf.float32)  # [batch_size]
        s_t = tf.constant(transitions.s_t, dtype=tf.float32)  # [batch_size, state_shape]
        done = tf.constant(transitions.done, dtype=tf.bool) # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), tf.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)
        base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
        base.assert_rank_and_dtype(r_t, 1, tf.float32)
        base.assert_rank_and_dtype(done, 1, tf.bool)

        discount_t = tf.cast(~done, dtype=tf.float32) * self._discount

        with tf.GradientTape() as tape:
            loss = self._calc_loss(s_tm1, a_tm1, r_t, s_t, discount_t)
        grads = tape.gradient(loss, self._policy_network.trainable_variables)

        if self._clip_grad:
            tf.clip_by_global_norm(grads, self._max_grad_norm)
        
        self._policy_optimizer.apply_gradients(zip(grads, self._policy_network.trainable_variables))
        self._update_t += 1

    def _calc_loss(self, s_tm1, a_tm1, r_t, s_t, discount_t) -> tf.Tensor:
        """Calculate loss summed over the trajectories of a single episode"""
        
        # Get policy action logits and value for s_tm1.
        policy_output = self._policy_network(s_tm1)
        logits_tm1 = policy_output.pi_logits
        value_tm1 = tf.squeeze(policy_output.value, axis=1) # [batch_size]

        # Calculates TD n-step target and advantages.
        
        baseline_s_t = tf.squeeze(self._policy_network(s_t).value, axis=1)  # [batch_size]
        target_baseline = r_t + discount_t * baseline_s_t
        advantages = target_baseline - value_tm1

        # Compute policy gradient a.k.a. log-likelihood loss.
        policy_loss = rl.policy_gradient_loss(logits_tm1, a_tm1, advantages).loss

        # Compute entropy loss.
        entropy_loss = rl.entropy_loss(logits_tm1).loss

        # Compute state-value loss.
        value_loss = rl.value_loss(target_baseline, value_tm1).loss

        # Averaging over batch dimension.
        policy_loss = tf.reduce_mean(policy_loss, axis=0)
        entropy_loss = tf.reduce_mean(entropy_loss, axis=0)
        value_loss = tf.reduce_mean(value_loss, axis=0)

        # Combine policy loss, value loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + self._entropy_coef * entropy_loss) + self._value_coef * value_loss

        # For logging only.
        self._policy_loss_t = np.array(policy_loss).item()
        self._value_loss_t = np.array(value_loss).item()
        self._entropy_loss_t = np.array(entropy_loss).item()

        return loss

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
        }
