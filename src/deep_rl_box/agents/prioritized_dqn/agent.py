"""Prioritized Experience Replay DQN agent class.

From the paper "Prioritized Experience Replay" http://arxiv.org/abs/1511.05952.

This agent combines:

*   Double Q-learning
*   TD n-step bootstrap
*   Prioritized experience replay
"""
from typing import Callable, Tuple
import numpy as np
import tensorflow as tf

import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
import deep_rl_box.utils.value_learning as rl
from deep_rl_box.utils import base


class PrioritizedDqn(types_lib.Agent):
    """Prioritized DQN agent"""

    def __init__(
        self,
        network: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        random_state: np.random.RandomState,  
        replay: replay_lib.PrioritizedReplay, # <<== changed
        transition_accumulator: replay_lib.TransitionAccumulator,
        exploration_epsilon: Callable[[int], float],
        learn_interval: int,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        action_dim: int,
        discount: float,
        clip_grad: bool,
        max_grad_norm: float,
    ):
        """
        Args:
            network: the Q network we want to optimize.
            optimizer: the optimizer for Q network.
            random_state: used to sample random actions for e-greedy policy.
            replay: prioritized experience replay.
            transition_accumulator: external helper class to build n-step transition.
            exploration_epsilon: external schedule of e in e-greedy exploration rate.
            learn_interval: the frequency (measured in agent steps) to do learning.
            target_net_update_interval: the frequency (measured in number of online Q network parameter updates)
                 to Update target network parameters.
            min_replay_size: Minimum replay size before start to do learning.
            batch_size: sample batch size.
            action_dim: number of valid actions in the environment.
            discount: gamma discount for future rewards.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
        """
        if not 1 <= learn_interval:
            raise ValueError(f'Expect learn_interval to be positive integer, got {learn_interval}')
        if not 1 <= target_net_update_interval:
            raise ValueError(f'Expect target_net_update_interval to be positive integer, got {target_net_update_interval}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be positive integer, got {min_replay_size}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size [1, 512], got {batch_size}')
        if not batch_size <= min_replay_size <= replay.capacity:
            raise ValueError(f'Expect min_replay_size >= {batch_size} and <= {replay.capacity} and, got {min_replay_size}')
        if not 0 < action_dim:
            raise ValueError(f'Expect action_dim to be positive integer, got {action_dim}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount [0.0, 1.0], got {discount}')

        self.agent_name = 'PER-DQN'

        self._random_state = random_state
        self._action_dim = action_dim

        # Online Q network
        self._online_network = network
        self._optimizer = optimizer

        # create target Q network
        online_network_config = self._online_network.get_config()
        self._target_network = type(self._online_network).from_config(online_network_config)
        self._target_network.set_weights(self._online_network.get_weights())
        # Disable autograd for target network
        self._target_network.trainable = False

        # Experience replay parameters
        self._transition_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._replay = replay
        self._max_seen_priority = 1.0 # <<== added

        # Learning related parameters
        self._discount = discount
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_size = min_replay_size
        self._learn_interval = learn_interval
        self._target_net_update_interval = target_net_update_interval
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, do a action selection and a series of learn related activities"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try build transition and add into replay
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._replay.add(transition, priority=self._max_seen_priority) # <<== extra argument

        # Return if replay is ready
        if self._replay.size < self._min_replay_size:
            return a_t

        # Start to learn
        if self._step_t % self._learn_interval == 0:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        self.exploration_epsilon = self._exploration_epsilon(self._step_t)
        if self._random_state.rand() <= self.exploration_epsilon:
            # randint() returns random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._action_dim)
            return a_t
        
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        q_values = self._choose_action(s_t)
        a_t = tf.argmax(q_values, axis=-1)
        return np.array(a_t).item()

    @tf.function
    def _choose_action(self, s_t: tf.Tensor) -> tf.Tensor:
        """
        Choose action by following the e-greedy policy with respect to Q values
        Args:
            timestep: the current timestep from env
            epsilon: the e in e-greedy exploration
        Returns:
            a_t: the action to take at s_t
        """
        q_values = self._online_network(s_t).q_values
        return q_values

    def _learn(self) -> None:
        """Sample a batch of transitions and learn."""
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update(transitions, weights) # <<== weights added as extra argument

        # Update target network parameters
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape {self._batch_size}, got {priorities.shape}')
        
        priorities = tf.abs(priorities)
        self._max_seen_priority = tf.reduce_max([self._max_seen_priority, tf.reduce_max(priorities)])
        self._replay.update_priorities(indices, priorities)

    def _update(self, transitions: replay_lib.Transition, weights: np.ndarray) -> np.ndarray:
        weights = tf.constant(weights, dtype=tf.float32) # [batch_size]
        base.assert_rank_and_dtype(weights, 1, tf.float32)

        with tf.GradientTape() as tape:
            loss, priorities = self._calc_loss(transitions)
            # Multiply loss by sampling weights, averaging over batch dimension
            loss = tf.reduce_mean(loss * tf.stop_gradient(weights))
        grads = tape.gradient(loss, self._online_network.trainable_variables)
        if self._clip_grad:
            grads = [tf.clip_by_norm(g, self._max_grad_norm) for g in grads]
        self._optimizer.apply_gradients(zip(grads, self._online_network.trainable_variables))
        self._update_t += 1

        # For logging only.
        self._loss_t = loss.numpy().item()

        return priorities

    def _calc_loss(self, transitions: replay_lib.Transition) -> Tuple[tf.Tensor, np.ndarray]:
        """Calculate loss for a given batch of transitions"""
        s_tm1 = tf.convert_to_tensor(transitions.s_tm1, dtype=tf.float32)  # [batch_size, state_shape]
        a_tm1 = tf.convert_to_tensor(transitions.a_tm1, dtype=tf.int64)  # [batch_size]
        r_t = tf.convert_to_tensor(transitions.r_t, dtype=tf.float32)  # [batch_size]
        s_t = tf.convert_to_tensor(transitions.s_t, dtype=tf.float32)  # [batch_size, state_shape]
        done = tf.convert_to_tensor(transitions.done, dtype=tf.bool)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), tf.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)
        base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
        base.assert_rank_and_dtype(r_t, 1, tf.float32)
        base.assert_rank_and_dtype(done, 1, tf.bool)

        discount_t = tf.cast(~done, tf.float32) * self._discount

        # Compute predicted q values for s_tm1, using online Q network
        q_tm1 = self._online_network(s_tm1).q_values  # [batch_size, action_dim]

        # Compute predicted q values for s_t, using target Q network and double Q
        q_t_selector = self._online_network(s_t).q_values  # [batch_size, action_dim]
        target_q_t = self._target_network(s_t).q_values  # [batch_size, action_dim]

        # Compute loss which is 0.5 * square(td_errors)
        loss_output = rl.double_qlearning(q_tm1, a_tm1, r_t, discount_t, target_q_t, q_t_selector)
        # Averaging over batch dimension
        loss = tf.reduce_mean(loss_output.loss, axis=0)

        # Extract TD errors as priorities.
        priorities = tf.stop_gradient(loss_output.extra.td_error).numpy()  # [batch_size]

        return loss, priorities

    def _update_target_network(self):
        """Copy online network parameters to target network."""
        self._target_network.set_weights(self._online_network.get_weights())
        self._target_update_t += 1

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._optimizer.param_groups[0]['lr'],
            'loss': self._loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
            'exploration_epsilon': self.exploration_epsilon,
        }
