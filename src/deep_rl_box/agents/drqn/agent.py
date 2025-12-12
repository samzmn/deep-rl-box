"""DRQN agent class.

From the paper "Deep Recurrent Q-Learning for Partially Observable MDPs"
https://arxiv.org/abs/1507.06527.

This agent combines:
*   Double Q-learning
*   TD n-step bootstrap
"""
from typing import Callable
import numpy as np
import tensorflow as tf

import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
import deep_rl_box.utils.value_learning as rl
from deep_rl_box.utils import base
from deep_rl_box.networks.value import DrqnMlpNet
from deep_rl_box.networks.common import clone_network, disable_trainability

class Drqn(types_lib.Agent):
    'DRQN agent'

    def __init__(
        self,
        network: DrqnMlpNet,
        optimizer: tf.optimizers.Optimizer,
        random_state: np.random.RandomState,
        replay: replay_lib.UniformReplay,
        transition_accumulator: replay_lib.TransitionAccumulator,
        exploration_epsilon: Callable[[int], float],
        learn_interval: int,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        action_dim: int,
        discount: float,
        unroll_length: int,
        clip_grad: bool,
        max_grad_norm: float,
    ):
        """
        Args:
            network: the Q network we want to optimize.
            optimizer: the optimizer for Q network.
            random_state: used to sample random actions for e-greedy policy.
            replay: experience replay.
            transition_accumulator: external helper class to build n-step transition.
            exploration_epsilon: external schedule of e in e-greedy exploration rate.
            learn_interval: the frequency (measured in agent steps) to do learning.
            target_net_update_interval: the frequency (measured in number of online Q network parameter updates)
                 to Update target network parameters.
            min_replay_size: Minimum replay size before start to do learning.
            batch_size: sample batch size.
            action_dim: number of valid actions in the environment.
            discount: gamma discount for future rewards.
            unroll_length: the unroll transition length
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
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
        if not 0 < unroll_length:
            raise ValueError(f'Expect unroll_length to be positive integer, got {unroll_length}')

        self.agent_name = 'DRQN'
        self._random_state = random_state
        self._action_dim = action_dim

        # Online Q network
        self._online_network = network
        self._optimizer = optimizer

        # Lazy way to create target Q network
        self._target_network = clone_network(self._online_network)
        # Disable autograd for target network
        disable_trainability(self._target_network)

        # Experience replay parameters
        self._transition_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._replay = replay

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=1,
            structure=replay_lib.TransitionStructure,
            cross_episode=True,
        )

        # Learning related parameters
        self._discount = discount
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_size = min_replay_size
        self._learn_interval = learn_interval
        self._target_net_update_interval = target_net_update_interval
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        self._lstm_state = None  # Stores LSTM hidden state and cell state

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, do a action selection and a series of learn related activities"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try build transition and add into intermediate queue
        for transition in self._transition_accumulator.step(timestep, a_t):
            unrolled_transition = self._unroll.add(transition, timestep.done)
            if unrolled_transition is not None:
                self._replay.add(unrolled_transition)

        # Return if replay is not ready
        if self._replay.size < self._min_replay_size:
            return a_t

        # Start to learn
        if self._step_t % self._learn_interval == 0:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()
        self._lstm_state = self._online_network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        a_t = self._choose_action(timestep, self.exploration_epsilon)
        return a_t

    def _choose_action(self, timestep: types_lib.TimeStep, epsilon: float) -> types_lib.Action:
        """
        Choose action by following the e-greedy policy with respect to Q values
        Args:
            timestep: the current timestep from env
            epsilon: the e in e-greedy exploration
        Returns:
            a_t: the action to take at s_t
        """

        if self._random_state.rand() <= epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._action_dim)
            return a_t

        s_t = tf.constant(timestep.observation[None, None, ...], dtype=tf.float32)
        network_output = self._online_network(s_t, self._lstm_state)
        self._lstm_state = network_output.hidden_s
        q_values = network_output.q_values
        a_t = tf.argmax(q_values, axis=-1)
        return a_t.numpy().item()

    def _learn(self):
        """Sample a batch of transitions and learn."""
        transitions = self._replay.sample(self._batch_size)
        self._update(transitions)

        # Update target network parameters
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()

    def _update(self, transitions):
        with tf.GradientTape() as tape:
            loss = self._calc_loss(transitions)
        
        grads = tape.gradient(loss, self._online_network.trainable_variables)

        if self._clip_grad:
            grads = [tf.clip_by_norm(grad, self._max_grad_norm) for grad in grads]
            
        self._optimizer.apply_gradients(zip(grads, self._online_network.trainable_variables))
        self._update_t += 1

        # For logging only.
        self._loss_t = loss.numpy().item()

    def _calc_loss(self, transitions: replay_lib.Transition) -> tf.Tensor:
        """Calculate loss for a given batch of transitions"""
        s_tm1 = tf.constant(transitions.s_tm1, dtype=tf.float32)  # [B, T, state_shape]
        a_tm1 = tf.constant(transitions.a_tm1, dtype=tf.int64)  # [B, T]
        r_t = tf.constant(transitions.r_t, dtype=tf.float32)  # [B, T]
        s_t = tf.constant(transitions.s_t, dtype=tf.float32)  # [B, T, state_shape]
        done = tf.constant(transitions.done, dtype=tf.bool)  # [B, T]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_tm1, (3, 5), tf.float32)
        base.assert_rank_and_dtype(s_t, (3, 5), tf.float32)
        base.assert_rank_and_dtype(a_tm1, 2, tf.int64)
        base.assert_rank_and_dtype(r_t, 2, tf.float32)
        base.assert_rank_and_dtype(done, 2, tf.bool)

        # Using zero start for RNN
        hidden_s = self._online_network.get_initial_hidden_state(batch_size=self._batch_size)
        target_hidden_s = tuple([tf.identity(hx) for hx in hidden_s])

        discount_t = tf.cast(~done, tf.float32) * self._discount

        # Compute predicted q values for s_tm1, using online Q network
        q_tm1 = self._online_network(s_tm1, hidden_s).q_values  # [B, T, action_dim]

        # Compute predicted q values for s_t, using target Q network and double Q
        q_t_selector = self._online_network(s_t, hidden_s).q_values  # [batch_size, action_dim]
        target_q_t = self._target_network(s_t, target_hidden_s).q_values  # [batch_size, action_dim]

        # Merge batch and time dimensions.
        B, T = s_tm1.shape[:2]
        q_tm1 = tf.reshape(q_tm1, (B * T, -1))
        a_tm1 = tf.reshape(a_tm1, (B * T))
        r_t = tf.reshape(r_t, (B * T))
        discount_t = tf.reshape(discount_t, (B * T))
        target_q_t = tf.reshape(target_q_t, (B * T, -1))
        q_t_selector = tf.reshape(q_t_selector, (B * T, -1))

        # Compute loss which is 0.5 * square(td_errors)
        loss = rl.double_qlearning(q_tm1, a_tm1, r_t, discount_t, target_q_t, q_t_selector).loss

        # Averaging over batch dimension
        loss = tf.reduce_mean(loss, axis=0)
        return loss

    def _update_target_network(self):
        """Copy online network parameters to target network."""
        self._target_network.set_weights(self._online_network.get_weights())
        self._target_update_t += 1

    @property
    def exploration_epsilon(self):
        """Call external schedule function"""
        return self._exploration_epsilon(self._step_t)

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            'loss': self._loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
            'exploration_epsilon': self.exploration_epsilon,
        }
