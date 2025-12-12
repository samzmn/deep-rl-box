"""SAC (for Discrete Action) agent class.

From the paper "Soft Actor-Critic for Discrete Action Settings"
https://arxiv.org/abs/1910.07207.

From the paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
https://arxiv.org/abs/1801.01290.
"""

from typing import Iterable, Mapping, Tuple, Text
import copy
import multiprocessing
import numpy as np
import tensorflow as tf
#from torch import nn
#import torch.nn.functional as F
#from torch.distributions import Categorical

# pylint: disable=import-error
import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
from deep_rl_box.utils import value_learning
from deep_rl_box.utils import base


class Actor(types_lib.Agent):
    """SAC actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: tf.keras.Model,
        transition_accumulator: replay_lib.TransitionAccumulator,
        min_replay_size: int,
        action_dim: int,
        device: str,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            transition_accumulator: external helper class to build n-step transition.
            min_replay_size: minimum replay size before do learning.
            action_dim: number of actions for the environment.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """

        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer greater than or equal to 1, got {min_replay_size}')
        if not 1 <= action_dim:
            raise ValueError(f'Expect action_dim to be integer greater than or equal to 1, got {action_dim}')

        self.rank = rank
        self.agent_name = f'SAC-actor{rank}'
        self._policy_network = policy_network
        # Disable autograd for actor networks.
        for layer in self._policy_network.layers:
            layer.trainable = False

        self._device = device

        self._shared_params = shared_params

        self._queue = data_queue
        self._transition_accumulator = transition_accumulator
        self._action_dim = action_dim
        self._min_replay_size = min_replay_size

        self._step_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try build transition and add to global queue
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._queue.put(transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()
        self._update_actor_network()

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        s_t = tf.constant(timestep.observation[None, ...], dtype=tf.float32)
        a_t = self._choose_action(s_t)
        return np.array(a_t).item()

    def _update_actor_network(self):
        weights = self._shared_params['policy_network']
        if weights is not None:
            self._policy_network.set_weights(weights)

    def _choose_action(self, s_t: tf.Tensor) -> tf.Tensor:
        """Given timestep, choose action a_t"""
        if self._step_t < self._min_replay_size:  # Act randomly when staring out.
            a_t = np.random.randint(0, self._action_dim)
            return a_t
        else:
            logits_t = self._policy_network(s_t).pi_logits
            a_t = tf.random.categorical(logits=logits_t, num_samples=1)
            return a_t

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {}


class Learner(types_lib.Learner):
    """SAC learner"""

    def __init__(
        self,
        replay: replay_lib.UniformReplay,
        policy_network: tf.keras.Model,
        policy_optimizer: tf.keras.optimizers.Optimizer,
        q1_network: tf.keras.Model,
        q1_optimizer: tf.keras.optimizers.Optimizer,
        q2_network: tf.keras.Model,
        q2_optimizer: tf.keras.optimizers.Optimizer,
        discount: float,
        batch_size: int,
        action_dim: int,
        min_replay_size: int,
        learn_interval: int,
        q_target_tau: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: str,
        shared_params: dict,
    ) -> None:
        """
        Args:
            replay: simple experience replay to store transitions.
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            q1_network: the first Q network.
            q1_optimizer: the optimizer for the first Q network.
            q2_network: the second Q network.
            q2_optimizer: the optimizer for the second Q network.
            discount: the gamma discount for future rewards.
            batch_size: sample batch_size of transitions.
            action_dim: number of actions for the environment.
            min_replay_size: minimum replay size before do learning.
            learn_interval: how often should the agent learn.
            q_target_tau: the coefficient of target Q network parameters.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= action_dim:
            raise ValueError(f'Expect action_dim to be integer greater than or equal to 1, got {action_dim}')
        if not 1 <= learn_interval:
            raise ValueError(f'Expect learn_interval to be positive integer, got {learn_interval}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be positive integer, got {min_replay_size}')
        if not batch_size <= min_replay_size <= replay.capacity:
            raise ValueError(f'Expect min_replay_size >= {batch_size} and <= {replay.capacity} and, got {min_replay_size}')
        if not 0.0 < q_target_tau <= 1.0:
            raise ValueError(f'Expect q_target_tau to (0.0, 1.0], got {q_target_tau}')

        self.agent_name = 'SAC-learner'
        self._device = device
        self._policy_network = policy_network
        self._policy_optimizer = policy_optimizer
        self._q1_network = q1_network
        self._q1_optimizer = q1_optimizer
        self._q2_network = q2_network
        self._q2_optimizer = q2_optimizer

        # Lazy way to create target Q networks
        self._q1_target_network = q1_network.__class__.from_config(q1_network.get_config())
        self._q2_target_network = q2_network.__class__.from_config(q2_network.get_config())
        # Disable require gradients for target Q networks to improve performance
        for l1, l2 in zip(self._q1_target_network.layers, self._q2_target_network.layers):
            l1.trainable = False
            l2.trainable = False            

        self._shared_params = shared_params

        # Entropy temperature parameters is learned
        # Automating Entropy Adjustment for Maximum Entropy RL section of https://arxiv.org/abs/1812.05905
        self._target_entropy = -tf.math.log(1.0 / action_dim) * 0.98
        # Use log is more numerical stable as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        self._log_ent_coef = tf.math.log(tf.ones(1))
        lr = self._policy_optimizer.learning_rate  # Copy learning rate from policy network optimizer
        self._ent_coef_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self._q_target_tau = q_target_tau

        self._replay = replay
        self._discount = discount
        self._batch_size = batch_size
        self._min_replay_size = min_replay_size
        self._learn_interval = learn_interval
        self._action_dim = action_dim

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._q1_loss_t = np.nan
        self._q2_loss_t = np.nan
        self._policy_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if self._replay.size < self._batch_size or self._step_t % self._batch_size != 0:
            return

        self._learn()
        yield self.statistics

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""
        self._replay.reset()

    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._replay.add(item)

    def get_policy_network_weights(self):
        # To keep things consistent, we move the parameters to CPU
        return self._policy_network.get_weights()

    def _learn(self) -> None:
        # Note we don't clear old samples since this off-policy learning
        transitions = self._replay.sample(self._batch_size)
        self._update(transitions)
        self._update_t += 1

        self._shared_params['policy_network'] = self.get_policy_network_weights()

    def _update(self, transitions: replay_lib.Transition) -> None:
        self._update_q(transitions)  # Policy evaluation
        self._update_pi(transitions)  # Policy improvement
        self._update_target_q_networks()

    def _update_q(self, transitions: replay_lib.Transition) -> None:
        s_tm1 = tf.constant(transitions.s_tm1, dtype=tf.float32)  # [batch_size, state_shape]
        a_tm1 = tf.constant(transitions.a_tm1, dtype=tf.int64)  # [batch_size]
        r_t = tf.constant(transitions.r_t, dtype=tf.float32)  # [batch_size]
        s_t = tf.constant(transitions.s_t, dtype=tf.float32)  # [batch_size, state_shape]
        done = tf.constant(transitions.done, dtype=tf.bool)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), tf.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)
        base.assert_rank_and_dtype(a_tm1, 1, tf.int64)
        base.assert_rank_and_dtype(r_t, 1, tf.float32)
        base.assert_rank_and_dtype(done, 1, tf.bool)

        discount_t = tf.cast(~done, dtype=tf.float32) * self._discount

        with tf.GradientTape(persistent=True) as tape:
            q1_loss, q2_loss = self._calc_q_loss(s_tm1, a_tm1, r_t, s_t, discount_t)

        gradients_q1 = tape.gradient(q1_loss, self._q1_network.trainable_variables)
        gradients_q2 = tape.gradient(q2_loss, self._q2_network.trainable_variables)
        # Manually delete the tape to free resources
        del tape

        if self._clip_grad:
            gradients_q1 = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients_q1]
            gradients_q2 = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients_q2]
        
        self._q1_optimizer.apply_gradients(zip(gradients_q1, self._q1_network.trainable_variables))
        self._q2_optimizer.apply_gradients(zip(gradients_q2, self._q2_network.trainable_variables))

        # For logging only.
        self._q1_loss_t = q1_loss.numpy().item()
        self._q2_loss_t = q2_loss.numpy().item()

    def _update_pi(self, transitions: replay_lib.Transition) -> None:
        """Calculate policy network loss and entropy temperature loss"""
        s_tm1 = tf.constant(transitions.s_tm1, dtype=tf.float32)  # [batch_size, state_shape]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Calculate policy loss
            loss, ent_coef_loss = self._calc_policy_loss(s_tm1)
         # Compute gradients
        gradients_policy = tape.gradient(loss, self._policy_network.trainable_variables)
        gradients_ent_coef = tape.gradient(ent_coef_loss, self._policy_network.trainable_variables)
        # Manually delete the tape to free resources
        del tape
        if self._clip_grad:
            gradients_policy = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients_policy]
            gradients_ent_coef = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients_ent_coef]
        
        # Apply gradients
        self._policy_optimizer.apply_gradients(zip(gradients_policy, self._policy_network.trainable_variables))
        self._ent_coef_optimizer.apply_gradients(zip(gradients_ent_coef, self._policy_network.trainable_variables))

        # For logging only.
        self._policy_loss_t = loss.numpy().item()

    def _calc_q_loss(self, s_tm1, a_tm1, r_t, s_t, discount_t) -> Tuple[tf.Tensor, tf.Tensor]:
        # Calculate estimated q values for state-action pair (s_tm1, a_tm1) using two Q networks.
        q1_tm1 = self._q1_network(s_tm1).q_values
        q2_tm1 = self._q2_network(s_tm1).q_values

        # Get action a_t probabilities for s_t from current policy
        logits_t = self._policy_network(s_t).pi_logits  # [batch_size, action_dim]

        # Calculate log probabilities for all actions
        logprob_t = tf.nn.log_softmax(logits=logits_t, axis=1)
        prob_t = tf.nn.softmax(logits_t, axis=1)

        # Get estimated q values from target Q networks
        q1_s_t = self._q1_target_network(s_t).q_values  # [batch_size, action_dim]
        q2_s_t = self._q2_target_network(s_t).q_values  # [batch_size, action_dim]
        q_s_t = tf.minimum(q1_s_t, q2_s_t)  # [batch_size, action_dim]

        # Calculate soft state-value for s_t with respect to current policy
        target_q_t = prob_t * (q_s_t - self.ent_coef * logprob_t)  # eq 10, (batch_size, action_dim)

        # Compute q loss is 0.5 * square(td_errors)
        q1_loss = value_learning.qlearning(q1_tm1, a_tm1, r_t, discount_t, target_q_t).loss
        q2_loss = value_learning.qlearning(q2_tm1, a_tm1, r_t, discount_t, target_q_t).loss

        # Averaging over batch dimension.
        q1_loss = tf.reduce_mean(q1_loss)
        q2_loss = tf.reduce_mean(q2_loss)


        return q1_loss, q2_loss

    def _calc_policy_loss(self, s_tm1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        

        # Compute action logits for s_tm1.
        logits_tm1 = self._policy_network(s_tm1).pi_logits  # [batch_size, action_dim]
        logprob_tm1 = tf.nn.log_softmax(logits_tm1, axis=1)
        prob_tm1 = tf.nn.softmax(logits_tm1, axis=1)

        # Compute the minimum q values for s_tm1 from the two Q networks.
        q1_tm1 = self._q1_network(s_tm1).q_values
        q2_tm1 = self._q2_network(s_tm1).q_values
        min_q_tm1 = tf.minimum(q1_tm1, q2_tm1)

        # Compute expected q values with action probabilities from current policy.
        q_tm1 = tf.reduce_sum(min_q_tm1 * prob_tm1, axis=1, keepdims=True)

        # Compute entropy temperature parameter loss
        ent_coef_losses = self._log_ent_coef * (logprob_tm1 + tf.cast(self._target_entropy, dtype=tf.float32))  # eq 11, (batch_size, action_dim)

        # Compute SAC policy gradient loss.
        policy_losses = prob_tm1 * (q_tm1 - self.ent_coef * logprob_tm1)  # [batch_size, action_dim]
        # alternative, we can calculate it according to original paper eq 12
        # policy_losses = prob_tm1 * (self.ent_coef * logprob_tm1 - q_tm1)  # eq 12, (batch_size, action_dim)

        # Sum over all actions, averaging over batch dimension.
        # Negative sign to indicate we want to maximize the policy gradient objective function
        ent_coef_loss = -tf.reduce_mean(tf.reduce_sum(ent_coef_losses, axis=-1), axis=0)
        policy_loss = -tf.reduce_mean(tf.reduce_sum(policy_losses, axis=-1), axis=0)

        return policy_loss, ent_coef_loss

    def _update_target_q_networks(self) -> None:
        self._polyak_update_target_q(self._q1_network, self._q1_target_network, self._q_target_tau)
        self._polyak_update_target_q(self._q2_network, self._q2_target_network, self._q_target_tau)

    def _polyak_update_target_q(self, q: tf.keras.Model, target: tf.keras.Model, tau: float) -> None:
        for q_var, target_var in zip(q.trainable_variables, target.trainable_variables):
            target_var.assign(tau * target_var + (1 - tau) * q_var)

    @property
    def ent_coef(self) -> tf.Tensor:
        """Detached entropy temperature parameter, avoid passing into policy or Q networks"""
        return tf.exp(self._log_ent_coef)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'discount': self._discount,
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            # 'q1_learning_rate': self._q1_optimizer.param_groups[0]['lr'],
            # 'q2_learning_rate': self._q2_optimizer.param_groups[0]['lr'],
            'policy_loss': self._policy_loss_t,
            'q1_loss': self._q1_loss_t,
            'q2_loss': self._q2_loss_t,
        }
