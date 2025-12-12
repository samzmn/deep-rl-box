"""PPO agent class.

Notice in this implementation we follow the following naming convention when referring to unroll sequence:
sₜ, aₜ, rₜ, sₜ₊₁, aₜ₊₁, rₜ₊₁, ...

From the paper "Proximal Policy Optimization Algorithms"
https://arxiv.org/abs/1707.06347.
"""
from typing import Mapping, Iterable, Tuple, Text, Optional, NamedTuple
import multiprocessing
import numpy as np
import tensorflow as tf

from deep_rl_box.utils.schedules import LinearSchedule
from deep_rl_box.utils import utils
from deep_rl_box.utils import base
from deep_rl_box.utils import distributions
from deep_rl_box.utils import multistep
import deep_rl_box.utils.policy_gradient as rl
import deep_rl_box.utils.types as types_lib


class Transition(NamedTuple):
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    logprob_a_t: Optional[float]
    returns_t: Optional[float]
    advantage_t: Optional[float]


class Actor(types_lib.Agent):
    """PPO actor"""

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        policy_network: tf.keras.Model,
        unroll_length: int,
        device: str,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            unroll_length: rollout length.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')

        self.rank = rank
        self.agent_name = f'PPO-actor{rank}'
        self._queue = data_queue
        self._policy_network = policy_network

        # Disable autograd for actor networks.
        for layer in self._policy_network.layers:
            layer.trainable = False

        self._device = device

        self._shared_params = shared_params

        self._unroll_length = unroll_length
        self._unroll_sequence = []

        self._step_t = -1

        self._s_tm1 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given current timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        a_t, logprob_a_t = self.act(timestep)

        if self._a_tm1 is not None:
            self._unroll_sequence.append(
                (
                    self._s_tm1,  # s_t
                    self._a_tm1,  # a_t
                    self._logprob_a_tm1,  # logprob_a_t
                    timestep.reward,  # r_t
                    timestep.observation,  # s_tp1
                    timestep.done,
                )
            )

            if len(self._unroll_sequence) == self._unroll_length:
                self._queue.put(self._unroll_sequence)
                self._unroll_sequence = []

                self._update_actor_network()

        self._s_tm1 = timestep.observation
        self._a_tm1 = a_t
        self._logprob_a_tm1 = logprob_a_t

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode."""
        self._s_tm1 = None
        self._a_tm1 = None
        self._logprob_a_tm1 = None

    def act(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action]:
        'Given timestep, return an action.'
        return self._choose_action(timestep)

    def _update_actor_network(self):
        weights = self._shared_params['policy_network']
        if weights is not None:
            self._policy_network.set_weights(weights)

    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[types_lib.Action]:
        """Given timestep, choose action a_t"""
        s_t = tf.constant(timestep.observation[None, ...], dtype=tf.float32)
        pi_output = self._policy_network(s_t)
        pi_logits_t = pi_output.pi_logits
        # Sample an action
        pi_dist_t = distributions.categorical_distribution(logits=pi_logits_t)

        a_t = pi_dist_t.sample()
        logprob_a_t = pi_dist_t.log_prob(a_t)
        return np.array(a_t).item(), np.array(logprob_a_t).item()

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {}


class Learner(types_lib.Learner):
    """PPO learner"""

    def __init__(
        self,
        policy_network: tf.keras.Model,
        policy_optimizer: tf.keras.optimizers.Optimizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: str,
        shared_params: dict,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            clip_epsilon: external scheduler to decay clip epsilon.
            discount: the gamma discount for future rewards.
            gae_lambda: lambda for the GAE general advantage estimator.
            total_unroll_length: wait until collects this samples before update networks, computed as num_actors x rollout_length.
            update_k: update k times when it's time to do learning.
            batch_size: batch size for learning.
            entropy_coef: the coefficient of entropy loss.
            value_coef: the coefficient of state-value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """

        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length to be greater than 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer greater than or equal to 1, got {update_k}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to [0.0, 1.0], got {entropy_coef}')
        if not 0.0 < value_coef <= 1.0:
            raise ValueError(f'Expect value_coef to (0.0, 1.0], got {value_coef}')

        self.agent_name = 'PPO-learner'
        self._policy_network = policy_network
        self._policy_network.trainable = True
        self._policy_optimizer = policy_optimizer
        self._device = device

        self._shared_params = shared_params

        self._storage = []

        self._total_unroll_length = total_unroll_length

        # For each update epoch, try best to process all samples in 4 batches
        self._batch_size = min(512, int(np.ceil(total_unroll_length / 4).item()))

        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._discount = discount
        self._gae_lambda = gae_lambda

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.
        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if len(self._storage) < self._total_unroll_length:
            return

        return self._learn()

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        """Received item send by actors through multiprocessing queue."""

        # Unpack list of tuples into separate lists.
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))

        returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_t, s_tp1, done_tp1)

        # Zip multiple lists into list of tuples, only keep relevant data
        zipped_sequence = zip(s_t, a_t, logprob_a_t, returns_t, advantage_t)

        self._storage += zipped_sequence

    def get_policy_network_weights(self):
        # To keep things consistent, we move the parameters to CPU
        return self._policy_network.trainable_variables

    def _learn(self) -> Iterable[Mapping[Text, float]]:
        num_samples = len(self._storage)

        # Go over the samples for K epochs
        for _ in range(self._update_k):
            # For each update epoch, split indices into 'bins' with batch_size.
            binned_indices = utils.split_indices_into_bins(self._batch_size, num_samples, shuffle=True)
            # Update on a batch of transitions.
            for indices in binned_indices:
                transitions = [self._storage[i] for i in indices]

                # Stack list of transitions, follow our code convention.
                s_t, a_t, logprob_a_t, returns_t, advantage_t = map(list, zip(*transitions))
                stacked_transitions = Transition(
                    s_t=np.stack(s_t, axis=0),
                    a_t=np.stack(a_t, axis=0),
                    logprob_a_t=np.stack(logprob_a_t, axis=0),
                    returns_t=np.stack(returns_t, axis=0),
                    advantage_t=np.stack(advantage_t, axis=0),
                )
                self._update(stacked_transitions)
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_network_weights()

        del self._storage[:]  # discard old samples after using it

    def _update(self, transitions: Transition) -> None:
        s_t = tf.constant(transitions.s_t, dtype=tf.float32)  # [batch_size, state_shape]
        a_t = tf.constant(transitions.a_t, dtype=tf.int64)  # [batch_size]
        behavior_logprob_a_t = tf.constant(transitions.logprob_a_t, dtype=tf.float32)  # [batch_size]
        returns_t = tf.constant(transitions.returns_t, dtype=tf.float32)  # [batch_size]
        advantage_t = tf.constant(transitions.advantage_t, dtype=tf.float32)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)
        base.assert_rank_and_dtype(a_t, 1, tf.int64)
        base.assert_rank_and_dtype(returns_t, 1, tf.float32)
        base.assert_rank_and_dtype(advantage_t, 1, tf.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, tf.float32)

        with tf.GradientTape() as tape:
            loss = self._calc_loss(s_t, a_t, behavior_logprob_a_t, returns_t, advantage_t)
        gradients = tape.gradient(loss, self._policy_network.trainable_variables)

        if self._clip_grad:
            gradients = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients]

        self._policy_optimizer.apply_gradients(zip(gradients, self._policy_network.trainable_variables))
        self._update_t += 1

    def _calc_loss(self, s_t, a_t, behavior_logprob_a_t, returns_t, advantage_t) -> tf.Tensor:
        """Calculate loss for a batch transitions"""
        # Get policy action logits and value for s_tm1.
        policy_output = self._policy_network(s_t)
        pi_logits_t = policy_output.pi_logits
        v_t = tf.squeeze(policy_output.value, axis=-1) # [batch_size]

        pi_dist_t = distributions.categorical_distribution(logits=pi_logits_t)

        # Compute entropy loss.
        entropy_loss = pi_dist_t.entropy()

        # Compute clipped surrogate policy gradient loss.
        pi_logprob_a_t = pi_dist_t.log_prob(a_t)
        ratio = tf.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantages have same shape, got {ratio.shape} and {advantage_t.shape}')
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss

        # Compute state-value loss.
        value_loss = rl.value_loss(returns_t, v_t).loss

        # Average over batch dimension.
        policy_loss = tf.reduce_mean(policy_loss)
        entropy_loss = tf.reduce_mean(entropy_loss)
        value_loss = tf.reduce_mean(value_loss)

        # Combine policy loss, value loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + self._entropy_coef * entropy_loss) + self._value_coef * value_loss

        # For logging only.
        self._policy_loss_t = np.array(policy_loss).item()
        self._value_loss_t = np.array(value_loss).item()
        self._entropy_loss_t = np.array(entropy_loss).item()

        return loss

    def _compute_returns_and_advantages(
        self,
        s_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        s_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute returns, GAE estimated advantages"""
        stacked_s_t = tf.constant(np.stack(s_t, axis=0), dtype=tf.float32)
        stacked_r_t = tf.constant(np.stack(r_t, axis=0), dtype=tf.float32)
        stacked_s_tp1 = tf.constant(np.stack(s_tp1, axis=0), dtype=tf.float32)
        stacked_done_tp1 = tf.constant(np.stack(done_tp1, axis=0), dtype=tf.bool)

        discount_tp1 = tf.cast(~stacked_done_tp1, dtype=tf.float32) * self._discount

        # Get output from old policy
        output_t = self._policy_network(stacked_s_t)
        v_t = tf.squeeze(output_t.value, axis=-1)

        v_tp1 = tf.squeeze(self._policy_network(stacked_s_tp1).value, axis=-1)
        advantage_t = multistep.truncated_generalized_advantage_estimation(
            stacked_r_t, v_t, v_tp1, discount_tp1, self._gae_lambda
        )

        return_t = advantage_t + v_t

        # Normalize advantages
        advantage_t = (advantage_t - tf.reduce_mean(advantage_t)) / (tf.math.reduce_std(advantage_t) + 1e-8)
        
        advantage_t = np.array(advantage_t)
        return_t = np.array(return_t)

        return (return_t, advantage_t)

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            # 'discount': self._discount,
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }


class GaussianActor(Actor):
    """Gaussian PPO actor for continuous action space"""

    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray]:
        """Given timestep, choose action a_t"""
        s_t = tf.constant(timestep.observation[None, ...], dtype=tf.float32)
        pi_mu, pi_sigma = self._policy_network(s_t)

        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)
        a_t = pi_dist_t.sample()
        logprob_a_t = tf.reduce_sum(pi_dist_t.log_prob(a_t), axis=-1)

        return np.array(tf.squeeze(a_t, axis=0)), np.array(tf.squeeze(logprob_a_t, axis=0))


class GaussianLearner(types_lib.Learner):
    """Learner PPO learner for continuous action space"""

    def __init__(
        self,
        policy_network: tf.keras.Model,
        policy_optimizer: tf.keras.optimizers.Optimizer,
        critic_network: tf.keras.Model,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        clip_epsilon: LinearSchedule,
        discount: float,
        gae_lambda: float,
        total_unroll_length: int,
        update_k: int,
        entropy_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: str,
        shared_params: dict,
    ) -> None:
        """
        Args:
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            critic_network: the value network we want to train.
            critic_optimizer: the optimizer for value network.
            clip_epsilon: external scheduler to decay clip epsilon.
            discount: the gamma discount for future rewards.
            gae_lambda: lambda for the GAE general advantage estimator.
            total_unroll_length: wait until collects this samples before update networks, computed as num_actors x rollout_length.
            update_k: update k times when it's time to do learning.
            entropy_coef: the coefficient of entropy loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """

        if not 1 <= total_unroll_length:
            raise ValueError(f'Expect total_unroll_length to be greater than 1, got {total_unroll_length}')
        if not 1 <= update_k:
            raise ValueError(f'Expect update_k to be integer greater than or equal to 1, got {update_k}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to [0.0, 1.0], got {entropy_coef}')

        self.agent_name = 'PPO-GaussianLearner'
        self._policy_network = policy_network
        self._policy_network.trainable = True
        self._policy_optimizer = policy_optimizer

        self._critic_network = critic_network
        self._critic_optimizer = critic_optimizer
        self._device = device

        self._shared_params = shared_params

        self._storage = []

        self._total_unroll_length = total_unroll_length

        # For each update epoch, try best to process all samples in 4 batches
        self._batch_size = min(512, int(np.ceil(total_unroll_length / 4).item()))

        self._update_k = update_k

        self._entropy_coef = entropy_coef
        self._clip_epsilon = clip_epsilon

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm
        self._discount = discount
        self._gae_lambda = gae_lambda

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
        self._entropy_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.
        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if len(self._storage) < self._total_unroll_length:
            return

        return self._learn()

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""
        self._storage = []

    def received_item_from_queue(self, unroll_sequences: Iterable[Tuple]) -> None:
        """Received item send by actors through multiprocessing queue."""

        # Unpack list of tuples into separate lists.
        s_t, a_t, logprob_a_t, r_t, s_tp1, done_tp1 = map(list, zip(*unroll_sequences))

        returns_t, advantage_t = self._compute_returns_and_advantages(s_t, r_t, s_tp1, done_tp1)

        # Zip multiple lists into list of tuples, only keep relevant data
        zipped_sequence = zip(s_t, a_t, logprob_a_t, returns_t, advantage_t)

        self._storage += zipped_sequence

    def get_policy_network_weights(self):
        # To keep things consistent, we move the parameters to CPU
        return self._policy_network.trainable_variables

    def _learn(self) -> Iterable[Mapping[Text, float]]:
        num_samples = len(self._storage)

        # Go over the samples for K epochs
        for _ in range(self._update_k):
            # For each update epoch, split indices into 'bins' with batch_size.
            binned_indices = utils.split_indices_into_bins(self._batch_size, num_samples, shuffle=True)
            # Update on a batch of transitions.
            for indices in binned_indices:
                transitions = [self._storage[i] for i in indices]
                self._update_policy_network(transitions)
                self._update_value_network(transitions)
                self._update_t += 1
                yield self.statistics

        self._shared_params['policy_network'] = self.get_policy_network_weights()

        del self._storage[:]  # discard old samples after using it

    def _update_policy_network(self, transitions: Iterable[Tuple]) -> None:
        # Unpack list of tuples into separate lists
        s_t, a_t, logprob_a_t, _, advantage_t = map(list, zip(*transitions))

        s_t = tf.constant(tf.stack(s_t, axis=0), dtype=tf.float32)  # [batch_size, state_shape]
        a_t = tf.constant(tf.stack(a_t, axis=0), dtype=tf.float32)  # [batch_size]
        behavior_logprob_a_t = tf.constant(tf.stack(logprob_a_t, axis=0), dtype=tf.float32)  # [batch_size]
        advantage_t = tf.constant(np.stack(advantage_t, axis=0), dtype=tf.float32)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)
        base.assert_rank_and_dtype(a_t, 2, tf.float32)
        base.assert_rank_and_dtype(behavior_logprob_a_t, 1, tf.float32)

        with tf.GradientTape() as tape:
            loss = self._calc_policy_loss(s_t, a_t, behavior_logprob_a_t, advantage_t)
        gradients = tape.gradient(loss, self._policy_network.trainable_variables)

        if self._clip_grad:
            gradients = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients]

        self._policy_optimizer.apply_gradients(zip(gradients, self._policy_network.trainable_variables))

    def _update_value_network(self, transitions: Iterable[Tuple]) -> None:
        # Unpack list of tuples into separate lists
        s_t, _, _, returns_t, _ = map(list, zip(*transitions))

        s_t = tf.constant(tf.stack(s_t, axis=0), dtype=tf.float32)  # [batch_size, state_shape]
        returns_t = tf.constant(tf.stack(returns_t, axis=0), dtype=tf.float32)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)
        base.assert_rank_and_dtype(returns_t, 1, tf.float32)

        with tf.GradientTape() as tape:
            loss = self._calc_value_loss(s_t, returns_t)
        gradients = tape.gradient(loss, self._critic_network.trainable_variables)

        if self._clip_grad:
            gradients = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients]

        self._critic_optimizer.apply_gradients(zip(gradients, self._critic_network.trainable_variables))

    def _calc_policy_loss(self, s_t, a_t, behavior_logprob_a_t, advantage_t) -> tf.Tensor:
        """Calculate loss for a batch transitions"""
        # Get policy action logits and value for s_tm1.
        pi_mu, pi_sigma = self._policy_network(s_t)

        pi_dist_t = distributions.normal_distribution(pi_mu, pi_sigma)

        # Compute entropy loss.
        entropy_loss = pi_dist_t.entropy()

        # Compute clipped surrogate policy gradient loss.
        pi_logprob_a_t = tf.reduce_sum(pi_dist_t.log_prob(a_t), axis=-1)
        ratio = tf.exp(pi_logprob_a_t - behavior_logprob_a_t)

        if ratio.shape != advantage_t.shape:
            raise RuntimeError(f'Expect ratio and advantages have same shape, got {ratio.shape} and {advantage_t.shape}')
        policy_loss = rl.clipped_surrogate_gradient_loss(ratio, advantage_t, self.clip_epsilon).loss

        # Average over batch dimension.
        policy_loss = tf.reduce_mean(policy_loss)
        entropy_loss = tf.reduce_mean(entropy_loss)

        # Combine policy loss, value loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + self._entropy_coef * entropy_loss)

        # For logging only.
        self._policy_loss_t = np.array(policy_loss).item()
        self._entropy_loss_t = np.array(entropy_loss).item()

        return loss

    def _calc_value_loss(self, s_t, returns_t) -> tf.Tensor:
        """Calculate loss for a batch transitions"""
        v_t = tf.squeeze(self._critic_network(s_t), axis=-1)  # [batch_size]

        # Compute state-value loss.
        value_loss = rl.value_loss(returns_t, v_t).loss

        # Average over batch dimension.
        value_loss = tf.reduce_mean(value_loss)

        # For logging only.
        self._value_loss_t = np.array(value_loss).item()

        return value_loss

    def _compute_returns_and_advantages(
        self,
        s_t: Iterable[np.ndarray],
        r_t: Iterable[float],
        s_tp1: Iterable[np.ndarray],
        done_tp1: Iterable[bool],
    ):
        """Compute returns, GAE estimated advantages"""
        stacked_s_t = tf.constant(np.stack(s_t, axis=0), dtype=tf.float32)
        stacked_r_t = tf.constant(np.stack(r_t, axis=0), dtype=tf.float32)
        stacked_s_tp1 = tf.constant(np.stack(s_tp1, axis=0), dtype=tf.float32)
        stacked_done_tp1 = tf.constant(np.stack(done_tp1, axis=0), dtype=tf.bool)

        discount_tp1 = tf.cast(~stacked_done_tp1, dtype=tf.float32) * self._discount

        # Get output from old policy
        v_t = tf.squeeze(self._critic_network(stacked_s_t), axis=-1)
        v_tp1 = tf.squeeze(self._critic_network(stacked_s_tp1), axis=-1)
        advantage_t = multistep.truncated_generalized_advantage_estimation(
            stacked_r_t, v_t, v_tp1, discount_tp1, self._gae_lambda
        )

        return_t = advantage_t + v_t

        # Normalize advantages
        advantage_t = (advantage_t - tf.reduce_mean(advantage_t)) / (tf.reduce_std(advantage_t) + 1e-8)

        advantage_t = advantage_t.numpy()
        return_t = return_t.numpy()

        return (return_t, advantage_t)

    @property
    def clip_epsilon(self):
        """Call external clip epsilon scheduler"""
        return self._clip_epsilon(self._step_t)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            # 'discount': self._discount,
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            'updates': self._update_t,
            'clip_epsilon': self.clip_epsilon,
        }
