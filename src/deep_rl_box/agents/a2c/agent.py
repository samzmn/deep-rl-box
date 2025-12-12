"""A2C agent class.

Specifically:
    * Actors collects transitions and send to master through multiprocessing.Queue.
    * Learner will sample batch of transitions then do the learning step.
    * Learner update policy network parameters for actors.

Note only supports training on single machine.

From the paper "Asynchronous Methods for Deep Reinforcement Learning"
https://arxiv.org/abs/1602.01783

Synchronous, Deterministic variant of A3C
https://openai.com/blog/baselines-acktr-a2c/.

"""

from typing import Iterable, Mapping, Text
import multiprocessing
import numpy as np
import tensorflow as tf

# pylint: disable=import-error
import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
import deep_rl_box.utils.policy_gradient as rl
from deep_rl_box.utils import base
from deep_rl_box.utils import distributions

class Actor(types_lib.Agent):
    """A2C actor"""

    def __init__(
        self,
        rank: int,
        lock: multiprocessing.Lock, # type: ignore
        data_queue: multiprocessing.Queue,
        policy_network: tf.keras.Model,
        transition_accumulator: replay_lib.TransitionAccumulator,
        device: str,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank for the actor.
            lock: multiprocessing.Lock to synchronize with learner process.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            policy_network: the policy network for worker to make action choice.
            transition_accumulator: external helper class to build n-step transition.
            device: Pytf runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """

        self.agent_name = f'A2C-actor{rank}'
        self.rank = rank
        self._lock = lock
        self._queue = data_queue
        self._transition_accumulator = transition_accumulator
        self._policy_network = policy_network

        # Disable autograd for actor networks.
        for layer in self._policy_network.layers:
            layer.trainable = False

        self._device = device

        self._shared_params = shared_params

        # Counters
        self._step_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        # Wait for learner process update to finish before make any move
        with self._lock:
            self._step_t += 1

        a_t = self.act(timestep)

        # Try build transition and add to global queue
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._queue.put(transition)

        return a_t

    def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        'Given timestep, return an action.'
        s_t = tf.constant(timestep.observation[None, ...], dtype=tf.float32)
        a_t = self._choose_action(s_t)
        return np.array(a_t).item()

    def reset(self) -> None:
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()
        self._update_actor_network()

    def _update_actor_network(self) -> None:
        weights = self._shared_params['policy_network']
        if weights is not None:
            self._policy_network.set_weights(weights)

    def _choose_action(self, s_t: tf.Tensor) -> tf.Tensor:
        """Given timestep, choose action a_t"""
        logits_t = self._policy_network(s_t).pi_logits
        # Sample an action
        a_t = distributions.categorical_distribution(logits_t).sample()
        return a_t

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {}


class Learner(types_lib.Learner):
    """A2C learner"""

    def __init__(
        self,
        lock: multiprocessing.Lock, # type: ignore
        policy_network: tf.keras.Model,
        policy_optimizer: tf.keras.optimizers.Optimizer,
        replay: replay_lib.UniformReplay,
        discount: float,
        batch_size: int,
        entropy_coef: float,
        value_coef: float,
        clip_grad: bool,
        max_grad_norm: float,
        device: str,
        shared_params: dict,
    ) -> None:
        """
        Args:
            lock: multiprocessing.Lock to synchronize with worker processes.
            policy_network: the policy network we want to train.
            policy_optimizer: the optimizer for policy network.
            replay: simple experience replay to store transitions.
            discount: the gamma discount for future rewards.
            batch_size: sample batch_size of transitions.
            entropy_coef: the coefficient of entropy loss.
            value_coef: the coefficient of state-value value loss.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: Pytf runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """

        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to be [0.0, 1.0], got {discount}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to be [1, 512], got {batch_size}')
        if not 0.0 <= entropy_coef <= 1.0:
            raise ValueError(f'Expect entropy_coef to be (0.0, 1.0], got {entropy_coef}')
        if not 0.0 <= value_coef <= 1.0:
            raise ValueError(f'Expect value_coef to be (0.0, 1.0], got {value_coef}')

        self.agent_name = 'A2C-learner'
        self._lock = lock

        self._device = device
        self._policy_network = policy_network
        self._policy_network.trainable = True
        self._policy_optimizer = policy_optimizer

        self._shared_params = shared_params

        self._replay = replay
        self._discount = discount
        self._batch_size = batch_size

        self._entropy_coef = entropy_coef
        self._value_coef = value_coef

        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

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

        if self._replay.size < self._batch_size:
            return

        # Blocking while master is updating network parameters
        with self._lock:
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
        transitions = self._replay.sample(self._batch_size)
        self._update(transitions)
        self._replay.reset()  # discard old samples after using it

        self._update_t += 1

        self._shared_params['policy_network'] = self.get_policy_network_weights()

    def _update(self, transitions: replay_lib.Transition) -> None:
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

        discount_t = tf.cast(~done, tf.float32) * self._discount

        with tf.GradientTape() as tape:
            loss = self._calc_loss(s_tm1, a_tm1, r_t, s_t, discount_t)
        grads = tape.gradient(loss, self._policy_network.trainable_variables)

        if self._clip_grad:
            tf.clip_by_global_norm(grads, self._max_grad_norm)
        
        self._policy_optimizer.apply_gradients(zip(grads, self._policy_network.trainable_variables))

    def _calc_loss(self, s_tm1, a_tm1, r_t, s_t, discount_t) -> tf.Tensor:
        """Calculate loss for a batch transitions"""
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
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._policy_optimizer.param_groups[0]['lr'],
            'policy_loss': self._policy_loss_t,
            'value_loss': self._value_loss_t,
            'entropy_loss': self._entropy_loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
        }
