"""NGU agent class.

From the paper "Never Give Up: Learning Directed Exploration Strategies"
https://arxiv.org/abs/2002.06038.
"""

from typing import Iterable, Mapping, Optional, Tuple, NamedTuple, Text, Any
import queue
import numpy as np
import tensorflow as tf

# pylint: disable=import-error
import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
from deep_rl_box.utils import normalizer
from deep_rl_box.utils import nonlinear_bellman
from deep_rl_box.utils import base
from deep_rl_box.utils import distributed
from deep_rl_box.utils.curiosity import EpisodicBonusModule, RndLifeLongBonusModule
from deep_rl_box.networks.value import NguDqnConvNet, NguNetworkInputs
from deep_rl_box.networks.curiosity import NGURndConvNet, NguEmbeddingConvNet

HiddenState = Tuple[tf.Tensor, tf.Tensor]


class NguTransition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env.step().

    last_action is the last action the agent took, before in s_t.
    """

    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]  # q values for s_t
    prob_a_t: Optional[np.ndarray]  # probability of choose a_t in s_t
    last_action: Optional[int]  # for network input only
    ext_r_t: Optional[float]  # extrinsic reward for (s_tm1, a_tm1)
    int_r_t: Optional[float]  # intrinsic reward for (s_tm1)
    policy_index: Optional[int]  # intrinsic reward scale beta index
    beta: Optional[float]  # intrinsic reward scale beta value
    discount: Optional[float]
    done: Optional[bool]
    init_h: Optional[np.ndarray]  # LSTM initial hidden state
    init_c: Optional[np.ndarray]  # LSTM initial cell state


TransitionStructure = NguTransition(
    s_t=None,
    a_t=None,
    q_t=None,
    prob_a_t=None,
    last_action=None,
    ext_r_t=None,
    int_r_t=None,
    policy_index=None,
    beta=None,
    discount=None,
    done=None,
    init_h=None,
    init_c=None,
)


def no_autograd(net: tf.keras.Model):
    """Disable autograd for a network."""
    for layer in net.layers:
        layer.trainable = False


class Actor(types_lib.Agent):
    """NGU actor"""

    def __init__(
        self,
        rank: int,
        data_queue: queue.Queue,
        network: NguDqnConvNet,
        rnd_target_network: NGURndConvNet,
        rnd_predictor_network: NGURndConvNet,
        embedding_network: NguEmbeddingConvNet,
        random_state: np.random.RandomState,
        ext_discount: float,
        int_discount: float,
        num_actors: int,
        action_dim: int,
        state_dim: Any,
        unroll_length: int,
        burn_in: int,
        num_policies: int,
        policy_beta: float,
        episodic_memory_capacity: int,
        reset_episodic_memory: bool,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        actor_update_interval: int,
        shared_params: dict,
    ) -> None:
        """
        Args:
            rank: the rank number for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to learner process.
            network: the Q network for actor to make action choice.
            rnd_target_network: RND random target network.
            rnd_predictor_network: RND predictor target network.
            embedding_network: NGU action prediction network.
            random_state: random state.
            ext_discount: extrinsic reward discount.
            int_discount: intrinsic reward discount.
            num_actors: number of actors.
            action_dim: number of valid actions in the environment.
            unroll_length: how many agent time step to unroll transitions before put on to queue.
            burn_in: two consecutive unrolls will overlap on burn_in+1 steps.
            num_policies: number of exploring and exploiting policies.
            policy_beta: intrinsic reward scale beta.
            episodic_memory_capacity: maximum capacity of episodic memory.
            reset_episodic_memory: Reset the episodic_memory on every episode.
            num_neighbors: number of K-NN neighbors for compute episodic bonus.
            cluster_distance: K-NN neighbors cluster distance for compute episodic bonus.
            kernel_epsilon: K-NN kernel epsilon for compute episodic bonus.
            max_similarity: maximum similarity for compute episodic bonus.
            actor_update_interval: the frequency to update actor's Q network.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 0.0 <= ext_discount <= 1.0:
            raise ValueError(f'Expect ext_discount to be [0.0, 1.0], got {ext_discount}')
        if not 0.0 <= int_discount <= 1.0:
            raise ValueError(f'Expect int_discount to be [0.0, 1.0], got {int_discount}')
        if not 0 < num_actors:
            raise ValueError(f'Expect num_actors to be positive integer, got {num_actors}')
        if not 0 < action_dim:
            raise ValueError(f'Expect action_dim to be positive integer, got {action_dim}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect burn_in length to be [0, {unroll_length}), got {burn_in}')
        if not 1 <= num_policies:
            raise ValueError(f'Expect num_policies to be integer greater than or equal to 1, got {num_policies}')
        if not 0.0 <= policy_beta <= 1.0:
            raise ValueError(f'Expect policy_beta to be [0.0, 1.0], got {policy_beta}')
        if not 1 <= episodic_memory_capacity:
            raise ValueError(
                f'Expect episodic_memory_capacity to be integer greater than or equal to 1, got {episodic_memory_capacity}'
            )
        if not 1 <= num_neighbors:
            raise ValueError(f'Expect num_neighbors to be integer greater than or equal to 1, got {num_neighbors}')
        if not 0.0 <= cluster_distance <= 1.0:
            raise ValueError(f'Expect cluster_distance to be [0.0, 1.0], got {cluster_distance}')
        if not 0.0 <= kernel_epsilon <= 1.0:
            raise ValueError(f'Expect kernel_epsilon to be [0.0, 1.0], got {kernel_epsilon}')
        if not 1 <= actor_update_interval:
            raise ValueError(
                f'Expect actor_update_interval to be integer greater than or equal to 1, got {actor_update_interval}'
            )

        self.rank = rank  # Needs to make sure rank always start from 0
        self.agent_name = f'NGU-actor{rank}'

        self._network = network
        self._rnd_target_network = rnd_target_network
        self._rnd_predictor_network = rnd_predictor_network
        self._embedding_network = embedding_network

        # Disable autograd for actor's local networks
        no_autograd(self._network)
        no_autograd(self._rnd_target_network)
        no_autograd(self._rnd_predictor_network)
        no_autograd(self._embedding_network)

        self._shared_params = shared_params

        self._queue = data_queue

        self._random_state = random_state
        self._num_actors = num_actors
        self._action_dim = action_dim
        self._actor_update_q_network_interval = actor_update_interval
        self._num_policies = num_policies

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,  # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        self._betas, self._gammas = distributed.get_ngu_policy_betas_and_discounts(
            num_policies=num_policies,
            beta=policy_beta,
            gamma_max=ext_discount,
            gamma_min=int_discount,
        )

        self._policy_index = None
        self._policy_beta = None
        self._policy_discount = None
        self._sample_policy()

        self._reset_episodic_memory = reset_episodic_memory

        # E-greedy policy epsilon, rank 0 has the lowest noise, while rank N-1 has the highest noise.
        epsilons = distributed.get_actor_exploration_epsilon(num_actors)
        self._exploration_epsilon = epsilons[self.rank]

        # Episodic intrinsic bonus module
        self._episodic_module = EpisodicBonusModule(
            embedding_network=self._embedding_network,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        # Lifelong intrinsic bonus module
        self._lifelong_module = RndLifeLongBonusModule(
            target_network=self._rnd_target_network,
            predictor_network=self._rnd_predictor_network,
            discount=int_discount,
            Observation_shape=[state_dim],
        )

        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._lstm_state = None  # Stores LSTM hidden state and cell state

        self._step_t = -1

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1

        if self._step_t % self._actor_update_q_network_interval == 0:
            self._update_actor_network(False)

        q_t, a_t, prob_a_t, hidden_s = self.act(timestep)

        transition = NguTransition(
            s_t=timestep.observation,
            a_t=a_t,
            q_t=q_t,
            prob_a_t=prob_a_t,
            last_action=self._last_action,
            ext_r_t=timestep.reward,
            int_r_t=self.intrinsic_reward,
            policy_index=self._policy_index,
            beta=self._policy_beta,
            discount=self._policy_discount,
            done=timestep.done,
            init_h=tf.squeeze(self._lstm_state[0],axis=0),  # remove batch dimension
            init_c=tf.squeeze(self._lstm_state[1], axis=0),
        )

        unrolled_transition = self._unroll.add(transition, timestep.done)

        s_t = tf.constant(timestep.observation[None, ...], dtype=tf.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        # Update local state
        self._last_action, self._lstm_state = a_t, hidden_s

        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._unroll.reset()

        # From NGU Paper on MONTEZUMA’S REVENGE:
        """
        Instead of resetting the memory after every episode, we do it after a small number of
        consecutive episodes, which we call a meta-episode. This structure plays an important role when the
        agent faces irreversible choices.
        """

        if self._reset_episodic_memory:
            self._episodic_module.reset()

        self._update_actor_network(True)

        self._sample_policy()

        # During the first step of a new episode,
        # use 'fake' previous action and 'intrinsic' reward for network pass
        self._last_action = self._random_state.randint(0, self._action_dim)  # Initialize a_tm1 randomly
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, float, HiddenState]:
        'Given state s_t and done marks, return an action.'
        return self._choose_action(timestep)

    def _choose_action(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, float, HiddenState]:
        """Given state s_t, choose action a_t"""
        pi_output = self._network(self._prepare_network_input(timestep))
        q_t = tf.squeeze(pi_output.q_values)

        a_t = tf.argmax(q_t, axis=-1)

        # Policy probability for a_t, the detailed equation is mentioned in Agent57 paper.
        prob_a_t = 1 - (self._exploration_epsilon * ((self._action_dim - 1) / self._action_dim))

        # To make sure every actors generates the same amount of samples, we apply e-greedy after the network pass,
        # otherwise the actor with higher epsilons will generate more samples,
        # while the actor with lower epsilon will generate less samples.
        if self._random_state.rand() < self._exploration_epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._action_dim)
            prob_a_t = self._exploration_epsilon / self._action_dim

        return (np.array(q_t), np.array(a_t).item(), prob_a_t, pi_output.hidden_s)

    def _prepare_network_input(self, timestep: types_lib.TimeStep) -> NguNetworkInputs:
        # NGU network expect input shape [B, T, state_shape],
        # and additionally 'last action', 'extrinsic reward for last action', last intrinsic reward, and intrinsic reward scale beta index.
        s_t = tf.constant(timestep.observation[None, ...], dtype=tf.float32)
        last_action = tf.constant(self._last_action, dtype=tf.int64)
        ext_r_t = tf.constant(timestep.reward, dtype=tf.float32)
        int_r_t = tf.constant(self.intrinsic_reward, dtype=tf.float32)
        policy_index = tf.constant(self._policy_index, dtype=tf.int64)
        hidden_s = tuple(s for s in self._lstm_state)

        assert not tf.reduce_any(tf.math.is_nan(int_r_t))

        return NguNetworkInputs(
            s_t = np.expand_dims(s_t, axis=1),  # [B, T, state_shape]
            a_tm1 = tf.expand_dims(last_action, axis=0),  # [B, T]
            ext_r_t = tf.expand_dims(ext_r_t, axis=0),  # [B, T]
            int_r_t = tf.expand_dims(int_r_t, axis=0),  # [B, T]
            policy_index = tf.expand_dims(policy_index, axis=0),  # [B, T]
            hidden_s=hidden_s,
        )

    def _put_unroll_onto_queue(self, unrolled_transition):
        # Important note, store hidden states for every step in the unroll will consume HUGE memory.
        self._queue.put(unrolled_transition)

    def _update_actor_network(self, update_embed: bool = False):
        q_kernel = self._shared_params['network']
        embed_kernel = self._shared_params['embedding_network']
        rnd_kernel = self._shared_params['rnd_predictor_network']

        if update_embed:
            state_net_pairs = zip(
                (q_kernel, embed_kernel, rnd_kernel),
                (self._network, self._embedding_network, self._rnd_predictor_network),
            )
        else:
            state_net_pairs = zip(
                (q_kernel, rnd_kernel),
                (self._network, self._rnd_predictor_network),
            )

        for kernel, network in state_net_pairs:
            if kernel is not None:
                network.set_weights(kernel)

    def _sample_policy(self):
        self._policy_index = np.random.randint(0, self._num_policies)
        self._policy_beta = self._betas[self._policy_index]
        self._policy_discount = self._gammas[self._policy_index]

    @property
    def intrinsic_reward(self) -> float:
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the paper.
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current actor's statistics as a dictionary."""
        return {
            # 'policy_index': self._policy_index,
            'policy_discount': self._policy_discount,
            'policy_beta': self._policy_beta,
            'exploration_epsilon': self._exploration_epsilon,
            'intrinsic_reward': self.intrinsic_reward,
            # 'episodic_bonus': self._episodic_bonus_t,
            # 'lifelong_bonus': self._lifelong_bonus_t,
        }


class Learner(types_lib.Learner):
    """NGU learner"""

    def __init__(
        self,
        network: NguDqnConvNet,
        optimizer: tf.keras.optimizers.Optimizer,
        embedding_network: NguEmbeddingConvNet,
        rnd_target_network: NGURndConvNet,
        rnd_predictor_network: NGURndConvNet,
        intrinsic_optimizer: tf.keras.optimizers.Optimizer,
        replay: replay_lib.PrioritizedReplay,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        unroll_length: int,
        burn_in: int,
        retrace_lambda: float,
        transformed_retrace: bool,
        priority_eta: float,
        clip_grad: bool,
        max_grad_norm: float,
        shared_params: dict,
    ) -> None:
        """
        Args:
            network: the Q network we want to train and optimize.
            optimizer: the optimizer for Q network.
            embedding_network: NGU action prediction network.
            rnd_target_network: RND random network.
            rnd_predictor_network: RND predictor network.
            intrinsic_optimizer: the optimizer for action prediction and RND predictor networks.
            replay: prioritized recurrent experience replay.
            target_net_update_interval: how often to copy online network parameters to target.
            min_replay_size: wait till experience replay buffer this number before start to learn.
            batch_size: sample batch_size of transitions.
            burn_in: burn n transitions to generate initial hidden state before learning.
            unroll_length: transition sequence length.
            retrace_lambda: coefficient of the retrace lambda.
            transformed_retrace: if True, use transformed retrace.
            priority_eta: coefficient to mix the max and mean absolute TD errors.
            clip_grad: if True, clip gradients norm.
            max_grad_norm: the maximum gradient norm for clip grad, only works if clip_grad is True.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for actors.
        """
        if not 1 <= target_net_update_interval:
            raise ValueError(f'Expect target_net_update_interval to be positive integer, got {target_net_update_interval}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer greater than or equal to 1, got {min_replay_size}')
        if not 1 <= batch_size <= 128:
            raise ValueError(f'Expect batch_size to in the range [1, 128], got {batch_size}')
        if not 1 <= unroll_length:
            raise ValueError(f'Expect unroll_length to be greater than or equal to 1, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect burn_in length to be [0, {unroll_length}), got {burn_in}')
        if not 0.0 <= retrace_lambda <= 1.0:
            raise ValueError(f'Expect retrace_lambda to in the range [0.0, 1.0], got {retrace_lambda}')
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError(f'Expect priority_eta to in the range [0.0, 1.0], got {priority_eta}')

        self.agent_name = 'NGU-learner'
        self._network = network
        self._network.trainable = True
        self._optimizer = optimizer
        self._embedding_network = embedding_network
        self._embedding_network.trainable = True
        self._rnd_predictor_network = rnd_predictor_network
        self._rnd_predictor_network.trainable = True
        self._intrinsic_optimizer = intrinsic_optimizer

        self._rnd_target_network = rnd_target_network
        # Lazy way to create target Q network
        self._target_network = self._network.__class__.from_config(self._network.get_config())

        # Disable autograd for target Q network and RND target network
        no_autograd(self._target_network)
        no_autograd(self._rnd_target_network)

        self._shared_params = shared_params

        self._batch_size = batch_size
        self._burn_in = burn_in
        self._unroll_length = unroll_length
        self._total_unroll_length = unroll_length + 1
        self._target_net_update_interval = target_net_update_interval
        self._clip_grad = clip_grad
        self._max_grad_norm = max_grad_norm

        # Accumulate running statistics to calculate mean and std
        self._rnd_obs_normalizer = normalizer.TensorFlowRunningMeanStd(shape=(network.state_dim)) # Observation shape

        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._max_seen_priority = 1.0  # New unroll will use this as priority

        self._retrace_lambda = retrace_lambda
        self._transformed_retrace = transformed_retrace

        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._retrace_loss_t = np.nan
        self._rnd_loss_t = np.nan
        self._embed_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if self._replay.size < self._min_replay_size or self._step_t % max(4, int(self._batch_size * 0.25)) != 0:
            if self._replay.size % 50 == 0:
                print("replay size: ", self._replay.size)
            return

        self._learn()
        
        yield self.statistics

    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""

    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._replay.add(item, self._max_seen_priority)

    def get_network_weights(self):
        return self._network.get_weights()

    def get_embedding_network_weights(self):
        return self._embedding_network.get_weights()

    def get_rnd_predictor_network_weights(self):
        return self._rnd_predictor_network.get_weights()

    def _learn(self) -> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update_q_network(transitions, weights)
        self._update_embed_and_rnd_predictor_networks(transitions, weights)
        self._update_t += 1

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape ({self._batch_size}), got {priorities.shape}')
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority, np.max(priorities)])
        self._replay.update_priorities(indices, priorities)

        self._shared_params['network'] = self.get_network_weights()
        self._shared_params['embedding_network'] = self.get_embedding_network_weights()
        self._shared_params['rnd_predictor_network'] = self.get_rnd_predictor_network_weights()

        # Copy Q network parameters to target Q network, every m updates
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()

    def _update_q_network(self, transitions: NguTransition, weights: np.ndarray) -> np.ndarray:
        """Update online Q network."""
        weights = tf.constant(weights, dtype=tf.float32)  # [batch_size]
        base.assert_rank_and_dtype(weights, 1, tf.float32)

        # Get initial hidden state, handle possible burn in.
        init_hidden_s = self._extract_first_step_hidden_state(transitions)
        burn_transitions, learn_transitions = replay_lib.split_structure(transitions, TransitionStructure, self._burn_in, axis=1) # over the time axis

        if burn_transitions is not None:
            hidden_s, target_hidden_s = self._burn_in_unroll_q_networks(burn_transitions, init_hidden_s)
        else:
            hidden_s = tuple(tf.identity(s) for s in init_hidden_s)
            target_hidden_s = tuple(tf.identity(s) for s in init_hidden_s)

        with tf.GradientTape() as tape:
            # Compute predicted q values using online and target Q networks.
            q_t = self._get_predicted_q_values(learn_transitions, self._network, hidden_s)
            target_q_t = self._get_predicted_q_values(learn_transitions, self._target_network, target_hidden_s)

            # [batch_size]
            retrace_loss, priorities = self._calc_retrace_loss(learn_transitions, q_t, target_q_t)
            # Multiply loss by sampling weights, averaging over batch dimension
            loss = tf.reduce_mean(retrace_loss * weights)

        gradients = tape.gradient(loss, self._network.trainable_variables)

        if self._clip_grad:
            gradients = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients]

        self._optimizer.apply_gradients(zip(gradients, self._network.trainable_variables))
        
        # For logging only.
        self._retrace_loss_t = loss.numpy().item()

        return priorities

    def _prepare_network_input(self, transitions: NguTransition, hidden_state: HiddenState) -> NguNetworkInputs:
        s_t = tf.constant(transitions.s_t, dtype=tf.float32)  # [B, T+1, state_shape]
        last_action = tf.constant(transitions.last_action, dtype=tf.int64)  # [B, T+1]
        ext_r_t = tf.constant(transitions.ext_r_t, dtype=tf.float32)  # [B, T+1]
        int_r_t = tf.constant(transitions.int_r_t, dtype=tf.float32)  # [B, T+1]
        policy_index = tf.constant(transitions.policy_index, dtype=tf.int64)  # [B, T+1]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(s_t, (3, 5), tf.float32)
        base.assert_rank_and_dtype(last_action, 2, tf.int64)
        base.assert_rank_and_dtype(ext_r_t, 2, tf.float32)
        base.assert_rank_and_dtype(int_r_t, 2, tf.float32)
        base.assert_rank_and_dtype(policy_index, 2, tf.int64)
        
        assert not tf.reduce_any(tf.math.is_nan(s_t))
        assert not tf.reduce_any(tf.math.is_nan(tf.cast(last_action, dtype=tf.float32)))
        assert not tf.reduce_any(tf.math.is_nan(ext_r_t))
        assert not tf.reduce_any(tf.math.is_nan(int_r_t))
        assert not tf.reduce_any(tf.math.is_nan(tf.cast(policy_index, dtype=tf.float32)))
        
        return NguNetworkInputs(
            s_t=s_t,
            a_tm1=last_action,
            ext_r_t=ext_r_t,
            int_r_t=int_r_t,
            policy_index=policy_index,
            hidden_s=hidden_state,
        )

    def _get_predicted_q_values(
        self, transitions: NguTransition, q_network: tf.keras.Model, hidden_state: HiddenState
    ) -> tf.Tensor:
        """Returns the predicted q values from the 'q_network' for a given batch of sampled unrolls.

        Args:
            transitions: sampled batch of unrolls, this should not include the burn_in part.
            q_network: this could be one of the online or target Q networks.
            hidden_state: initial hidden states for the 'q_network'.
        """

        # Get q values from Q network
        q_t = q_network(self._prepare_network_input(transitions, hidden_state)).q_values

        assert not tf.reduce_any(tf.math.is_nan(q_t))

        return q_t

    def _calc_retrace_loss(
        self,
        transitions: NguTransition,
        q_t: tf.Tensor,
        target_q_t: tf.Tensor,
    ) -> Tuple[tf.Tensor, np.ndarray]:
        """Calculate loss and priorities for given unroll sequence transitions."""
        a_t = tf.constant(transitions.a_t, dtype=tf.int64)  # [B, T+1]
        behavior_prob_a_t = tf.constant(transitions.prob_a_t, dtype=tf.float32)  # [B, T+1]
        ext_r_t = tf.constant(transitions.ext_r_t, dtype=tf.float32)  # [B, T+1]
        int_r_t = tf.constant(transitions.int_r_t, dtype=tf.float32)  # [B, T+1]
        beta = tf.constant(transitions.beta, dtype=tf.float32)  # [B, T+1]
        discount = tf.constant(transitions.discount, dtype=tf.float32)  # [B, T+1]
        done = tf.constant(transitions.done, dtype=tf.bool)  # [B, T+1]

        # Rank and dtype checks, note we have a new unroll time dimension, states may be images, which is rank 5.
        base.assert_rank_and_dtype(behavior_prob_a_t, 2, tf.float32)
        base.assert_rank_and_dtype(a_t, 2, tf.int64)
        base.assert_rank_and_dtype(ext_r_t, 2, tf.float32)
        base.assert_rank_and_dtype(int_r_t, 2, tf.float32)
        base.assert_rank_and_dtype(beta, 2, tf.float32)
        base.assert_rank_and_dtype(discount, 2, tf.float32)
        base.assert_rank_and_dtype(done, 2, tf.bool)

        r_t = ext_r_t + beta * int_r_t  # Augmented rewards
        discount_t = tf.cast(~done, dtype=tf.float32) * discount  # (B, T+1)

        # Derive target policy probabilities from q values.
        target_policy_probs = tf.nn.softmax(target_q_t, axis=-1)  # [B, T+1, action_dim]

        if self._transformed_retrace:
            transform_tx_pair = nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR
        else:
            transform_tx_pair = nonlinear_bellman.IDENTITY_PAIR  # No transform

        # Compute retrace loss.
        retrace_out = nonlinear_bellman.transformed_retrace(
            q_tm1=q_t[:, :-1],
            q_t=target_q_t[:, 1:],
            a_tm1=a_t[:, :-1],
            a_t=a_t[:, 1:],
            r_t=r_t[:, 1:],
            discount_t=discount_t[:, 1:],
            pi_t=target_policy_probs[:, 1:],
            mu_t=behavior_prob_a_t[:, 1:],
            lambda_=self._retrace_lambda,
            tx_pair=transform_tx_pair,
        )

        # Compute priority.
        priorities = distributed.calculate_dist_priorities_from_td_error(retrace_out.extra.td_error, self._priority_eta)
        # Sums over time dimension.
        loss = tf.reduce_sum(retrace_out.loss, axis=1)
        return (loss, priorities)

    def _update_embed_and_rnd_predictor_networks(self, transitions: NguTransition, weights: np.ndarray) -> None:
        """Update the embedding action prediction and RND predictor networks."""
        b = self._batch_size
        weights = tf.constant(weights[-b:], dtype=tf.float32)  # [B]
        base.assert_rank_and_dtype(weights, 1, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # [batch_size]
            rnd_loss = self._calc_rnd_loss(transitions)
            embed_loss = self._calc_embed_inverse_loss(transitions)

            # Multiply loss by sampling weights, averaging over batch dimension
            loss = tf.reduce_mean((rnd_loss + embed_loss) * weights)

        rnd_gradients = tape.gradient(loss, self._rnd_predictor_network.trainable_variables)
        embed_gradients = tape.gradient(loss, self._embedding_network.trainable_variables)
        del tape
        if self._clip_grad:
            rnd_gradients = [tf.clip_by_norm(g, self._max_grad_norm) for g in rnd_gradients]
            embed_gradients = [tf.clip_by_norm(g, self._max_grad_norm) for g in embed_gradients]

        self._intrinsic_optimizer.apply_gradients(zip(rnd_gradients, self._rnd_predictor_network.trainable_variables))
        self._intrinsic_optimizer.apply_gradients(zip(embed_gradients, self._embedding_network.trainable_variables))

        # For logging only.
        self._rnd_loss_t = np.array(rnd_loss).mean().item()
        self._embed_loss_t = np.array(embed_loss).mean().item()

    def _calc_rnd_loss(self, transitions: NguTransition) -> tf.Tensor:
        s_t = tf.constant(transitions.s_t[:, -5:], dtype=tf.float32)  # [B, 5, state_shape]
        # Rank and dtype checks.
        base.assert_rank_and_dtype(s_t, (3, 5), tf.float32)
        # Merge batch and time dimension.
        # Flatten the tensor from dimension 0 to 1 
        s_t = tf.reshape(s_t, shape=[-1, *s_t.shape[2:]])

        normed_s_t = self._normalize_rnd_obs(s_t)

        pred_s_t = self._rnd_predictor_network(normed_s_t)
        target_s_t = self._rnd_target_network(normed_s_t)

        rnd_loss = tf.reduce_mean(tf.square(pred_s_t - target_s_t), axis=1)
        # Reshape loss into [B, 5].
        rnd_loss = tf.reshape(rnd_loss, (-1, 5))

        # Sums over time dimension. shape [B]
        loss = tf.reduce_sum(rnd_loss, axis=1)

        return loss

    def _calc_embed_inverse_loss(self, transitions: NguTransition) -> tf.Tensor:
        s_t = tf.constant(transitions.s_t[:, -6:], dtype=tf.float32)  # [B, 6, state_shape]
        a_t = tf.constant(transitions.a_t[:, -6:], dtype=tf.int64)  # [B, 6]

        # Rank and dtype checks.
        base.assert_rank_and_dtype(s_t, (3, 5), tf.float32)
        base.assert_rank_and_dtype(a_t, 2, tf.int64)

        s_tm1 = s_t[:, 0:-1, ...]  # [B, 5, state_shape]
        s_t = s_t[:, 1:, ...]  # [B, 5, state_shape]
        a_tm1 = a_t[:, :-1]  # [B, 5]

        # Merge batch and time dimension.
        s_tm1 = tf.reshape(s_tm1, shape=[-1, *s_tm1.shape[2:]])
        s_t = tf.reshape(s_t, shape=[-1, *s_t.shape[2:]])
        a_tm1 = tf.reshape(a_tm1, shape=[-1, *a_tm1.shape[2:]])

        # Compute action prediction loss.
        embedding_s_tm1 = self._embedding_network(s_tm1)  # [B*5, latent_dim]
        embedding_s_t = self._embedding_network(s_t)  # [B*5, latent_dim]
        embeddings = tf.concat([embedding_s_tm1, embedding_s_t], axis=-1)
        pi_logits = self._embedding_network.inverse_prediction(embeddings)  # [B*5, action_dim]
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=a_tm1, y_pred=pi_logits, from_logits=True) # [B*5,]
        # Reshape loss into [B, 5].
        loss = tf.reshape(loss, shape=(-1, 5))

        # Sums over time dimension. shape [B]
        loss = tf.reduce_sum(loss, axis=1)
        return loss

    def _normalize_rnd_obs(self, rnd_obs):
        rnd_obs = tf.cast(rnd_obs, dtype=tf.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(rnd_obs)
        # Clamp the values to the range [-5, 5]
        normed_obs = tf.clip_by_value(normed_obs, clip_value_min=-5, clip_value_max=5)

        self._rnd_obs_normalizer.update(rnd_obs)

        return normed_obs

    def _burn_in_unroll_q_networks(
        self,
        transitions: NguTransition,
        init_hidden_s: HiddenState,
    ) -> Tuple[HiddenState, HiddenState]:
        """Unroll both online and target q networks to generate hidden states for LSTM."""

        _hidden_s = tuple(tf.identity(s) for s in init_hidden_s)
        _target_hidden_s = tuple(tf.identity(s) for s in init_hidden_s)

        # Burn in to generate hidden states for LSTM, we unroll both online and target Q networks
        hidden_s = self._network(self._prepare_network_input(transitions, _hidden_s)).hidden_s
        target_hidden_s = self._target_network(self._prepare_network_input(transitions, _target_hidden_s)).hidden_s

        return (hidden_s, target_hidden_s)

    def _extract_first_step_hidden_state(self, transitions: NguTransition) -> HiddenState:
        # We only need the first step hidden states in replay, shape [batch_size, lstm_hidden_size]
        init_h = tf.constant(tf.squeeze(transitions.init_h[:, 0:1], axis=1), dtype=tf.float32)
        init_c = tf.constant(tf.squeeze(transitions.init_c[:, 0:1], axis=1), dtype=tf.float32)

        # Rank and dtype checks.
        base.assert_rank_and_dtype(init_h, 2, tf.float32)
        base.assert_rank_and_dtype(init_c, 2, tf.float32)

        # Batch dimension checks.
        base.assert_batch_dimension(init_h, self._batch_size, 0)
        base.assert_batch_dimension(init_c, self._batch_size, 0)

        return (init_h, init_c)

    def _update_target_network(self):
        self._target_network.set_weights(self._network.get_weights())
        self._target_update_t += 1

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            # 'ext_lr': self._optimizer.param_groups[0]['lr'],
            # 'int_lr': self._intrinsic_optimizer.param_groups[0]['lr'],
            'retrace_loss': self._retrace_loss_t,
            'rnd_loss': self._rnd_loss_t,
            'embed_loss': self._embed_loss_t,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
        }
