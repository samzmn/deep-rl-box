"""Agent57 agent class.

From the paper "Agent57: Outperforming the Atari Human Benchmark"
https://arxiv.org/pdf/2003.13350.
"""

from typing import Iterable, Mapping, Optional, Tuple, NamedTuple, Text
import multiprocessing
import numpy as np
import tensorflow as tf

import deep_rl_box.utils.replay as replay_lib
import deep_rl_box.utils.types as types_lib
from deep_rl_box.utils import normalizer
from deep_rl_box.utils import transforms
from deep_rl_box.utils import nonlinear_bellman
from deep_rl_box.utils import base
from deep_rl_box.utils import distributed
from deep_rl_box.utils import bandit
from deep_rl_box.utils.curiosity import EpisodicBonusModule, RndLifeLongBonusModule
from deep_rl_box.networks.value import Agent57NetworkInputs


HiddenState = Tuple[tf.Tensor, tf.Tensor]


class Agent57Transition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env.step().

    last_action is the last action the agent took, before in s_t.
    """
    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]  # q values for s_t, computed from both ext_q_network and int_q_network
    prob_a_t: Optional[np.ndarray]  # probability of choose a_t in s_t
    last_action: Optional[int]  # for network input only
    ext_r_t: Optional[float]  # extrinsic reward for (s_tm1, a_tm1)
    int_r_t: Optional[float]  # intrinsic reward for (s_tm1)
    policy_index: Optional[int]  # intrinsic reward scale beta index
    beta: Optional[float]  # intrinsic reward scale beta value
    discount: Optional[float]
    done: Optional[bool]
    ext_init_h: Optional[np.ndarray]  # LSTM initial hidden state, from ext_q_network
    ext_init_c: Optional[np.ndarray]  # LSTM initial cell state, from ext_q_network
    int_init_h: Optional[np.ndarray]  # LSTM initial hidden state, from int_q_network
    int_init_c: Optional[np.ndarray]  # LSTM initial cell state, from int_q_network


TransitionStructure = Agent57Transition(
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
    ext_init_h=None,
    ext_init_c=None,
    int_init_h=None,
    int_init_c=None,
)


def compute_transformed_q(ext_q: tf.Tensor, int_q: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """Returns transformed state-action values from ext_q and int_q."""
    if not isinstance(beta, tf.Tensor):
        beta = tf.identity(beta)
        beta = tf.broadcast_to(tf.convert_to_tensor(beta), shape=tf.shape(int_q))
    # Check if the rank (number of dimensions) of beta is less than that of int_q
    if len(beta.shape) < len(int_q.shape):
        # Expand dimensions of beta to match int_q
        beta = tf.expand_dims(beta, axis=-1)
        beta = tf.broadcast_to(beta, int_q.shape)
        
    return transforms.signed_hyperbolic(transforms.signed_parabolic(ext_q) + beta * transforms.signed_parabolic(int_q))


def no_autograd(net: tf.keras.Model):
    """Disable autograd for a network."""
    for layer in net.layers:
        layer.trainable = False


class Actor(types_lib.Agent):
    """Agent57 actor"""
    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        network: tf.keras.Model,
        rnd_target_network: tf.keras.Model,
        rnd_predictor_network: tf.keras.Model,
        embedding_network: tf.keras.Model,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        ext_discount: float,
        int_discount: float,
        num_actors: int,
        action_dim: int,
        unroll_length: int,
        burn_in: int,
        num_policies: int,
        policy_beta: float,
        ucb_window_size: int,
        ucb_beta: float,
        ucb_epsilon: float,
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
            ucb_window_size: window size of the sliding window UCB algorithm.
            ucb_beta: beta for the sliding window UCB algorithm.
            ucb_epsilon: exploration epsilon for sliding window UCB algorithm.
            episodic_memory_capacity: maximum capacity of episodic memory.
            reset_episodic_memory: Reset the episodic_memory on every episode.
            num_neighbors: number of K-NN neighbors for compute episodic bonus.
            cluster_distance: K-NN neighbors cluster distance for compute episodic bonus.
            kernel_epsilon: K-NN kernel epsilon for compute episodic bonus.
            max_similarity: maximum similarity for compute episodic bonus.
            actor_update_interval: the frequency to update actor's Q network.
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
        if not 1 <= ucb_window_size:
            raise ValueError(f'Expect ucb_window_size to be integer greater than or equal to 1, got {ucb_window_size}')
        if not 0.0 <= ucb_beta <= 100.0:
            raise ValueError(f'Expect ucb_beta to be [0.0, 100.0], got {ucb_beta}')
        if not 0.0 <= ucb_epsilon <= 1.0:
            raise ValueError(f'Expect ucb_epsilon to be [0.0, 1.0], got {ucb_epsilon}')
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
        self.agent_name = f'Agent57-actor{rank}'

        self._network = network
        self._rnd_target_network = rnd_target_network
        self._rnd_predictor_network = rnd_predictor_network
        self._embedding_network = embedding_network

        # Disable autograd for actor's Q networks, embedding, and RND networks.
        no_autograd(self._network)
        no_autograd(self._rnd_target_network)
        no_autograd(self._rnd_predictor_network)
        no_autograd(self._embedding_network)

        self._shared_params = shared_params

        self._queue = data_queue
        self._random_state = random_state
        self._num_actors = num_actors
        self._action_dim = action_dim
        self._actor_update_interval = actor_update_interval
        self._num_policies = num_policies

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,  # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        # Meta-collector
        self._meta_coll = bandit.SimplifiedSlidingWindowUCB(
            self._num_policies, ucb_window_size, self._random_state, ucb_beta, ucb_epsilon
        )

        self._betas, self._gammas = distributed.get_ngu_policy_betas_and_discounts(
            num_policies=num_policies,
            beta=policy_beta,
            gamma_max=ext_discount,
            gamma_min=int_discount,
        )
        self._betas, self._gammas = tf.cast(self._betas, dtype=tf.float32), tf.cast(self._gammas, dtype=tf.float32)
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
            Observation_shape=[network.state_dim],
        )

        self._episode_returns = 0.0
        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._ext_lstm_state = None  # Stores LSTM hidden state and cell state. for extrinsic Q network
        self._int_lstm_state = None  # Stores LSTM hidden state and cell state. for intrinsic Q network

        self._step_t = -1


    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Given timestep, return action a_t, and push transition into global queue"""
        self._step_t += 1
        self._episode_returns += timestep.reward

        if self._step_t % self._actor_update_interval == 0:
            self._update_actor_network(False)

        q_t, a_t, prob_a_t, ext_hidden_s, int_hidden_s = self.act(timestep)

        transition = Agent57Transition(
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
            ext_init_h=self._ext_lstm_state[0].numpy().squeeze(0),  # remove batch dimension
            ext_init_c=self._ext_lstm_state[1].numpy().squeeze(0),
            int_init_h=self._int_lstm_state[0].numpy().squeeze(0), # remove batch dimension
            int_init_c=self._int_lstm_state[1].numpy().squeeze(0),
        )

        unrolled_transition = self._unroll.add(transition, timestep.done)

        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        # Update local state
        self._last_action, self._ext_lstm_state, self._int_lstm_state = a_t, ext_hidden_s, int_hidden_s

        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        return a_t

    def reset(self) -> None:
        """This method should be called at the beginning of every episode before take any action."""
        self._unroll.reset()

        if self._reset_episodic_memory:
            self._episodic_module.reset()

        self._update_actor_network(True)

        # Update Sliding Window UCB statistics.
        self._meta_coll.update(self._policy_index, self._episode_returns)

        self._episode_returns = 0.0

        # Agent57 actor samples a policy using the Sliding Window UCB algorithm, then play a single episode.
        self._sample_policy()

        # During the first step of a new episode,
        # use 'fake' previous action and 'intrinsic' reward for network pass
        self._last_action = self._random_state.randint(0, self._action_dim)  # Initialize a_tm1 randomly
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._ext_lstm_state, self._int_lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    def act(self, timestep: types_lib.TimeStep) -> Tuple[np.ndarray, types_lib.Action, float, HiddenState, HiddenState]:
        'Given state s_t and done marks, return an action.'
        return self._choose_action(timestep)

    def _choose_action(self, timestep):
        """Given state s_t, choose action a_t"""
        input_ = self._prepare_network_input(timestep)

        output = self._network(input_)
        ext_q_t = tf.squeeze(output.ext_q_values)
        int_q_t = tf.squeeze(output.int_q_values)

        q_t = compute_transformed_q(ext_q_t, int_q_t, self._policy_beta)

        a_t = tf.argmax(q_t, axis=-1).numpy()

        # Policy probability for a_t, the detailed equation is mentioned in Agent57 paper.
        prob_a_t = 1 - (self._exploration_epsilon * ((self._action_dim - 1) / self._action_dim))

        # To ensure every actor generates the same amount of samples, apply e-greedy after the network pass.
        if self._random_state.rand() < self._exploration_epsilon:
            a_t = self._random_state.randint(0, self._action_dim)
            prob_a_t = self._exploration_epsilon / self._action_dim

        return (
            q_t.numpy(),
            a_t,
            prob_a_t,
            output.ext_hidden_s,
            output.int_hidden_s,
        )

    def _prepare_network_input(self, timestep: types_lib.TimeStep) -> Agent57NetworkInputs:
        # Agent57 network expect input shape [B, T, state_shape],
        # and additionally 'last action', 'extrinsic reward for last action', last intrinsic reward, and intrinsic reward scale beta index.
        """Prepare network input for Agent57."""
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        last_action = tf.convert_to_tensor(self._last_action, dtype=tf.int64)
        ext_r_t = tf.convert_to_tensor(timestep.reward, dtype=tf.float32)
        int_r_t = tf.convert_to_tensor(self.intrinsic_reward, dtype=tf.float32)
        policy_index = tf.convert_to_tensor(self._policy_index, dtype=tf.int64)
        ext_hidden_s = tuple(tf.identity(s) for s in self._ext_lstm_state)
        int_hidden_s = tuple(tf.identity(s) for s in self._int_lstm_state)
        return Agent57NetworkInputs(
            s_t=tf.expand_dims(s_t, axis=1),  # [B, T, state_shape]
            a_tm1=tf.expand_dims(last_action, axis=0),  # [B, T]
            ext_r_t=tf.expand_dims(ext_r_t, axis=0),  # [B, T]
            int_r_t=tf.expand_dims(int_r_t, axis=0),  # [B, T]
            policy_index=tf.expand_dims(policy_index, axis=0),  # [B, T]
            ext_hidden_s=ext_hidden_s,
            int_hidden_s=int_hidden_s,
        )

    def _put_unroll_onto_queue(self, unrolled_transition):
        # Important note, store hidden states for every step in the unroll will consume HUGE memory.
        self._queue.put(unrolled_transition)

    def _sample_policy(self):
        """Sample new policy from meta collector."""
        self._policy_index = self._meta_coll.sample()
        self._policy_beta = self._betas[self._policy_index]
        self._policy_discount = self._gammas[self._policy_index]

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

        for weights, network in state_net_pairs:
            if weights is not None:
                network.set_weights(weights)

    @property
    def intrinsic_reward(self) -> float:
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the NGU paper.
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
    """Agent57 learner"""

    def __init__(
        self,
        network: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        embedding_network: tf.keras.Model,
        rnd_target_network: tf.keras.Model,
        rnd_predictor_network: tf.keras.Model,
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

        self.agent_name = 'Agent57-learner'
        self._network = network
        self._optimizer = optimizer
        self._embedding_network = embedding_network
        self._rnd_predictor_network = rnd_predictor_network
        self._intrinsic_optimizer = intrinsic_optimizer

        self._rnd_target_network = rnd_target_network
        # create target Q network
        online_network_config = self._network.get_config()
        self._target_network = self._network.__class__.from_config(online_network_config)
        
        # Disable autograd for target Q networks, and RND target networks.
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
        self._rnd_obs_normalizer = normalizer.TensorFlowRunningMeanStd(shape=(network.state_dim))
        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._max_seen_priority = 1.0  # New unroll will use this as priority

        self._retrace_lambda = retrace_lambda
        self._transformed_retrace = transformed_retrace

        self._step_t = -1
        self._update_t = 1
        self._target_update_t = 0
        self._retrace_loss_t = np.nan
        self._embed_loss_t = np.nan

    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise returns None.
        """
        self._step_t += 1

        if self._replay.size < self._min_replay_size or self._step_t % max(4, int(self._batch_size * 0.25)) != 0:
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
        if self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()

    def _update_q_network(self, transitions: Agent57Transition, weights: np.ndarray) -> np.ndarray:
        weights = tf.constant(weights, dtype=tf.float32)  # [B,]
        base.assert_rank_and_dtype(weights, 1, tf.float32)

        # Get initial hidden state for both extrinsic and intrinsic Q networks, handle possible burn in.
        init_ext_hidden_s, init_int_hidden_s = self._extract_first_step_hidden_state(transitions)
        burn_transitions, learn_transitions = replay_lib.split_structure(transitions, TransitionStructure, self._burn_in, axis=1) # over the time axis
        if burn_transitions is not None:
            # Burn in for extrinsic and intrinsic Q networks.
            ext_hidden_s, int_hidden_s, target_ext_hidden_s, target_int_hidden_s = self._burn_in_unroll_q_networks(
                burn_transitions,
                self._network,
                self._target_network,
                init_ext_hidden_s,
                init_int_hidden_s,
            )
        else:
            # Make copy of hidden state for extrinsic Q networks.
            ext_hidden_s = tuple(tf.identity(s) for s in init_ext_hidden_s)
            target_ext_hidden_s = tuple(tf.identity(s) for s in init_ext_hidden_s)

            # Make copy of hidden state for intrinsic Q networks.
            int_hidden_s = tuple(tf.identity(s) for s in init_int_hidden_s)
            target_int_hidden_s = tuple(tf.identity(s) for s in init_int_hidden_s)

        # Update Q network.
        with tf.GradientTape() as tape:
            # Do network pass for all four Q networks to get estimated q values.
            ext_q_t, int_q_t = self._get_predicted_q_values(learn_transitions, self._network, ext_hidden_s, int_hidden_s)

            target_ext_q_t, target_int_q_t = self._get_predicted_q_values(
                learn_transitions, self._target_network, target_ext_hidden_s, target_int_hidden_s
            )

            ext_retrace_loss, ext_priorities = self._calc_retrace_loss(learn_transitions, ext_q_t, target_ext_q_t)
            int_retrace_loss, int_priorities = self._calc_retrace_loss(learn_transitions, int_q_t, target_int_q_t)

            # Multiply loss by sampling weights, averaging over batch dimension
            loss = tf.reduce_mean((ext_retrace_loss + int_retrace_loss) * weights)

        gradients = tape.gradient(loss, self._network.trainable_variables)

        if self._clip_grad:
            gradients = [tf.clip_by_norm(g, self._max_grad_norm) for g in gradients]

        self._optimizer.apply_gradients(zip(gradients, self._network.trainable_variables))

        priorities = 0.8 * ext_priorities + 0.2 * int_priorities

        # For logging only.
        self._retrace_loss_t = loss.numpy().item()

        return priorities

    def _get_predicted_q_values(
        self,
        transitions: Agent57Transition,
        network: tf.keras.Model,
        ext_hidden_state: HiddenState,
        int_hidden_state: HiddenState,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Returns the predicted q values from the network for a given batch of sampled unrolls.

        Args:
            transitions: sampled batch of unrolls, this should not include the burn_in part.
            network: this could be any one of the extrinsic and intrinsic (online or target) networks.
            ext_hidden_state: initial hidden states for the network.
            int_hidden_state: initial hidden states for the network.
        """
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

        # Rank and dtype checks for hidden state.
        base.assert_rank_and_dtype(ext_hidden_state[0], 3, tf.float32)
        base.assert_rank_and_dtype(ext_hidden_state[1], 3, tf.float32)
        base.assert_batch_dimension(ext_hidden_state[0], self._batch_size, 0)
        base.assert_batch_dimension(ext_hidden_state[1], self._batch_size, 0)
        base.assert_rank_and_dtype(int_hidden_state[0], 3, tf.float32)
        base.assert_rank_and_dtype(int_hidden_state[1], 3, tf.float32)
        base.assert_batch_dimension(int_hidden_state[0], self._batch_size, 0)
        base.assert_batch_dimension(int_hidden_state[1], self._batch_size, 0)

        # Get q values from Q network,
        output = network(
            Agent57NetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                ext_hidden_s=ext_hidden_state,
                int_hidden_s=int_hidden_state,
            )
        )

        return (output.ext_q_values, output.int_q_values)

    def _calc_retrace_loss(
        self,
        transitions: Agent57Transition,
        q_t: tf.Tensor,
        target_q_t: tf.Tensor,
    ) -> Tuple[tf.Tensor, np.ndarray]:
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

        return loss, priorities

    def _update_embed_and_rnd_predictor_networks(self, transitions: Agent57Transition, weights: np.ndarray) -> None:
        """Use last 5 frames to update the embedding and RND predictor networks."""
        b = self._batch_size
        weights = tf.constant(weights[-b:], dtype=tf.float32)  # [B]
        base.assert_rank_and_dtype(weights, 1, tf.float32)

        with tf.GradientTape() as tape:
            # Calculate losses
            rnd_loss = self._calc_rnd_loss(transitions)
            embed_loss = self._calc_embed_inverse_loss(transitions)

            # Multiply loss by sampling weights, averaging over batch dimension
            loss = tf.reduce_mean((rnd_loss + embed_loss) * weights)

        # Compute gradients for both models
        combined_grads = tape.gradient(loss, self._rnd_predictor_network.trainable_variables + self._embedding_network.trainable_variables)

        if self._clip_grad:
            combined_grads = [tf.clip_by_norm(g, self._max_grad_norm) for g in combined_grads]

        self._intrinsic_optimizer.apply_gradients(zip(combined_grads, self._rnd_predictor_network.trainable_variables + self._embedding_network.trainable_variables))

        # For logging only.
        self._rnd_loss_t = tf.reduce_mean(rnd_loss).numpy().item()
        self._embed_loss_t = tf.reduce_mean(embed_loss).numpy().item()


    def _calc_rnd_loss(self, transitions: Agent57Transition) -> tf.Tensor:
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

    def _calc_embed_inverse_loss(self, transitions: Agent57Transition) -> tf.Tensor:
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
        rnd_obs = tf.convert_to_tensor(rnd_obs, dtype=tf.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(rnd_obs)
        normed_obs = tf.clip_by_value(normed_obs, -5, 5)

        self._rnd_obs_normalizer.update(rnd_obs)

        return normed_obs

    def _burn_in_unroll_q_networks(
        self,
        transitions: Agent57Transition,
        network: tf.keras.Model,
        target_network: tf.keras.Model,
        ext_hidden_state: HiddenState,
        int_hidden_state: HiddenState,
    ) -> Tuple[HiddenState, HiddenState, HiddenState, HiddenState]:
        """Unroll both online and target Q networks to generate hidden states for LSTM."""
        s_t = tf.convert_to_tensor(transitions.s_t, dtype=tf.float32)  # [B, burn_in, state_shape]
        last_action = tf.convert_to_tensor(transitions.last_action, dtype=tf.int64)  # [B, burn_in]
        ext_r_t = tf.convert_to_tensor(transitions.ext_r_t, dtype=tf.float32)  # [B, burn_in]
        int_r_t = tf.convert_to_tensor(transitions.int_r_t, dtype=tf.float32)  # [B, burn_in]
        policy_index = tf.convert_to_tensor(transitions.policy_index, dtype=tf.int64)  # [B, burn_in]

        # Rank and dtype checks
        tf.debugging.assert_rank(s_t, 3)
        tf.debugging.assert_rank(last_action, 2)
        tf.debugging.assert_rank(ext_r_t, 2)
        tf.debugging.assert_rank(int_r_t, 2)
        tf.debugging.assert_rank(policy_index, 2)

        _ext_hidden_s = tuple(tf.identity(s) for s in ext_hidden_state)
        _int_hidden_s = tuple(tf.identity(s) for s in int_hidden_state)
        _target_ext_hidden_s = tuple(tf.identity(s) for s in ext_hidden_state)
        _target_int_hidden_s = tuple(tf.identity(s) for s in int_hidden_state)

        # Burn in to generate hidden states for LSTM, we unroll both online and target Q networks
        output = network(
            Agent57NetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                ext_hidden_s=_ext_hidden_s,
                int_hidden_s=_int_hidden_s,
            )
        )

        target_output = target_network(
            Agent57NetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                ext_hidden_s=_target_ext_hidden_s,
                int_hidden_s=_target_int_hidden_s,
            )
        )

        return (output.ext_hidden_s, output.int_hidden_s, target_output.ext_hidden_s, target_output.int_hidden_s)

    def _extract_first_step_hidden_state(self, transitions: Agent57Transition) -> Tuple[HiddenState, HiddenState]:
        """Returns ext_hidden_state and int_hidden_state."""
        # We only need the first step hidden states in replay, shape [batch_size, num_lstm_layers, lstm_hidden_size]
        ext_init_h = tf.convert_to_tensor(tf.squeeze(transitions.ext_init_h[:, 0:1], axis=1), dtype=tf.float32)  # [B, lstm_hidden_size]
        ext_init_c = tf.convert_to_tensor(tf.squeeze(transitions.ext_init_c[:, 0:1], axis=1), dtype=tf.float32)  # [B, lstm_hidden_size]
        int_init_h = tf.convert_to_tensor(tf.squeeze(transitions.int_init_h[:, 0:1], axis=1), dtype=tf.float32)  # [B, lstm_hidden_size]
        int_init_c = tf.convert_to_tensor(tf.squeeze(transitions.int_init_c[:, 0:1], axis=1), dtype=tf.float32)  # [B, lstm_hidden_size]

        # Rank and dtype checks.
        base.assert_rank_and_dtype(ext_init_h, 2, tf.float32)
        base.assert_rank_and_dtype(ext_init_c, 2, tf.float32)
        base.assert_rank_and_dtype(int_init_h, 2, tf.float32)
        base.assert_rank_and_dtype(int_init_c, 2, tf.float32)

        # Batch dimension checks.
        base.assert_batch_dimension(ext_init_h, self._batch_size, 0)
        base.assert_batch_dimension(ext_init_c, self._batch_size, 0)
        base.assert_batch_dimension(int_init_h, self._batch_size, 0)
        base.assert_batch_dimension(int_init_c, self._batch_size, 0)

        return ((ext_init_h, ext_init_c), (int_init_h, int_init_c))

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
            'embed_loss': self._embed_loss_t,
            'rnd_loss': self._rnd_loss_t,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
        }
