"""Greedy actors for testing and evaluation."""
from typing import Mapping, Tuple, Text
import numpy as np
import tensorflow as tf

# pylint: disable=import-error
import deep_rl_box.utils.types as types_lib
from deep_rl_box.networks.policy import ImpalaActorCriticNetworkInputs
from deep_rl_box.networks.value import RnnDqnNetworkInputs, NguNetworkInputs, Agent57NetworkInputs
from deep_rl_box.utils.curiosity import EpisodicBonusModule, RndLifeLongBonusModule
from deep_rl_box.agents.agent57.agent import compute_transformed_q


HiddenState = Tuple[tf.Tensor, tf.Tensor]


def apply_egreedy_policy(
    q_values: np.ndarray,
    epsilon: float,
    random_state: np.random.RandomState,  # pylint: disable=no-member
) -> types_lib.Action:
    """Apply e-greedy policy."""
    action_dim = q_values.shape[-1]
    if random_state.rand() <= epsilon:
        a_t = random_state.randint(0, action_dim)
    else:
        a_t = np.argmax(q_values, axis=-1)
    return np.array(a_t).item()


class EpsilonGreedyActor(types_lib.Agent):
    """DQN e-greedy actor."""

    def __init__(
        self,
        network: tf.keras.Model,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        name: str = 'DQN-greedy',
    ):
        self.agent_name = name
        self._network = network
        self._exploration_epsilon = exploration_epsilon
        self._random_state = random_state

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state"""
        """Give current timestep, return best action"""
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        q_t = self._select_action(s_t)
        q_t = np.array(q_t)
        return apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.
        This method should be called at the beginning of every episode.
        """

    def _select_action(self, s_t: tf.Tensor) -> tf.Tensor:
        """returns actions Q-values at given state."""
        q_t = self._network(s_t, training=False).q_values
        return q_t

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'exploration_epsilon': self._exploration_epsilon,
        }


class IqnEpsilonGreedyActor(EpsilonGreedyActor):
    """IQN e-greedy actor."""

    def __init__(
        self,
        network: tf.keras.Model,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        tau_samples: int,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            'IQN-greedy',
        )
        self._tau_samples = tau_samples

    def _select_action(self, s_t: tf.Tensor) -> tf.Tensor:
        q_t = self._network(s_t, num_taus=self._tau_samples).q_values
        return q_t


class DrqnEpsilonGreedyActor(EpsilonGreedyActor):
    """DRQN e-greedy actor."""

    def __init__(
        self,
        network: tf.keras.Model,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            'DRQN-greedy',
        )
        self._lstm_state = None

    def _select_action(self, s_t) -> tf.Tensor:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        network_output = self._network(s_t[None, ...], self._lstm_state)
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s
        return q_t

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


class R2d2EpsilonGreedyActor(EpsilonGreedyActor):
    """R2D2 e-greedy actor."""

    def __init__(
        self,
        network: tf.keras.Model,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            'R2D2-greedy',
        )
        self._last_action = None
        self._lstm_state = None

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Samples action from eps-greedy policy wrt Q-values at given state"""
        """Give current timestep, return best action"""
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        r_t = tf.convert_to_tensor(timestep.reward, dtype=tf.float32)
        q_t = self._select_action(s_t, r_t)
        q_t = np.array(q_t)
        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t
    
    def _select_action(self, s_t: tf.Tensor, r_t: tf.Tensor) -> tf.Tensor:
        a_tm1 = tf.convert_to_tensor(self._last_action, dtype=tf.int64)
        hidden_s = tuple(tf.convert_to_tensor(s) for s in self._lstm_state)
        network_output = self._network(
            RnnDqnNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                hidden_s=hidden_s,
            )
        )
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s
        return q_t

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._last_action = 0  # During the first step of a new episode, use 'fake' previous action for network pass
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


class NguEpsilonGreedyActor(EpsilonGreedyActor):
    """NGU e-greedy actor."""

    def __init__(
        self,
        network: tf.keras.Model,
        embedding_network: tf.keras.Model,
        rnd_target_network: tf.keras.Model,
        rnd_predictor_network: tf.keras.Model,
        episodic_memory_capacity: int,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        exploration_epsilon: float,
        random_state: np.random.RandomState,  # pylint: disable=no-member
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            'NGU-greedy',
        )

        self._policy_index = 0
        self._policy_beta = 0

        # Episodic intrinsic bonus module
        self._episodic_module = EpisodicBonusModule(
            embedding_network=embedding_network,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        # Lifelong intrinsic bonus module
        self._lifelong_module = RndLifeLongBonusModule(
            target_network=rnd_target_network, predictor_network=rnd_predictor_network, discount=0.99, Observation_shape=network.state_dim
        )

        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)

    def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
        """Give current timestep, return best action"""
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        a_tm1 = tf.convert_to_tensor(self._last_action, dtype=tf.int64)
        ext_r_t = tf.convert_to_tensor(timestep.reward, dtype=tf.float32)
        int_r_t = tf.convert_to_tensor(self.intrinsic_reward, dtype=tf.float32)
        policy_index = tf.convert_to_tensor(self._policy_index, dtype=tf.int64)
        hidden_s = tuple(tf.convert_to_tensor(s) for s in self._lstm_state)
        ngu_network_inputs = NguNetworkInputs(
                s_t=s_t[None, ...],  # [B, T, state_shape]
                a_tm1=a_tm1[None, ...],  # [B, T]
                ext_r_t=ext_r_t[None, ...],  # [B, T]
                int_r_t=int_r_t[None, ...],  # [B, T]
                policy_index=policy_index[None, ...],  # [B, T]
                hidden_s=hidden_s,
        )
        q_t = self._select_action(ngu_network_inputs)
        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t

        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)
        return a_t

    def _select_action(self, ngu_network_inputs: NguNetworkInputs) -> tf.Tensor:
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        pi_output = self._network(ngu_network_inputs)

        q_t = pi_output.q_values
        self._lstm_state = pi_output.hidden_s

        return q_t

    def reset(self) -> None:
        """Reset hidden state to zeros at new episodes."""
        self._last_action = 0  # Initialize a_tm1 to 0.
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
        self._episodic_module.reset()

    @property
    def intrinsic_reward(self) -> float:
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the NGU paper.
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)


class Agent57EpsilonGreedyActor(types_lib.Agent):
    """Agent57 e-greedy actor."""

    def __init__(
        self,
        network: tf.keras.Model,
        embedding_network: tf.keras.Model,
        rnd_target_network: tf.keras.Model,
        rnd_predictor_network: tf.keras.Model,
        episodic_memory_capacity: int,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
    ):
        self.agent_name = 'Agent57-greedy'
        self._network = network

        self._random_state = random_state
        self._exploration_epsilon = exploration_epsilon

        self._policy_index = 0
        self._policy_beta = tf.constant(0.0, tf.float32)

        # Episodic intrinsic bonus module
        self._episodic_module = EpisodicBonusModule(
            embedding_network=embedding_network,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        # Lifelong intrinsic bonus module
        self._lifelong_module = RndLifeLongBonusModule(
            target_network=rnd_target_network,
            predictor_network=rnd_predictor_network,
            discount=0.99,
            Observation_shape=network.state_dim
        )

        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._ext_lstm_state = None  # Stores LSTM hidden state and cell state for extrinsic Q network
        self._int_lstm_state = None  # Stores LSTM hidden state and cell state for intrinsic Q network

    def step(self, timestep):
        """Give current timestep, return best action"""
        a_t = self._select_action(timestep)

        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        return a_t

    def reset(self):
        """Reset hidden state to zeros at new episodes."""
        self._last_action = 0  # Initialize a_tm1 to 0.
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._ext_lstm_state, self._int_lstm_state = self._network.get_initial_hidden_state(batch_size=1)

        self._episodic_module.reset()

    def _select_action(self, timestep):
        """Samples action from eps-greedy policy wrt Q-values at given state."""
        input_ = self._prepare_network_input(timestep)

        output = self._network(input_)
        ext_q_t = tf.squeeze(output.ext_q_values)
        int_q_t = tf.squeeze(output.int_q_values)

        q_t = compute_transformed_q(ext_q_t, int_q_t, self._policy_beta)

        self._ext_lstm_state = output.ext_hidden_s
        self._int_lstm_state = output.int_hidden_s

        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t

    def _prepare_network_input(self, timestep: types_lib.TimeStep):
        # NGU network expect input shape [B, T, state_shape],
        # and additionally 'last action', 'extrinsic reward for last action', last intrinsic reward, and intrinsic reward scale beta index.
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        a_tm1 = tf.convert_to_tensor(self._last_action, dtype=tf.int64)
        ext_r_t = tf.convert_to_tensor(timestep.reward, dtype=tf.float32)
        int_r_t = tf.convert_to_tensor(self.intrinsic_reward, dtype=tf.float32)
        policy_index = tf.convert_to_tensor(self._policy_index, dtype=tf.int64)
        ext_hidden_s = tuple(tf.convert_to_tensor(s) for s in self._ext_lstm_state)
        int_hidden_s = tuple(tf.convert_to_tensor(s) for s in self._int_lstm_state)
        return Agent57NetworkInputs(
            s_t=s_t[None, ...],  # [B, T, state_shape]
            a_tm1=a_tm1[None, ...],  # [B, T]
            ext_r_t=ext_r_t[None, ...],  # [B, T]
            int_r_t=int_r_t[None, ...],  # [B, T]
            policy_index=policy_index[None, ...],  # [B, T]
            ext_hidden_s=ext_hidden_s,
            int_hidden_s=int_hidden_s,
        )

    @property
    def intrinsic_reward(self):
        """Returns intrinsic reward for last state s_tm1."""
        # Equation 1 of the NGU paper.
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t, 1.0), 5.0)

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            'exploration_epsilon': self._exploration_epsilon,
        }


class PolicyGreedyActor(types_lib.Agent):
    """Agent that acts with a given set of policy network parameters."""

    def __init__(
        self,
        network: tf.keras.Model,
        name: str = '',
    ):
        self.agent_name = name
        self._network = network

    def step(self, timestep):
        """Give current timestep, return best action"""
        return self.act(timestep)

    def act(self, timestep):
        """Selects action given a timestep."""
        return self._select_action(timestep)

    def reset(self):
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """

    def _select_action(self, timestep):
        """Samples action from policy at given state."""
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        pi_logits_t = self._network(s_t).pi_logits

        # Sample an action
        a_t = tf.random.categorical(pi_logits_t, num_samples=1)

        # # Can also try to act greedy
        # prob_t = tf.nn.softmax(pi_logits_t, axis=1)
        # a_t = tf.argmax(prob_t, axis=1)

        return np.array(a_t).item()

    @property
    def statistics(self):
        """Empty statistics"""
        return {}


class ImpalaGreedyActor(PolicyGreedyActor):
    """IMPALA greedy actor to do evaluation during training"""

    def __init__(
        self,
        network: tf.keras.Model,
    ):
        super().__init__(
            network,
            'IMPALA',
        )

        self._last_action = None
        self._hidden_s = self._network.get_initial_hidden_state(batch_size=1)

    def step(self, timestep):
        """Given timestep, return action a_t"""
        a_t = self.act(timestep)

        # Update local states after create the transition
        self._last_action = a_t

        return a_t

    def act(self, timestep):
        'Given state s_t and done marks, return an action.'
        a_t = self._choose_action(timestep)
        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode before take any action."""
        self._last_action = 0  # During the first step of a new episode, use 'fake' previous action for network pass
        self._hidden_s = self._network.get_initial_hidden_state(batch_size=1)

    def _choose_action(self, timestep):
        """Given state s_t, choose action a_t"""
        # IMPALA network requires more than just the state input, but also last action, and reward for last action
        # optionally the last hidden state from LSTM and done mask if using LSTM
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        a_tm1 = tf.convert_to_tensor(self._last_action, dtype=tf.int64)
        r_t = tf.convert_to_tensor(timestep.reward, dtype=tf.float32)
        done = tf.convert_to_tensor(timestep.done, dtype=tf.bool)

        hidden_s = tuple(tf.convert_to_tensor(s) for s in self._hidden_s)

        network_output = self._network(
            ImpalaActorCriticNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                done=done[None, ...],
                hidden_s=hidden_s,
            )
        )
        pi_logits_t = tf.squeeze(network_output.pi_logits, axis=0)  # Remove T dimension

        # Sample an action
        a_t = tf.random.categorical(pi_logits_t, num_samples=1)

        # # Can also try to act greedy
        # prob_t = tf.nn.softmax(pi_logits_t, axis=-1)
        # a_t = tf.argmax(prob_t, axis=-1)

        self._hidden_s = network_output.hidden_s  # Save last hidden state for next pass
        return np.array(a_t).item()

    @property
    def statistics(self):
        """Returns current actor's statistics as a dictionary."""
        return {}


class GaussianPolicyGreedyActor(PolicyGreedyActor):
    """Gaussian Agent that acts with a given set of policy network parameters."""

    def _select_action(self, timestep):
        """Samples action from policy at given state."""
        s_t = tf.convert_to_tensor(timestep.observation[None, ...], dtype=tf.float32)
        pi_mu, pi_sigma = self._network(s_t)
        # Sample an action
        a_t = tf.random.normal(shape=pi_mu.shape, mean=pi_mu, stddev=pi_sigma)
        return np.array(a_t).squeeze(0)
