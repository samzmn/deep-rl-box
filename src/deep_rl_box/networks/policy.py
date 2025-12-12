"""Networks for policy-based learning methods like Actor-Critic and its variants"""
import numpy as np
import tensorflow as tf
from typing import NamedTuple, Optional, Tuple

# pylint: disable=import-error
from deep_rl_box.networks import common


class ActorNetworkOutputs(NamedTuple):
    pi_logits: tf.Tensor


class CriticNetworkOutputs(NamedTuple):
    value: tf.Tensor


class ActorCriticNetworkOutputs(NamedTuple):
    pi_logits: tf.Tensor
    value: tf.Tensor


class ImpalaActorCriticNetworkOutputs(NamedTuple):
    pi_logits: tf.Tensor
    value: tf.Tensor
    hidden_s: tf.Tensor


class ImpalaActorCriticNetworkInputs(NamedTuple):
    s_t: tf.Tensor
    a_tm1: tf.Tensor
    r_t: tf.Tensor  # reward for (s_tm1, a_tm1), but received at current timestep.
    done: tf.Tensor
    hidden_s: Optional[Tuple[tf.Tensor]]


class RndActorCriticNetworkOutputs(NamedTuple):
    """Random Network Distillation"""

    pi_logits: tf.Tensor
    int_baseline: tf.Tensor  # intrinsic value head
    ext_baseline: tf.Tensor  # extrinsic value head


# =================================================================
# Fully connected Neural Networks
# =================================================================


class ActorMlpNet(tf.keras.Model):
    """Actor MLP network."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

    def call(self, x: tf.Tensor) -> ActorNetworkOutputs:
        """Given raw state x, predict the action probability distribution."""
        # Predict action distributions wrt policy
        pi_logits = self.net(x)

        return ActorNetworkOutputs(pi_logits=pi_logits)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CriticMlpNet(tf.keras.Model):
    """Critic MLP network."""

    def __init__(self, state_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x: tf.Tensor) -> CriticNetworkOutputs:
        """Given raw state x, predict the state-value."""
        value = self.net(x)
        return CriticNetworkOutputs(value=value)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ActorCriticMlpNet(tf.keras.Model):
    """Actor-Critic MLP network."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
        ])

        self.policy_head = tf.keras.Sequential([
            # tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")
            tf.keras.layers.Dense(action_dim),
        ])
        self.baseline_head = tf.keras.Sequential([
            # tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")
            tf.keras.layers.Dense(1),
        ])

    def call(self, x: tf.Tensor) -> ActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution
        and state-value."""
        # Extract features from raw input state
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        value = self.baseline_head(features)

        return ActorCriticNetworkOutputs(pi_logits=pi_logits, value=value)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GaussianActorMlpNet(tf.keras.Model):
    """Gaussian Actor MLP network for continuous action space."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
        ])

        self.mu_head = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[hidden_size]),
            # tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(action_dim),
        ])

        # self.sigma_head = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=[hidden_size]),
        #     # tf.keras.layers.Dense(hidden_size, activation='tanh'),
        #     tf.keras.layers.Dense(action_dim),
        # )

        self.logstd = tf.Variable(tf.zeros((1, action_dim)), trainable=True)

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Given raw state x, predict the action probability distribution
        and state-value."""
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_mu = self.mu_head(features)

        logstd = tf.broadcast_to(self.logstd, tf.shape(pi_mu))
        pi_sigma = tf.exp(logstd)

        return pi_mu, pi_sigma

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GaussianCriticMlpNet(tf.keras.Model):
    """Gaussian Critic MLP network for continuous action space."""

    def __init__(self, state_dim: int, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(hidden_size, activation='tanh'),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Given raw state x, predict the state-value."""

        # Predict state-value
        value = self.net(x)

        return value

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "hidden_size": self.hidden_size
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ImpalaActorCriticMlpNet(tf.keras.Model):
    """IMPALA(Importance Weighted Actor-Learner Architecture) Actor-Critic MLP network, with LSTM."""

    def __init__(self, state_dim: int, action_dim: int, use_lstm: bool = False, **kwargs) -> None:
        """
        Args:
            state_dim: state space size of environment state dimension
            action_dim: action space size of number of actions of the environment
            feature_size: number of units of the last feature representation linear layer
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[self.state_dim]),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
        ])

        # Feature representation output size + one-hot of last action + last reward.
        core_output_size = 64 + self.action_dim + 1

        if self.use_lstm: # input_size=core_output_size
            self.lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
            core_output_size = 64

        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"), # input_shape=core_output_size
            tf.keras.layers.Dense(action_dim),
        ])

        self.baseline_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"), # input_shape=core_output_size
            tf.keras.layers.Dense(1),
        ])

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode."""
        if self.use_lstm:
            # Shape should be num_tf.keras.layers, batch_size, hidden_size, note lstm expects two hidden states.
            return tuple(tf.zeros((1, batch_size, 64)) for _ in range(2))
        else:
            return tuple()

    def call(self, input_: ImpalaActorCriticNetworkInputs) -> ImpalaActorCriticNetworkOutputs:
        """
        Given state, predict the action probability distribution and state-value,
        T refers to the time dimension ranging from 0 to T-1. B refers to the batch size

        If self.use_lstm is set to True, and no hidden_s is given, will use zero start method.

        Args:
            input_: the ImpalaActorCriticNetworkInputs which contains the follow attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tm1: action taken in t-1 timestep, shape [T, B].
                r_t: reward for state-action pair (stm1, a_tm1), shape [T, B].
                done: current timestep state s_t is done state, shape [T, B].
                hidden_s: (optional) LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_tf.keras.layers, B, lstm_hidden_size).

        Returns:
            ImpalaActorCriticNetworkOutputs object with the following attributes:
                pi_logits: action probability logits.
                value: state-value.
                hidden_s: (optional) hidden state from LSTM layer output.
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        done = input_.done
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape  # [T, B, state_shape]
        x = tf.reshape(s_t, (T * B, -1))  # Merge time and batch.

        # Extract features from raw input state
        x = self.body(x)

        # Append clipped last reward and one hot last action.
        one_hot_a_tm1 = tf.one_hot(tf.reshape(a_tm1, (T * B,)), self.action_dim, dtype=tf.float32)
        rewards = tf.clip_by_value(tf.reshape(r_t, (T * B, 1)), -1, 1)  # Clip reward [-1, 1]
        core_input = tf.concat([x, rewards, one_hot_a_tm1], axis=-1)

        if self.use_lstm:
            assert done.dtype == tf.bool
            # Pass through RNN LSTM layer
            core_input = tf.reshape(core_input, (T, B, -1))
            lstm_output_list = []
            notdone = tf.cast(~done, tf.float32)

            # Use zero start if not given
            if hidden_s is None:
                hidden_s = self.get_initial_hidden_state(B)

            for inpt, n_d in zip(tf.unstack(core_input), tf.unstack(notdone)):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_tf.keras.layers, B, hidden_size)
                n_d = tf.reshape(n_d, (1, -1, 1))
                hidden_s = tuple(n_d * s for s in hidden_s)
                output, hidden_s = self.lstm(tf.expand_dims(inpt, 0), initial_state=hidden_s)  # LSTM takes input x and previous hidden units
                lstm_output_list.append(output)

            core_output = tf.reshape(tf.concat(lstm_output_list, axis=0), (T * B, -1))
        else:
            core_output = core_input
            hidden_s = tuple()

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(core_output)

        # Predict state-value value
        value = self.baseline_head(core_output)

        # Reshape to matching original shape
        pi_logits = tf.reshape(pi_logits, (T, B, self.action_dim))
        value = tf.reshape(value, (T, B))
        return ImpalaActorCriticNetworkOutputs(pi_logits=pi_logits, value=value, hidden_s=hidden_s)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_lstm": self.use_lstm
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RndActorCriticMlpNet(tf.keras.Model):
    """Actor-Critic MLP network with two value heads.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
        ])

        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

        self.ext_baseline_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1),
        ])

        self.int_baseline_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x: tf.Tensor) -> RndActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution,
        and extrinsic and intrinsic value values."""
        # Extract features from raw input state
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        ext_baseline = self.ext_baseline_head(features)
        int_baseline = self.int_baseline_head(features)

        return RndActorCriticNetworkOutputs(pi_logits=pi_logits, ext_baseline=ext_baseline, int_baseline=int_baseline)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =================================================================
# Convolutional Neural Networks
# =================================================================


class ActorConvNet(tf.keras.Model):
    """Actor Conv2D network."""

    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

    def call(self, x: tf.Tensor) -> ActorNetworkOutputs:
        """Given raw state x, predict the action probability distribution."""
        # Extract features from raw input state
        x = tf.cast(x, tf.float32) / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)
        return ActorNetworkOutputs(pi_logits=pi_logits)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CriticConvNet(tf.keras.Model):
    """Critic Conv2D network."""

    def __init__(self, state_dim: Tuple[int, int, int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.baseline_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x: tf.Tensor) -> CriticNetworkOutputs:
        """Given raw state x, predict the state-value."""
        # Extract features from raw input state
        x = tf.cast(x, tf.float32) / 255.0
        features = self.body(x)

        # Predict state-value
        value = self.baseline_head(features)
        return CriticNetworkOutputs(value=value)

    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ActorCriticConvNet(tf.keras.Model):
    """Actor-Critic Conv2D network."""

    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

        self.baseline_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x: tf.Tensor) -> ActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution
        and state-value."""
        # Extract features from raw input state
        x = tf.cast(x, tf.float32) / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        value = self.baseline_head(features)

        return ActorCriticNetworkOutputs(pi_logits=pi_logits, value=value)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class ImpalaActorCriticConvNet(tf.keras.Model):
    """IMPALA Actor-Critic Conv2D network, with LSTM.

    Reference code from Facebook Torchbeast:
    https://github.com/facebookresearch/torchbeast/blob/0af07b051a2176a8f9fd20c36891ba2bba6bae68/torchbeast/polybeast_learner.py#L135
    """

    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, use_lstm: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm

        assert state_dim[1] == state_dim[2] == 84

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        input_channels = state_dim[0]
        for num_ch in [16, 32, 32]:
            feats_convs = [
                tf.keras.layers.Input(shape=[84, 84, input_channels]),
                tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal"),
                tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', kernel_initializer="he_normal")
            ]
            self.feat_convs.append(tf.keras.Sequential(feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = [
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal"),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")
                ]
                if i == 0:
                    self.resnet1.append(tf.keras.Sequential(resnet_block))
                else:
                    self.resnet2.append(tf.keras.Sequential(resnet_block))

        self.fc = tf.keras.layers.Dense(256)

        # Feature representation output size + one-hot of last action + last reward.
        core_output_size = 256 + self.action_dim + 1

        if self.use_lstm:
            self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
            core_output_size = 256

        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

        self.baseline_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1),
        ])

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode.
        """
        if self.use_lstm:
            # Shape should be num_tf.keras.layers, batch_size, hidden_size, note lstm expects two hidden states.
            return tuple(tf.zeros((1, batch_size, 256)) for _ in range(2))
        else:
            return tuple()

    def call(self, input_: ImpalaActorCriticNetworkInputs) -> ImpalaActorCriticNetworkOutputs:
        """
        Given state, predict the action probability distribution and state-value,
        T refers to the time dimension ranging from 0 to T-1. B refers to the batch size

        If self.use_lstm is set to True, and no hidden_s is given, will use zero start method.

        Args:
            input_: the ImpalaActorCriticNetworkInputs which contains the follow attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tm1: action taken in t-1 timestep, shape [T, B].
                r_t: reward for state-action pair (stm1, a_tm1), shape [T, B].
                done: current timestep state s_t is done state, shape [T, B].
                hidden_s: (optional) LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_tf.keras.layers, B, lstm_hidden_size).

        Returns:
            ImpalaActorCriticNetworkOutputs object with the following attributes:
                pi_logits: action probability logits.
                value: state-value.
                hidden_s: (optional) hidden state from LSTM layer output.
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        done = input_.done
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape  # [T, B, state_dim].
        x = tf.reshape(s_t, (T * B, 84, 84, -1))  # Merge time and batch.
        x = tf.cast(x, tf.float32) / 255.0

        # Extract features from raw input state
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1i
            x += res_input
            res_input = x
            x = self.resnet2i
            x += res_input

        x = tf.nn.relu(x)
        x = tf.reshape(x, (T * B, -1))
        x = tf.nn.relu(self.fc(x))

        # Append clipped last reward and one hot last action.
        one_hot_a_tm1 = tf.one_hot(tf.reshape(a_tm1, (T * B,)), self.action_dim, dtype=tf.float32)
        rewards = tf.clip_by_value(tf.reshape(r_t, (T * B, 1)), -1, 1)  # Clip reward [-1, 1]
        core_input = tf.concat([x, rewards, one_hot_a_tm1], axis=-1)

        if self.use_lstm:
            assert done.dtype == tf.bool

            # Pass through RNN LSTM layer
            core_input = tf.reshape(core_input, (T, B, -1))
            lstm_output_list = []
            notdone = tf.cast(~done, tf.float32)

            # Use zero start if not given
            if hidden_s is None:
                hidden_s = self.get_initial_hidden_state(B)

            for inpt, nd in zip(tf.unstack(core_input), tf.unstack(notdone)):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_tf.keras.layers, B, hidden_size)
                nd = tf.reshape(nd, (1, -1, 1))
                hidden_s = tuple(nd * s for s in hidden_s)
                output, hidden_s = self.lstm(tf.expand_dims(inpt, 0), initial_state=hidden_s)  # LSTM takes input x and previous hidden units
                lstm_output_list.append(output)
            core_output = tf.reshape(tf.concat(lstm_output_list, axis=0), (T * B, -1))
        else:
            core_output = core_input
            hidden_s = tuple()

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(core_output)

        # Predict state-value value
        value = self.baseline_head(core_output)

        # Reshape to matching original shape
        pi_logits = tf.reshape(pi_logits, (T, B, self.action_dim))
        value = tf.reshape(value, (T, B))
        return ImpalaActorCriticNetworkOutputs(pi_logits=pi_logits, value=value, hidden_s=hidden_s)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_lstm": self.use_lstm
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class RndActorCriticConvNet(tf.keras.Model):
    """Actor-Critic Conv2D network with two value heads.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=state_dim),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(448, activation='relu', kernel_initializer="he_normal"),
        ])

        self.extra_policy_fc = tf.keras.layers.Dense(448, activation='relu', kernel_initializer="he_normal")
        self.extra_value_fc = tf.keras.layers.Dense(448, activation='relu', kernel_initializer="he_normal")

        self.policy_head = tf.keras.layers.Dense(action_dim)
        self.ext_value_head = tf.keras.layers.Dense(1)
        self.int_value_head = tf.keras.layers.Dense(1)

        # Initialize weights
        for layer in self.body.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                layer.kernel_initializer = tf.keras.initializers.Orthogonal(np.sqrt(2))
                layer.bias_initializer = tf.keras.initializers.Zeros()

        for layer in [self.extra_policy_fc, self.extra_value_fc]:
            layer.kernel_initializer = tf.keras.initializers.Orthogonal(np.sqrt(0.1))
            layer.bias_initializer = tf.keras.initializers.Zeros()

        for layer in [self.policy_head, self.ext_value_head, self.int_value_head]:
            layer.kernel_initializer = tf.keras.initializers.Orthogonal(np.sqrt(0.01))
            layer.bias_initializer = tf.keras.initializers.Zeros()

    def call(self, x: tf.Tensor) -> RndActorCriticNetworkOutputs:
        """Given raw state x, predict the action probability distribution,
        and extrinsic and intrinsic value values."""
        # Extract features from raw input state
        x = tf.cast(x, tf.float32) / 255.0
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_features = features + self.extra_policy_fc(features)
        pi_logits = self.policy_head(pi_features)

        # Predict state-value
        value_features = features + self.extra_value_fc(features)
        ext_baseline = self.ext_value_head(value_features)
        int_baseline = self.int_value_head(value_features)

        return RndActorCriticNetworkOutputs(pi_logits=pi_logits, ext_baseline=ext_baseline, int_baseline=int_baseline)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
