"""Networks for value-based learning methods like DQN and its variants"""
from itertools import chain
from typing import NamedTuple, Optional, Tuple
import tensorflow as tf

import deep_rl_box.networks.common as common


class DqnNetworkOutputs(NamedTuple):
    q_values: tf.Tensor


class C51NetworkOutputs(NamedTuple):
    q_values: tf.Tensor
    q_logits: tf.Tensor  # use logits and log_softmax() when calculate loss to avoid log() on zero cause NaN


class QRDqnNetworkOutputs(NamedTuple):
    q_values: tf.Tensor
    q_dist: tf.Tensor


class IqnNetworkOutputs(NamedTuple):
    q_values: tf.Tensor
    q_dist: tf.Tensor
    taus: tf.Tensor


class RnnDqnNetworkInputs(NamedTuple):
    s_t: tf.Tensor
    a_tm1: tf.Tensor
    r_t: tf.Tensor  # reward for (s_tm1, a_tm1), but received at current timestep.
    hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]


class RnnDqnNetworkOutputs(NamedTuple):
    q_values: tf.Tensor
    hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]


class NguNetworkInputs(NamedTuple):
    """Never give up agent network input."""

    s_t: tf.Tensor
    a_tm1: tf.Tensor
    ext_r_t: tf.Tensor  # extrinsic reward for (s_tm1, a_tm1), but received at current timestep.
    int_r_t: tf.Tensor  # intrinsic reward for (s_tm1)
    policy_index: tf.Tensor  # index for intrinsic reward scale beta and discount
    hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]


class Agent57NetworkInputs(NamedTuple):
    """Never give up agent network input."""

    s_t: tf.Tensor
    a_tm1: tf.Tensor
    ext_r_t: tf.Tensor  # extrinsic reward for (s_tm1, a_tm1), but received at current timestep.
    int_r_t: tf.Tensor  # intrinsic reward for (s_tm1)
    policy_index: tf.Tensor  # index for intrinsic reward scale beta and discount
    ext_hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]
    int_hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]


class Agent57NetworkOutputs(NamedTuple):
    ext_q_values: tf.Tensor
    int_q_values: tf.Tensor
    ext_hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]
    int_hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]]


# =================================================================
# Fully connected Neural Networks
# =================================================================


class DqnMlpNet(tf.keras.Model):
    """MLP DQN network."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])

    def call(self, x: tf.Tensor) -> DqnNetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        q_values = self.body(x)  # [batch_size, action_dim]
        return DqnNetworkOutputs(q_values=q_values)

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


class DuelingDqnMlpNet(tf.keras.Model):
    """MLP Dueling DQN network."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
        ])

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x: tf.Tensor) -> DqnNetworkOutputs:
        """Given state, return state-action value for all possible actions"""

        features = self.body(x)

        advantages = self.advantage_head(features)  # [batch_size, action_dim]
        values = self.value_head(features)  # [batch_size, 1]

        q_values = values + (advantages - tf.reduce_max(advantages, axis=1, keepdims=True))  # [batch_size, action_dim]

        return DqnNetworkOutputs(q_values=q_values)

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


class C51DqnMlpNet(tf.keras.Model):
    """C51 DQN MLP network."""

    def __init__(self, state_dim: int, action_dim: int, atoms: tf.Tensor, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            atoms: the support for q value distribution, used here to turn Z into Q values
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.shape[0]

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim * self.num_atoms),
        ])

    def call(self, x: tf.Tensor) -> C51NetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        x = self.body(x)

        q_logits = tf.reshape(x, [-1, self.action_dim, self.num_atoms])  # [batch_size, action_dim, num_atoms]
        q_dist = tf.nn.softmax(q_logits, axis=-1)
        atoms = tf.expand_dims(tf.expand_dims(self.atoms, axis=0), axis=0)
        q_values = tf.reduce_sum(q_dist * atoms, axis=-1)  # [batch_size, action_dim]

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "atoms": self.atoms.numpy().tolist()
        }

    @classmethod
    def from_config(cls, config):
        atoms = tf.convert_to_tensor(config.pop('atoms'))
        return cls(atoms=atoms, **config)


class RainbowSimpleDqnMlpNet(tf.keras.Model):
    def __init__(self, state_dim: int, action_dim: int, atoms: tf.Tensor, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output liner layer
            atoms: the support for q value distribution, used here to turn Z into Q values
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.shape[0]
        
        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
        ])

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="elu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(action_dim * self.num_atoms),
        ])
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="elu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1 * self.num_atoms),
        ])

    def call(self, x: tf.Tensor) -> C51NetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        x = self.body(x)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        advantages = tf.reshape(advantages, [-1, self.action_dim, self.num_atoms])
        values = tf.reshape(values, [-1, 1, self.num_atoms])

        q_logits = values + (advantages - tf.reduce_max(advantages, axis=-1, keepdims=True))

        q_logits = tf.reshape(q_logits, [-1, self.action_dim, self.num_atoms])  # [batch_size, action_dim, num_atoms]

        q_dist = tf.nn.softmax(q_logits, axis=-1)
        atoms = tf.expand_dims(tf.expand_dims(self.atoms, axis=0), axis=0)
        q_values = tf.reduce_sum(q_dist * atoms, axis=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "atoms": self.atoms.numpy().tolist()
        }

    @classmethod
    def from_config(cls, config):
        config['atoms'] = tf.convert_to_tensor(config.pop('atoms'))
        return cls(**config)


def get_rainbow_dqn_mlp_net(state_dim: int, action_dim: int, atoms: tf.Tensor, units: int = 128, activation="elu", sigma_init=0.5) -> tf.keras.Model:
    """
    Args:
        state_dim: the shape of the input tensor to the neural network
        action_dim: the number of units for the output liner layer
        atoms: the support for q value distribution, used here to turn Z into Q values
    """
    num_atoms = atoms.shape[0]
    if action_dim < 1:
        raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
    if state_dim < 1:
        raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
    if len(atoms.shape) != 1:
        raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')
    
    input_states = tf.keras.layers.Input(shape=[state_dim])
    input_states = tf.keras.layers.BatchNormalization()(input_states)
    hidden1 = tf.keras.layers.Dense(units, activation=activation)(input_states)
    hidden1 = tf.keras.layers.BatchNormalization()(hidden1)
    hidden2 = tf.keras.layers.Dense(units*2, activation=activation)(hidden1)
    hidden2 = tf.keras.layers.BatchNormalization()(hidden2)

    state_values = common.NoisyDense(units, sigma_init=sigma_init, activation=activation)(hidden2)
    state_values = common.NoisyDense(1 * num_atoms, sigma_init=sigma_init)(state_values)
    state_values = tf.keras.layers.Reshape(target_shape=[1, num_atoms])(state_values)

    raw_advantages = common.NoisyDense(units, sigma_init=sigma_init, activation=activation)(hidden2)
    raw_advantages = common.NoisyDense(action_dim * num_atoms, sigma_init=sigma_init)(raw_advantages)
    raw_advantages = tf.keras.layers.Reshape(target_shape=[action_dim, num_atoms])(raw_advantages)

    advantages = raw_advantages - tf.reduce_max(raw_advantages, axis=-1, keepdims=True)
    q_logits = state_values + advantages # [batch_size, action_dim, num_atoms]

    q_dist = tf.nn.softmax(q_logits, axis=-1)
    q_values = tf.reduce_sum(q_dist * atoms[None, None, ...], axis=-1)

    model = tf.keras.Model(inputs=[input_states], outputs={"q_logits": q_logits, "q_values": q_values})
    return model


class RainbowDqnMlpNet(tf.keras.Model):
    """Rainbow combines C51, dueling architecture, and noisy net."""

    def __init__(self, state_dim: int, action_dim: int, atoms: tf.Tensor, units: int = 128, sigma_init: int = 0.5, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output liner layer
            atoms: the support for q value distribution, used here to turn Z into Q values
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.shape[0]
        self.units = units
        hidden_dense_1_units = int(units / 2)

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(hidden_dense_1_units, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.units, activation='elu', kernel_initializer="he_normal"),
        ])

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[self.units]),
            tf.keras.layers.BatchNormalization(),
            common.NoisyDense(hidden_dense_1_units, sigma_init=sigma_init, activation="elu"),
            tf.keras.layers.BatchNormalization(),
            common.NoisyDense(action_dim * self.num_atoms, sigma_init=sigma_init),
        ])
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[self.units]),
            tf.keras.layers.BatchNormalization(),
            common.NoisyDense(hidden_dense_1_units, sigma_init=sigma_init, activation="elu"),
            tf.keras.layers.BatchNormalization(),
            common.NoisyDense(1 * self.num_atoms, sigma_init=sigma_init),
        ])

    def build(self, input_shape):
        super().build(input_shape)
        self.body.build([None, self.state_dim])
        self.advantage_head.build([None, self.units])
        self.value_head.build([None, self.units])

    def call(self, x: tf.Tensor, training=None) -> C51NetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        x = self.body(x)
        advantages = self.advantage_head(x, training=training)
        values = self.value_head(x, training=training)

        advantages = tf.reshape(advantages, [-1, self.action_dim, self.num_atoms])
        values = tf.reshape(values, [-1, 1, self.num_atoms])

        q_logits = values + (advantages - tf.reduce_max(advantages, axis=1, keepdims=True))

        q_logits = tf.reshape(q_logits, [-1, self.action_dim, self.num_atoms])  # [batch_size, action_dim, num_atoms]

        q_dist = tf.nn.softmax(q_logits, axis=-1)
        atoms = tf.expand_dims(tf.expand_dims(self.atoms, axis=0), axis=0)
        q_values = tf.reduce_sum(q_dist * atoms, axis=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

    # def reset_noise(self) -> None:
    #     """Reset noisy layer"""
    #     # combine two lists into one: list(chain(*zip(a, b)))
    #     for layer in list(chain(*zip(self.advantage_head.layers, self.value_head.layers))):
    #         if isinstance(layer, common.NoisyDense):
    #             layer.reset_noise()

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "atoms": self.atoms.numpy().tolist(),
            "units": self.units,
        }

    @classmethod
    def from_config(cls, config):
        config['atoms'] = tf.convert_to_tensor(config.pop('atoms'))
        return cls(**config)


class QRDqnMlpNet(tf.keras.Model):
    """Quantile Regression DQN MLP network."""

    def __init__(self, state_dim: int, action_dim: int, quantiles: tf.Tensor, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output liner layer
            quantiles: the quantiles for QR DQN
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if len(quantiles.shape) != 1:
            raise ValueError(f'Expect quantiles to be a 1D tensor, got {quantiles.shape}')

        super().__init__(**kwargs)
        self.taus = quantiles
        self.num_taus = quantiles.shape[0]
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim * self.num_taus),
        ])

    def call(self, x: tf.Tensor) -> QRDqnNetworkOutputs:
        """Given state, return state-action value for all possible actions."""
        # No softmax as the model is trying to approximate the 'whole' probability distributions
        q_dist = tf.reshape(self.body(x), [-1, self.num_taus, self.action_dim])  # [batch_size, num_taus, action_dim]
        q_values = tf.reduce_mean(q_dist, axis=1)

        return QRDqnNetworkOutputs(q_values=q_values, q_dist=q_dist)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "quantiles": self.taus.numpy().tolist()
        }

    @classmethod
    def from_config(cls, config):
        config['quantiles'] = tf.convert_to_tensor(config.pop('quantiles'))
        return cls(**config)


class IqnMlpNet(tf.keras.Model):
    """Implicit Quantile MLP network."""

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            latent_dim: the cos embedding linear layer input shapes
        """
        super().__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        if latent_dim < 1:
            raise ValueError(f'Expect latent_dim to be a positive integer, got {latent_dim}')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.pis = tf.range(1, self.latent_dim + 1, dtype=tf.float32) * 3.141592653589793  # [latent_dim]

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal")
        ])

        self.embedding_layer = tf.keras.layers.Dense(128)
        self.value_head = tf.keras.layers.Dense(action_dim)

    def sample_taus(self, batch_size: int, num_taus: int) -> tf.Tensor:
        """Returns sampled batch taus."""
        taus = tf.random.uniform((batch_size, num_taus), dtype=tf.float32)
        return taus

    def call(self, x: tf.Tensor, num_taus: int = 32) -> IqnNetworkOutputs:
        """
        Args:
            state: environment state, shape (B, state_shape)
            taus: tau embedding samples, shape (B, num_taus)

        Returns:
            q_values: # [batch_size, action_dim]
            q_dist: # [batch_size, action_dim, num_taus]
        """
        batch_size = tf.shape(x)[0]
        features = self.body(x)

        taus = self.sample_taus(batch_size, num_taus)

        pis = tf.expand_dims(tf.expand_dims(self.pis, 0), 0)
        tau_embedding = tf.cos(pis * tf.expand_dims(taus, -1))  # [batch_size, num_taus, latent_dim]

        tau_embedding = tf.reshape(tau_embedding, (batch_size * num_taus, -1))  # [batch_size x num_taus, latent_dim]
        tau_embedding = self.embedding_layer(tau_embedding)  # [batch_size x num_taus, embedding_layer_output]

        tau_embedding = tf.reshape(tau_embedding, (batch_size, num_taus, -1))
        head_input = tau_embedding * tf.expand_dims(features, 1)  # [batch_size, num_taus, embedding_layer_output]

        head_input = tf.reshape(head_input, (-1, self.embedding_layer.units))

        q_dist = self.value_head(head_input)  # [batch_size x num_taus, action_dim]
        q_dist = tf.reshape(q_dist, (batch_size, num_taus, self.action_dim))  # [batch_size, num_taus, action_dim]
        q_values = tf.reduce_mean(q_dist, axis=1)  # [batch_size, action_dim]

        return IqnNetworkOutputs(q_values=q_values, q_dist=q_dist, taus=taus)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DrqnMlpNet(tf.keras.Model):
    """DRQN MLP network."""

    def __init__(self, state_dim: int, action_dim: int, units: int = 128, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        super(DrqnMlpNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_units = units

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal")
        ])

        self.lstm = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, return_state=True)

        self.value_head = tf.keras.layers.Dense(action_dim)

    def call(self, x: tf.Tensor, hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the batch of state tensor, shape [B, T, state_shape]
            hidden_s: the initial/last time step hidden state from lstm
        Returns:
            q_values: state-action values
            hidden_s: hidden state from LSTM layer
        """
        # Expect x shape to be [B, T, state_shape]
        assert len(x.shape) == 3
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        x = tf.reshape(x, shape=[T * B, *x.shape[2:]])  # Merge batch and time dimension.

        x = self.body(x)
        x = tf.reshape(x, (B, T, -1))  # LSTM expect rank 3

        x, hidden_h, hidden_c = self.lstm(x, initial_state=hidden_s)

        x = tf.reshape(x, shape=[T * B, *x.shape[2:]])  # Merge batch and time dimension.
        q_values = self.value_head(x)
        q_values = tf.reshape(q_values, (B, T, -1))  # reshape to in the range [B, T, action_dim]
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=(hidden_h, hidden_c))

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""
        return (tf.zeros((batch_size, self.lstm_units)), tf.zeros((batch_size, self.lstm_units)))

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "units": self.lstm_units,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class R2d2DqnMlpNet(tf.keras.Model):
    """R2D2 DQN MLP network."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        super().__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal")
        ])

        core_output_size = 128 + action_dim + 1

        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, input_: RnnDqnNetworkInputs) -> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the RnnDqnNetworkInputs object which contains the following attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tm1: action taken in t-1 timestep, shape [T, B].
                r_t: reward for state-action pair (s_tm1, a_tm1), shape [T, B].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            RnnDqnNetworkOutputs object with the following attributes:
                q_values: state-action values.
                hidden_s: hidden state from LSTM layer output, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        hidden_s = input_.hidden_s

        T = tf.shape(s_t)[0]
        B = tf.shape(s_t)[1]

        x = tf.reshape(s_t, shape=tf.concat([[T * B], x.shape[2:]], axis=-1))  # Merge batch and time dimension.

        x = self.body(x)
        x = tf.reshape(x, (T * B, -1))

        one_hot_a_tm1 = tf.one_hot(tf.reshape(a_tm1, (T * B,)), self.action_dim, dtype=tf.float32)

        reward = tf.reshape(r_t, (T * B, 1))
        core_input = tf.concat([x, reward, one_hot_a_tm1], axis=-1)
        core_input = tf.reshape(core_input, (T, B, -1))  # LSTM expect rank 3 tensor.

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)

        x, hidden_h, hidden_c = self.lstm(core_input, initial_state=hidden_s)

        x = tf.reshape(x, shape=tf.concat([[T * B], x.shape[2:]], axis=-1))  # Merge batch and time dimension.
        advantages = self.advantage_head(x)  # [T*B, action_dim]
        values = self.value_head(x)  # [T*B, 1]

        q_values = values + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        q_values = tf.reshape(q_values, (T, B, -1))  # reshape to in the range [B, T, action_dim]
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=(hidden_h, hidden_c))

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""
        return (tf.zeros((batch_size, 128)), tf.zeros((batch_size, 128)))

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


class NguDqnMlpNet(tf.keras.Model):
    """NGU(Never Give UP) DQN MLP network."""

    def __init__(self, state_dim: int, action_dim: int, num_policies: int, units = 128, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output linear layer.
            num_policies: the number of mixtures for intrinsic reward scale betas.
        """
        super().__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if state_dim < 1:
            raise ValueError(f'Expect state_dim to be a positive integer, got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_policies = num_policies  # intrinsic reward scale betas
        self.n_lstm_units = units

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(units//2, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units, activation='elu', kernel_initializer="he_normal")
        ])

        # Core input includes:
        # feature representation output size
        # one-hot of intrinsic reward scale beta
        # one-hot of last action
        # last intrinsic reward
        # last extrinsic reward
        # core_output_size = 128 + self.num_policies + self.action_dim + 1 + 1

        self.lstm = tf.keras.layers.LSTM(self.n_lstm_units, return_sequences=True, return_state=True)

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1)
        ])

    def build(self, input_shape):
        input_shape = input_shape.s_t
        super().build(input_shape)
        self.body.build((None, input_shape[-1]))
        self.lstm.build((None, None, self.n_lstm_units + self.num_policies + self.action_dim + 2))
        self.advantage_head.build((None, self.n_lstm_units))
        self.value_head.build((None, self.n_lstm_units))

    def call(self, input_: NguNetworkInputs) -> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the NguNetworkInputs object which contains the following attributes:
                s_t: timestep t environment state, shape [B, T, state_shape].
                a_tm1: action taken in s_tm1, shape [B, T].
                ext_r_t: extrinsic reward for state-action pair (s_tm1, a_tm1), shape [B, T].
                int_r_t: intrinsic reward for state s_tm1, shape [B, T].
                policy_index: the index for the pair of intrinsic reward scale beta and discount gamma, shape [B, T].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            RnnDqnNetworkOutputs object with the following attributes:
                q_values: state-action values.
                hidden_s: hidden state from LSTM layer output.
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        ext_r_t = input_.ext_r_t
        int_r_t = input_.int_r_t
        policy_index = input_.policy_index
        hidden_s = input_.hidden_s

        B = tf.shape(s_t)[0]
        T = tf.shape(s_t)[1]
    
        x = tf.reshape(s_t, shape=[-1, *s_t.shape[2:]])  # Merge batch and time dimension.
        x = self.body(x)
        x = tf.reshape(x, (B * T, -1))

        one_hot_beta = tf.one_hot(tf.reshape(policy_index, (-1)), self.num_policies, dtype=tf.float32) # (B * T)
        one_hot_a_tm1 = tf.one_hot(tf.reshape(a_tm1, (-1)), self.action_dim, dtype=tf.float32) # (B * T)
        int_reward = tf.reshape(int_r_t, (-1, 1)) # (B * T)
        ext_reward = tf.reshape(ext_r_t, (-1, 1)) # (B * T)

        core_input = tf.concat([x, ext_reward, one_hot_a_tm1, int_reward, one_hot_beta], axis=-1)
        core_input = tf.reshape(core_input, (B, T, -1))  # LSTM expect rank 3 tensor.

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)

        x, hidden_h, hidden_c = self.lstm(core_input, initial_state=hidden_s)

        x = tf.reshape(x, shape=[-1, *x.shape[2:]])  # Merge batch and time dimension.
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        q_values = values + (advantages - tf.reduce_mean(advantages, axis=0, keepdims=True))
        q_values = tf.reshape(q_values, (B, T, -1))  # reshape to in the range [B, T, action_dim]
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=(hidden_h, hidden_c))

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""
        return (tf.zeros((batch_size, self.n_lstm_units)), tf.zeros((batch_size, self.n_lstm_units)))

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_policies": self.num_policies,
            "units": self.n_lstm_units,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_ngu_mlp_net(state_dim: int, action_dim: int, num_policies: int, units=128):
    body = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[state_dim]),
        tf.keras.layers.Dense(units//2, activation='elu', kernel_initializer="he_normal"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units, activation='elu', kernel_initializer="he_normal")
    ])

    lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, stateful=True)

    advantage_head = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='elu', kernel_initializer="he_normal"),
        tf.keras.layers.Dense(action_dim)
    ])
    value_head = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='elu', kernel_initializer="he_normal"),
        tf.keras.layers.Dense(1)
    ])

    state_input = tf.keras.layers.Input()
    state_input = tf.keras.layers.TimeDistributed(body(state_input))
    
    int_reward_input = tf.keras.layers.Input()
    ext_reward_input = tf.keras.layers.Input()
    a_tm1_input = tf.keras.layers.Input()
    policy_input = tf.keras.layers.Input()
    
    core_input = tf.keras.layers.concatenate([state_input, ext_reward_input, a_tm1_input, int_reward_input, policy_input])

    x, hidden_states = lstm(core_input)

    advantages = advantage_head(x)
    values = value_head(x)

    q_values = values + (advantages - tf.reduce_max(advantages, axis=0, keepdims=True)) # [B, T, action_dim]

    model = tf.keras.Model(inputs=[state_input, a_tm1_input, ext_reward_input, int_reward_input, policy_input],
                           outputs=[q_values, hidden_states])
    return model

def get_agent57_mlp_net(state_dim: int, action_dim: int, num_policies: int, units=128):
    ext_q = get_ngu_mlp_net(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies, units=units)
    int_q = get_ngu_mlp_net(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies, units=units)
    state_input = tf.keras.layers.Input()
    int_reward_input = tf.keras.layers.Input()
    ext_reward_input = tf.keras.layers.Input()
    a_tm1_input = tf.keras.layers.Input()
    policy_input = tf.keras.layers.Input()

    ext_q_values, ext_hidden_states = ext_q([state_input, a_tm1_input, ext_reward_input, int_reward_input, policy_input])
    int_q_values, int_hidden_states = int_q([state_input, a_tm1_input, ext_reward_input, int_reward_input, policy_input])

    model = tf.keras.Model(inputs=[state_input, a_tm1_input, ext_reward_input, int_reward_input, policy_input],
                           outputs=[ext_q_values, int_q_values, ext_hidden_states, int_hidden_states]) # outputs={"ext_q_values": ext_q_values, "int_q_values": int_q_values}
    return model


class Agent57MlpNet(tf.keras.Model):
    """Agent57 DQN Conv2d network."""

    def __init__(self, state_dim: int, action_dim: int, num_policies: int, units=128, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output linear layer.
            num_policies: the number of mixtures for intrinsic reward scale betas.
        """
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_policies = num_policies  # intrinsic reward scale betas
        self.units = units

        self.ext_q = NguDqnMlpNet(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies, units=units)
        self.int_q = NguDqnMlpNet(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies, units=units)

    def build(self, input_shape): 
        super().build(input_shape) 
        self.ext_q.build(input_shape) 
        self.int_q.build(input_shape)

    def call(self, input_: Agent57NetworkInputs) -> Agent57NetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the NguNetworkInputs object which contains the following attributes:
                s_t: timestep t environment state, shape [B, T, state_shape].
                a_tm1: action taken in s_tm1, shape [B, T].
                ext_r_t: extrinsic reward for state-action pair (s_tm1, a_tm1), shape [B, T].
                int_r_t: intrinsic reward for state s_tm1, shape [B, T].
                policy_index: the index for the pair of intrinsic reward scale beta and discount gamma, shape [B, T].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            Agent57NetworkOutputs object with the following attributes:
                q_values: state-action values.
                ext_hidden_s: hidden state from LSTM layer output, from extrinsic Q network.
                int_hidden_s: hidden state from LSTM layer output, from intrinsic Q network.
        """

        ext_input = NguNetworkInputs(
            s_t=tf.identity(input_.s_t),
            a_tm1=tf.identity(input_.a_tm1),
            ext_r_t=tf.identity(input_.ext_r_t),
            int_r_t=tf.identity(input_.int_r_t),
            policy_index=tf.identity(input_.policy_index),
            hidden_s=input_.ext_hidden_s,
        )

        int_input = NguNetworkInputs(
            s_t=input_.s_t,
            a_tm1=input_.a_tm1,
            ext_r_t=input_.ext_r_t,
            int_r_t=input_.int_r_t,
            policy_index=input_.policy_index,
            hidden_s=input_.int_hidden_s,
        )

        ext_output = self.ext_q(ext_input)
        int_output = self.int_q(int_input)

        return Agent57NetworkOutputs(
            ext_q_values=ext_output.q_values,
            int_q_values=int_output.q_values,
            ext_hidden_s=ext_output.hidden_s,
            int_hidden_s=int_output.hidden_s,
        )

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""

        ext_state = self.ext_q.get_initial_hidden_state(batch_size)
        int_state = self.int_q.get_initial_hidden_state(batch_size)

        return (ext_state, int_state)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_policies": self.num_policies,
            "units": self.units,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =================================================================
# Convolutional Neural Networks
# =================================================================


class DqnConvNet(tf.keras.Model):
    """DQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        super(DqnConvNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [H, W, C], got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])

    def call(self, x: tf.Tensor) -> DqnNetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        x = tf.cast(x, tf.float32) / 255.0
        x = self.body(x)
        q_values = self.value_head(x)  # [batch_size, action_dim]
        return DqnNetworkOutputs(q_values=q_values)

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


class DuelingDqnConvNet(tf.keras.Model):
    """Dueling DQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        super(DuelingDqnConvNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x: tf.Tensor) -> DqnNetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        features = self.body(x)

        advantages = self.advantage_head(features)  # [batch_size, action_dim]
        values = self.value_head(features)  # [batch_size, 1]

        q_values = values + (advantages - tf.reduce_max(advantages, axis=1, keepdims=True))  # [batch_size, action_dim]

        return DqnNetworkOutputs(q_values=q_values)

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


class C51DqnConvNet(tf.keras.Model):
    """C51 DQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, atoms: tf.Tensor, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            atoms: the support for q value distribution, used here to turn Z into Q values
        """
        super(C51DqnConvNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.shape[0]
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim * self.num_atoms)
        ])

    def call(self, x: tf.Tensor) -> C51NetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        x = tf.cast(x, tf.float32) / 255.0
        x = self.body(x)
        x = self.value_head(x)

        q_logits = tf.reshape(x, (-1, self.action_dim, self.num_atoms))  # [batch_size, action_dim, num_atoms]
        q_dist = tf.nn.softmax(q_logits, axis=-1)
        atoms = tf.expand_dims(tf.expand_dims(self.atoms, 0), 0)
        q_values = tf.reduce_sum(q_dist * atoms, axis=-1)  # [batch_size, action_dim]

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "atoms": self.atoms.numpy().tolist()
        }

    @classmethod
    def from_config(cls, config):
        config['atoms'] = tf.convert_to_tensor(config.pop('atoms'))
        return cls(**config)


class RainbowDqnConvNet(tf.keras.Model):
    """Rainbow combines C51, dueling architecture, and noisy net."""

    def __init__(self, state_dim: tuple, action_dim: int, atoms: tf.Tensor, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            atoms: the support for q value distribution, used here to turn Z into Q values
        """
        super(RainbowDqnConvNet, self).__init__(**kwargs)
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.shape[0]

        self.body = common.NatureCnnBackboneNet(state_dim)

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.GaussianNoise,
            common.NoisyDense(512, "elu"),
            common.NoisyDense(action_dim * self.num_atoms)
        ])

        self.value_head = tf.keras.Sequential([
            common.NoisyDense(512, "elu"),
            common.NoisyDense(1 * self.num_atoms)
        ])

    def call(self, x: tf.Tensor) -> C51NetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        x = tf.cast(x, tf.float32) / 255.0
        x = self.body(x)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        advantages = tf.reshape(advantages, (-1, self.action_dim, self.num_atoms))
        values = tf.reshape(values, (-1, 1, self.num_atoms))

        q_logits = values + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))

        q_logits = tf.reshape(q_logits, (-1, self.action_dim, self.num_atoms))  # [batch_size, action_dim, num_atoms]

        q_dist = tf.nn.softmax(q_logits, axis=-1)
        atoms = tf.expand_dims(tf.expand_dims(self.atoms, 0), 0)
        q_values = tf.reduce_sum(q_dist * atoms, axis=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

    def reset_noise(self) -> None:
        """Reset noisy layer"""
        for module in list(chain(*zip(self.advantage_head.layers, self.value_head.layers))):
            if isinstance(module, common.NoisyDense):
                module.reset_noise()

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "atoms": self.atoms.numpy().tolist()
        }

    @classmethod
    def from_config(cls, config):
        config['atoms'] = tf.convert_to_tensor(config.pop('atoms'))
        return cls(**config)


class QRDqnConvNet(tf.keras.Model):
    """Quantile Regression DQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, quantiles: tf.Tensor, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            quantiles: the quantiles for QR DQN
        """
        super(QRDqnConvNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')
        if len(quantiles.shape) != 1:
            raise ValueError(f'Expect quantiles to be a 1D tensor, got {quantiles.shape}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.taus = quantiles
        self.num_taus = quantiles.shape[0]

        self.body = common.NatureCnnBackboneNet(state_dim)

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim * self.num_taus)
        ])

    def call(self, x: tf.Tensor) -> QRDqnNetworkOutputs:
        """Given state, return state-action value for all possible actions"""
        x = tf.cast(x, tf.float32) / 255.0
        x = self.body(x)
        q_dist = self.value_head(x)
        # No softmax as the model is trying to approximate the 'whole' probability distributions
        q_dist = tf.reshape(q_dist, (-1, self.num_taus, self.action_dim))  # [batch_size, num_taus, action_dim]
        q_values = tf.reduce_mean(q_dist, axis=1)

        return QRDqnNetworkOutputs(q_values=q_values, q_dist=q_dist)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "quantiles": self.taus.numpy().tolist()
        }

    @classmethod
    def from_config(cls, config):
        config['quantiles'] = tf.convert_to_tensor(config.pop('quantiles'))
        return cls(**config)


class IqnConvNet(tf.keras.Model):
    """Implicit Quantile Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, latent_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            latent_dim: the cos embedding linear layer input shapes
        """
        super(IqnConvNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')
        if latent_dim < 1:
            raise ValueError(f'Expect latent_dim to be a positive integer, got {latent_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.pis = tf.range(1, self.latent_dim + 1, dtype=tf.float32) * 3.141592653589793  # [latent_dim]

        self.body = common.NatureCnnBackboneNet(state_dim)
        self.embedding_layer = tf.keras.layers.Dense(self.body.output_shape[-1], activation='elu', kernel_initializer="he_normal")

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])

    def sample_taus(self, batch_size: int, num_taus: int) -> tf.Tensor:
        """Returns sampled batch taus."""
        taus = tf.random.uniform((batch_size, num_taus), dtype=tf.float32)
        return taus

    def call(self, x: tf.Tensor, num_taus: int = 64) -> IqnNetworkOutputs:
        """
        Args:
            state: environment state # batch x state_shape
            taus: tau embedding samples # batch x samples

        Returns:
            q_values: # [batch_size, action_dim]
            q_dist: # [batch_size, num_taus, action_dim]
        """
        batch_size = tf.shape(x)[0]

        # Apply ConvDQN to embed state.
        x = tf.cast(x, tf.float32) / 255.0
        features = self.body(x)

        taus = self.sample_taus(batch_size, num_taus)

        # Embed taus with cosine embedding + linear layer.
        # cos(pi * i * tau) for i = 1,...,latents for each batch_element x sample.
        # Broadcast everything to batch x num_taus x latent_dim.
        pis = tf.expand_dims(tf.expand_dims(self.pis, 0), 0)
        tau_embedding = tf.cos(pis * tf.expand_dims(taus, -1))  # [batch_size, num_taus, latent_dim]

        # Merge batch and taus dimension before input to embedding layer.
        tau_embedding = tf.reshape(tau_embedding, (batch_size * num_taus, -1))  # [batch_size x num_taus, latent_dim]
        tau_embedding = self.embedding_layer(tau_embedding)  # [batch_size x num_taus, embedding_layer_output]

        # Reshape/broadcast both embeddings to batch x num_taus x state_dim
        # and multiply together, before applying value head.
        tau_embedding = tf.reshape(tau_embedding, (batch_size, num_taus, -1))
        head_input = tau_embedding * tf.expand_dims(features, 1)  # [batch_size, num_taus, embedding_layer_output]

        # Merge head input dimensions.
        head_input = tf.reshape(head_input, (-1, self.embedding_layer.units))

        # No softmax as the model is trying to approximate the 'whole' probability distributions
        q_dist = self.value_head(head_input)  # [batch_size x num_taus, action_dim]
        q_dist = tf.reshape(q_dist, (batch_size, num_taus, self.action_dim))  # [batch_size, num_taus, action_dim]
        q_values = tf.reduce_mean(q_dist, axis=1)  # [batch_size, action_dim]
        return IqnNetworkOutputs(q_values=q_values, q_dist=q_dist, taus=taus)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DrqnConvNet(tf.keras.Model):
    """DRQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        super(DrqnConvNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.body = common.NatureCnnBackboneNet(state_dim)

        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])

    def call(self, x: tf.Tensor, hidden_s: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the batch of state tensor, shape [B, T, state_shape]
            hidden_s: the initial/last time step hidden state from lstm
        Returns:
            q_values: state-action values
            hidden_s: hidden state from LSTM layer
        """
        # Expect x shape to be [B, T, state_shape]
        assert len(x.shape) == 5
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        x = tf.reshape(x, shape=tf.concat([[T * B], x.shape[2:]], axis=-1))  # Merge batch and time dimension.
        x = tf.cast(x, tf.float32) / 255.0
        x = self.body(x)
        x = tf.reshape(x, (B, T, -1))  # LSTM expect rank 3

        x, hidden_h, hidden_c = self.lstm(x, initial_state=hidden_s)

        x = tf.reshape(x, shape=tf.concat([[T * B], x.shape[2:]], axis=-1))  # Merge batch and time dimension.
        q_values = self.value_head(x)
        q_values = tf.reshape(q_values, (B, T, -1))  # reshape to in the range [B, T, action_dim]
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=(hidden_h, hidden_c))

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""
        return (tf.zeros((batch_size, 256)), tf.zeros((batch_size, 256)))

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


class R2d2DqnConvNet(tf.keras.Model):
    """R2D2 DQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        super(R2d2DqnConvNet, self).__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.body = common.NatureCnnBackboneNet(state_dim)

        core_output_size = self.body.output_shape[-1] + self.action_dim + 1

        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, input_: RnnDqnNetworkInputs) -> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            input_: the RnnDqnNetworkInputs object which contains the following attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tm1: action taken in t-1 timestep, shape [T, B].
                r_t: reward for state-action pair (s_tm1, a_tm1), shape [T, B].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            RnnDqnNetworkOutputs object with the following attributes:
                q_values: state-action values.
                hidden_s: hidden state from LSTM layer output, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        hidden_s = input_.hidden_s

        T = tf.shape(s_t)[0]
        B = tf.shape(s_t)[1]

        x = tf.reshape(s_t, shape=tf.concat([[T * B], x.shape[2:]], axis=-1))  # Merge batch and time dimension.
        x = tf.cast(x, tf.float32) / 255.0
        x = self.body(x)
        x = tf.reshape(x, (T * B, -1))

        one_hot_a_tm1 = tf.one_hot(tf.reshape(a_tm1, (T * B,)), self.action_dim, dtype=tf.float32)

        reward = tf.reshape(r_t, (T * B, 1))
        core_input = tf.concat([x, reward, one_hot_a_tm1], axis=-1)
        core_input = tf.reshape(core_input, (T, B, -1))  # LSTM expect rank 3 tensor.

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)

        x, hidden_h, hidden_c = self.lstm(core_input, initial_state=hidden_s)

        x = tf.reshape(x, shape=tf.concat([[T * B], x.shape[2:]], axis=-1))  # Merge batch and time dimension.
        advantages = self.advantage_head(x)  # [T*B, action_dim]
        values = self.value_head(x)  # [T*B, 1]

        q_values = values + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        q_values = tf.reshape(q_values, (T, B, -1))  # reshape to in the range [T, B, action_dim]
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=(hidden_h, hidden_c))

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""
        return (tf.zeros((batch_size, 512)), tf.zeros((batch_size, 512)))

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


class NguDqnConvNet(tf.keras.Model):
    """NGU DQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, num_policies: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output linear layer.
            num_policies: the number of mixtures for intrinsic reward scale betas.
        """
        super().__init__(**kwargs)
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [H, W, C], got {state_dim}')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_policies = num_policies  # intrinsic reward scale betas

        self.body = common.NatureCnnBackboneNet(state_dim)

        # Core input includes:
        # feature representation output size
        # one-hot of intrinsic reward scale beta
        # one-hot of last action
        # last intrinsic reward
        # last extrinsic reward
        # core_output_size = self.body.output_shape[-1] + self.num_policies + self.action_dim + 1 + 1

        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)

        self.advantage_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim)
        ])

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='elu', kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, input_: NguNetworkInputs) -> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the NguNetworkInputs object which contains the following attributes:
                s_t: timestep t environment state, shape [B, T, state_shape].
                a_tm1: action taken in s_tm1, shape [B, T].
                ext_r_t: extrinsic reward for state-action pair (s_tm1, a_tm1), shape [B, T].
                int_r_t: intrinsic reward for state s_tm1, shape [B, T].
                policy_index: the index for the pair of intrinsic reward scale beta and discount gamma, shape [B, T].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            RnnDqnNetworkOutputs object with the following attributes:
                q_values: state-action values.
                hidden_s: hidden state from LSTM layer output.
        """
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        ext_r_t = input_.ext_r_t
        int_r_t = input_.int_r_t
        policy_index = input_.policy_index
        hidden_s = input_.hidden_s

        T = s_t.shape[0]
        B = s_t.shape[1]

        x = tf.reshape(s_t, shape=[-1, *s_t.shape[2:]])  # Merge batch and time dimension.
        x = tf.cast(x, tf.float32) / 255.0
        x = self.body(x)
        x = tf.reshape(x, [T * B, -1])

        one_hot_beta = tf.one_hot(tf.reshape(policy_index, (T * B,)), self.num_policies, dtype=tf.float32)
        one_hot_a_tm1 = tf.one_hot(tf.reshape(a_tm1, (T * B,)), self.action_dim, dtype=tf.float32)

        int_reward = tf.reshape(int_r_t, (T * B, 1))
        ext_reward = tf.reshape(ext_r_t, (T * B, 1))

        core_input = tf.concat([x, ext_reward, one_hot_a_tm1, int_reward, one_hot_beta], axis=-1)
        core_input = tf.reshape(core_input, (T, B, -1))  # LSTM expect rank 3 tensor.

        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)

        x, hidden_h, hidden_c = self.lstm(core_input, initial_state=hidden_s)

        x = tf.reshape(x, shape=[-1, *x.shape[2:]]) # Merge batch and time dimension.
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        q_values = values + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        q_values = tf.reshape(q_values, (T, B, -1))  # reshape to in the range [T, B, action_dim]
        return RnnDqnNetworkOutputs(q_values=q_values, hidden_s=(hidden_h, hidden_c))

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""
        return (tf.zeros((batch_size, 512)), tf.zeros((batch_size, 512)))

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_policies": self.num_policies
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Agent57Conv2dNet(tf.keras.Model):
    """Agent57 DQN Conv2d network."""

    def __init__(self, state_dim: tuple, action_dim: int, num_policies: int, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output linear layer.
            num_policies: the number of mixtures for intrinsic reward scale betas.
        """
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_policies = num_policies  # intrinsic reward scale betas

        self.ext_q = NguDqnConvNet(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies)
        self.int_q = NguDqnConvNet(state_dim=state_dim, action_dim=action_dim, num_policies=num_policies)

    def call(self, input_: Agent57NetworkInputs) -> Agent57NetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the NguNetworkInputs object which contains the following attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tm1: action taken in s_tm1, shape [T, B].
                ext_r_t: extrinsic reward for state-action pair (s_tm1, a_tm1), shape [T, B].
                int_r_t: intrinsic reward for state s_tm1, shape [T, B].
                policy_index: the index for the pair of intrinsic reward scale beta and discount gamma, shape [T, B].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tensors each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            Agent57NetworkOutputs object with the following attributes:
                q_values: state-action values.
                ext_hidden_s: hidden state from LSTM layer output, from extrinsic Q network.
                int_hidden_s: hidden state from LSTM layer output, from intrinsic Q network.
        """

        ext_input = NguNetworkInputs(
            s_t=tf.identity(input_.s_t),
            a_tm1=tf.identity(input_.a_tm1),
            ext_r_t=tf.identity(input_.ext_r_t),
            int_r_t=tf.identity(input_.int_r_t),
            policy_index=tf.identity(input_.policy_index),
            hidden_s=input_.ext_hidden_s,
        )

        int_input = NguNetworkInputs(
            s_t=input_.s_t,
            a_tm1=input_.a_tm1,
            ext_r_t=input_.ext_r_t,
            int_r_t=input_.int_r_t,
            policy_index=input_.policy_index,
            hidden_s=input_.int_hidden_s,
        )

        ext_output = self.ext_q(ext_input)
        int_output = self.int_q(int_input)

        return Agent57NetworkOutputs(
            ext_q_values=ext_output.q_values,
            int_q_values=int_output.q_values,
            ext_hidden_s=ext_output.hidden_s,
            int_hidden_s=int_output.hidden_s,
        )

    def get_initial_hidden_state(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get initial LSTM hidden state, which is all zeros,
        should call at the beginning of new episode, or every training batch"""

        ext_state = self.ext_q.get_initial_hidden_state(batch_size)
        int_state = self.int_q.get_initial_hidden_state(batch_size)

        return (ext_state, int_state)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_policies": self.num_policies
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

