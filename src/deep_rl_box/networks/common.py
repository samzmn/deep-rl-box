"""Common components for network."""
import tensorflow as tf


def clone_network(network: tf.keras.Model, clone_weights: bool = False, input_shape=None) -> tf.keras.Model:
        """clones subclassed network's architectures (not weights) from its configuration and returns it"""
        cloned = network.__class__.from_config(network.get_config())
        if clone_weights and input_shape is not None:
            network.build(input_shape=input_shape)
            cloned.set_weights(network.get_weights())
        return cloned

def disable_trainability(network: tf.keras.Model) -> None:
    for layer in network.layers:
        layer.trainable = False
    network.trainable = False


class NatureCnnBackboneNet(tf.keras.Model):
    """DQN Nature paper conv2d layers backbone, returns feature representation vector."""

    def __init__(self, state_dim: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state_dim = state_dim

        self.net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=state_dim),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Flatten(),
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Given raw state images, returns feature representation vector"""
        return self.net(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResNetBlock(tf.keras.Model):
    """Basic 3x3 residual block."""

    def __init__(self, num_planes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_planes = num_planes
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=num_planes, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=num_planes, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = tf.nn.relu(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_planes": self.num_planes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NoisyDense000(tf.keras.layers.Layer):
    """Factorized NoisyDense layer with bias.

    Code adapted from:
    https://github.com/Kaixhin/Rainbow/blob/master/model.py
    """

    def __init__(self, units, std_init=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.std_init = std_init

    def build(self, input_shape):
        super().build(input_shape)
        self.in_features = input_shape[-1]
        mu_range = 1 / tf.sqrt(tf.cast(self.in_features, dtype=tf.float32))
        self.weight_mu = self.add_weight(name='weight_mu',
                                         shape=(self.in_features, self.units),
                                         initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
                                         trainable=True)
        self.weight_sigma = self.add_weight(name='weight_sigma',
                                            shape=(self.in_features, self.units),
                                            initializer=tf.keras.initializers.Constant(self.std_init * mu_range),
                                            trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=tf.keras.initializers.RandomUniform(-mu_range, mu_range),
                                       trainable=True)
        self.bias_sigma = self.add_weight(name='bias_sigma',
                                          shape=(self.units,),
                                          initializer=tf.keras.initializers.Constant(self.std_init / tf.sqrt(tf.cast(self.units, tf.float32))),
                                          trainable=True)
        self.weight_epsilon = tf.Variable(tf.zeros((self.in_features, self.units)), trainable=False)
        self.bias_epsilon = tf.Variable(tf.zeros((self.units,)), trainable=False)
        self.reset_noise()
        self.built = True

    def _scale_noise(self, size):
        x = tf.random.normal(size)
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def reset_noise(self):
        epsilon_in = self._scale_noise([self.in_features])
        epsilon_out = self._scale_noise([self.units])
        self.weight_epsilon.assign(tf.tensordot(epsilon_in, epsilon_out, axes=0))
        self.bias_epsilon.assign(epsilon_out)

    def call(self, inputs, training=None):
        if self.trainable or training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return tf.matmul(inputs, weight) + bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "std_init": self.std_init,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

import numpy as np
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, sigma_init=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        super().build(input_shape)
        self.mu_weight = self.add_weight(
            name="mu_weight",
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True)
        
        self.sigma_weight = self.add_weight(
            name="sigma_weight",
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.Constant(value=self.sigma_init),
            trainable=True)
        
        self.mu_bias = self.add_weight(
            name="mu_bias",
            shape=(self.units,),
            initializer='zeros',
            trainable=True)
        
        self.sigma_bias = self.add_weight(
            name="sigma_bias",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(value=self.sigma_init),
            trainable=True)
        
        self.built = True

    def call(self, inputs, training=None):
        if training:
            # epsilon_weight = np.random.normal(loc=0.0, scale=1.0, size=self.mu_weight.shape)
            # epsilon_bias = np.random.normal(loc=0.0, scale=1.0, size=self.mu_bias.shape)
            epsilon_weight = tf.random.normal(shape=self.mu_weight.shape, mean=0.0, stddev=1.0)
            epsilon_bias = tf.random.normal(shape=self.mu_bias.shape, mean=0.0, stddev=1.0)
            weights = self.mu_weight + self.sigma_weight * epsilon_weight
            bias = self.mu_bias + self.sigma_bias * epsilon_bias
        else:
            weights = self.mu_weight
            bias = self.mu_bias 
        
        out = tf.matmul(inputs, weights) + bias
        return self.activation(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sigma_init": self.sigma_init,
            "activation": tf.keras.activations.deserialize(self.activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

