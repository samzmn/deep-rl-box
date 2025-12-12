"""Networks for curiosity-driven-exploration."""

import numpy as np
import tensorflow as tf
from typing import NamedTuple, Tuple

# pylint: disable=import-error
from deep_rl_box.utils import base
from deep_rl_box.networks import common


class IcmNetworkOutput(NamedTuple):
    """ICM module"""

    pi_logits: tf.Tensor
    features: tf.Tensor
    pred_features: tf.Tensor


class IcmMlpNet(tf.keras.Model):
    """ICM(Intrinsic Curiosity Module) module of curiosity driven exploration for Mlp networks.

    From the paper "Curiosity-driven Exploration by Self-supervised Prediction"
    https://arxiv.org/abs/1705.05363.
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output liner layer.
        """
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim

        feature_vector_size = 128

        # Feature representations
        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(feature_vector_size, activation="relu", kernel_initializer="he_normal"),
        ])

        # Forward model, predict feature vector of s_t from s_tm1 and a_t
        self.forward_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(feature_vector_size + action_dim,)),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(feature_vector_size, activation="relu", kernel_initializer="he_normal"),
        ])

        # Inverse model, predict a_tm1 from feature vectors of s_tm1, s_t
        self.inverse_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(feature_vector_size * 2,)),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

    def call(self, s_tm1: tf.Tensor, a_tm1: tf.Tensor, s_t: tf.Tensor) -> IcmNetworkOutput:
        """Given raw state s_tm1, s_t, and action a_tm1,
        call forward model and inverse model to predict a_tm1 and feature vector of s_t"""
        base.assert_rank(s_tm1, 2)
        base.assert_rank(s_t, 2)
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = tf.one_hot(a_tm1, self.action_dim, dtype=tf.float32)

        # Get feature vectors of s_tm1 and s_t
        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        # Predict feature vector of s_t
        forward_input = tf.concat([features_tm1, a_tm1_onehot], axis=-1)
        pred_features_t = self.forward_net(forward_input)

        # Predict actions a_tm1 from feature vectors s_tm1 and s_t
        inverse_input = tf.concat([features_tm1, features_t], axis=-1)
        pi_logits_a_tm1 = self.inverse_net(inverse_input)  # Returns logits not probability distribution

        return IcmNetworkOutput(pi_logits=pi_logits_a_tm1, pred_features=pred_features_t, features=features_t)


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
    

class IcmNatureConvNet(tf.keras.Model):
    """ICM module of curiosity driven exploration for Conv networks.

    From the paper "Curiosity-driven Exploration by Self-supervised Prediction"
    https://arxiv.org/abs/1705.05363.
    """

    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, **kwargs) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output liner layer.
        """
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Compute the output shape of final conv2d layer
        h, w, c = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 2, 1)
        conv2d_out_size = 32 * h * w  # output size 288

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[h, w, c]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Flatten(),
        ])

        # Forward model, predict feature vector of s_t from s_tm1 and a_t
        self.forward_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[conv2d_out_size + self.action_dim]),
            tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(conv2d_out_size, activation="relu", kernel_initializer="he_normal"),
        ])

        # Inverse model, predict a_tm1 from feature vectors of s_tm1, s_t
        self.inverse_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[conv2d_out_size * 2]),
            tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

    def call(self, s_tm1: tf.Tensor, a_tm1: tf.Tensor, s_t: tf.Tensor) -> IcmNetworkOutput:
        """Given raw state s_tm1, s_t, and action a_tm1,
        call forward model and inverse model to predict a_tm1 and feature vector of s_t"""
        base.assert_rank(s_tm1, (2, 4))
        base.assert_rank(s_t, (2, 4))
        base.assert_rank(a_tm1, 1)

        a_tm1_onehot = tf.one_hot(a_tm1, self.action_dim, dtype=tf.float32)

        # Get feature vectors of s_tm1 and s_t
        s_tm1 = tf.cast(s_tm1, tf.float32) / 255.0
        s_t = tf.cast(s_t, tf.float32) / 255.0
        features_tm1 = self.body(s_tm1)
        features_t = self.body(s_t)

        # Predict feature vector of s_t
        forward_input = tf.concat([features_tm1, a_tm1_onehot], axis=-1)
        pred_features_t = self.forward_net(forward_input)

        # Predict actions a_tm1 from feature vectors s_tm1 and s_t
        inverse_input = tf.concat([features_tm1, features_t], axis=-1)
        pi_logits_a_tm1 = self.inverse_net(inverse_input)  # Returns logits not probability distribution

        return IcmNetworkOutput(pi_logits=pi_logits_a_tm1, pred_features=pred_features_t, features=features_t)

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
    

class RndConvNet(tf.keras.Model):
    """RND Conv2D network.

    From the paper "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894
    """

    def __init__(self, state_dim: Tuple[int, int, int], is_target: bool = False, latent_dim: int = 256, **kwargs) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            is_target: if True, use one single linear layer at the head, default False.
            latent_dim: the embedding latent dimension, default 256.
        """
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.is_target = is_target
        self.latent_dim = latent_dim

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=state_dim),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="leaky_relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="leaky_relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="leaky_relu", kernel_initializer="he_normal"),
            tf.keras.layers.Flatten(),
        ])

        if is_target:
            self.head = tf.keras.layers.Dense(latent_dim)
        else:
            self.head = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal"),
                tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal"),
                tf.keras.layers.Dense(latent_dim),
            ])

        # Initialize weights.
        for layer in self.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                layer.kernel_initializer = tf.keras.initializers.Orthogonal(np.sqrt(2))
                layer.bias_initializer = tf.keras.initializers.Zeros()

        if is_target:
            for param in self.trainable_variables:
                param.trainable = False

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Given raw state x, returns the feature embedding."""
        # RND normalizes state using a running mean and std instead of divide by 255.
        x = self.body(x)
        return self.head(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "state_dim": self.state_dim,
            "is_target": self.is_target,
            "latent_dim": self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NguEmbeddingConvNet(tf.keras.Model):
    """Conv2D Embedding networks for NGU.

    From the paper "Never Give Up: Learning Directed Exploration Strategies"
    https://arxiv.org/abs/2002.06038
    """

    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int, embed_size: int = 32, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output liner layer.
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.embed_size = embed_size

        self.net = common.NatureCnnBackboneNet(state_dim)

        self.fc = tf.keras.layers.Dense(self.embed_size)

        # *2 because the input to inverse head is two embeddings [s_t, s_tp1]
        self.inverse_head = tf.keras.Sequential([
            tf.keras.Input(shape=[2 * self.embed_size]),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Given state x, return the embedding."""
        x = tf.cast(x, tf.float32) / 255.0
        x = self.net(x)

        return tf.nn.relu(self.fc(x))

    def inverse_prediction(self, x: tf.Tensor) -> tf.Tensor:
        """Given combined embedding features of (s_tm1 + s_t), returns the raw logits of predicted action a_tm1."""
        pi_logits = self.inverse_head(x)  # [batch_size, action_dim]
        return pi_logits

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


class NGURndConvNet(tf.keras.Model):
    """RND Conv2D network for NGU agent.

    From the paper "Never Give Up: Learning Directed Exploration Strategies"
    https://arxiv.org/abs/2002.06038
    """

    def __init__(self, state_dim: Tuple[int, int, int], latent_dim: int = 128, is_target: bool = False, **kwargs) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            latent_dim: the latent vector dimension, default 128.
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.is_target = is_target

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=state_dim),
            tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Flatten(),
        ])

        self.head = tf.keras.layers.Dense(latent_dim)

        if is_target:
            self.trainable = False

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Given raw state x, returns the latent feature vector."""
        # RND normalizes state using a running mean and std instead of divide by 255.
        x = self.body(x)
        return self.head(x)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "latent_dim": self.latent_dim,
            "is_target": self.is_target
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class NguEmbeddingMlpNet(tf.keras.Model):
    """Embedding networks for NGU.
    """

    def __init__(self, state_dim: int, action_dim: int, embed_size: int = 32, **kwargs):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output liner layer.
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.embed_size = embed_size

        self.net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(self.embed_size, activation="relu", kernel_initializer="he_normal")
        ])

        # *2 because the input to inverse head is two embeddings [s_t, s_tp1]
        self.inverse_head = tf.keras.Sequential([
            tf.keras.Input(shape=[2 * self.embed_size]),
            tf.keras.layers.Dense(self.embed_size, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(action_dim),
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Given state x, return the embedding."""
        return self.net(x)

    def inverse_prediction(self, x: tf.Tensor) -> tf.Tensor:
        """Given combined embedding features of (s_tm1 + s_t), returns the raw logits of predicted action a_tm1."""
        return self.inverse_head(x)  # [batch_size, action_dim]

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "embed_size": self.embed_size,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NGURndMlpNet(tf.keras.Model):
    """RND network for NGU agent.
    """

    def __init__(self, state_dim: int, latent_dim: int = 128, is_target: bool = False, **kwargs) -> None:
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            latent_dim: the latent vector dimension, default 128.
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.is_target = is_target

        self.body = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[state_dim]),
            tf.keras.layers.Dense(latent_dim//2, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(latent_dim, kernel_initializer="he_normal"),
        ])

        if is_target:
            self.trainable = False
            for layer in self.body.layers:
                layer.trainable = False

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Given raw state x, returns the latent feature vector."""
        # RND normalizes state using a running mean and std instead of divide by 255.
        return self.body(x)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "state_dim": self.state_dim,
            "latent_dim": self.latent_dim,
            "is_target": self.is_target
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
