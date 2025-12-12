"""Implementing functions and class for curiosity driven exploration."""

from typing import NamedTuple, Dict
import numpy as np
import tensorflow as tf

from deep_rl_box.utils import base
import deep_rl_box.utils.normalizer as normalizer


class KNNQueryResult(NamedTuple):
    neighbors: tf.Tensor
    neighbor_indices: tf.Tensor
    neighbor_distances: tf.Tensor


def knn_query(current: tf.Tensor, memory: tf.Tensor, num_neighbors: int) -> KNNQueryResult:
    """Finds closest neighbors and their squared euclidean distances.

    Args:
      current: tensor of current embedding, shape [embedding_size].
      memory: tensor of previous embedded data, shape [m, embedding_size],
        where m is the number of previous embeddings.
      num_neighbors: number of neighbors to find.

    Returns:
      KNNQueryResult with (all sorted by squared euclidean distance):
        - neighbors, shape [num_neighbors, feature size].
        - neighbor_indices, shape [num_neighbors].
        - neighbor_distances, shape [num_neighbors].
    """
    base.assert_rank_and_dtype(current, 1, tf.float32)
    base.assert_rank_and_dtype(memory, 2, tf.float32)
    base.assert_batch_dimension(current, memory.shape[-1], -1)

    assert memory.shape[0] >= num_neighbors

    distances = tf.norm(tf.expand_dims(current, 0) - memory, axis=1, ord='euclidean')**2
    distances = -distances
    distances, indices = tf.math.top_k(distances, k=num_neighbors)
    
    neighbors = tf.gather(memory, indices)
    return KNNQueryResult(neighbors=neighbors, neighbor_indices=indices, neighbor_distances=distances)


class EpisodicBonusModule:
    """Episodic memory for calculate intrinsic bonus, used in NGU and Agent57."""

    def __init__(
        self,
        embedding_network: tf.keras.Model,
        capacity: int,
        num_neighbors: int,
        kernel_epsilon: float = 0.0001,
        cluster_distance: float = 0.008,
        max_similarity: float = 8.0,
        c_constant: float = 0.001,
    ) -> None:
        self._embedding_network = embedding_network

        # Initialize memory tensor
        self._memory = np.empty((capacity, embedding_network.embed_size), dtype=np.float32)

        # Initialize mask
        self._mask = np.zeros(capacity, dtype=np.bool_)
        
        self._capacity = capacity
        self._counter = 0

        # Compute the running mean dₘ².
        self._cdist_normalizer = normalizer.TensorFlowRunningMeanStd(shape=(1,))

        self._num_neighbors = num_neighbors
        self._kernel_epsilon = kernel_epsilon
        self._cluster_distance = cluster_distance
        self._max_similarity = max_similarity
        self._c_constant = c_constant

    def _add_to_memory(self, embedding: tf.Tensor) -> None:
        # Insert new embedding
        idx = self._counter % self._capacity
        self._memory[idx] = embedding
        self._mask[idx] = True
        self._counter += 1

    def compute_bonus(self, s_t: tf.Tensor) -> float:
        return np.array(self._compute_bonus(s_t)).item()
    
    def _compute_bonus(self, s_t: tf.Tensor) -> tf.Tensor:
        """Compute episodic intrinsic bonus for given state."""
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)

        embedding = self._embedding_network(s_t)
        embedding = embedding[0]

        # Make a copy of mask because we don't want to use the current embedding when compute the distance
        prev_mask = np.copy(self._mask)
        
        self._add_to_memory(embedding)

        if self._counter <= self._num_neighbors:
            return 0.0

        memory = tf.constant(self._memory[prev_mask])
        knn_query_result = knn_query(embedding, memory, self._num_neighbors)

        # neighbor_distances from knn_query is the squared Euclidean distances.
        nn_distances_sq = knn_query_result.neighbor_distances

        # Update the running mean dₘ².
        self._cdist_normalizer.update_single(nn_distances_sq)

        # Normalize distances with running mean dₘ².
        distance_rate = nn_distances_sq / (self._cdist_normalizer.mean + 1e-8)

        # The distance rate becomes 0 if already small: r <- max(r-ξ, 0).
        distance_rate = tf.maximum((distance_rate - self._cluster_distance), 0.0)

        # Compute the Kernel value K(xₖ, x) = ε/(rate + ε).
        kernel_output = self._kernel_epsilon / (distance_rate + self._kernel_epsilon)

        # Compute the similarity for the embedding x:
        # s = √(Σ_{xₖ ∈ Nₖ} K(xₖ, x)) + c
        similarity = tf.sqrt(tf.reduce_sum(kernel_output)) + self._c_constant

        if tf.math.is_nan(similarity):
            return 0.0

        # Compute the intrinsic reward:
        # r = 1 / s.
        if similarity > self._max_similarity:
            return 0.0

        return (1 / similarity)

    def reset(self):
        self._mask = np.zeros(self._capacity, dtype=np.bool_)  # Initialize mask
        self._counter = 0

    def update_embedding_network(self, weights) -> None:
        """Update embedding network."""
        self._embedding_network.set_weights(weights)


class RndLifeLongBonusModule:
    """RND lifelong intrinsic bonus module, used in NGU and Agent57."""

    def __init__(self, target_network: tf.keras.Model, predictor_network: tf.keras.Model, discount: float, Observation_shape=(84, 84, 1)) -> None:
        self._target_network = target_network
        self._predictor_network = predictor_network
        self._discount = discount

        # RND module observation and lifeline intrinsic reward normalizers
        self._int_reward_normalizer = normalizer.RunningMeanStd(shape=(1,))
        self._rnd_obs_normalizer = normalizer.TensorFlowRunningMeanStd(shape=Observation_shape)

    def _normalize_rnd_obs(self, rnd_obs) -> tf.Tensor:
        rnd_obs = tf.convert_to_tensor(rnd_obs, dtype=tf.float32)

        normed_obs = self._rnd_obs_normalizer.normalize(rnd_obs)

        normed_obs = tf.clip_by_value(normed_obs, -5, 5)

        self._rnd_obs_normalizer.update_single(rnd_obs)

        return normed_obs

    def _normalize_int_rewards(self, int_rewards) -> tf.Tensor:
        """Compute returns then normalize the intrinsic reward based on these returns"""

        self._int_reward_normalizer.update_single(int_rewards)

        # normed_int_rewards = int_rewards / tf.sqrt(self._int_reward_normalizer.var + 1e-8)
        normed_int_rewards = self._int_reward_normalizer.normalize(int_rewards)

        return normed_int_rewards

    def compute_bonus(self, s_t: tf.Tensor) -> float:
        """Compute lifelong bonus for a given state."""
        base.assert_rank_and_dtype(s_t, (2, 4), tf.float32)

        normed_int_r_t = self._compute_bonus(s_t)

        return np.array(normed_int_r_t).item()

    def _compute_bonus(self, s_t: tf.Tensor) -> tf.Tensor:
        """Compute lifelong bonus for a given state."""
        normed_s_t = self._normalize_rnd_obs(s_t)

        pred = self._predictor_network(normed_s_t)
        target = self._target_network(normed_s_t)

        int_r_t = tf.reduce_mean(tf.square(pred - target), axis=1)

        # Normalize intrinsic reward
        normed_int_r_t = self._normalize_int_rewards(int_r_t)

        return normed_int_r_t

    def update_predictor_network(self, weights) -> None:
        """Update RND predictor network."""
        self._predictor_network.set_weights(weights)
