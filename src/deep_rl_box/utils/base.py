"""Utilities for Reinforcement Learning ops."""
from typing import NamedTuple, Optional, Tuple, Union
import tensorflow as tf

class LossOutput(NamedTuple):
    loss: tf.Tensor
    extra: Optional[NamedTuple]

def assert_rank_and_dtype(tensor: tf.Tensor, rank: Union[int, Tuple[int]], dtype: Union[tf.dtypes.DType, Tuple[tf.dtypes.DType]]):
    """Asserts that the tensor has the correct rank and dtype.

    Args:
        tensor: The tensor to check.
        rank: A scalar or tuple of scalars specifying the acceptable ranks.
        dtype: A single TensorFlow dtype or tuple of acceptable dtypes.

    Raises:
        ValueError: If the tensor fails the rank and dtype checks.
    """
    assert_rank(tensor, rank)
    assert_dtype(tensor, dtype)

def assert_rank(tensor: tf.Tensor, rank: Union[int, Tuple[int]]) -> None:
    """Asserts that the tensor has the correct rank.

    Args:
        tensor: The tensor to check.
        rank: A scalar or tuple of scalars specifying the acceptable ranks.

    Raises:
        ValueError: If the tensor fails the rank checks.
    """
    if not isinstance(tensor, tf.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid tf.Tensor.')
    if isinstance(rank, int):
        rank = (rank,)
    if len(tensor.shape) not in rank:
        f'Error in rank and/or compatibility check. The input tensor should be rank {rank} torch.Tensor, got {tensor.shape}.'

def assert_dtype(tensor: tf.Tensor, dtype: Union[tf.dtypes.DType, Tuple[tf.dtypes.DType]]) -> None:
    """Asserts that the tensor has the correct dtype.

    Args:
        tensor: The tensor to check.
        dtype: A single TensorFlow dtype or tuple of acceptable dtypes.

    Raises:
        ValueError: If the tensor fails the dtype checks.
    """
    if not isinstance(tensor, tf.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid tf.Tensor.')
    if isinstance(dtype, tf.dtypes.DType):
        dtype = (dtype,)
    if tensor.dtype not in dtype:
        raise ValueError(f'Error in rank and/or compatibility check. The input tensor should be {dtype}, got {tensor.dtype}.')

def assert_batch_dimension(tensor: tf.Tensor, batch_size: int, dim: int = 0) -> None:
    """Asserts that the tensor has the correct batch dimension.

    Args:
        tensor: The tensor to check.
        batch_size: The expected batch size.
        dim: The dimension to check for the batch size.

    Raises:
        ValueError: If the tensor fails the batch dimension check.
    """
    if not isinstance(tensor, tf.Tensor):
        raise ValueError('Error in rank and/or compatibility check. The input tensor should be a valid tf.Tensor.')
    if tensor.shape[dim] != batch_size:
        raise ValueError(f'Error in rank and/or compatibility check. The input tensor should have {batch_size} entry on batch dimension {dim}, got {tensor.shape}.')

def batched_index(values: tf.Tensor, indices: tf.Tensor, dim: int = -1, keepdims: bool = False) -> tf.Tensor:
    """Performs batched indexing on the given tensor.

    Args:
        values: Tensor of shape `[B, num_values]` or `[T, B, num_values]`.
        indices: Tensor of shape `[B]` or `[T, B]` containing indices.
        dim: The dimension to perform the selection on.
        keepdims: If `True`, retains reduced dimensions with length 1.

    Returns:
        Tensor containing values for the given indices.

    Raises:
        ValueError: If the tensor shapes are incompatible.
    """
    assert_rank(values, (2, 3))
    assert_rank_and_dtype(indices, (1, 2), tf.int64)
    assert_batch_dimension(indices, values.shape[0], 0)
    if len(indices.shape) == 2:
        assert_batch_dimension(indices, values.shape[1], 1)

    one_hot_indices = tf.one_hot(indices, values.shape[dim], dtype=values.dtype)
    if len(values.shape) == 3 and len(one_hot_indices.shape) == 2:
        one_hot_indices = tf.expand_dims(one_hot_indices, 1)
    return tf.reduce_sum(values * one_hot_indices, axis=dim, keepdims=keepdims)
