"""Utility module."""

from typing import Iterable

import numpy as np


def split_indices_into_bins(bin_size: int, max_indices: int, min_indices: int = 0, shuffle: bool = False) -> Iterable[int]:
    """Split indices to small bins."""

    bin_size = int(bin_size)
    max_indices = int(max_indices)
    min_indices = int(min_indices)

    if max_indices < bin_size:
        raise ValueError(f'Expect max_indices to be greater than bin_size, got {max_indices} and {bin_size}')

    # Split indices into 'bins' with bin_size.
    indices = np.arange(min_indices, max_indices)

    if shuffle:
        np.random.shuffle(indices)

    indices_list = []
    for i in range(0, len(indices), bin_size):
        indices_list.append(indices[i : i + bin_size])  # noqa: E203

    # Make sure the last one has the same 'bin_size'.
    if len(indices_list[-1]) != bin_size:
        indices_list[-1] = indices[-bin_size:]

    return indices_list
