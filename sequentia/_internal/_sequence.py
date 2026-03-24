# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Core utilities for sequence data handling with sklearn compatibility."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np

from sequentia._internal._typing import Array, IntArray

__all__ = [
    "SequenceData",
    "split_sequences",
    "extract_sequences",
    "concatenate_sequences",
]


@dataclass
class SequenceData:
    """Container for sequence data with lazy evaluation support.

    This class provides a memory-efficient way to handle sequence data
    without unnecessary copying or stacking operations.
    """

    X: Array
    lengths: IntArray
    y: Array | None = None

    def __post_init__(self) -> None:
        self._idxs = get_sequence_indices(self.lengths)
        self._n_sequences = len(self.lengths)

    @property
    def n_sequences(self) -> int:
        return self._n_sequences

    @property
    def indices(self) -> IntArray:
        return self._idxs

    def get_sequence(self, i: int) -> Array:
        start, end = self._idxs[i]
        return self.X[start:end]

    def get_sequences(self, indices: IntArray | slice) -> list[Array]:
        idxs = self._idxs[indices]
        return [self.X[start:end] for start, end in idxs]

    def get_labels(self, indices: IntArray | slice) -> Array | None:
        if self.y is None:
            return None
        return self.y[indices]

    def get_lengths(self, indices: IntArray | slice) -> IntArray:
        return self.lengths[indices]

    def subset(self, indices: IntArray | slice) -> SequenceData:
        idxs = self._idxs[indices]
        X_subset = np.concatenate([self.X[start:end] for start, end in idxs])
        lengths_subset = self.lengths[indices]
        y_subset = self.y[indices] if self.y is not None else None
        return SequenceData(X_subset, lengths_subset, y_subset)

    def subset_view(self, indices: IntArray | slice) -> SequenceDataView:
        return SequenceDataView(self, indices)


class SequenceDataView:
    """Lazy view of a subset of sequence data.

    Provides efficient access to sequence subsets without copying data.
    """

    def __init__(
        self, data: SequenceData, indices: IntArray | slice
    ) -> None:
        self._data = data
        self._indices = indices
        self._idxs = data.indices[indices]
        self._lengths = data.lengths[indices]
        self._y = data.y[indices] if data.y is not None else None

    @property
    def lengths(self) -> IntArray:
        return self._lengths

    @property
    def y(self) -> Array | None:
        return self._y

    @property
    def n_sequences(self) -> int:
        return len(self._lengths)

    def materialize(self) -> tuple[Array, IntArray, Array | None]:
        X_concat = np.concatenate(
            [self._data.X[start:end] for start, end in self._idxs]
        )
        return X_concat, self._lengths, self._y


def get_sequence_indices(lengths: IntArray) -> IntArray:
    """Compute start and end indices for each sequence."""
    ends = lengths.cumsum()
    starts = np.zeros_like(ends)
    starts[1:] = ends[:-1]
    return np.column_stack([starts, ends])


def split_sequences(
    X: Array,
    lengths: IntArray,
    y: Array | None,
    train_indices: IntArray,
    test_indices: IntArray,
) -> tuple[tuple[Array, IntArray, Array | None], tuple[Array, IntArray, Array | None]]:
    """Split sequences into train and test sets efficiently.

    Parameters
    ----------
    X : Array
        Concatenated observation sequences.
    lengths : IntArray
        Lengths of each sequence.
    y : Array | None
        Labels for each sequence.
    train_indices : IntArray
        Indices of sequences for training.
    test_indices : IntArray
        Indices of sequences for testing.

    Returns
    -------
    tuple
        ((X_train, lengths_train, y_train), (X_test, lengths_test, y_test))
    """
    idxs = get_sequence_indices(lengths)

    def extract(indices: IntArray) -> tuple[Array, IntArray, Array | None]:
        seq_idxs = idxs[indices]
        X_subset = np.concatenate([X[start:end] for start, end in seq_idxs])
        lengths_subset = lengths[indices]
        y_subset = y[indices] if y is not None else None
        return X_subset, lengths_subset, y_subset

    return extract(train_indices), extract(test_indices)


def extract_sequences(
    X: Array,
    lengths: IntArray,
    indices: IntArray,
) -> tuple[Array, IntArray]:
    """Extract a subset of sequences by indices.

    Parameters
    ----------
    X : Array
        Concatenated observation sequences.
    lengths : IntArray
        Lengths of each sequence.
    indices : IntArray
        Indices of sequences to extract.

    Returns
    -------
    tuple
        (X_subset, lengths_subset)
    """
    idxs = get_sequence_indices(lengths)
    seq_idxs = idxs[indices]
    X_subset = np.concatenate([X[start:end] for start, end in seq_idxs])
    return X_subset, lengths[indices]


def concatenate_sequences(sequences: list[Array]) -> tuple[Array, IntArray]:
    """Concatenate a list of sequences into a single array.

    Parameters
    ----------
    sequences : list[Array]
        List of sequence arrays.

    Returns
    -------
    tuple
        (X_concatenated, lengths)
    """
    lengths = np.array([len(seq) for seq in sequences])
    X = np.concatenate(sequences)
    return X, lengths


def iter_sequences(X: Array, lengths: IntArray) -> t.Iterator[Array]:
    """Iterate over individual sequences."""
    idxs = get_sequence_indices(lengths)
    for start, end in idxs:
        yield X[start:end]
