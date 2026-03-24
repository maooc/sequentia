# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import typing as t

import numpy as np

from sequentia._internal._typing import Array, IntArray

__all__ = ["get_idxs", "iter_X", "SequentialArray"]


def get_idxs(lengths: IntArray, /) -> IntArray:
    ends = lengths.cumsum()
    starts = np.zeros_like(ends)
    starts[1:] = ends[:-1]
    return np.c_[starts, ends]


def iter_X(X: Array, /, *, idxs: IntArray) -> t.Iterator[Array]:
    for start, end in idxs:
        yield X[start:end]


class SequentialArray:
    """A data container for variable-length sequential data that implements
    scikit-learn compatible indexing interfaces.

    This class wraps concatenated observation sequences with their lengths,
    enabling transparent slicing and indexing operations that preserve the
    relationship between sequences and their metadata.

    The container is designed to work seamlessly with scikit-learn's
    cross-validation and parameter search utilities through the metadata
    routing mechanism.

    Parameters
    ----------
    X : Array
        Concatenated observation sequences with shape (n_observations, n_features).

    lengths : IntArray
        Lengths of each individual sequence. The sum must equal n_observations.

    Attributes
    ----------
    X : Array
        The underlying concatenated observation sequences.

    lengths : IntArray
        Lengths of each individual sequence.

    Examples
    --------
    >>> import numpy as np
    >>> from sequentia._internal._data import SequentialArray
    >>> X = np.random.rand(100, 5)  # 100 observations, 5 features
    >>> lengths = np.array([30, 40, 30])  # 3 sequences
    >>> data = SequentialArray(X, lengths)
    >>> len(data)
    3
    >>> subset = data[[0, 2]]  # Get first and third sequences
    >>> len(subset)
    2
    """

    def __init__(self, X: Array, lengths: IntArray) -> None:
        self._X = np.asarray(X)
        self._lengths = np.asarray(lengths, dtype=np.int64)
        self._idxs = get_idxs(self._lengths)

        if self._lengths.sum() != len(self._X):
            msg = (
                f"Sum of lengths ({self._lengths.sum()}) does not match "
                f"number of observations ({len(self._X)})"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self._lengths)

    def __getitem__(
        self, idx: int | slice | list | np.ndarray | tuple
    ) -> SequentialArray | Array:
        """Index into the sequential data.

        Parameters
        ----------
        idx : int, slice, list, ndarray, or tuple
            Index or indices of sequences to retrieve.
            If a tuple, it's assumed to be (indices, ...) format from sklearn's
            _array_indexing function.

        Returns
        -------
        SequentialArray or Array
            If a single integer index, returns the raw sequence array.
            Otherwise, returns a new SequentialArray with the selected sequences.
        """
        if isinstance(idx, tuple):
            idx = idx[0]

        if isinstance(idx, (int, np.integer)):
            start, end = self._idxs[int(idx)]
            return self._X[start:end].copy()

        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(len(self)))
        else:
            idx = np.atleast_1d(idx)

        if len(idx) == 0:
            return SequentialArray(
                np.empty((0, self._X.shape[1]), dtype=self._X.dtype),
                np.array([], dtype=np.int64),
            )

        idx = np.asarray(idx)
        selected_idxs = self._idxs[idx]
        new_lengths = self._lengths[idx]

        new_X = np.vstack(
            [self._X[start:end] for start, end in selected_idxs]
        )

        return SequentialArray(new_X, new_lengths)

    def __array__(self, dtype: np.dtype | None = None) -> Array:
        """Return the underlying X array for numpy compatibility."""
        if dtype is not None:
            return self._X.astype(dtype)
        return self._X

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape compatible with sklearn's expectations.

        Returns (n_sequences,) for indexing purposes.
        """
        return (len(self),)

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return 1

    @property
    def X(self) -> Array:
        """Return the underlying concatenated observation sequences."""
        return self._X

    @property
    def lengths(self) -> IntArray:
        """Return the lengths of each sequence."""
        return self._lengths

    @property
    def idxs(self) -> IntArray:
        """Return start and end indices for each sequence."""
        return self._idxs

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return self._X.shape[1]

    @property
    def n_observations(self) -> int:
        """Return the total number of observations."""
        return len(self._X)

    def iter_sequences(self) -> t.Iterator[Array]:
        """Iterate over individual sequences.

        Yields
        ------
        Array
            Individual sequence arrays.
        """
        for start, end in self._idxs:
            yield self._X[start:end]

    def copy(self) -> SequentialArray:
        """Return a deep copy of the container."""
        return SequentialArray(
            self._X.copy(),
            self._lengths.copy(),
        )

    def to_arrays(self) -> tuple[Array, IntArray]:
        """Return the underlying X and lengths arrays.

        Returns
        -------
        tuple[Array, IntArray]
            The concatenated X array and lengths array.
        """
        return self._X, self._lengths

    @classmethod
    def from_arrays(
        cls, X: Array, lengths: IntArray | None = None
    ) -> SequentialArray:
        """Create a SequentialArray from X and optional lengths.

        If lengths is None, X is treated as a single sequence.

        Parameters
        ----------
        X : Array
            Observation sequences.
        lengths : IntArray, optional
            Lengths of sequences. If None, X is treated as one sequence.

        Returns
        -------
        SequentialArray
            The created container.
        """
        X = np.asarray(X)
        if lengths is None:
            lengths = np.array([len(X)], dtype=np.int64)
        return cls(X, lengths)

    def __repr__(self) -> str:
        return (
            f"SequentialArray("
            f"n_sequences={len(self)}, "
            f"n_observations={self.n_observations}, "
            f"n_features={self.n_features})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequentialArray):
            return NotImplemented
        return (
            np.array_equal(self._X, other._X)
            and np.array_equal(self._lengths, other._lengths)
        )
