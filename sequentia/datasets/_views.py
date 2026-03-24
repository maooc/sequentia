# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Lazy views for sequential data to avoid unnecessary copying."""

from __future__ import annotations

import typing as t

import numpy as np

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray

__all__ = ["SequenceView", "ConcatSequenceView", "IndexedSequenceView"]


class SequenceView:
    """A lazy view of a single sequence."""

    __slots__ = ("_X", "_start", "_end")

    def __init__(self, X: Array, start: int, end: int) -> None:
        """Initialize a sequence view.

        Parameters
        ----------
        X:
            The underlying concatenated array.

        start:
            Start index of the sequence (inclusive).

        end:
            End index of the sequence (exclusive).
        """
        self._X = X
        self._start = start
        self._end = end

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the sequence."""
        return (self._end - self._start,) + self._X.shape[1:]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the sequence."""
        return self._X.dtype

    def __len__(self) -> int:
        """Length of the sequence."""
        return self._end - self._start

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert to a numpy array."""
        return np.asarray(self._X[self._start : self._end], dtype=dtype)

    def __getitem__(self, key: t.Any) -> Array | SequenceView:
        """Index into the sequence."""
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                return self._X[self._start : self._end : step][start:stop]
            new_start = self._start + start
            new_end = min(self._start + stop, self._end)
            return SequenceView(self._X, new_start, new_end)
        return self._X[self._start : self._end][key]

    def __repr__(self) -> str:
        return f"SequenceView(shape={self.shape}, dtype={self.dtype})"


class ConcatSequenceView:
    """A lazy view of concatenated sequences."""

    __slots__ = ("_X", "_lengths")

    def __init__(self, X: Array, lengths: IntArray) -> None:
        """Initialize a concatenated sequence view.

        Parameters
        ----------
        X:
            The underlying concatenated array.

        lengths:
            Lengths of each sequence.
        """
        self._X = X
        self._lengths = lengths

    @property
    def X(self) -> Array:
        """The underlying concatenated array."""
        return self._X

    @property
    def lengths(self) -> IntArray:
        """Lengths of each sequence."""
        return self._lengths

    @property
    def shape(self) -> tuple[int, ...]:
        """Total shape of all sequences concatenated."""
        return self._X.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the sequences."""
        return self._X.dtype

    def __len__(self) -> int:
        """Number of sequences."""
        return len(self._lengths)

    def __getitem__(
        self, key: int | slice | np.ndarray
    ) -> Array | SequenceView | ConcatSequenceView:
        """Get one or more sequences by index."""
        idxs = _data.get_idxs(self._lengths)

        if isinstance(key, (int, np.integer)):
            start, end = idxs[key]
            return SequenceView(self._X, start, end)

        if isinstance(key, slice):
            sub_lengths = self._lengths[key]
            return ConcatSequenceView(self._X, sub_lengths)

        if isinstance(key, (list, np.ndarray)):
            if np.ndim(key) == 1:
                sub_lengths = self._lengths[key]
                views = []
                for idx in key:
                    start, end = idxs[idx]
                    views.append(self._X[start:end])
                return np.concatenate(views) if views else np.array([])
            raise IndexError("Too many indices for ConcatSequenceView")

        raise IndexError(f"Invalid index type: {type(key)}")

    def __iter__(self) -> t.Iterator[SequenceView]:
        """Iterate over sequences."""
        idxs = _data.get_idxs(self._lengths)
        for start, end in idxs:
            yield SequenceView(self._X, start, end)

    def toarray(self) -> Array:
        """Convert to a single concatenated array."""
        return np.array(self._X)

    def tolist(self) -> list[Array]:
        """Convert to a list of sequence arrays."""
        idxs = _data.get_idxs(self._lengths)
        return [self._X[start:end] for start, end in idxs]

    def __repr__(self) -> str:
        return (
            f"ConcatSequenceView(n_sequences={len(self)}, "
            f"total_length={len(self._X)}, "
            f"n_features={self._X.shape[1] if self._X.ndim > 1 else 1})"
        )


class IndexedSequenceView:
    """A view of non-contiguous sequences selected by indices.

    This view allows lazy access to arbitrary sequences without physically
    concatenating the data, which avoids memory overhead and data copying.
    """

    __slots__ = ("_X", "_all_lengths", "_selected_indices", "_idxs")

    def __init__(self, X: Array, all_lengths: IntArray, selected_indices: IntArray) -> None:
        """Initialize an indexed sequence view.

        Parameters
        ----------
        X:
            The underlying concatenated array of all sequences.

        all_lengths:
            Lengths of all sequences in the underlying data.

        selected_indices:
            Indices of sequences to include in this view.
        """
        self._X = X
        self._all_lengths = all_lengths
        self._selected_indices = np.asarray(selected_indices)
        self._idxs = _data.get_idxs(all_lengths)

    @property
    def X(self) -> Array:
        """Materialize the view into a single concatenated array."""
        return self.toarray()

    @property
    def lengths(self) -> IntArray:
        """Lengths of selected sequences."""
        return self._all_lengths[self._selected_indices]

    @property
    def shape(self) -> tuple[int, ...]:
        """Total shape if materialized."""
        total_length = np.sum(self.lengths)
        return (total_length,) + self._X.shape[1:]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the sequences."""
        return self._X.dtype

    def __len__(self) -> int:
        """Number of selected sequences."""
        return len(self._selected_indices)

    def __getitem__(
        self, key: int | slice | np.ndarray
    ) -> Array | SequenceView | IndexedSequenceView:
        """Get one or more sequences by index."""
        if isinstance(key, (int, np.integer)):
            seq_idx = self._selected_indices[key]
            start, end = self._idxs[seq_idx]
            return SequenceView(self._X, start, end)

        if isinstance(key, slice):
            new_indices = self._selected_indices[key]
            return IndexedSequenceView(self._X, self._all_lengths, new_indices)

        if isinstance(key, (list, np.ndarray)):
            if np.ndim(key) == 1:
                new_indices = self._selected_indices[key]
                return IndexedSequenceView(self._X, self._all_lengths, new_indices)
            raise IndexError("Too many indices for IndexedSequenceView")

        raise IndexError(f"Invalid index type: {type(key)}")

    def __iter__(self) -> t.Iterator[SequenceView]:
        """Iterate over selected sequences."""
        for idx in self._selected_indices:
            start, end = self._idxs[idx]
            yield SequenceView(self._X, start, end)

    def toarray(self) -> Array:
        """Materialize into a single concatenated array."""
        selected_idxs = self._idxs[self._selected_indices]
        sequence_ranges = np.concatenate([np.arange(start, end) for start, end in selected_idxs])
        return self._X[sequence_ranges]

    def tolist(self) -> list[Array]:
        """Convert to a list of sequence arrays."""
        selected_idxs = self._idxs[self._selected_indices]
        return [self._X[start:end] for start, end in selected_idxs]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert to a numpy array."""
        return np.asarray(self.toarray(), dtype=dtype)

    def __repr__(self) -> str:
        return (
            f"IndexedSequenceView(n_sequences={len(self)}, "
            f"total_length={sum(self.lengths)}, "
            f"n_features={self._X.shape[1] if self._X.ndim > 1 else 1})"
        )
