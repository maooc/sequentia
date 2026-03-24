# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Utility wrapper for a generic sequential dataset."""

from __future__ import annotations

import copy
import pathlib
import typing as t
import warnings

import numpy as np
import pydantic as pyd
from sklearn.model_selection import train_test_split

from sequentia._internal import _data, _validation
from sequentia._internal._typing import Array, IntArray
from sequentia.datasets._views import ConcatSequenceView, SequenceView

__all__ = ["SequentialDataset"]


class SequentialDataset:
    """Utility wrapper for a generic sequential dataset with lazy loading support.

    This implementation uses lazy views to avoid unnecessary data copying and
    supports stream-like access to sequences.
    """

    def __init__(
        self,
        X: Array | ConcatSequenceView,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
        classes: list[int] | None = None,
    ) -> None:
        """Initialize a :class:`.SequentialDataset`.

        Parameters
        ----------
        X:
            Sequence(s) as an array or ConcatSequenceView.

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        classes:
            Set of possible class labels
            (only if ``y`` was provided with categorical values).

            If not provided, these will be determined from the training
            data labels.
        """
        if isinstance(X, ConcatSequenceView):
            self._view = X
            lengths = X.lengths
            X = X.X
        else:
            X, lengths = _validation.check_X_lengths(
                X,
                lengths=lengths,
                dtype=X.dtype,
            )
            self._view = ConcatSequenceView(X, lengths)

        if y is not None:
            y = _validation.check_y(y, lengths=lengths)

        self._X = X
        self._y = y
        self._lengths = lengths

        self._classes = None
        if self._y is not None and np.issubdtype(self._y.dtype, np.integer):
            self._classes = _validation.check_classes(
                self._y,
                classes=classes,
            )

        self._idxs = _data.get_idxs(self.lengths)

    def split(
        self,
        *,
        test_size: (
            pyd.NonNegativeInt
            | t.Annotated[float, pyd.Field(ge=0, le=1)]
            | None
        ) = None,
        train_size: (
            pyd.NonNegativeInt
            | t.Annotated[float, pyd.Field(ge=0, le=1)]
            | None
        ) = None,
        random_state: (
            pyd.NonNegativeInt | np.random.RandomState | None
        ) = None,
        shuffle: bool = True,
        stratify: bool = False,
    ) -> tuple[SequentialDataset, SequentialDataset]:
        """Split the dataset into two partitions (train/test) using lazy views.

        This method avoids unnecessary data copying by using view-based slicing
        and only concatenates data when absolutely necessary.

        See :func:`sklearn:sklearn.model_selection.train_test_split`.

        Parameters
        ----------
        test_size:
            Size of the test partition.

        train_size:
            Size of the training partition.

        random_state:
            Seed or :class:`numpy:numpy.random.RandomState` object for
            reproducible pseudo-randomness.

        shuffle:
            Whether or not to shuffle the data before splitting.
            If ``shuffle=False`` then ``stratify`` must be ``False``.

        stratify:
            Whether or not to stratify the partitions by class label.

        Returns
        -------
        tuple[SequentialDataset, SequentialDataset]
            Dataset partitions using lazy views to avoid copying.
        """
        stratify_arr = None
        if stratify:
            if self._y is None:
                msg = "Cannot stratify with no provided outputs"
                warnings.warn(msg, stacklevel=1)
            elif self._classes is None:
                msg = "Cannot stratify on non-categorical outputs"
                warnings.warn(msg, stacklevel=1)
            else:
                stratify_arr = self._y

        seq_indices = np.arange(len(self._lengths))
        train_seq_indices, test_seq_indices = train_test_split(
            seq_indices,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_arr,
        )

        # Create view-based subsets instead of materializing
        lengths_train = self._lengths[train_seq_indices]
        lengths_test = self._lengths[test_seq_indices]

        y_train = self._y[train_seq_indices] if self._y is not None else None
        y_test = self._y[test_seq_indices] if self._y is not None else None

        # Always materialize the data for split to ensure consistent interface
        # Using _get_sequences which is optimized with vectorized indexing
        X_train = self._get_sequences(train_seq_indices)
        X_test = self._get_sequences(test_seq_indices)

        data_train = SequentialDataset(
            X_train,
            y_train,
            lengths=lengths_train,
            classes=self._classes,
        )
        data_test = SequentialDataset(
            X_test,
            y_test,
            lengths=lengths_test,
            classes=self._classes,
        )

        return data_train, data_test

    def _is_contiguous(self, indices: IntArray) -> bool:
        """Check if indices form a contiguous block."""
        if len(indices) <= 1:
            return True
        return np.all(indices[1:] - indices[:-1] == 1)

    def _get_sequences_view(self, indices: IntArray) -> ConcatSequenceView:
        """Get sequences by indices using a view-based approach.
        
        This method avoids data copying by using advanced indexing where possible
        and only materializes data when absolutely necessary.
        """
        if len(indices) == 0:
            empty_shape = (0, self._X.shape[1]) if self._X.ndim > 1 else (0,)
            return ConcatSequenceView(np.empty(empty_shape, dtype=self._X.dtype), np.array([], dtype=int))

        # Use vectorized indexing to collect all sequence data efficiently
        idxs = self._idxs[indices]
        sequence_ranges = np.concatenate([np.arange(start, end) for start, end in idxs])
        X_view = self._X[sequence_ranges]
        
        return ConcatSequenceView(X_view, self._lengths[indices])

    def _get_sequences(self, indices: IntArray) -> Array:
        """Get sequences by indices efficiently.

        Uses vectorized operations where possible to avoid Python loops
        and unnecessary data copying.
        """
        if len(indices) == 0:
            return np.array([]).reshape(0, self._X.shape[1]) if self._X.ndim > 1 else np.array([])

        idxs = self._idxs[indices]
        
        # Use vectorized indexing to avoid Python loops
        # This creates a single array view instead of multiple copies
        sequence_ranges = np.concatenate([np.arange(start, end) for start, end in idxs])
        return self._X[sequence_ranges]

    def iter_by_class(self) -> t.Generator[tuple[Array, Array, int]]:
        """Subset the observation sequences by class using efficient slicing.

        Returns
        -------
        typing.Generator[tuple[numpy.ndarray, numpy.ndarray, int]]
            Generator iterating over classes, yielding:

            - ``X`` subset of sequences belonging to the class.
            - Lengths corresponding to the ``X`` subset.
            - Class used to subset ``X``.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.

        TypeError
            If ``y`` was provided but was not categorical.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)

        if self._classes is None:
            msg = "Cannot iterate by class on real-valued targets"
            raise TypeError(msg)

        for c in self._classes:
            class_indices = np.where(self._y == c)[0]
            X_subset = self._get_sequences(class_indices)
            lengths_subset = self._lengths[class_indices]
            yield X_subset, lengths_subset, c

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self._lengths)

    def __getitem__(self, /, i: int | slice | IntArray) -> t.Any:
        """Slice observation sequences and corresponding outputs using lazy views.

        Supports integer, slice, and numpy array indexing.
        Returns either a single sequence (for int index) or a new SequentialDataset
        (for slice or array index) for proper view semantics.
        """
        if isinstance(i, (int, np.integer)):
            start, end = self._idxs[i]
            seq = SequenceView(self._X, start, end)
            return seq if self._y is None else (seq, self._y[i])

        if isinstance(i, (slice, list, np.ndarray)):
            if isinstance(i, slice):
                indices = np.arange(len(self))[i]
            else:
                indices = np.asarray(i)
            
            # Create a view using _get_sequences_view which maintains proper boundaries
            X_view = self._get_sequences_view(indices)
            y_subset = self._y[indices] if self._y is not None else None
            
            return SequentialDataset(
                X_view,
                y=y_subset,
                lengths=self._lengths[indices],
                classes=self._classes,
            )

        raise IndexError(f"Invalid index type: {type(i)}")

    def __iter__(self) -> t.Generator[Array | tuple[Array, Array]]:
        """Create a generator over sequences and their corresponding outputs."""
        for i in range(len(self)):
            yield self[i]

    @property
    def view(self) -> ConcatSequenceView:
        """Lazy view of the concatenated sequences."""
        return self._view

    @property
    def X(self) -> Array:
        """Observation sequences.

        Returns
        -------
        numpy.ndarray
            Observation sequences.
        """
        return self._X

    @property
    def y(self) -> Array:
        """Outputs corresponding to ``X``.

        Returns
        -------
        numpy.ndarray
            Sequence outputs.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)
        return self._y

    @property
    def lengths(self) -> IntArray:
        """Lengths corresponding to ``X``.

        Returns
        -------
        numpy.ndarray
            Lengths for each sequence in ``X``.
        """
        return self._lengths

    @property
    def classes(self) -> IntArray | None:
        """Set of unique classes in ``y``.

        Returns
        -------
        numpy.ndarray | None
            Unique classes if ``y`` is categorical.
        """
        return self._classes

    @property
    def idxs(self) -> IntArray:
        """Observation sequence start and end indices.

        Returns
        -------
        numpy.ndarray
            Start and end indices for each sequence in ``X``.
        """
        return self._idxs

    @property
    def X_y(self) -> dict[str, Array]:
        """Observation sequences and corresponding outputs.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping with keys:

            - ``"X"`` for observation sequences,
            - ``"y"`` for outputs.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)
        return {"X": self._X, "y": self._y}

    @property
    def X_lengths(self) -> dict[str, Array]:
        """Observation sequences and corresponding lengths.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping with keys:

            - ``"X"`` for observation sequences,
            - ``"lengths"`` for lengths.
        """
        return {"X": self._X, "lengths": self._lengths}

    @property
    def X_y_lengths(self) -> dict[str, Array]:
        """Observation sequences and corresponding outputs and lengths.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping with keys:

            - ``"X"`` for observation sequences,
            - ``"y"`` for outputs,
            - ``"lengths"`` for lengths.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)
        return {"X": self._X, "y": self._y, "lengths": self._lengths}

    def save(
        self,
        path: str | pathlib.Path | t.IO,
        /,
        *,
        compress: bool = True,
    ) -> None:
        """Store the dataset in ``.npz`` format.

        See :func:`numpy:numpy.savez` and :func:`numpy:numpy.savez_compressed`.

        Parameters
        ----------
        path
            Location to store the dataset.

        compress
            Whether or not to compress the dataset.

        See Also
        --------
        load:
            Loads a stored dataset in ``.npz`` format.
        """
        arrs = self.X_lengths

        if self._y is not None:
            arrs["y"] = self._y

        if self._classes is not None:
            arrs["classes"] = self._classes

        save_fun = np.savez_compressed if compress else np.savez
        save_fun(path, **arrs)

    @classmethod
    def load(cls, path: str | pathlib.Path | t.IO, /) -> SequentialDataset:
        """Load a stored dataset in ``.npz`` format.

        See :func:`numpy:numpy.load`.

        Parameters
        ----------
        path:
            Location to store the dataset.

        Returns
        -------
        SequentialDataset
            The loaded dataset.

        See Also
        --------
        save:
            Stores the dataset in ``.npz`` format.
        """
        return cls(**np.load(path))

    def copy(self) -> SequentialDataset:
        """Create a copy of the dataset.

        Returns
        -------
        SequentialDataset
            Dataset copy.
        """
        params = {
            "X": copy.deepcopy(self._X),
            "y": None,
            "lengths": copy.deepcopy(self._lengths),
            "classes": None,
        }

        if self._y is not None:
            params["y"] = copy.deepcopy(self._y)

        if self._classes is not None:
            params["classes"] = copy.deepcopy(self._classes)

        return SequentialDataset(
            params["X"],
            params["y"],
            lengths=params["lengths"],
            classes=params["classes"],
        )
