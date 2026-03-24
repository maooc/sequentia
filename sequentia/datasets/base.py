# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Utility wrapper for a generic sequential dataset with lazy evaluation support.

Key Design Principles
---------------------
1. Lazy evaluation: Views avoid unnecessary data copying
2. Stream mode: Iterators provide memory-efficient access to large datasets
3. Zero-copy operations: Subsets reference original data until materialization
4. sklearn compatibility: Works with Pipeline and model selection
"""

from __future__ import annotations

import copy
import pathlib
import typing as t
import warnings

import numpy as np
import pydantic as pyd

from sequentia._internal import _sequence, _validation
from sequentia._internal._typing import Array, IntArray

__all__ = ["SequentialDataset", "SequentialDatasetView", "SequenceStream"]


class SequenceStream:
    """Stream interface for memory-efficient iteration over sequences.

    This class provides a generator-based interface for accessing sequences
    without loading all data into memory at once. Useful for very large
    datasets that don't fit in memory.

    The stream yields (X, y) tuples for each sequence, where X is the
    observation sequence and y is the label (if available).
    """

    def __init__(
        self,
        X: Array,
        lengths: IntArray,
        y: Array | None = None,
        idxs: IntArray | None = None,
        indices: IntArray | None = None,
    ) -> None:
        self._X = X
        self._lengths = lengths
        self._y = y
        self._idxs = idxs if idxs is not None else _sequence.get_sequence_indices(lengths)
        self._indices = indices if indices is not None else np.arange(len(lengths))

    def __iter__(self) -> t.Generator[tuple[Array, Array | None]]:
        for i, idx in enumerate(self._indices):
            start, end = self._idxs[i]
            X = self._X[start:end]
            y = self._y[idx] if self._y is not None else None
            yield (X, y) if y is not None else X

    def __len__(self) -> int:
        return len(self._indices)

    def subset(self, indices: IntArray) -> SequenceStream:
        return SequenceStream(
            self._X,
            self._lengths[self._indices],
            self._y[self._indices] if self._y is not None else None,
            self._idxs,
            indices,
        )


class SequentialDatasetView:
    """Lazy view of a subset of a SequentialDataset.

    This class provides efficient access to sequence subsets without
    materializing (copying) the data until explicitly requested.

    Key Features
    ------------
    - Zero-copy: References original data without duplication
    - Lazy materialization: Only copies data when needed
    - Stream interface: Memory-efficient iteration
    - sklearn compatible: Can be used with estimators

    This is useful for cross-validation and train/test splits where
    you want to avoid unnecessary memory allocations.
    """

    def __init__(
        self,
        dataset: SequentialDataset,
        indices: IntArray,
    ) -> None:
        self._dataset = dataset
        self._indices = np.asarray(indices)
        self._idxs = dataset._idxs[self._indices]
        self._lengths = dataset._lengths[self._indices]
        self._y = dataset._y[self._indices] if dataset._y is not None else None
        self._classes = dataset._classes

    @property
    def lengths(self) -> IntArray:
        return self._lengths

    @property
    def y(self) -> Array | None:
        return self._y

    @property
    def classes(self) -> IntArray | None:
        return self._classes

    @property
    def n_sequences(self) -> int:
        return len(self._lengths)

    def materialize(self) -> tuple[Array, IntArray, Array | None]:
        """Materialize the view into actual arrays.

        Note
        ----
        This operation copies data and should be avoided when possible.
        Use iter() or stream() for memory-efficient access.

        Returns
        -------
        tuple
            (X, lengths, y) where X is the concatenated sequences,
            lengths are the sequence lengths, and y are the labels.
        """
        total_len = np.sum(self._lengths)
        n_features = self._dataset._X.shape[1]
        X_concat = np.empty((total_len, n_features), dtype=self._dataset._X.dtype)

        offset = 0
        for start, end in self._idxs:
            n = end - start
            X_concat[offset : offset + n] = self._dataset._X[start:end]
            offset += n

        return (
            X_concat,
            self._lengths.copy(),
            self._y.copy() if self._y is not None else None,
        )

    def to_dataset(self) -> SequentialDataset:
        """Convert the view to a standalone SequentialDataset.

        Note
        ----
        This operation materializes the data. Use sparingly.

        Returns
        -------
        SequentialDataset
            A new dataset containing the materialized data.
        """
        X, lengths, y = self.materialize()
        return SequentialDataset(X, y, lengths=lengths, classes=self._classes)

    def stream(self) -> SequenceStream:
        """Get a stream interface for memory-efficient iteration.

        Returns
        -------
        SequenceStream
            A stream that yields sequences without copying.
        """
        return SequenceStream(
            self._dataset._X,
            self._lengths,
            self._y,
            self._idxs,
            np.arange(len(self._indices)),
        )

    def __len__(self) -> int:
        return len(self._lengths)

    def __iter__(self) -> t.Generator[Array | tuple[Array, Array]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(
        self, i: int | slice | IntArray
    ) -> Array | tuple[Array, Array]:
        if isinstance(i, int):
            start, end = self._idxs[i]
            X = self._dataset._X[start:end]
            return X if self._y is None else (X, self._y[i])
        else:
            indices = self._indices[i]
            return self._dataset[indices]

    def get_X_y_lengths(self) -> dict[str, Array | None]:
        """Get X, y, lengths as a dictionary for sklearn compatibility.

        Note
        ----
        This materializes the data. Use stream() for lazy access.

        Returns
        -------
        dict
            Dictionary with 'X', 'y', 'lengths' keys.
        """
        X, lengths, y = self.materialize()
        result = {"X": X, "lengths": lengths}
        if y is not None:
            result["y"] = y
        return result


class SequentialDataset:
    """Utility wrapper for a generic sequential dataset.

    This class provides efficient storage and access to sequence data,
    with support for lazy evaluation through views and streaming.

    Key Features
    ------------
    - Efficient storage: Sequences are stored concatenated in a single array
    - Lazy views: Create subsets without copying data via split_view()
    - Stream mode: Memory-efficient iteration via stream()
    - Stratified splitting: Proper stratification at the sequence level
    - sklearn compatibility: Works with sklearn's Pipeline and model selection

    Memory Efficiency
    -----------------
    For large datasets, prefer:
    - split_view() over split() - avoids data copying
    - stream() over materialize() - avoids loading all data at once
    - iter() over X property - yields sequences one at a time

    Examples
    --------
    Basic usage::

        from sequentia.datasets import SequentialDataset, load_digits

        # Load data
        data = load_digits()

        # Create a lazy view for cross-validation
        train_view, test_view = data.split_view(test_size=0.2, stratify=True)

        # Stream sequences for memory-efficient processing
        for X, y in train_view.stream():
            process(X, y)

        # Materialize when needed for sklearn estimators
        X_train, lengths_train, y_train = train_view.materialize()

    Integration with sklearn Pipeline::

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import minmax_scale
        from sequentia.preprocessing import IndependentFunctionTransformer
        from sequentia.models import KNNClassifier

        pipeline = Pipeline([
            ("scale", IndependentFunctionTransformer(minmax_scale)),
            ("clf", KNNClassifier(k=1)),
        ])

        # Fit using the dataset's properties
        pipeline.fit(data.X, data.y, lengths=data.lengths)
    """

    def __init__(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
        classes: list[int] | None = None,
    ) -> None:
        """Initialize a :class:`.SequentialDataset`.

        Parameters
        ----------
        X : Array
            Sequence(s).

        y : Array | None
            Outputs corresponding to sequence(s) in ``X``.

        lengths : IntArray | None
            Lengths of the sequence(s) provided in ``X``.
            If ``None``, then ``X`` is assumed to be a single sequence.
            ``len(X)`` should be equal to ``sum(lengths)``.

        classes : list[int] | None
            Set of possible class labels
            (only if ``y`` was provided with categorical values).
            If not provided, these will be determined from the training
            data labels.
        """
        X, lengths = _validation.check_X_lengths(
            X,
            lengths=lengths,
            dtype=X.dtype,
        )
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

        self._idxs = _sequence.get_sequence_indices(self._lengths)

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
        """Split the dataset into two partitions (train/test).

        Warning
        -------
        This method materializes data. For large datasets, use split_view()
        instead to avoid unnecessary memory allocation.

        Parameters
        ----------
        test_size : int | float | None
            Size of the test partition.

        train_size : int | float | None
            Size of the training partition.

        random_state : int | RandomState | None
            Seed or :class:`numpy:numpy.random.RandomState` object for
            reproducible pseudo-randomness.

        shuffle : bool
            Whether or not to shuffle the data before splitting.
            If ``shuffle=False`` then ``stratify`` must be ``False``.

        stratify : bool
            Whether or not to stratify the partitions by class label.

        Returns
        -------
        tuple[SequentialDataset, SequentialDataset]
            Dataset partitions.
        """
        train_view, test_view = self.split_view(
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        return train_view.to_dataset(), test_view.to_dataset()

    def split_view(
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
    ) -> tuple[SequentialDatasetView, SequentialDatasetView]:
        """Split the dataset into two lazy views (train/test).

        This method creates lazy views that reference the original data
        without copying, making it efficient for cross-validation.

        Parameters
        ----------
        test_size : int | float | None
            Size of the test partition.

        train_size : int | float | None
            Size of the training partition.

        random_state : int | RandomState | None
            Seed or :class:`numpy:numpy.random.RandomState` object for
            reproducible pseudo-randomness.

        shuffle : bool
            Whether or not to shuffle the data before splitting.
            If ``shuffle=False`` then ``stratify`` must be ``False``.

        stratify : bool
            Whether or not to stratify the partitions by class label.

        Returns
        -------
        tuple[SequentialDatasetView, SequentialDatasetView]
            Lazy views of the dataset partitions.
        """
        stratify_labels = None
        if stratify:
            if self._y is None:
                msg = "Cannot stratify with no provided outputs"
                warnings.warn(msg, stacklevel=1)
            elif self._classes is None:
                msg = "Cannot stratify on non-categorical outputs"
                warnings.warn(msg, stacklevel=1)
            else:
                stratify_labels = self._y

        from sklearn.model_selection import train_test_split

        idxs = np.arange(len(self._lengths))
        train_idxs, test_idxs = train_test_split(
            idxs,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_labels,
        )

        return (
            SequentialDatasetView(self, train_idxs),
            SequentialDatasetView(self, test_idxs),
        )

    def stream(self) -> SequenceStream:
        """Get a stream interface for memory-efficient iteration.

        Returns
        -------
        SequenceStream
            A stream that yields sequences without copying.
        """
        return SequenceStream(self._X, self._lengths, self._y, self._idxs)

    def iter_by_class(self) -> t.Generator[tuple[Array, Array, int]]:
        """Subset the observation sequences by class.

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
            ind = np.argwhere(self._y == c).flatten()
            idxs = self._idxs[ind]
            total_len = np.sum(self._lengths[ind])
            X = np.empty((total_len, self._X.shape[1]), dtype=self._X.dtype)
            offset = 0
            for start, end in idxs:
                n = end - start
                X[offset : offset + n] = self._X[start:end]
                offset += n
            yield X, self._lengths[ind], c

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self._lengths)

    def __getitem__(
        self, i: int | slice | IntArray
    ) -> Array | tuple[Array, Array]:
        """Slice observation sequences and corresponding outputs."""
        if isinstance(i, int):
            start, end = self._idxs[i]
            X = self._X[start:end]
            return X if self._y is None else (X, self._y[i])

        idxs = self._idxs[i]
        lengths = self._lengths[i]
        total_len = np.sum(lengths)
        X = np.empty((total_len, self._X.shape[1]), dtype=self._X.dtype)
        offset = 0
        for start, end in idxs:
            n = end - start
            X[offset : offset + n] = self._X[start:end]
            offset += n
        return X if self._y is None else (X, self._y[i])

    def __iter__(self) -> t.Generator[Array | tuple[Array, Array]]:
        """Create a generator over sequences and their corresponding
        outputs.
        """
        for i in range(len(self)):
            yield self[i]

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
