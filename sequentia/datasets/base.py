# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Utility wrapper for a generic sequential dataset with lazy loading support."""

from __future__ import annotations

import copy
import pathlib
import typing as t
import warnings
from collections.abc import Iterator, Sequence

import numpy as np
import pydantic as pyd
from sklearn.model_selection import train_test_split

from sequentia._internal import _data, _validation
from sequentia._internal._typing import Array, IntArray

__all__ = ["SequentialDataset", "LazySequentialDataset"]


class SequentialDataset:
    """Utility wrapper for a generic sequential dataset.
    
    This class provides efficient storage and access to sequential data
    with support for lazy operations and streaming.
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
        X:
            Sequence(s).

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
        """Split the dataset into two partitions (train/test).

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
            Dataset partitions.
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

        idxs = np.arange(len(self._lengths))
        train_idxs, test_idxs = train_test_split(
            idxs,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_arr,
        )

        # Build datasets without unnecessary stacking
        data_train = self._create_subset(train_idxs)
        data_test = self._create_subset(test_idxs)

        return data_train, data_test
    
    def _create_subset(self, idxs: IntArray) -> SequentialDataset:
        """Create a subset dataset from indices without unnecessary copying.
        
        Parameters
        ----------
        idxs:
            Indices of sequences to include in the subset.
            
        Returns
        -------
        SequentialDataset
            Subset dataset.
        """
        idxs = np.atleast_1d(idxs)
        
        # Get sequence boundaries
        subset_idxs = self._idxs[idxs]
        
        # Extract sequences using views where possible
        X_parts = []
        for start, end in subset_idxs:
            X_parts.append(self._X[start:end])
        
        # Concatenate only once
        X_subset = np.concatenate(X_parts) if X_parts else np.array([])
        
        y_subset = self._y[idxs] if self._y is not None else None
        lengths_subset = self._lengths[idxs]
        
        return SequentialDataset(
            X_subset,
            y_subset,
            lengths=lengths_subset,
            classes=self._classes,
        )

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
            subset = self._create_subset(ind)
            yield subset.X, subset.lengths, c

    def iter_sequences(self) -> t.Generator[Array | tuple[Array, Array]]:
        """Iterate over sequences lazily without materializing all at once.
        
        Yields
        ------
        Array or tuple[Array, Array]
            Single sequence or (sequence, label) pair.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self._lengths)

    def __getitem__(self, /, i: int | slice | IntArray) -> Array | tuple[Array, Array]:
        """Slice observation sequences and corresponding outputs.
        
        Supports integer indexing, slicing, and array indexing.
        """
        if isinstance(i, slice):
            idxs = np.arange(len(self._lengths))[i]
            return self._create_subset(idxs)
        
        idxs = np.atleast_2d(self._idxs[i])
        X = list(_data.iter_X(self._X, idxs=idxs))
        
        if isinstance(i, int) and len(X) == 1:
            X = X[0]
            return X if self._y is None else (X, self._y[i])
        else:
            # Multiple indices - return subset dataset
            return self._create_subset(np.atleast_1d(i))

    def __iter__(self) -> t.Generator[Array | tuple[Array, Array]]:
        """Create a generator over sequences and their corresponding
        outputs.
        """
        return self.iter_sequences()

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


class LazySequentialDataset:
    """Lazy loading dataset for large sequential data that doesn't fit in memory.
    
    This class provides an interface for streaming sequence data, loading
    sequences on-demand rather than storing all data in memory.
    
    Parameters
    ----------
    loader : callable
        Function that takes an index and returns the sequence at that index.
        Should return either just X (Array) or (X, y) tuple.
        
    n_sequences : int
        Total number of sequences in the dataset.
        
    lengths : IntArray | None
        Pre-computed sequence lengths. If None, will be determined lazily.
        
    classes : list[int] | None
        Set of possible class labels.
        
    Examples
    --------
    >>> def load_sequence(idx):
    ...     # Load from disk or database
    ...     return np.load(f"sequence_{idx}.npy")
    >>> dataset = LazySequentialDataset(load_sequence, n_sequences=1000)
    >>> for seq in dataset:
    ...     process(seq)
    """
    
    def __init__(
        self,
        loader: t.Callable[[int], Array | tuple[Array, Array]],
        n_sequences: int,
        *,
        lengths: IntArray | None = None,
        classes: list[int] | None = None,
    ) -> None:
        self._loader = loader
        self._n_sequences = n_sequences
        self._lengths = lengths
        self._classes = np.array(classes) if classes is not None else None
        self._y: Array | None = None
        
    def __len__(self) -> int:
        """Return the number of sequences."""
        return self._n_sequences
    
    def __getitem__(self, idx: int) -> Array | tuple[Array, Array]:
        """Load and return a single sequence."""
        if idx < 0 or idx >= self._n_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self._n_sequences})")
        return self._loader(idx)
    
    def __iter__(self) -> t.Generator[Array | tuple[Array, Array]]:
        """Iterate over all sequences."""
        for i in range(self._n_sequences):
            yield self[i]
    
    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> t.Generator[list[Array] | tuple[list[Array], list[Array]]]:
        """Iterate over sequences in batches.
        
        Parameters
        ----------
        batch_size:
            Number of sequences per batch.
        shuffle:
            Whether to shuffle the order of sequences.
        random_state:
            Random seed for shuffling.
            
        Yields
        ------
        list of arrays or tuple of lists
            Batch of sequences or (sequences, labels) if labels are available.
        """
        indices = np.arange(self._n_sequences)
        
        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)
        
        for i in range(0, self._n_sequences, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [self[idx] for idx in batch_indices]
            
            # Check if we have labels
            if batch and isinstance(batch[0], tuple):
                X_batch = [item[0] for item in batch]
                y_batch = [item[1] for item in batch]
                yield X_batch, y_batch
            else:
                yield batch
    
    def to_eager(self) -> SequentialDataset:
        """Convert lazy dataset to eager SequentialDataset.
        
        Note: This will load all sequences into memory.
        
        Returns
        -------
        SequentialDataset
            Eager dataset with all sequences loaded.
        """
        X_parts = []
        y_parts = []
        lengths = []
        
        for item in self:
            if isinstance(item, tuple):
                X_parts.append(item[0])
                y_parts.append(item[1])
            else:
                X_parts.append(item)
            lengths.append(len(X_parts[-1]))
        
        X = np.concatenate(X_parts)
        y = np.array(y_parts) if y_parts else None
        
        return SequentialDataset(
            X,
            y,
            lengths=np.array(lengths),
            classes=self._classes,
        )
    
    @property
    def classes(self) -> IntArray | None:
        """Set of unique classes."""
        return self._classes
    
    def get_lengths(self) -> IntArray:
        """Get sequence lengths, computing lazily if needed."""
        if self._lengths is None:
            # Compute lengths lazily
            lengths = []
            for i in range(self._n_sequences):
                item = self[i]
                X = item[0] if isinstance(item, tuple) else item
                lengths.append(len(X))
            self._lengths = np.array(lengths)
        return self._lengths
