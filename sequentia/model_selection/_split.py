# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation splitters for sequence data.

These splitters work on the sequence level (not sample level),
ensuring that entire sequences are kept together in train/test splits.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.model_selection import (
    KFold as SklearnKFold,
    StratifiedKFold as SklearnStratifiedKFold,
    ShuffleSplit as SklearnShuffleSplit,
    StratifiedShuffleSplit as SklearnStratifiedShuffleSplit,
    RepeatedKFold as SklearnRepeatedKFold,
    RepeatedStratifiedKFold as SklearnRepeatedStratifiedKFold,
)

__all__ = [
    "KFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
]


class KFold(SklearnKFold):
    """K-Fold cross-validator for sequence data.

    Provides train/test indices to split sequence data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the
    k - 1 remaining folds form the training set.

    This splitter operates on the sequence level, not the sample level,
    ensuring that entire sequences are kept together in train/test splits.

    See Also
    --------
    :class:`sklearn.model_selection.KFold`
        :class:`.KFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: t.Any = None
    ) -> t.Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (ignored, only used for length check).
        y : array-like of shape (n_sequences,)
            Sequence labels/targets. The length of y determines the number of sequences.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for this split.
        test : ndarray
            The testing set indices for this split.
        """
        # Use y to determine number of sequences
        n_sequences = len(y) if y is not None else len(X)
        # Create dummy array of sequence indices
        sequence_indices = np.arange(n_sequences)
        # Delegate to parent class with sequence indices
        yield from super().split(sequence_indices, sequence_indices, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            The number of splits.
        """
        # Use sequence count for determining splits
        n_sequences = len(y) if y is not None else (len(X) if X is not None else 0)
        return super().get_n_splits(X=np.zeros(n_sequences), y=None, groups=groups)


class StratifiedKFold(SklearnStratifiedKFold):
    """Stratified K-Fold cross-validator for sequence data.

    Provides train/test indices to split sequence data in train/test sets.

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage
    of samples for each class.

    This splitter operates on the sequence level, not the sample level.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedKFold`
        :class:`.StratifiedKFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: t.Any = None
    ) -> t.Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (ignored).
        y : array-like of shape (n_sequences,)
            Sequence labels for stratification.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for this split.
        test : ndarray
            The testing set indices for this split.
        """
        # y contains sequence labels for stratification
        n_sequences = len(y)
        sequence_indices = np.arange(n_sequences)
        yield from super().split(sequence_indices, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            The number of splits.
        """
        n_sequences = len(y) if y is not None else (len(X) if X is not None else 0)
        return super().get_n_splits(X=np.zeros(n_sequences), y=y, groups=groups)


class ShuffleSplit(SklearnShuffleSplit):
    """Random permutation cross-validator for sequence data.

    Yields indices to split sequence data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits do not
    guarantee that test sets across all folds will be mutually exclusive,
    and might include overlapping samples. However, this is still very likely
    for sizeable datasets.

    This splitter operates on the sequence level, not the sample level.

    See Also
    --------
    :class:`sklearn.model_selection.ShuffleSplit`
        :class:`.ShuffleSplit` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (ignored).
        y : array-like of shape (n_sequences,), optional
            Sequence labels.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for this split.
        test : ndarray
            The testing set indices for this split.
        """
        n_sequences = len(y) if y is not None else len(X)
        sequence_indices = np.arange(n_sequences)
        yield from super().split(sequence_indices, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            The number of splits.
        """
        # ShuffleSplit's n_splits is determined by n_splits parameter, not data
        return self.n_splits


class StratifiedShuffleSplit(SklearnStratifiedShuffleSplit):
    """Stratified :class:`.ShuffleSplit` cross-validator for sequence data.

    Provides train/test indices to split sequence data into train/test sets.

    This cross-validation object is a merge of :class:`.StratifiedKFold`
    and :class:`.ShuffleSplit`, which returns stratified randomized folds.
    The folds are made by preserving the percentage of samples for each class.

    This splitter operates on the sequence level, not the sample level.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedShuffleSplit`
        :class:`.StratifiedShuffleSplit` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (ignored).
        y : array-like of shape (n_sequences,)
            Sequence labels for stratification.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for this split.
        test : ndarray
            The testing set indices for this split.
        """
        n_sequences = len(y) if y is not None else len(X)
        sequence_indices = np.arange(n_sequences)
        yield from super().split(sequence_indices, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            The number of splits.
        """
        return self.n_splits


class RepeatedKFold(SklearnRepeatedKFold):
    """Repeated :class:`.KFold` cross validator for sequence data.

    Repeats :class:`.KFold` n times with different randomization in each repetition.

    This splitter operates on the sequence level, not the sample level.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedKFold`
        :class:`.RepeatedKFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (ignored).
        y : array-like of shape (n_sequences,)
            Sequence labels.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for this split.
        test : ndarray
            The testing set indices for this split.
        """
        n_sequences = len(y) if y is not None else len(X)
        sequence_indices = np.arange(n_sequences)
        yield from super().split(sequence_indices, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            The number of splits.
        """
        # Access the underlying CV's n_splits attribute
        return self.n_repeats * self.cvargs.get('n_splits', 5)


class RepeatedStratifiedKFold(SklearnRepeatedStratifiedKFold):
    """Repeated :class:`.StratifiedKFold` cross validator for sequence data.

    Repeats :class:`.StratifiedKFold` n times with different randomization
    in each repetition.

    This splitter operates on the sequence level, not the sample level.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedStratifiedKFold`
        :class:`.RepeatedStratifiedKFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (ignored).
        y : array-like of shape (n_sequences,)
            Sequence labels for stratification.
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for this split.
        test : ndarray
            The testing set indices for this split.
        """
        n_sequences = len(y) if y is not None else len(X)
        sequence_indices = np.arange(n_sequences)
        yield from super().split(sequence_indices, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            The number of splits.
        """
        # Access the underlying CV's n_splits attribute
        return self.n_repeats * self.cvargs.get('n_splits', 5)
