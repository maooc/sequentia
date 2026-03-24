# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware cross-validation splitters.

This module provides cross-validation splitters that operate on sequence
indices rather than observation indices, ensuring proper handling of
variable-length sequences.

Key Design Principles
---------------------
1. All splitters operate on **sequence indices**, not observation indices
2. `split()` expects `y` to be sequence labels (one per sequence)
3. `get_n_splits()` returns the number of folds based on sequence count
4. All indices returned refer to sequences, not individual observations
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.model_selection import _split

__all__ = [
    "KFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
]


class KFold(_split.KFold):
    """K-Fold cross-validator for sequence data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the
    k - 1 remaining folds form the training set.

    Important
    ---------
    This cross-validator operates on **sequence indices**, not observation
    indices. The `split` method expects `y` to be the sequence labels
    (one per sequence), and returns indices that refer to sequences.

    The number of folds is determined by the number of sequences, not
    the number of observations.

    Examples
    --------
    >>> import numpy as np
    >>> from sequentia.model_selection import KFold
    >>>
    >>> # 4 sequences with different lengths
    >>> lengths = np.array([10, 15, 8, 12])
    >>> y = np.array([0, 1, 0, 1])  # labels for each sequence
    >>> X = np.random.randn(sum(lengths), 3)  # concatenated observations
    >>>
    >>> cv = KFold(n_splits=2)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     print(f"Train sequences: {train_idx}, Test sequences: {test_idx}")
    Train sequences: [2 3], Test sequences: [0 1]
    Train sequences: [0 1], Test sequences: [2 3]

    See Also
    --------
    :class:`sklearn.model_selection.KFold`
        :class:`.KFold` is a modified version of this class that supports sequences.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

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
            Returns the number of splitting iterations in the cross-validator.
        """
        return super().get_n_splits(y, y, groups)

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None) -> t.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : np.ndarray
            Observation sequences (concatenated). Not used directly,
            but kept for API compatibility.
        y : np.ndarray
            The target variable for supervised learning problems.
            Should be sequence labels (one per sequence).
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split (sequence indices).
        test : ndarray
            The testing set indices for that split (sequence indices).
        """
        return super().split(y, y, groups)


class StratifiedKFold(_split.StratifiedKFold):
    """Stratified K-Fold cross-validator for sequence data.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of
    KFold that returns stratified folds.

    The folds are made by preserving the percentage of samples for each class.

    Important
    ---------
    This cross-validator operates on **sequence indices**, not observation
    indices. The `split` method expects `y` to be the sequence labels
    (one per sequence), and returns indices that refer to sequences.

    The stratification is done at the sequence level, ensuring that
    each fold has approximately the same proportion of sequence labels.

    Examples
    --------
    >>> import numpy as np
    >>> from sequentia.model_selection import StratifiedKFold
    >>>
    >>> # 6 sequences with binary labels
    >>> lengths = np.array([10, 15, 8, 12, 20, 5])
    >>> y = np.array([0, 1, 0, 1, 0, 1])  # labels for each sequence
    >>> X = np.random.randn(sum(lengths), 3)  # concatenated observations
    >>>
    >>> cv = StratifiedKFold(n_splits=2)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     print(f"Train: {train_idx}, Test: {test_idx}")

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedKFold`
        :class:`.StratifiedKFold` is a modified version of this class that supports sequences.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

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
            Returns the number of splitting iterations in the cross-validator.
        """
        return super().get_n_splits(y, y, groups)

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None) -> t.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : np.ndarray
            Observation sequences (concatenated). Not used directly,
            but kept for API compatibility.
        y : np.ndarray
            The target variable for supervised learning problems.
            Should be sequence labels (one per sequence).
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split (sequence indices).
        test : ndarray
            The testing set indices for that split (sequence indices).
        """
        return super().split(y, y, groups)


class ShuffleSplit(_split.ShuffleSplit):
    """Random permutation cross-validator for sequence data.

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits do not
    guarantee that test sets across all folds will be mutually exclusive,
    and might include overlapping samples. However, this is still very likely
    for sizeable datasets.

    Important
    ---------
    This cross-validator operates on **sequence indices**, not observation
    indices. The `split` method expects `y` to be the sequence labels
    (one per sequence), and returns indices that refer to sequences.

    The random sampling is done at the sequence level.

    See Also
    --------
    :class:`sklearn.model_selection.ShuffleSplit`
        :class:`.ShuffleSplit` is a modified version of this class that supports sequences.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

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
            Returns the number of splitting iterations in the cross-validator.
        """
        return super().get_n_splits(y, y, groups)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : np.ndarray
            Observation sequences (concatenated). Not used directly,
            but kept for API compatibility.
        y : np.ndarray | None
            The target variable for supervised learning problems.
            Should be sequence labels (one per sequence).
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split (sequence indices).
        test : ndarray
            The testing set indices for that split (sequence indices).
        """
        return super().split(y, y, groups)


class StratifiedShuffleSplit(_split.StratifiedShuffleSplit):
    """Stratified :class:`.ShuffleSplit` cross-validator for sequence data.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a merge of :class:`.StratifiedKFold`
    and :class:`.ShuffleSplit`, which returns stratified randomized folds.
    The folds are made by preserving the percentage of samples for each class.

    Important
    ---------
    This cross-validator operates on **sequence indices**, not observation
    indices. The `split` method expects `y` to be the sequence labels
    (one per sequence), and returns indices that refer to sequences.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedShuffleSplit`
        :class:`.StratifiedShuffleSplit` is a modified version of this class that supports sequences.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

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
            Returns the number of splitting iterations in the cross-validator.
        """
        return super().get_n_splits(y, y, groups)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : np.ndarray
            Observation sequences (concatenated). Not used directly,
            but kept for API compatibility.
        y : np.ndarray | None
            The target variable for supervised learning problems.
            Should be sequence labels (one per sequence).
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split (sequence indices).
        test : ndarray
            The testing set indices for that split (sequence indices).
        """
        return super().split(y, y, groups)


class RepeatedKFold(_split.RepeatedKFold):
    """Repeated :class:`.KFold` cross validator for sequence data.

    Repeats :class:`.KFold` n times with different randomization in each repetition.

    Important
    ---------
    This cross-validator operates on **sequence indices**, not observation
    indices. The `split` method expects `y` to be the sequence labels
    (one per sequence), and returns indices that refer to sequences.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedKFold`
        :class:`.RepeatedKFold` is a modified version of this class that supports sequences.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

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
            Returns the number of splitting iterations in the cross-validator.
        """
        return super().get_n_splits(y, y, groups)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : np.ndarray
            Observation sequences (concatenated). Not used directly,
            but kept for API compatibility.
        y : np.ndarray | None
            The target variable for supervised learning problems.
            Should be sequence labels (one per sequence).
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split (sequence indices).
        test : ndarray
            The testing set indices for that split (sequence indices).
        """
        return super().split(y, y, groups)


class RepeatedStratifiedKFold(_split.RepeatedStratifiedKFold):
    """Repeated :class:`.StratifiedKFold` cross validator for sequence data.

    Repeats :class:`.StratifiedKFold` n times with different randomization
    in each repetition.

    Important
    ---------
    This cross-validator operates on **sequence indices**, not observation
    indices. The `split` method expects `y` to be the sequence labels
    (one per sequence), and returns indices that refer to sequences.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedStratifiedKFold`
        :class:`.RepeatedStratifiedKFold` is a modified version of this class that supports sequences.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

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
            Returns the number of splitting iterations in the cross-validator.
        """
        return super().get_n_splits(y, y, groups)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> t.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : np.ndarray
            Observation sequences (concatenated). Not used directly,
            but kept for API compatibility.
        y : np.ndarray | None
            The target variable for supervised learning problems.
            Should be sequence labels (one per sequence).
        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split (sequence indices).
        test : ndarray
            The testing set indices for that split (sequence indices).
        """
        return super().split(y, y, groups)
