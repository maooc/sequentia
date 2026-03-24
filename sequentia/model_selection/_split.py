# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation splitters for sequence data.

This module provides cross-validation splitters that work with sequence data
by operating on sequence indices rather than observation indices.

All splitters are thin wrappers around scikit-learn's public API splitters,
delegating to them for the actual split logic.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.model_selection import KFold as _SKKFold
from sklearn.model_selection import RepeatedKFold as _SKRepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold as _SKRepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit as _SKShuffleSplit
from sklearn.model_selection import StratifiedKFold as _SKStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit as _SKStratifiedShuffleSplit

__all__ = [
    "KFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
]


class KFold(_SKKFold):
    """K-Fold cross-validator for sequence data.

    Provides train/test indices to split sequence data.
    Split dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the
    k - 1 remaining folds form the training set.

    This splitter operates on sequence indices, not observation indices.
    Pass the labels array (y) as the first argument to split().

    See Also
    --------
    :class:`sklearn.model_selection.KFold`
        This class wraps the sklearn KFold for sequence data.
    """

    def split(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, optional
            Ignored. Present for API compatibility.
        y : array-like
            Labels for each sequence. Used to determine the number of sequences.
        groups : object, optional
            Ignored. Present for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if y is None:
            raise ValueError("y must be provided to determine sequence indices")
        n_samples = len(y)
        return super().split(np.zeros(n_samples), y, groups)


class StratifiedKFold(_SKStratifiedKFold):
    """Stratified K-Fold cross-validator for sequence data.

    Provides train/test indices to split sequence data.

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    This splitter operates on sequence indices, not observation indices.
    Pass the labels array (y) as the first argument to split().

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedKFold`
        This class wraps the sklearn StratifiedKFold for sequence data.
    """

    def split(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, optional
            Ignored. Present for API compatibility.
        y : array-like
            Labels for each sequence. Used to determine sequence indices
            and stratification.
        groups : object, optional
            Ignored. Present for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if y is None:
            raise ValueError("y must be provided to determine sequence indices")
        n_samples = len(y)
        return super().split(np.zeros(n_samples), y, groups)


class ShuffleSplit(_SKShuffleSplit):
    """Random permutation cross-validator for sequence data.

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that test sets across all folds will be mutually
    exclusive, and might include overlapping samples. However, this is
    still very likely for sizeable datasets.

    This splitter operates on sequence indices, not observation indices.
    Pass the labels array (y) as the first argument to split().

    See Also
    --------
    :class:`sklearn.model_selection.ShuffleSplit`
        This class wraps the sklearn ShuffleSplit for sequence data.
    """

    def split(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, optional
            Ignored. Present for API compatibility.
        y : array-like, optional
            Labels for each sequence. Used to determine the number of sequences.
        groups : object, optional
            Ignored. Present for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = len(y) if y is not None else self.n_samples
        return super().split(np.zeros(n_samples), y, groups)


class StratifiedShuffleSplit(_SKStratifiedShuffleSplit):
    """Stratified ShuffleSplit cross-validator for sequence data.

    Provides train/test indices to split data into training and test sets.

    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds.
    The folds are made by preserving the percentage of samples for each class.

    This splitter operates on sequence indices, not observation indices.
    Pass the labels array (y) as the first argument to split().

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedShuffleSplit`
        This class wraps the sklearn StratifiedShuffleSplit for sequence data.
    """

    def split(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, optional
            Ignored. Present for API compatibility.
        y : array-like
            Labels for each sequence. Used to determine sequence indices
            and stratification.
        groups : object, optional
            Ignored. Present for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if y is None:
            raise ValueError("y must be provided to determine sequence indices")
        n_samples = len(y)
        return super().split(np.zeros(n_samples), y, groups)


class RepeatedKFold(_SKRepeatedKFold):
    """Repeated K-Fold cross validator for sequence data.

    Repeats KFold n times with different randomization in each repetition.

    This splitter operates on sequence indices, not observation indices.
    Pass the labels array (y) as the first argument to split().

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedKFold`
        This class wraps the sklearn RepeatedKFold for sequence data.
    """

    def split(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, optional
            Ignored. Present for API compatibility.
        y : array-like, optional
            Labels for each sequence. Used to determine the number of sequences.
        groups : object, optional
            Ignored. Present for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = len(y) if y is not None else self.n_splits
        return super().split(np.zeros(n_samples), y, groups)


class RepeatedStratifiedKFold(_SKRepeatedStratifiedKFold):
    """Repeated Stratified K-Fold cross validator for sequence data.

    Repeats StratifiedKFold n times with different randomization
    in each repetition.

    This splitter operates on sequence indices, not observation indices.
    Pass the labels array (y) as the first argument to split().

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedStratifiedKFold`
        This class wraps the sklearn RepeatedStratifiedKFold for sequence data.
    """

    def split(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, optional
            Ignored. Present for API compatibility.
        y : array-like
            Labels for each sequence. Used to determine sequence indices
            and stratification.
        groups : object, optional
            Ignored. Present for API compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if y is None:
            raise ValueError("y must be provided to determine sequence indices")
        n_samples = len(y)
        return super().split(np.zeros(n_samples), y, groups)
