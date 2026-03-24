# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation splitters for sequential data.

This module provides scikit-learn compatible cross-validation splitters that
properly handle sequential data. All splitters work at the sequence level
rather than the individual observation level.
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
    "TimeSeriesSplit",
]


class SequenceSplitterMixin:
    """Mixin for sequence-aware cross-validation splitters.

    Ensures that splits are performed at the sequence level rather than the
    observation level by using the sequence labels (y) to determine the
    number of sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> t.Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_timesteps, n_features)
            The concatenated training data, where `n_timesteps` is the number
            of timesteps across all sequences and `n_features` is the number
            of features.

        y : array-like of shape (n_sequences,)
            The target variable for supervised learning problems (classification
            or regression). This has length equal to the number of sequences,
            not the number of timesteps.

        groups : array-like of shape (n_sequences,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split. These are sequence-level
            indices, not timestep-level indices.

        test : ndarray
            The testing set indices for that split. These are sequence-level
            indices, not timestep-level indices.
        """
        if y is None:
            raise ValueError(
                "y must be provided for sequence splitters - it defines "
                "the number of sequences and is used for stratification"
            )

        n_sequences = len(y)
        dummy_X = np.arange(n_sequences).reshape(-1, 1)
        return super().split(dummy_X, y, groups)


class KFold(SequenceSplitterMixin, _split.KFold):
    """K-Fold cross-validator for sequential data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the
    k - 1 remaining folds form the training set.

    See Also
    --------
    :class:`sklearn.model_selection.KFold`
        :class:`.KFold` is a modified version
        of this class that supports sequences.
    """


class StratifiedKFold(SequenceSplitterMixin, _split.StratifiedKFold):
    """Stratified K-Fold cross-validator for sequential data.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of
    KFold that returns stratified folds.

    The folds are made by preserving the percentage of samples for each class.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedKFold`
        :class:`.StratifiedKFold` is a modified version
        of this class that supports sequences.
    """


class ShuffleSplit(SequenceSplitterMixin, _split.ShuffleSplit):
    """Random permutation cross-validator for sequential data.

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits do not
    guarantee that test sets across all folds will be mutually exclusive,
    and might include overlapping samples. However, this is still very likely
    for sizeable datasets.

    See Also
    --------
    :class:`sklearn.model_selection.ShuffleSplit`
        :class:`.ShuffleSplit` is a modified version
        of this class that supports sequences.
    """


class StratifiedShuffleSplit(SequenceSplitterMixin, _split.StratifiedShuffleSplit):
    """Stratified :class:`.ShuffleSplit` cross-validator for sequential data.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a merge of :class:`.StratifiedKFold`
    and :class:`.ShuffleSplit`, which returns stratified randomized folds.
    The folds are made by preserving the percentage of samples for each class.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedShuffleSplit`
        :class:`.StratifiedShuffleSplit` is a modified version
        of this class that supports sequences.
    """


class RepeatedKFold(SequenceSplitterMixin, _split.RepeatedKFold):
    """Repeated :class:`.KFold` cross validator for sequential data.

    Repeats :class:`.KFold` n times with different randomization in each repetition.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedKFold`
        :class:`.RepeatedKFold` is a modified version
        of this class that supports sequences.
    """


class RepeatedStratifiedKFold(SequenceSplitterMixin, _split.RepeatedStratifiedKFold):
    """Repeated :class:`.StratifiedKFold` cross validator for sequential data.

    Repeats :class:`.StratifiedKFold` n times with different randomization
    in each repetition.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedStratifiedKFold`
        :class:`.RepeatedStratifiedKFold` is a modified version
        of this class that supports sequences.
    """


class TimeSeriesSplit(SequenceSplitterMixin, _split.TimeSeriesSplit):
    """Time Series cross-validator for sequential data.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    See Also
    --------
    :class:`sklearn.model_selection.TimeSeriesSplit`
        :class:`.TimeSeriesSplit` is a modified version
        of this class that supports sequences.
    """
