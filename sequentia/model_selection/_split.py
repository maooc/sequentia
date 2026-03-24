# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation splitters for sequence data.

This module provides cross-validation splitters that work correctly with
sequence data, handling the mismatch between sequence-level labels and
observation-level features.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.model_selection import (
    KFold as _KFold,
    RepeatedKFold as _RepeatedKFold,
    RepeatedStratifiedKFold as _RepeatedStratifiedKFold,
    ShuffleSplit as _ShuffleSplit,
    StratifiedKFold as _StratifiedKFold,
    StratifiedShuffleSplit as _StratifiedShuffleSplit,
)

from sequentia._internal import _data
from sequentia._internal._typing import IntArray

__all__ = [
    "KFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
]


def _seq_indices_to_obs_indices(seq_indices: IntArray, lengths: IntArray) -> IntArray:
    """Convert sequence-level indices to observation-level indices.
    
    Parameters
    ----------
    seq_indices : array-like of shape (n_selected_sequences,)
        Indices of selected sequences.
    lengths : array-like of shape (n_sequences,)
        Lengths of all sequences.
        
    Returns
    -------
    obs_indices : ndarray of shape (n_selected_observations,)
        Indices of observations corresponding to selected sequences.
    """
    seq_indices = np.asarray(seq_indices)
    lengths = np.asarray(lengths)
    
    # Calculate start indices for each sequence
    start_indices = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    
    # Build observation indices
    obs_indices = []
    for seq_idx in seq_indices:
        start = start_indices[seq_idx]
        end = start + lengths[seq_idx]
        obs_indices.extend(range(start, end))
    
    return np.array(obs_indices)


def _split_lengths(seq_indices: IntArray, lengths: IntArray) -> IntArray:
    """Split lengths array based on sequence indices.
    
    Parameters
    ----------
    seq_indices : array-like of shape (n_selected_sequences,)
        Indices of selected sequences.
    lengths : array-like of shape (n_sequences,)
        Lengths of all sequences.
        
    Returns
    -------
    selected_lengths : ndarray of shape (n_selected_sequences,)
        Lengths of selected sequences.
    """
    return np.asarray(lengths)[np.asarray(seq_indices)]


class _SequenceCVSplitter:
    """Wrapper for sequence-aware cross-validation.
    
    This class wraps a standard sklearn CV splitter and handles the
    conversion between sequence-level and observation-level indices.
    
    Parameters
    ----------
    base_splitter : object
        The base CV splitter to wrap (e.g., KFold, StratifiedKFold).
    """
    
    def __init__(self, base_splitter):
        self.base_splitter = base_splitter
    
    def split(self, X: np.ndarray, y: np.ndarray, lengths: IntArray, groups: t.Any = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Training data (concatenated observations).
        y : array-like of shape (n_sequences,)
            Target labels (one per sequence).
        lengths : array-like of shape (n_sequences,)
            Lengths of each sequence in X.
        groups : array-like of shape (n_sequences,), default=None
            Group labels for the samples used while splitting the dataset.
            
        Yields
        ------
        train_indices : dict
            Dictionary containing 'X', 'y', and 'lengths' for training set.
        test_indices : dict
            Dictionary containing 'X', 'y', and 'lengths' for test set.
        """
        lengths_arr = np.asarray(lengths)
        n_sequences = len(lengths_arr)
        
        # Create dummy array for sklearn splitter (it will split based on y)
        X_dummy = np.zeros(n_sequences)
        
        for train_seq_idx, test_seq_idx in self.base_splitter.split(X_dummy, y, groups):
            # Convert sequence indices to observation indices
            train_obs_idx = _seq_indices_to_obs_indices(train_seq_idx, lengths_arr)
            test_obs_idx = _seq_indices_to_obs_indices(test_seq_idx, lengths_arr)
            
            # Split lengths
            train_lengths = _split_lengths(train_seq_idx, lengths_arr)
            test_lengths = _split_lengths(test_seq_idx, lengths_arr)
            
            yield {
                'indices': train_obs_idx,
                'lengths': train_lengths,
                'seq_indices': train_seq_idx,
            }, {
                'indices': test_obs_idx,
                'lengths': test_lengths,
                'seq_indices': test_seq_idx,
            }
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.base_splitter.get_n_splits(X, y, groups)


class KFold(_KFold):
    """K-Folds cross-validator for sequence data.
    
    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).
    
    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.
    
    This splitter operates on sequences (not individual observations),
    ensuring that entire sequences are kept together in train/test splits.
    
    See Also
    --------
    sklearn.model_selection.KFold : The base KFold implementation.
    """

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Training data.
        y : array-like of shape (n_sequences,)
            Target labels (one per sequence).
        groups : array-like of shape (n_sequences,), default=None
            Group labels.
            
        Yields
        ------
        train : ndarray of shape (n_train_sequences,)
            The training set indices (sequence-level).
        test : ndarray of shape (n_test_sequences,)
            The testing set indices (sequence-level).
        """
        # Use y for splitting since it has the correct shape (n_sequences,)
        return super().split(y, y, groups)


class StratifiedKFold(_StratifiedKFold):
    """Stratified K-Folds cross-validator for sequence data.
    
    Provides train/test indices to split data in train/test sets.
    
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage
    of samples for each class.
    
    This splitter operates on sequences, ensuring that entire sequences
    are kept together in train/test splits while maintaining class
    distribution.
    
    See Also
    --------
    sklearn.model_selection.StratifiedKFold : The base implementation.
    """

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None):
        """Generate indices to split data into training and test set."""
        # Use y for splitting since it has the correct shape (n_sequences,)
        return super().split(y, y, groups)


class ShuffleSplit(_ShuffleSplit):
    """Random permutation cross-validator for sequence data.
    
    Yields indices to split data into training and test sets.
    
    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.
    
    See Also
    --------
    sklearn.model_selection.ShuffleSplit : The base implementation.
    """

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None):
        """Generate indices to split data into training and test set."""
        # Use y for splitting since it has the correct shape (n_sequences,)
        return super().split(y, y, groups)


class StratifiedShuffleSplit(_StratifiedShuffleSplit):
    """Stratified ShuffleSplit cross-validator for sequence data.
    
    Yields indices to split data into training and test sets with
    stratification.
    
    See Also
    --------
    sklearn.model_selection.StratifiedShuffleSplit : The base implementation.
    """

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None):
        """Generate indices to split data into training and test set."""
        # Use y for splitting since it has the correct shape (n_sequences,)
        return super().split(y, y, groups)


class RepeatedKFold(_RepeatedKFold):
    """Repeated K-Fold cross validator for sequence data.
    
    Repeats K-Fold n times with different randomization in each repetition.
    
    See Also
    --------
    sklearn.model_selection.RepeatedKFold : The base implementation.
    """

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None):
        """Generate indices to split data into training and test set."""
        # Use y for splitting since it has the correct shape (n_sequences,)
        return super().split(y, y, groups)


class RepeatedStratifiedKFold(_RepeatedStratifiedKFold):
    """Repeated Stratified K-Fold cross validator for sequence data.
    
    Repeats Stratified K-Fold n times with different randomization
    in each repetition.
    
    See Also
    --------
    sklearn.model_selection.RepeatedStratifiedKFold : The base implementation.
    """

    def split(self, X: np.ndarray, y: np.ndarray, groups: t.Any = None):
        """Generate indices to split data into training and test set."""
        # Use y for splitting since it has the correct shape (n_sequences,)
        return super().split(y, y, groups)
