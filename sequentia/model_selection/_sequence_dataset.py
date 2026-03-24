# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence dataset wrapper for cross-validation.

This module provides a wrapper for sequence data that allows it to work
with standard sklearn cross-validation and parameter search utilities.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray

__all__ = ["SequenceDataset", "SequenceCVSplitter"]


class SequenceDataset:
    """Wrapper for sequence data that provides sklearn-compatible interface.
    
    This class wraps sequence data (X, y, lengths) and provides methods
    to split and index the data correctly for cross-validation.
    
    Parameters
    ----------
    X : array-like of shape (n_observations, n_features)
        Observation data.
    y : array-like of shape (n_sequences,)
        Target labels (one per sequence).
    lengths : array-like of shape (n_sequences,)
        Lengths of each sequence.
        
    Attributes
    ----------
    X : ndarray
        Observation data.
    y : ndarray
        Target labels.
    lengths : ndarray
        Sequence lengths.
    n_sequences : int
        Number of sequences.
    n_observations : int
        Total number of observations.
    """
    
    def __init__(
        self,
        X: Array,
        y: Array,
        lengths: IntArray,
    ):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.lengths = np.asarray(lengths)
        
        # Validate dimensions
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {self.y.shape}")
        if self.lengths.ndim != 1:
            raise ValueError(f"lengths must be 1D, got shape {self.lengths.shape}")
        if len(self.y) != len(self.lengths):
            raise ValueError(
                f"y and lengths must have same length, got {len(self.y)} and {len(self.lengths)}"
            )
        if self.X.shape[0] != self.lengths.sum():
            raise ValueError(
                f"X.shape[0] ({self.X.shape[0]}) must equal sum of lengths ({self.lengths.sum()})"
            )
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.lengths)
    
    @property
    def n_sequences(self) -> int:
        """Return number of sequences."""
        return len(self.lengths)
    
    @property
    def n_observations(self) -> int:
        """Return total number of observations."""
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        """Return number of features."""
        return self.X.shape[1]
    
    def get_sequence(self, idx: int) -> tuple[Array, Array]:
        """Get a single sequence by index.
        
        Parameters
        ----------
        idx : int
            Sequence index.
            
        Returns
        -------
        X_seq : ndarray
            Observations for the sequence.
        y_seq : ndarray
            Label for the sequence.
        """
        start_idx = self.lengths[:idx].sum()
        end_idx = start_idx + self.lengths[idx]
        return self.X[start_idx:end_idx], self.y[idx:idx+1]
    
    def split_by_indices(
        self,
        seq_indices: IntArray,
    ) -> "SequenceDataset":
        """Split dataset by sequence indices.
        
        Parameters
        ----------
        seq_indices : array-like
            Indices of sequences to select.
            
        Returns
        -------
        SequenceDataset
            New dataset containing only the selected sequences.
        """
        seq_indices = np.asarray(seq_indices)
        
        # Calculate observation indices
        start_indices = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        
        obs_indices = []
        new_lengths = []
        for seq_idx in seq_indices:
            start = start_indices[seq_idx]
            end = start + self.lengths[seq_idx]
            obs_indices.extend(range(start, end))
            new_lengths.append(self.lengths[seq_idx])
        
        return SequenceDataset(
            X=self.X[obs_indices],
            y=self.y[seq_indices],
            lengths=np.array(new_lengths),
        )


class SequenceCVSplitter:
    """Cross-validation splitter for sequence data.
    
    This class wraps a standard sklearn CV splitter and properly handles
    the splitting of sequence data, ensuring that X, y, and lengths are
    all split consistently.
    
    Parameters
    ----------
    cv : object
        A sklearn cross-validation splitter (e.g., KFold, StratifiedKFold).
        
    Examples
    --------
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sequentia.model_selection import SequenceCVSplitter
    >>> 
    >>> cv = SequenceCVSplitter(StratifiedKFold(n_splits=3))
    >>> for train_data, test_data in cv.split(X, y, lengths):
    ...     print(f"Train: {len(train_data.y)} sequences")
    ...     print(f"Test: {len(test_data.y)} sequences")
    """
    
    def __init__(self, cv):
        self.cv = cv
    
    def split(
        self,
        X: Array,
        y: Array,
        lengths: IntArray,
        groups: t.Any = None,
    ):
        """Generate sequence-aware train/test splits.
        
        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Observation data.
        y : array-like of shape (n_sequences,)
            Target labels.
        lengths : array-like of shape (n_sequences,)
            Sequence lengths.
        groups : array-like of shape (n_sequences,), default=None
            Group labels.
            
        Yields
        ------
        train_dataset : SequenceDataset
            Training dataset.
        test_dataset : SequenceDataset
            Test dataset.
        """
        dataset = SequenceDataset(X, y, lengths)
        
        # Use y for splitting (sequence-level)
        for train_seq_idx, test_seq_idx in self.cv.split(dataset.y, dataset.y, groups):
            train_dataset = dataset.split_by_indices(train_seq_idx)
            test_dataset = dataset.split_by_indices(test_seq_idx)
            yield train_dataset, test_dataset
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.cv.get_n_splits(X, y, groups)


class SequenceAwareEstimator(BaseEstimator):
    """Wrapper to make an estimator sequence-aware for CV.
    
    This wrapper handles the unpacking of SequenceDataset objects
    and passes X, y, and lengths to the underlying estimator.
    
    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to wrap. Must accept lengths parameter in fit/predict/score.
    """
    
    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
    
    def fit(self, dataset: SequenceDataset, y=None):
        """Fit the estimator on sequence data.
        
        Parameters
        ----------
        dataset : SequenceDataset
            The sequence dataset to fit on.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : SequenceAwareEstimator
            The fitted estimator.
        """
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(
            dataset.X,
            dataset.y,
            lengths=dataset.lengths,
        )
        return self
    
    def predict(self, dataset: SequenceDataset):
        """Predict on sequence data.
        
        Parameters
        ----------
        dataset : SequenceDataset
            The sequence dataset to predict on.
            
        Returns
        -------
        predictions : ndarray
            Predicted values.
        """
        return self.estimator_.predict(
            dataset.X,
            lengths=dataset.lengths,
        )
    
    def score(self, dataset: SequenceDataset, y=None):
        """Score on sequence data.
        
        Parameters
        ----------
        dataset : SequenceDataset
            The sequence dataset to score on.
        y : ignored
            Not used, present for API consistency.
            
        Returns
        -------
        score : float
            The score.
        """
        return self.estimator_.score(
            dataset.X,
            dataset.y,
            lengths=dataset.lengths,
        )
