# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Validation utilities for sequence data.

This module provides wrappers around scikit-learn's cross-validation utilities
that are aware of sequence data and handle metadata routing for sequence lengths
using the public scikit-learn API only.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.model_selection import cross_val_predict as _cross_val_predict
from sklearn.model_selection import cross_val_score as _cross_val_score
from sklearn.model_selection import cross_validate as _cross_validate
from sklearn.utils import indexable

from sequentia._internal._sequence_data import SequenceIndexProxy
from sequentia._internal._typing import Array, IntArray

__all__ = ["cross_val_predict", "cross_val_score", "cross_validate"]


def _prepare_sequence_routing(
    estimator: t.Any,
    X: Array,
    y: Array | None = None,
    *,
    lengths: IntArray | None = None,
) -> tuple[Array, Array | None, dict[str, t.Any]]:
    """Prepare sequence data for cross-validation using metadata routing.
    
    This function wraps the sequence data in a SequenceIndexProxy that can
    be properly indexed by scikit-learn's cross-validation utilities, while
    preserving sequence length metadata.
    
    Parameters
    ----------
    estimator : estimator object
        The estimator to use for fitting.
        
    X : array-like of shape (n_total_time_steps, n_features)
        Combined sequence array.
        
    y : array-like of shape (n_sequences,), default=None
        Sequence-level target variable.
        
    lengths : array-like of shape (n_sequences,), default=None
        Lengths of the sequences. If None, X is treated as a single sequence.
        
    Returns
    -------
    X_wrapped : array-like or SequenceIndexProxy
        Wrapped X array that can be properly indexed by scikit-learn.
        
    y : array-like or None
        The target array (unchanged).
        
    fit_params : dict
        Fit parameters with metadata routing configuration.
    """
    fit_params: dict[str, t.Any] = {}
    
    if lengths is None:
        # No lengths provided, treat as standard data
        return X, y, fit_params
    
    # Wrap X in a SequenceIndexProxy to allow proper indexing by scikit-learn
    # The proxy preserves sequence metadata through sklearn's indexing operations
    # This is the key: the proxy carries lengths inside it, so when sklearn
    # splits it during cross-validation, each split correctly contains
    # the subset of sequences with corresponding lengths
    X_wrapped = SequenceIndexProxy(X, lengths=lengths)
    
    # Configure metadata routing for the estimator
    # This tells scikit-learn that our estimator can accept 'lengths' parameter
    # The decorator in estimator will extract lengths from X_wrapped when needed
    # We use try/except to handle cases where the estimator doesn't support lengths
    # (e.g., sklearn Pipeline where only specific steps need lengths)
    try:
        if hasattr(estimator, "set_fit_request"):
            estimator.set_fit_request(lengths=True)
    except TypeError:
        # Estimator (or Pipeline) doesn't support lengths parameter globally,
        # but individual steps might be configured externally
        pass
    
    try:
        if hasattr(estimator, "set_predict_request"):
            estimator.set_predict_request(lengths=True)
    except TypeError:
        pass
    
    try:
        if hasattr(estimator, "set_score_request"):
            estimator.set_score_request(lengths=True)
    except TypeError:
        pass
    
    # Note: We don't add 'lengths' to fit_params explicitly
    # because the data is carried INSIDE the X_wrapped proxy
    # When the estimator receives X_wrapped in fit/predict/score,
    # the @with_resolved_sequence_data decorator will extract
    # both X data and lengths from the proxy/view
    
    return X_wrapped, y, fit_params


def cross_validate(
    estimator,
    X: Array,
    y: Array | None = None,
    *,
    lengths: IntArray | None = None,
    groups: Array | None = None,
    scoring: t.Any = None,
    cv: t.Any = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    fit_params: dict[str, t.Any] | None = None,
    pre_dispatch: str = "2*n_jobs",
    return_train_score: bool = False,
    return_estimator: bool = False,
    error_score: str | float = np.nan,
    params: dict[str, t.Any] | None = None,
) -> dict[str, Array]:
    """Evaluate metric(s) by cross-validation and also record fit/score times.
    
    This is a sequence-aware wrapper around scikit-learn's cross_validate function.
    It properly handles variable-length sequence data by using metadata routing
    for sequence lengths through scikit-learn's public API.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
        
    X : array-like of shape (n_total_time_steps, n_features)
        Combined sequence array.
        
    y : array-like of shape (n_sequences,), default=None
        Sequence-level target variable.
        
    lengths : array-like of shape (n_sequences,), default=None
        Lengths of the sequences. If None, X is treated as a single sequence.
        
    groups : array-like of shape (n_sequences,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
        
    scoring : str, callable, list, tuple, or dict, default=None
        A str (see model evaluation documentation) or a scorer callable object
        / function with signature scorer(estimator, X, y).
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. For sequence data,
        use a splitter from sequentia.model_selection.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        
    verbose : int, default=0
        The verbosity level.
        
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
        
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
        
    return_train_score : bool, default=False
        Whether to include train scores.
        
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
        
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        
    params : dict, default=None
        Parameters to pass to the search object.
        
    Returns
    -------
    dict of float arrays
        A dict with keys:
        - test_score: Array of scores on the test set for each CV fold
        - train_score: Array of scores on the train set for each CV fold
          (only if return_train_score is True)
        - fit_time: Time (in seconds) for fitting the model on each fold
        - score_time: Time (in seconds) for scoring the model on each fold
        - estimator: The estimator objects for each fold (only if
          return_estimator is True)
          
    See Also
    --------
    sklearn.model_selection.cross_validate
        The original scikit-learn implementation.
    """
    fit_params = fit_params or {}
    
    # Prepare sequence data using metadata routing
    X, y, seq_fit_params = _prepare_sequence_routing(
        estimator, X, y, lengths=lengths
    )
    fit_params.update(seq_fit_params)
    
    # Use standard sklearn cross_validate with the prepared data
    # No monkey-patching or private API usage required!
    X, y = indexable(X, y)
    
    # sklearn cross_validate uses 'params' parameter for consistency across versions
    all_params = fit_params.copy()
    if params:
        all_params.update(params)
    
    return _cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=all_params,
        pre_dispatch=pre_dispatch,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
        error_score=error_score,
    )


def cross_val_score(
    estimator,
    X: Array,
    y: Array | None = None,
    *,
    lengths: IntArray | None = None,
    groups: Array | None = None,
    scoring: t.Any = None,
    cv: t.Any = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    fit_params: dict[str, t.Any] | None = None,
    pre_dispatch: str = "2*n_jobs",
    error_score: str | float = np.nan,
) -> Array:
    """Evaluate a score by cross-validation.
    
    This is a sequence-aware wrapper around scikit-learn's cross_val_score function.
    It properly handles variable-length sequence data by using metadata routing
    for sequence lengths through scikit-learn's public API.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
        
    X : array-like of shape (n_total_time_steps, n_features)
        Combined sequence array.
        
    y : array-like of shape (n_sequences,), default=None
        Sequence-level target variable.
        
    lengths : array-like of shape (n_sequences,), default=None
        Lengths of the sequences. If None, X is treated as a single sequence.
        
    groups : array-like of shape (n_sequences,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
        
    scoring : str, callable, or None, default=None
        A str (see model evaluation documentation) or a scorer callable object
        / function with signature scorer(estimator, X, y).
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. For sequence data,
        use a splitter from sequentia.model_selection.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        
    verbose : int, default=0
        The verbosity level.
        
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
        
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
        
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        
    Returns
    -------
    array of float of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
        
    See Also
    --------
    sklearn.model_selection.cross_val_score
        The original scikit-learn implementation.
    """
    fit_params = fit_params or {}
    
    # Prepare sequence data using metadata routing
    X, y, seq_fit_params = _prepare_sequence_routing(
        estimator, X, y, lengths=lengths
    )
    fit_params.update(seq_fit_params)
    
    # Use standard sklearn cross_val_score
    X, y = indexable(X, y)
    return _cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=fit_params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
    )


def cross_val_predict(
    estimator,
    X: Array,
    y: Array | None = None,
    *,
    lengths: IntArray | None = None,
    groups: Array | None = None,
    cv: t.Any = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    fit_params: dict[str, t.Any] | None = None,
    pre_dispatch: str = "2*n_jobs",
    method: str = "predict",
) -> Array:
    """Generate cross-validated estimates for each input data point.
    
    This is a sequence-aware wrapper around scikit-learn's cross_val_predict function.
    It properly handles variable-length sequence data by using metadata routing
    for sequence lengths through scikit-learn's public API.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
        
    X : array-like of shape (n_total_time_steps, n_features)
        Combined sequence array.
        
    y : array-like of shape (n_sequences,), default=None
        Sequence-level target variable.
        
    lengths : array-like of shape (n_sequences,), default=None
        Lengths of the sequences. If None, X is treated as a single sequence.
        
    groups : array-like of shape (n_sequences,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. For sequence data,
        use a splitter from sequentia.model_selection.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        
    verbose : int, default=0
        The verbosity level.
        
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
        
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
        
    method : str, default='predict'
        Invokes the passed method name of the passed estimator.
        
    Returns
    -------
    array of shape (n_sequences,) or (n_sequences, n_classes)
        This is the result of calling method on each sequence in the input.
        
    See Also
    --------
    sklearn.model_selection.cross_val_predict
        The original scikit-learn implementation.
    """
    fit_params = fit_params or {}
    
    # Prepare sequence data using metadata routing
    X, y, seq_fit_params = _prepare_sequence_routing(
        estimator, X, y, lengths=lengths
    )
    fit_params.update(seq_fit_params)
    
    # Use standard sklearn cross_val_predict
    X, y = indexable(X, y)
    return _cross_val_predict(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        method=method,
    )
