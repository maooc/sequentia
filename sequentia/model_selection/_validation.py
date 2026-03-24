# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Model validation utilities for sequential data.

This module provides a minimal wrapper around scikit-learn's native validation
infrastructure to handle sequence-specific data formats. The key insight is
that we delegate almost all functionality to scikit-learn's native implementation,
only intercepting to translate sequence indices to actual data slices when needed.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection._validation import (
    _fit_and_score as _sklearn_fit_and_score,
    is_classifier,
)
from sklearn.utils import indexable
from sklearn.utils.validation import _check_method_params

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray
from sequentia.datasets._views import IndexedSequenceView

__all__ = ["_fit_and_score", "cross_val_score", "cross_val_predict"]


class _SequenceDataInterceptor:
    """Interceptor that wraps the estimator to handle sequence data transformations.

    This is a lightweight wrapper that translates the indices passed by scikit-learn
    (which refer to sequences) into the actual concatenated data slices expected by
    the sequence-aware estimator.
    """

    __slots__ = ("estimator", "_lengths", "_X", "_y")

    def __init__(
        self,
        estimator: t.Any,
        X: Array,
        y: Array | None,
        lengths: IntArray,
    ) -> None:
        self.estimator = estimator
        self._X = X
        self._y = y
        self._lengths = lengths

    def _prepare_data(
        self, indices: np.ndarray
    ) -> tuple[Array | IndexedSequenceView, Array | None, IntArray]:
        """Prepare the actual data for the given sequence indices.

        Uses IndexedSequenceView for lazy evaluation, avoiding unnecessary
        data copying and memory overhead until absolutely necessary.
        """
        # Create a lazy view instead of eagerly concatenating
        X_view = IndexedSequenceView(self._X, self._lengths, indices)
        lengths_selected = self._lengths[indices]

        if self._y is not None:
            y_selected = self._y[indices]
            return X_view, y_selected, lengths_selected

        return X_view, None, lengths_selected

    def fit(
        self,
        X_indices: np.ndarray,
        y: t.Any = None,
        **fit_params: t.Any,
    ) -> "_SequenceDataInterceptor":
        """Fit using sequence indices.

        Note: scikit-learn passes dummy X here because we passed a dummy array
        with shape (n_sequences, 1). The "indices" are actually just values from 0..n_sequences.
        """
        # Extract the actual sequence indices from the dummy X
        seq_indices = X_indices.ravel() if hasattr(X_indices, "ravel") else X_indices

        # Prepare actual data for these sequences
        X_data, y_data, lengths_data = self._prepare_data(seq_indices)

        # Update fit_params with lengths
        fit_params = fit_params.copy()
        fit_params["lengths"] = lengths_data

        # Delegate to the actual estimator
        if y_data is not None:
            self.estimator.fit(X_data, y_data, **fit_params)
        else:
            self.estimator.fit(X_data, **fit_params)

        return self

    def predict(self, X_indices: np.ndarray, **predict_params: t.Any) -> Array:
        seq_indices = X_indices.ravel()
        X_data, _, lengths_data = self._prepare_data(seq_indices)
        predict_params["lengths"] = lengths_data
        return self.estimator.predict(X_data, **predict_params)

    def predict_proba(
        self, X_indices: np.ndarray, **predict_params: t.Any
    ) -> Array:
        seq_indices = X_indices.ravel()
        X_data, _, lengths_data = self._prepare_data(seq_indices)
        predict_params["lengths"] = lengths_data
        return self.estimator.predict_proba(X_data, **predict_params)

    def predict_log_proba(
        self, X_indices: np.ndarray, **predict_params: t.Any
    ) -> Array:
        seq_indices = X_indices.ravel()
        X_data, _, lengths_data = self._prepare_data(seq_indices)
        predict_params["lengths"] = lengths_data
        return self.estimator.predict_log_proba(X_data, **predict_params)

    def score(
        self,
        X_indices: np.ndarray,
        y: Array | None = None,
        sample_weight: Array | None = None,
        **score_params: t.Any,
    ) -> float:
        seq_indices = X_indices.ravel()
        X_data, y_data, lengths_data = self._prepare_data(seq_indices)
        score_params["lengths"] = lengths_data

        if sample_weight is not None:
            return self.estimator.score(
                X_data, y_data, sample_weight=sample_weight, **score_params
            )
        return self.estimator.score(X_data, y_data, **score_params)

    def __getattr__(self, name: str) -> t.Any:
        """Delegate all other attributes to the wrapped estimator."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.estimator, name)

    @property
    def __sklearn_is_fitted__(self) -> bool:
        from sklearn.utils.validation import check_is_fitted

        try:
            check_is_fitted(self.estimator)
            return True
        except Exception:
            return False

    @property
    def _estimator_type(self) -> str:
        return getattr(self.estimator, "_estimator_type", None)

    @property
    def classes_(self) -> Array:
        return getattr(self.estimator, "classes_", None)

    def __sklearn_tags__(self) -> t.Any:
        """Delegate sklearn tags to the wrapped estimator."""
        return self.estimator.__sklearn_tags__()

    def get_params(self, deep: bool = True) -> dict[str, t.Any]:
        """Get parameters for this estimator wrapper.

        Required for scikit-learn's clone operation.
        """
        return {
            "estimator": self.estimator,
            "X": self._X,
            "y": self._y,
            "lengths": self._lengths,
        }

    def set_params(self, **params: t.Any) -> "_SequenceDataInterceptor":
        """Set the parameters of this estimator wrapper.

        Required for scikit-learn's clone operation.
        """
        for key, value in params.items():
            if key == "estimator":
                self.estimator = value
            elif key == "X":
                self._X = value
            elif key == "y":
                self._y = value
            elif key == "lengths":
                self._lengths = value
        return self

    def __getstate__(self) -> dict[str, t.Any]:
        """Support for pickling when using __slots__."""
        return {
            "estimator": self.estimator,
            "_X": self._X,
            "_y": self._y,
            "_lengths": self._lengths,
        }

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        """Support for unpickling when using __slots__."""
        self.estimator = state["estimator"]
        self._X = state["_X"]
        self._y = state["_y"]
        self._lengths = state["_lengths"]


def _fit_and_score(
    estimator: t.Any,
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    scorer: t.Any,
    train: np.ndarray,
    test: np.ndarray,
    verbose: int,
    parameters: dict[str, t.Any] | None,
    fit_params: dict[str, t.Any],
    score_params: dict[str, t.Any],
    return_train_score: bool = False,
    return_parameters: bool = False,
    return_n_test_samples: bool = False,
    return_times: bool = False,
    return_estimator: bool = False,
    split_progress: tuple[int, int] | None = None,
    candidate_progress: tuple[int, int] | None = None,
    error_score: float | str = np.nan,
) -> dict[str, t.Any]:
    """Fit estimator and compute scores for a given dataset split.

    This delegates almost entirely to scikit-learn's native _fit_and_score,
    only intercepting to wrap the estimator with our sequence data handler.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_timesteps, n_features)
        The concatenated sequence data.

    y : array-like of shape (n_sequences,) or None
        The target variable (one per sequence).

    scorer : callable
        A scorer callable object.

    train : array-like of shape (n_train_sequences,)
        Indices of training sequences.

    test : array-like of shape (n_test_sequences,)
        Indices of test sequences.

    **kwargs :
        Additional parameters passed to scikit-learn's _fit_and_score.

    Returns
    -------
    result : dict
        Dictionary of scores.
    """
    # Extract lengths from fit_params
    lengths = fit_params.pop("lengths", None) or fit_params.pop("lengths_", None)
    if lengths is None:
        raise ValueError(
            "lengths must be provided via fit_params for sequence validation"
        )

    # Create a dummy X with shape (n_sequences, 1) for scikit-learn
    # This tricks scikit-learn into thinking we're dealing with n_sequences samples
    n_sequences = len(lengths)
    X_dummy = np.arange(n_sequences).reshape(-1, 1)

    # Wrap the estimator to intercept calls and translate to actual data
    wrapped_estimator = _SequenceDataInterceptor(estimator, X, y, lengths)

    # Prepare parameters - ensure lengths is not passed through directly
    # but instead handled by our interceptor
    fit_params_clean = fit_params.copy()
    score_params_clean = score_params.copy() if score_params else {}

    # Delegate to scikit-learn's native implementation
    return _sklearn_fit_and_score(
        wrapped_estimator,
        X_dummy,  # scikit-learn sees this as the "X" with shape (n_sequences, 1)
        y,  # scikit-learn sees this with shape (n_sequences,)
        scorer=scorer,
        train=train,  # these are indices into the n_sequences dimension
        test=test,  # these are indices into the n_sequences dimension
        verbose=verbose,
        parameters=parameters,
        fit_params=fit_params_clean,
        score_params=score_params_clean,
        return_train_score=return_train_score,
        return_parameters=return_parameters,
        return_n_test_samples=return_n_test_samples,
        return_times=return_times,
        return_estimator=return_estimator,
        split_progress=split_progress,
        candidate_progress=candidate_progress,
        error_score=error_score,
    )


def cross_val_score(
    estimator: t.Any,
    X: Array,
    y: Array | None = None,
    *,
    groups: Array | None = None,
    scoring: t.Any = None,
    cv: t.Any = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    params: dict[str, t.Any] | None = None,
    pre_dispatch: str = "2*n_jobs",
    error_score: float | str = np.nan,
    **fit_params: t.Any,
) -> np.ndarray:
    """Evaluate a score by cross-validation with sequence data support.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_timesteps, n_features)
        The concatenated sequence data.

    y : array-like of shape (n_sequences,) or None
        The target variable (one per sequence).

    groups : array-like of shape (n_sequences,) or None
        Group labels for the samples used while splitting the dataset into
        train/test set.

    scoring : str, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y, sample_weight=None).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.

    n_jobs : int, default=None
        Number of jobs to run in parallel.

    verbose : int, default=0
        The verbosity level.

    params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs.

    **fit_params : dict
        Additional parameters passed to the fit method, including 'lengths'.

    Returns
    -------
    scores : array of float, shape=(n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """
    # Extract lengths from fit_params
    lengths = fit_params.pop("lengths", None)
    if lengths is None:
        raise ValueError("lengths must be provided for sequence-aware cross validation")

    # Skip X from indexable since it has a different shape (n_timesteps vs n_sequences)
    # We pass dummy data to scikit-learn instead
    _, y, groups = indexable(np.zeros((len(lengths), 1)), y, groups)

    # Create a dummy X with shape (n_sequences, 1) for scikit-learn
    # This tricks scikit-learn into thinking we're dealing with n_sequences samples
    n_sequences = len(lengths)
    X_dummy = np.arange(n_sequences).reshape(-1, 1)

    # Wrap estimator with a sequence data handler
    wrapped_estimator = _SequenceDataInterceptor(clone(estimator), X, y, lengths)

    # Delegate to scikit-learn's native implementation
    return model_selection.cross_val_score(
        wrapped_estimator,
        X_dummy,
        y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
        **fit_params,
    )


def cross_val_predict(
    estimator: t.Any,
    X: Array,
    y: Array | None = None,
    *,
    groups: Array | None = None,
    cv: t.Any = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    params: dict[str, t.Any] | None = None,
    pre_dispatch: str = "2*n_jobs",
    method: str = "predict",
    **fit_params: t.Any,
) -> Array:
    """Generate cross-validated estimates for each input data point.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_timesteps, n_features)
        The concatenated sequence data.

    y : array-like of shape (n_sequences,) or None
        The target variable (one per sequence).

    groups : array-like of shape (n_sequences,) or None
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.

    n_jobs : int, default=None
        Number of jobs to run in parallel.

    verbose : int, default=0
        The verbosity level.

    params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution.

    method : str, default='predict'
        Invokes the passed method name of the passed estimator.

    **fit_params : dict
        Additional parameters passed to the fit method, including 'lengths'.

    Returns
    -------
    predictions : array
        This is the result of calling `method` on the estimator
        for each data point in X in sequence order.
    """
    # Extract lengths from fit_params
    lengths = fit_params.pop("lengths", None)
    if lengths is None:
        raise ValueError("lengths must be provided for sequence-aware cross validation")

    # Skip X from indexable since it has a different shape (n_timesteps vs n_sequences)
    _, y, groups = indexable(np.zeros((len(lengths), 1)), y, groups)

    # Create a dummy X with shape (n_sequences, 1) for scikit-learn
    n_sequences = len(lengths)
    X_dummy = np.arange(n_sequences).reshape(-1, 1)

    # Wrap estimator with a sequence data handler
    wrapped_estimator = _SequenceDataInterceptor(clone(estimator), X, y, lengths)

    # Use scikit-learn's cross_val_predict
    return model_selection.cross_val_predict(
        wrapped_estimator,
        X_dummy,
        y,
        groups=groups,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        method=method,
        **fit_params,
    )
