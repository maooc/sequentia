# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
"""Hyperparameter search utilities for sequential data.

This module provides minimal wrappers around scikit-learn's search estimators
that properly handle sequential data by using a lightweight estimator wrapper
that handles the sequence-specific data format translation.
"""

from __future__ import annotations

import typing as t
from itertools import product

import numpy as np
from sklearn.base import _fit_context, clone, BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection._search import (
    BaseSearchCV as _BaseSearchCV,
    GridSearchCV as _GridSearchCV,
    RandomizedSearchCV as _RandomizedSearchCV,
)
from sklearn.utils.validation import _check_method_params

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray

__all__ = ["BaseSearchCV", "GridSearchCV", "RandomizedSearchCV", "param_grid"]


def param_grid(**kwargs: list[t.Any]) -> list[dict[str, t.Any]]:
    """Generates a hyper-parameter grid for a nested object.

    Examples
    --------
    Using :func:`.param_grid` in a grid search to cross-validate over
    settings for :class:`.GaussianMixtureHMM`, which is a nested model
    specified in the constructor of a :class:`.HMMClassifier`. ::

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import minmax_scale

        from sequentia.enums import PriorMode, CovarianceMode, TopologyMode
        from sequentia.models import HMMClassifier, GaussianMixtureHMM
        from sequentia.preprocessing import IndependentFunctionTransformer
        from sequentia.model_selection import GridSearchCV, StratifiedKFold

        GridSearchCV(
            estimator=Pipeline(
                [
                    ("scale", IndependentFunctionTransformer(minmax_scale)),
                    ("clf", HMMClassifier(variant=GaussianMixtureHMM)),
                ]
            ),
            param_grid={
                "clf__prior": [PriorMode.UNIFORM, PriorMode.FREQUENCY],
                "clf__model_kwargs": param_grid(
                    n_states=[3, 5, 7],
                    n_components=[2, 3, 4],
                    covariance=[
                        CovarianceMode.DIAGONAL, CovarianceMode.SPHERICAL
                    ],
                    topology=[
                        TopologyMode.LEFT_RIGHT, TopologyMode.LINEAR
                    ],
                )
            },
            cv=StratifiedKFold(),
        )

    Parameters
    ----------
    **kwargs:
        Hyper-parameter name and corresponding values.

    Returns
    -------
    Hyper-parameter grid for a nested object.
    """
    return [dict(zip(kwargs.keys(), values)) for values in product(*kwargs.values())]


class _SequenceEstimatorAdapter(MetaEstimatorMixin, BaseEstimator):
    """Adapts a sequence-aware estimator to work with scikit-learn's search APIs.

    This adapter intercepts calls to fit/predict/score and translates the
    dummy indices passed by scikit-learn into actual sequence data slices.

    The design principle is:
    - Store the full sequence data at adapter creation time
    - When fit/predict is called with dummy indices, reconstruct the actual data
    - Pass the actual data to the underlying sequence-aware estimator
    """

    def __init__(
        self,
        estimator: t.Any,
        X_full: Array | None = None,
        y_full: Array | None = None,
        lengths_full: IntArray | None = None,
    ) -> None:
        self.estimator = estimator
        self.X_full = X_full
        self.y_full = y_full
        self.lengths_full = lengths_full
        self._fit_lengths: IntArray | None = None
        self._estimator_type = getattr(estimator, "_estimator_type", None)
        
    def __sklearn_tags__(self) -> t.Any:
        return getattr(self.estimator, "__sklearn_tags__", lambda: None)()

    def _resolve_indices(
        self, X_indices: np.ndarray
    ) -> tuple[Array, Array | None, IntArray | None]:
        """Resolve dummy indices to actual sequence data.

        Returns
        -------
        X_actual : array-like or None
            Actual sequence data, or None if input was not dummy indices
        y_actual : array-like or None
            Actual labels
        lengths_actual : array-like or None
            Actual lengths
        """
        # Extract sequence indices from dummy X
        # Only treat as dummy if:
        # 1. It's a numpy array
        # 2. Shape is (n_samples, 1) where n_samples > 0
        # 3. Values are small integers (likely indices, not feature values)
        if (
            isinstance(X_indices, np.ndarray)
            and X_indices.ndim == 2
            and X_indices.shape[1] == 1
            and X_indices.shape[0] > 0
        ):
            # Additional check: values should be integers representing indices
            seq_indices = X_indices.ravel()
            # Check if values look like indices (integers within reasonable range)
            if np.issubdtype(seq_indices.dtype, np.integer) or np.allclose(
                seq_indices, seq_indices.astype(int)
            ):
                seq_indices = seq_indices.astype(int)
                # Check that indices are valid for our stored sequence data
                if self.lengths_full is not None and len(seq_indices) > 0:
                    max_idx = np.max(seq_indices)
                    if max_idx < len(self.lengths_full):
                        # Valid dummy indices - resolve them
                        idxs = _data.get_idxs(self.lengths_full)
                        selected_idxs = idxs[seq_indices]

                        # Build actual X
                        X_list = [
                            self.X_full[start:end] for start, end in selected_idxs
                        ]
                        X_actual = (
                            np.concatenate(X_list) if X_list else np.array([])
                        )
                        lengths_actual = self.lengths_full[seq_indices]

                        # Build actual y if available
                        y_actual = (
                            self.y_full[seq_indices]
                            if self.y_full is not None
                            else None
                        )

                        return X_actual, y_actual, lengths_actual

        # Not dummy indices - pass through as-is (refit case or edge case)
        return None, None, None

    def fit(
        self,
        X: np.ndarray,
        y: Array | None = None,
        **fit_params: t.Any,
    ) -> "_SequenceEstimatorAdapter":
        """Fit using either dummy indices or actual data."""
        # Try to resolve indices (for CV case)
        X_actual, y_actual, lengths_actual = self._resolve_indices(X)

        # For the final refit case, use what was passed directly
        if X_actual is None:
            X_actual = X
            y_actual = y
            lengths_actual = fit_params.pop("lengths", None) or fit_params.pop(
                "lengths_", None
            )

        fit_params = fit_params.copy()
        fit_params["lengths"] = lengths_actual

        if y_actual is not None:
            self.estimator.fit(X_actual, y_actual, **fit_params)
        else:
            self.estimator.fit(X_actual, **fit_params)

        self._fit_lengths = lengths_actual
        return self

    def predict(self, X: np.ndarray, **predict_params: t.Any) -> Array:
        X_actual, _, lengths_actual = self._resolve_indices(X)

        if X_actual is None:
            X_actual = X
            lengths_actual = predict_params.pop("lengths", self._fit_lengths)

        predict_params["lengths"] = lengths_actual
        return self.estimator.predict(X_actual, **predict_params)

    def predict_proba(self, X: np.ndarray, **predict_params: t.Any) -> Array:
        X_actual, _, lengths_actual = self._resolve_indices(X)

        if X_actual is None:
            X_actual = X
            lengths_actual = predict_params.pop("lengths", self._fit_lengths)

        predict_params["lengths"] = lengths_actual
        return self.estimator.predict_proba(X_actual, **predict_params)

    def predict_log_proba(self, X: np.ndarray, **predict_params: t.Any) -> Array:
        X_actual, _, lengths_actual = self._resolve_indices(X)

        if X_actual is None:
            X_actual = X
            lengths_actual = predict_params.pop("lengths", self._fit_lengths)

        predict_params["lengths"] = lengths_actual
        return self.estimator.predict_log_proba(X_actual, **predict_params)

    def score(
        self,
        X: np.ndarray,
        y: Array | None = None,
        sample_weight: Array | None = None,
        **score_params: t.Any,
    ) -> float:
        X_actual, y_actual, lengths_actual = self._resolve_indices(X)

        if X_actual is None:
            X_actual = X
            y_actual = y
            lengths_actual = score_params.pop("lengths", self._fit_lengths)

        score_params["lengths"] = lengths_actual

        if sample_weight is not None:
            return self.estimator.score(
                X_actual, y_actual, sample_weight=sample_weight, **score_params
            )
        return self.estimator.score(X_actual, y_actual, **score_params)

    def __getattr__(self, name: str) -> t.Any:
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
    def classes_(self) -> Array:
        return getattr(self.estimator, "classes_", None)
    
    def get_params(self, deep: bool = True) -> dict[str, t.Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=False)
        if deep:
            params.update(self.estimator.get_params(deep=deep))
            # Convert to estimator__ format
            for k, v in list(params.items()):
                if k not in ["estimator", "X_full", "y_full", "lengths_full"]:
                    params[f"estimator__{k}"] = v
                    del params[k]
        return params
    
    def set_params(self, **params: t.Any) -> "_SequenceEstimatorAdapter":
        """Set the parameters of this estimator."""
        valid_params = self.get_params(deep=True)
        
        # Separate adapter params from estimator params
        adapter_params = {}
        estimator_params = {}
        
        for key, value in params.items():
            if key in ["estimator", "X_full", "y_full", "lengths_full"]:
                adapter_params[key] = value
            elif key.startswith("estimator__"):
                estimator_params[key[len("estimator__"):]] = value
            else:
                # Try to pass through to estimator directly
                estimator_params[key] = value
        
        # Set adapter params
        for key, value in adapter_params.items():
            setattr(self, key, value)
        
        # Set estimator params
        if estimator_params:
            self.estimator.set_params(**estimator_params)
        
        return self


class _SearchCVWrapper:
    """Mixin that wraps scikit-learn's search CV classes to handle sequence data.

    This wrapper:
    1. Stores the full sequence data during fit()
    2. Creates dummy indices for scikit-learn to iterate over
    3. Overrides _run_search to use our sequence-aware validation
    4. Lets scikit-learn do its work naturally

    The key insight is that we need to use our own validation module
    (_validation.py) which properly handles the sequence data.
    """

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: t.Any, y: t.Any = None, **params: t.Any) -> t.Self:
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_timesteps, n_features)
            Training vectors - concatenated array of all sequences.

        y : array-like of shape (n_sequences,) or None
            Target relative to X for classification or regression
            (one label per sequence).

        **params : dict of str -> object
            Must include 'lengths' specifying the lengths of each sequence.
        """
        # Manually handle the lengths parameter before calling parent fit to avoid routing issues
        lengths = params.pop("lengths", None)
        if lengths is None:
            raise ValueError(
                "lengths must be provided for sequence-aware hyperparameter search"
            )

        params = _check_method_params(X, params=params)

        # Store the actual sequence data
        self._sequence_X = X
        self._sequence_y = y
        self._sequence_lengths = lengths

        # Create a dummy X with shape (n_sequences, 1) for scikit-learn
        # This tricks scikit-learn into thinking we're dealing with n_sequences samples
        n_sequences = len(lengths)
        X_dummy = np.arange(n_sequences).reshape(-1, 1)

        # Wrap the estimator BEFORE calling fit
        # This ensures that all validation uses the adapter
        self.estimator = _SequenceEstimatorAdapter(
            self.estimator,
            X_full=self._sequence_X,
            y_full=self._sequence_y,
            lengths_full=self._sequence_lengths,
        )

        # Now delegate to the parent fit with our dummy X
        result = super().fit(X_dummy, y, **params)

        # Unwrap the estimator for the final refit
        if isinstance(self.estimator, _SequenceEstimatorAdapter):
            self.estimator = self.estimator.estimator

        return result
    
    def predict(self, X: np.ndarray, **predict_params: t.Any) -> Array:
        lengths = predict_params.pop("lengths", None)
        return self.best_estimator_.predict(X, lengths=lengths, **predict_params)
    
    def predict_proba(self, X: np.ndarray, **predict_params: t.Any) -> Array:
        lengths = predict_params.pop("lengths", None)
        return self.best_estimator_.predict_proba(X, lengths=lengths, **predict_params)
    
    def predict_log_proba(self, X: np.ndarray, **predict_params: t.Any) -> Array:
        lengths = predict_params.pop("lengths", None)
        return self.best_estimator_.predict_log_proba(X, lengths=lengths, **predict_params)
    
    def score(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **score_params: t.Any
    ) -> float:
        lengths = score_params.pop("lengths", None)
        return self.best_estimator_.score(X, y, lengths=lengths, **score_params)
    

class BaseSearchCV(_SearchCVWrapper, _BaseSearchCV):
    """Base class for hyperparameter search with sequential data support.

    This class extends scikit-learn's BaseSearchCV to properly handle
    sequential data with variable lengths.

    See Also
    --------
    :class:`sklearn.model_selection.BaseSearchCV`
        For detailed documentation of parameters.
    """

    def __init__(
        self,
        estimator: t.Any,
        *,
        n_jobs: int | None = None,
        refit: bool = True,
        cv: int | t.Any = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        error_score: float | str = np.nan,
        return_train_score: bool = False,
    ) -> None:
        super().__init__(
            estimator=estimator,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )


class GridSearchCV(_SearchCVWrapper, _GridSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.GridSearchCV`
        For detailed documentation of parameters.
    """

    def __init__(
        self,
        estimator: t.Any,
        param_grid: dict[str, list[t.Any]] | list[dict[str, t.Any]],
        *,
        n_jobs: int | None = None,
        refit: bool = True,
        cv: int | t.Any = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        error_score: float | str = np.nan,
        return_train_score: bool = False,
    ) -> None:
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )


class RandomizedSearchCV(_SearchCVWrapper, _RandomizedSearchCV):
    """Randomized search on hyper parameters.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.RandomizedSearchCV`
        For detailed documentation of parameters.
    """

    def __init__(
        self,
        estimator: t.Any,
        param_distributions: dict[str, t.Any],
        *,
        n_iter: int = 10,
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int | None = None,
        refit: bool = True,
        cv: int | t.Any = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        error_score: float | str = np.nan,
        return_train_score: bool = False,
    ) -> None:
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
