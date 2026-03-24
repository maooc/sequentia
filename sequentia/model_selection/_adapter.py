# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Adapter layer for scikit-learn integration.

This module provides adapters that allow seamless integration of sequential
data processing with scikit-learn's model selection ecosystem.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.model_selection._validation import _fit_and_score as _sklearn_fit_and_score
from sklearn.utils.validation import _check_method_params

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray

__all__ = [
    "SequenceClassifierAdapter",
    "SequenceEstimatorAdapter",
    "SequenceRegressorAdapter",
    "make_sequence_scorer",
]



class SequenceEstimatorAdapter:
    """Adapter that wraps a sequence-aware estimator to work with scikit-learn's
    model selection utilities.

    This adapter handles the conversion between standard scikit-learn X/y inputs
    and the sequence-specific format required by Sequentia estimators.
    """

    def __init__(
        self,
        estimator: t.Any,
        lengths: IntArray | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        estimator:
            The sequence-aware estimator to wrap.

        lengths:
            Lengths of the sequences. If provided, this is the global lengths
            array that will be indexed by CV splits.
        """
        self.estimator = estimator
        self.lengths = lengths
        self._fit_params: dict[str, t.Any] = {}

    def fit(
        self,
        X: Array,
        y: Array | None = None,
        **fit_params: t.Any,
    ) -> t.Self:
        """Fit the wrapped estimator on the given data.

        Parameters
        ----------
        X:
            The concatenated sequence array of shape (n_timesteps, n_features).

        y:
            The target array of shape (n_sequences,).

        **fit_params:
            Additional fit parameters. May contain 'lengths' or 'lengths_' (routed).
        """
        # Extract lengths from fit_params (metadata routing may have renamed it)
        lengths = fit_params.pop("lengths", None) or fit_params.pop("lengths_", None)

        if lengths is None and self.lengths is not None:
            # Need to compute lengths for this specific split
            # X here contains the concatenated array of the selected sequences
            # We need to determine which sequences were selected and get their lengths
            if hasattr(X, "shape") and self.lengths is not None:
                lengths = self._infer_split_lengths(X)

        if lengths is None:
            raise ValueError(
                "lengths must be provided either during adapter initialization "
                "or via fit_params"
            )

        # Store fit params for potential use in predict/score
        self._fit_params = fit_params.copy()

        if y is None:
            self.estimator.fit(X, lengths=lengths, **fit_params)
        else:
            self.estimator.fit(X, y, lengths=lengths, **fit_params)

        return self

    def _infer_split_lengths(self, X_split: Array) -> IntArray:
        """Infer the lengths for a CV split.

        This is tricky because we only have the concatenated array for the split.
        We use some heuristics to determine which sequences are in this split.
        """
        if self.lengths is None:
            raise ValueError("Cannot infer split lengths without global lengths")

        total_length = len(X_split)
        cum_lengths = np.cumsum(self.lengths)

        # Find all possible ways to select a subset of sequences that sum to total_length
        # This is a fallback - ideally lengths should be passed via fit_params
        candidates = []
        for i in range(len(self.lengths)):
            for j in range(i, len(self.lengths)):
                if cum_lengths[j] - (cum_lengths[i-1] if i > 0 else 0) == total_length:
                    candidates.append(self.lengths[i:j+1])

        if candidates:
            return candidates[0]  # Return first match

        # If no exact match, just return lengths that sum to total_length
        # This is a fallback and may not be correct!
        msg = (
            "Could not infer split lengths from global lengths array. "
            "Please ensure lengths are passed via fit_params."
        )
        import warnings
        warnings.warn(msg, UserWarning, stacklevel=2)

        remaining = total_length
        inferred = []
        for l in self.lengths:
            if remaining >= l:
                inferred.append(l)
                remaining -= l
            if remaining == 0:
                break

        return np.array(inferred)

    def predict(self, X: Array, **predict_params: t.Any) -> Array:
        """Make predictions using the wrapped estimator."""
        if "lengths" not in predict_params and self.lengths is not None:
            predict_params["lengths"] = self._infer_split_lengths(X)
        return self.estimator.predict(X, **predict_params)

    def predict_proba(self, X: Array, **predict_params: t.Any) -> Array:
        """Predict class probabilities using the wrapped estimator."""
        if "lengths" not in predict_params and self.lengths is not None:
            predict_params["lengths"] = self._infer_split_lengths(X)
        return self.estimator.predict_proba(X, **predict_params)

    def predict_log_proba(self, X: Array, **predict_params: t.Any) -> Array:
        """Predict log class probabilities using the wrapped estimator."""
        if "lengths" not in predict_params and self.lengths is not None:
            predict_params["lengths"] = self._infer_split_lengths(X)
        return self.estimator.predict_log_proba(X, **predict_params)

    def score(
        self,
        X: Array,
        y: Array,
        sample_weight: Array | None = None,
        **score_params: t.Any,
    ) -> float:
        """Score the estimator's predictions."""
        if "lengths" not in score_params and self.lengths is not None:
            score_params["lengths"] = self._infer_split_lengths(X)
        if sample_weight is not None:
            return self.estimator.score(X, y, sample_weight=sample_weight, **score_params)
        return self.estimator.score(X, y, **score_params)

    def __getattr__(self, name: str) -> t.Any:
        """Delegate all other attributes to the wrapped estimator."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.estimator, name)

    @property
    def __sklearn_is_fitted__(self) -> bool:
        """Check if the wrapped estimator is fitted."""
        from sklearn.utils.validation import check_is_fitted

        try:
            check_is_fitted(self.estimator)
            return True
        except Exception:
            return False


class SequenceClassifierAdapter(SequenceEstimatorAdapter):
    """Adapter for sequence-aware classifiers.

    Adds classifier-specific attributes required by scikit-learn.
    """

    _estimator_type = "classifier"

    @property
    def classes_(self) -> Array:
        return self.estimator.classes_


class SequenceRegressorAdapter(SequenceEstimatorAdapter):
    """Adapter for sequence-aware regressors.

    Adds regressor-specific attributes required by scikit-learn.
    """

    _estimator_type = "regressor"


def make_sequence_scorer(
    scorer: t.Callable,
    **kwargs: t.Any,
) -> t.Callable:
    """Wrap a scorer to handle sequence data properly.

    Parameters
    ----------
    scorer:
        The scorer function to wrap.

    **kwargs:
        Additional arguments to pass to the scorer.

    Returns
    -------
    callable:
        A wrapped scorer function that handles sequence data formats.
    """
    def sequence_scorer(
        estimator: t.Any,
        X: Array,
        y: Array,
        lengths: IntArray | None = None,
        **scorer_params: t.Any,
    ) -> float:
        all_params = kwargs.copy()
        all_params.update(scorer_params)

        if lengths is not None:
            return scorer(estimator, X, y, lengths=lengths, **all_params)
        return scorer(estimator, X, y, **all_params)

    return sequence_scorer


def _adapt_fit_params_for_split(
    fit_params: dict[str, t.Any],
    train: IntArray,
    test: IntArray,
    global_lengths: IntArray | None = None,
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """Adapt fit parameters for a CV split.

    This handles splitting the lengths array appropriately for train/test indices.
    """
    fit_params = fit_params.copy() if fit_params is not None else {}

    # Extract lengths from fit_params
    lengths = fit_params.pop("lengths", None) or fit_params.pop("lengths_", None)
    if lengths is None:
        lengths = global_lengths

    if lengths is None:
        return fit_params, fit_params.copy()

    # Compute actual data indices for train/test sequences
    train_data_mask = np.zeros(lengths.sum(), dtype=bool)
    test_data_mask = np.zeros(lengths.sum(), dtype=bool)

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    for idx in train:
        start = offsets[idx]
        end = start + lengths[idx]
        train_data_mask[start:end] = True

    for idx in test:
        start = offsets[idx]
        end = start + lengths[idx]
        test_data_mask[start:end] = True

    # Get train and test lengths
    lengths_train = lengths[train]
    lengths_test = lengths[test]

    # Prepare fit params for train
    fit_params_train = fit_params.copy()
    fit_params_train["lengths"] = lengths_train

    # Prepare fit params for test
    fit_params_test = fit_params.copy()
    fit_params_test["lengths"] = lengths_test

    return fit_params_train, fit_params_test


def _prepare_sequence_data_for_split(
    X: Array,
    y: Array | None,
    lengths: IntArray,
    train: IntArray,
    test: IntArray,
) -> tuple[Array, Array, Array | None, Array | None, IntArray, IntArray]:
    """Prepare sequence data for a CV split.

    Returns the actual data arrays for train/test splits.
    """
    idxs = _data.get_idxs(lengths)

    # Get train data
    idxs_train = idxs[train]
    X_train = np.concatenate([X[start:end] for start, end in idxs_train])
    lengths_train = lengths[train]
    y_train = y[train] if y is not None else None

    # Get test data
    idxs_test = idxs[test]
    X_test = np.concatenate([X[start:end] for start, end in idxs_test])
    lengths_test = lengths[test]
    y_test = y[test] if y is not None else None

    return X_train, X_test, y_train, y_test, lengths_train, lengths_test
