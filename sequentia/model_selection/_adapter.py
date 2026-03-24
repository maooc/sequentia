# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Adapter for using sequence models with standard sklearn components.

This module provides an adapter that allows sequence models to work with
standard sklearn GridSearchCV, cross_val_score, etc. without modifying
sklearn's internal code.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray

__all__ = ["SequenceAdapter"]


class SequenceAdapter(BaseEstimator):
    """Adapter to use sequence models with standard sklearn CV.
    
    This adapter wraps a sequence model and handles the conversion between
    observation-level indices (used by sklearn) and sequence-level data.
    
    Parameters
    ----------
    estimator : BaseEstimator
        The sequence model to wrap. Must accept `lengths` parameter in
        fit, predict, and score methods.
        
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import minmax_scale
    >>> from sequentia.preprocessing import IndependentFunctionTransformer
    >>> from sequentia.models import KNNClassifier
    >>> from sequentia.model_selection import SequenceAdapter, GridSearchCV, KFold
    >>>
    >>> # Create a pipeline with sequence adapter
    >>> pipe = Pipeline([
    ...     ('scale', IndependentFunctionTransformer(minmax_scale)),
    ...     ('clf', SequenceAdapter(KNNClassifier(k=1))),
    ... ])
    >>>
    >>> # Use with GridSearchCV
    >>> optimizer = GridSearchCV(
    ...     estimator=pipe,
    ...     param_grid={'clf__estimator__k': [1, 3, 5]},
    ...     cv=KFold(n_splits=3),
    ... )
    >>> optimizer.fit(X, y, lengths=lengths)
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
        **fit_params,
    ) -> t.Self:
        """Fit the estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Training data.
        y : array-like of shape (n_sequences,) or (n_observations,), default=None
            Target values.
        lengths : array-like of shape (n_sequences,), default=None
            Lengths of each sequence in X.
        **fit_params : dict
            Additional fit parameters.
            
        Returns
        -------
        self : SequenceAdapter
            The fitted adapter.
        """
        if lengths is None:
            raise ValueError(
                "lengths must be provided for sequence data. "
                "If you have a single sequence, pass lengths=[len(X)]."
            )
        
        self.estimator_ = clone(self.estimator)
        self.lengths_ = np.asarray(lengths)
        
        if y is not None:
            self.estimator_.fit(X, y, lengths=self.lengths_, **fit_params)
        else:
            self.estimator_.fit(X, lengths=self.lengths_, **fit_params)
            
        return self

    def predict(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        """Predict using the estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Test data.
        lengths : array-like of shape (n_sequences,), default=None
            Lengths of each sequence in X.
            
        Returns
        -------
        y_pred : array-like
            Predicted values.
        """
        if lengths is None:
            lengths = self.lengths_
        return self.estimator_.predict(X, lengths=lengths)

    def score(
        self,
        X: Array,
        y: Array,
        *,
        lengths: IntArray | None = None,
        **score_params,
    ) -> float:
        """Return the mean accuracy on the given test data.
        
        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Test data.
        y : array-like of shape (n_sequences,)
            True labels.
        lengths : array-like of shape (n_sequences,), default=None
            Lengths of each sequence in X.
        **score_params : dict
            Additional score parameters.
            
        Returns
        -------
        score : float
            Mean accuracy.
        """
        if lengths is None:
            lengths = self.lengths_
        return self.estimator_.score(X, y, lengths=lengths, **score_params)

    def set_fit_request(self, *, lengths: bool = False) -> "SequenceAdapter":
        """Set metadata request for fit method.
        
        Parameters
        ----------
        lengths : bool, default=False
            Whether to request lengths metadata.
            
        Returns
        -------
        self : SequenceAdapter
            The adapter instance.
        """
        # Store the request configuration
        self._fit_request_lengths = lengths
        return self

    def set_predict_request(self, *, lengths: bool = False) -> "SequenceAdapter":
        """Set metadata request for predict method.
        
        Parameters
        ----------
        lengths : bool, default=False
            Whether to request lengths metadata.
            
        Returns
        -------
        self : SequenceAdapter
            The adapter instance.
        """
        self._predict_request_lengths = lengths
        return self

    def set_score_request(self, *, lengths: bool = False) -> "SequenceAdapter":
        """Set metadata request for score method.
        
        Parameters
        ----------
        lengths : bool, default=False
            Whether to request lengths metadata.
            
        Returns
        -------
        self : SequenceAdapter
            The adapter instance.
        """
        self._score_request_lengths = lengths
        return self

    def get_metadata_routing(self) -> dict:
        """Get metadata routing configuration.
        
        Returns
        -------
        routing : dict
            Metadata routing configuration.
        """
        return {
            "fit": {"lengths": getattr(self, "_fit_request_lengths", False)},
            "predict": {"lengths": getattr(self, "_predict_request_lengths", False)},
            "score": {"lengths": getattr(self, "_score_request_lengths", False)},
        }
