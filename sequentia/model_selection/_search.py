# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Hyperparameter search utilities for sequence models.

This module provides wrappers around scikit-learn's hyperparameter search
utilities that are compatible with sequence data. The main difference from
standard scikit-learn search utilities is the handling of sequence lengths
through metadata routing using public scikit-learn APIs only.
"""

from __future__ import annotations

import typing as t
from itertools import product

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV as SKGridSearchCV,
    RandomizedSearchCV as SKRandomizedSearchCV,
)
from sklearn.utils import indexable

from sequentia._internal._sequence_data import SequenceIndexProxy
from sequentia._internal._typing import Array, IntArray

__all__ = ["GridSearchCV", "RandomizedSearchCV", "param_grid"]


def param_grid(**kwargs: list[t.Any]) -> list[dict[str, t.Any]]:
    """Generates a hyper-parameter grid for a nested object.

    Parameters
    ----------
    **kwargs:
        Hyper-parameter name and corresponding values.

    Returns
    -------
    list[dict[str, Any]]
        Hyper-parameter grid for a nested object.

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
    """
    return [
        dict(zip(kwargs.keys(), values))
        for values in product(*kwargs.values())
    ]


class _SequenceSearchCVMixin:
    """Mixin class that adds sequence data support to scikit-learn search classes.

    This mixin leverages scikit-learn's metadata routing system to properly
    handle sequence data. SequenceIndexProxy ensures that sequence metadata
    (lengths) is properly carried through the indexing operations.
    """

    def _call_method(self, method_name, X, **kwargs):
        """Call a method on the estimator with proper sequence metadata."""
        if not hasattr(self, "best_estimator_"):
            raise AttributeError("This GridSearchCV instance is not fitted yet.")

        # Delegate to the best estimator
        method = getattr(self.best_estimator_, method_name)
        return method(X, **kwargs)

    def predict(self, X: Array, **kwargs: t.Any) -> Array:
        """Call predict on the estimator with the best found parameters."""
        return self._call_method("predict", X, **kwargs)

    def predict_proba(self, X: Array, **kwargs: t.Any) -> Array:
        """Call predict_proba on the estimator with the best found parameters."""
        return self._call_method("predict_proba", X, **kwargs)

    def predict_log_proba(self, X: Array, **kwargs: t.Any) -> Array:
        """Call predict_log_proba on the estimator with the best found parameters."""
        return self._call_method("predict_log_proba", X, **kwargs)

    def decision_function(self, X: Array, **kwargs: t.Any) -> Array:
        """Call decision_function on the estimator with the best found parameters."""
        return self._call_method("decision_function", X, **kwargs)

    def transform(self, X: Array, **kwargs: t.Any) -> Array:
        """Call transform on the estimator with the best found parameters."""
        return self._call_method("transform", X, **kwargs)

    def inverse_transform(self, Xt: Array, **kwargs: t.Any) -> Array:
        """Call inverse_transform on the estimator with the best found parameters."""
        return self._call_method("inverse_transform", Xt, **kwargs)

    def score(self, X: Array, y: Array | None = None, **kwargs: t.Any) -> float:
        """Call score on the estimator with the best found parameters."""
        if not hasattr(self, "best_estimator_"):
            raise AttributeError("This GridSearchCV instance is not fitted yet.")
        return self.best_estimator_.score(X, y, **kwargs)

    def _configure_metadata_routing(self, estimator: t.Any) -> None:
        """Configure metadata routing for sequence-aware estimator.

        This method configures the estimator to accept and propagate
        sequence metadata (lengths) during fit, predict, and score.
        """
        # Configure metadata routing for the estimator
        # This tells scikit-learn that our estimator accepts a 'lengths' parameter
        # during fit, predict, and score calls
        # Use try/except for estimators that don't support lengths (like Pipeline)
        try:
            if hasattr(estimator, "set_fit_request"):
                estimator.set_fit_request(lengths=True)
        except TypeError:
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

        # Configure scoring also needs to accept lengths for scoring during cross-validation
        if hasattr(self, "scoring") and callable(self.scoring):
            # Custom scorer functions may need to be configured to accept lengths
            pass

    def fit(self, X: Array, y: Array | None = None, **params: t.Any) -> t.Self:
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_total_time_steps, n_features)
            Training vectors, containing sequences concatenated together.

        y : array-like of shape (n_sequences,), default=None
            Target relative to X for classification or regression.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator, the scorer,
            and the CV splitter. Typically includes ``lengths`` for sequence data.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        # Extract lengths from params if provided
        # This is ONLY at the entry point - lengths are then EMBEDDED in X_wrapped
        # and flow through sklearn's pipeline transparently via the proxy object
        # No further manual handling of lengths is needed after this point
        lengths = params.pop("lengths", None)
        fit_params = params.copy()

        # Configure metadata routing on a clone of the estimator
        # The real configuration happens on clones during cross-validation
        self._configure_metadata_routing(self.estimator)

        # Handle sequence data by wrapping X in a SequenceIndexProxy
        # This proxy CARRIES the lengths metadata through sklearn's indexing operations
        # and into estimator methods, where the decorator resolves them
        if lengths is not None:
            # Wrap X with sequence metadata proxy - THIS IS THE KEY INJECTION POINT
            # After this, lengths travel INSIDE X_wrapped, not as a separate parameter
            X_wrapped = SequenceIndexProxy(X, lengths=lengths)
            
            # NOTE: We do NOT add lengths to fit_params anymore!
            # The @with_resolved_sequence_data decorator in the estimator
            # extracts lengths DIRECTLY from the wrapped X object,
            # enabling true metadata routing without manual parameter passing
        else:
            X_wrapped = X

        # Use standard sklearn fit with prepared data
        # lengths are now CARRIED in X_wrapped and will be extracted
        # by the estimator's decorator when needed
        X_wrapped, y = indexable(X_wrapped, y)
        return super().fit(X_wrapped, y, **fit_params)


class GridSearchCV(_SequenceSearchCVMixin, SKGridSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    This is a sequence-aware version of :class:`sklearn.model_selection.GridSearchCV`
    that properly handles variable-length sequence data through metadata routing.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which
        case the grids spanned by each dictionary in the list are explored.

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    n_jobs : int, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole dataset.

    cv : int, cross-validation generator, or an iterable, default=None
        Determines the cross-validation splitting strategy. For sequence data,
        use a splitter from :mod:`sequentia.model_selection`.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator which gave highest
        score (or smallest loss if specified) on the left out data.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best model.

    n_splits_ : int
        The number of cross-validation splits used.

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

    multimetric_ : bool
        Whether the scorer returns multiple metrics.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    See Also
    --------
    :class:`sklearn.model_selection.GridSearchCV`
        The original scikit-learn implementation.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import minmax_scale
    >>> from sequentia.enums import PriorMode, CovarianceMode
    >>> from sequentia.models import HMMClassifier, GaussianMixtureHMM
    >>> from sequentia.preprocessing import IndependentFunctionTransformer
    >>> from sequentia.model_selection import GridSearchCV, StratifiedKFold
    >>> # Create a pipeline with preprocessing and a sequence model
    >>> pipeline = Pipeline([
    ...     ('scale', IndependentFunctionTransformer(minmax_scale)),
    ...     ('clf', HMMClassifier(variant=GaussianMixtureHMM, random_state=0)),
    ... ])
    >>> # Define the hyperparameter grid
    >>> param_grid = {
    ...     'clf__prior': [PriorMode.UNIFORM, PriorMode.FREQUENCY],
    ...     'clf__model_kwargs': [
    ...         {'n_states': 3, 'covariance': CovarianceMode.DIAGONAL},
    ...         {'n_states': 5, 'covariance': CovarianceMode.SPHERICAL},
    ...     ],
    ... }
    >>> # Create and fit the grid search
    >>> grid_search = GridSearchCV(
    ...     estimator=pipeline,
    ...     param_grid=param_grid,
    ...     cv=StratifiedKFold(n_splits=3),
    ...     verbose=1,
    ... )
    >>> # Fit with sequence lengths
    >>> grid_search.fit(X, y, lengths=lengths)  # doctest: +SKIP
    """
    pass


class RandomizedSearchCV(_SequenceSearchCVMixin, SKRandomizedSearchCV):
    """Randomized search on hyper parameters.

    This is a sequence-aware version of :class:`sklearn.model_selection.RandomizedSearchCV`
    that properly handles variable-length sequence data through metadata routing.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (str) as keys and distributions or
        lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).

    n_iter : int, default=10
        Number of parameter settings that are sampled. ``n_iter`` trades off
        runtime vs quality of the solution.

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    n_jobs : int, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole dataset.

    cv : int, cross-validation generator, or an iterable, default=None
        Determines the cross-validation splitting strategy. For sequence data,
        use a splitter from :mod:`sequentia.model_selection`.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator which gave highest
        score (or smallest loss if specified) on the left out data.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best model.

    n_splits_ : int
        The number of cross-validation splits used.

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

    multimetric_ : bool
        Whether the scorer returns multiple metrics.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    See Also
    --------
    :class:`sklearn.model_selection.RandomizedSearchCV`
        The original scikit-learn implementation.

    :class:`GridSearchCV`:
        Does exhaustive search over a grid of parameters.
    """
    pass
