# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware hyperparameter search utilities.

This module provides sequence-aware alternatives to sklearn's hyperparameter
search classes, designed to work seamlessly with sequence data.

Key Design Principles
---------------------
1. Use sklearn public APIs only - no private module imports
2. Only implement sequence-specific logic
3. Leverage sklearn's BaseSearchCV infrastructure
"""

from __future__ import annotations

import time
import typing as t
from collections import defaultdict
from itertools import product

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.model_selection import RandomizedSearchCV as sklearn_RandomizedSearchCV
from sklearn.model_selection import check_cv
from sklearn.utils.parallel import Parallel, delayed

from sequentia.model_selection._validation import _fit_and_score

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
    return [
        dict(zip(kwargs.keys(), values))
        for values in product(*kwargs.values())
    ]


class BaseSearchCV:
    """Base class for hyperparameter search with sequence data support.

    This class provides sequence-aware hyperparameter search that properly
    handles the `lengths` parameter for sequence data.

    Unlike sklearn's BaseSearchCV, this implementation:
    1. Uses only public sklearn APIs
    2. Properly routes the `lengths` parameter through the evaluation pipeline
    3. Operates on sequence indices, not observation indices
    """

    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def _get_scorer(self):
        """Get the scorer from the estimator."""
        return check_scoring(self.estimator, scoring=self.scoring)

    def _check_input(self, X, y, params):
        """Validate input parameters."""
        fit_params = params.copy() if params else {}
        return fit_params

    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Training vectors, where `n_observations` is the total number of
            observations across all sequences.

        y : array-like of shape (n_sequences,), default=None
            Target relative to X for classification or regression.
            One label per sequence, not per observation.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator.
            Must include `lengths` parameter for sequence data.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        scorer = self._get_scorer()
        fit_params = self._check_input(X, y, params)

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        n_splits = cv.get_n_splits(X, y)

        base_estimator = clone(self.estimator)

        candidate_params = list(self._iter_candidates())

        n_candidates = len(candidate_params)
        if self.verbose > 0:
            print(
                f"Fitting {n_splits} folds for each of {n_candidates} candidates, "
                f"totalling {n_candidates * n_splits} fits"
            )

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, verbose=self.verbose)

        with parallel:
            out = parallel(
                delayed(_fit_and_score)(
                    clone(base_estimator),
                    X,
                    y,
                    scorer=scorer,
                    train=train,
                    test=test,
                    verbose=self.verbose,
                    parameters=params,
                    fit_params=fit_params,
                    score_params={},
                    return_train_score=self.return_train_score,
                    return_times=True,
                    return_parameters=True,
                    error_score=self.error_score,
                    split_progress=(split_idx, n_splits),
                    candidate_progress=(cand_idx, n_candidates),
                )
                for (cand_idx, params), (split_idx, (train, test)) in product(
                    enumerate(candidate_params),
                    enumerate(cv.split(X, y)),
                )
            )

        n_failed = sum(1 for r in out if r.get("fit_error") is not None)
        if n_failed == len(out):
            raise ValueError(
                f"All the {len(out)} fits failed. "
                "It is very likely that your model is misconfigured. "
                "You can try to debug the error by setting error_score='raise'."
            )

        results = self._format_results(candidate_params, n_splits, out)

        self.cv_results_ = results
        self.n_splits_ = n_splits
        self.scorer_ = scorer
        self.multimetric_ = False

        if self.refit:
            best_index = self._select_best_index(results)
            self.best_index_ = best_index
            self.best_score_ = results["mean_test_score"][best_index]
            self.best_params_ = results["params"][best_index]

            self.best_estimator_ = clone(base_estimator).set_params(**clone(self.best_params_, safe=False))

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            self.refit_time_ = time.time() - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        return self

    def _iter_candidates(self):
        """Iterate over candidate parameter settings.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _select_best_index(self, results):
        """Select the best parameter setting index."""
        return np.argmax(results["mean_test_score"])

    def _format_results(self, candidate_params, n_splits, out):
        """Format the results dictionary."""
        n_candidates = len(candidate_params)

        results = {}

        test_scores = [r["test_scores"] for r in out]
        results["test_score"] = np.array(test_scores).reshape(n_candidates, n_splits).T

        if self.return_train_score:
            train_scores = [r["train_scores"] for r in out]
            results["train_score"] = np.array(train_scores).reshape(n_candidates, n_splits).T

        fit_times = [r["fit_time"] for r in out]
        results["fit_time"] = np.array(fit_times).reshape(n_candidates, n_splits).T

        score_times = [r["score_time"] for r in out]
        results["score_time"] = np.array(score_times).reshape(n_candidates, n_splits).T

        params_list = [r["parameters"] for r in out]
        results["params"] = [params_list[i * n_splits] for i in range(n_candidates)]

        results["mean_test_score"] = np.mean(results["test_score"], axis=0)
        results["std_test_score"] = np.std(results["test_score"], axis=0)
        results["rank_test_score"] = np.zeros(n_candidates, dtype=int)
        ranks = np.argsort(-results["mean_test_score"])
        results["rank_test_score"][ranks] = np.arange(1, n_candidates + 1)

        if self.return_train_score:
            results["mean_train_score"] = np.mean(results["train_score"], axis=0)
            results["std_train_score"] = np.std(results["train_score"], axis=0)

        results["mean_fit_time"] = np.mean(results["fit_time"], axis=0)
        results["std_fit_time"] = np.std(results["fit_time"], axis=0)
        results["mean_score_time"] = np.mean(results["score_time"], axis=0)
        results["std_score_time"] = np.std(results["score_time"], axis=0)

        return results

    def predict(self, X, **params):
        """Call predict on the estimator with the best found parameters.

        Parameters
        ----------
        X : array-like
            Input data.
        **params : dict
            Parameters to pass to predict.

        Returns
        -------
        array-like
            Predicted values.
        """
        self._check_is_fitted()
        return self.best_estimator_.predict(X, **params)

    def predict_proba(self, X, **params):
        """Call predict_proba on the estimator with the best found parameters.

        Parameters
        ----------
        X : array-like
            Input data.
        **params : dict
            Parameters to pass to predict_proba.

        Returns
        -------
        array-like
            Predicted probabilities.
        """
        self._check_is_fitted()
        return self.best_estimator_.predict_proba(X, **params)

    def predict_log_proba(self, X, **params):
        """Call predict_log_proba on the estimator with the best found parameters.

        Parameters
        ----------
        X : array-like
            Input data.
        **params : dict
            Parameters to pass to predict_log_proba.

        Returns
        -------
        array-like
            Predicted log probabilities.
        """
        self._check_is_fitted()
        return self.best_estimator_.predict_log_proba(X, **params)

    def score(self, X, y=None, **params):
        """Return the score on the given data.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like, default=None
            Target values.
        **params : dict
            Parameters to pass to score.

        Returns
        -------
        float
            Score of the best estimator.
        """
        self._check_is_fitted()
        return self.best_estimator_.score(X, y, **params)

    def _check_is_fitted(self):
        """Check if the estimator has been fitted."""
        if not hasattr(self, "best_estimator_"):
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    This class extends sklearn's GridSearchCV to properly handle sequence
    data where each sample consists of multiple observations.

    Important
    ---------
    The `cv` parameter should be a splitter from :mod:`sequentia.model_selection`
    that operates on sequence indices.

    Parameters
    ----------
    estimator : estimator object
        The estimator to optimize.

    param_grid : dict
        Parameter grid to search.

    scoring : str, callable, or None
        Scoring metric to use.

    n_jobs : int or None
        Number of parallel jobs.

    refit : bool
        Whether to refit the best estimator.

    cv : cross-validator
        Cross-validation splitter. Should be from sequentia.model_selection.

    verbose : int
        Verbosity level.

    pre_dispatch : str
        Pre-dispatch for parallel jobs.

    error_score : float or 'raise'
        Value to assign if fitting fails.

    return_train_score : bool
        Whether to return training scores.

    Examples
    --------
    >>> from sequentia.model_selection import GridSearchCV, StratifiedKFold
    >>> from sequentia.models import KNNClassifier
    >>> from sequentia.datasets import load_digits
    >>>
    >>> data = load_digits()
    >>> cv = StratifiedKFold(n_splits=3)
    >>> search = GridSearchCV(
    ...     KNNClassifier(),
    ...     param_grid={"k": [1, 3, 5]},
    ...     cv=cv,
    ... )
    >>> search.fit(data.X, data.y, lengths=data.lengths)
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid

    def _iter_candidates(self):
        """Iterate over all parameter combinations."""
        from sklearn.model_selection import ParameterGrid

        return iter(ParameterGrid(self.param_grid))


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    This class extends sklearn's RandomizedSearchCV to properly handle
    sequence data where each sample consists of multiple observations.

    Important
    ---------
    The `cv` parameter should be a splitter from :mod:`sequentia.model_selection`
    that operates on sequence indices.

    Parameters
    ----------
    estimator : estimator object
        The estimator to optimize.

    param_distributions : dict
        Parameter distributions to sample from.

    n_iter : int
        Number of parameter settings to sample.

    scoring : str, callable, or None
        Scoring metric to use.

    n_jobs : int or None
        Number of parallel jobs.

    refit : bool
        Whether to refit the best estimator.

    cv : cross-validator
        Cross-validation splitter. Should be from sequentia.model_selection.

    verbose : int
        Verbosity level.

    random_state : int, RandomState, or None
        Random state for reproducibility.

    pre_dispatch : str
        Pre-dispatch for parallel jobs.

    error_score : float or 'raise'
        Value to assign if fitting fails.

    return_train_score : bool
        Whether to return training scores.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        random_state=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _iter_candidates(self):
        """Iterate over sampled parameter settings."""
        from sklearn.model_selection import ParameterSampler

        return iter(
            ParameterSampler(
                self.param_distributions,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
        )
