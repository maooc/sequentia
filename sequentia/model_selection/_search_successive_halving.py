# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware successive halving search utilities.

This module provides sequence-aware alternatives to sklearn's successive halving
search classes, designed to work seamlessly with sequence data.

Note
----
These classes are sequence-aware versions of sklearn's HalvingGridSearchCV and
HalvingRandomSearchCV, using sequentia's cross-validation splitters.
"""

from __future__ import annotations

import numbers
import time
import typing as t
from itertools import product

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import ParameterGrid, ParameterSampler
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from sequentia.model_selection._validation import _fit_and_score

__all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]


class BaseHalvingSearchCV:
    """Base class for halving search with sequence data support."""

    _required_parameters = ["estimator"]

    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        error_score=np.nan,
        return_train_score=True,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        random_state=None,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.factor = factor
        self.resource = resource
        self.max_resources = max_resources
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination
        self.random_state = random_state

    def _check_input(self, X, y, params):
        """Validate input parameters."""
        fit_params = params.copy() if params else {}
        return fit_params

    def _get_scorer(self):
        """Get scorer from scoring parameter."""
        from sklearn.metrics import check_scoring

        return check_scoring(self.estimator, scoring=self.scoring)

    def fit(self, X, y=None, *, lengths=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Training vectors.

        y : array-like of shape (n_sequences,), default=None
            Target relative to X.

        lengths : array-like of shape (n_sequences,), default=None
            Lengths of each sequence in X.

        **params : dict
            Parameters passed to the fit method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        scorer = self._get_scorer()
        fit_params = self._check_input(X, y, params)
        if lengths is not None:
            fit_params["lengths"] = lengths

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        n_splits = cv.get_n_splits(X, y)

        n_sequences = len(lengths) if lengths is not None else len(y) if y is not None else X.shape[0]

        if self.max_resources == "auto":
            max_resources = n_sequences
        else:
            max_resources = self.max_resources

        if self.min_resources == "exhaust":
            min_resources = self._compute_min_resources(max_resources, n_sequences)
        else:
            min_resources = self.min_resources

        base_estimator = clone(self.estimator)
        candidate_params = list(self._iter_candidates())

        if self.verbose > 0:
            print(f"Fitting {n_splits} folds for each of {len(candidate_params)} candidates...")

        results = []
        n_candidates = len(candidate_params)
        resources = min_resources

        while True:
            n_candidates_to_keep = max(1, int(np.floor(n_candidates / self.factor)))

            parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

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
                        enumerate(candidate_params[:n_candidates]),
                        enumerate(cv.split(X, y)),
                    )
                )

            scores = [r["test_scores"] for r in out]
            mean_scores = np.array(scores).reshape(n_candidates, n_splits).mean(axis=1)

            best_indices = np.argsort(mean_scores)[::-1][:n_candidates_to_keep]

            for i, result in enumerate(out[:n_candidates]):
                result["resources"] = resources
                results.append(result)

            candidate_params = [candidate_params[i] for i in best_indices]
            n_candidates = len(candidate_params)

            if resources >= max_resources or n_candidates <= 1:
                break

            resources = min(resources * self.factor, max_resources)

        self.cv_results_ = self._format_results(results)
        self.n_splits_ = n_splits
        self.scorer_ = scorer

        if self.refit and len(candidate_params) > 0:
            best_index = np.argmax(mean_scores[best_indices[0] : best_indices[0] + 1])
            self.best_index_ = best_indices[0]
            self.best_score_ = mean_scores[self.best_index_]
            self.best_params_ = candidate_params[0]

            self.best_estimator_ = clone(base_estimator).set_params(**clone(self.best_params_, safe=False))

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            self.refit_time_ = time.time() - refit_start_time

        return self

    def _compute_min_resources(self, max_resources, n_sequences):
        """Compute minimum resources for first iteration."""
        resources = 1
        while resources * self.factor <= max_resources:
            resources *= self.factor
        return max(1, resources // self.factor)

    def _format_results(self, results):
        """Format the results dictionary."""
        n_results = len(results)

        cv_results = {
            "mean_test_score": np.array([r["test_scores"] for r in results]),
            "std_test_score": np.zeros(n_results),
            "rank_test_score": np.zeros(n_results, dtype=int),
            "mean_fit_time": np.array([r["fit_time"] for r in results]),
            "std_fit_time": np.zeros(n_results),
            "mean_score_time": np.array([r["score_time"] for r in results]),
            "std_score_time": np.zeros(n_results),
            "params": [r["parameters"] for r in results],
            "resources": np.array([r.get("resources", 1) for r in results]),
        }

        if self.return_train_score:
            cv_results["mean_train_score"] = np.array([r.get("train_scores", np.nan) for r in results])
            cv_results["std_train_score"] = np.zeros(n_results)

        return cv_results

    def predict(self, X, *, lengths=None, **params):
        """Call predict on the estimator with the best found parameters."""
        check_is_fitted(self)
        predict_params = params.copy() if params else {}
        if lengths is not None:
            predict_params["lengths"] = lengths
        return self.best_estimator_.predict(X, **predict_params)

    def predict_proba(self, X, *, lengths=None, **params):
        """Call predict_proba on the estimator with the best found parameters."""
        check_is_fitted(self)
        predict_params = params.copy() if params else {}
        if lengths is not None:
            predict_params["lengths"] = lengths
        return self.best_estimator_.predict_proba(X, **predict_params)

    def predict_log_proba(self, X, *, lengths=None, **params):
        """Call predict_log_proba on the estimator with the best found parameters."""
        check_is_fitted(self)
        predict_params = params.copy() if params else {}
        if lengths is not None:
            predict_params["lengths"] = lengths
        return self.best_estimator_.predict_log_proba(X, **predict_params)

    def score(self, X, y=None, *, lengths=None, **params):
        """Return the score on the given data."""
        check_is_fitted(self)
        score_params = params.copy() if params else {}
        if lengths is not None:
            score_params["lengths"] = lengths
        return self.best_estimator_.score(X, y, **score_params)


class HalvingGridSearchCV(BaseHalvingSearchCV):
    """Search over specified parameter values with successive halving.

    This class provides a sequence-aware version of sklearn's HalvingGridSearchCV
    that properly handles sequence data.

    Parameters
    ----------
    estimator : estimator object
        The estimator to optimize.

    param_grid : dict
        Parameter grid to search.

    factor : int or float, default=3
        The 'halving' parameter.

    resource : str, default='n_samples'
        Defines the resource which increases with each iteration.

    max_resources : int, default='auto'
        The maximum amount of resource.

    min_resources : int or 'exhaust', default='exhaust'
        The minimum amount of resource.

    aggressive_elimination : bool, default=False
        Whether to use aggressive elimination.

    cv : cross-validator
        Cross-validation splitter. Should be from sequentia.model_selection.

    scoring : str, callable, or None
        Scoring metric to use.

    n_jobs : int or None
        Number of parallel jobs.

    refit : bool
        Whether to refit the best estimator.

    verbose : int
        Verbosity level.

    random_state : int, RandomState, or None
        Random state for reproducibility.

    error_score : float or 'raise'
        Value to assign if fitting fails.

    return_train_score : bool
        Whether to return training scores.

    Examples
    --------
    >>> from sequentia.model_selection import HalvingGridSearchCV, StratifiedKFold
    >>> from sequentia.models import KNNClassifier
    >>> from sequentia.datasets import load_digits
    >>>
    >>> data = load_digits()
    >>> cv = StratifiedKFold(n_splits=3)
    >>> search = HalvingGridSearchCV(
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
        error_score=np.nan,
        return_train_score=True,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
            factor=factor,
            resource=resource,
            max_resources=max_resources,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
            random_state=random_state,
        )
        self.param_grid = param_grid

    def _iter_candidates(self):
        """Iterate over candidate parameter settings."""
        return ParameterGrid(self.param_grid)


class HalvingRandomSearchCV(BaseHalvingSearchCV):
    """Randomized search with successive halving.

    This class provides a sequence-aware version of sklearn's HalvingRandomSearchCV
    that properly handles sequence data.

    Parameters
    ----------
    estimator : estimator object
        The estimator to optimize.

    param_distributions : dict
        Parameter distributions to sample from.

    n_candidates : int, default='exhaust'
        Number of candidate parameter settings to sample.

    factor : int or float, default=3
        The 'halving' parameter.

    resource : str, default='n_samples'
        Defines the resource which increases with each iteration.

    max_resources : int, default='auto'
        The maximum amount of resource.

    min_resources : int or 'exhaust', default='exhaust'
        The minimum amount of resource.

    aggressive_elimination : bool, default=False
        Whether to use aggressive elimination.

    cv : cross-validator
        Cross-validation splitter. Should be from sequentia.model_selection.

    scoring : str, callable, or None
        Scoring metric to use.

    n_jobs : int or None
        Number of parallel jobs.

    refit : bool
        Whether to refit the best estimator.

    verbose : int
        Verbosity level.

    random_state : int, RandomState, or None
        Random state for reproducibility.

    error_score : float or 'raise'
        Value to assign if fitting fails.

    return_train_score : bool
        Whether to return training scores.

    Examples
    --------
    >>> from sequentia.model_selection import HalvingRandomSearchCV, StratifiedKFold
    >>> from sequentia.models import KNNClassifier
    >>> from sequentia.datasets import load_digits
    >>>
    >>> data = load_digits()
    >>> cv = StratifiedKFold(n_splits=3)
    >>> search = HalvingRandomSearchCV(
    ...     KNNClassifier(),
    ...     param_distributions={"k": [1, 3, 5, 7, 9]},
    ...     cv=cv,
    ... )
    >>> search.fit(data.X, data.y, lengths=data.lengths)
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_candidates="exhaust",
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        error_score=np.nan,
        return_train_score=True,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
            factor=factor,
            resource=resource,
            max_resources=max_resources,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
            random_state=random_state,
        )
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates

    def _iter_candidates(self):
        """Iterate over candidate parameter settings."""
        n_candidates = self.n_candidates
        if n_candidates == "exhaust":
            n_candidates = 10
        return ParameterSampler(self.param_distributions, n_candidates, random_state=self.random_state)
