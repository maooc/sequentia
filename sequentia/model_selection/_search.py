# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Hyper-parameter search utilities using sklearn's public API."""

from __future__ import annotations

import time
import typing as t
import warnings
from collections import defaultdict
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV
from sklearn.model_selection import RandomizedSearchCV as SklearnRandomizedSearchCV
from sklearn.utils.parallel import Parallel, delayed

from sequentia.model_selection._validation import _fit_and_score, _check_method_params

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


def _check_cv(cv, y=None, classifier=False):
    """Check and build a cross-validator using sklearn's public API.
    
    This is a compatibility wrapper that creates cross-validators
    without relying on sklearn internal APIs.
    """
    from sklearn.model_selection import (
        KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit,
        LeaveOneOut, LeavePOut, StratifiedGroupKFold, GroupKFold,
        TimeSeriesSplit
    )
    
    if cv is None:
        cv = 5
    
    if isinstance(cv, int):
        if classifier and y is not None:
            cv = StratifiedKFold(cv)
        else:
            cv = KFold(cv)
    
    return cv


def _warn_or_raise_about_fit_failures(results, error_score):
    """Warn or raise errors for fit failures during cross-validation.
    
    This is a compatibility wrapper to handle fit failures without
    relying on sklearn internal APIs.
    """
    fit_errors = [result.get("fit_error") for result in results if result.get("fit_error") is not None]
    
    if fit_errors:
        n_failures = len(fit_errors)
        n_total = len(results)
        
        if error_score == "raise":
            # Re-raise the first error
            raise RuntimeError(f"Fit failed for {n_failures}/{n_total} splits. First error:\n{fit_errors[0]}")
        else:
            warnings.warn(
                f"Fit failed for {n_failures}/{n_total} splits. "
                f"Scores for these splits will be set to {error_score}. "
                f"First error:\n{fit_errors[0]}",
                UserWarning
            )


def _insert_error_scores(results, error_score):
    """Insert error scores into results for failed fits.
    
    This is a compatibility wrapper to handle error score insertion
    without relying on sklearn internal APIs.
    """
    for result in results:
        if result.get("fit_error") is not None:
            if isinstance(error_score, dict):
                result["test_scores"] = {name: error_score for name in error_score}
            else:
                result["test_scores"] = error_score


class BaseSearchCV(MetaEstimatorMixin, BaseEstimator):
    """Base class for hyper-parameter search with cross-validation.
    
    This class reimplements sklearn's BaseSearchCV functionality
    without relying on sklearn internal APIs, to properly handle
    sequence data with lengths parameter.
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

    def _get_scorers(self):
        """Get scorers based on scoring parameter."""
        if self.scoring is None:
            # Use estimator's default scorer
            if hasattr(self.estimator, 'score'):
                scorers = lambda est, X, y=None, **kwargs: est.score(X, y, **kwargs) if y is not None else est.score(X, **kwargs)
            else:
                raise ValueError("No scoring method specified and estimator has no score method.")
            refit_metric = "score"
        elif isinstance(self.scoring, str):
            scorers = get_scorer(self.scoring)
            refit_metric = self.scoring
        elif callable(self.scoring):
            scorers = self.scoring
            refit_metric = "score"
        elif isinstance(self.scoring, dict):
            scorers = {name: get_scorer(s) if isinstance(s, str) else s for name, s in self.scoring.items()}
            refit_metric = self.refit if isinstance(self.refit, str) else "score"
        else:
            raise ValueError(f"Invalid scoring type: {type(self.scoring)}")
        
        return scorers, refit_metric

    def _get_routed_params_for_fit(self, params):
        """Extract routed parameters for fit method.
        
        This is a simplified version that handles common parameter routing.
        """
        # For simplicity, pass all params to estimator.fit
        class SimpleRouter:
            def __init__(self, params):
                self.estimator = SimpleNamespace(fit=params)
                self.scorer = SimpleNamespace(score={})
                self.splitter = SimpleNamespace(split={})
        
        from types import SimpleNamespace
        return SimpleRouter(params)

    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        """Format results from parallel evaluation."""
        n_candidates = len(candidate_params)
        
        results = {
            "params": candidate_params,
        }
        
        # Extract test scores
        test_scores = [r["test_scores"] for r in out]
        
        # Handle single scorer
        if test_scores and not isinstance(test_scores[0], dict):
            test_scores_arr = np.array(test_scores).reshape(n_candidates, n_splits)
            results["mean_test_score"] = np.mean(test_scores_arr, axis=1)
            results["std_test_score"] = np.std(test_scores_arr, axis=1)
            results["rank_test_score"] = np.argsort(-results["mean_test_score"]) + 1
            for i in range(n_splits):
                results[f"split{i}_test_score"] = test_scores_arr[:, i]
        else:
            # Handle multiple scorers
            scorer_names = list(test_scores[0].keys()) if test_scores else []
            for name in scorer_names:
                scores = [r[name] for r in test_scores]
                scores_arr = np.array(scores).reshape(n_candidates, n_splits)
                results[f"mean_test_{name}"] = np.mean(scores_arr, axis=1)
                results[f"std_test_{name}"] = np.std(scores_arr, axis=1)
                results[f"rank_test_{name}"] = np.argsort(-results[f"mean_test_{name}"]) + 1
                for i in range(n_splits):
                    results[f"split{i}_test_{name}"] = scores_arr[:, i]
        
        # Extract timing info
        if out and "fit_time" in out[0]:
            fit_times = [r["fit_time"] for r in out]
            results["mean_fit_time"] = np.mean(fit_times)
            results["std_fit_time"] = np.std(fit_times)
        
        if out and "score_time" in out[0]:
            score_times = [r["score_time"] for r in out]
            results["mean_score_time"] = np.mean(score_times)
            results["std_score_time"] = np.std(score_times)
        
        return results

    def _select_best_index(self, refit, refit_metric, results):
        """Select the best parameter index based on refit strategy."""
        if isinstance(refit, str):
            metric_key = f"mean_test_{refit}"
        else:
            metric_key = f"mean_test_{refit_metric}"
        
        if metric_key in results:
            return np.argmax(results[metric_key])
        else:
            return 0

    def _check_refit_for_multimetric(self, first_test_score):
        """Check that refit is valid for multimetric scoring."""
        if isinstance(self.refit, str):
            if self.refit not in first_test_score:
                raise ValueError(
                    f"For multi-metric scoring, refit must be one of the scorer names. "
                    f"Got {self.refit}, available: {list(first_test_score.keys())}"
                )

    def _run_search(self, evaluate_candidates):
        """Run the search with the given evaluation function.
        
        Subclasses must implement this method.
        """
        raise NotImplementedError("_run_search must be implemented by subclasses")

    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target relative to X for classification or regression.

        **params : dict of str -> object
            Parameters passed to the fit method.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        scorers, refit_metric = self._get_scorers()

        # Validate parameters
        params = _check_method_params(X, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        cv = _check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv.get_n_splits(X, y)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.estimator.fit,
            score_params=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(
                candidate_params, cv_iter=None, more_results=None
            ):
                cv_iter = cv_iter or cv
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (
                        split_idx,
                        (train, test),
                    ) in product(
                        enumerate(candidate_params),
                        enumerate(cv_iter.split(X, y)),
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        f"inconsistent results. Expected {n_splits} "
                        f"splits, got {len(out) // n_candidates}"
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # here we clone the estimator as well as the parameters, since
            # sometimes the parameters themselves might be estimators, e.g.
            # when we search over different estimators in a pipeline.
            self.best_estimator_ = clone(base_estimator).set_params(
                **clone(self.best_params_, safe=False)
            )

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    @property
    def _estimator_type(self):
        """Return the estimator type based on the base estimator."""
        if hasattr(self.estimator, '_estimator_type'):
            return self.estimator._estimator_type
        return None


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.
    
    This class extends sklearn's GridSearchCV to properly handle
    sequence data with lengths parameter.
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
            estimator=estimator,
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

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid."""
        from sklearn.model_selection import ParameterGrid
        
        param_grid = self.param_grid
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        
        evaluate_candidates(ParameterGrid(param_grid))


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.
    
    This class extends sklearn's RandomizedSearchCV to properly handle
    sequence data with lengths parameter.
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
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
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

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions."""
        from sklearn.model_selection import ParameterSampler
        
        evaluate_candidates(ParameterSampler(
            self.param_distributions,
            self.n_iter,
            random_state=self.random_state,
        ))
