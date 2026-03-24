# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware hyper-parameter search using scikit-learn's public API only.

This module provides GridSearchCV and RandomizedSearchCV that work with
sequence data through scikit-learn's metadata routing mechanism.
"""

from itertools import product
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid, ParameterSampler

from sequentia._internal._typing import Array, IntArray
from sequentia.model_selection._sequence_dataset import SequenceCVSplitter

__all__ = ["GridSearchCV", "RandomizedSearchCV", "param_grid"]


def param_grid(**kwargs):
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


def _is_pipeline(estimator):
    """Check if estimator is a sklearn Pipeline."""
    return hasattr(estimator, 'steps') and hasattr(estimator, 'named_steps')


def _get_step_params(params, step_name):
    """Extract parameters for a specific step from param dict."""
    prefix = f"{step_name}__"
    step_params = {}
    for key, value in params.items():
        if key.startswith(prefix):
            step_params[key[len(prefix):]] = value
    return step_params


def _set_params(estimator, **params):
    """Set parameters on estimator, handling nested pipelines."""
    if _is_pipeline(estimator):
        # For Pipeline, set params on steps
        for step_name, step in estimator.named_steps.items():
            step_params = _get_step_params(params, step_name)
            if step_params:
                step.set_params(**step_params)
        return estimator
    else:
        # Regular estimator
        return estimator.set_params(**params)


def _fit_with_params(estimator, X, y, lengths, **fit_params):
    """Fit estimator with parameters, handling pipelines."""
    if _is_pipeline(estimator):
        # For Pipeline, manually apply each step
        Xt = X
        for step_name, step in estimator.steps[:-1]:
            # Fit and transform preprocessing steps
            if hasattr(step, 'fit'):
                step.fit(Xt, y, lengths=lengths)
            if hasattr(step, 'transform'):
                Xt = step.transform(Xt, lengths=lengths)
        
        # Fit final estimator
        final_estimator = estimator.steps[-1][1]
        final_estimator.fit(Xt, y, lengths=lengths, **fit_params)
        return estimator
    else:
        # Regular estimator
        return estimator.fit(X, y, lengths=lengths, **fit_params)


def _transform_with_params(estimator, X, lengths):
    """Transform X through pipeline steps."""
    if _is_pipeline(estimator):
        Xt = X
        for step_name, step in estimator.steps[:-1]:
            if hasattr(step, 'transform'):
                Xt = step.transform(Xt, lengths=lengths)
        return Xt
    return X


def _score_with_params(estimator, X, y, lengths):
    """Score estimator with parameters, handling pipelines."""
    if _is_pipeline(estimator):
        Xt = _transform_with_params(estimator, X, lengths)
        final_estimator = estimator.steps[-1][1]
        return final_estimator.score(Xt, y, lengths=lengths)
    else:
        return estimator.score(X, y, lengths=lengths)


def _predict_with_params(estimator, X, lengths):
    """Predict with estimator, handling pipelines."""
    if _is_pipeline(estimator):
        Xt = _transform_with_params(estimator, X, lengths)
        final_estimator = estimator.steps[-1][1]
        return final_estimator.predict(Xt, lengths=lengths)
    else:
        return estimator.predict(X, lengths=lengths)


def _predict_proba_with_params(estimator, X, lengths):
    """Predict probabilities with estimator, handling pipelines."""
    if _is_pipeline(estimator):
        Xt = _transform_with_params(estimator, X, lengths)
        final_estimator = estimator.steps[-1][1]
        return final_estimator.predict_proba(Xt, lengths=lengths)
    else:
        return estimator.predict_proba(X, lengths=lengths)


def _predict_log_proba_with_params(estimator, X, lengths):
    """Predict log probabilities with estimator, handling pipelines."""
    if _is_pipeline(estimator):
        Xt = _transform_with_params(estimator, X, lengths)
        final_estimator = estimator.steps[-1][1]
        return final_estimator.predict_log_proba(Xt, lengths=lengths)
    else:
        return estimator.predict_log_proba(X, lengths=lengths)


class BaseSearchCV(BaseEstimator):
    """Base class for hyper parameter search with cross-validation.
    
    This class implements a sequence-aware version of sklearn's BaseSearchCV
    that properly handles the mismatch between observation-level X and 
    sequence-level y.
    """

    def __init__(
        self,
        estimator,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        return_train_score=False,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.return_train_score = return_train_score

    def _run_search(self, evaluate_candidates):
        """Run the search with the given evaluate_candidates function."""
        raise NotImplementedError("_run_search must be implemented by subclasses")

    def fit(self, X: Array, y: Array, *, lengths: IntArray, groups: Array | None = None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Training data.
        y : array-like of shape (n_sequences,)
            Target values.
        lengths : array-like of shape (n_sequences,)
            Lengths of each sequence.
        groups : array-like of shape (n_sequences,), default=None
            Group labels for the samples used while splitting the dataset.
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        from sequentia.model_selection._sequence_dataset import SequenceDataset
        
        # Create sequence dataset
        self.dataset_ = SequenceDataset(X, y, lengths)
        
        # Set up CV splitter
        if self.cv is None:
            from sequentia.model_selection._split import KFold
            cv = KFold(n_splits=5)
        else:
            cv = self.cv
        
        self.cv_ = SequenceCVSplitter(cv)
        
        # Run the search
        def evaluate_candidates(candidate_params):
            """Evaluate all candidate parameter sets."""
            results = []
            
            for params in candidate_params:
                scores = self._evaluate_params(params, groups)
                results.append({
                    "params": params,
                    "mean_test_score": np.mean(scores),
                    "std_test_score": np.std(scores),
                    "scores": scores,
                })
            
            return results
        
        all_results = self._run_search(evaluate_candidates)
        
        # Store results
        self.cv_results_ = all_results
        
        # Find best params
        best_idx = np.argmax([r["mean_test_score"] for r in all_results])
        self.best_params_ = all_results[best_idx]["params"]
        self.best_score_ = all_results[best_idx]["mean_test_score"]
        
        # Refit with best params if requested
        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            _set_params(self.best_estimator_, **self.best_params_)
            _fit_with_params(self.best_estimator_, X, y, lengths, **fit_params)
        
        return self

    def _evaluate_params(self, params: dict, groups: Array | None = None) -> list[float]:
        """Evaluate a single parameter set across all CV folds."""
        scores = []
        
        for train_data, test_data in self.cv_.split(
            self.dataset_.X, self.dataset_.y, self.dataset_.lengths, groups
        ):
            # Clone and configure estimator
            estimator = clone(self.estimator)
            _set_params(estimator, **params)
            
            # Fit on training data
            _fit_with_params(
                estimator,
                train_data.X,
                train_data.y,
                train_data.lengths,
            )
            
            # Score on test data
            score = _score_with_params(
                estimator,
                test_data.X,
                test_data.y,
                test_data.lengths,
            )
            scores.append(score)
        
        return scores

    @property
    def _estimator_type(self):
        """Return the estimator type."""
        return getattr(self.estimator, "_estimator_type", None)

    def score(self, X, y, *, lengths=None):
        """Return the score on the given data."""
        if hasattr(self, "best_estimator_"):
            return _score_with_params(self.best_estimator_, X, y, lengths)
        raise NotFittedError("This GridSearchCV instance is not fitted yet.")

    def predict(self, X, *, lengths=None):
        """Call predict on the best estimator."""
        if hasattr(self, "best_estimator_"):
            return _predict_with_params(self.best_estimator_, X, lengths)
        raise NotFittedError("This GridSearchCV instance is not fitted yet.")

    def predict_proba(self, X, *, lengths=None):
        """Call predict_proba on the best estimator."""
        if hasattr(self, "best_estimator_"):
            return _predict_proba_with_params(self.best_estimator_, X, lengths)
        raise NotFittedError("This GridSearchCV instance is not fitted yet.")

    def predict_log_proba(self, X, *, lengths=None):
        """Call predict_log_proba on the best estimator."""
        if hasattr(self, "best_estimator_"):
            return _predict_log_proba_with_params(self.best_estimator_, X, lengths)
        raise NotFittedError("This GridSearchCV instance is not fitted yet.")


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    This class implements a sequence-aware version of sklearn's GridSearchCV
    that properly handles the mismatch between observation-level X and 
    sequence-level y.

    Parameters
    ----------
    estimator : estimator object
        The estimator to use for fitting and scoring.
    param_grid : dict or list of dicts
        Dictionary with parameters names as keys and lists of parameter
        settings to try as values.
    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.
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
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid."""
        param_grid = ParameterGrid(self.param_grid)
        return evaluate_candidates(param_grid)


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    This class implements a sequence-aware version of sklearn's RandomizedSearchCV
    that properly handles the mismatch between observation-level X and 
    sequence-level y.

    Parameters
    ----------
    estimator : estimator object
        The estimator to use for fitting and scoring.
    param_distributions : dict or list of dicts
        Dictionary with parameters names as keys and distributions or lists of
        parameters to try.
    n_iter : int, default=10
        Number of parameter settings that are sampled.
    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values.
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
        return_train_score=False,
        random_state=None,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions."""
        param_iter = ParameterSampler(
            self.param_distributions,
            self.n_iter,
            random_state=self.random_state,
        )
        return evaluate_candidates(param_iter)


class NotFittedError(Exception):
    """Exception raised when an estimator is not fitted."""
    pass
