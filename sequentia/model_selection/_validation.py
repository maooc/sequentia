# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation utilities using sklearn's public API."""

from __future__ import annotations

import time
import typing as t
from traceback import format_exc

import numpy as np
from joblib import logger
from sklearn.base import clone
from sklearn.metrics import check_scoring, get_scorer
from sklearn.utils.parallel import Parallel, delayed

from sequentia._internal import _data

__all__ = ["_fit_and_score", "cross_val_score"]


def _check_method_params(X, params, indices=None):
    """Check and adjust method parameters for cross-validation.
    
    This is a compatibility wrapper that handles parameter validation
    without relying on sklearn internal APIs.
    """
    if params is None:
        return {}
    
    result = {}
    for key, value in params.items():
        if value is None:
            result[key] = None
        elif hasattr(value, '__len__') and len(value) == len(X):
            # Parameter is sample-aligned, slice it
            if indices is not None:
                result[key] = value[indices] if isinstance(value, np.ndarray) else [value[i] for i in indices]
            else:
                result[key] = value
        else:
            # Parameter is not sample-aligned, pass as-is
            result[key] = value
    
    return result


def _num_samples(X):
    """Return the number of samples in an array-like."""
    if hasattr(X, '__len__'):
        return len(X)
    return X.shape[0]


def _score(estimator, X_test, y_test, scorer, score_params, error_score):
    """Compute the score(s) of an estimator on a given test set.
    
    This is a compatibility wrapper that handles scoring without relying
    on sklearn internal APIs.
    """
    try:
        if y_test is None:
            score = scorer(estimator, X_test, **score_params)
        else:
            score = scorer(estimator, X_test, y_test, **score_params)
    except Exception:
        if error_score == "raise":
            raise
        else:
            if isinstance(scorer, dict):
                score = {name: error_score for name in scorer}
            else:
                score = error_score
    return score


def _fit_and_score(
    estimator,
    X,
    y,
    *,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    score_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    """Fit estimator and compute scores for a given dataset split.
    
    This function is a wrapper around sklearn's validation logic that
    properly handles sequence data with lengths parameter.
    """
    # Ensure train/test are arrays
    train = np.asarray(train)
    test = np.asarray(test)

    if not isinstance(error_score, (int, float, np.number)) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    # Build progress message
    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += (
                f"; {candidate_progress[0]+1}/{candidate_progress[1]}"
            )

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Extract and validate lengths parameter
    lengths = fit_params.get("lengths") if fit_params else None
    
    # Validate fit_params and score_params
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    score_params = score_params if score_params is not None else {}
    score_params_train = _check_method_params(
        X, params=score_params, indices=train
    )
    score_params_test = _check_method_params(
        X, params=score_params, indices=test
    )

    if parameters is not None:
        estimator = estimator.set_params(**clone(parameters, safe=False))

    start_time = time.time()

    # Handle sequence data splitting with lengths
    if lengths is not None:
        idxs = _data.get_idxs(lengths)
        idxs_train, idxs_test = idxs[train], idxs[test]
        y_train, y_test = y[train], y[test]
        lengths_train, lengths_test = lengths[train], lengths[test]
        
        # Use list-based approach to avoid unnecessary stacking
        X_train_list = list(_data.iter_X(X, idxs=idxs_train))
        X_test_list = list(_data.iter_X(X, idxs=idxs_test))
        X_train = np.concatenate(X_train_list) if X_train_list else np.array([])
        X_test = np.concatenate(X_test_list) if X_test_list else np.array([])
        
        fit_params["lengths"] = lengths_train
        score_params_train["lengths"] = lengths_train
        score_params_test["lengths"] = lengths_test
    else:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, (int, float, np.number)):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(
            estimator, X_test, y_test, scorer, score_params_test, error_score
        )
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(
                estimator,
                X_train,
                y_train,
                scorer,
                score_params_train,
                error_score,
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += (
                        f"(train={train_scores:.3f}, test={test_scores:.3f})"
                    )
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def cross_val_score(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    score_params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
):
    """Evaluate a score by cross-validation.
    
    This is a wrapper around sklearn's cross_val_score that properly
    handles sequence data with lengths parameter.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be a sequence dataset with lengths.
    y : array-like, optional
        The target variable to try to predict.
    groups : array-like, optional
        Group labels for the samples used while splitting the dataset.
    scoring : str or callable, optional
        A str or a scorer callable object / function.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    n_jobs : int, optional
        Number of jobs to run in parallel.
    verbose : int, optional
        The verbosity level.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    score_params : dict, optional
        Parameters to pass to the score method of the estimator.
    pre_dispatch : int or str, optional
        Controls the number of jobs that get dispatched during parallel execution.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        
    Returns
    -------
    scores : ndarray of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """
    from sklearn.base import is_classifier
    from sklearn.model_selection import KFold, StratifiedKFold
    
    # Build cross-validator without relying on sklearn's private check_cv
    if cv is None:
        cv = 5
    
    if isinstance(cv, int):
        if is_classifier(estimator) and y is not None:
            cv = StratifiedKFold(cv)
        else:
            cv = KFold(cv)
    
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    
    fit_params = fit_params if fit_params is not None else {}
    score_params = score_params if score_params is not None else {}
    
    # Get scorer using sklearn's public API
    if scoring is None:
        # Use estimator's default scorer
        if hasattr(estimator, 'score'):
            scorer = lambda est, X, y=None, **kwargs: est.score(X, y, **kwargs) if y is not None else est.score(X, **kwargs)
        else:
            raise ValueError("No scoring method specified and estimator has no score method.")
    elif isinstance(scoring, str):
        scorer = get_scorer(scoring)
    elif callable(scoring):
        scorer = scoring
    elif isinstance(scoring, dict):
        scorers = {name: get_scorer(s) if isinstance(s, str) else s for name, s in scoring.items()}
        scorer = scorers
    else:
        raise ValueError(f"Invalid scoring type: {type(scoring)}")
    
    scores = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorer=scorer,
            train=train,
            test=test,
            verbose=verbose,
            parameters=None,
            fit_params=fit_params,
            score_params=score_params,
            return_train_score=False,
            return_n_test_samples=False,
            return_times=False,
            return_estimator=False,
            split_progress=(split_idx, cv.get_n_splits(X, y, groups)),
            error_score=error_score,
        )
        for split_idx, (train, test) in enumerate(cv.split(X, y, groups))
    )
    
    return np.array([score["test_scores"] for score in scores])
