# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware model validation utilities.

This module provides sequence-aware alternatives to sklearn's validation
utilities, designed to work seamlessly with sklearn's cross-validation
and hyperparameter search infrastructure.

Key Design Principles
---------------------
1. Use sklearn public APIs only - no private module imports
2. Only implement sequence-specific logic
3. Metadata routing: Support sklearn's metadata routing for `lengths`
"""

from __future__ import annotations

import numbers
import time
from traceback import format_exc

import numpy as np
from joblib import logger
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.utils.parallel import Parallel, delayed

from sequentia._internal import _sequence

__all__ = ["_fit_and_score", "prepare_sequence_split", "cross_validate"]


def prepare_sequence_split(
    X: np.ndarray,
    y: np.ndarray | None,
    lengths: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict:
    """Prepare sequence data for a train/test split.

    This function efficiently extracts train and test subsets from
    concatenated sequence data without unnecessary copying.

    Parameters
    ----------
    X : np.ndarray
        Concatenated observation sequences.
    y : np.ndarray | None
        Labels for each sequence.
    lengths : np.ndarray
        Lengths of each sequence.
    train_indices : np.ndarray
        Indices of sequences for training.
    test_indices : np.ndarray
        Indices of sequences for testing.

    Returns
    -------
    dict
        Dictionary containing prepared data for fitting and scoring.
    """
    (X_train, lengths_train, y_train), (X_test, lengths_test, y_test) = _sequence.split_sequences(
        X, lengths, y, train_indices, test_indices
    )

    return {
        "train": {"X": X_train, "y": y_train, "lengths": lengths_train},
        "test": {"X": X_test, "y": y_test, "lengths": lengths_test},
    }


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

    This is a sequence-aware wrapper that:
    1. Extracts the `lengths` parameter from fit_params
    2. Splits the data at the sequence level
    3. Updates fit_params and score_params with the correct lengths
    4. Delegates fitting and scoring to the estimator

    Parameters
    ----------
    estimator : estimator object
        The estimator to fit.
    X : array-like
        Concatenated observation sequences.
    y : array-like
        Sequence labels.
    scorer : scorer object
        The scorer to use.
    train : array-like
        Sequence indices for training.
    test : array-like
        Sequence indices for testing.
    verbose : int
        Verbosity level.
    parameters : dict or None
        Parameters to set on the estimator.
    fit_params : dict
        Parameters passed to the fit method.
    score_params : dict
        Parameters passed to the score method.
    return_train_score : bool
        Whether to return training scores.
    return_parameters : bool
        Whether to return parameters.
    return_n_test_samples : bool
        Whether to return number of test samples.
    return_times : bool
        Whether to return fit and score times.
    return_estimator : bool
        Whether to return the fitted estimator.
    split_progress : tuple or None
        Progress information for the split.
    candidate_progress : tuple or None
        Progress information for the candidate.
    error_score : float or 'raise'
        Value to assign to the score if an error occurs.

    Returns
    -------
    dict
        Dictionary containing scores, times, and other requested information.
    """
    train = np.asarray(train)
    test = np.asarray(test)

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    lengths = fit_params.get("lengths") if fit_params else None
    if lengths is None:
        lengths = np.array([len(X)])

    fit_params = dict(fit_params) if fit_params else {}
    score_params = dict(score_params) if score_params else {}

    if parameters is not None:
        estimator = estimator.set_params(**clone(parameters, safe=False))

    start_time = time.time()

    split_data = prepare_sequence_split(X, y, lengths, train, test)

    X_train = split_data["train"]["X"]
    y_train = split_data["train"]["y"]
    lengths_train = split_data["train"]["lengths"]

    X_test = split_data["test"]["X"]
    y_test = split_data["test"]["y"]
    lengths_test = split_data["test"]["lengths"]

    fit_params["lengths"] = lengths_train
    score_params_train = dict(score_params)
    score_params_train["lengths"] = lengths_train
    score_params_test = dict(score_params)
    score_params_test["lengths"] = lengths_test

    result = {}
    fit_error = None
    test_scores = error_score
    train_scores = error_score

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        fit_time = time.time() - start_time
        score_time = 0.0
        fit_error = format_exc()
        if error_score == "raise":
            raise
    else:
        fit_time = time.time() - start_time

        try:
            if hasattr(scorer, "__call__"):
                test_scores = scorer(estimator, X_test, y_test, **score_params_test)
            else:
                test_scores = scorer(estimator, X_test, y_test, **score_params_test)
        except Exception:
            if error_score == "raise":
                raise
            test_scores = error_score

        score_time = time.time() - start_time - fit_time

        if return_train_score:
            try:
                if hasattr(scorer, "__call__"):
                    train_scores = scorer(estimator, X_train, y_train, **score_params_train)
                else:
                    train_scores = scorer(estimator, X_train, y_train, **score_params_train)
            except Exception:
                if error_score == "raise":
                    raise
                train_scores = error_score

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
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    result["fit_error"] = fit_error
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = len(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def cross_validate(
    estimator,
    X,
    y=None,
    *,
    cv=None,
    scoring=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    score_params=None,
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
):
    """Evaluate metric(s) by cross-validation for sequence data.

    This is a sequence-aware version of sklearn's cross_validate that
    properly handles the `lengths` parameter for sequence data.

    Parameters
    ----------
    estimator : estimator object
        The estimator to fit.
    X : array-like
        Concatenated observation sequences.
    y : array-like
        Sequence labels.
    cv : cross-validation generator or int
        Cross-validation strategy. Should be from sequentia.model_selection.
    scoring : str, callable, list, tuple, or dict
        Scoring metric(s) to use.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : int
        Verbosity level.
    fit_params : dict
        Parameters passed to the fit method. Must include 'lengths'.
    score_params : dict
        Parameters passed to the score method.
    return_train_score : bool
        Whether to return training scores.
    return_estimator : bool
        Whether to return the fitted estimators.
    error_score : float or 'raise'
        Value to assign to the score if an error occurs.

    Returns
    -------
    dict
        Dictionary of arrays containing the scores.
    """
    from sklearn.model_selection import check_cv

    cv = check_cv(cv, y, classifier=False)

    scorer = check_scoring(estimator, scoring=scoring)

    fit_params = fit_params or {}
    score_params = score_params or {}

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

    with parallel:
        out = parallel(
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
                return_train_score=return_train_score,
                return_times=True,
                return_estimator=return_estimator,
                error_score=error_score,
            )
            for train, test in cv.split(X, y)
        )

    results = _aggregate_cv_results(out, return_train_score, return_estimator, error_score)

    return results


def _aggregate_cv_results(all_out, return_train_score, return_estimator, error_score):
    """Aggregate cross-validation results."""
    results = {}

    n_fits = len(all_out)
    n_failed = sum(1 for out in all_out if out["fit_error"] is not None)

    if n_failed == n_fits:
        raise ValueError(
            f"All the {n_fits} fits failed. "
            "It is very likely that your model is misconfigured. "
            "You can try to debug the error by setting error_score='raise'."
        )

    test_scores = [out["test_scores"] for out in all_out]
    if isinstance(test_scores[0], dict):
        for key in test_scores[0]:
            results[f"test_{key}"] = np.array([d[key] for d in test_scores])
    else:
        results["test_score"] = np.array(test_scores)

    if return_train_score:
        train_scores = [out["train_scores"] for out in all_out]
        if isinstance(train_scores[0], dict):
            for key in train_scores[0]:
                results[f"train_{key}"] = np.array([d[key] for d in train_scores])
        else:
            results["train_score"] = np.array(train_scores)

    results["fit_time"] = np.array([out["fit_time"] for out in all_out])
    results["score_time"] = np.array([out["score_time"] for out in all_out])

    if return_estimator:
        results["estimator"] = [out["estimator"] for out in all_out]

    return results
