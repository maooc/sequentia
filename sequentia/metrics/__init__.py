# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware evaluation metrics.

This module provides metrics that automatically align predictions with
true values based on sequence lengths, eliminating the need for manual
alignment in evaluation pipelines.

Key Features
------------
1. Automatic alignment: Predictions and true values are aligned using lengths
2. sklearn-compatible: Works with sklearn's scoring infrastructure
3. Per-sequence and aggregate metrics: Support for both granular and overall scores
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from sequentia._internal import _sequence
from sequentia._internal._typing import Array, IntArray

__all__ = [
    "sequence_accuracy_score",
    "sequence_f1_score",
    "sequence_precision_score",
    "sequence_recall_score",
    "sequence_mean_squared_error",
    "sequence_mean_absolute_error",
    "sequence_r2_score",
    "make_sequence_scorer",
    "align_predictions",
]


def align_predictions(
    y_true: Array,
    y_pred: Array,
    lengths: IntArray,
) -> tuple[Array, Array]:
    """Align predictions with true values based on sequence lengths.

    This function handles cases where predictions might be:
    - Per-sequence (one prediction per sequence)
    - Per-observation (one prediction per observation point)

    For per-sequence predictions, the true values are expected to be
    per-sequence as well. For per-observation predictions, both are
    expected to have the same total length.

    Parameters
    ----------
    y_true : Array
        True values. Either per-sequence or per-observation.
    y_pred : Array
        Predicted values. Either per-sequence or per-observation.
    lengths : IntArray
        Lengths of each sequence.

    Returns
    -------
    tuple[Array, Array]
        Aligned (y_true, y_pred) suitable for metric computation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lengths = np.asarray(lengths)

    n_sequences = len(lengths)
    total_observations = np.sum(lengths)

    if len(y_true) == n_sequences and len(y_pred) == n_sequences:
        return y_true, y_pred

    if len(y_true) == total_observations and len(y_pred) == total_observations:
        return y_true, y_pred

    if len(y_true) == n_sequences and len(y_pred) == total_observations:
        y_true_aligned = np.empty(total_observations, dtype=y_true.dtype)
        offset = 0
        for i, length in enumerate(lengths):
            y_true_aligned[offset : offset + length] = y_true[i]
            offset += length
        return y_true_aligned, y_pred

    if len(y_true) == total_observations and len(y_pred) == n_sequences:
        y_pred_aligned = np.empty(total_observations, dtype=y_pred.dtype)
        offset = 0
        for i, length in enumerate(lengths):
            y_pred_aligned[offset : offset + length] = y_pred[i]
            offset += length
        return y_true, y_pred_aligned

    raise ValueError(
        f"Cannot align predictions: y_true has {len(y_true)} elements, "
        f"y_pred has {len(y_pred)} elements, "
        f"n_sequences={n_sequences}, total_observations={total_observations}"
    )


def sequence_accuracy_score(
    y_true: Array,
    y_pred: Array,
    *,
    lengths: IntArray,
    normalize: bool = True,
    sample_weight: Array | None = None,
) -> float:
    """Accuracy classification score for sequence data.

    This function automatically aligns predictions with true values
    based on sequence lengths.

    Parameters
    ----------
    y_true : Array
        Ground truth (correct) labels.
    y_pred : Array
        Predicted labels, as returned by a classifier.
    lengths : IntArray
        Lengths of each sequence.
    normalize : bool, default=True
        If False, return the number of correctly classified samples.
    sample_weight : Array | None, default=None
        Sample weights.

    Returns
    -------
    float
        Accuracy score.
    """
    y_true_aligned, y_pred_aligned = align_predictions(y_true, y_pred, lengths)
    return accuracy_score(
        y_true_aligned,
        y_pred_aligned,
        normalize=normalize,
        sample_weight=sample_weight,
    )


def sequence_precision_score(
    y_true: Array,
    y_pred: Array,
    *,
    lengths: IntArray,
    labels: Array | None = None,
    pos_label: int | str = 1,
    average: str = "binary",
    sample_weight: Array | None = None,
    zero_division: str | float = "warn",
) -> float:
    """Compute precision score for sequence data.

    Parameters
    ----------
    y_true : Array
        Ground truth (correct) labels.
    y_pred : Array
        Predicted labels.
    lengths : IntArray
        Lengths of each sequence.
    labels : Array | None, default=None
        The set of labels to include.
    pos_label : int | str, default=1
        The class to report if average='binary'.
    average : str, default='binary'
        Averaging method: 'micro', 'macro', 'weighted', or 'binary'.
    sample_weight : Array | None, default=None
        Sample weights.
    zero_division : str | float, default='warn'
        Value to return when there is a zero division.

    Returns
    -------
    float
        Precision score.
    """
    y_true_aligned, y_pred_aligned = align_predictions(y_true, y_pred, lengths)
    return precision_score(
        y_true_aligned,
        y_pred_aligned,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def sequence_recall_score(
    y_true: Array,
    y_pred: Array,
    *,
    lengths: IntArray,
    labels: Array | None = None,
    pos_label: int | str = 1,
    average: str = "binary",
    sample_weight: Array | None = None,
    zero_division: str | float = "warn",
) -> float:
    """Compute recall score for sequence data.

    Parameters
    ----------
    y_true : Array
        Ground truth (correct) labels.
    y_pred : Array
        Predicted labels.
    lengths : IntArray
        Lengths of each sequence.
    labels : Array | None, default=None
        The set of labels to include.
    pos_label : int | str, default=1
        The class to report if average='binary'.
    average : str, default='binary'
        Averaging method: 'micro', 'macro', 'weighted', or 'binary'.
    sample_weight : Array | None, default=None
        Sample weights.
    zero_division : str | float, default='warn'
        Value to return when there is a zero division.

    Returns
    -------
    float
        Recall score.
    """
    y_true_aligned, y_pred_aligned = align_predictions(y_true, y_pred, lengths)
    return recall_score(
        y_true_aligned,
        y_pred_aligned,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def sequence_f1_score(
    y_true: Array,
    y_pred: Array,
    *,
    lengths: IntArray,
    labels: Array | None = None,
    pos_label: int | str = 1,
    average: str = "binary",
    sample_weight: Array | None = None,
    zero_division: str | float = "warn",
) -> float:
    """Compute F1 score for sequence data.

    Parameters
    ----------
    y_true : Array
        Ground truth (correct) labels.
    y_pred : Array
        Predicted labels.
    lengths : IntArray
        Lengths of each sequence.
    labels : Array | None, default=None
        The set of labels to include.
    pos_label : int | str, default=1
        The class to report if average='binary'.
    average : str, default='binary'
        Averaging method: 'micro', 'macro', 'weighted', or 'binary'.
    sample_weight : Array | None, default=None
        Sample weights.
    zero_division : str | float, default='warn'
        Value to return when there is a zero division.

    Returns
    -------
    float
        F1 score.
    """
    y_true_aligned, y_pred_aligned = align_predictions(y_true, y_pred, lengths)
    return f1_score(
        y_true_aligned,
        y_pred_aligned,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def sequence_mean_squared_error(
    y_true: Array,
    y_pred: Array,
    *,
    lengths: IntArray,
    sample_weight: Array | None = None,
    multioutput: str = "uniform_average",
    squared: bool = True,
) -> float:
    """Mean squared error regression loss for sequence data.

    Parameters
    ----------
    y_true : Array
        Ground truth (correct) target values.
    y_pred : Array
        Estimated target values.
    lengths : IntArray
        Lengths of each sequence.
    sample_weight : Array | None, default=None
        Sample weights.
    multioutput : str, default='uniform_average'
        Defines aggregating of multiple output values.
    squared : bool, default=True
        If True, returns MSE. If False, returns RMSE.

    Returns
    -------
    float
        Mean squared error.
    """
    y_true_aligned, y_pred_aligned = align_predictions(y_true, y_pred, lengths)
    return mean_squared_error(
        y_true_aligned,
        y_pred_aligned,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared,
    )


def sequence_mean_absolute_error(
    y_true: Array,
    y_pred: Array,
    *,
    lengths: IntArray,
    sample_weight: Array | None = None,
    multioutput: str = "uniform_average",
) -> float:
    """Mean absolute error regression loss for sequence data.

    Parameters
    ----------
    y_true : Array
        Ground truth (correct) target values.
    y_pred : Array
        Estimated target values.
    lengths : IntArray
        Lengths of each sequence.
    sample_weight : Array | None, default=None
        Sample weights.
    multioutput : str, default='uniform_average'
        Defines aggregating of multiple output values.

    Returns
    -------
    float
        Mean absolute error.
    """
    y_true_aligned, y_pred_aligned = align_predictions(y_true, y_pred, lengths)
    return mean_absolute_error(
        y_true_aligned,
        y_pred_aligned,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


def sequence_r2_score(
    y_true: Array,
    y_pred: Array,
    *,
    lengths: IntArray,
    sample_weight: Array | None = None,
    multioutput: str = "uniform_average",
) -> float:
    """R² (coefficient of determination) regression score for sequence data.

    Parameters
    ----------
    y_true : Array
        Ground truth (correct) target values.
    y_pred : Array
        Estimated target values.
    lengths : IntArray
        Lengths of each sequence.
    sample_weight : Array | None, default=None
        Sample weights.
    multioutput : str, default='uniform_average'
        Defines aggregating of multiple output values.

    Returns
    -------
    float
        R² score.
    """
    y_true_aligned, y_pred_aligned = align_predictions(y_true, y_pred, lengths)
    return r2_score(
        y_true_aligned,
        y_pred_aligned,
        sample_weight=sample_weight,
        multioutput=multioutput,
    )


def make_sequence_scorer(
    metric_func: t.Callable,
    **kwargs: t.Any,
) -> t.Callable:
    """Make a sequence-aware scorer from a metric function.

    This creates a scorer that automatically handles the `lengths`
    parameter for sequence data.

    Parameters
    ----------
    metric_func : callable
        A metric function that takes y_true, y_pred, and lengths.
    **kwargs : dict
        Additional keyword arguments to pass to the metric function.

    Returns
    -------
    callable
        A scorer function compatible with sklearn's scoring infrastructure.

    Examples
    --------
    >>> from sequentia.metrics import make_sequence_scorer, sequence_accuracy_score
    >>> from sklearn.model_selection import cross_validate
    >>>
    >>> scorer = make_sequence_scorer(sequence_accuracy_score)
    >>> results = cross_validate(estimator, X, y, scoring=scorer, fit_params={'lengths': lengths})
    """
    def scorer(estimator, X, y, lengths=None, **score_kwargs):
        if lengths is None:
            lengths = np.array([len(X)])
        y_pred = estimator.predict(X, lengths=lengths)
        return metric_func(y, y_pred, lengths=lengths, **kwargs, **score_kwargs)

    return scorer
