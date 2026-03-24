# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Metadata routing utilities for sequence data.

This module provides utilities to enable proper metadata routing for sequence
models in scikit-learn pipelines and cross-validation.
"""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray

__all__ = ["enable_sequence_routing"]


def enable_sequence_routing(estimator: BaseEstimator) -> BaseEstimator:
    """Enable metadata routing for sequence data on an estimator.

    This function configures an estimator (or pipeline) to properly route
    the `lengths` metadata through scikit-learn's metadata routing system.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to configure. This can be a single estimator or a Pipeline.

    Returns
    -------
    BaseEstimator
        The configured estimator with metadata routing enabled.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import minmax_scale
    >>> from sequentia.preprocessing import IndependentFunctionTransformer
    >>> from sequentia.models import KNNClassifier
    >>> from sequentia.model_selection import enable_sequence_routing
    >>>
    >>> pipe = Pipeline([
    ...     ('scale', IndependentFunctionTransformer(minmax_scale)),
    ...     ('clf', KNNClassifier(k=1)),
    ... ])
    >>> pipe = enable_sequence_routing(pipe)
    """
    import sklearn

    # Ensure metadata routing is enabled
    sklearn.set_config(enable_metadata_routing=True)

    # Configure the estimator and all its steps
    _configure_estimator(estimator)

    return estimator


def _configure_estimator(estimator: BaseEstimator) -> None:
    """Recursively configure an estimator for sequence metadata routing."""
    from sklearn.pipeline import Pipeline

    # If it's a pipeline, configure each step
    if isinstance(estimator, Pipeline):
        for name, step in estimator.steps:
            if step is not None:
                _configure_step(step, name)
    else:
        # Single estimator
        _configure_step(estimator, None)


def _configure_step(estimator: BaseEstimator, name: str | None) -> None:
    """Configure a single estimator step for sequence metadata routing."""
    # Check if the estimator has metadata request methods
    if hasattr(estimator, "set_fit_request"):
        try:
            estimator.set_fit_request(lengths=True)
        except Exception:
            pass

    if hasattr(estimator, "set_transform_request"):
        try:
            estimator.set_transform_request(lengths=True)
        except Exception:
            pass

    if hasattr(estimator, "set_predict_request"):
        try:
            estimator.set_predict_request(lengths=True)
        except Exception:
            pass

    if hasattr(estimator, "set_score_request"):
        try:
            estimator.set_score_request(lengths=True)
        except Exception:
            pass

    # Handle nested pipelines (e.g., in a meta-estimator)
    from sklearn.pipeline import Pipeline

    if hasattr(estimator, "steps"):
        for step_name, step in estimator.steps:
            if step is not None:
                _configure_step(step, step_name)
