# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""scikit-learn compatibility utilities."""

from __future__ import annotations

import typing as t

import sklearn
from sklearn import set_config
from sklearn.base import BaseEstimator  # noqa: F401

# Enable metadata routing by default for sequence models
# This is required for proper lengths parameter routing through sklearn pipelines
set_config(enable_metadata_routing=True)

from sequentia._internal._metadata import (  # noqa: F401
    MetadataRouter,
    MethodMapping,
    get_routing_for_object,
    get_sequence_lengths_router,
    make_sequence_router,
    process_routing,
    routing_enabled,
)

__all__ = [
    "routing_enabled",
    "BaseEstimator",
    "MetadataRouter",
    "MethodMapping",
    "get_routing_for_object",
    "process_routing",
    "make_sequence_router",
    "get_sequence_lengths_router",
    "set_default_metadata_requests",
]


def set_default_metadata_requests(
    estimator: BaseEstimator,
    *,
    fit: list[str] | None = None,
    predict: list[str] | None = None,
    transform: list[str] | None = None,
    score: list[str] | None = None,
    split: list[str] | None = None,
) -> None:
    """Set default metadata requests on an estimator based on method signatures.

    This function inspects the method signatures of the estimator and sets
    appropriate metadata requests for parameters that end with '_request'
    or have known metadata names like 'lengths'.

    Parameters
    ----------
    estimator:
        The estimator to configure.

    fit:
        Additional fit method metadata parameter names.

    predict:
        Additional predict method metadata parameter names.

    transform:
        Additional transform method metadata parameter names.

    score:
        Additional score method metadata parameter names.
        
    split:
        Additional split method metadata parameter names (for CV splitters).
    """
    fit = fit or []
    predict = predict or []
    transform = transform or []
    score = score or []
    split = split or []

    # Always request 'lengths' since it's core to sequence models
    base_requests = {"lengths": True}

    # Configure fit requests
    if hasattr(estimator, "set_fit_request"):
        fit_requests = {**base_requests}
        for param in fit:
            fit_requests[param] = True
        try:
            estimator.set_fit_request(**fit_requests)
        except (ValueError, TypeError):
            pass

    # Configure predict requests
    if hasattr(estimator, "set_predict_request"):
        predict_requests = {**base_requests}
        for param in predict:
            predict_requests[param] = True
        try:
            estimator.set_predict_request(**predict_requests)
        except (ValueError, TypeError):
            pass

    # Configure predict_proba requests
    if hasattr(estimator, "set_predict_proba_request"):
        proba_requests = {**base_requests}
        try:
            estimator.set_predict_proba_request(**proba_requests)
        except (ValueError, TypeError):
            pass

    # Configure predict_log_proba requests
    if hasattr(estimator, "set_predict_log_proba_request"):
        log_proba_requests = {**base_requests}
        try:
            estimator.set_predict_log_proba_request(**log_proba_requests)
        except (ValueError, TypeError):
            pass

    # Configure transform requests
    if hasattr(estimator, "set_transform_request"):
        transform_requests = {**base_requests}
        for param in transform:
            transform_requests[param] = True
        try:
            estimator.set_transform_request(**transform_requests)
        except (ValueError, TypeError):
            pass

    # Configure inverse_transform requests
    if hasattr(estimator, "set_inverse_transform_request"):
        inv_transform_requests = {**base_requests}
        try:
            estimator.set_inverse_transform_request(**inv_transform_requests)
        except (ValueError, TypeError):
            pass

    # Configure score requests
    if hasattr(estimator, "set_score_request"):
        score_requests = {**base_requests}
        score_requests["sample_weight"] = True
        score_requests["normalize"] = True
        for param in score:
            score_requests[param] = True
        try:
            estimator.set_score_request(**score_requests)
        except (ValueError, TypeError):
            pass
            
    # Configure split requests (for CV splitters)
    if hasattr(estimator, "set_split_request"):
        split_requests = {**base_requests}
        for param in split:
            split_requests[param] = True
        try:
            estimator.set_split_request(**split_requests)
        except (ValueError, TypeError):
            pass
