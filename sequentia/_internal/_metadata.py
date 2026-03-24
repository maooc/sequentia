# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Metadata routing utilities for scikit-learn compatibility."""

from __future__ import annotations

import typing as t

import sklearn
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    get_routing_for_object,
    process_routing,
)

__all__ = [
    "MetadataRouter",
    "MethodMapping",
    "get_routing_for_object",
    "process_routing",
    "routing_enabled",
    "make_sequence_router",
    "get_sequence_lengths_router",
]


def routing_enabled() -> bool:
    """Check if metadata routing is enabled in scikit-learn configuration."""
    return sklearn.get_config()["enable_metadata_routing"]


def make_sequence_router(
    owner: t.Any,
    estimator: t.Any,
    transform_included: bool = False,
    predict_included: bool = True,
) -> MetadataRouter:
    """Create a MetadataRouter for sequence models that needs lengths routing.

    This utility configures routing for meta-estimators wrapping sequence models
    that require 'lengths' metadata to be routed to their methods.

    Parameters
    ----------
    owner:
        The meta-estimator instance that owns this router.

    estimator:
        The underlying estimator requiring the lengths metadata.

    transform_included:
        Whether to route lengths to transform methods (used in Pipeline).

    predict_included:
        Whether to route lengths to predict methods (used in meta-estimators).

    Returns
    -------
    MetadataRouter
        Configured router for sequence models.
    """
    router = MetadataRouter(owner=owner)

    # Basic fit mapping
    fit_mapping = MethodMapping().add(caller="fit", callee="fit")

    if transform_included:
        fit_mapping.add(caller="fit", callee="transform").add(
            caller="transform", callee="transform"
        )

    if predict_included:
        predict_mapping = MethodMapping()
        predict_mapping.add(caller="predict", callee="predict").add(
            caller="predict_proba", callee="predict_proba"
        ).add(caller="predict_log_proba", callee="predict_log_proba").add(
            caller="score", callee="score"
        )

        if transform_included:
            predict_mapping.add(caller="predict", callee="transform")

    router.add(estimator=estimator, method_mapping=fit_mapping)

    if predict_included:
        router.add(estimator=estimator, method_mapping=predict_mapping)

    return router


def get_sequence_lengths_router(
    obj: t.Any,
    method: str,
    **kwargs: t.Any,
) -> t.Any:
    """Get routed parameters for sequence methods requiring lengths.

    This is a convenience wrapper around process_routing for sequence models.

    Parameters
    ----------
    obj:
        The estimator/router object to get routing for.

    method:
        The method name (e.g., "fit", "predict").

    **kwargs:
        Additional parameters including lengths.

    Returns
    -------
    Bunch
        Routed parameters for underlying estimators.
    """
    lengths = kwargs.pop("lengths", None)
    if lengths is not None:
        kwargs["lengths"] = lengths

    return process_routing(obj, method, **kwargs)
