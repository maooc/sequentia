# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Utilities for scikit-learn compatibility.

This module provides utilities for integrating sequentia estimators with
scikit-learn's metadata routing protocol.

The metadata routing protocol allows estimators to declare which metadata
they accept (like 'lengths' for sequence data), and scikit-learn will
automatically route this metadata through pipelines, cross-validation,
and hyper-parameter search.
"""

from __future__ import annotations

import typing as t

import sklearn

__all__ = [
    "routing_enabled",
    "LengthsRoutingMixin",
]


def routing_enabled() -> bool:
    """Check if scikit-learn metadata routing is enabled.

    Metadata routing must be enabled for the lengths parameter to be
    automatically routed through sklearn's cross-validation and
    hyper-parameter search utilities.

    Enable it with:
        sklearn.set_config(enable_metadata_routing=True)

    Returns
    -------
    bool
        True if metadata routing is enabled.
    """
    return sklearn.get_config().get("enable_metadata_routing", False)


def _sklearn_version() -> tuple[int, int]:
    """Get scikit-learn version as tuple of (major, minor)."""
    parts = sklearn.__version__.split(".")
    return (int(parts[0]), int(parts[1]))


class LengthsRoutingMixin:
    """Mixin class providing lengths metadata routing support.

    This mixin provides methods to set up metadata routing for the
    'lengths' parameter, which is used by sequence estimators.

    Classes using this mixin should call _setup_lengths_routing() in
    their __init__ method after calling super().__init__().

    The mixin uses scikit-learn's public metadata routing API:
    - set_fit_request(lengths=True)
    - set_predict_request(lengths=True)
    - etc.

    These methods are available in scikit-learn 1.4+ when the estimator
    inherits from BaseEstimator.
    """

    def _setup_lengths_routing(self) -> None:
        """Set up metadata routing for lengths.

        This method should be called in __init__ after super().__init__().
        It configures the estimator to request the 'lengths' metadata
        for all applicable methods.
        """
        if not routing_enabled():
            return

        if hasattr(self, "set_fit_request"):
            self.set_fit_request(lengths=True)

        if hasattr(self, "set_predict_request"):
            self.set_predict_request(lengths=True)

        if hasattr(self, "set_predict_proba_request"):
            self.set_predict_proba_request(lengths=True)

        if hasattr(self, "set_predict_log_proba_request"):
            self.set_predict_log_proba_request(lengths=True)

        if hasattr(self, "set_score_request"):
            self.set_score_request(lengths=True)

        if hasattr(self, "set_transform_request"):
            self.set_transform_request(lengths=True)

        if hasattr(self, "set_inverse_transform_request"):
            self.set_inverse_transform_request(lengths=True)

    def get_metadata_routing(self) -> t.Any:
        """Get metadata routing for this estimator.

        This method is required by scikit-learn's metadata routing protocol.
        It returns the routing information for all metadata parameters.

        Returns
        -------
        MetadataRequest
            The metadata routing information.
        """
        if hasattr(super(), "get_metadata_routing"):
            return super().get_metadata_routing()
        from sklearn.utils.metadata_routing import MetadataRouter

        router = MetadataRouter(owner=self.__class__.__name__)
        return router
