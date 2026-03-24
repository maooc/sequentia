# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Metadata routing support for sequence data.

This module provides utilities for enabling sklearn metadata routing
for the `lengths` parameter, allowing seamless integration with
sklearn Pipelines and model selection tools.
"""

from __future__ import annotations

import typing as t

import sklearn

__all__ = [
    "SequenceMetadataMixin",
    "enable_metadata_routing",
    "routing_enabled",
]


def routing_enabled() -> bool:
    """Check if sklearn metadata routing is enabled.

    Returns
    -------
    bool
        True if metadata routing is enabled.
    """
    try:
        from sklearn import get_config

        return get_config().get("enable_metadata_routing", False)
    except (ImportError, AttributeError):
        return False


def enable_metadata_routing() -> None:
    """Enable sklearn metadata routing globally."""
    sklearn.set_config(enable_metadata_routing=True)


class SequenceMetadataMixin:
    """Mixin class that adds metadata routing support for `lengths` parameter.

    This mixin automatically generates `set_*_request` methods for the
    `lengths` parameter when sklearn metadata routing is enabled.

    Classes that inherit from this mixin should define which methods
    accept the `lengths` parameter by setting the `_metadata_requests`
    class attribute.

    Example
    -------
    >>> class MyTransformer(SequenceMetadataMixin, BaseEstimator, TransformerMixin):
    ...     _metadata_requests = ("fit", "transform", "fit_transform")
    ...
    ...     def fit(self, X, y=None, *, lengths=None):
    ...         return self
    ...
    ...     def transform(self, X, *, lengths=None):
    ...         return X
    """

    _metadata_requests: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)

        if routing_enabled():
            for method in cls._metadata_requests:
                attr_name = f"__metadata_request__{method}"
                if not hasattr(cls, attr_name):
                    setattr(cls, attr_name, {"lengths": None})

    def _enable_length_routing(self) -> None:
        """Enable metadata routing for the lengths parameter.

        This method should be called at the end of __init__ in subclasses.
        """
        if routing_enabled():
            for method in self._metadata_requests:
                setter_name = f"set_{method}_request"
                if hasattr(self, setter_name):
                    setter = getattr(self, setter_name)
                    try:
                        setter(lengths=True)
                    except TypeError:
                        pass
