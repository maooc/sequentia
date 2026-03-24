# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Metadata routing mixin for sequence models."""

from __future__ import annotations

import typing as t

import numpy as np
from sklearn.utils.metadata_routing import (
    get_routing_for_object,
)

from sequentia._internal import _data, _sklearn
from sequentia._internal._typing import Array, IntArray

__all__ = ["SequenceMetadataRouterMixin"]


class SequenceMetadataRouterMixin:
    """Mixin providing sequence-aware metadata routing capabilities.

    This mixin provides methods to handle sequence data and lengths metadata
    routing for sequence models, estimators, and transformers.
    """

    def _get_sequence_indices(
        self,
        lengths: IntArray,
        /,
        indices: IntArray | None = None,
    ) -> IntArray:
        """Get sequence start/end indices, optionally subset by indices.

        Parameters
        ----------
        lengths:
            Lengths of sequences.

        indices:
            Optional subset indices for selecting specific sequences.

        Returns
        -------
        numpy.ndarray
            (n_sequences, 2) array of start/end indices.
        """
        if indices is not None:
            lengths = lengths[indices]
        return _data.get_idxs(lengths)

    def _subset_sequence_data(
        self,
        X: Array,
        lengths: IntArray,
        /,
        indices: IntArray,
    ) -> tuple[Array, IntArray]:
        """Subset sequence data by sequence indices.

        Parameters
        ----------
        X:
            Combined sequence array.

        lengths:
            Original sequence lengths.

        indices:
            Indices of sequences to keep.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Subset X array and corresponding lengths.
        """
        idxs = self._get_sequence_indices(lengths, indices)
        lengths_subset = lengths[indices]
        X_subset = np.concatenate(list(_data.iter_X(X, idxs=idxs)))
        return X_subset, lengths_subset

    def _route_and_subset(
        self,
        method: str,
        X: Array,
        y: Array | None = None,
        **kwargs: t.Any,
    ) -> tuple[Array, Array | None, IntArray, t.Any]:
        """Route metadata and subset sequence data for cross-validation.

        This method handles the metadata routing and returns the properly
        subsetted data for the current fold.

        Parameters
        ----------
        method:
            The method being called (e.g., "fit", "predict").

        X:
            Input features.

        y:
            Target values, optional.

        **kwargs:
            Additional parameters including lengths and optional indices.

        Returns
        -------
        tuple
            (X_subset, y_subset, lengths_subset, routed_params)
        """
        lengths = kwargs.pop("lengths", None)
        if lengths is None:
            raise ValueError("lengths metadata is required for sequence models")

        # Process routing
        routed = get_routing_for_object(self).route_params(
            params={"lengths": lengths, **kwargs}, caller=method
        )

        # Handle indices if provided (for cross-validation splits)
        indices = kwargs.get("indices")
        if indices is not None:
            X, lengths = self._subset_sequence_data(X, lengths, indices)
            if y is not None:
                y = y[indices]

        return X, y, lengths, routed
