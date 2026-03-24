# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware preprocessing transformers.

This module provides transformers that apply functions independently to each
sequence in a dataset, with optimized performance and sklearn Pipeline compatibility.

Key Design Principles
---------------------
1. Vectorized operations where possible - minimize explicit Python loops
2. Pre-allocated output arrays - avoid memory fragmentation from vstack/concat
3. Stateless transformations - consistent with sklearn conventions
4. Metadata routing support - seamless Pipeline integration
"""

from __future__ import annotations

import typing as t
import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy.signal
import sklearn.base
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import _allclose_dense_sparse, check_array

from sequentia._internal import _sequence, _validation
from sequentia._internal._routing import SequenceMetadataMixin, routing_enabled
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = [
    "IndependentFunctionTransformer",
    "SequenceTransformer",
    "SequenceFeatureExtractor",
    "mean_filter",
    "median_filter",
    "extract_mean",
    "extract_std",
    "extract_min",
    "extract_max",
    "extract_range",
    "extract_quantiles",
]


class SequenceTransformer(
    SequenceMetadataMixin,
    sklearn.base.BaseEstimator,
    sklearn.base.TransformerMixin,
    ABC,
):
    """Abstract base class for sequence-aware transformers.

    This class provides a foundation for creating transformers that operate
    on sequence data, ensuring proper handling of the `lengths` parameter
    and seamless integration with sklearn Pipelines via metadata routing.

    Subclasses must implement the `_transform_sequence` method to define
    the transformation applied to each individual sequence.

    Metadata Routing
    ----------------
    When sklearn metadata routing is enabled, this transformer automatically
    requests the `lengths` parameter for fit, transform, and inverse_transform
    methods, allowing it to work seamlessly within sklearn Pipelines.
    """

    _metadata_requests = ("fit", "transform", "fit_transform", "inverse_transform")

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._enable_length_routing()

    def fit(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> t.Self:
        return self

    def transform(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        X, lengths = _validation.check_X_lengths(X, lengths=lengths, dtype=X.dtype)
        return self._transform_implementation(X, lengths)

    def fit_transform(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        return self.fit(X, y, lengths=lengths).transform(X, lengths=lengths)

    def inverse_transform(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        raise NotImplementedError(f"{self.__class__.__name__} does not support inverse_transform")

    @abstractmethod
    def _transform_sequence(self, x: Array) -> Array:
        raise NotImplementedError

    def _transform_implementation(self, X: Array, lengths: IntArray) -> Array:
        idxs = _sequence.get_sequence_indices(lengths)
        n_sequences = len(lengths)

        transformed_seqs = []
        total_len = 0
        for i in range(n_sequences):
            start, end = idxs[i]
            seq = X[start:end]
            transformed = self._transform_sequence(seq)
            transformed_seqs.append(transformed)
            total_len += len(transformed)

        n_features = transformed_seqs[0].shape[1] if len(transformed_seqs) > 0 else X.shape[1]
        result = np.empty((total_len, n_features), dtype=X.dtype)

        offset = 0
        for transformed in transformed_seqs:
            n = len(transformed)
            result[offset : offset + n] = transformed
            offset += n

        return result


class IndependentFunctionTransformer(SequenceMetadataMixin, FunctionTransformer):
    """Constructs a transformer from an arbitrary callable,
    applying the transform independently to each sequence.

    This transform forwards its ``X`` and ``lengths`` arguments
    to a user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Performance Optimizations
    -------------------------
    This implementation uses pre-allocated output arrays instead of vstack/concat
    to reduce memory fragmentation and improve performance for large datasets.

    Metadata Routing
    ----------------
    When sklearn metadata routing is enabled, this transformer automatically
    requests the `lengths` parameter, allowing it to work seamlessly within
    sklearn Pipelines without manual parameter passing.
    """

    _metadata_requests = ("fit", "transform", "fit_transform", "inverse_transform")

    def __init__(
        self,
        func=None,
        inverse_func=None,
        *,
        validate=False,
        accept_sparse=False,
        check_inverse=True,
        feature_names_out=None,
        kw_args=None,
        inv_kw_args=None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
        self._enable_length_routing()

    def _check_input(self, X, *, lengths, reset):
        if self.validate:
            X, lengths = _validation.check_X_lengths(X, lengths=lengths, dtype=X.dtype)
            return (
                self._validate_data(X, accept_sparse=self.accept_sparse, reset=reset),
                lengths,
            )
        return X, lengths

    def _check_inverse_transform(self, X, *, lengths):
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        X_round_trip = self.inverse_transform(
            self.transform(X[idx_selected], lengths=lengths),
            lengths=lengths,
        )

        if hasattr(X, "dtype"):
            dtypes = [X.dtype]
        elif hasattr(X, "dtypes"):
            dtypes = X.dtypes

        if not all(np.issubdtype(d, np.number) for d in dtypes):
            raise ValueError(
                "'check_inverse' is only supported when all the elements in `X` is"
                " numerical."
            )

        if not _allclose_dense_sparse(X[idx_selected], X_round_trip):
            warnings.warn(
                (
                    "The provided functions are not strictly"
                    " inverse of each other. If you are sure you"
                    " want to proceed regardless, set"
                    " 'check_inverse=False'."
                ),
                UserWarning,
            )

    @sklearn.base._fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> t.Self:
        X, lengths = self._check_input(X, lengths=lengths, reset=True)
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(X, lengths=lengths)
        return self

    def transform(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        X, lengths = self._check_input(X, lengths=lengths, reset=False)
        return self._transform(X, lengths=lengths, func=self.func, kw_args=self.kw_args)

    def inverse_transform(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        if self.validate:
            X = check_array(X, accept_sparse=False)
            X, lengths = _validation.check_X_lengths(X, lengths=lengths, dtype=X.dtype)
        return self._transform(
            X,
            lengths=lengths,
            func=self.inverse_func,
            kw_args=self.inv_kw_args,
        )

    def fit_transform(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        return self.fit(X, lengths=lengths).transform(X, lengths=lengths)

    def _transform(self, X, *, lengths, func=None, kw_args=None):
        if func is None:
            return X

        kw = kw_args if kw_args else {}

        idxs = _sequence.get_sequence_indices(lengths)
        n_sequences = len(lengths)

        transformed_seqs = []
        total_len = 0
        n_features = None

        for i in range(n_sequences):
            start, end = idxs[i]
            seq = X[start:end]
            transformed = func(seq, **kw)
            transformed_seqs.append(transformed)
            total_len += len(transformed)
            if n_features is None:
                n_features = transformed.shape[1] if transformed.ndim > 1 else 1

        if n_features is None:
            n_features = X.shape[1] if X.ndim > 1 else 1

        result = np.empty((total_len, n_features), dtype=X.dtype)

        offset = 0
        for transformed in transformed_seqs:
            n = len(transformed)
            if transformed.ndim == 1:
                result[offset : offset + n, 0] = transformed
            else:
                result[offset : offset + n] = transformed
            offset += n

        return result


class SequenceFeatureExtractor(SequenceTransformer):
    """Base class for feature extractors that reduce sequences to fixed-length vectors.

    Feature extractors transform variable-length sequences into fixed-length
    feature vectors, one per sequence. This is useful for downstream classifiers
    that require fixed-length input.

    Performance Optimizations
    -------------------------
    Uses pre-allocated output arrays instead of vstack to avoid memory
    fragmentation when processing large numbers of sequences.
    """

    def transform(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        X, lengths = _validation.check_X_lengths(X, lengths=lengths, dtype=X.dtype)
        return self._transform_implementation(X, lengths)

    def _transform_implementation(self, X: Array, lengths: IntArray) -> Array:
        n_sequences = len(lengths)

        first_features = None
        for seq in _sequence.iter_sequences(X, lengths):
            first_features = self._extract_features(seq)
            break

        if first_features is None:
            return np.empty((0, X.shape[1] if X.ndim > 1 else 1))

        n_features = len(first_features)
        result = np.empty((n_sequences, n_features), dtype=X.dtype)
        result[0] = first_features

        idx = 1
        for seq in list(_sequence.iter_sequences(X, lengths))[1:]:
            result[idx] = self._extract_features(seq)
            idx += 1

        return result

    @abstractmethod
    def _extract_features(self, x: Array) -> Array:
        raise NotImplementedError

    def _transform_sequence(self, x: Array) -> Array:
        raise NotImplementedError("SequenceFeatureExtractor uses _extract_features instead")


class MeanFeatureExtractor(SequenceFeatureExtractor):
    """Extract mean of each feature across the sequence.

    Uses vectorized numpy operations for optimal performance.
    """

    def _extract_features(self, x: Array) -> Array:
        return np.mean(x, axis=0)


class StdFeatureExtractor(SequenceFeatureExtractor):
    """Extract standard deviation of each feature across the sequence.

    Parameters
    ----------
    ddof : int, default=0
        Delta degrees of freedom for std calculation.
    """

    def __init__(self, ddof: int = 0):
        self.ddof = ddof

    def _extract_features(self, x: Array) -> Array:
        return np.std(x, axis=0, ddof=self.ddof)


class MinFeatureExtractor(SequenceFeatureExtractor):
    """Extract minimum of each feature across the sequence."""

    def _extract_features(self, x: Array) -> Array:
        return np.min(x, axis=0)


class MaxFeatureExtractor(SequenceFeatureExtractor):
    """Extract maximum of each feature across the sequence."""

    def _extract_features(self, x: Array) -> Array:
        return np.max(x, axis=0)


class RangeFeatureExtractor(SequenceFeatureExtractor):
    """Extract range (max - min) of each feature across the sequence."""

    def _extract_features(self, x: Array) -> Array:
        return np.ptp(x, axis=0)


class QuantileFeatureExtractor(SequenceFeatureExtractor):
    """Extract quantiles of each feature across the sequence.

    Parameters
    ----------
    quantiles : array-like, default=(0.25, 0.5, 0.75)
        Quantiles to extract. Each quantile is computed for each feature,
        resulting in n_features * n_quantiles output features.
    """

    def __init__(self, quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)):
        self.quantiles = quantiles

    def _extract_features(self, x: Array) -> Array:
        return np.quantile(x, self.quantiles, axis=0).flatten()


def extract_mean(x: FloatArray) -> FloatArray:
    """Extract mean of each feature from a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.

    Returns
    -------
    FloatArray
        Mean of each feature.
    """
    return np.mean(x, axis=0)


def extract_std(x: FloatArray, ddof: int = 0) -> FloatArray:
    """Extract standard deviation of each feature from a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.
    ddof : int
        Delta degrees of freedom.

    Returns
    -------
    FloatArray
        Standard deviation of each feature.
    """
    return np.std(x, axis=0, ddof=ddof)


def extract_min(x: FloatArray) -> FloatArray:
    """Extract minimum of each feature from a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.

    Returns
    -------
    FloatArray
        Minimum of each feature.
    """
    return np.min(x, axis=0)


def extract_max(x: FloatArray) -> FloatArray:
    """Extract maximum of each feature from a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.

    Returns
    -------
    FloatArray
        Maximum of each feature.
    """
    return np.max(x, axis=0)


def extract_range(x: FloatArray) -> FloatArray:
    """Extract range (max - min) of each feature from a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.

    Returns
    -------
    FloatArray
        Range of each feature.
    """
    return np.ptp(x, axis=0)


def extract_quantiles(x: FloatArray, quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)) -> FloatArray:
    """Extract quantiles of each feature from a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.
    quantiles : tuple
        Quantiles to extract.

    Returns
    -------
    FloatArray
        Quantiles of each feature.
    """
    return np.quantile(x, quantiles, axis=0).flatten()


def mean_filter(x: FloatArray, window_size: int = 3) -> FloatArray:
    """Apply a mean filter to a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.
    window_size : int
        Size of the filter window.

    Returns
    -------
    FloatArray
        Filtered sequence.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if len(x) < window_size:
        return x.copy()

    result = np.empty_like(x)
    half_window = window_size // 2

    for i in range(len(x)):
        start = max(0, i - half_window)
        end = min(len(x), i + half_window + 1)
        result[i] = np.mean(x[start:end], axis=0)

    return result


def median_filter(x: FloatArray, window_size: int = 3) -> FloatArray:
    """Apply a median filter to a sequence.

    Parameters
    ----------
    x : FloatArray
        Observation sequence.
    window_size : int
        Size of the filter window.

    Returns
    -------
    FloatArray
        Filtered sequence.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if len(x) < window_size:
        return x.copy()

    return scipy.signal.medfilt(x, kernel_size=window_size)
