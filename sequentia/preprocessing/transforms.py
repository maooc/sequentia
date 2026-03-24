# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence preprocessing transforms compatible with sklearn pipelines."""

from __future__ import annotations

import typing as t
import warnings
from functools import partial

import numpy as np
import scipy.signal
import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import _allclose_dense_sparse, check_array

from sequentia._internal import _data, _sklearn, _validation
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = [
    "IndependentFunctionTransformer",
    "mean_filter",
    "median_filter",
    "SequenceFeatureExtractor",
]


class IndependentFunctionTransformer(FunctionTransformer):
    """Constructs a transformer from an arbitrary callable,
    applying the transform independently to each sequence.

    This transform forwards its ``X`` and ``lengths`` arguments
    to a user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    This works conveniently with functions in :mod:`sklearn.preprocessing`
    such as :func:`~sklearn.preprocessing.scale` or :func:`~sklearn.preprocessing.normalize`.

    :note: This is a stateless transform, meaning :func:`fit` and :func:`fit_transform` will not fit on any data.

    See Also
    --------
    :class:`sklearn.preprocessing.FunctionTransformer`
        :class:`.IndependentFunctionTransformer` is based on this class,
        which applies the callable to the entire input array ``X`` as if it were a single sequence.
        Read more in the :ref:`User Guide <function_transformer>`.

    Examples
    --------
    Using an :class:`IndependentFunctionTransformer` with :func:`sklearn.preprocessing.minmax_scale` to
    scale features to the range [0, 1] independently for each sequence in the spoken digits dataset. ::

        from sklearn.preprocessing import minmax_scale
        from sequentia.preprocessing import IndependentFunctionTransformer
        from sequentia.datasets import load_digits

        # Fetch MFCCs of spoken digits
        data = load_digits()

        # Create an independent min-max transform
        transform = IndependentFunctionTransformer(minmax_scale)

        # Apply the transform to the data
        Xt = transform.transform(data.X, lengths=data.lengths)
    """

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
        """See :class:`sklearn:sklearn.preprocessing.FunctionTransformer`."""
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

        # Allow metadata routing for lengths
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_transform_request(lengths=True)
            self.set_inverse_transform_request(lengths=True)

    def _check_input(self, X, *, lengths, reset):
        if self.validate:
            X, lengths = _validation.check_X_lengths(
                X, lengths=lengths, dtype=X.dtype
            )
            return (
                self._validate_data(
                    X, accept_sparse=self.accept_sparse, reset=reset
                ),
                lengths,
            )
        return X, lengths

    def _check_inverse_transform(self, X, *, lengths):
        """Check that func and inverse_func are the inverse."""
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        X_round_trip = self.inverse_transform(
            self.transform(X[idx_selected], lengths=lengths),
            lengths=lengths,
        )

        if hasattr(X, "dtype"):
            dtypes = [X.dtype]
        elif hasattr(X, "dtypes"):
            # Dataframes can have multiple dtypes
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
        """Fits the transformer to ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        IndependentFunctionTransformer
            The fitted transformer.
        """
        X, lengths = self._check_input(X, lengths=lengths, reset=True)
        if self.check_inverse and not (
            self.func is None or self.inverse_func is None
        ):
            self._check_inverse_transform(X, lengths=lengths)
        return self

    def transform(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        """Applies the transformation to ``X``,
        producing a transformed version of ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            The transformed array.
        """
        X, lengths = self._check_input(X, lengths=lengths, reset=False)
        return self._transform(
            X, lengths=lengths, func=self.func, kw_args=self.kw_args
        )

    def inverse_transform(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        """Applies the inverse transformation to ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            The inverse transformed array.
        """
        if self.validate:
            X = check_array(X, accept_sparse=False)
            X, lengths = _validation.check_X_lengths(
                X, lengths=lengths, dtype=X.dtype
            )
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
        """Fits the transformer to the sequence(s) in ``X`` and returns a
        transformed version of ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            The transformed data.
        """
        return self.fit(X, lengths=lengths).transform(X, lengths=lengths)

    def _transform(self, X, *, lengths, func=None, kw_args=None):
        """Apply function to each sequence efficiently.
        
        Uses pre-allocated array and direct indexing for better performance
        compared to np.vstack with list comprehension.
        """
        if func is None:
            return X
        
        kw_args = kw_args if kw_args else {}
        apply = partial(func, **kw_args)
        
        if lengths is None:
            # Single sequence case
            return apply(X)
        
        # Get sequence indices
        idxs = _data.get_idxs(lengths)
        
        # Pre-allocate result array for better memory efficiency
        n_features = X.shape[1] if X.ndim > 1 else 1
        result = np.empty((len(X), n_features), dtype=X.dtype)
        
        # Apply function to each sequence using direct indexing
        pos = 0
        for start, end in idxs:
            length = end - start
            transformed = apply(X[start:end])
            result[pos:pos + length] = transformed
            pos += length
        
        return result


class SequenceFeatureExtractor(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Extract features from sequences for use with standard sklearn estimators.
    
    This transformer converts variable-length sequences into fixed-length feature
    vectors by applying aggregation functions to each sequence. This allows
    sequences to be used with standard sklearn models that expect fixed-length
    inputs.
    
    Parameters
    ----------
    aggregations : list of callable, default=[np.mean, np.std]
        List of aggregation functions to apply to each sequence.
        Each function should accept an array of shape (n_timesteps, n_features)
        and return an array of shape (n_features,).
        
    axis : int, default=0
        Axis along which to apply aggregations. Default is 0 (time axis).
        
    Examples
    --------
    >>> from sequentia.preprocessing import SequenceFeatureExtractor
    >>> from sequentia.datasets import load_digits
    >>> data = load_digits()
    >>> extractor = SequenceFeatureExtractor([np.mean, np.std, np.min, np.max])
    >>> X_features = extractor.fit_transform(data.X, lengths=data.lengths)
    >>> X_features.shape
    (250, 52)  # 250 sequences, 13 features * 4 aggregations
    """
    
    def __init__(
        self,
        aggregations: list[t.Callable] | None = None,
        *,
        axis: int = 0,
    ):
        self.aggregations = aggregations or [np.mean, np.std]
        self.axis = axis
        
        # Allow metadata routing for lengths
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_transform_request(lengths=True)
    
    def fit(self, X: Array, y: Array | None = None, *, lengths: IntArray | None = None) -> t.Self:
        """Fit the transformer (no-op for stateless transform).
        
        Parameters
        ----------
        X:
            Sequence(s).
        y:
            Ignored, present for API consistency.
        lengths:
            Lengths of the sequence(s) provided in ``X``.
            
        Returns
        -------
        self
        """
        X, lengths = _validation.check_X_lengths(
            X, lengths=lengths, dtype=X.dtype
        )
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        self.n_aggregations_ = len(self.aggregations)
        return self

    def fit_transform(self, X: Array, y: Array | None = None, *, lengths: IntArray | None = None) -> FloatArray:
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X:
            Sequence(s).
        y:
            Ignored, present for API consistency.
        lengths:
            Lengths of the sequence(s) provided in ``X``.
            
        Returns
        -------
        numpy.ndarray:
            Fixed-length feature vectors of shape (n_sequences, n_features * n_aggregations).
        """
        return self.fit(X, y, lengths=lengths).transform(X, lengths=lengths)

    def transform(self, X: Array, *, lengths: IntArray | None = None) -> FloatArray:
        """Transform sequences to fixed-length feature vectors.
        
        Parameters
        ----------
        X:
            Sequence(s).
        lengths:
            Lengths of the sequence(s) provided in ``X``.
            
        Returns
        -------
        numpy.ndarray:
            Fixed-length feature vectors of shape (n_sequences, n_features * n_aggregations).
        """
        X, lengths = _validation.check_X_lengths(
            X, lengths=lengths, dtype=X.dtype
        )
        
        idxs = _data.get_idxs(lengths)
        n_sequences = len(lengths)
        n_features = X.shape[1] if X.ndim > 1 else 1
        n_aggregations = len(self.aggregations)
        
        # Map string aggregations to numpy functions
        agg_map = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'sum': np.sum,
            'median': np.median,
        }
        
        # Resolve aggregation functions
        aggregations = []
        for agg in self.aggregations:
            if isinstance(agg, str):
                if agg not in agg_map:
                    raise ValueError(f"Unknown aggregation: {agg}. "
                                     f"Supported: {list(agg_map.keys())}")
                aggregations.append(agg_map[agg])
            else:
                aggregations.append(agg)
        
        # Pre-allocate result
        result = np.empty((n_sequences, n_features * n_aggregations), dtype=np.float64)
        
        # Extract features for each sequence
        for i, (start, end) in enumerate(idxs):
            seq = X[start:end]
            features = []
            for agg in aggregations:
                agg_result = agg(seq, axis=self.axis)
                # Ensure 1D array - np.mean/std return arrays when axis is specified
                if np.isscalar(agg_result):
                    agg_result = np.array([agg_result])
                else:
                    agg_result = np.ravel(agg_result)
                features.append(agg_result)
            result[i] = np.concatenate(features)
        
        return result


def mean_filter(x: FloatArray, *, k: int = 5) -> FloatArray:
    """Applies a mean filter of size ``k`` independently to each feature of
    the sequence, retaining the original input shape by using appropriate
    padding.

    This is implemented as a 1D convolution with a kernel of size ``k`` and
    values ``1 / k``.

    Parameters
    ----------
    x:
        Observation sequence.

    k:
        Width of the filter.

    Returns
    -------
    numpy.ndarray:
        The filtered array.

    Examples
    --------
    Applying a :func:`mean_filter` to a single sequence
    and multiple sequences (independently via :class:`IndependentFunctionTransformer`) from the spoken digits dataset. ::

        from sequentia.preprocessing import IndependentFunctionTransformer, mean_filter
        from sequentia.datasets import load_digits

        # Fetch MFCCs of spoken digits
        data = load_digits()

        # Apply the mean filter to the first sequence
        x, _ = data[0]
        xt = mean_filter(x, k=7)

        # Create an independent mean filter transform
        transform = IndependentFunctionTransformer(mean_filter, kw_args={"k": 7})

        # Apply the transform to all sequences
        Xt = transform.transform(data.X, lengths=data.lengths)
    """
    return scipy.signal.convolve(x, np.ones((k, 1)) / k, mode="same")


def median_filter(x: FloatArray, *, k: int = 5) -> FloatArray:
    """Applies a median filter of size ``k`` independently to each feature of
    the sequence, retaining the original input shape by using appropriate
    padding.

    Parameters
    ----------
    x:
        Observation sequence.

    k:
        Width of the filter.

    Returns
    -------
    numpy.ndarray:
        The filtered array.

    Examples
    --------
    Applying a :func:`median_filter` to a single sequence
    and multiple sequences (independently via :class:`IndependentFunctionTransformer`) from the spoken digits dataset. ::

        from sequentia.preprocessing import IndependentFunctionTransformer, median_filter
        from sequentia.datasets import load_digits

        # Fetch MFCCs of spoken digits
        data = load_digits()

        # Apply the median filter to the first sequence
        x, _ = data[0]
        xt = median_filter(x, k=7)

        # Create an independent median filter transform
        transform = IndependentFunctionTransformer(median_filter, kw_args={"k": 7})

        # Apply the transform to all sequences
        Xt = transform.transform(data.X, lengths=data.lengths)
    """
    return scipy.signal.medfilt2d(x, kernel_size=(k, 1))
