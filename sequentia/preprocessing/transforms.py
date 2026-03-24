# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Transformers for sequence preprocessing and feature engineering."""

from __future__ import annotations

import typing as t
import warnings

import numpy as np
import scipy.signal
import sklearn.base
from sklearn.utils.validation import _allclose_dense_sparse, check_array

from sequentia._internal import _data, _sklearn, _validation
from sequentia._internal._mixin import SequenceTransformerMixin
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = [
    "IndependentFunctionTransformer",
    "mean_filter",
    "median_filter",
    "downsample",
    "normalize",
    "standardize",
]


class IndependentFunctionTransformer(SequenceTransformerMixin):
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

    _parameter_constraints: dict[str, list[t.Any]] = {
        "func": [callable, None],
        "inverse_func": [callable, None],
        "validate": ["boolean"],
        "accept_sparse": ["boolean"],
        "check_inverse": ["boolean"],
        "feature_names_out": [str, None, callable],
        "kw_args": [dict, None],
        "inv_kw_args": [dict, None],
    }

    def __init__(
        self,
        func: t.Callable[[Array], Array] | None = None,
        inverse_func: t.Callable[[Array], Array] | None = None,
        *,
        validate: bool = False,
        accept_sparse: bool = False,
        check_inverse: bool = True,
        feature_names_out: str | t.Callable | None = None,
        kw_args: dict[str, t.Any] | None = None,
        inv_kw_args: dict[str, t.Any] | None = None,
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

    def _check_input(
        self, X: Array, *, lengths: IntArray, reset: bool
    ) -> tuple[Array, IntArray]:
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

    def _check_inverse_transform(self, X: Array, *, lengths: IntArray) -> None:
        """Check that func and inverse_func are the inverse."""
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        X_round_trip = self.inverse_transform(
            self.transform(X[idx_selected], lengths=lengths),
            lengths=lengths,
        )

        if hasattr(X, "dtype"):
            dtypes = [X.dtype]
        elif hasattr(X, "dtypes"):
            dtypes = X.dtypes
        else:
            dtypes = []

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
                stacklevel=2,
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
        _, lengths = self._check_input(X, lengths=lengths, reset=True)
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

    def _transform(
        self,
        X: Array,
        *,
        lengths: IntArray,
        func: t.Callable[[Array], Array] | None = None,
        kw_args: dict[str, t.Any] | None = None,
    ) -> Array:
        """Apply the transformation efficiently using vectorized operations.

        Uses the optimized _transform_independent method from SequenceTransformerMixin
        to avoid unnecessary memory copies and to process sequences efficiently.
        """
        if func is None:
            return X

        apply = lambda x: func(x, **(kw_args if kw_args else {}))
        return self._transform_independent(X, lengths=lengths, func=apply)



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


def downsample(x: FloatArray, *, factor: int = 2, method: str = "mean") -> FloatArray:
    """Downsamples a sequence by the given factor, using the specified aggregation method.

    Parameters
    ----------
    x:
        Observation sequence of shape (n_timesteps, n_features).

    factor:
        Downsampling factor.

    method:
        Aggregation method to use:

        - ``"mean"``: Take mean of each window
        - ``"median"``: Take median of each window
        - ``"first"``: Take first element of each window
        - ``"last"``: Take last element of each window

    Returns
    -------
    numpy.ndarray:
        Downsampled sequence of shape (n_timesteps // factor, n_features).

    Examples
    --------
    Downsampling sequences using :class:`IndependentFunctionTransformer`: ::

        from sequentia.preprocessing import IndependentFunctionTransformer, downsample
        from sequentia.datasets import load_digits

        data = load_digits()

        # Downsample by factor of 2 using mean aggregation
        transform = IndependentFunctionTransformer(downsample, kw_args={"factor": 2, "method": "mean"})
        Xt = transform.transform(data.X, lengths=data.lengths)
    """
    n_timesteps, n_features = x.shape
    n_output = n_timesteps // factor

    if n_output == 0:
        return x[:1, :]

    x_reshaped = x[: n_output * factor].reshape(n_output, factor, n_features)

    if method == "mean":
        return x_reshaped.mean(axis=1)
    if method == "median":
        return np.median(x_reshaped, axis=1)
    if method == "first":
        return x_reshaped[:, 0, :]
    if method == "last":
        return x_reshaped[:, -1, :]

    raise ValueError(f"Unknown downsampling method: {method}")


def normalize(x: FloatArray, *, axis: int = 0, order: int = 2) -> FloatArray:
    """Normalizes each feature (axis=0) or each timestep (axis=1) to have unit norm.

    Parameters
    ----------
    x:
        Observation sequence of shape (n_timesteps, n_features).

    axis:
        Axis along which to normalize:

        - ``0``: Normalize each feature independently across timesteps
        - ``1``: Normalize each timestep independently across features

    order:
        Order of the norm to use (1 for L1 norm, 2 for L2 norm).

    Returns
    -------
    numpy.ndarray:
        Normalized sequence of the same shape as input.

    Examples
    --------
    Normalizing sequences using :class:`IndependentFunctionTransformer`: ::

        from sequentia.preprocessing import IndependentFunctionTransformer, normalize
        from sequentia.datasets import load_digits

        data = load_digits()

        # L2 normalize each feature across timesteps
        transform = IndependentFunctionTransformer(normalize, kw_args={"axis": 0, "order": 2})
        Xt = transform.transform(data.X, lengths=data.lengths)
    """
    eps = np.finfo(x.dtype).eps
    norm = np.linalg.norm(x, ord=order, axis=axis, keepdims=True)
    return x / (norm + eps)


def standardize(x: FloatArray, *, axis: int = 0) -> FloatArray:
    """Standardizes each feature (axis=0) or each timestep (axis=1) to have
    zero mean and unit variance.

    Parameters
    ----------
    x:
        Observation sequence of shape (n_timesteps, n_features).

    axis:
        Axis along which to standardize:

        - ``0``: Standardize each feature independently across timesteps
        - ``1``: Standardize each timestep independently across features

    Returns
    -------
    numpy.ndarray:
        Standardized sequence of the same shape as input.

    Examples
    --------
    Standardizing sequences using :class:`IndependentFunctionTransformer`: ::

        from sequentia.preprocessing import IndependentFunctionTransformer, standardize
        from sequentia.datasets import load_digits

        data = load_digits()

        # Standardize each feature to zero mean and unit variance
        transform = IndependentFunctionTransformer(standardize, kw_args={"axis": 0})
        Xt = transform.transform(data.X, lengths=data.lengths)
    """
    eps = np.finfo(x.dtype).eps
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)
