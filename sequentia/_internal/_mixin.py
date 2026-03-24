# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Mixin classes for sequence-aware estimators and transformers."""

from __future__ import annotations

import typing as t

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from sequentia._internal import _data, _validation
from sequentia._internal._typing import Array, IntArray

__all__ = [
    "EstimatorMixin",
    "TransformerMixin",
    "SequenceEstimatorMixin",
    "SequenceTransformerMixin",
]


def _safe_set_request(estimator: t.Any, method: str, **kwargs: t.Any) -> None:
    """Safely set metadata request if routing is enabled."""
    try:
        setter = getattr(estimator, f"set_{method}_request", None)
        if setter is not None:
            setter(**kwargs)
    except RuntimeError:
        # Metadata routing not enabled, skip
        pass


class EstimatorMixin(BaseEstimator):
    """Base mixin for all estimators."""

    def _more_tags(self: t.Self) -> dict[str, bool]:
        return {"allow_nan": True, "no_validation": True}


class TransformerMixin(EstimatorMixin, TransformerMixin):
    """Base mixin for all transformers."""

    def fit_transform(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        """Fit the transformer and transform the input data.

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
        if lengths is not None:
            return self.fit(X, y, lengths=lengths).transform(X, lengths=lengths)
        return self.fit(X, y).transform(X)


class SequenceEstimatorMixin(EstimatorMixin):
    """Mixin for sequence-aware estimators that handles length metadata routing."""

    # Class-level metadata request configuration for sklearn's metadata routing
    # This tells sklearn that the 'lengths' parameter is requested for these methods
    __metadata_request__fit = {"lengths": True}
    __metadata_request__predict = {"lengths": True}
    __metadata_request__predict_proba = {"lengths": True}
    __metadata_request__predict_log_proba = {"lengths": True}
    __metadata_request__score = {"lengths": True}

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip registration for the mixin classes themselves
        if cls.__name__ not in ("SequenceEstimatorMixin", "SequenceTransformerMixin"):
            cls._register_routing()

    @classmethod
    def _register_routing(cls) -> None:
        """Register metadata routing for lengths parameter.
        
        This method ensures that subclasses properly inherit and propagate
        the metadata request configuration.
        """
        # Ensure metadata routing is properly configured at the class level
        # by inheriting from the parent class and updating with our own
        for method in ["fit", "predict", "predict_proba", "predict_log_proba", "score"]:
            attr_name = f"__metadata_request__{method}"
            # Start with parent's configuration if available
            parent_config = {}
            for base in cls.__mro__[1:]:
                if hasattr(base, attr_name):
                    parent_config = getattr(base, attr_name, {}).copy()
                    break
            # Add our configuration
            our_config = getattr(cls, attr_name, {})
            if our_config:
                parent_config.update(our_config)
                setattr(cls, attr_name, parent_config)

    def _validate_sequence_data(
        self,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
        reset: bool = True,
        dtype: np.float64 | np.int64 = np.float64,
    ) -> tuple[Array, Array | None, IntArray]:
        """Validate sequence data and lengths.

        Parameters
        ----------
        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

        reset:
            Whether to reset the internal feature names.

        dtype:
            Target dtype for X.

        Returns
        -------
        tuple[Array, Array | None, IntArray]
            Validated X, y (if provided), and lengths.
        """
        X, lengths = _validation.check_X_lengths(X, lengths=lengths, dtype=dtype)
        if y is not None:
            y = _validation.check_y(y, lengths=lengths)
        if hasattr(self, "_validate_data"):
            X = self._validate_data(X, reset=reset, dtype=None)
        return X, y, lengths


class SequenceTransformerMixin(SequenceEstimatorMixin, TransformerMixin):
    """Mixin for sequence-aware transformers that handles length metadata routing."""

    # Additional metadata request configuration for transformer methods
    __metadata_request__transform = {"lengths": True}
    __metadata_request__inverse_transform = {"lengths": True}
    __metadata_request__fit_transform = {"lengths": True}

    @classmethod
    def _register_routing(cls) -> None:
        """Register metadata routing for lengths parameter including transform methods."""
        super()._register_routing()
        
        # Ensure transformer methods are properly configured
        for method in ["transform", "inverse_transform", "fit_transform"]:
            attr_name = f"__metadata_request__{method}"
            # Start with parent's configuration if available
            parent_config = {}
            for base in cls.__mro__[1:]:
                if hasattr(base, attr_name):
                    parent_config = getattr(base, attr_name, {}).copy()
                    break
            # Add our configuration
            our_config = getattr(cls, attr_name, {})
            if our_config:
                parent_config.update(our_config)
                setattr(cls, attr_name, parent_config)

    def _transform_independent(
        self,
        X: Array,
        *,
        lengths: IntArray | None,
        func: t.Callable[[Array], Array],
        **kwargs: t.Any,
    ) -> Array:
        """Apply a function independently to each sequence in X.

        This method provides optimized vectorized processing of sequences by utilizing
        the underlying structure of the data without explicit loops where possible.

        Parameters
        ----------
        X:
            Concatenated sequence array.

        lengths:
            Lengths of each sequence. If None, X is treated as a single sequence.

        func:
            Function to apply to each sequence.

        **kwargs:
            Additional keyword arguments to pass to func.

        Returns
        -------
        Array
            Transformed array with the same structure as X.
        """
        if lengths is None:
            return func(X, **kwargs)
        
        idxs = _data.get_idxs(lengths)
        n_sequences = len(lengths)
        
        if n_sequences == 0:
            return X
        
        # Test apply to first sequence to determine output shape and type
        first_seq = X[idxs[0, 0]:idxs[0, 1]]
        first_result = func(first_seq, **kwargs)
        
        # Check if function preserves number of timesteps per sequence
        func_preserves_timesteps = first_result.shape[0] == first_seq.shape[0]
        
        if func_preserves_timesteps:
            # Same number of timesteps per sequence
            # Can use pre-allocated array
            output_shape = list(X.shape)
            output_shape[1:] = first_result.shape[1:]
            output = np.empty(output_shape, dtype=first_result.dtype)
            
            # Apply function to first sequence
            output[idxs[0, 0]:idxs[0, 1]] = first_result
            
            # Process remaining sequences
            for i in range(1, n_sequences):
                start, end = idxs[i]
                output[start:end] = func(X[start:end], **kwargs)
            
            return output
        else:
            # Function changes timesteps per sequence (e.g., downsampling)
            # Must use list concatenation since output
            results = [first_result]
            for i in range(1, n_sequences):
                start, end = idxs[i]
                results.append(func(X[start:end], **kwargs))
            return np.concatenate(results)


