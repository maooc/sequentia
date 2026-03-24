"""Sequence data wrapper that provides a view-like indexing mechanism.

This module enables transparent sequence data handling in scikit-learn pipelines
by providing a dummy array that passes sklearn's dimension checks but carries
the necessary information to reconstruct actual sequence data subsets.
"""

from __future__ import annotations

import functools
import typing as t

import numpy as np

from sequentia._internal import _data
from sequentia._internal._typing import Array, IntArray


class SequenceIndexProxy:
    """A proxy array that carries sequence information through sklearn indexing.
    
    This class behaves like a numpy array for sklearn's validation checks
    but carries the actual sequence data and lengths. When estimators receive
    this object, they can resolve the actual data via resolve_sequence_data.
    """
    
    __array_ufunc__ = None  # Disable numpy ufuncs
    
    def __init__(
        self,
        X: Array,
        *,
        lengths: IntArray,
    ) -> None:
        """Initialize the sequence index proxy.
        
        Parameters
        ----------
        X:
            Full sequence array (concatenated time steps).
            
        lengths:
            Lengths of each sequence.
        """
        self._X_full = X
        self._lengths_full = lengths
        self._n_sequences = len(lengths)
        self._indices = np.arange(self._n_sequences)
        
    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape for sklearn validation (n_sequences, 1)."""
        return (self._n_sequences, 1)
    
    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return 2
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return self._n_sequences
    
    def __getitem__(self, key: t.Any) -> _SequenceIndexView:
        """Create a view of the sequence data based on indices."""
        if isinstance(key, (int, np.integer)):
            key = [key]
        return _SequenceIndexView(
            self._X_full,
            lengths=self._lengths_full,
            indices=self._indices[key],
        )
    
    def __array__(self) -> np.ndarray:
        """Convert to a dummy numpy array for sklearn validation."""
        return np.empty(self.shape)


class _SequenceIndexView:
    """A view on a subset of sequences."""
    
    __array_ufunc__ = None
    
    def __init__(
        self,
        X_full: Array,
        *,
        lengths: IntArray,
        indices: IntArray,
    ) -> None:
        self._X_full = X_full
        self._lengths_full = lengths
        self._indices = indices
        self._n_sequences_view = len(indices)
        
    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of this view."""
        return (self._n_sequences_view, 1)
    
    @property
    def ndim(self) -> int:
        return 2
    
    def __len__(self) -> int:
        return self._n_sequences_view
    
    def __array__(self) -> np.ndarray:
        """Convert to dummy array when needed."""
        return np.empty(self.shape)


def _adapt_sequence_data(
    X: Array,
    *,
    lengths: IntArray,
    indices: IntArray | None = None,
) -> tuple[Array, IntArray]:
    """Adapt sequence data by extracting a subset based on sequence indices."""
    if indices is None:
        return X, lengths
    
    # Get the start/end indices of each sequence
    idxs = _data.get_idxs(lengths)
    
    # Subset the sequence indices
    idxs_subset = idxs[indices]
    lengths_subset = lengths[indices]
    
    # Concatenate the selected sequences
    X_subset = np.concatenate(list(_data.iter_X(X, idxs=idxs_subset)))
    
    return X_subset, lengths_subset


def resolve_sequence_data(
    X: Array | SequenceIndexProxy | _SequenceIndexView,
    lengths: IntArray | None = None,
) -> tuple[Array, IntArray]:
    """Extract actual sequence data and lengths.
    
    This utility resolves sequence data from a potentially wrapped X.
    If X is a proxy/view, reconstruct the actual data.
    If X is a regular array and lengths are provided (metadata routing), return them.
    
    Returns
    -------
    X_resolved:
        The actual (possibly subsetted) sequence array.
        
    lengths_resolved:
        The lengths corresponding to the resolved sequence data.
    """
    if isinstance(X, SequenceIndexProxy):
        return X._X_full, X._lengths_full
    elif isinstance(X, _SequenceIndexView):
        return _adapt_sequence_data(
            X._X_full,
            lengths=X._lengths_full,
            indices=X._indices,
        )
    
    # Regular array - return as-is with provided lengths
    if lengths is None:
        raise ValueError(
            "Sequence model requires 'lengths' parameter when X is a regular array. "
            "Use set_fit_request(lengths=True) etc. to enable metadata routing, or "
            "pass X through a SequenceIndexProxy for cross-validation."
        )
    return X, lengths


def with_resolved_sequence_data(method: t.Callable) -> t.Callable:
    """Decorator that automatically resolves sequence data before method calls.
    
    This decorator should be applied to estimator methods (fit, predict, score)
    that need to handle potentially wrapped sequence data. It resolves the actual
    X and lengths and calls the method with the resolved data.
    
    Example
    -------
    >>> class MyEstimator:
    ...     @with_resolved_sequence_data
    ...     def fit(self, X, y, lengths=None):
    ...         # X and lengths are now resolved to actual values
    ...         pass
    """
    @functools.wraps(method)
    def wrapped(
        self: t.Any,
        X: Array | SequenceIndexProxy | _SequenceIndexView,
        *args: t.Any,
        lengths: IntArray | None = None,
        **kwargs: t.Any,
    ) -> t.Any:
        X_resolved, lengths_resolved = resolve_sequence_data(X, lengths)
        return method(self, X_resolved, *args, lengths=lengths_resolved, **kwargs)
    return wrapped
