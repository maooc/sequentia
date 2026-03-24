# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation splitters for sequence data.

These splitters are designed to work with sequence data where each sample is a
sequence of variable length. The splitters operate on the sequence level rather
than on individual time steps, and support scikit-learn's metadata routing for
passing sequence length information.
"""

from __future__ import annotations

import functools
import typing as t

import numpy as np
from sklearn.model_selection import (
    KFold as SKKFold,
    RepeatedKFold as SKRepeatedKFold,
    RepeatedStratifiedKFold as SKRepeatedStratifiedKFold,
    ShuffleSplit as SKShuffleSplit,
    StratifiedKFold as SKStratifiedKFold,
    StratifiedShuffleSplit as SKStratifiedShuffleSplit,
)
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
)

from sequentia._internal import _sklearn

__all__ = [
    "KFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
]


class _SequenceSplitterMixin:
    """Mixin class for sequence-aware cross-validation splitters.
    
    This mixin enables scikit-learn's metadata routing for sequence length
    information and adjusts the split logic to operate on sequences rather than
    individual time steps.
    """

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """Initialize metadata routing for subclasses."""
        super().__init_subclass__(**kwargs)
        # Wrap __init__ to set up metadata routing after initialization
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args: t.Any, **init_kwargs: t.Any) -> None:
            original_init(self, *args, **init_kwargs)
            # Enable metadata routing for lengths in split method
            # Using scikit-learn's internal metadata request mechanism
            if hasattr(self, "_get_metadata_request"):
                # Set the metadata request directly for the 'split' method
                metadata_request = {"lengths": True}
                # Store using the private attribute pattern used by scikit-learn
                attr_name = f"_{self.__class__.__name__}__metadata_request__split"
                if not hasattr(self, attr_name):
                    # Try the parent class name
                    for base in self.__class__.__mro__:
                        if base.__name__ in ["_BaseCrossValidator", "_BaseShuffleSplit"]:
                            attr_name = f"_{base.__name__}__metadata_request__split"
                            break
                if hasattr(self, attr_name):
                    getattr(self, attr_name).update(metadata_request)
                else:
                    # Fall back to setting it directly
                    setattr(
                        self,
                        attr_name,
                        metadata_request,
                    )

        cls.__init__ = wrapped_init

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing for this splitter.
        
        Returns
        -------
        MetadataRouter
            The metadata router configured to route ``lengths`` to the split method.
        """
        router = MetadataRouter(owner=self.__class__)
        router.add(
            method_mapping=MethodMapping().add(
                caller="split",
                callee="split",
            ),
        )
        return router

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        *,
        lengths: np.ndarray | None = None,
    ) -> t.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_total_time_steps, n_features)
            The combined sequence array.
            
        y : array-like of shape (n_sequences,) or (n_total_time_steps,), default=None
            The target variable. If shape is (n_total_time_steps,), it's assumed
            to be per-time-step labels and the sequence-level label is inferred
            based on the first element of each sequence.
            
        groups : array-like of shape (n_sequences,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            
        lengths : array-like of shape (n_sequences,), default=None
            Lengths of the sequences. If None, X is assumed to be a single sequence.
            
        Yields
        ------
        train : ndarray
            The training set indices for that split (sequence-level indices).
            
        test : ndarray
            The testing set indices for that split (sequence-level indices).
        """
        # Determine number of sequences from lengths or assume single sequence
        if lengths is None:
            # Try X's first dimension as a fallback if no lengths
            n_sequences = X.shape[0] if hasattr(X, 'shape') else len(X) if X is not None else 1
        else:
            n_sequences = len(lengths)
        
        # Create sequence-level indices
        seq_indices = np.arange(n_sequences)
        
        # Adjust y for splitting: use sequence-level labels
        if y is not None:
            if lengths is not None and len(y) == sum(lengths):
                # y is per-time-step, extract sequence-level labels (first element)
                seq_y = np.zeros(n_sequences, dtype=y.dtype)
                idx = 0
                for i, seq_len in enumerate(lengths):
                    seq_y[i] = y[idx]
                    idx += seq_len
            elif len(y) == n_sequences:
                # y is already sequence-level
                seq_y = y
            else:
                # Fall back to sequence count if y has incompatible shape
                seq_y = np.zeros(n_sequences, dtype=np.intp)
        else:
            # For unsupervised cases, use sequence indices as "labels" for splitting
            seq_y = seq_indices
        
        # Ensure seq_indices and seq_y have the correct shape and dimension
        # Parent split expects X to have shape (n_sequences, n_features)
        # So we pass dummy 2D array with proper sequence-level dimensions
        seq_indices_dummy = np.arange(n_sequences).reshape(-1, 1)
        
        # seq_y should have shape (n_sequences,) for proper label handling
        seq_y = np.asarray(seq_y).ravel()
        if len(seq_y) != n_sequences:
            seq_y = np.zeros(n_sequences, dtype=np.intp)
        
        # Use the parent splitter's split logic on sequence-level data
        # Note: We pass None for groups since we handle sequence-level indexing
        for train_seq_idx, test_seq_idx in super().split(seq_indices_dummy, seq_y, None):
            yield train_seq_idx, test_seq_idx


class KFold(_SequenceSplitterMixin, SKKFold):
    """K-Fold cross-validator for sequence data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining folds
    form the training set. Splits are performed on the sequence level, not on
    individual time steps.

    See Also
    --------
    :class:`sklearn.model_selection.KFold`
        The original scikit-learn implementation.
    """
    pass


class StratifiedKFold(_SequenceSplitterMixin, SKStratifiedKFold):
    """Stratified K-Fold cross-validator for sequence data.

    Provides train/test indices to split data in train/test sets. This
    cross-validation object is a variation of KFold that returns stratified
    folds. The folds are made by preserving the percentage of samples for each
    class. Splits are performed on the sequence level, not on individual time
    steps.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedKFold`
        The original scikit-learn implementation.
    """
    pass


class ShuffleSplit(_SequenceSplitterMixin, SKShuffleSplit):
    """Random permutation cross-validator for sequence data.

    Yields indices to split data into training and test sets. Note: contrary to
    other cross-validation strategies, random splits do not guarantee that test
    sets across all folds will be mutually exclusive, and might include
    overlapping samples. However, this is still very likely for sizeable
    datasets. Splits are performed on the sequence level, not on individual time
    steps.

    See Also
    --------
    :class:`sklearn.model_selection.ShuffleSplit`
        The original scikit-learn implementation.
    """
    pass


class StratifiedShuffleSplit(_SequenceSplitterMixin, SKStratifiedShuffleSplit):
    """Stratified ShuffleSplit cross-validator for sequence data.

    Provides train/test indices to split data in train/test sets. This
    cross-validation object is a merge of StratifiedKFold and ShuffleSplit,
    which returns stratified randomized folds. The folds are made by preserving
    the percentage of samples for each class. Splits are performed on the
    sequence level, not on individual time steps.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedShuffleSplit`
        The original scikit-learn implementation.
    """
    pass


class RepeatedKFold(_SequenceSplitterMixin, SKRepeatedKFold):
    """Repeated KFold cross-validator for sequence data.

    Repeats KFold n times with different randomization in each repetition.
    Splits are performed on the sequence level, not on individual time steps.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedKFold`
        The original scikit-learn implementation.
    """
    pass


class RepeatedStratifiedKFold(_SequenceSplitterMixin, SKRepeatedStratifiedKFold):
    """Repeated Stratified KFold cross-validator for sequence data.

    Repeats StratifiedKFold n times with different randomization in each
    repetition. Splits are performed on the sequence level, not on individual
    time steps.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedStratifiedKFold`
        The original scikit-learn implementation.
    """
    pass
