# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Successive halving hyperparameter search utilities for sequence models.

This module provides wrappers around scikit-learn's successive halving
hyperparameter search utilities that are compatible with sequence data.
The main difference from standard scikit-learn search utilities is the
handling of sequence lengths through metadata routing.
"""

from __future__ import annotations

import typing as t

# Enable experimental successive halving search
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    HalvingGridSearchCV as SKHalvingGridSearchCV,
    HalvingRandomSearchCV as SKHalvingRandomSearchCV,
)

from sequentia._internal._typing import Array
from sequentia.model_selection._search import _SequenceSearchCVMixin

__all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]


class HalvingGridSearchCV(_SequenceSearchCVMixin, SKHalvingGridSearchCV):
    """Search over specified parameter values with successive halving.

    This is a sequence-aware version of
    :class:`sklearn.model_selection.HalvingGridSearchCV`
    that properly handles variable-length sequence data through metadata routing.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which
        case the grids spanned by each dictionary in the list are explored.

    factor : int or float, default=3
        The determining factor of the resource allocation between iterations.

    resource : str, default="n_samples"
        Defines the resource to use for successive halving.

    max_resources : int, default=None
        The maximum number of resources that any candidate is allowed to use
        for a given iteration.

    min_resources : {"exhaust", "smallest"} or int, default="exhaust"
        The minimum amount of resources that candidates are allocated
        at the first iteration.

    aggressive_elimination : bool, default=False
        Whether to perform aggressive elimination during the first iteration.

    cv : int, cross-validation generator, or an iterable, default=None
        Determines the cross-validation splitting strategy. For sequence data,
        use a splitter from :mod:`sequentia.model_selection`.

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole dataset.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training scores.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state.

    n_jobs : int, default=None
        Number of jobs to run in parallel.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each iteration.

    n_remaining_candidates_ : int
        The number of candidates remaining after the last iteration.

    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration.

    min_resources_ : int
        The minimum amount of resources that candidates are allocated
        at the first iteration.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns.

    best_estimator_ : estimator
        Estimator that was chosen by the search.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index which corresponds to the best candidate parameter setting.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best model.

    n_splits_ : int
        The number of cross-validation splits used.

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

    multimetric_ : bool
        Whether the scorer returns multiple metrics.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    See Also
    --------
    :class:`sklearn.model_selection.HalvingGridSearchCV`
        The original scikit-learn implementation.

    :class:`HalvingRandomSearchCV`:
        Randomized search on hyper parameters with successive halving.
    """

    pass


class HalvingRandomSearchCV(_SequenceSearchCVMixin, SKHalvingRandomSearchCV):
    """Randomized search on hyper parameters with successive halving.

    This is a sequence-aware version of
    :class:`sklearn.model_selection.HalvingRandomSearchCV`
    that properly handles variable-length sequence data through metadata routing.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (str) as keys and distributions or
        lists of parameters to try.

    n_iter : int, default=10
        Number of parameter settings that are sampled.

    factor : int or float, default=3
        The determining factor of the resource allocation between iterations.

    resource : str, default="n_samples"
        Defines the resource to use for successive halving.

    max_resources : int, default=None
        The maximum number of resources that any candidate is allowed to use
        for a given iteration.

    min_resources : {"exhaust", "smallest"} or int, default="exhaust"
        The minimum amount of resources that candidates are allocated
        at the first iteration.

    aggressive_elimination : bool, default=False
        Whether to perform aggressive elimination during the first iteration.

    cv : int, cross-validation generator, or an iterable, default=None
        Determines the cross-validation splitting strategy. For sequence data,
        use a splitter from :mod:`sequentia.model_selection`.

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole dataset.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training scores.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state.

    n_jobs : int, default=None
        Number of jobs to run in parallel.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    n_resources_ : list of int
        The amount of resources used at each iteration.

    n_candidates_ : list of int
        The number of candidate parameters that were evaluated at each iteration.

    n_remaining_candidates_ : int
        The number of candidates remaining after the last iteration.

    max_resources_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration.

    min_resources_ : int
        The minimum amount of resources that candidates are allocated
        at the first iteration.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns.

    best_estimator_ : estimator
        Estimator that was chosen by the search.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index which corresponds to the best candidate parameter setting.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best model.

    n_splits_ : int
        The number of cross-validation splits used.

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

    multimetric_ : bool
        Whether the scorer returns multiple metrics.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    See Also
    --------
    :class:`sklearn.model_selection.HalvingRandomSearchCV`
        The original scikit-learn implementation.

    :class:`HalvingGridSearchCV`:
        Search over specified parameter values with successive halving.
    """

    pass
