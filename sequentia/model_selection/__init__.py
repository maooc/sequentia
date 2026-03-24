# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Hyper-parameter search and dataset splitting utilities."""

from sequentia.model_selection._adapter import (
    SequenceClassifierAdapter,
    SequenceEstimatorAdapter,
    SequenceRegressorAdapter,
    make_sequence_scorer,
)
from sequentia.model_selection._search import (
    GridSearchCV,
    RandomizedSearchCV,
    param_grid,
)
from sequentia.model_selection._search_successive_halving import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
from sequentia.model_selection._split import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sequentia.model_selection._validation import (
    cross_val_predict,
    cross_val_score,
)

__all__ = [
    "GridSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "KFold",
    "RandomizedSearchCV",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "SequenceClassifierAdapter",
    "SequenceEstimatorAdapter",
    "SequenceRegressorAdapter",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "TimeSeriesSplit",
    "cross_val_predict",
    "cross_val_score",
    "make_sequence_scorer",
    "param_grid",
]
