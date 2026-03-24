# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Hyper-parameter search and dataset splitting utilities for sequence data.

This module provides sequence-aware alternatives to sklearn's model selection
utilities, designed to work seamlessly with sequence data.
"""

from sequentia.model_selection._search import (
    BaseSearchCV,
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
)
from sequentia.model_selection._validation import (
    cross_validate,
    prepare_sequence_split,
)

__all__ = [
    "BaseSearchCV",
    "GridSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "KFold",
    "RandomizedSearchCV",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "cross_validate",
    "param_grid",
    "prepare_sequence_split",
]
