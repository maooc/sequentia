# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Hyper-parameter search and dataset splitting utilities."""

from sequentia.model_selection._adapter import SequenceAdapter
from sequentia.model_selection._metadata_router import enable_sequence_routing
from sequentia.model_selection._search import (
    GridSearchCV,
    RandomizedSearchCV,
    param_grid,
)
from sequentia.model_selection._search_successive_halving import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
from sequentia.model_selection._sequence_dataset import (
    SequenceAwareEstimator,
    SequenceCVSplitter,
    SequenceDataset,
)
from sequentia.model_selection._split import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

__all__ = [
    "GridSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "KFold",
    "RandomizedSearchCV",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "SequenceAdapter",
    "SequenceAwareEstimator",
    "SequenceCVSplitter",
    "SequenceDataset",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "enable_sequence_routing",
    "param_grid",
]
