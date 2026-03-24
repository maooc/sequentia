# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation splitters and utilities for sequence data.

This module provides:
- Cross-validation splitters that operate on sequence indices
- The param_grid utility for generating hyper-parameter grids

For hyper-parameter search, use sklearn's GridSearchCV, RandomizedSearchCV, etc.
directly with metadata routing enabled:

    import sklearn
    sklearn.set_config(enable_metadata_routing=True)
    
    from sklearn.model_selection import GridSearchCV
    from sequentia.model_selection import KFold, param_grid
    from sequentia.models import KNNClassifier
    
    clf = KNNClassifier()
    search = GridSearchCV(clf, {"k": [1, 3, 5]}, cv=KFold(n_splits=3))
    search.fit(X, y, lengths=lengths)  # lengths is routed automatically
"""

from sequentia.model_selection._search import param_grid
from sequentia.model_selection._split import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

__all__ = [
    "KFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "param_grid",
]
