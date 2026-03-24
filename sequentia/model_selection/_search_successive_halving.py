# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Sequence-aware successive halving search using scikit-learn's public API only.

This module provides HalvingGridSearchCV and HalvingRandomSearchCV that work with
sequence data through scikit-learn's metadata routing mechanism.
"""

# Import experimental feature first
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV as _HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV as _HalvingRandomSearchCV

__all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]


class HalvingGridSearchCV(_HalvingGridSearchCV):
    """Search over specified parameter values with successive halving.

    This is a thin wrapper around sklearn's HalvingGridSearchCV that ensures
    compatibility with sequence data through metadata routing.

    To use with sequence data:
    1. Ensure your estimator/transformer accepts `lengths` in fit/score/predict
    2. Pass `lengths` to the fit method
    3. Enable metadata routing: sklearn.set_config(enable_metadata_routing=True)

    See Also
    --------
    :class:`sklearn.model_selection.HalvingGridSearchCV`
        This class inherits from sklearn's HalvingGridSearchCV.
    """


class HalvingRandomSearchCV(_HalvingRandomSearchCV):
    """Randomized search on hyper parameters with successive halving.

    This is a thin wrapper around sklearn's HalvingRandomSearchCV that ensures
    compatibility with sequence data through metadata routing.

    To use with sequence data:
    1. Ensure your estimator/transformer accepts `lengths` in fit/score/predict
    2. Pass `lengths` to the fit method
    3. Enable metadata routing: sklearn.set_config(enable_metadata_routing=True)

    See Also
    --------
    :class:`sklearn.model_selection.HalvingRandomSearchCV`
        This class inherits from sklearn's HalvingRandomSearchCV.
    """
