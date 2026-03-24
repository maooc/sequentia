# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Successive halving search utilities using sklearn's public API.

This module provides wrappers around sklearn's successive halving search
classes that properly handle sequence data with lengths parameter.
"""

from __future__ import annotations

# Enable experimental successive halving search
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV as SklearnHalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV as SklearnHalvingRandomSearchCV

from sequentia.model_selection._search import BaseSearchCV

__all__ = ["HalvingGridSearchCV", "HalvingRandomSearchCV"]


class HalvingGridSearchCV(SklearnHalvingGridSearchCV):
    """Search over specified parameter values with successive halving.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.HalvingGridSearchCV`
        :class:`.HalvingGridSearchCV` is a modified version
        of this class that supports sequences.
    """
    
    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.
        
        Overrides the parent fit to properly handle sequence data
        with lengths parameter.
        """
        # Store lengths for use during fitting
        self._fit_params = params
        return super().fit(X, y, **params)


class HalvingRandomSearchCV(SklearnHalvingRandomSearchCV):
    """Randomized search on hyper parameters with successive halving.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.HalvingRandomSearchCV`
        :class:`.HalvingRandomSearchCV` is a modified version
        of this class that supports sequences.
    """
    
    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.
        
        Overrides the parent fit to properly handle sequence data
        with lengths parameter.
        """
        # Store lengths for use during fitting
        self._fit_params = params
        return super().fit(X, y, **params)
