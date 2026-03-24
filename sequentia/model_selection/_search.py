# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Hyper-parameter search utilities for sequence data.

This module provides search utilities that work with scikit-learn's public API,
without relying on any private sklearn modules.

The key insight is that when using scikit-learn's metadata routing (enabled via
sklearn.set_config(enable_metadata_routing=True)), the lengths metadata is
automatically routed through the standard GridSearchCV/RandomizedSearchCV
to the estimators that request it.
"""

from __future__ import annotations

import typing as t
from itertools import product

__all__ = ["param_grid"]


def param_grid(**kwargs: list[t.Any]) -> list[dict[str, t.Any]]:
    """Generates a hyper-parameter grid for a nested object.

    Examples
    --------
    Using :func:`.param_grid` in a grid search to cross-validate over
    settings for :class:`.GaussianMixtureHMM`, which is a nested model
    specified in the constructor of a :class:`.HMMClassifier`. ::

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import minmax_scale

        from sequentia.enums import PriorMode, CovarianceMode, TopologyMode
        from sequentia.models import HMMClassifier, GaussianMixtureHMM
        from sequentia.preprocessing import IndependentFunctionTransformer
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        GridSearchCV(
            estimator=Pipeline(
                [
                    ("scale", IndependentFunctionTransformer(minmax_scale)),
                    ("clf", HMMClassifier(variant=GaussianMixtureHMM)),
                ]
            ),
            param_grid={
                "clf__prior": [PriorMode.UNIFORM, PriorMode.FREQUENCY],
                "clf__model_kwargs": param_grid(
                    n_states=[3, 5, 7],
                    n_components=[2, 3, 4],
                    covariance=[
                        CovarianceMode.DIAGONAL, CovarianceMode.SPHERICAL
                    ],
                    topology=[
                        TopologyMode.LEFT_RIGHT, TopologyMode.LINEAR
                    ],
                )
            },
            cv=StratifiedKFold(),
        )

    Parameters
    ----------
    **kwargs:
        Hyper-parameter name and corresponding values.

    Returns
    -------
    Hyper-parameter grid for a nested object.
    """
    return [
        dict(zip(kwargs.keys(), values))
        for values in product(*kwargs.values())
    ]
