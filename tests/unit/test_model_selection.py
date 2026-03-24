# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GridSearchCV,
    HalvingGridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale

from sequentia._internal._data import SequentialArray
from sequentia.datasets import SequentialDataset, load_digits
from sequentia.enums import CovarianceMode, PriorMode, TopologyMode
from sequentia.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    param_grid,
)
from sequentia.models import (
    GaussianMixtureHMM,
    HMMClassifier,
    KNNClassifier,
    KNNRegressor,
)
from sequentia.preprocessing import IndependentFunctionTransformer

EPS: np.float32 = np.finfo(np.float32).eps
random_state: np.random.RandomState = np.random.RandomState(0)


def exp_weight(x: np.ndarray) -> np.ndarray:
    return np.exp(-x)


def inv_weight(x: np.ndarray) -> np.ndarray:
    return 1 / (x + EPS)


@pytest.fixture(scope="module")
def data() -> SequentialDataset:
    """Small subset of the spoken digits dataset."""
    digits = load_digits(digits={0, 1})
    _, digits = digits.split(
        test_size=0.1,
        random_state=random_state,
        shuffle=True,
        stratify=True,
    )
    return digits


@pytest.fixture(autouse=True)
def enable_metadata_routing():
    """Enable metadata routing for all tests."""
    sklearn.set_config(enable_metadata_routing=True)
    yield
    sklearn.set_config(enable_metadata_routing=False)


@pytest.mark.parametrize(
    "cv",
    [
        KFold,
        StratifiedKFold,
        ShuffleSplit,
        StratifiedShuffleSplit,
        RepeatedKFold,
        RepeatedStratifiedKFold,
    ],
)
@pytest.mark.parametrize(
    "search", [GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV]
)
def test_knn_classifier(
    data: SequentialDataset,
    search: type[GridSearchCV] | type[RandomizedSearchCV] | type[HalvingGridSearchCV],
    cv: type[BaseCrossValidator] | type[BaseShuffleSplit],
) -> None:
    cv_kwargs = {"random_state": 0, "n_splits": 2}
    if cv in (KFold, StratifiedKFold):
        cv_kwargs["shuffle"] = True

    seq_data = SequentialArray(data.X, data.lengths)

    optimizer = search(
        Pipeline(
            [
                ("scale", IndependentFunctionTransformer(minmax_scale)),
                ("knn", KNNClassifier(use_c=True, n_jobs=-1)),
            ]
        ),
        {
            "knn__k": [1, 5],
            "knn__weighting": [exp_weight, inv_weight],
        },
        cv=cv(**cv_kwargs),
        n_jobs=-1,
    )

    optimizer.fit(seq_data, data.y, lengths=data.lengths)
    assert optimizer.best_score_ > 0.8
    clf = optimizer.best_estimator_

    y_pred = clf.predict(data.X, lengths=data.lengths)
    assert np.isin(y_pred, (0, 1)).all()

    y_probs = clf.predict_proba(data.X, lengths=data.lengths)
    assert ((y_probs >= 0) & (y_probs <= 1)).all()
    npt.assert_almost_equal(y_probs.sum(axis=1), 1.0)

    y_log_probs = clf.predict_log_proba(data.X, lengths=data.lengths)
    assert (y_log_probs <= 0).all()
    npt.assert_almost_equal(y_log_probs, np.log(y_probs))

    acc = clf.score(data.X, data.y, lengths=data.lengths)
    assert acc > 0.8


@pytest.mark.parametrize(
    "cv",
    [
        KFold,
        StratifiedKFold,
        ShuffleSplit,
        StratifiedShuffleSplit,
        RepeatedKFold,
        RepeatedStratifiedKFold,
    ],
)
@pytest.mark.parametrize(
    "search", [GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV]
)
def test_knn_regressor(
    data: SequentialDataset,
    search: type[GridSearchCV] | type[RandomizedSearchCV] | type[HalvingGridSearchCV],
    cv: type[BaseCrossValidator] | type[BaseShuffleSplit],
) -> None:
    cv_kwargs = {"random_state": 0, "n_splits": 2}
    if cv in (KFold, StratifiedKFold):
        cv_kwargs["shuffle"] = True

    seq_data = SequentialArray(data.X, data.lengths)

    optimizer = search(
        Pipeline(
            [
                ("scale", IndependentFunctionTransformer(minmax_scale)),
                ("knn", KNNRegressor(use_c=True, n_jobs=-1)),
            ]
        ),
        {
            "knn__k": [3, 5],
            "knn__weighting": [exp_weight, inv_weight],
        },
        cv=cv(**cv_kwargs),
        n_jobs=-1,
    )

    y = data.y.astype(np.float64)

    optimizer.fit(seq_data, y, lengths=data.lengths)
    assert optimizer.best_score_ > 0.8
    model = optimizer.best_estimator_

    y_pred = model.predict(data.X, lengths=data.lengths)
    assert ((y_pred >= 0) & (y_pred <= 1)).all()

    r2 = model.score(data.X, y, lengths=data.lengths)
    assert r2 > 0.8


def test_hmm_classifier(data: SequentialDataset) -> None:
    seq_data = SequentialArray(data.X, data.lengths)

    optimizer = GridSearchCV(
        estimator=Pipeline(
            [
                ("scale", IndependentFunctionTransformer(minmax_scale)),
                ("clf", HMMClassifier(variant=GaussianMixtureHMM, n_jobs=-1)),
            ]
        ),
        param_grid={
            "clf__prior": [PriorMode.UNIFORM, PriorMode.FREQUENCY],
            "clf__model_kwargs": param_grid(
                n_states=[3, 4, 5],
                n_components=[2, 3, 4],
                covariance=[CovarianceMode.DIAGONAL, CovarianceMode.SPHERICAL],
                topology=[TopologyMode.LEFT_RIGHT, TopologyMode.LINEAR],
            ),
        },
        cv=StratifiedKFold(),
        n_jobs=-1,
    )

    optimizer.fit(seq_data, data.y, lengths=data.lengths)
    assert optimizer.best_score_ > 0.8
    clf = optimizer.best_estimator_

    y_pred = clf.predict(data.X, lengths=data.lengths)
    assert np.isin(y_pred, (0, 1)).all()

    y_probs = clf.predict_proba(data.X, lengths=data.lengths)
    assert ((y_probs >= 0) & (y_probs <= 1)).all()
    npt.assert_almost_equal(y_probs.sum(axis=1), 1.0)

    clf.predict_log_proba(data.X, lengths=data.lengths)

    acc = clf.score(data.X, data.y, lengths=data.lengths)
    assert acc > 0.8


def test_cv_splitter(data: SequentialDataset) -> None:
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    splits = list(cv.split(y=data.y))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert len(train_idx) + len(test_idx) == len(data.y)
        assert len(set(train_idx) & set(test_idx)) == 0


def test_kfold_basic() -> None:
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    y = np.array([0, 0, 0, 1, 1, 1])
    splits = list(cv.split(y=y))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) == 4
        assert len(test_idx) == 2


def test_stratified_kfold_basic() -> None:
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    y = np.array([0, 0, 0, 1, 1, 1])
    splits = list(cv.split(y=y))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert len(train_idx) == 3
        assert len(test_idx) == 3


def test_sequential_array_indexing() -> None:
    X = np.random.rand(100, 5)
    lengths = np.array([30, 40, 30])
    seq_data = SequentialArray(X, lengths)

    assert len(seq_data) == 3
    assert seq_data.shape == (3,)
    assert seq_data.n_observations == 100
    assert seq_data.n_features == 5

    subset = seq_data[[0, 2]]
    assert len(subset) == 2
    assert subset.n_observations == 60

    single = seq_data[1]
    assert single.shape == (40, 5)


def test_cross_val_score_with_sequential_array(data: SequentialDataset) -> None:
    from sklearn.model_selection import cross_val_score

    seq_data = SequentialArray(data.X, data.lengths)

    pipe = Pipeline([
        ("scale", IndependentFunctionTransformer(minmax_scale)),
        ("clf", KNNClassifier(k=1, use_c=True)),
    ])

    scores = cross_val_score(
        pipe, seq_data, data.y,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
        params={'lengths': data.lengths}
    )

    assert len(scores) == 2
    assert all(0 <= s <= 1 for s in scores)
