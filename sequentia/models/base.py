# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Base classifier and regressor mixin classes."""

from __future__ import annotations

import abc
import typing as t

import numpy as np
import sklearn.base
import sklearn.metrics

from sequentia._internal import _sklearn, _validation
from sequentia._internal._data import SequentialArray
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = ["ClassifierMixin", "RegressorMixin"]


class ClassifierMixin(
    sklearn.base.BaseEstimator,
    sklearn.base.ClassifierMixin,
    metaclass=abc.ABCMeta,
):
    """Represents a generic sequential classifier.

    This class provides the foundation for all sequential classifiers in
    sequentia, implementing scikit-learn's metadata routing protocol for
    transparent handling of sequence lengths metadata.

    Subclasses must implement the abstract methods:
    - fit(X, y, *, lengths=None)
    - predict(X, *, lengths=None)
    - predict_proba(X, *, lengths=None)
    - predict_scores(X, *, lengths=None)
    """

    @abc.abstractmethod
    def fit(
        self,
        X: Array,
        y: IntArray,
        *,
        lengths: IntArray | None = None,
    ) -> t.Self:
        """Fit the classifier with the provided sequences and outputs.

        Parameters
        ----------
        X : Array
            Observation sequences, either as:
            - A SequentialArray container
            - A concatenated array with shape (n_observations, n_features)
        y : IntArray
            Class labels for each sequence.
        lengths : IntArray, optional
            Lengths of each sequence. Required if X is a concatenated array.

        Returns
        -------
        self
            The fitted classifier.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> IntArray:
        """Predict outputs for the provided sequences."""
        raise NotImplementedError

    def fit_predict(
        self,
        X: Array,
        y: IntArray,
        *,
        lengths: IntArray | None = None,
    ) -> IntArray:
        """Fit the model to the sequence(s) in ``X`` and predicts outputs for
        ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Output predictions.
        """
        return self.fit(X, y, lengths=lengths).predict(X, lengths=lengths)

    @abc.abstractmethod
    def predict_proba(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predict class probabilities for the provided sequences."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_scores(
        self,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predict class scores for the provided sequences."""
        raise NotImplementedError

    @_validation.requires_fit
    def score(
        self,
        X: Array,
        y: IntArray,
        *,
        lengths: IntArray | None = None,
        normalize: bool = True,
        sample_weight: Array | None = None,
    ) -> float:
        """Calculate the predictive accuracy for the sequence(s) in ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        normalize:
            See :func:`sklearn:sklearn.metrics.accuracy_score`.

        sample_weight:
            See :func:`sklearn:sklearn.metrics.accuracy_score`.

        Returns
        -------
        float
            Predictive accuracy.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        y = _validation.check_y(y, lengths=lengths, dtype=np.int8)
        y_pred = self.predict(X, lengths=lengths)
        return sklearn.metrics.accuracy_score(
            y, y_pred, normalize=normalize, sample_weight=sample_weight
        )

    def get_metadata_routing(self):
        """Get metadata routing for this estimator.

        Returns
        -------
        MetadataRouter
            A MetadataRouter encapsulating routing information.
        """
        from sklearn.utils.metadata_routing import MetadataRouter

        router = MetadataRouter(owner=self).add_self_request(self)
        return router

    def _setup_metadata_routing(self) -> None:
        """Set up metadata routing for lengths.

        This should be called in __init__ after parameter setup.
        """
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_predict_request(lengths=True)
            self.set_predict_proba_request(lengths=True)
            self.set_predict_log_proba_request(lengths=True)
            self.set_score_request(lengths=True, normalize=True, sample_weight=True)

    @staticmethod
    def _extract_X_lengths(
        X: Array | SequentialArray,
        lengths: IntArray | None = None,
    ) -> tuple[Array, IntArray]:
        """Extract X and lengths from input, handling SequentialArray.

        Parameters
        ----------
        X : Array or SequentialArray
            Input data.
        lengths : IntArray, optional
            Sequence lengths.

        Returns
        -------
        tuple[Array, IntArray]
            (X, lengths) tuple.
        """
        if isinstance(X, SequentialArray):
            return X.X, X.lengths
        return X, lengths


class RegressorMixin(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Represents a generic sequential regressor.

    This class provides the foundation for all sequential regressors in
    sequentia, implementing scikit-learn's metadata routing protocol for
    transparent handling of sequence lengths metadata.

    Subclasses must implement the abstract methods:
    - fit(X, y, *, lengths=None)
    - predict(X, *, lengths=None)
    """

    @abc.abstractmethod
    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> t.Self:
        """Fit the regressor with the provided sequences and outputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self, X: FloatArray, lengths: IntArray | None = None
    ) -> FloatArray:
        """Predict outputs for the provided sequences."""
        raise NotImplementedError

    def fit_predict(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Fit the model to the sequence(s) in ``X`` and predicts outputs for
        ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray
            Output predictions.
        """
        return self.fit(X, y, lengths=lengths).predict(X, lengths=lengths)

    @_validation.requires_fit
    def score(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        lengths: IntArray | None = None,
        sample_weight: Array | None = None,
    ) -> float:
        r"""Calculate the predictive coefficient of determination
        (R\ :sup:`2`) for the sequence(s) in ``X``.

        Parameters
        ----------
        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        sample_weight:
            See :func:`sklearn:sklearn.metrics.r2_score`.

        Returns
        -------
        float
            Coefficient of determination.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        y = _validation.check_y(y, lengths=lengths, dtype=np.float64)
        y_pred = self.predict(X, lengths=lengths)
        return sklearn.metrics.r2_score(y, y_pred, sample_weight=sample_weight)

    def get_metadata_routing(self):
        """Get metadata routing for this estimator.

        Returns
        -------
        MetadataRouter
            A MetadataRouter encapsulating routing information.
        """
        from sklearn.utils.metadata_routing import MetadataRouter

        router = MetadataRouter(owner=self).add_self_request(self)
        return router

    def _setup_metadata_routing(self) -> None:
        """Set up metadata routing for lengths.

        This should be called in __init__ after parameter setup.
        """
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_predict_request(lengths=True)
            self.set_score_request(lengths=True, sample_weight=True)

    @staticmethod
    def _extract_X_lengths(
        X: Array | SequentialArray,
        lengths: IntArray | None = None,
    ) -> tuple[Array, IntArray]:
        """Extract X and lengths from input, handling SequentialArray.

        Parameters
        ----------
        X : Array or SequentialArray
            Input data.
        lengths : IntArray, optional
            Sequence lengths.

        Returns
        -------
        tuple[Array, IntArray]
            (X, lengths) tuple.
        """
        if isinstance(X, SequentialArray):
            return X.X, X.lengths
        return X, lengths
