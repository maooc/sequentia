# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Preprocessing utilities for sequence data.

This module provides transformers and feature extractors designed for
sequence data, with full sklearn Pipeline compatibility via metadata routing.
"""

from sequentia.preprocessing.transforms import (
    IndependentFunctionTransformer,
    MaxFeatureExtractor,
    MeanFeatureExtractor,
    MinFeatureExtractor,
    QuantileFeatureExtractor,
    RangeFeatureExtractor,
    SequenceFeatureExtractor,
    SequenceTransformer,
    StdFeatureExtractor,
    extract_max,
    extract_mean,
    extract_min,
    extract_quantiles,
    extract_range,
    extract_std,
    mean_filter,
    median_filter,
)

__all__ = [
    "IndependentFunctionTransformer",
    "MaxFeatureExtractor",
    "MeanFeatureExtractor",
    "MinFeatureExtractor",
    "QuantileFeatureExtractor",
    "RangeFeatureExtractor",
    "SequenceFeatureExtractor",
    "SequenceTransformer",
    "StdFeatureExtractor",
    "extract_max",
    "extract_mean",
    "extract_min",
    "extract_quantiles",
    "extract_range",
    "extract_std",
    "mean_filter",
    "median_filter",
]
