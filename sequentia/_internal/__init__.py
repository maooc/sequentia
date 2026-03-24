# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from sequentia._internal._data import get_idxs, iter_X
from sequentia._internal._routing import (
    SequenceMetadataMixin,
    enable_metadata_routing,
    routing_enabled,
)
from sequentia._internal._sequence import (
    SequenceData,
    SequenceDataView,
    concatenate_sequences,
    extract_sequences,
    get_sequence_indices,
    iter_sequences,
    split_sequences,
)
from sequentia._internal._typing import Array, FloatArray, IntArray
from sequentia._internal._validation import (
    check_X,
    check_X_lengths,
    check_classes,
    check_is_fitted,
    check_random_state,
    check_use_c,
    check_weighting,
    check_y,
    requires_fit,
)

__all__ = [
    "_data",
    "_hmm",
    "_multiprocessing",
    "_routing",
    "_sequence",
    "_sklearn",
    "_typing",
    "_validation",
    "Array",
    "FloatArray",
    "IntArray",
    "SequenceData",
    "SequenceDataView",
    "SequenceMetadataMixin",
    "check_X",
    "check_X_lengths",
    "check_classes",
    "check_is_fitted",
    "check_random_state",
    "check_use_c",
    "check_weighting",
    "check_y",
    "concatenate_sequences",
    "enable_metadata_routing",
    "extract_sequences",
    "get_idxs",
    "get_sequence_indices",
    "iter_X",
    "iter_sequences",
    "requires_fit",
    "routing_enabled",
    "split_sequences",
]
