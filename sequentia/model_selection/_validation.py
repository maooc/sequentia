# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Cross-validation utilities for sequence data.

This module provides utilities that work with scikit-learn's public API
for cross-validation, without relying on any private sklearn modules.
"""

from sequentia._internal._data import SequentialArray

__all__ = ["SequentialArray"]
