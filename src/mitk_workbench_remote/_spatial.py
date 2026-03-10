# SPDX-FileCopyrightText: 2026, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or find it in LICENSE.txt.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared geometry helpers -- used by both Image and MultiLabelSegmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _normalize_spacing(
    spacing: Sequence[float] | np.ndarray | None,
    *,
    ndim: int,
) -> tuple[float, ...]:
    """Normalize spacing to a tuple of floats.

    Args:
        spacing: Input spacing. ``None`` defaults to ``(1.0, ...)``.
        ndim: Expected number of dimensions.

    Returns:
        Tuple of floats with length ``ndim``.

    Raises:
        ValueError: If the length does not match ``ndim``.
    """
    if spacing is None:
        return tuple(1.0 for _ in range(ndim))
    result = tuple(float(s) for s in spacing)
    if len(result) != ndim:
        raise ValueError(f"spacing has length {len(result)}, expected {ndim}")
    return result


def _normalize_origin(
    origin: Sequence[float] | np.ndarray | None,
    *,
    ndim: int,
) -> tuple[float, ...]:
    """Normalize origin to a tuple of floats.

    Args:
        origin: Input origin. ``None`` defaults to ``(0.0, ...)``.
        ndim: Expected number of dimensions.

    Returns:
        Tuple of floats with length ``ndim``.

    Raises:
        ValueError: If the length does not match ``ndim``.
    """
    if origin is None:
        return tuple(0.0 for _ in range(ndim))
    result = tuple(float(o) for o in origin)
    if len(result) != ndim:
        raise ValueError(f"origin has length {len(result)}, expected {ndim}")
    return result


def _normalize_direction(
    direction: Sequence[Any] | np.ndarray | None,
    *,
    ndim: int,
) -> np.ndarray:
    """Normalize direction to an ndim x ndim float64 matrix.

    Args:
        direction: Input direction matrix. ``None`` defaults to the identity matrix.
            Accepts flat sequences (length ndim*ndim), nested sequences, or ndarrays.
        ndim: Expected number of dimensions.

    Returns:
        Float64 ndarray with shape ``(ndim, ndim)``.

    Raises:
        ValueError: If the shape cannot be reshaped to ``(ndim, ndim)``.
    """
    if direction is None:
        return np.eye(ndim, dtype=np.float64)
    arr = np.asarray(direction, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != ndim * ndim:
            raise ValueError(f"flat direction has length {arr.shape[0]}, expected {ndim * ndim}")
        arr = arr.reshape(ndim, ndim)
    if arr.shape != (ndim, ndim):
        raise ValueError(f"direction has shape {arr.shape}, expected ({ndim}, {ndim})")
    return arr
