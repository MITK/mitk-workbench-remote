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

"""Tests for _spatial.py."""

import numpy as np
import pytest

from mitk_workbench_remote._spatial import (
    _normalize_direction,
    _normalize_origin,
    _normalize_spacing,
)

# ---------------------------------------------------------------------------
# _normalize_spacing
# ---------------------------------------------------------------------------


def test_spacing_none_returns_ones() -> None:
    assert _normalize_spacing(None, ndim=3) == (1.0, 1.0, 1.0)


def test_spacing_none_2d() -> None:
    assert _normalize_spacing(None, ndim=2) == (1.0, 1.0)


def test_spacing_from_list() -> None:
    assert _normalize_spacing([0.5, 1.0, 2.0], ndim=3) == (0.5, 1.0, 2.0)


def test_spacing_from_tuple() -> None:
    assert _normalize_spacing((2.0, 3.0), ndim=2) == (2.0, 3.0)


def test_spacing_from_ndarray() -> None:
    result = _normalize_spacing(np.array([1.5, 2.5, 3.5]), ndim=3)
    assert result == (1.5, 2.5, 3.5)


def test_spacing_wrong_length_raises_ValueError() -> None:
    with pytest.raises(ValueError, match="length 2, expected 3"):
        _normalize_spacing([1.0, 2.0], ndim=3)


def test_spacing_coerces_int_to_float() -> None:
    result = _normalize_spacing([1, 2, 3], ndim=3)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# _normalize_origin
# ---------------------------------------------------------------------------


def test_origin_none_returns_zeros() -> None:
    assert _normalize_origin(None, ndim=3) == (0.0, 0.0, 0.0)


def test_origin_from_list() -> None:
    assert _normalize_origin([10.0, 20.0, 30.0], ndim=3) == (10.0, 20.0, 30.0)


def test_origin_wrong_length_raises_ValueError() -> None:
    with pytest.raises(ValueError, match="length 4, expected 3"):
        _normalize_origin([1.0, 2.0, 3.0, 4.0], ndim=3)


def test_origin_coerces_int_to_float() -> None:
    result = _normalize_origin([0, 0, 0], ndim=3)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# _normalize_direction
# ---------------------------------------------------------------------------


def test_direction_none_returns_identity() -> None:
    result = _normalize_direction(None, ndim=3)
    np.testing.assert_array_equal(result, np.eye(3))
    assert result.dtype == np.float64


def test_direction_none_2d() -> None:
    result = _normalize_direction(None, ndim=2)
    np.testing.assert_array_equal(result, np.eye(2))


def test_direction_from_2d_array() -> None:
    mat = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    result = _normalize_direction(mat, ndim=3)
    expected = np.array(mat, dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_direction_from_flat_list() -> None:
    flat = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    result = _normalize_direction(flat, ndim=3)
    np.testing.assert_array_equal(result, np.eye(3))


def test_direction_from_ndarray() -> None:
    arr = np.eye(3, dtype=np.float32)
    result = _normalize_direction(arr, ndim=3)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.eye(3))


def test_direction_wrong_flat_length_raises_ValueError() -> None:
    with pytest.raises(ValueError, match="flat direction has length 8"):
        _normalize_direction([1, 0, 0, 0, 1, 0, 0, 0], ndim=3)


def test_direction_wrong_shape_raises_ValueError() -> None:
    with pytest.raises(ValueError, match="shape"):
        _normalize_direction(np.eye(2), ndim=3)
