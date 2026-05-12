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

"""Tests for converters/."""

import numpy as np

from mitk_workbench_remote.converters import find_converter_for_type, find_image_converter
from mitk_workbench_remote.converters._numpy import NumpyConverter
from mitk_workbench_remote.image import Image

# ---------------------------------------------------------------------------
# find_image_converter
# ---------------------------------------------------------------------------


def test_find_converter_ndarray() -> None:
    arr = np.zeros((3, 4, 5))
    converter = find_image_converter(arr)
    assert converter is not None
    assert isinstance(converter, NumpyConverter)


def test_find_converter_unknown_type_returns_none() -> None:
    assert find_image_converter(42) is None
    assert find_image_converter(object()) is None
    assert find_image_converter("some_string") is None


# ---------------------------------------------------------------------------
# find_converter_for_type
# ---------------------------------------------------------------------------


def test_find_converter_for_type_ndarray() -> None:
    converter = find_converter_for_type(np.ndarray)
    assert converter is not None
    assert isinstance(converter, NumpyConverter)


# ---------------------------------------------------------------------------
# NumpyConverter
# ---------------------------------------------------------------------------


def test_numpy_converter_can_handle() -> None:
    c = NumpyConverter()
    assert c.can_handle(np.zeros(3))
    assert not c.can_handle([1, 2, 3])


def test_numpy_converter_extract_geometry_empty() -> None:
    c = NumpyConverter()
    assert c.extract_geometry(np.zeros(3)) == {}


def test_numpy_converter_extract_properties_empty() -> None:
    c = NumpyConverter()
    assert c.extract_properties(np.zeros(3)) == {}


def test_numpy_converter_to_ndarray_returns_same_object() -> None:
    c = NumpyConverter()
    arr = np.zeros(3)
    assert c.to_ndarray(arr) is arr


def test_numpy_converter_roundtrip_via_nrrd() -> None:
    c = NumpyConverter()
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    nrrd_bytes = c.to_nrrd_bytes(arr)
    assert isinstance(nrrd_bytes, bytes)
    assert len(nrrd_bytes) > 0


def test_numpy_converter_from_image() -> None:
    c = NumpyConverter()
    arr = np.arange(6).reshape(2, 3)
    img = Image(arr)
    result = c.from_image(img)
    assert result is arr
