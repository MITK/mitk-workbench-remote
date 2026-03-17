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

"""Tests for _io/nrrd.py."""

from pathlib import Path

import numpy as np

from mitk_workbench_remote._io.nrrd import read_nrrd, write_nrrd
from mitk_workbench_remote.image import Image

# ---------------------------------------------------------------------------
# read_nrrd / write_nrrd roundtrip (bytes)
# ---------------------------------------------------------------------------


def test_roundtrip_bytes_3d() -> None:
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    img = Image(arr, spacing=(0.5, 1.0, 2.0), origin=(10.0, 20.0, 30.0))
    nrrd_bytes = write_nrrd(img)
    img2 = read_nrrd(nrrd_bytes)
    np.testing.assert_array_equal(img.array, img2.array)
    assert np.allclose(img.spacing, img2.spacing)
    assert np.allclose(img.origin, img2.origin)
    np.testing.assert_allclose(img.direction, img2.direction)


def test_roundtrip_bytes_2d() -> None:
    arr = np.arange(12, dtype=np.int16).reshape(3, 4)
    img = Image(arr, spacing=(0.25, 0.5))
    nrrd_bytes = write_nrrd(img)
    img2 = read_nrrd(nrrd_bytes)
    np.testing.assert_array_equal(img.array, img2.array)
    assert np.allclose(img.spacing, img2.spacing)


def test_roundtrip_default_geometry() -> None:
    arr = np.zeros((3, 4, 5), dtype=np.uint8)
    img = Image(arr)
    nrrd_bytes = write_nrrd(img)
    img2 = read_nrrd(nrrd_bytes)
    assert img2.spacing == (1.0, 1.0, 1.0)
    assert img2.origin == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(img2.direction, np.eye(3))


# ---------------------------------------------------------------------------
# read_nrrd / write_nrrd roundtrip (file)
# ---------------------------------------------------------------------------


def test_roundtrip_file(tmp_path: Path) -> None:
    arr = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    img = Image(arr, spacing=(1.5, 2.5, 3.5), origin=(-1.0, -2.0, -3.0))
    file_path = tmp_path / "test.nrrd"
    write_nrrd(img, path=file_path)
    assert file_path.exists()
    img2 = read_nrrd(file_path)
    np.testing.assert_array_equal(img.array, img2.array)
    assert np.allclose(img.spacing, img2.spacing)


def test_read_from_string_path(tmp_path: Path) -> None:
    arr = np.zeros((2, 3), dtype=np.float32)
    img = Image(arr)
    file_path = tmp_path / "test.nrrd"
    write_nrrd(img, path=file_path)
    img2 = read_nrrd(str(file_path))
    np.testing.assert_array_equal(img.array, img2.array)


# ---------------------------------------------------------------------------
# Custom properties roundtrip
# ---------------------------------------------------------------------------


def test_custom_metadata_roundtrip() -> None:
    arr = np.zeros((3, 4, 5), dtype=np.uint8)
    meta = {"my_custom_field": "hello_world", "another_field": "42"}
    img = Image(arr, properties=meta)
    nrrd_bytes = write_nrrd(img)
    img2 = read_nrrd(nrrd_bytes)
    assert img2.metadata.get("my_custom_field") == "hello_world"
    assert img2.metadata.get("another_field") == "42"


# ---------------------------------------------------------------------------
# Non-identity direction
# ---------------------------------------------------------------------------


def test_roundtrip_with_custom_direction() -> None:
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    # 90-degree rotation around z-axis
    direction = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    img = Image(arr, spacing=(1.0, 2.0, 3.0), direction=direction)
    nrrd_bytes = write_nrrd(img)
    img2 = read_nrrd(nrrd_bytes)
    np.testing.assert_allclose(img2.direction, direction, atol=1e-10)
    assert np.allclose(img2.spacing, img.spacing)
