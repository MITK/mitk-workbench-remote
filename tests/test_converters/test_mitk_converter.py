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

"""Tests for converters/_mitk.py -- requires the ``mitk`` package."""

import numpy as np
import pytest

mitk = pytest.importorskip("mitk")

from mitk_workbench_remote.converters import (  # noqa: E402  -- requires `mitk`
    find_converter_for_type,
    find_image_converter,
)
from mitk_workbench_remote.converters._mitk import (  # noqa: E402  -- requires `mitk`
    MitkImageConverter,
)
from mitk_workbench_remote.image import Image  # noqa: E402  -- requires `mitk`

_OBLIQUE = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _make_3d_mitk_image() -> "mitk.Image":
    arr = np.zeros((4, 5, 6), dtype=np.uint8)
    return mitk.Image.from_numpy(
        arr,
        spacing=(1.1, 1.2, 1.3),
        origin=(10.0, 20.0, 30.0),
        direction=_OBLIQUE,
    )


# ---------------------------------------------------------------------------
# target_type / can_handle
# ---------------------------------------------------------------------------


def test_target_type_is_mitk_image() -> None:
    converter = MitkImageConverter()
    assert converter.target_type is mitk.Image


def test_can_handle_mitk_image() -> None:
    converter = MitkImageConverter()
    img = _make_3d_mitk_image()
    assert converter.can_handle(img) is True


def test_can_handle_ndarray_false() -> None:
    converter = MitkImageConverter()
    assert converter.can_handle(np.zeros((3, 4, 5))) is False


def test_can_handle_mw_image_false() -> None:
    converter = MitkImageConverter()
    mw_img = Image(np.zeros((3, 4, 5)))
    assert converter.can_handle(mw_img) is False


# ---------------------------------------------------------------------------
# extract_geometry
# ---------------------------------------------------------------------------


def test_extract_geometry_3d() -> None:
    converter = MitkImageConverter()
    img = _make_3d_mitk_image()
    geo = converter.extract_geometry(img)

    np.testing.assert_allclose(geo["spacing"], (1.1, 1.2, 1.3), rtol=1e-6)
    np.testing.assert_allclose(geo["origin"], (10.0, 20.0, 30.0), rtol=1e-6)
    np.testing.assert_allclose(geo["direction"], _OBLIQUE, atol=1e-10)


def test_extract_geometry_2d_caps_dimensions() -> None:
    converter = MitkImageConverter()
    arr = np.zeros((4, 5), dtype=np.uint8)
    img = mitk.Image.from_numpy(arr, spacing=(0.5, 0.5, 1.0), origin=(1.0, 2.0, 0.0))
    geo = converter.extract_geometry(img)

    ndim = img.ndim
    assert len(geo["spacing"]) == ndim
    assert len(geo["origin"]) == ndim
    assert geo["direction"].shape == (ndim, ndim)


# ---------------------------------------------------------------------------
# extract_properties
# ---------------------------------------------------------------------------


def test_extract_properties_returns_scalar_primitives() -> None:
    converter = MitkImageConverter()
    img = _make_3d_mitk_image()
    img.set_property("name", "abc")
    img.set_property("opacity", 0.5)
    img.set_property("visible", True)

    properties = converter.extract_properties(img)

    assert properties.get("name") == "abc"
    assert isinstance(properties["name"], str)
    assert abs(properties.get("opacity", -1) - 0.5) < 1e-5
    assert properties.get("visible") is True


def test_extract_properties_color_property_round_trips() -> None:
    converter = MitkImageConverter()
    img = _make_3d_mitk_image()
    img.set_property("color", mitk.ColorProperty.from_rgb(1.0, 0.5, 0.0))

    properties = converter.extract_properties(img)
    val = properties.get("color")

    # get_property() coerces ColorProperty to a 3-tuple via propertyToPythonValue
    assert val is not None
    assert isinstance(val, tuple)
    assert len(val) == 3
    np.testing.assert_allclose(val, (1.0, 0.5, 0.0), atol=1e-5)


# ---------------------------------------------------------------------------
# to_ndarray
# ---------------------------------------------------------------------------


def test_to_ndarray_is_view_not_copy() -> None:
    converter = MitkImageConverter()
    img = _make_3d_mitk_image()
    arr = converter.to_ndarray(img)

    assert arr.base is not None
    assert arr.flags.writeable is False


# ---------------------------------------------------------------------------
# to_nrrd_bytes
# ---------------------------------------------------------------------------


def test_to_nrrd_bytes_writes_valid_nrrd() -> None:
    from mitk_workbench_remote._io import read_nrrd

    converter = MitkImageConverter()
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    img = mitk.Image.from_numpy(arr, spacing=(1.0, 2.0, 3.0), origin=(4.0, 5.0, 6.0))

    nrrd_bytes = converter.to_nrrd_bytes(img)
    mw_img = read_nrrd(nrrd_bytes)

    np.testing.assert_allclose(mw_img.spacing, (1.0, 2.0, 3.0), rtol=1e-5)
    np.testing.assert_allclose(mw_img.origin, (4.0, 5.0, 6.0), rtol=1e-5)
    assert np.array_equal(mw_img.array, arr)


def test_to_nrrd_bytes_cleans_up_tempfile(tmp_path: pytest.TempPathFactory) -> None:
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    converter = MitkImageConverter()
    img = _make_3d_mitk_image()

    captured_path: list[Path] = []
    original_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(*args: object, **kwargs: object) -> object:
        ctx = original_ntf(*args, **kwargs)
        captured_path.append(Path(ctx.name))
        return ctx

    with patch.object(tempfile, "NamedTemporaryFile", side_effect=tracking_ntf):
        converter.to_nrrd_bytes(img)

    assert captured_path, "NamedTemporaryFile was never called"
    assert not captured_path[0].exists(), "Temp file was not cleaned up"


# ---------------------------------------------------------------------------
# from_image
# ---------------------------------------------------------------------------


def test_from_image_preserves_geometry_and_pixels() -> None:
    converter = MitkImageConverter()
    arr = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    mw_img = Image(arr, spacing=(1.0, 2.0, 3.0), origin=(4.0, 5.0, 6.0), direction=_OBLIQUE)

    mitk_img = converter.from_image(mw_img)

    np.testing.assert_allclose(mitk_img.get_spacing(time_step=0), (1.0, 2.0, 3.0), rtol=1e-5)
    np.testing.assert_allclose(mitk_img.get_origin(time_step=0), (4.0, 5.0, 6.0), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(mitk_img.get_direction(time_step=0)), _OBLIQUE, atol=1e-10
    )
    assert np.array_equal(np.asarray(mitk_img), arr)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_mw_to_mitk_to_mw() -> None:
    converter = MitkImageConverter()
    arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    original = Image(arr, spacing=(1.0, 2.0, 3.0), origin=(4.0, 5.0, 6.0), direction=_OBLIQUE)

    mitk_img = converter.from_image(original)
    restored = Image(mitk_img)

    assert original == restored


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------


def test_is_registered_after_import() -> None:
    assert find_converter_for_type(mitk.Image) is not None
    assert find_image_converter(mitk.Image()) is not None
