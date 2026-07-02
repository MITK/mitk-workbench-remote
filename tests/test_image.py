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

"""Tests for image.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mitk_workbench_remote.image import Image

# ---------------------------------------------------------------------------
# Construction from ndarray
# ---------------------------------------------------------------------------


def test_image_from_ndarray_stores_array() -> None:
    arr = np.zeros((3, 4, 5))
    img = Image(arr)
    assert img.array is arr


def test_image_default_spacing() -> None:
    img = Image(np.zeros((3, 4, 5)))
    assert img.spacing == (1.0, 1.0, 1.0)


def test_image_default_origin() -> None:
    img = Image(np.zeros((3, 4, 5)))
    assert img.origin == (0.0, 0.0, 0.0)


def test_image_default_direction() -> None:
    img = Image(np.zeros((3, 4, 5)))
    np.testing.assert_array_equal(img.direction, np.eye(3))


def test_image_custom_spacing() -> None:
    img = Image(np.zeros((3, 4, 5)), spacing=(0.5, 1.0, 2.0))
    assert img.spacing == (0.5, 1.0, 2.0)


def test_image_custom_origin() -> None:
    img = Image(np.zeros((3, 4, 5)), origin=(10.0, 20.0, 30.0))
    assert img.origin == (10.0, 20.0, 30.0)


def test_image_custom_direction() -> None:
    d = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    img = Image(np.zeros((3, 4, 5)), direction=d)
    np.testing.assert_array_equal(img.direction, d)


def test_image_default_properties_is_empty_dict() -> None:
    img = Image(np.zeros((3, 4, 5)))
    assert img.properties == {}


def test_image_custom_properties() -> None:
    meta = {"key1": "value1", "key2": 42}
    img = Image(np.zeros((3, 4, 5)), properties=meta)
    assert img.properties == meta


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_image_ndim() -> None:
    assert Image(np.zeros((3, 4, 5))).ndim == 3
    assert Image(np.zeros((10, 20))).ndim == 2


def test_image_shape() -> None:
    assert Image(np.zeros((3, 4, 5))).shape == (3, 4, 5)


def test_image_dtype() -> None:
    assert Image(np.zeros((3,), dtype=np.float32)).dtype == np.float32


# ---------------------------------------------------------------------------
# Lazy conversion via converter
# ---------------------------------------------------------------------------


def test_image_from_converter_lazy_array() -> None:
    """Array is not materialized until .array is accessed."""
    mock_converter = MagicMock()
    mock_converter.can_handle.return_value = True
    mock_converter.extract_geometry.return_value = {"spacing": (1.0, 1.0, 1.0)}
    mock_converter.extract_properties.return_value = {}
    mock_converter.to_ndarray.return_value = np.zeros((3, 4, 5))

    with patch(
        "mitk_workbench_remote.converters.find_image_converter", return_value=mock_converter
    ):
        img = Image("fake_data")
        mock_converter.to_ndarray.assert_not_called()
        _ = img.array
        mock_converter.to_ndarray.assert_called_once()


def test_image_from_converter_extracts_geometry() -> None:
    mock_converter = MagicMock()
    mock_converter.can_handle.return_value = True
    mock_converter.extract_geometry.return_value = {
        "spacing": (0.5, 1.0, 2.0),
        "origin": (10.0, 20.0, 30.0),
    }
    mock_converter.extract_properties.return_value = {"my_key": "my_value"}
    mock_converter.to_ndarray.return_value = np.zeros((3, 4, 5))

    with patch(
        "mitk_workbench_remote.converters.find_image_converter", return_value=mock_converter
    ):
        img = Image("fake_data")
        assert img.spacing == (0.5, 1.0, 2.0)
        assert img.origin == (10.0, 20.0, 30.0)
        assert img.properties == {"my_key": "my_value"}


def test_image_explicit_kwargs_override_converter_geometry() -> None:
    mock_converter = MagicMock()
    mock_converter.can_handle.return_value = True
    mock_converter.extract_geometry.return_value = {
        "spacing": (0.5, 1.0, 2.0),
        "origin": (10.0, 20.0, 30.0),
    }
    mock_converter.extract_properties.return_value = {"extracted": True}
    mock_converter.to_ndarray.return_value = np.zeros((3, 4, 5))

    with patch(
        "mitk_workbench_remote.converters.find_image_converter", return_value=mock_converter
    ):
        img = Image("fake_data", spacing=(2.0, 2.0, 2.0), properties={"custom": True})
        assert img.spacing == (2.0, 2.0, 2.0)
        assert img.origin == (10.0, 20.0, 30.0)  # not overridden
        assert img.properties == {"custom": True}  # overridden


def test_image_unsupported_type_raises_TypeError() -> None:
    with pytest.raises(TypeError, match="No converter found"):
        Image(object())


def test_image_unsupported_type_message_enumerates_accepted_inputs() -> None:
    with pytest.raises(TypeError) as excinfo:
        Image(object())
    msg = str(excinfo.value)
    assert "No converter found" in msg
    # The message must point the way forward by naming what IS accepted.
    for accepted in ("SimpleITK.Image", "mitk.Image", "mlarray.MLArray", "register_converter"):
        assert accepted in msg, f"expected {accepted!r} in TypeError message: {msg!r}"


def test_image_unsupported_mitk_type_does_not_steer_to_as_type_remote() -> None:
    # A native non-image mitk type (e.g. mitk.PointSet) hits this same branch.
    # The message must NOT advise get_data(as_type=REMOTE): such nodes have no
    # remote Image (they raise UnsupportedDataTypeError), so that advice would
    # misdirect. The constructor cannot infer get_data provenance from the type.
    class _FakeMitkPointSet:
        pass

    with pytest.raises(TypeError) as excinfo:
        Image(_FakeMitkPointSet())
    msg = str(excinfo.value)
    assert "No converter found" in msg
    assert "as_type" not in msg
    assert "REMOTE" not in msg


# ---------------------------------------------------------------------------
# Conversion methods
# ---------------------------------------------------------------------------


def test_to_numpy_returns_array() -> None:
    arr = np.arange(12).reshape(3, 4)
    img = Image(arr)
    assert img.to_numpy() is arr


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


def test_image_equality_same() -> None:
    arr = np.arange(6).reshape(2, 3)
    img1 = Image(arr, spacing=(1.0, 2.0))
    img2 = Image(arr.copy(), spacing=(1.0, 2.0))
    assert img1 == img2


def test_image_inequality_different_array() -> None:
    img1 = Image(np.zeros((2, 3)))
    img2 = Image(np.ones((2, 3)))
    assert img1 != img2


def test_image_inequality_different_spacing() -> None:
    arr = np.zeros((2, 3))
    img1 = Image(arr, spacing=(1.0, 1.0))
    img2 = Image(arr.copy(), spacing=(2.0, 2.0))
    assert img1 != img2


def test_image_inequality_different_origin() -> None:
    arr = np.zeros((2, 3))
    img1 = Image(arr, origin=(0.0, 0.0))
    img2 = Image(arr.copy(), origin=(1.0, 1.0))
    assert img1 != img2


def test_image_eq_not_implemented_for_other_types() -> None:
    img = Image(np.zeros((2, 3)))
    assert img.__eq__("not an image") is NotImplemented


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr_includes_shape() -> None:
    img = Image(np.zeros((3, 4, 5)))
    assert "(3, 4, 5)" in repr(img)


def test_repr_includes_spacing() -> None:
    img = Image(np.zeros((3, 4, 5)), spacing=(0.5, 1.0, 2.0))
    assert "(0.5, 1.0, 2.0)" in repr(img)


# ---------------------------------------------------------------------------
# _repr_html_
# ---------------------------------------------------------------------------


def test_repr_html_returns_string() -> None:
    img = Image(np.zeros((3, 4, 5)))
    assert isinstance(img._repr_html_(), str)


def test_repr_html_contains_shape() -> None:
    img = Image(np.zeros((3, 4, 5)))
    assert "(3, 4, 5)" in img._repr_html_()


def test_repr_html_contains_dtype() -> None:
    img = Image(np.zeros((3, 4, 5), dtype=np.float32))
    assert "float32" in img._repr_html_()


def test_repr_html_contains_spacing() -> None:
    img = Image(np.zeros((3, 4, 5)), spacing=(0.5, 1.0, 2.0))
    assert "(0.5, 1.0, 2.0)" in img._repr_html_()


def test_repr_html_contains_origin() -> None:
    img = Image(np.zeros((3, 4, 5)), origin=(10.0, 20.0, 30.0))
    assert "(10.0, 20.0, 30.0)" in img._repr_html_()


def test_repr_html_contains_direction() -> None:
    img = Image(np.zeros((3, 4, 5)))
    html = img._repr_html_()
    # Identity direction: rows separated by |
    assert "|" in html


def test_repr_html_no_properties_section_when_empty() -> None:
    img = Image(np.zeros((3, 4, 5)))
    html = img._repr_html_()
    assert "Fetched properties" not in html


def test_repr_html_shows_properties_section_when_present() -> None:
    img = Image(np.zeros((3, 4, 5)), properties={"my_key": "my_value"})
    html = img._repr_html_()
    assert "Fetched properties" in html
    assert "my_key" in html
    assert "my_value" in html


def test_repr_html_properties_section_includes_snapshot_note() -> None:
    img = Image(np.zeros((3, 4, 5)), properties={"k": "v"})
    html = img._repr_html_()
    assert "snapshot" in html.lower()


def test_repr_html_escapes_special_characters_in_property_values() -> None:
    img = Image(np.zeros((3, 4, 5)), properties={"<script>": "<b>xss</b>"})
    html = img._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;xss&lt;/b&gt;" in html


# ---------------------------------------------------------------------------
# to_mitk -- requires the ``mitk`` package (WP-7)
# ---------------------------------------------------------------------------

_MITK_OBLIQUE = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


class TestToMitk:
    """Tests for Image.to_mitk() -- all gated on the mitk package."""

    @pytest.fixture(autouse=True)
    def require_mitk(self) -> None:
        pytest.importorskip("mitk")

    def _make_image(self) -> Image:
        arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        return Image(
            arr,
            spacing=(1.0, 2.0, 3.0),
            origin=(4.0, 5.0, 6.0),
            direction=np.array(_MITK_OBLIQUE, dtype=np.float64),
        )

    def test_to_mitk_returns_mitk_image(self) -> None:
        import mitk

        result = self._make_image().to_mitk()
        assert isinstance(result, mitk.Image)

    def test_to_mitk_preserves_geometry(self) -> None:
        result = self._make_image().to_mitk()
        np.testing.assert_allclose(result.get_spacing(time_step=0), (1.0, 2.0, 3.0), rtol=1e-5)
        np.testing.assert_allclose(result.get_origin(time_step=0), (4.0, 5.0, 6.0), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(result.get_direction(time_step=0)),
            np.array(_MITK_OBLIQUE, dtype=np.float64),
            atol=1e-10,
        )

    def test_to_mitk_preserves_pixels(self) -> None:
        mw_img = self._make_image()
        result = mw_img.to_mitk()
        assert np.array_equal(np.asarray(result), mw_img.array)

    def test_to_mitk_raises_importerror_when_mitk_absent(self) -> None:
        import sys
        from unittest.mock import patch

        mw_img = self._make_image()
        with (
            patch.dict(sys.modules, {"mitk": None}),
            pytest.raises(ImportError, match="mitk"),
        ):
            mw_img.to_mitk()

    def test_to_mitk_does_not_transfer_properties(self) -> None:
        arr = np.zeros((3, 4, 5), dtype=np.uint8)
        mw_img = Image(arr, properties={"name": "X"})
        result = mw_img.to_mitk()
        assert "name" not in result.property_keys
