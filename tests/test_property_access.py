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


import numpy as np
import pytest

from mitk_workbench_remote.image import Image
from mitk_workbench_remote.multilabel import MultiLabelSegmentation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_image(props: dict | None = None) -> Image:
    return Image(np.zeros((3, 4, 5)), properties=props or {})


def _make_seg(props: dict | None = None) -> MultiLabelSegmentation:
    return MultiLabelSegmentation(groups=[], labels={}, properties=props or {})


# ===========================================================================
# Image
# ===========================================================================


def test_image_get_property_returns_value() -> None:
    img = _make_image({"k": "v"})
    assert img.get_property("k") == "v"


def test_image_get_property_missing_returns_none() -> None:
    img = _make_image()
    assert img.get_property("nonexistent") is None


def test_image_set_property_adds_new_key() -> None:
    img = _make_image()
    img.set_property("k", "v")
    assert img.get_property("k") == "v"


def test_image_set_property_overwrites_existing() -> None:
    img = _make_image({"k": "first"})
    img.set_property("k", "second")
    assert img.get_property("k") == "second"


def test_image_remove_property_deletes_key() -> None:
    img = _make_image({"k": "v"})
    img.remove_property("k")
    assert img.get_property("k") is None


def test_image_remove_property_missing_raises_keyerror() -> None:
    img = _make_image()
    with pytest.raises(KeyError):
        img.remove_property("missing")


def test_image_property_keys_returns_list() -> None:
    img = _make_image({"a": 1, "b": 2})
    assert sorted(img.property_keys) == ["a", "b"]


def test_image_property_keys_empty_by_default() -> None:
    img = _make_image()
    assert img.property_keys == []


def test_image_properties_returns_snapshot() -> None:
    img = _make_image({"k": "v"})
    snapshot = img.properties
    assert snapshot == {"k": "v"}
    snapshot["k"] = "mutated"
    assert img._properties["k"] == "v"


def test_image_properties_supports_dict_iteration() -> None:
    img = _make_image({"a": 1, "b": 2})
    keys = [k for k in img.properties]
    assert sorted(keys) == ["a", "b"]
    assert len(img.properties) == 2
    assert "a" in img.properties


# ===========================================================================
# MultiLabelSegmentation
# ===========================================================================


def test_multilabel_get_property_returns_value() -> None:
    seg = _make_seg({"k": "v"})
    assert seg.get_property("k") == "v"


def test_multilabel_get_property_missing_returns_none() -> None:
    seg = _make_seg()
    assert seg.get_property("nonexistent") is None


def test_multilabel_set_property_adds_new_key() -> None:
    seg = _make_seg()
    seg.set_property("k", "v")
    assert seg.get_property("k") == "v"


def test_multilabel_set_property_overwrites_existing() -> None:
    seg = _make_seg({"k": "first"})
    seg.set_property("k", "second")
    assert seg.get_property("k") == "second"


def test_multilabel_remove_property_deletes_key() -> None:
    seg = _make_seg({"k": "v"})
    seg.remove_property("k")
    assert seg.get_property("k") is None


def test_multilabel_remove_property_missing_raises_keyerror() -> None:
    seg = _make_seg()
    with pytest.raises(KeyError):
        seg.remove_property("missing")


def test_multilabel_property_keys_returns_list() -> None:
    seg = _make_seg({"a": 1, "b": 2})
    assert sorted(seg.property_keys) == ["a", "b"]


def test_multilabel_property_keys_empty_by_default() -> None:
    seg = _make_seg()
    assert seg.property_keys == []


def test_multilabel_properties_returns_snapshot() -> None:
    seg = _make_seg({"k": "v"})
    snapshot = seg.properties
    assert snapshot == {"k": "v"}
    snapshot["k"] = "mutated"
    assert seg._properties["k"] == "v"


def test_multilabel_properties_supports_dict_iteration() -> None:
    seg = _make_seg({"a": 1, "b": 2})
    keys = [k for k in seg.properties]
    assert sorted(keys) == ["a", "b"]
    assert len(seg.properties) == 2
    assert "a" in seg.properties
