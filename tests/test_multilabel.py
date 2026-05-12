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

"""Tests for multilabel.py."""

import numpy as np
import pytest

from mitk_workbench_remote.image import Image
from mitk_workbench_remote.multilabel import LABEL_DTYPE, Label, LabelGroup, MultiLabelSegmentation

# ===========================================================================
# Label
# ===========================================================================


def test_label_construction_with_value() -> None:
    label = Label(1, "Liver", color=(0.8, 0.2, 0.1))
    assert label.value == 1
    assert label.name == "Liver"
    assert label.color == (0.8, 0.2, 0.1)
    assert label.opacity == 1.0
    assert label.visible is True
    assert label.locked is False
    assert label.tracking_id is None
    assert label.tracking_uid is None
    assert label.description is None


def test_label_construction_defaults() -> None:
    label = Label()
    assert label.value is None
    assert label.name == ""
    assert label.color == (1.0, 1.0, 1.0)
    assert label.opacity == 1.0
    assert label.visible is True
    assert label.locked is False


def test_label_value_zero_raises() -> None:
    with pytest.raises(ValueError, match="UNLABELED_VALUE"):
        Label(0)


def test_label_assign_value_sets_value() -> None:
    label = Label(None, "Spleen")
    label._assign_value(5)
    assert label.value == 5


def test_label_assign_value_twice_raises() -> None:
    label = Label(3, "Liver")
    with pytest.raises(RuntimeError, match="already assigned"):
        label._assign_value(4)


def test_label_assign_value_when_none_raises_if_called_twice() -> None:
    label = Label(None)
    label._assign_value(2)
    with pytest.raises(RuntimeError):
        label._assign_value(3)


def test_label_name_setter() -> None:
    label = Label(1, "Old")
    label.name = "New"
    assert label.name == "New"


def test_label_color_setter_valid() -> None:
    label = Label(1)
    label.color = (0.5, 0.5, 0.5)
    assert label.color == (0.5, 0.5, 0.5)


def test_label_color_setter_wrong_length() -> None:
    label = Label(1)
    with pytest.raises(ValueError, match="3 components"):
        label.color = (0.5, 0.5)  # type: ignore[assignment]


def test_label_color_setter_out_of_range() -> None:
    label = Label(1)
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        label.color = (1.5, 0.5, 0.5)


def test_label_opacity_setter() -> None:
    label = Label(1)
    label.opacity = 0.7
    assert label.opacity == pytest.approx(0.7)


def test_label_visible_setter() -> None:
    label = Label(1)
    label.visible = False
    assert label.visible is False


def test_label_locked_setter() -> None:
    label = Label(1)
    label.locked = True
    assert label.locked is True


def test_label_optional_fields() -> None:
    label = Label(
        1,
        "X",
        tracking_id="tid",
        tracking_uid="tuid",
        description="desc",
    )
    assert label.tracking_id == "tid"
    assert label.tracking_uid == "tuid"
    assert label.description == "desc"


def test_label_repr() -> None:
    label = Label(1, "Liver", color=(0.8, 0.2, 0.1))
    r = repr(label)
    assert "1" in r
    assert "Liver" in r


# ===========================================================================
# LabelGroup
# ===========================================================================


def test_labelgroup_construction_with_name() -> None:
    g = LabelGroup(name="Organs")
    assert g.name == "Organs"
    assert g.label_ids == []


def test_labelgroup_construction_no_name() -> None:
    g = LabelGroup()
    assert g.name is None
    assert g.label_ids == []


def test_labelgroup_label_ids_is_copy() -> None:
    g = LabelGroup()
    g._label_ids.append(1)
    ids = g.label_ids
    ids.append(99)  # mutate the copy
    assert g._label_ids == [1]  # internal state unchanged


def test_labelgroup_name_setter() -> None:
    g = LabelGroup("A")
    g.name = "B"
    assert g.name == "B"


def test_labelgroup_repr() -> None:
    g = LabelGroup("Organs")
    g._label_ids.append(1)
    r = repr(g)
    assert "Organs" in r
    assert "1" in r


# ===========================================================================
# MultiLabelSegmentation construction
# ===========================================================================


def test_create_empty() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    assert len(seg.groups) == 0
    assert len(seg.labels) == 0


def test_create_with_spacing() -> None:
    seg = MultiLabelSegmentation.create(shape=(10, 20, 30), spacing=(1.0, 2.0, 3.0))
    assert seg.spacing == (1.0, 2.0, 3.0)


def test_create_neither_reference_nor_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"reference.*shape|shape.*reference"):
        MultiLabelSegmentation.create()


def test_create_from_reference() -> None:
    ref = Image(
        np.zeros((5, 6, 7), dtype=np.float32),
        spacing=(1.0, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
    )
    seg = MultiLabelSegmentation.create(reference=ref)
    assert seg._shape == (5, 6, 7)
    assert seg.spacing == (1.0, 2.0, 3.0)
    assert seg.origin == (10.0, 20.0, 30.0)


def test_constructor_duplicate_label_id_raises() -> None:
    g1 = LabelGroup("A")
    g1._label_ids.append(1)
    g2 = LabelGroup("B")
    g2._label_ids.append(1)  # duplicate
    label = Label(1, "X")
    with pytest.raises(ValueError, match="Duplicate"):
        MultiLabelSegmentation(groups=[g1, g2], labels={1: label})


def test_constructor_missing_label_in_dict_raises() -> None:
    g = LabelGroup("A")
    g._label_ids.append(5)
    with pytest.raises(ValueError, match="not in the labels dict"):
        MultiLabelSegmentation(groups=[g], labels={})


def test_constructor_group_image_count_mismatch_raises() -> None:
    g = LabelGroup("A")
    with pytest.raises(ValueError, match="len\\(group_images\\)"):
        MultiLabelSegmentation(groups=[g], labels={}, group_images=[])


# ===========================================================================
# add_group / add_label
# ===========================================================================


def test_add_group_returns_index() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    idx0 = seg.add_group("Organs")
    idx1 = seg.add_group("Lesions")
    assert idx0 == 0
    assert idx1 == 1
    assert len(seg.groups) == 2
    assert seg.get_group(0).name == "Organs"


def test_add_label_auto_assign_starting_from_1() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    lbl = seg.add_label(Label(None, "Liver"), group=g)
    assert lbl.value == 1


def test_add_label_auto_assign_skips_existing() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(None, "A"), group=g)  # value 1
    seg.add_label(Label(None, "B"), group=g)  # value 2
    lbl = seg.add_label(Label(None, "C"), group=g)
    assert lbl.value == 3


def test_add_label_explicit_value() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    lbl = seg.add_label(Label(7, "Spleen"), group=g)
    assert lbl.value == 7
    # next auto value is max+1 (gaps are not reused, matching MITK strategy)
    lbl2 = seg.add_label(Label(None, "Liver"), group=g)
    assert lbl2.value == 8


def test_add_label_duplicate_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(3, "X"), group=g)
    with pytest.raises(ValueError, match="already exists"):
        seg.add_label(Label(3, "Y"), group=g)


def test_add_label_group_oob_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    with pytest.raises(IndexError):
        seg.add_label(Label(1, "X"), group=0)


# ===========================================================================
# remove_label
# ===========================================================================


def test_remove_label_clear_pixels() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(1, "Liver"), group=g)
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[2, 2, 2] = 1
    seg.set_group_image(g, arr)

    seg.remove_label(1, clear_pixels=True)
    assert seg.get_label(1) is None
    assert seg._group_images[g].array[2, 2, 2] == 0


def test_remove_label_no_clear_pixels() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(1, "Liver"), group=g)
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[2, 2, 2] = 1
    seg.set_group_image(g, arr)

    seg.remove_label(1, clear_pixels=False)
    assert seg.get_label(1) is None
    # pixel not cleared
    assert seg._group_images[g].array[2, 2, 2] == 1


def test_remove_label_unknown_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    with pytest.raises(ValueError, match="not found"):
        seg.remove_label(99)


# ===========================================================================
# Lookup
# ===========================================================================


def test_get_label_found() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(2, "Spleen"), group=g)
    lbl = seg.get_label(2)
    assert lbl is not None
    assert lbl.name == "Spleen"


def test_get_label_not_found() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    assert seg.get_label(99) is None


def test_get_group_oob_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    with pytest.raises(IndexError):
        seg.get_group(0)


def test_get_group_by_name_found() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    seg.add_group("Organs")
    g = seg.get_group_by_name("Organs")
    assert g is not None
    assert g.name == "Organs"


def test_get_group_by_name_not_found() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    assert seg.get_group_by_name("Missing") is None


def test_get_group_labels() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(None, "A"), group=g)
    seg.add_label(Label(None, "B"), group=g)
    labels = seg.get_group_labels(g)
    assert len(labels) == 2
    names = {lbl.name for lbl in labels}
    assert names == {"A", "B"}


def test_labels_property_raises_on_label_with_none_value() -> None:
    # ``add_label`` always assigns a value before storing, so a label with
    # ``value=None`` in ``_labels`` is internal-state corruption. The sort
    # accessor must surface that loudly rather than aliasing it onto the
    # reserved UNLABELED_VALUE (0).
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(1, "A"), group=g)
    # Force the corrupt state by mutating an entry's underlying value back
    # to None. Public API never lets this happen, but the sort accessor must
    # still fail loudly if it does.
    next(iter(seg._labels.values()))._value = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="value=None"):
        _ = seg.labels


# ===========================================================================
# get_group_image
# ===========================================================================


def test_get_group_image_geometry() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 6, 7), spacing=(1.0, 2.0, 3.0))
    g = seg.add_group("G")
    img = seg.get_group_image(g)
    assert img.shape == (5, 6, 7)
    assert img.spacing == (1.0, 2.0, 3.0)


def test_get_group_image_lazy_allocation() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    assert seg._group_images[g] is None
    img = seg.get_group_image(g)
    assert img is not None
    assert seg._group_images[g] is not None
    assert img.shape == (5, 5, 5)


def test_get_group_image_view_semantics() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    img1 = seg.get_group_image(g)
    img1.array[2, 2, 2] = 99
    img2 = seg.get_group_image(g)
    assert img2.array[2, 2, 2] == 99


def test_get_group_image_no_shape_raises() -> None:
    # Construct without shape and no images
    seg = MultiLabelSegmentation(groups=[], labels={})
    seg._groups.append(LabelGroup("G"))
    seg._group_images.append(None)
    with pytest.raises(ValueError, match="shape is unknown"):
        seg.get_group_image(0)


# ===========================================================================
# set_group_image
# ===========================================================================


def test_set_group_image_valid() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.ones((4, 4, 4), dtype=np.uint8)
    seg.set_group_image(g, arr)
    np.testing.assert_array_equal(seg.get_group_image(g).array, arr)


def test_set_group_image_shape_mismatch_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        seg.set_group_image(g, arr)


def test_set_group_image_non_integer_dtype_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="integer"):
        seg.set_group_image(g, arr)


def test_set_group_image_ndim_too_low_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="ndim"):
        seg.set_group_image(g, arr)


def test_set_group_image_accepts_image_object() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    img = Image(np.ones((4, 4, 4), dtype=np.uint8))
    seg.set_group_image(g, img)
    assert seg._group_images[g] is not None


# ===========================================================================
# import_group_image
# ===========================================================================


def _make_seg_with_two_labels() -> tuple[MultiLabelSegmentation, int]:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(1, "Liver"), group=g)
    seg.add_label(Label(2, "Spleen"), group=g)
    return seg, g


def test_import_group_image_int_remap() -> None:
    seg, g = _make_seg_with_two_labels()
    src = np.zeros((5, 5, 5), dtype=np.uint8)
    src[0, 0, 0] = 10  # 10 -> label 1
    src[1, 1, 1] = 20  # 20 -> label 2
    seg.import_group_image(g, src, value_map={10: 1, 20: 2})
    arr = seg.get_group_image(g).array
    assert arr[0, 0, 0] == 1
    assert arr[1, 1, 1] == 2


def test_import_group_image_str_remap() -> None:
    seg, g = _make_seg_with_two_labels()
    src = np.zeros((5, 5, 5), dtype=np.uint8)
    src[0, 0, 0] = 5  # 5 -> "Liver" (value 1)
    seg.import_group_image(g, src, value_map={5: "Liver"})
    arr = seg.get_group_image(g).array
    assert arr[0, 0, 0] == 1


def test_import_group_image_unknown_str_target_raises() -> None:
    seg, g = _make_seg_with_two_labels()
    src = np.zeros((5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="No label named"):
        seg.import_group_image(g, src, value_map={5: "Unknown"})


def test_import_group_image_wrong_group_int_target_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g0 = seg.add_group("G0")
    g1 = seg.add_group("G1")
    seg.add_label(Label(1, "Liver"), group=g0)
    seg.add_label(Label(2, "Spleen"), group=g1)
    src = np.zeros((5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not belong to group"):
        seg.import_group_image(g0, src, value_map={5: 2})  # label 2 is in g1, not g0


# ===========================================================================
# validate
# ===========================================================================


def test_validate_consistent() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(1, "Liver"), group=g)
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[2, 2, 2] = 1
    seg.set_group_image(g, arr)
    assert seg.validate() == []


def test_validate_undeclared_pixel_value() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    # No labels added, but pixel value 1 exists
    arr = np.zeros((5, 5, 5), dtype=np.uint8)
    arr[0, 0, 0] = 1
    seg.set_group_image(g, arr)
    warnings = seg.validate()
    assert any("1" in w and "not declared" in w for w in warnings)


def test_validate_declared_label_absent_from_pixels() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("G")
    seg.add_label(Label(1, "Liver"), group=g)
    arr = np.zeros((5, 5, 5), dtype=np.uint8)  # all zero — no label pixels
    seg.set_group_image(g, arr)
    warnings = seg.validate()
    assert any("1" in w and "not present" in w for w in warnings)


# ===========================================================================
# repr
# ===========================================================================


def test_repr() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    seg.add_group("G")
    seg.add_label(Label(1, "X"), group=0)
    r = repr(seg)
    assert "groups=1" in r
    assert "labels=1" in r


def test_repr_html_contains_table() -> None:
    seg = MultiLabelSegmentation.create(shape=(5, 5, 5))
    g = seg.add_group("Organs")
    seg.add_label(Label(1, "Liver", color=(0.8, 0.2, 0.1)), group=g)
    html = seg._repr_html_()
    assert "<table" in html
    assert "Liver" in html
    assert "Organs" in html


# ===========================================================================
# UNLABELED_VALUE class constant
# ===========================================================================


def test_unlabeled_value_constant() -> None:
    assert MultiLabelSegmentation.UNLABELED_VALUE == 0


# ===========================================================================
# LABEL_DTYPE enforcement
# ===========================================================================


def test_label_dtype_constant_is_uint16() -> None:
    assert np.dtype(np.uint16) == LABEL_DTYPE


def test_get_group_image_lazy_alloc_dtype_is_label_dtype() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    img = seg.get_group_image(g)
    assert img.array.dtype == LABEL_DTYPE


def test_set_group_image_casts_uint8_to_label_dtype() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.ones((4, 4, 4), dtype=np.uint8)
    seg.set_group_image(g, arr)
    assert seg.get_group_image(g).array.dtype == LABEL_DTYPE


def test_set_group_image_casts_int32_to_label_dtype() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.ones((4, 4, 4), dtype=np.int32)
    seg.set_group_image(g, arr)
    assert seg.get_group_image(g).array.dtype == LABEL_DTYPE


def test_set_group_image_uint16_input_stays_label_dtype() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.ones((4, 4, 4), dtype=np.uint16)
    seg.set_group_image(g, arr)
    assert seg.get_group_image(g).array.dtype == LABEL_DTYPE


def test_set_group_image_preserves_pixel_values_after_cast() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    seg.add_label(Label(1, "A"), group=g)
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[2, 2, 2] = 1
    seg.set_group_image(g, arr)
    stored = seg.get_group_image(g).array
    assert stored.dtype == LABEL_DTYPE
    assert stored[2, 2, 2] == 1
    assert stored[0, 0, 0] == 0


def test_set_group_image_float_still_raises() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g = seg.add_group("G")
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="integer"):
        seg.set_group_image(g, arr)


def test_compose_array_dtype_is_label_dtype() -> None:
    seg = MultiLabelSegmentation.create(shape=(4, 4, 4))
    g0 = seg.add_group("G0")
    seg.add_group("G1")
    # g0 has explicit pixel data (cast from uint8), g1 is lazy-zero
    arr = np.ones((4, 4, 4), dtype=np.uint8)
    seg.set_group_image(g0, arr)
    composed = seg._compose_array()
    assert composed.dtype == LABEL_DTYPE
    assert composed.shape == (2, 4, 4, 4)


def test_compose_array_mixed_sources_dtype_is_label_dtype() -> None:
    """All-None groups (zero-filled) also produce LABEL_DTYPE."""
    seg = MultiLabelSegmentation.create(shape=(3, 3, 3))
    seg.add_group("G0")
    seg.add_group("G1")
    composed = seg._compose_array()
    assert composed.dtype == LABEL_DTYPE


# ===========================================================================
# to_mitk / from_mitk (require mitk package)
# ===========================================================================


class TestMitkInterop:
    """Tests for MultiLabelSegmentation.to_mitk() and from_mitk()."""

    @pytest.fixture(autouse=True)
    def require_mitk(self) -> None:
        pytest.importorskip("mitk")

    def _make_seg(self) -> MultiLabelSegmentation:
        seg = MultiLabelSegmentation.create(
            shape=(4, 5, 6), spacing=(1.0, 2.0, 3.0), origin=(10.0, 20.0, 30.0)
        )
        g0 = seg.add_group("Anatomy")
        seg.add_label(Label(1, "Liver", color=(0.8, 0.4, 0.1)), group=g0)
        seg.add_label(Label(2, "Spleen", color=(0.3, 0.6, 0.8)), group=g0)
        g1 = seg.add_group("Findings")
        seg.add_label(Label(3, "Tumor", color=(1.0, 0.1, 0.1), locked=True), group=g1)
        img0 = seg.get_group_image(0)
        img0.array[0, 0, 0] = 1
        img0.array[1, 1, 1] = 2
        img1 = seg.get_group_image(1)
        img1.array[2, 2, 2] = 3
        return seg

    def test_to_mitk_returns_mitk_mls(self) -> None:
        import mitk

        result = self._make_seg().to_mitk()
        assert isinstance(result, mitk.MultiLabelSegmentation)

    def test_to_mitk_preserves_groups(self) -> None:
        result = self._make_seg().to_mitk()
        assert result.num_groups == 2

    def test_to_mitk_preserves_labels(self) -> None:
        result = self._make_seg().to_mitk()
        values = sorted(int(v) for v in result.label_values if int(v) != 0)
        assert values == [1, 2, 3]
        names = {int(lbl.value): lbl.name for lbl in result.labels}
        assert names[1] == "Liver"
        assert names[3] == "Tumor"

    def test_to_mitk_preserves_pixels(self) -> None:
        result = self._make_seg().to_mitk()
        arr0 = np.asarray(result.get_group_image(0), copy=False)
        assert int(arr0[0, 0, 0]) == 1
        assert int(arr0[1, 1, 1]) == 2
        arr1 = np.asarray(result.get_group_image(1), copy=False)
        assert int(arr1[2, 2, 2]) == 3

    def test_to_mitk_preserves_geometry(self) -> None:
        result = self._make_seg().to_mitk()
        np.testing.assert_allclose(result.spacing, (1.0, 2.0, 3.0), rtol=1e-5)
        np.testing.assert_allclose(result.origin, (10.0, 20.0, 30.0), rtol=1e-5)

    def test_to_mitk_raises_importerror_when_mitk_absent(self) -> None:
        import sys
        from unittest.mock import patch

        seg = self._make_seg()
        with patch.dict(sys.modules, {"mitk": None}), pytest.raises(ImportError, match="mitk"):
            seg.to_mitk()

    def test_from_mitk_returns_mw_mls(self) -> None:

        seg = self._make_seg()
        mitk_seg = seg.to_mitk()
        result = MultiLabelSegmentation.from_mitk(mitk_seg)
        assert isinstance(result, MultiLabelSegmentation)

    def test_from_mitk_preserves_structure(self) -> None:
        seg = self._make_seg()
        mitk_seg = seg.to_mitk()
        result = MultiLabelSegmentation.from_mitk(mitk_seg)

        assert len(result.groups) == 2
        values = sorted(lbl.value for lbl in result.labels)
        assert values == [1, 2, 3]
        arr0 = result.get_group_image(0).array
        assert int(arr0[0, 0, 0]) == 1
        assert int(arr0[1, 1, 1]) == 2
        np.testing.assert_allclose(result.spacing, (1.0, 2.0, 3.0), rtol=1e-5)

    def test_from_mitk_raises_importerror_when_mitk_absent(self) -> None:
        import sys
        from unittest.mock import patch

        seg = self._make_seg()
        mitk_seg = seg.to_mitk()
        with patch.dict(sys.modules, {"mitk": None}), pytest.raises(ImportError, match="mitk"):
            MultiLabelSegmentation.from_mitk(mitk_seg)
