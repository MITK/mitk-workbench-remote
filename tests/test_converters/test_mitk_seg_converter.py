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

"""Tests for converters/_mitk_seg.py -- requires the ``mitk`` package."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

mitk = pytest.importorskip("mitk")

from mitk_workbench_remote.converters._mitk_seg import MitkSegmentationConverter
from mitk_workbench_remote.multilabel import Label, LabelGroup, MultiLabelSegmentation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mitk_seg() -> "mitk.MultiLabelSegmentation":
    """Build a mitk.MultiLabelSegmentation with two groups and three labels."""
    ref = mitk.Image.from_numpy(
        np.zeros((4, 5, 6), dtype=np.uint16),
        spacing=(1.0, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
    )
    # mitk.MultiLabelSegmentation(ref) always creates one empty group (the
    # C++ flag steering that is not exposed in the binding). Populate it as
    # "Anatomy" instead of calling add_group for the first group.
    seg = mitk.MultiLabelSegmentation(ref)

    # Group 0: Anatomy
    liver = mitk.Label(1, "Liver")
    liver.color = (0.8, 0.4, 0.1)
    spleen = mitk.Label(2, "Spleen")
    spleen.color = (0.3, 0.6, 0.8)
    seg.add_label(liver, group=0)
    seg.add_label(spleen, group=0)

    # Group 1: Findings
    tumor = mitk.Label(3, "Tumor")
    tumor.color = (1.0, 0.1, 0.1)
    tumor.locked = True
    seg.add_group(None, [tumor])

    # Paint simple pixel data so round-trips have non-trivial content.
    # Use as_numpy() which returns a writeable view; np.asarray / .array
    # return a read-only view by convention.
    arr0 = seg.get_group_image(0).as_numpy()
    arr0[0, 0, 0] = 1
    arr0[1, 1, 1] = 2

    arr1 = seg.get_group_image(1).as_numpy()
    arr1[2, 2, 2] = 3

    return seg


def _make_mw_seg() -> MultiLabelSegmentation:
    """Build an mw.MultiLabelSegmentation with two groups and three labels."""
    seg = MultiLabelSegmentation.create(
        shape=(4, 5, 6),
        spacing=(1.0, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
    )
    g0 = seg.add_group("Anatomy")
    seg.add_label(Label(1, "Liver", color=(0.8, 0.4, 0.1)), group=g0)
    seg.add_label(Label(2, "Spleen", color=(0.3, 0.6, 0.8)), group=g0)

    g1 = seg.add_group("Findings")
    seg.add_label(Label(3, "Tumor", color=(1.0, 0.1, 0.1), locked=True), group=g1)

    # Paint pixel data
    img0 = seg.get_group_image(0)
    img0.array[0, 0, 0] = 1
    img0.array[1, 1, 1] = 2

    img1 = seg.get_group_image(1)
    img1.array[2, 2, 2] = 3

    return seg


# ---------------------------------------------------------------------------
# from_segmentation
# ---------------------------------------------------------------------------


def test_from_segmentation_preserves_group_count() -> None:
    converter = MitkSegmentationConverter()
    mw_seg = _make_mw_seg()
    mitk_seg = converter.from_segmentation(mw_seg)
    assert mitk_seg.num_groups == 2


def test_from_segmentation_preserves_label_values() -> None:
    converter = MitkSegmentationConverter()
    mw_seg = _make_mw_seg()
    mitk_seg = converter.from_segmentation(mw_seg)
    values = sorted(int(v) for v in mitk_seg.label_values if int(v) != 0)
    assert values == [1, 2, 3]


def test_from_segmentation_preserves_label_names() -> None:
    converter = MitkSegmentationConverter()
    mw_seg = _make_mw_seg()
    mitk_seg = converter.from_segmentation(mw_seg)
    names = {int(lbl.value): lbl.name for lbl in mitk_seg.labels}
    assert names[1] == "Liver"
    assert names[2] == "Spleen"
    assert names[3] == "Tumor"


def test_from_segmentation_preserves_label_colors() -> None:
    converter = MitkSegmentationConverter()
    mw_seg = _make_mw_seg()
    mitk_seg = converter.from_segmentation(mw_seg)
    labels_by_value = {int(lbl.value): lbl for lbl in mitk_seg.labels}
    np.testing.assert_allclose(labels_by_value[1].color, (0.8, 0.4, 0.1), atol=1e-3)
    np.testing.assert_allclose(labels_by_value[3].color, (1.0, 0.1, 0.1), atol=1e-3)


def test_from_segmentation_preserves_geometry() -> None:
    converter = MitkSegmentationConverter()
    mw_seg = _make_mw_seg()
    mitk_seg = converter.from_segmentation(mw_seg)
    np.testing.assert_allclose(mitk_seg.spacing, (1.0, 2.0, 3.0), rtol=1e-5)
    np.testing.assert_allclose(mitk_seg.origin, (10.0, 20.0, 30.0), rtol=1e-5)


def test_from_segmentation_preserves_pixel_data() -> None:
    converter = MitkSegmentationConverter()
    mw_seg = _make_mw_seg()
    mitk_seg = converter.from_segmentation(mw_seg)

    arr0 = np.asarray(mitk_seg.get_group_image(0), copy=False)
    assert int(arr0[0, 0, 0]) == 1
    assert int(arr0[1, 1, 1]) == 2

    arr1 = np.asarray(mitk_seg.get_group_image(1), copy=False)
    assert int(arr1[2, 2, 2]) == 3


def test_from_segmentation_preserves_label_properties() -> None:
    converter = MitkSegmentationConverter()
    seg = MultiLabelSegmentation.create(shape=(3, 3, 3), spacing=(1.0, 1.0, 1.0))
    g = seg.add_group("G")
    seg.add_label(
        Label(
            1,
            "Organ",
            locked=True,
            visible=False,
            opacity=0.5,
            description="test desc",
            tracking_id="tid-1",
            tracking_uid="tuid-1",
        ),
        group=g,
    )
    mitk_seg = converter.from_segmentation(seg)
    lbl = mitk_seg.get_label(1)
    assert lbl.locked is True
    assert lbl.visible is False
    assert abs(lbl.opacity - 0.5) < 1e-3
    assert lbl.description == "test desc"
    assert lbl.tracking_id == "tid-1"
    assert lbl.tracking_uid == "tuid-1"


def test_from_segmentation_cleans_up_tempfile() -> None:
    converter = MitkSegmentationConverter()
    mw_seg = _make_mw_seg()

    captured: list[Path] = []
    original_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(*args: object, **kwargs: object) -> object:
        ctx = original_ntf(*args, **kwargs)
        captured.append(Path(ctx.name))
        return ctx

    with patch.object(tempfile, "NamedTemporaryFile", side_effect=tracking_ntf):
        converter.from_segmentation(mw_seg)

    assert captured, "NamedTemporaryFile was never called"
    assert not captured[0].exists(), "Temp file was not cleaned up"


# ---------------------------------------------------------------------------
# to_segmentation
# ---------------------------------------------------------------------------


def test_to_segmentation_preserves_group_count() -> None:
    converter = MitkSegmentationConverter()
    mitk_seg = _make_mitk_seg()
    mw_seg = converter.to_segmentation(mitk_seg)
    assert len(mw_seg.groups) == 2


def test_to_segmentation_preserves_label_values() -> None:
    converter = MitkSegmentationConverter()
    mitk_seg = _make_mitk_seg()
    mw_seg = converter.to_segmentation(mitk_seg)
    values = sorted(lbl.value for lbl in mw_seg.labels)
    assert values == [1, 2, 3]


def test_to_segmentation_preserves_label_names() -> None:
    converter = MitkSegmentationConverter()
    mitk_seg = _make_mitk_seg()
    mw_seg = converter.to_segmentation(mitk_seg)
    names = {lbl.value: lbl.name for lbl in mw_seg.labels}
    assert names[1] == "Liver"
    assert names[2] == "Spleen"
    assert names[3] == "Tumor"


def test_to_segmentation_preserves_pixel_data() -> None:
    converter = MitkSegmentationConverter()
    mitk_seg = _make_mitk_seg()
    mw_seg = converter.to_segmentation(mitk_seg)

    arr0 = mw_seg.get_group_image(0).array
    assert int(arr0[0, 0, 0]) == 1
    assert int(arr0[1, 1, 1]) == 2

    arr1 = mw_seg.get_group_image(1).array
    assert int(arr1[2, 2, 2]) == 3


def test_to_segmentation_cleans_up_tempfile() -> None:
    converter = MitkSegmentationConverter()
    mitk_seg = _make_mitk_seg()

    captured: list[Path] = []
    original_ntf = tempfile.NamedTemporaryFile

    def tracking_ntf(*args: object, **kwargs: object) -> object:
        ctx = original_ntf(*args, **kwargs)
        captured.append(Path(ctx.name))
        return ctx

    with patch.object(tempfile, "NamedTemporaryFile", side_effect=tracking_ntf):
        converter.to_segmentation(mitk_seg)

    assert captured, "NamedTemporaryFile was never called"
    assert not captured[0].exists(), "Temp file was not cleaned up"


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


def test_round_trip_mw_to_mitk_to_mw() -> None:
    converter = MitkSegmentationConverter()
    original = _make_mw_seg()
    mitk_seg = converter.from_segmentation(original)
    restored = converter.to_segmentation(mitk_seg)

    assert len(restored.groups) == len(original.groups)
    assert sorted(lbl.value for lbl in restored.labels) == sorted(
        lbl.value for lbl in original.labels
    )
    assert {lbl.value: lbl.name for lbl in restored.labels} == {
        lbl.value: lbl.name for lbl in original.labels
    }
    np.testing.assert_allclose(restored.spacing, original.spacing, rtol=1e-5)
    np.testing.assert_allclose(restored.origin, original.origin, rtol=1e-5)

    arr0_orig = original.get_group_image(0).array
    arr0_rest = restored.get_group_image(0).array
    assert np.array_equal(arr0_orig, arr0_rest)


def test_round_trip_mitk_to_mw_to_mitk() -> None:
    converter = MitkSegmentationConverter()
    original = _make_mitk_seg()
    mw_seg = converter.to_segmentation(original)
    restored = converter.from_segmentation(mw_seg)

    assert restored.num_groups == original.num_groups
    orig_values = sorted(int(v) for v in original.label_values if int(v) != 0)
    rest_values = sorted(int(v) for v in restored.label_values if int(v) != 0)
    assert rest_values == orig_values
