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

"""High-level roundtrip tests for read_multilabel_nrrd / write_multilabel_nrrd."""

import numpy as np
import pytest

from mitk_workbench_remote._io.multilabel_nrrd import read_multilabel_nrrd, write_multilabel_nrrd
from mitk_workbench_remote._io.nrrd import read_nrrd, write_nrrd
from mitk_workbench_remote.image import Image
from mitk_workbench_remote.multilabel import LABEL_DTYPE, Label, MultiLabelSegmentation


def _make_full_seg() -> MultiLabelSegmentation:
    """Build a segmentation with two groups and full label metadata."""
    seg = MultiLabelSegmentation.create(
        shape=(4, 5, 6),
        spacing=(0.5, 1.0, 2.0),
        origin=(10.0, 20.0, 30.0),
    )
    g0 = seg.add_group(name="Organs")
    g1 = seg.add_group(name="Lesions")

    seg.add_label(
        Label(
            None,
            "Liver",
            color=(0.8, 0.2, 0.1),
            opacity=0.9,
            visible=True,
            locked=False,
            tracking_id="track-1",
            tracking_uid="uid-1",
            description="Main liver label",
        ),
        group=g0,
    )
    seg.add_label(
        Label(None, "Spleen", color=(0.2, 0.8, 0.3), visible=False, locked=True),
        group=g0,
    )
    seg.add_label(
        Label(None, "Tumor", color=(1.0, 0.0, 0.0)),
        group=g1,
    )
    return seg


def test_roundtrip_all_fields() -> None:
    seg = _make_full_seg()
    nrrd_bytes = write_multilabel_nrrd(seg)
    seg2 = read_multilabel_nrrd(nrrd_bytes)

    # Group count and names
    assert len(seg2.groups) == 2
    assert seg2.groups[0].name == "Organs"
    assert seg2.groups[1].name == "Lesions"

    # Label count per group
    assert len(seg2.get_group_labels(0)) == 2
    assert len(seg2.get_group_labels(1)) == 1

    # Find labels by value
    liver = seg2.get_label(1)
    assert liver is not None
    assert liver.name == "Liver"
    assert liver.color == pytest.approx((0.8, 0.2, 0.1))
    assert liver.opacity == pytest.approx(0.9)
    assert liver.visible is True
    assert liver.locked is False
    assert liver.tracking_id == "track-1"
    assert liver.tracking_uid == "uid-1"
    assert liver.description == "Main liver label"

    spleen = seg2.get_label(2)
    assert spleen is not None
    assert spleen.visible is False
    assert spleen.locked is True

    tumor = seg2.get_label(3)
    assert tumor is not None
    assert tumor.name == "Tumor"
    assert tumor.color == pytest.approx((1.0, 0.0, 0.0))


def test_roundtrip_array_data() -> None:
    seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
    g = seg.add_group("G")
    seg.add_label(Label(1, "A"), group=g)
    seg.add_label(Label(2, "B"), group=g)

    arr = np.zeros((3, 4, 5), dtype=np.uint8)
    arr[0, 0, 0] = 1
    arr[1, 1, 1] = 2
    seg.set_group_image(g, arr)

    nrrd_bytes = write_multilabel_nrrd(seg)
    seg2 = read_multilabel_nrrd(nrrd_bytes)

    recovered = seg2.get_group_image(0).array
    assert recovered[0, 0, 0] == 1
    assert recovered[1, 1, 1] == 2
    assert recovered[2, 2, 2] == 0


def test_roundtrip_geometry_preserved() -> None:
    seg = MultiLabelSegmentation.create(
        shape=(4, 5, 6),
        spacing=(0.5, 1.0, 2.0),
        origin=(10.0, 20.0, 30.0),
    )
    seg.add_group("G")

    nrrd_bytes = write_multilabel_nrrd(seg)
    seg2 = read_multilabel_nrrd(nrrd_bytes)

    assert seg2.spacing == pytest.approx((0.5, 1.0, 2.0))
    assert seg2.origin == pytest.approx((10.0, 20.0, 30.0))


def test_group_with_no_name() -> None:
    seg = MultiLabelSegmentation.create(shape=(3, 3, 3))
    seg.add_group(name=None)  # unnamed group
    seg.add_label(Label(1, "X"), group=0)

    nrrd_bytes = write_multilabel_nrrd(seg)
    seg2 = read_multilabel_nrrd(nrrd_bytes)

    assert len(seg2.groups) == 1
    assert seg2.groups[0].name is None
    assert seg2.get_label(1) is not None
    assert seg2.get_label(1).name == "X"


def test_roundtrip_group_image_dtype_is_label_dtype() -> None:
    """Group images read back from NRRD must use LABEL_DTYPE (uint16)."""
    seg = MultiLabelSegmentation.create(shape=(3, 3, 3))
    g = seg.add_group("G")
    seg.add_label(Label(1, "A"), group=g)

    # Write with uint8 source — set_group_image casts to uint16
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[1, 1, 1] = 1
    seg.set_group_image(g, arr)

    # Verify the stored array is already uint16 before serialisation
    assert seg.get_group_image(g).array.dtype == LABEL_DTYPE

    # Serialise and read back
    nrrd_bytes = write_multilabel_nrrd(seg)
    seg2 = read_multilabel_nrrd(nrrd_bytes)

    recovered = seg2.get_group_image(0)
    assert recovered.array.dtype == LABEL_DTYPE, (
        f"Expected LABEL_DTYPE ({LABEL_DTYPE}) after roundtrip, got {recovered.array.dtype}"
    )
    assert recovered.array[1, 1, 1] == 1


def test_roundtrip_multiple_groups_pixel_data() -> None:
    seg = MultiLabelSegmentation.create(shape=(3, 3, 3))
    g0 = seg.add_group("G0")
    g1 = seg.add_group("G1")
    seg.add_label(Label(1, "A"), group=g0)
    seg.add_label(Label(2, "B"), group=g1)

    arr0 = np.zeros((3, 3, 3), dtype=np.uint8)
    arr0[0, 0, 0] = 1
    seg.set_group_image(g0, arr0)

    arr1 = np.zeros((3, 3, 3), dtype=np.uint8)
    arr1[2, 2, 2] = 2
    seg.set_group_image(g1, arr1)

    nrrd_bytes = write_multilabel_nrrd(seg)
    seg2 = read_multilabel_nrrd(nrrd_bytes)

    assert seg2.get_group_image(0).array[0, 0, 0] == 1
    assert seg2.get_group_image(1).array[2, 2, 2] == 2


def test_seg_and_image_align_in_world_space_after_io_roundtrip() -> None:
    """A label and an image voxel placed at the same numpy index must
    end up at the same world position after both go through their
    respective NRRD I/O paths.

    Regression for the F-order vs C-order multilabel-NRRD axis-order
    bug. Prior to the fix, MultiLabelSegmentation per-group arrays were
    written with their spatial axes transposed relative to Image, so
    the seg's NRRD described a different physical volume from the
    image's NRRD whenever the spatial dims were not all equal. This
    test uses a non-cubic anisotropic shape so any such mismatch
    surfaces.
    """
    # Three distinct spatial sizes so axis swaps cannot hide.
    img_arr = np.zeros((4, 5, 6), dtype=np.float32)
    img_arr[1, 2, 3] = 999.0  # voxel at numpy index (k=1, j=2, i=3)
    spacing = (0.5, 1.0, 2.0)  # (sx, sy, sz)
    img = Image(img_arr, spacing=spacing, origin=(0.0, 0.0, 0.0))

    seg = MultiLabelSegmentation.create(reference=img, dtype=LABEL_DTYPE)
    seg.add_label(Label(1, "spot", color=(1.0, 0.0, 0.0)), group=seg.add_group("G"))
    seg_arr = np.zeros(img.shape, dtype=LABEL_DTYPE)
    seg_arr[1, 2, 3] = 1  # SAME numpy index as the image voxel
    seg.set_group_image(0, seg_arr)

    # Round-trip both through their respective NRRD writers/readers.
    img_back = read_nrrd(write_nrrd(img))
    seg_back = read_multilabel_nrrd(write_multilabel_nrrd(seg))

    # In-memory shape and spacing must remain Image-compatible after
    # round-trip (this alone catches the F-order axis-swap bug).
    assert img_back.array.shape == img.array.shape
    assert seg_back.get_group_image(0).array.shape == img.array.shape
    assert img_back.spacing == img.spacing
    assert seg_back.spacing == img.spacing

    # The label voxel must round-trip to the SAME numpy index as the
    # image voxel: same physical position in world space.
    assert seg_back.get_group_image(0).array[1, 2, 3] == 1
    assert img_back.array[1, 2, 3] == 999.0
    # And asymmetric coords -- arr[1, 2, 3] must NOT equal arr[3, 2, 1]
    # under either round-trip; an axis swap would silently make them so.
    assert seg_back.get_group_image(0).array[3, 2, 1] == 0
    assert img_back.array[3, 2, 1] == 0.0
