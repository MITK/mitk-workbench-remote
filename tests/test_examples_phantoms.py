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

"""Tests for the synthetic phantom helpers used by the example notebooks."""

from __future__ import annotations

import numpy as np
import pytest

from mitk_workbench_remote.examples.phantoms import (
    DEFAULT_ORIGIN,
    DEFAULT_SHAPE,
    DEFAULT_SPACING,
    CaseBundle,
    make_body_with_tumor,
    make_cohort,
    make_ct_like,
    make_mr_like,
    make_segmentation,
)
from mitk_workbench_remote.image import Image
from mitk_workbench_remote.multilabel import LABEL_DTYPE, MultiLabelSegmentation

# ---------------------------------------------------------------------------
# make_ct_like / make_mr_like
# ---------------------------------------------------------------------------


def test_make_ct_like_returns_image_with_default_geometry() -> None:
    img = make_ct_like()
    assert isinstance(img, Image)
    assert img.shape == DEFAULT_SHAPE
    assert img.spacing == DEFAULT_SPACING
    assert img.origin == DEFAULT_ORIGIN
    assert img.array.dtype == np.float32


def test_make_ct_like_intensity_range_is_hu_ish() -> None:
    img = make_ct_like()
    arr = img.array
    # Three distinct populations: -1000 (air), 40 (body), 200 (inner).
    assert arr.min() == pytest.approx(-1000.0)
    assert arr.max() == pytest.approx(200.0)
    assert (arr == 40.0).any()
    assert (arr == 200.0).any()


def test_make_mr_like_intensity_range_is_t1_ish() -> None:
    img = make_mr_like()
    arr = img.array
    assert arr.min() == pytest.approx(0.0)
    assert arr.max() == pytest.approx(1200.0)
    assert (arr == 600.0).any()
    assert (arr == 1200.0).any()


def test_make_ct_like_respects_custom_geometry() -> None:
    shape = (16, 16, 16)
    spacing = (0.5, 0.5, 1.0)
    origin = (10.0, -20.0, 5.0)
    img = make_ct_like(shape=shape, spacing=spacing, origin=origin)
    assert img.shape == shape
    assert img.spacing == spacing
    assert img.origin == origin


# ---------------------------------------------------------------------------
# make_body_with_tumor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["ct", "mr"])
def test_make_body_with_tumor_returns_image(modality: str) -> None:
    img = make_body_with_tumor(modality)  # type: ignore[arg-type]
    assert isinstance(img, Image)
    assert img.shape == DEFAULT_SHAPE


def test_make_body_with_tumor_ct_uses_hu_intensities() -> None:
    img = make_body_with_tumor("ct")
    arr = img.array
    assert arr.min() == pytest.approx(-1000.0)
    assert (arr == 40.0).any()
    assert (arr == 200.0).any()


def test_make_body_with_tumor_mr_uses_t1_intensities() -> None:
    img = make_body_with_tumor("mr")
    arr = img.array
    assert (arr == 600.0).any()
    assert (arr == 1200.0).any()


def test_make_body_with_tumor_rejects_unknown_modality() -> None:
    with pytest.raises(ValueError, match="modality"):
        make_body_with_tumor("pet")  # type: ignore[arg-type]


def test_make_body_with_tumor_offset_moves_the_tumor() -> None:
    centered = make_body_with_tumor("mr", tumor_offset_mm=(0.0, 0.0, 0.0))
    offset = make_body_with_tumor("mr", tumor_offset_mm=(10.0, 0.0, 0.0))
    # Centroid of the tumor-only mask should shift along axis 0 (Z).
    centered_mask = centered.array == 1200.0
    offset_mask = offset.array == 1200.0
    centered_centroid = np.array(np.where(centered_mask)).mean(axis=1)
    offset_centroid = np.array(np.where(offset_mask)).mean(axis=1)
    assert offset_centroid[0] > centered_centroid[0] + 1.0


# ---------------------------------------------------------------------------
# make_segmentation
# ---------------------------------------------------------------------------


def test_make_segmentation_inherits_reference_geometry() -> None:
    img = make_mr_like()
    seg = make_segmentation(img, label_names=("a", "b"))
    assert isinstance(seg, MultiLabelSegmentation)
    assert seg.spacing == img.spacing
    assert seg.origin == img.origin
    np.testing.assert_array_equal(seg.direction, img.direction)


def test_make_segmentation_creates_one_group_with_named_labels() -> None:
    img = make_mr_like()
    seg = make_segmentation(img, label_names=("tumor", "edema"), group_name="primary")
    assert len(seg.groups) == 1
    assert seg.groups[0].name == "primary"
    labels = seg.labels
    assert [label.name for label in labels] == ["tumor", "edema"]
    assert [label.value for label in labels] == [1, 2]


def test_make_segmentation_label_voxels_are_non_empty_and_disjoint() -> None:
    img = make_mr_like()
    seg = make_segmentation(img, label_names=("a", "b", "c"), seed=42)
    arr = seg.get_group_image(0).array
    assert arr.dtype == LABEL_DTYPE
    for value in (1, 2, 3):
        assert (arr == value).any(), f"label {value} produced no voxels"
    # Voxels carry one label each (no fractional / overlap encoding).
    assert set(np.unique(arr).tolist()) == {0, 1, 2, 3}


def test_make_segmentation_is_reproducible_under_a_seed() -> None:
    img = make_mr_like()
    a = make_segmentation(img, seed=7)
    b = make_segmentation(img, seed=7)
    np.testing.assert_array_equal(a.get_group_image(0).array, b.get_group_image(0).array)


def test_make_segmentation_rejects_empty_label_names() -> None:
    img = make_mr_like()
    with pytest.raises(ValueError, match="label_names"):
        make_segmentation(img, label_names=())


# ---------------------------------------------------------------------------
# make_cohort
# ---------------------------------------------------------------------------


def test_make_cohort_returns_n_distinct_cases() -> None:
    cases = make_cohort(n=3, seed=0)
    assert len(cases) == 3
    case_ids = [c.case_id for c in cases]
    assert case_ids == ["case_001", "case_002", "case_003"]
    assert len(set(case_ids)) == 3


def test_make_cohort_each_case_has_consistent_geometry() -> None:
    cases = make_cohort(n=2, seed=1)
    for case in cases:
        assert isinstance(case, CaseBundle)
        assert case.image.shape == case.segmentation.get_group_image(0).shape
        assert case.image.spacing == case.segmentation.spacing
        assert case.image.origin == case.segmentation.origin


def test_make_cohort_metadata_carries_modality_and_offset() -> None:
    case = make_cohort(n=1, seed=0, modality="ct")[0]
    assert case.metadata["modality"] == "ct"
    assert "tumor_offset_mm" in case.metadata
    assert "seed" in case.metadata


def test_make_cohort_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        make_cohort(n=0)
