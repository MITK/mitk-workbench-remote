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

import mitk_workbench_remote as mw
from mitk_workbench_remote.protocols import SpatialImage


def test_spatial_image_is_runtime_checkable() -> None:
    img = mw.Image(np.zeros((3, 4, 5)))
    assert isinstance(img, SpatialImage)


def test_spatial_image_with_custom_geometry() -> None:
    img = mw.Image(
        np.zeros((3, 4, 5)),
        spacing=(0.5, 0.5, 1.0),
        origin=(10.0, 20.0, 30.0),
        direction=np.eye(3),
    )
    assert isinstance(img, SpatialImage)


def test_multilabel_does_not_satisfy_spatial_image() -> None:
    seg = mw.MultiLabelSegmentation(groups=[], labels={})
    assert not isinstance(seg, SpatialImage)


def test_plain_ndarray_does_not_satisfy_spatial_image() -> None:
    arr = np.zeros((3, 4, 5))
    assert not isinstance(arr, SpatialImage)


def test_custom_class_satisfying_protocol() -> None:
    class MinimalImage:
        @property
        def array(self) -> np.ndarray:
            return np.zeros((2, 2))

        @property
        def spacing(self) -> tuple[float, ...]:
            return (1.0, 1.0)

        @property
        def origin(self) -> tuple[float, ...]:
            return (0.0, 0.0)

        @property
        def direction(self) -> np.ndarray:
            return np.eye(2)

        @property
        def ndim(self) -> int:
            return 2

        @property
        def shape(self) -> tuple[int, ...]:
            return (2, 2)

        @property
        def dtype(self) -> np.dtype:
            return np.dtype(np.float64)

    assert isinstance(MinimalImage(), SpatialImage)


def test_protocol_members_accessible() -> None:
    img = mw.Image(np.ones((3, 4, 5), dtype=np.float32), spacing=(0.5, 0.5, 1.0))
    spatial: SpatialImage = img
    assert isinstance(spatial.array, np.ndarray)
    assert isinstance(spatial.spacing, tuple)
    assert isinstance(spatial.origin, tuple)
    assert isinstance(spatial.direction, np.ndarray)
    assert isinstance(spatial.ndim, int)
    assert isinstance(spatial.shape, tuple)
    assert spatial.dtype == np.dtype(np.float32)


def test_spatial_image_usable_in_type_hints() -> None:
    def process(img: SpatialImage) -> tuple[int, ...]:
        return img.shape

    img = mw.Image(np.zeros((3, 4, 5)))
    result = process(img)
    assert result == (3, 4, 5)
