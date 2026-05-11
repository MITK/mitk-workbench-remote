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

"""SimpleITK converter -- optional dependency.

Registered automatically at import time if SimpleITK is installed.
Preserves spacing, origin, and direction through the conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk


class SitkConverter:
    """Converter for SimpleITK Image objects."""

    @property
    def target_type(self) -> type:
        return sitk.Image

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, sitk.Image)

    def extract_geometry(self, obj: Any) -> dict[str, Any]:
        img: Any = obj
        ndim = img.GetDimension()  # type: ignore[no-untyped-call]
        spacing = img.GetSpacing()  # type: ignore[no-untyped-call]
        origin = img.GetOrigin()  # type: ignore[no-untyped-call]
        # SimpleITK direction is a flat tuple of ndim*ndim elements
        direction_flat = img.GetDirection()  # type: ignore[no-untyped-call]
        direction = np.array(direction_flat, dtype=np.float64).reshape(ndim, ndim)
        return {
            "spacing": tuple(spacing),
            "origin": tuple(origin),
            "direction": direction,
        }

    def extract_properties(self, obj: Any) -> dict[str, Any]:
        img: Any = obj
        properties: dict[str, Any] = {}
        for key in img.GetMetaDataKeys():  # type: ignore[no-untyped-call]
            properties[key] = img.GetMetaData(key)  # type: ignore[no-untyped-call]
        return properties

    def to_ndarray(self, obj: Any) -> np.ndarray:
        return sitk.GetArrayFromImage(obj)

    def to_nrrd_bytes(self, obj: Any) -> bytes:
        from mitk_workbench_remote._io.nrrd import write_nrrd
        from mitk_workbench_remote.image import Image

        image = Image(obj)
        return write_nrrd(image)

    def from_image(self, image: Any) -> sitk.Image:
        arr = image.array
        sitk_image: Any = sitk.GetImageFromArray(arr)
        sitk_image.SetSpacing(image.spacing)  # type: ignore[no-untyped-call]
        sitk_image.SetOrigin(image.origin)  # type: ignore[no-untyped-call]
        direction_flat = image.direction.flatten().tolist()
        sitk_image.SetDirection(direction_flat)  # type: ignore[no-untyped-call]
        for key, value in image.properties.items():
            sitk_image.SetMetaData(key, str(value))  # type: ignore[no-untyped-call]
        result: sitk.Image = sitk_image
        return result
