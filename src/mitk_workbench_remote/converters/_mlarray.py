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

"""mlarray converter -- optional dependency.

Registered automatically at import time if mlarray is installed.
mlarray is numpy-backed, so conversion is trivial; spatial properties
(spacing, origin, direction) are preserved via mlarray's attributes.
Custom metadata is stored in and read from mlarray's ``meta.extra`` field.
"""

from __future__ import annotations

from typing import Any

import mlarray
import numpy as np


class MLArrayConverter:
    """Converter for mlarray.MLArray objects."""

    @property
    def target_type(self) -> type:
        return mlarray.MLArray  # type: ignore[no-any-return]

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, mlarray.MLArray)

    def extract_geometry(self, obj: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if hasattr(obj, "spacing"):
            result["spacing"] = tuple(obj.spacing)
        if hasattr(obj, "origin"):
            result["origin"] = tuple(obj.origin)
        if hasattr(obj, "direction"):
            result["direction"] = np.asarray(obj.direction, dtype=np.float64)
        return result

    def extract_properties(self, obj: Any) -> dict[str, Any]:
        if hasattr(obj, "meta") and hasattr(obj.meta, "extra") and obj.meta.extra:
            return dict(obj.meta.extra)
        return {}

    def to_ndarray(self, obj: Any) -> np.ndarray:
        return np.asarray(obj)

    def to_nrrd_bytes(self, obj: Any) -> bytes:
        from mitk_workbench_remote._io.nrrd import write_nrrd
        from mitk_workbench_remote.image import Image

        image = Image(obj)
        return write_nrrd(image)

    def from_image(self, image: Any) -> mlarray.MLArray:
        arr = np.asarray(image.array)
        result = mlarray.MLArray(arr)
        if hasattr(result, "spacing"):
            result.spacing = image.spacing
        if hasattr(result, "origin"):
            result.origin = image.origin
        if hasattr(result, "direction"):
            result.direction = image.direction.tolist()
        if image.properties and hasattr(result, "meta") and hasattr(result.meta, "extra"):
            result.meta.extra = dict(image.properties)
        return result
