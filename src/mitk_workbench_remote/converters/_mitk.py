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

"""MITK native Image converter -- optional dependency.

Registered automatically at import time if the ``mitk`` package is installed.
Preserves spacing, origin, direction, and properties through the conversion.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mitk
import numpy as np

_log = logging.getLogger(__name__)


class MitkImageConverter:
    """Converter for ``mitk.Image`` objects (native MITK Python binding).

    Auto-registered when the ``mitk`` package is importable. Provides
    round-trip fidelity for geometry (spacing, origin, direction) and pixel
    data.  Properties are extracted via the binding's ``get_property()``
    coercion path (``propertyToPythonValue``) for known types, falling back
    to ``BaseProperty.to_json()`` for unknown types.
    """

    @property
    def target_type(self) -> type:
        return mitk.Image  # type: ignore[no-any-return]

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, mitk.Image)

    def extract_geometry(self, obj: Any) -> dict[str, Any]:
        img: mitk.Image = obj
        ndim: int = img.ndim

        # MITK geometry is always 3D internally; cap to img.ndim for 2D images.
        spacing_3d = tuple(img.get_spacing(time_step=0))
        origin_3d = tuple(img.get_origin(time_step=0))
        direction_3d: np.ndarray = np.asarray(img.get_direction(time_step=0), dtype=np.float64)

        if ndim < 3:
            spacing = spacing_3d[:ndim]
            origin = origin_3d[:ndim]
            direction = direction_3d[:ndim, :ndim]
        else:
            spacing = spacing_3d
            origin = origin_3d
            direction = direction_3d

        return {
            "spacing": spacing,
            "origin": origin,
            "direction": direction,
        }

    def extract_metadata(self, obj: Any) -> dict[str, Any]:
        img: mitk.Image = obj
        metadata: dict[str, Any] = {}
        for key in img.property_keys:
            val = img.get_property(key)  # raw=False: returns coerced Python value
            if val is None:
                continue
            if isinstance(val, mitk.BaseProperty):
                # Unknown type -- no native Python equivalent.
                # to_json() is guaranteed on every BaseProperty subclass.
                metadata[key] = json.loads(val.to_json())
            else:
                metadata[key] = val
        return metadata

    def to_ndarray(self, obj: Any) -> np.ndarray:
        img: mitk.Image = obj
        return img.as_numpy(writeable=False)  # type: ignore[no-any-return]

    def to_nrrd_bytes(self, obj: Any) -> bytes:
        img: mitk.Image = obj
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            mitk.IOUtil.save(img, str(tmp_path))
            return tmp_path.read_bytes()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def from_image(self, image: Any) -> mitk.Image:
        from mitk_workbench_remote.image import Image as _Image

        img: _Image = image
        direction = np.asarray(img.direction, dtype=np.float64)
        return mitk.Image.from_numpy(  # type: ignore[no-any-return]
            img.array,
            spacing=img.spacing,
            origin=img.origin,
            direction=direction,
            copy=True,
        )
