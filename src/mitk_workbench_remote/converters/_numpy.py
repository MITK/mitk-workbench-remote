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

"""numpy converter -- always available."""

from __future__ import annotations

from typing import Any

import numpy as np


class NumpyConverter:
    """Converter for numpy ndarrays."""

    @property
    def target_type(self) -> type:
        return np.ndarray

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def extract_geometry(self, obj: Any) -> dict[str, Any]:
        return {}

    def extract_properties(self, obj: Any) -> dict[str, Any]:
        return {}

    def to_ndarray(self, obj: Any) -> np.ndarray:
        return obj  # type: ignore[no-any-return]

    def to_nrrd_bytes(self, obj: Any) -> bytes:
        from mitk_workbench_remote._io.nrrd import write_nrrd
        from mitk_workbench_remote.image import Image

        image = Image(obj)
        return write_nrrd(image)

    def from_image(self, image: Any) -> np.ndarray:
        return image.array  # type: ignore[no-any-return]
