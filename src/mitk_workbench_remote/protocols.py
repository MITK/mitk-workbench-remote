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

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SpatialImage(Protocol):
    """Protocol for spatial image types with array data and geometry.

    Both ``mw.Image`` and ``mitk.Image`` satisfy this protocol, enabling
    functions to accept either via type hints::

        def process(img: SpatialImage) -> None:
            arr = img.array
            print(f"Spacing: {img.spacing}")

    Property types are intentionally broad:

    - ``spacing`` and ``origin`` may be ``tuple[float, ...]`` (mw) or
      ``np.ndarray`` (mitk).
    - ``direction`` is always ``np.ndarray``.

    This protocol covers geometry and array access only. Data-scope property
    methods (``properties``, ``property_keys``, ``get_property``,
    ``set_property``, ``remove_property``) are available on ``mw.Image`` and
    ``mw.MultiLabelSegmentation`` but are not part of this protocol contract.
    """

    @property
    def array(self) -> np.ndarray: ...

    @property
    def spacing(self) -> tuple[float, ...] | np.ndarray: ...

    @property
    def origin(self) -> tuple[float, ...] | np.ndarray: ...

    @property
    def direction(self) -> np.ndarray: ...

    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...
