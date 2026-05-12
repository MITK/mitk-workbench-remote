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

"""StdMultiWidget editor handle.

Wraps ``GET /rendering/editors/stdmulti`` and its sub-resources.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mitk_workbench_remote.editors._base import (
    EditorAlias,
    EditorBase,
    RenderWindow,
    ViewDirection,
    WindowKind,
)

# Engine-fixed window slots that the StdMultiWidget always exposes.
# The contract pins these as ``["axial", "sagittal", "coronal", "3d"]``.
_FIXED_WINDOW_KINDS: dict[str, tuple[WindowKind, ViewDirection | None]] = {
    "axial": (WindowKind.TWO_D, ViewDirection.AXIAL),
    "sagittal": (WindowKind.TWO_D, ViewDirection.SAGITTAL),
    "coronal": (WindowKind.TWO_D, ViewDirection.CORONAL),
    "3d": (WindowKind.THREE_D, None),
}


class StdMultiEditor(EditorBase):
    """Handle to the StdMultiWidget editor (axial / sagittal / coronal / 3d)."""

    ALIAS = EditorAlias.STD_MULTI.value

    # ------------------------------------------------------------------
    # Window construction — propagate eager kind/view_direction even when
    # only an ``id`` is known (e.g. ``editor["axial"]``).
    # ------------------------------------------------------------------

    def _make_window(self, payload: Mapping[str, Any]) -> RenderWindow:
        window_id = payload["id"]
        kind: WindowKind | None
        if "kind" in payload:
            kind = WindowKind(payload["kind"])
        else:
            fixed = _FIXED_WINDOW_KINDS.get(window_id)
            kind = None if fixed is None else fixed[0]
        return RenderWindow(
            self._transport,
            self.ALIAS,
            window_id,
            kind=kind,
        )

    # ------------------------------------------------------------------
    # Named-window properties
    # ------------------------------------------------------------------

    @property
    def axial(self) -> RenderWindow:
        """The axial 2D window."""
        return self._make_window({"id": "axial"})

    @property
    def sagittal(self) -> RenderWindow:
        """The sagittal 2D window."""
        return self._make_window({"id": "sagittal"})

    @property
    def coronal(self) -> RenderWindow:
        """The coronal 2D window."""
        return self._make_window({"id": "coronal"})

    @property
    def three_d(self) -> RenderWindow:
        """The 3D window. Equivalent to ``editor["3d"]`` (``"3d"`` is not a
        valid Python identifier, hence the spelled-out alias)."""
        return self._make_window({"id": "3d"})
