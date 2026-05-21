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

"""Editors subpackage — handles for the StdMultiWidget and MxN editors.

Re-exports the public surface so callers can write::

    from mitk_workbench_remote.editors import StdMultiEditor, MxNEditor
"""

from mitk_workbench_remote.editors._base import (
    Camera,
    EditorAlias,
    EditorBase,
    EditorDescriptor,
    EditorInfo,
    MxNWindowSummary,
    RenderWindow,
    SelectedSlice,
    SliceBounds,
    StandardView,
    ViewDirection,
    WindowKind,
    WindowSummary,
)
from mitk_workbench_remote.editors.mxn import MxNEditor, MxNRenderWindow
from mitk_workbench_remote.editors.std_multi import StdMultiEditor

__all__ = [
    "Camera",
    "EditorAlias",
    "EditorBase",
    "EditorDescriptor",
    "EditorInfo",
    "MxNEditor",
    "MxNRenderWindow",
    "MxNWindowSummary",
    "RenderWindow",
    "SelectedSlice",
    "SliceBounds",
    "StandardView",
    "StdMultiEditor",
    "ViewDirection",
    "WindowKind",
    "WindowSummary",
]
