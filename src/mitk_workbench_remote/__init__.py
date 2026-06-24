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

"""mitk-workbench-remote — Remote control for MITK Workbench from Python.

Example:
    >>> import mitk_workbench_remote as mw
    >>> wb = mw.connect("http://localhost:8080")
    >>> wb.ping()
    True
"""

import logging

from mitk_workbench_remote._version import __version__
from mitk_workbench_remote.discovery import discover, launch
from mitk_workbench_remote.editors import (
    Camera,
    EditorAlias,
    EditorBase,
    EditorDescriptor,
    EditorInfo,
    MxNEditor,
    MxNRenderWindow,
    MxNWindowSummary,
    RenderWindow,
    SelectedSlice,
    SliceBounds,
    StandardView,
    StdMultiEditor,
    ViewDirection,
    WindowKind,
    WindowSummary,
)
from mitk_workbench_remote.errors import (
    EditorNotActiveError,
    RenderingError,
    RenderWindowNotFoundError,
    UnsupportedDataTypeError,
    UnsupportedOperationError,
)
from mitk_workbench_remote.image import Image
from mitk_workbench_remote.multilabel import LABEL_DTYPE, Label, LabelGroup, MultiLabelSegmentation
from mitk_workbench_remote.node import DataNode, DataRepresentation, PropertyScope
from mitk_workbench_remote.protocols import SpatialImage
from mitk_workbench_remote.storage import DataStorage
from mitk_workbench_remote.transport import InsecureTransportWarning, TransferMode
from mitk_workbench_remote.workbench import (
    PositionBounds,
    ReinitMode,
    RenderWindows,
    ScreenshotFormat,
    SelectedPosition,
    SelectedTime,
    TimeBounds,
    Workbench,
    WorkbenchInfo,
    connect,
)

__all__ = [
    "LABEL_DTYPE",
    "Camera",
    "DataNode",
    "DataRepresentation",
    "DataStorage",
    "EditorAlias",
    "EditorBase",
    "EditorDescriptor",
    "EditorInfo",
    "EditorNotActiveError",
    "Image",
    "InsecureTransportWarning",
    "Label",
    "LabelGroup",
    "MultiLabelSegmentation",
    "MxNEditor",
    "MxNRenderWindow",
    "MxNWindowSummary",
    "PositionBounds",
    "PropertyScope",
    "ReinitMode",
    "RenderWindow",
    "RenderWindowNotFoundError",
    "RenderWindows",
    "RenderingError",
    "ScreenshotFormat",
    "SelectedPosition",
    "SelectedSlice",
    "SelectedTime",
    "SliceBounds",
    "SpatialImage",
    "StandardView",
    "StdMultiEditor",
    "TimeBounds",
    "TransferMode",
    "UnsupportedDataTypeError",
    "UnsupportedOperationError",
    "ViewDirection",
    "WindowKind",
    "WindowSummary",
    "Workbench",
    "WorkbenchInfo",
    "__version__",
    "connect",
    "discover",
    "launch",
]

logging.getLogger("mitk_workbench_remote").addHandler(logging.NullHandler())
