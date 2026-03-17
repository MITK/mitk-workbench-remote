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

from mitk_workbench_remote._version import __version__
from mitk_workbench_remote.discovery import discover, launch
from mitk_workbench_remote.errors import RenderingError, UnsupportedDataTypeError
from mitk_workbench_remote.image import Image
from mitk_workbench_remote.multilabel import LABEL_DTYPE, Label, LabelGroup, MultiLabelSegmentation
from mitk_workbench_remote.node import DataNode, PropertyScope
from mitk_workbench_remote.storage import DataStorage
from mitk_workbench_remote.workbench import Workbench, WorkbenchInfo, connect

__all__ = [
    "LABEL_DTYPE",
    "DataNode",
    "DataStorage",
    "Image",
    "Label",
    "LabelGroup",
    "MultiLabelSegmentation",
    "PropertyScope",
    "RenderingError",
    "UnsupportedDataTypeError",
    "Workbench",
    "WorkbenchInfo",
    "__version__",
    "connect",
    "discover",
    "launch",
]
