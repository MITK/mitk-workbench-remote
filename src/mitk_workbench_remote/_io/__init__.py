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

"""NRRD I/O and MultiLabel parsing -- internal sub-package."""

from mitk_workbench_remote._io.multilabel_nrrd import (
    is_multilabel_nrrd,
    parse_labelgroups_json,
    read_multilabel_nrrd_raw,
    serialize_labelgroups_json,
    write_multilabel_nrrd_raw,
)
from mitk_workbench_remote._io.nrrd import read_nrrd, write_nrrd

__all__ = [
    "is_multilabel_nrrd",
    "parse_labelgroups_json",
    "read_multilabel_nrrd_raw",
    "read_nrrd",
    "serialize_labelgroups_json",
    "write_multilabel_nrrd_raw",
    "write_nrrd",
]
