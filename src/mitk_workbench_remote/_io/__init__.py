# SPDX-FileCopyrightText: 2024, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
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

"""NRRD I/O and MultiLabel parsing — internal sub-package.

Public interface:
    read_nrrd(data: bytes) -> Image
    write_nrrd(image: Image) -> bytes
    read_multilabel_nrrd(data: bytes) -> MultiLabelSegmentation
    write_multilabel_nrrd(seg: MultiLabelSegmentation) -> bytes
    is_multilabel_nrrd(header: dict) -> bool
"""
