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

"""MultiLabelSegmentation — 4D label volume with group semantics.

Classes:
    MultiLabelSegmentation: Container of label groups backed by a 4D numpy array.
        Label values are globally unique across all groups; value 0 is reserved.
        Supports group-level image access, label metadata editing, and value remapping.
    LabelGroup: A named collection of labels within a MultiLabelSegmentation.
    Label: A single label with value, name, color, opacity, visibility, and lock state.
"""
