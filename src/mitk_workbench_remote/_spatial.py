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

"""Shared geometry helpers — used by both Image and MultiLabelSegmentation.

Functions:
    _normalize_spacing: Normalize spacing to a tuple of floats; defaults to (1.0, ...).
    _normalize_origin: Normalize origin to a tuple of floats; defaults to (0.0, ...).
    _normalize_direction: Normalize direction to an ndim x ndim identity-defaulting matrix.
"""
