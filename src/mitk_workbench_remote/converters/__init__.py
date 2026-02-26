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

"""Converter registry — maps Python types to NRRD I/O operations.

Functions:
    register_converter: Register a new ImageConverter for a custom type.
    find_converter: Look up the converter for a given object by type.

Built-in converters (registered at import time):
    NumpyConverter: numpy.ndarray ↔ NRRD (always available).
    ImageConverter: Image ↔ NRRD (always available).
    FilePathConverter: str/Path → file-reference upload (always available).
    SitkConverter: SimpleITK.Image ↔ Image (registered if SimpleITK is installed).
    MLArrayConverter: mlarray.MLArray ↔ Image (registered if mlarray is installed).
"""
