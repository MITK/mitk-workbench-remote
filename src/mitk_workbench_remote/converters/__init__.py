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

"""Converter registry -- maps Python types to NRRD I/O operations.

Functions:
    register_converter: Register a new ImageConverter for a custom type.
    find_image_converter: Look up the converter for a given object by type.
    find_converter_for_type: Reverse lookup -- find converter that produces a target type.

Built-in converters (registered at import time):
    NumpyConverter: numpy.ndarray (always available).
    SitkConverter: SimpleITK.Image (registered if SimpleITK is installed).
    MLArrayConverter: mlarray.MLArray (registered if mlarray is installed).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol, runtime_checkable

import numpy as np

_log = logging.getLogger(__name__)


@runtime_checkable
class ImageConverter(Protocol):
    """Protocol for converting between external image types and Image."""

    def can_handle(self, obj: Any) -> bool:
        """Return True if this converter can handle the given object."""
        ...

    def extract_geometry(self, obj: Any) -> dict[str, Any]:
        """Extract spatial geometry (spacing, origin, direction) from the object."""
        ...

    def extract_metadata(self, obj: Any) -> dict[str, Any]:
        """Extract custom properties/properties from the object."""
        ...

    def to_ndarray(self, obj: Any) -> np.ndarray:
        """Convert the object's pixel data to a numpy array."""
        ...

    def to_nrrd_bytes(self, obj: Any) -> bytes:
        """Serialize the object to NRRD bytes for upload."""
        ...

    def from_image(self, image: Any) -> Any:
        """Convert an Image to the target type."""
        ...

    @property
    def target_type(self) -> type | None:
        """The type this converter produces via from_image, or None."""
        ...


_converters: list[ImageConverter] = []
_converters_lock = threading.Lock()


def register_converter(converter: ImageConverter) -> None:
    """Register a new ImageConverter.

    Thread-safe. Converters should be registered before any concurrent image
    operations; registrations that race with lookups may not be visible
    immediately.
    """
    with _converters_lock:
        _converters.append(converter)
    _log.debug("Registered converter: %s", type(converter).__name__)


def find_image_converter(obj: Any) -> ImageConverter | None:
    """Find the first converter that can handle the given object."""
    for converter in _converters:
        if converter.can_handle(obj):
            _log.debug("Found converter for %s: %s", type(obj).__name__, type(converter).__name__)
            return converter
    _log.debug("No converter found for %s", type(obj).__name__)
    return None


def find_converter_for_type(target_type: type) -> ImageConverter | None:
    """Find a converter that produces the given target type via from_image."""
    for converter in _converters:
        if converter.target_type is not None and issubclass(converter.target_type, target_type):
            return converter
    return None


# --- Auto-register built-in converters ---

from mitk_workbench_remote.converters._numpy import NumpyConverter  # noqa: E402

register_converter(NumpyConverter())

try:
    from mitk_workbench_remote.converters._simpleitk import SitkConverter

    register_converter(SitkConverter())
except ImportError:
    pass

try:
    from mitk_workbench_remote.converters._mlarray import MLArrayConverter

    register_converter(MLArrayConverter())
except ImportError:
    pass
