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

"""Property helpers and serialization utilities.

Handles conversion between Python types and MITK REST API property representations.
Used internally by DataNode for get_property / set_property / update_properties.
"""

from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Property serialization helpers
# ---------------------------------------------------------------------------

# Dispatch table: known complex types with ergonomic Python representations.
# Only types where we know the safe inverse serialization go here.
_COMPLEX_DESERIALIZERS: dict[str, Callable[[Any], Any]] = {
    "ColorProperty": lambda v: tuple(v),  # [r, g, b] -> (r, g, b)
}


def _deserialize_property(raw: Any) -> Any:
    """Deserialize a property value from the wire format.

    Simple types (str, int, float, bool) pass through unchanged.
    Known complex types (e.g. ColorProperty) are converted to ergonomic
    Python objects. Unknown complex types pass through as raw dicts so
    they can be round-tripped via set_property.

    Args:
        raw: Raw value from the JSON response body.

    Returns:
        Deserialized Python value.
    """
    if isinstance(raw, dict) and "type" in raw and "value" in raw:
        fn = _COMPLEX_DESERIALIZERS.get(raw["type"])
        if fn is not None:
            return fn(raw["value"])
    return raw  # simple type OR unknown complex type -> passthrough


_COMPLEX_SERIALIZERS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "color": lambda v: {"type": "ColorProperty", "value": list(v)},
}


def _serialize_property(key: str, value: Any) -> Any:
    """Serialize a property value to the wire format.

    Args:
        key: Property key (used to select the correct wire type for known
            complex properties such as ``"color"``).
        value: Python value to serialize.

    Returns:
        JSON-serializable value in the server wire format.
    """
    if isinstance(value, dict) and "type" in value:
        return value  # already in wire format — escape hatch for unknown complex types
    fn = _COMPLEX_SERIALIZERS.get(key)
    if fn is not None:
        return fn(value)
    return value  # simple type -> passthrough
