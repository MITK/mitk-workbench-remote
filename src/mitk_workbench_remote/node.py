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

"""DataNode wrapper — a single node in the MITK DataStorage.

Classes:
    DataNode: Represents one node with property shortcuts (name, visible, opacity, color),
        data access (get_data, set_data), child management, and batch property updates.
"""
from __future__ import annotations

import html as _html
from collections.abc import Callable
from enum import Enum
from typing import Any, cast

from mitk_workbench_remote.transport import RestTransport, TransferMode


class PropertyScope(str, Enum):
    """Valid scopes for node property operations.

    Attributes:
        ALL: Include both node-scope and data-scope properties (read-only).
        NODE: Node-level display properties (e.g. visible, opacity, color).
        DATA: Data-level properties embedded in the data object.
    """

    ALL = "all"
    NODE = "node"
    DATA = "data"


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


# ---------------------------------------------------------------------------
# DataNode
# ---------------------------------------------------------------------------


class DataNode:
    """A single node in the MITK DataStorage.

    Do not construct directly — obtain instances from
    :class:`~mitk_workbench_remote.storage.DataStorage`.

    Display-only caches (``_name``, ``_path``, ``_parent_uid``) are seeded
    from the node dict at construction and refreshed by :meth:`refresh`.
    Property getters for :attr:`name`, :attr:`visible`, :attr:`opacity`, and
    :attr:`color` always make a live REST call.

    Args:
        uid: Immutable node identifier.
        name: Initial display name (cached; used by ``__repr__``).
        data_type: MITK data type string (e.g. ``"Image"``), updated by
            :meth:`refresh`.
        path: Full storage path (e.g. ``"/CT_Scan/seg"``), updated by
            :meth:`refresh`.
        parent_uid: UID of the parent node, or ``None`` for top-level nodes.
        transport: Transport used for all HTTP calls.
    """

    def __init__(
        self,
        uid: str,
        *,
        name: str,
        data_type: str | None,
        path: str,
        parent_uid: str | None,
        transport: RestTransport,
    ) -> None:
        self._uid = uid
        self._data_type = data_type
        self._name = name  # display cache, updated by name setter and refresh()
        self._path = path  # display cache, updated by refresh
        self._parent_uid = parent_uid  # display cache, updated by refresh
        self._transport = transport

    @classmethod
    def _from_node_dict(cls, data: dict[str, Any], transport: RestTransport) -> DataNode:
        """Construct a DataNode from a server node dict.

        Args:
            data: Parsed node object from the API response.
            transport: Transport used for all HTTP calls.

        Returns:
            A new :class:`DataNode` instance.
        """
        return cls(
            uid=data["uid"],
            name=data["name"],
            data_type=data.get("data_type"),
            path=data.get("path", ""),
            parent_uid=data.get("parent_uid"),
            transport=transport,
        )

    # ------------------------------------------------------------------
    # Identity (cached; immutable or refreshed)
    # ------------------------------------------------------------------

    @property
    def uid(self) -> str:
        """Immutable node identifier."""
        return self._uid

    @property
    def data_type(self) -> str | None:
        """MITK data type string (e.g. ``"Image"``). Updated by :meth:`refresh`."""
        return self._data_type

    @property
    def path(self) -> str:
        """Full storage path (e.g. ``"/CT_Scan/seg"``). Updated by :meth:`refresh`."""
        return self._path

    # ------------------------------------------------------------------
    # Common properties — live REST calls via get_property / set_property
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Node display name. Always fetches from the server."""
        return cast(str, self.get_property("name"))

    @name.setter
    def name(self, value: str) -> None:
        self.set_property("name", value)
        self._name = value  # keep display cache in sync

    @property
    def visible(self) -> bool:
        """Node visibility flag. Always fetches from the server."""
        return cast(bool, self.get_property("visible"))

    @visible.setter
    def visible(self, value: bool) -> None:
        self.set_property("visible", value)

    @property
    def opacity(self) -> float:
        """Node opacity (0.0 - 1.0). Always fetches from the server."""
        return cast(float, self.get_property("opacity"))

    @opacity.setter
    def opacity(self, value: float) -> None:
        self.set_property("opacity", value)

    @property
    def color(self) -> tuple[float, float, float]:
        """Node color as an (r, g, b) tuple. Always fetches from the server."""
        return cast(tuple[float, float, float], self.get_property("color"))

    @color.setter
    def color(self, value: tuple[float, float, float]) -> None:
        self.set_property("color", value)

    # ------------------------------------------------------------------
    # Batch property update
    # ------------------------------------------------------------------

    def update_properties(
        self,
        scope: PropertyScope = PropertyScope.NODE,
        *,
        context: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Update multiple properties in a single PATCH request.

        Property values are serialized using the same rules as
        :meth:`set_property`. Pass ``color`` as a ``(r, g, b)`` tuple;
        all other built-in properties accept plain Python values.

        Args:
            scope: Property scope (:attr:`~PropertyScope.NODE` or
                :attr:`~PropertyScope.DATA`).
            context: Optional renderer context identifier. Only relevant for
                scope :attr:`~PropertyScope.NODE`; ignored at data scope.
            **kwargs: Property key-value pairs to update.
        """
        body = {k: _serialize_property(k, v) for k, v in kwargs.items()}
        params: dict[str, str] = {"property_scope": scope}
        if context is not None:
            params["context"] = context
        self._transport.patch(
            f"/datastorage/nodes/{self._uid}/properties",
            json=body,
            params=params,
        )
        if "name" in kwargs:
            self._name = kwargs["name"]  # keep display cache in sync

    # ------------------------------------------------------------------
    # Property CRUD
    # ------------------------------------------------------------------

    def get_properties(
        self,
        *,
        scope: PropertyScope = PropertyScope.ALL,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Fetch all properties for this node.

        Args:
            scope: Property scope filter.
            context: Optional renderer context identifier.

        Returns:
            Mapping of property name to deserialized value.
        """
        params: dict[str, str] = {"property_scope": scope}
        if context is not None:
            params["context"] = context
        resp = self._transport.get(f"/datastorage/nodes/{self._uid}/properties", params=params)
        raw: dict[str, Any] = resp.json()["data"]["properties"]
        return {k: _deserialize_property(v) for k, v in raw.items()}

    def get_property(
        self,
        key: str,
        *,
        scope: PropertyScope = PropertyScope.ALL,
        context: str | None = None,
    ) -> Any:
        """Fetch a single property by key.

        Args:
            key: Property name (e.g. ``"name"``, ``"visible"``).
            scope: Property scope filter.
            context: Optional renderer context identifier.

        Returns:
            Deserialized property value.

        Raises:
            NodeNotFoundError: If the node or property does not exist.
        """
        params: dict[str, str] = {"property_scope": scope}
        if context is not None:
            params["context"] = context
        resp = self._transport.get(
            f"/datastorage/nodes/{self._uid}/properties/{key}", params=params
        )
        return _deserialize_property(resp.json()["data"][key])

    def set_property(
        self,
        key: str,
        value: Any,
        *,
        scope: PropertyScope = PropertyScope.NODE,
        context: str | None = None,
    ) -> None:
        """Set a single property by key.

        Pass ``color`` as a ``(r, g, b)`` tuple. Unknown complex types can
        be round-tripped by passing the raw wire dict (any dict that has a
        ``"type"`` key is passed through unchanged).

        Args:
            key: Property name.
            value: New value. Built-in types are serialized automatically;
                raw dicts with a ``"type"`` key are forwarded as-is.
            scope: Property scope (:attr:`~PropertyScope.NODE` or
                :attr:`~PropertyScope.DATA`).
            context: Optional renderer context identifier. Only relevant for
                scope :attr:`~PropertyScope.NODE`; ignored at data scope.
        """
        body = _serialize_property(key, value)
        params: dict[str, str] = {"property_scope": scope}
        if context is not None:
            params["context"] = context
        self._transport.put(
            f"/datastorage/nodes/{self._uid}/properties/{key}",
            json=body,
            params=params,
        )

    def delete_property(self, key: str, *, scope: PropertyScope = PropertyScope.NODE) -> None:
        """Delete a property by key.

        Args:
            key: Property name to remove.
            scope: Property scope (:attr:`~PropertyScope.NODE` or
                :attr:`~PropertyScope.DATA`).
        """
        self._transport.delete(
            f"/datastorage/nodes/{self._uid}/properties/{key}",
            params={"property_scope": scope},
        )

    # ------------------------------------------------------------------
    # Children
    # ------------------------------------------------------------------

    @property
    def children(self) -> list[DataNode]:
        """Child nodes of this node. Always fetches from the server."""
        resp = self._transport.get(f"/datastorage/nodes/{self._uid}/children")
        return [DataNode._from_node_dict(d, self._transport) for d in resp.json()["data"]]

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, *, recursive: bool = False) -> None:
        """Remove this node from the DataStorage.

        Args:
            recursive: When ``True``, also remove all child nodes.

        Raises:
            NodeNotFoundError: If the node no longer exists.
        """
        params: dict[str, str] = {}
        if recursive:
            params["recursive"] = "true"
        self._transport.delete(f"/datastorage/nodes/{self._uid}", params=params)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Refresh cached metadata from the server.

        Updates ``data_type``, ``path``, and the display caches for
        ``_name`` and ``_parent_uid``.

        Raises:
            NodeNotFoundError: If the node no longer exists.
        """
        resp = self._transport.get(f"/datastorage/nodes/{self._uid}")
        data: dict[str, Any] = resp.json()["data"]
        self._data_type = data.get("data_type")
        self._name = data["name"]
        self._path = data.get("path", "")
        self._parent_uid = data.get("parent_uid")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def reinit(self) -> None:
        """Fit render window cameras to this node's geometry.

        Convenience wrapper around ``POST /rendering/reinit`` for a single
        node. Equivalent to ``workbench.reinit([self])``.

        Raises:
            ApiError: If this node has no data (code ``NO_DATA``) or no
                usable geometry (code ``NO_GEOMETRY``).
            RenderingError: If the rendering framework reports a failure.
        """
        self._transport.post("/rendering/reinit", json={"uids": [self._uid]})

    # ------------------------------------------------------------------
    # Repr (no REST calls)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<DataNode {self._name!r} ({self._data_type})"
            f" @ {self._path} uid={self._uid}>"
        )

    def _repr_html_(self) -> str:
        e = _html.escape
        rows = "".join(
            [
                f"<tr><th>uid</th><td>{e(self._uid)}</td></tr>",
                f"<tr><th>name</th><td>{e(self._name)}</td></tr>",
                f"<tr><th>data_type</th><td>{e(self._data_type or '')}</td></tr>",
                f"<tr><th>path</th><td>{e(self._path)}</td></tr>",
                f"<tr><th>parent_uid</th><td>{e(self._parent_uid or '')}</td></tr>",
            ]
        )
        return f"<table>{rows}</table>"
