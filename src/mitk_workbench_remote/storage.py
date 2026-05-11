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

"""DataStorage wrapper — Pythonic access to the MITK DataStorage."""

from __future__ import annotations

import builtins
import html as _html
import json
import logging
from collections.abc import Iterator
from typing import Any

from mitk_workbench_remote import errors
from mitk_workbench_remote.node import DataNode, PropertyScope
from mitk_workbench_remote.properties import _serialize_property
from mitk_workbench_remote.transport import RestTransport

_log = logging.getLogger(__name__)


def _to_filter_str(value: Any) -> str:
    """Convert a serialized property value to a filter query-parameter string.

    Bool must be checked before int because bool is a subclass of int.
    Complex wire-format dicts (e.g. ColorProperty) are JSON-encoded.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


class DataStorage:
    """CRUD interface for DataNode objects.

    Obtained from a :class:`~mitk_workbench_remote.workbench.Workbench`
    instance via ``workbench.storage``. Do not construct directly.

    Args:
        transport: The REST transport used for all HTTP calls.
    """

    def __init__(self, transport: RestTransport) -> None:
        self._transport = transport

    # ------------------------------------------------------------------
    # List / Get
    # ------------------------------------------------------------------

    def list(
        self,
        *,
        data_type: str | None = None,
        toplevel: bool = False,
    ) -> list[DataNode]:
        """Return all nodes, optionally filtered.

        Args:
            data_type: If given, only return nodes whose MITK data type
                matches this string (e.g. ``"Image"``).
            toplevel: If ``True``, only return root-level nodes (no parent).

        Returns:
            List of :class:`~mitk_workbench_remote.node.DataNode` objects.
        """
        _log.debug(
            "[%s] list(data_type=%s, toplevel=%s)",
            self._transport.base_url,
            data_type,
            toplevel,
        )
        params: dict[str, str | int] = {"limit": 1000, "offset": 0}
        offset = 0
        if data_type is not None:
            params["data_type"] = data_type
        if toplevel:
            params["hierarchy"] = "toplevel"

        nodes: list[DataNode] = []
        while True:
            resp = self._transport.get("/datastorage/nodes", params=params)
            body = resp.json()
            nodes.extend(DataNode._from_node_dict(d, self._transport) for d in body["data"])
            total_count: int = int(body["meta"]["total_count"])
            offset += len(body["data"])
            params["offset"] = offset
            if offset >= total_count:
                break
        _log.debug("[%s] list -> %d node(s)", self._transport.base_url, len(nodes))
        return nodes

    def filter(
        self,
        *,
        data_type: str | None = None,
        toplevel: bool = False,
        parent_uid: str | None = None,
        properties: dict[str, Any] | None = None,
        scope: PropertyScope = PropertyScope.ALL,
        context: str | None = None,
    ) -> builtins.list[DataNode]:
        """Return nodes matching the given filters.

        Args:
            data_type: If given, only return nodes whose MITK data type
                matches this string (e.g. ``"Image"``).
            toplevel: If ``True``, only return root-level nodes (no parent).
            parent_uid: If given, only return direct children of this node.
            properties: If given, only return nodes whose properties match
                all entries. Values are serialized with the same rules as
                :meth:`~mitk_workbench_remote.node.DataNode.set_property`:
                simple types pass through, ``color`` is wrapped as a
                ``ColorProperty`` wire dict. Wildcard characters ``*`` and
                ``?`` are supported for string values.
            scope: Property scope for the property filters (``"all"``,
                ``"node"``, or ``"data"``). Only used when ``properties``
                is given.
            context: Renderer context for the property filters. Only used
                when ``properties`` is given.

        Returns:
            List of :class:`~mitk_workbench_remote.node.DataNode` objects.
        """
        params: dict[str, str | int] = {"limit": 1000, "offset": 0}
        offset = 0
        if data_type is not None:
            params["data_type"] = data_type
        if toplevel:
            params["hierarchy"] = "toplevel"
        if parent_uid is not None:
            params["parent_uid"] = parent_uid
        if properties is not None:
            for key, value in properties.items():
                params[f"filter.{key}"] = _to_filter_str(_serialize_property(key, value))
            params["property_scope"] = scope
            if context is not None:
                params["context"] = context

        nodes: list[DataNode] = []
        while True:
            resp = self._transport.get("/datastorage/nodes", params=params)
            body = resp.json()
            nodes.extend(DataNode._from_node_dict(d, self._transport) for d in body["data"])
            total_count: int = int(body["meta"]["total_count"])
            offset += len(body["data"])
            params["offset"] = offset
            if offset >= total_count:
                break
        return nodes

    def get(self, uid: str) -> DataNode:
        """Fetch a single node by UID.

        Args:
            uid: Node identifier.

        Returns:
            The matching :class:`~mitk_workbench_remote.node.DataNode`.

        Raises:
            NodeNotFoundError: If no node with this UID exists.
        """
        _log.debug("[%s] get(%s)", self._transport.base_url, uid)
        resp = self._transport.get(f"/datastorage/nodes/{uid}")
        return DataNode._from_node_dict(resp.json()["data"], self._transport)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        *,
        parent: DataNode | str | None = None,
    ) -> DataNode:
        """Create a new empty node in the DataStorage.

        Args:
            name: Display name for the new node.
            parent: Parent node (a :class:`~mitk_workbench_remote.node.DataNode`
                instance or a UID string). When ``None`` the node is created at
                the top level.

        Returns:
            The newly created :class:`~mitk_workbench_remote.node.DataNode`.
        """
        _log.debug("[%s] create('%s', parent=%s)", self._transport.base_url, name, parent)
        body = {"name": name}
        if parent is None:
            resp = self._transport.post("/datastorage/nodes", json=body)
        else:
            parent_uid = parent.uid if isinstance(parent, DataNode) else str(parent)
            resp = self._transport.post(f"/datastorage/nodes/{parent_uid}/children", json=body)
        return DataNode._from_node_dict(resp.json()["data"], self._transport)

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        resp = self._transport.get("/datastorage/nodes", params={"limit": 1})
        return int(resp.json()["meta"]["total_count"])

    def __iter__(self) -> Iterator[DataNode]:
        return iter(self.list())

    def __contains__(self, uid: object) -> bool:
        if isinstance(uid, DataNode):
            lookup = uid.uid
        elif isinstance(uid, str):
            lookup = uid
        else:
            return False
        try:
            self.get(lookup)
            return True
        except errors.NodeNotFoundError:
            return False

    def __getitem__(self, uid: str) -> DataNode:
        return self.get(uid)

    # ------------------------------------------------------------------
    # Repr (no REST calls except _repr_html_)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DataStorage(url={self._transport.base_url!r})"

    def _repr_html_(self) -> str:
        e = _html.escape
        nodes = self.list()
        header = "<tr><th>uid</th><th>name</th><th>type</th><th>path</th></tr>"
        rows = "".join(
            f"<tr>"
            f"<td>{e(n.uid)}</td>"
            f"<td>{e(n._name)}</td>"  # display cache - avoids REST call
            f"<td>{e(n.data_type or '')}</td>"
            f"<td>{e(n.path)}</td>"
            f"</tr>"
            for n in nodes
        )
        return f"<table>{header}{rows}</table>"
