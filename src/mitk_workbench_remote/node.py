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

from __future__ import annotations

import html as _html
import json
import logging
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, cast

from mitk_workbench_remote import _io
from mitk_workbench_remote.errors import TransferError, UnsupportedDataTypeError
from mitk_workbench_remote.properties import _deserialize_property, _serialize_property
from mitk_workbench_remote.transport import RestTransport, TransferMode

_log = logging.getLogger(__name__)


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


class DataRepresentation(str, Enum):
    """Return type selection for :meth:`DataNode.get_data`.

    Attributes:
        AUTO: Return ``mitk.Image`` if the ``mitk`` package is importable,
            otherwise fall back to :class:`~mitk_workbench_remote.image.Image`.
        REMOTE: Always return the remote Python type
            (:class:`~mitk_workbench_remote.image.Image` or
            :class:`~mitk_workbench_remote.multilabel.MultiLabelSegmentation`).
        MITK: Always return ``mitk.Image``; raises :exc:`ImportError` if the
            ``mitk`` package is absent.
    """

    AUTO = "auto"
    REMOTE = "remote"
    MITK = "mitk"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_remote_properties_to_mitk(mitk_img: Any, props: dict[str, Any]) -> None:
    """Apply a plain-Python property dict onto a ``mitk.Image``.

    Scalar types (``bool``, ``int``, ``float``, ``str``, ``(r, g, b)`` tuple)
    are passed directly to ``set_property()`` and auto-wrapped by the binding.
    Dict-form values (``{"type": "...", "value": ...}``) from the REST API are
    reconstructed via ``mitk.BaseProperty.from_json()``, which handles every
    property type that has a self-contained JSON representation.

    Args:
        mitk_img: A ``mitk.Image`` instance.
        props: Property dict as returned by
            :meth:`DataNode.get_properties`.
    """
    import mitk

    for key, value in props.items():
        if isinstance(value, dict) and "type" in value:
            mitk_img.set_property(key, mitk.BaseProperty.from_json(json.dumps(value)))
        else:
            mitk_img.set_property(key, value)


# ---------------------------------------------------------------------------
# DataNode
# ---------------------------------------------------------------------------


class DataNode:
    """A single node in the MITK DataStorage.

    Wrapper that represents one node with property shortcuts (name, visible,
    opacity, color), data access (get_data, set_data), child management,
    and batch property updates.

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

    def _validate_writable_scope(self, scope: PropertyScope) -> None:
        """Raise ValueError if scope is ALL (not valid for write operations)."""
        if scope == PropertyScope.ALL:
            raise ValueError(
                "PropertyScope.ALL is not valid for write operations; use PropertyScope.NODE or"
                " PropertyScope.DATA."
            )

    # ------------------------------------------------------------------
    # Batch property update
    # ------------------------------------------------------------------

    def update_properties(
        self,
        *,
        scope: PropertyScope = PropertyScope.NODE,
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
        self._validate_writable_scope(scope)
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
        _log.debug("[%s] get_property('%s', key='%s')", self._transport.base_url, self._name, key)
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
        self._validate_writable_scope(scope)
        _log.debug("[%s] set_property('%s', key='%s')", self._transport.base_url, self._name, key)
        body = _serialize_property(key, value)
        params: dict[str, str] = {"property_scope": scope}
        if context is not None:
            params["context"] = context
        self._transport.put(
            f"/datastorage/nodes/{self._uid}/properties/{key}",
            json=body,
            params=params,
        )

    def remove_property(self, key: str, *, scope: PropertyScope = PropertyScope.NODE) -> None:
        """Remove a property by key.

        Args:
            key: Property name to remove.
            scope: Property scope (:attr:`~PropertyScope.NODE` or
                :attr:`~PropertyScope.DATA`).
        """
        self._validate_writable_scope(scope)
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
        params: dict[str, str | int] = {"limit": 1000, "offset": 0}
        nodes: list[DataNode] = []
        while True:
            resp = self._transport.get(f"/datastorage/nodes/{self._uid}/children", params=params)
            body = resp.json()
            nodes.extend(DataNode._from_node_dict(d, self._transport) for d in body["data"])
            total_count: int = int(body["meta"]["total_count"])
            params["offset"] = int(params["offset"]) + len(body["data"])
            if params["offset"] >= total_count:
                break
        return nodes

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
        """Refresh cached properties from the server.

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
    # Data transfer
    # ------------------------------------------------------------------

    # Data types that can be reconstructed as in-memory Python objects.
    _SUPPORTED_DATA_TYPES: frozenset[str] = frozenset(
        {
            "Image",
            "MultiLabelSegmentation",
        }
    )

    def _check_data_type_supported(self) -> str:
        """Validate that the node's data type can be represented in Python.

        Returns:
            The validated data type string.

        Raises:
            UnsupportedDataTypeError: If the data type is not in
                :attr:`_SUPPORTED_DATA_TYPES`.
            UnsupportedDataTypeError: If the data type is ``None``
                (node has no data assigned).
        """
        dt = self._data_type
        if dt is None:
            raise UnsupportedDataTypeError("None (no data)")
        if dt not in self._SUPPORTED_DATA_TYPES:
            raise UnsupportedDataTypeError(dt)
        return dt

    def get_data(
        self,
        *,
        include_properties: bool = False,
        as_type: DataRepresentation = DataRepresentation.AUTO,
    ) -> Any:
        """Download this node's data and return a typed Python object.

        The returned type depends on the node's :attr:`data_type` and the
        ``as_type`` parameter:

        * ``"Image"`` with ``as_type=AUTO`` -- returns ``mitk.Image`` if the
          ``mitk`` package is importable, otherwise
          :class:`~mitk_workbench_remote.image.Image`.
        * ``"Image"`` with ``as_type=REMOTE`` -- always returns
          :class:`~mitk_workbench_remote.image.Image`.
        * ``"Image"`` with ``as_type=MITK`` -- always returns ``mitk.Image``;
          raises :exc:`ImportError` if the ``mitk`` package is absent.
        * ``"MultiLabelSegmentation"`` -- always returns
          :class:`~mitk_workbench_remote.multilabel.MultiLabelSegmentation`;
          ``as_type=MITK`` raises :exc:`NotImplementedError` (see WP-11).

        The transfer mode (direct or file-reference) is negotiated automatically
        based on the transport's :attr:`~RestTransport.transfer_mode`.

        Args:
            include_properties: If ``True``, also fetch data-scope properties
                and store them in the returned object's properties.  For
                ``mitk.Image`` results, scalar properties (str/bool/int/float/
                3-tuple) are applied via ``mitk.Image.set_property``; exotic
                types that cannot be auto-wrapped are logged at WARNING and
                skipped.
            as_type: Controls the Python type of the returned image object.
                Defaults to :attr:`DataRepresentation.AUTO`.

        Returns:
            A typed Python object matching the node's data type.

        Raises:
            UnsupportedDataTypeError: If the data type cannot be represented
                in Python. Use :meth:`save_data` to download raw bytes instead.
            ImportError: If ``as_type=MITK`` and the ``mitk`` package is
                not installed.
            NotImplementedError: If ``as_type=MITK`` and the data type is
                ``MultiLabelSegmentation`` (requires WP-11).
        """
        dt = self._check_data_type_supported()
        _log.info(
            "[%s] Downloading data for '%s' (%s, %s)",
            self._transport.base_url,
            self._name,
            dt,
            self._transport.transfer_mode,
        )
        nrrd_source = self._download_raw_source()

        if dt == "MultiLabelSegmentation":
            return self._get_data_multilabel(
                nrrd_source,
                include_properties=include_properties,
                as_type=as_type,
            )
        if dt == "Image":
            return self._get_data_image(
                nrrd_source,
                include_properties=include_properties,
                as_type=as_type,
            )
        # Unreachable after _check_data_type_supported, but explicit.
        raise UnsupportedDataTypeError(dt)

    def _get_data_image(
        self,
        nrrd_source: bytes | str,
        *,
        include_properties: bool,
        as_type: DataRepresentation,
    ) -> Any:
        """Deserialize NRRD data into an Image."""
        image = _io.read_nrrd(nrrd_source)
        if include_properties:
            image._properties = self.get_properties(scope=PropertyScope.DATA)

        if as_type == DataRepresentation.REMOTE:
            return image

        if as_type == DataRepresentation.MITK:
            mitk_img = image.to_mitk()  # raises ImportError if mitk absent
            if include_properties:
                _apply_remote_properties_to_mitk(mitk_img, image._properties)
            return mitk_img

        # AUTO: try mitk, fall back to remote image on ImportError
        try:
            mitk_img = image.to_mitk()
        except ImportError:
            return image
        if include_properties:
            _apply_remote_properties_to_mitk(mitk_img, image._properties)
        return mitk_img

    def _get_data_multilabel(
        self,
        nrrd_source: bytes | str,
        *,
        include_properties: bool,
        as_type: DataRepresentation,
    ) -> Any:
        """Deserialize NRRD data into a MultiLabelSegmentation."""
        if as_type == DataRepresentation.MITK:
            raise NotImplementedError(
                "as_type=MITK for MultiLabelSegmentation requires WP-11 "
                "(mitk.MultiLabelSegmentation binding + MitkSegmentationConverter). "
                "See MITK_Interoperability_Design.md §WP-11."
            )
        # TODO(WP-11): when mitk.MultiLabelSegmentation is available, AUTO should
        # resolve to it here (same pattern as _get_data_image).
        seg = _io.read_multilabel_nrrd(nrrd_source)
        if include_properties:
            seg._properties = self.get_properties(scope=PropertyScope.DATA)
        return seg

    def _download_raw_source(self) -> bytes | str:
        """Download raw data and return bytes or a file path.

        For direct transfer mode the raw bytes are returned.
        For file-reference mode the server-provided file path is returned.
        """
        resp = self._transport.get_binary(f"/datastorage/nodes/{self._uid}/data")
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            body = resp.json()
            transfer = body.get("transfer")
            if not isinstance(transfer, dict) or "file_path" not in transfer:
                raise TransferError(
                    "Malformed file-reference response from server: expected"
                    f" 'transfer.file_path' in JSON body, got: {body!r}"
                )
            return cast(str, transfer["file_path"])
        return resp.content

    def save_data(self, path: str | Path) -> Path:
        """Download this node's data and save the raw bytes to a file.

        This works for any data type, including types that cannot be
        represented in Python (e.g. Surface, PointSet). The file is saved
        in NRRD format as delivered by the server.

        Args:
            path: Destination file path. Parent directory must exist.

        Returns:
            The resolved Path that was written.
        """
        raw_source = self._download_raw_source()
        dest = Path(path)
        if isinstance(raw_source, str):
            # file-reference mode: server provided a local path, copy it.
            # Security note: the path is fully trusted — only use file-reference
            # mode with a trusted server, as a compromised server could supply
            # arbitrary local paths.
            dest.write_bytes(Path(raw_source).read_bytes())
        else:
            dest.write_bytes(raw_source)
        return dest

    def set_data(self, data: Any, *, include_properties: bool = False) -> None:
        """Upload data to this node.

        Accepts an :class:`~mitk_workbench_remote.image.Image`, a numpy
        ndarray, or any type handled by the converter registry (e.g.
        SimpleITK.Image, mlarray.MLArray). The transfer mode is chosen
        automatically.

        After a successful upload, :attr:`data_type` is updated locally to
        reflect the uploaded type without requiring an explicit
        :meth:`refresh` call.

        Args:
            data: Pixel data or a convertible object.
            include_properties: If ``True``, also upload the data's properties
                as data-scope properties.

        Raises:
            UnsupportedDataTypeError: If the node's data type does not match
                the data being uploaded.
            TypeError: If no converter is registered for ``data``'s type.
        """
        nrrd_bytes = self._resolve_serialized_bytes(data)
        mode = self._transport.transfer_mode
        _log.info(
            "[%s] Uploading data for '%s' (%d bytes, %s)",
            self._transport.base_url,
            self._name,
            len(nrrd_bytes),
            mode,
        )

        endpoint = f"/datastorage/nodes/{self._uid}/data"

        if mode == TransferMode.DIRECT:
            self._transport.put_binary(
                endpoint,
                data=nrrd_bytes,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": 'attachment; filename="data.nrrd"',
                },
            )

        elif mode == TransferMode.FILE_REFERENCE:
            tmp_dir = self._resolve_temp_dir()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".nrrd", dir=tmp_dir
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
            try:
                tmp_path.write_bytes(nrrd_bytes)
                self._transport.put_file_reference(endpoint, file_path=str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                "Cannot set data. Transfer mode requested by MITK via"
                f" transport layer is unknown. Unknown mode: {mode}"
            )

        # Update the cached data_type to reflect the uploaded data so that
        # get_data() works immediately after set_data() without requiring
        # an explicit refresh() call.
        # For known Python types (Image, MultiLabelSegmentation) we can set the
        # type locally. For all other types (numpy arrays, third-party converter
        # types) the server determines the resulting type, so refresh from there.
        from mitk_workbench_remote.image import Image as _Image
        from mitk_workbench_remote.multilabel import MultiLabelSegmentation as _MLS

        if isinstance(data, _MLS):
            self._data_type = "MultiLabelSegmentation"
        elif isinstance(data, _Image):
            self._data_type = "Image"
        else:
            self.refresh()

        if include_properties:
            from mitk_workbench_remote.converters import find_image_converter
            from mitk_workbench_remote.image import Image
            from mitk_workbench_remote.multilabel import MultiLabelSegmentation

            properties: dict[str, Any] = {}
            if isinstance(data, (MultiLabelSegmentation, Image)):
                properties = data.properties
            else:
                converter = find_image_converter(data)
                if converter is not None:
                    properties = converter.extract_metadata(data)
            if properties:
                self.update_properties(scope=PropertyScope.DATA, **properties)

    def _resolve_serialized_bytes(self, data: Any) -> bytes:
        """Convert data to serialized bytes for upload.

        Raises:
            TypeError: If no converter is registered for the data's type.
        """
        from mitk_workbench_remote import _io
        from mitk_workbench_remote.converters import find_image_converter
        from mitk_workbench_remote.image import Image
        from mitk_workbench_remote.multilabel import MultiLabelSegmentation

        if isinstance(data, MultiLabelSegmentation):
            return _io.write_multilabel_nrrd(data)
        if isinstance(data, Image):
            return _io.write_nrrd(data)
        image_converter = find_image_converter(data)
        if image_converter is not None:
            return image_converter.to_nrrd_bytes(data)
        raise TypeError(f"No converter found for {type(data).__name__}")

    def _resolve_temp_dir(self) -> str:
        """Choose a temp directory for file-reference uploads.

        If file-access restrictions are active, uses the first allowed path.
        Otherwise uses the system temp directory.

        Raises:
            TransferError: If the server-advertised allowed path does not exist
                on the local filesystem.
        """
        config = self._transport.file_access_config
        if config.restrictions_active and config.allowed_paths:
            p = Path(config.allowed_paths[0])
            if not p.exists():
                raise TransferError(
                    f"File-access allowed path does not exist on this filesystem: {p}"
                )
            return str(p)
        return tempfile.gettempdir()

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
        return f"<DataNode {self._name!r} ({self._data_type}) @ {self._path} uid={self._uid}>"

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
