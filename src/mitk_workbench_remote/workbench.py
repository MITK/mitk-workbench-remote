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

"""Workbench handle — primary entry point for controlling a MITK Workbench instance."""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from mitk_workbench_remote import errors
from mitk_workbench_remote.node import DataNode
from mitk_workbench_remote.storage import DataStorage
from mitk_workbench_remote.transport import RestTransport, TransferMode

if TYPE_CHECKING:
    from mitk_workbench_remote.editors import (
        EditorAlias,
        EditorBase,
        EditorDescriptor,
        MxNEditor,
        StdMultiEditor,
    )

_log = logging.getLogger(__name__)


def _xyz(seq: Any) -> tuple[float, float, float]:
    """Convert a 3-element JSON array to a typed ``(x, y, z)`` tuple."""
    x, y, z = seq
    return (float(x), float(y), float(z))


def _position_bounds_from_json(body: dict[str, Any]) -> PositionBounds:
    """Parse a ``{"min_position": ..., "max_position": ...}`` bounds object.

    Either field may be ``null`` when no geometry is loaded; both are
    optional in that case.
    """
    raw_min = body.get("min_position")
    raw_max = body.get("max_position")
    return PositionBounds(
        min_position=_xyz(raw_min) if raw_min is not None else None,
        max_position=_xyz(raw_max) if raw_max is not None else None,
    )


def _find_listening_pid(port: int) -> int | None:
    """Return the PID of the process listening on *port* (Windows only).

    Parses ``netstat -ano -p TCP`` output.  Returns ``None`` when no
    matching listener is found or if the command fails.
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    for line in result.stdout.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.strip().split()
        # Format: Proto  LocalAddr  ForeignAddr  State  PID
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        if local_addr.endswith(f":{port}"):
            try:
                return int(parts[-1])
            except (ValueError, IndexError):
                continue
    return None


@dataclass(frozen=True)
class WorkbenchInfo:
    """Metadata about a MITK Workbench instance.

    Attributes:
        name: Human-readable server name.
        api_version: REST API version string (e.g. ``"v1"``).
        mitk_version: Underlying MITK framework version string.
        url: Base URL of the Workbench REST server.
    """

    name: str
    api_version: str
    mitk_version: str
    url: str


@dataclass(frozen=True)
class PositionBounds:
    """World-space axis-aligned bounding box of the scene.

    Attributes:
        min_position: Minimum corner ``[x, y, z]``, or ``None`` if no geometry
            is loaded.
        max_position: Maximum corner ``[x, y, z]``, or ``None`` if no geometry
            is loaded.
    """

    min_position: tuple[float, float, float] | None
    max_position: tuple[float, float, float] | None


@dataclass(frozen=True)
class SelectedPosition:
    """Current crosshair position and scene bounds.

    Attributes:
        position: Crosshair position in world coordinates ``[x, y, z]``.
        bounds: Scene bounding box.
    """

    position: tuple[float, float, float]
    bounds: PositionBounds


@dataclass(frozen=True)
class TimeBounds:
    """Time geometry bounds.

    Attributes:
        min_timepoint_ms: Start of the time range in milliseconds.
        max_timepoint_ms: End of the time range in milliseconds.
        steps: Total number of time steps.
    """

    min_timepoint_ms: float
    max_timepoint_ms: float
    steps: int


@dataclass(frozen=True)
class SelectedTime:
    """Current time navigation state.

    Attributes:
        timepoint_ms: Selected time point in milliseconds.
        timestep: Selected time step index (zero-based).
        bounds: Time geometry bounds.
    """

    timepoint_ms: float
    timestep: int
    bounds: TimeBounds


class ReinitMode(str, Enum):
    """Controls automatic reinit behavior after show().

    Attributes:
        ALL_VISIBLE: Global reinit -- fit all views to all visible data.
        NODE: Node reinit -- fit views to the newly shown node only.
        NONE: Skip reinit entirely.
    """

    ALL_VISIBLE = "all"
    NODE = "node"
    NONE = "none"


class RenderWindows(str, Enum):
    """Which render windows to update.

    Attributes:
        ALL: Update all render windows.
        TWO_D: Update only 2-D render windows.
        THREE_D: Update only the 3-D render window.
    """

    ALL = "all"
    TWO_D = "2d"
    THREE_D = "3d"

    def __str__(self) -> str:
        return self.value


class ScreenshotFormat(str, Enum):
    """Image encoding format for screenshots.

    Attributes:
        PNG: Lossless PNG encoding.
        JPEG: Lossy JPEG encoding.
    """

    PNG = "png"
    JPEG = "jpeg"

    def __str__(self) -> str:
        return self.value


def _infer_name(data: Any) -> str:
    """Infer a default node name from the data object.

    Rules:
        1. ``str`` / ``Path`` -- file stem (e.g. ``"ct_scan.nrrd"`` -> ``"ct_scan"``)
        2. ``Image`` -- ``"Image"``
        3. ``MultiLabelSegmentation`` -- ``"Segmentation"``
        4. ``numpy.ndarray`` -- ``"Array"``
        5. Anything else -- ``type(data).__name__``
    """
    if isinstance(data, (str, Path)):
        return Path(data).stem

    from mitk_workbench_remote.image import Image as _Image

    if isinstance(data, _Image):
        return "Image"

    try:
        from mitk_workbench_remote.multilabel import MultiLabelSegmentation as _MLS

        if isinstance(data, _MLS):
            return "Segmentation"
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            return "Array"
    except ImportError:
        pass

    return type(data).__name__


class Workbench:
    """Handle to a running MITK Workbench instance.

    Do not instantiate directly — use :func:`connect` or
    :func:`~mitk_workbench_remote.discovery.launch`.

    Args:
        transport: Configured REST transport for the target instance.
        process: Optional subprocess handle when the instance was started
            via :func:`~mitk_workbench_remote.discovery.launch`.
    """

    def __init__(
        self,
        transport: RestTransport,
        *,
        process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        self._transport = transport
        self._process = process
        self._info: WorkbenchInfo | None = None
        self._storage: DataStorage | None = None
        self._std_multi: StdMultiEditor | None = None
        self._mxn: MxNEditor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Base URL of the Workbench REST server."""
        return self._transport.base_url

    @property
    def is_launched_remotely(self) -> bool:
        """Returns if the workbench was launched remotely (true) or was just connected to but is
        independent (false)."""
        return self._process is not None

    @property
    def info(self) -> WorkbenchInfo:
        """Metadata about the connected Workbench instance.

        Fetched lazily from ``GET /api/v1/info`` on first access and cached thereafter.
        """
        if self._info is None:
            si = self._transport.server_info
            self._info = WorkbenchInfo(
                name=si.name,
                api_version=si.api_version,
                mitk_version=si.mitk_version,
                url=self.url,
            )
        return self._info

    @property
    def storage(self) -> DataStorage:
        """The DataStorage for this Workbench instance. Lazy and cached."""
        if self._storage is None:
            self._storage = DataStorage(self._transport)
        return self._storage

    # ------------------------------------------------------------------
    # Editors
    # ------------------------------------------------------------------

    @property
    def std_multi(self) -> StdMultiEditor:
        """Handle to the StdMultiWidget editor. Lazy and cached.

        The handle holds only the transport — every state read goes to
        the network. Use :attr:`is_active` on the returned handle (or
        :meth:`editors`) to check whether the editor is currently open.
        """
        if self._std_multi is None:
            from mitk_workbench_remote.editors import StdMultiEditor

            self._std_multi = StdMultiEditor(self._transport)
        return self._std_multi

    @property
    def mxn(self) -> MxNEditor:
        """Handle to the MxN multi-widget editor. Lazy and cached."""
        if self._mxn is None:
            from mitk_workbench_remote.editors import MxNEditor

            self._mxn = MxNEditor(self._transport)
        return self._mxn

    def editor(self, alias: str | EditorAlias) -> EditorBase:
        """Return an editor handle by alias.

        Accepts either a raw alias string (``"stdmulti"``, ``"mxn"``) or
        an :class:`~mitk_workbench_remote.editors.EditorAlias` member.
        For the well-known aliases this returns the same instance as the
        named property (:attr:`std_multi` / :attr:`mxn`). Unknown aliases
        raise :class:`ValueError`.
        """
        from mitk_workbench_remote.editors import EditorAlias

        if alias == EditorAlias.STD_MULTI:
            return self.std_multi
        if alias == EditorAlias.MXN:
            return self.mxn
        raise ValueError(f"Unknown editor alias: {alias!r}")

    def editors(self) -> list[EditorDescriptor]:
        """List all known editors with their current activity state.

        Sends ``GET /rendering/editors``. Always live — the active state
        of an editor changes at runtime as the user opens or closes it.
        """
        from mitk_workbench_remote.editors import EditorDescriptor

        body = self._transport.get("/rendering/editors").json()
        return [
            EditorDescriptor(
                alias=item["alias"],
                plugin_id=item["plugin_id"],
                active=item["active"],
            )
            for item in body
        ]

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check whether the Workbench is reachable.

        Sends a ``GET /health`` request. Returns ``False`` for any failure —
        network errors, HTTP errors, or unexpected exceptions — without raising.

        Returns:
            ``True`` if the server responded with a 2xx status, ``False`` otherwise.
        """
        try:
            response = self._transport.get("/health")
            result = response.ok
        except Exception:
            result = False
        _log.debug("[%s] ping -> %s", self.url, result)
        return result

    @property
    def is_connected(self) -> bool:
        """Whether the Workbench is currently reachable. Alias for :meth:`ping`."""
        return self.ping()

    # ------------------------------------------------------------------
    # show()
    # ------------------------------------------------------------------

    def show(
        self,
        data: Any,
        *,
        name: str | None = None,
        visible: bool = True,
        opacity: float | None = None,
        color: tuple[float, float, float] | None = None,
        parent: DataNode | str | None = None,
        hide_others: bool = False,
        reinit: ReinitMode = ReinitMode.ALL_VISIBLE,
    ) -> DataNode:
        """One-liner to load data into the Workbench and display it.

        Creates a new node, uploads the data, sets display properties, and
        optionally reinitializes the render views.

        Args:
            data: Data to show. Accepts file paths (``str`` / ``Path``),
                :class:`~mitk_workbench_remote.image.Image`,
                :class:`~mitk_workbench_remote.multilabel.MultiLabelSegmentation`,
                numpy arrays, or any type handled by the converter registry.
            name: Node display name. Inferred from ``data`` when ``None``.
            visible: Whether the node is visible after showing.
            opacity: Node opacity (0.0 - 1.0). ``None`` keeps the server default.
            color: Node color as ``(r, g, b)``. ``None`` keeps the server default.
            parent: Parent node (instance or UID string). ``None`` for top-level.
            hide_others: When ``True``, hide all existing nodes before showing.
            reinit: Auto-reinit mode after showing the data.

        Returns:
            The newly created :class:`~mitk_workbench_remote.node.DataNode`.
        """
        if name is None:
            name = _infer_name(data)

        _log.info("[%s] Showing '%s'", self.url, name)

        if hide_others:
            for existing_node in self.storage.list():
                existing_node.update_properties(visible=False)

        node = self.storage.create(name, parent=parent)

        try:
            if isinstance(data, (str, Path)):
                self._upload_file(node, Path(data))
            else:
                node.set_data(data)
        except Exception:
            with contextlib.suppress(Exception):
                node.remove()
            raise

        props: dict[str, Any] = {"visible": visible}
        if opacity is not None:
            props["opacity"] = opacity
        if color is not None:
            props["color"] = color
        node.update_properties(**props)

        if reinit == ReinitMode.ALL_VISIBLE:
            self.reinit()
        elif reinit == ReinitMode.NODE:
            self.reinit([node])

        return node

    def _upload_file(self, node: DataNode, path: Path) -> None:
        """Upload a file to an existing node without deserializing.

        The server parses the file format. After upload the node is refreshed
        to pick up the server-assigned ``data_type``.
        """
        endpoint = f"/datastorage/nodes/{node.uid}/data"
        mode = self._transport.transfer_mode

        if mode == TransferMode.FILE_REFERENCE:
            self._transport.put_file_reference(endpoint, file_path=str(path.resolve()))
        elif mode == TransferMode.DIRECT:
            self._transport.put_binary(
                endpoint,
                data=path.read_bytes(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": f'attachment; filename="{path.name}"',
                },
            )
        else:
            raise RuntimeError(f"Unknown transfer mode: {mode}")

        node.refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(self, *, windows: RenderWindows | str = RenderWindows.ALL) -> None:
        """Request all render windows to redraw.

        Use after batching data or property changes to make them visible
        without per-change flicker.

        Args:
            windows: Which render windows to update: ``"all"`` (default),
                ``"2d"``, or ``"3d"``.

        Raises:
            RenderingError: If the rendering framework reports a failure.
        """
        self._transport.post("/rendering/update", json={"type": windows})

    def reinit(self, nodes: list[DataNode | str] | None = None) -> None:
        """Fit render window cameras to the bounding geometry of nodes.

        Three modes:

        - ``nodes=None`` — global reinit: fits all views to all visible data
          (equivalent to clicking the global reinit button in the Workbench).
        - single element list — fits views to that node's geometry.
        - multi-element list — fits views to the combined bounding geometry.

        Args:
            nodes: Nodes to reinit to. Accepts :class:`DataNode` instances or
                UID strings. ``None`` performs a global reinit.

        Raises:
            ValueError: If ``nodes`` is an empty list (use ``None`` for global reinit).
            NodeNotFoundError: If any listed node does not exist.
            ApiError: If a node has no data (code ``NO_DATA``) or no usable
                geometry (code ``NO_GEOMETRY``).
            RenderingError: If the rendering framework reports a failure.
        """
        if nodes is not None and len(nodes) == 0:
            raise ValueError("nodes must be None (global reinit) or a non-empty list")
        if nodes is None:
            _log.info("[%s] Reinit: all visible", self.url)
            body: dict[str, object] = {}
        else:
            uids = [n.uid if isinstance(n, DataNode) else str(n) for n in nodes]
            _log.info("[%s] Reinit: %d node(s)", self.url, len(uids))
            body = {"uids": uids}
        self._transport.post("/rendering/reinit", json=body)

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def get_position(self) -> SelectedPosition:
        """Get the current crosshair position and scene bounds.

        Returns:
            A :class:`SelectedPosition` with the crosshair coordinates and
            the world-space bounding box.

        Raises:
            RenderingError: If no render window is available.
        """
        _log.debug("[%s] get_position", self.url)
        body = self._transport.get("/rendering/selected-position").json()
        return SelectedPosition(
            position=_xyz(body["position"]),
            bounds=_position_bounds_from_json(body["bounds"]),
        )

    def set_position(self, position: tuple[float, float, float] | list[float]) -> None:
        """Move the crosshair to the given world position.

        Args:
            position: Target position ``[x, y, z]`` in world coordinates.

        Raises:
            ValueError: If position does not have exactly 3 elements.
            RenderingError: If no render window is available.
        """
        if len(position) != 3:
            raise ValueError(
                f"position must have exactly 3 elements [x, y, z], got {len(position)}"
            )
        _log.debug("[%s] set_position %s", self.url, list(position))
        self._transport.put("/rendering/selected-position", json={"position": list(position)})

    # ------------------------------------------------------------------
    # Time navigation
    # ------------------------------------------------------------------

    def get_time(self) -> SelectedTime:
        """Get the current time navigation state.

        Returns:
            A :class:`SelectedTime` with the active time point, step, and bounds.

        Raises:
            RenderingError: If no render window is available.
        """
        body = self._transport.get("/rendering/selected-time").json()
        b = body["bounds"]
        return SelectedTime(
            timepoint_ms=body["timepoint_ms"],
            timestep=body["timestep"],
            bounds=TimeBounds(
                min_timepoint_ms=b["min_timepoint_ms"],
                max_timepoint_ms=b["max_timepoint_ms"],
                steps=b["steps"],
            ),
        )

    def set_timepoint(self, timepoint_ms: float) -> None:
        """Set the active time point.

        Args:
            timepoint_ms: Target time point in milliseconds.

        Raises:
            RenderingError: If no render window is available.
        """
        self._transport.put("/rendering/selected-time", json={"timepoint_ms": timepoint_ms})

    def set_timestep(self, timestep: int) -> None:
        """Set the active time step.

        Args:
            timestep: Target time step index (zero-based).

        Raises:
            RenderingError: If no render window is available.
        """
        self._transport.put("/rendering/selected-time", json={"timestep": timestep})

    def get_timepoint(self) -> float:
        """Convenience: return the current time point in milliseconds."""
        return self.get_time().timepoint_ms

    def get_timestep(self) -> int:
        """Convenience: return the current time step index."""
        return self.get_time().timestep

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def screenshot(
        self,
        *,
        fmt: ScreenshotFormat | str = ScreenshotFormat.PNG,
        width: int | None = None,
        height: int | None = None,
        path: str | Path | None = None,
    ) -> bytes:
        """Capture a screenshot of the active application window.

        Args:
            fmt: Image encoding format (``"png"`` or ``"jpeg"``).
            width: Output width in pixels. Must be given together with ``height``.
            height: Output height in pixels. Must be given together with ``width``.
            path: If given, save the screenshot to this file path.

        Returns:
            Raw image bytes.

        Raises:
            ValueError: If only one of ``width`` / ``height`` is given.
            RenderingError: If no render window is available.
        """
        if (width is None) != (height is None):
            raise ValueError("width and height must both be given or both omitted")

        params: dict[str, str | int] = {"format": str(fmt)}
        if width is not None:
            params["width"] = width
        if height is not None:
            params["height"] = height

        resp = self._transport.get_binary("/rendering/screenshot", params=params)
        data = resp.content

        if path is not None:
            Path(path).write_bytes(data)

        return data

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Terminate the subprocess started by :func:`~mitk_workbench_remote.discovery.launch`.

        On Windows the MITK executable is often a batch-script wrapper, so
        the :class:`~subprocess.Popen` handle points to ``cmd.exe`` — not the
        real ``MitkWorkbench.exe``.  Killing the wrapper (or its tree) may
        leave the actual Workbench running.  We therefore look up the process
        that owns the REST port via ``netstat`` and kill *that*.

        On Unix, sends ``SIGTERM`` to the subprocess and falls back to
        ``SIGKILL`` after 10 seconds.

        Also closes the underlying transport session.

        Raises:
            MitkError: If this instance was not created by ``launch()``. Use property
            is_launched_remotely to check this.
        """
        if self._process is None:
            raise errors.MitkError("shutdown() is only valid for instances started with launch()")

        _log.info("[%s] Shutting down", self.url)

        if sys.platform == "win32":
            # Find the process actually listening on the REST port — this may
            # differ from self._process.pid when MITK is launched via a
            # batch-script wrapper (cmd.exe -> MitkWorkbench.exe).
            port = urlparse(self.url).port
            listener_pid: int | None = None
            if port is not None:
                listener_pid = _find_listening_pid(port)
                if listener_pid is not None:
                    result = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(listener_pid)],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        warnings.warn(
                            f"taskkill /PID {listener_pid} failed (code {result.returncode}): "
                            f"{result.stderr.strip()}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            # Also kill the wrapper process tree, unless it IS the listener
            # (no wrapper — MITK was started directly).
            if listener_pid != self._process.pid:
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    warnings.warn(
                        f"taskkill /PID {self._process.pid} failed (code {result.returncode}): "
                        f"{result.stderr.strip()}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        else:
            self._process.terminate()

        # Use communicate() rather than wait() to drain the stderr pipe that
        # launch() opens (stderr=PIPE).  wait() with an unread PIPE can
        # deadlock when the subprocess fills the pipe buffer during shutdown.
        try:
            self._process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.communicate()

        self._transport.close()

    def close(self) -> None:
        """Close the underlying transport session."""
        self._transport.close()

    def __enter__(self) -> Workbench:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Workbench(url={self.url!r})"


def connect(
    url: str = "http://localhost:8080",
    *,
    token: str | None = None,
    timeout: float = 30.0,
    transfer_mode: str | None = None,
) -> Workbench:
    """Connect to a running MITK Workbench REST server.

    No HTTP call is made during construction — the connection is established
    lazily on first use.

    Args:
        url: Base URL of the Workbench REST server.
        token: Optional API token (``X-MITK-API-Token`` header).
        timeout: Request timeout in seconds.
        transfer_mode: Override transfer mode (``"direct"`` or
            ``"file-reference"``). When ``None`` the mode is auto-detected
            from server capabilities on first data request.

    Returns:
        A :class:`Workbench` handle ready for use.
    """
    _log.info("[%s] Connecting", url)
    transport = RestTransport(url, token=token, timeout=timeout, transfer_mode=transfer_mode)
    return Workbench(transport)
