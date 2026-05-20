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

"""Editor and render-window primitives shared by all editor types.

Public types:
    WindowKind, ViewDirection, StandardView, EditorAlias — vocabulary.
    EditorDescriptor, EditorInfo                          — discovery payloads.
    WindowSummary, MxNWindowSummary                       — per-window summaries.
    Camera                                                — camera GET + PUT body.
    SliceBounds, SelectedSlice                            — per-window slice state.
    EditorBase, RenderWindow                              — base classes for editors.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from mitk_workbench_remote.errors import EditorNotActiveError
from mitk_workbench_remote.node import PropertyScope
from mitk_workbench_remote.transport import RestTransport
from mitk_workbench_remote.workbench import ScreenshotFormat

if TYPE_CHECKING:
    from mitk_workbench_remote.node import DataNode

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WindowKind(str, Enum):
    """Render-window kind (drives which sub-resources apply)."""

    TWO_D = "2d"
    THREE_D = "3d"

    def __str__(self) -> str:
        return self.value


class ViewDirection(str, Enum):
    """Anatomical plane / view direction.

    Mirrors :class:`mitk.mxn.layout.ViewDirection` value-for-value so that
    the editors module remains usable without the optional ``mitk`` package
    installed.
    """

    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    ORIGINAL = "original"

    def __str__(self) -> str:
        return self.value


class StandardView(str, Enum):
    """Standard camera orientations accepted by the camera PUT endpoint."""

    ANTERIOR = "anterior"
    POSTERIOR = "posterior"
    LEFT = "left"
    RIGHT = "right"
    CRANIAL = "cranial"
    CAUDAL = "caudal"

    def __str__(self) -> str:
        return self.value


class EditorAlias(str, Enum):
    """Stable short aliases for the built-in MITK editors."""

    STD_MULTI = "stdmulti"
    MXN = "mxn"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Discovery payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditorDescriptor:
    """Editor list entry returned by ``GET /rendering/editors``.

    Attributes:
        alias: Stable short alias used in URL paths.
        plugin_id: Berry editor plugin id.
        active: True if an editor instance is currently open.
    """

    alias: str
    plugin_id: str
    active: bool


@dataclass(frozen=True)
class EditorInfo:
    """Editor metadata returned by ``GET /rendering/editors/{alias}``.

    Attributes:
        alias: Stable short alias used in URL paths.
        plugin_id: Berry editor plugin id.
        active: True if an editor instance is currently open.
        windows: Current window identifiers (URL segments for sub-resources).
            Empty when the editor is inactive.
    """

    alias: str
    plugin_id: str
    active: bool
    windows: tuple[str, ...]


# ---------------------------------------------------------------------------
# Window summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSummary:
    """Per-window summary common to every editor type.

    The contract guarantees ``view_direction`` is present iff
    ``kind == WindowKind.TWO_D``; the StdMulti 3D window omits it.

    Attributes:
        id: Render window id (URL segment for sub-resources).
        kind: Window kind (2D vs. 3D).
        view_direction: Anatomical plane the slot renders. ``None`` for 3D.
        has_camera: Whether the camera sub-resource applies.
        has_selected_slice: Whether the selected-slice sub-resource applies.
    """

    id: str
    kind: WindowKind
    view_direction: ViewDirection | None
    has_camera: bool
    has_selected_slice: bool


@dataclass(frozen=True)
class MxNWindowSummary(WindowSummary):
    """MxN cell summary — extends :class:`WindowSummary` with layout fields.

    Attributes:
        display_name: Optional human-readable label from the layout document.
            ``None`` when the cell has no display name.
        links: Per-cell synchronisation links from the layout document
            (at minimum ``{"selection": <group>}`` under v2).
        has_selected_position: Always True under v2 (per-cell selected
            position is a v2 capability, distinct from the global resource).
    """

    display_name: str | None
    links: Mapping[str, str]
    has_selected_position: bool


# ---------------------------------------------------------------------------
# Slice state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceBounds:
    """Bounds of a slice navigator.

    Attributes:
        steps: Total step count.
        min_position: World position of the first slice. ``None`` when no
            geometry is loaded.
        max_position: World position of the last slice. ``None`` when no
            geometry is loaded.
    """

    steps: int
    min_position: tuple[float, float, float] | None
    max_position: tuple[float, float, float] | None


@dataclass(frozen=True)
class SelectedSlice:
    """Selected-slice state for a 2D window.

    Attributes:
        step: Current step index.
        position: World position on the live slice plane.
        bounds: Navigator bounds.
    """

    step: int
    position: tuple[float, float, float]
    bounds: SliceBounds


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """Render-window camera state.

    Used both as a GET response (full state, with ``kind`` set and the
    irrelevant axis nulled out) and as a PUT body (any subset of fields;
    ``None`` fields are omitted from the wire payload). 2D windows carry
    ``parallel_scale``; the 3D window carries ``perspective_angle``.

    ``kind`` is local-only metadata: the contract response carries no
    ``kind`` field. :meth:`RenderWindow.get_camera` populates it from the
    surrounding window's known kind.

    Attributes:
        kind: Local-only window kind label; ignored on PUT.
        position: Camera origin in world coordinates.
        focal_point: Point the camera looks at, in world coordinates.
        view_up: View-up vector in world coordinates.
        parallel_scale: Orthographic zoom (2D windows only).
        perspective_angle: Vertical FOV in degrees (3D window only).
        standard_view: Symbolic camera orientation. Cannot be combined
            with explicit pose fields (``position`` / ``focal_point`` /
            ``view_up``) on PUT — server rejects with 400.
    """

    kind: WindowKind | None = None
    position: tuple[float, float, float] | None = None
    focal_point: tuple[float, float, float] | None = None
    view_up: tuple[float, float, float] | None = None
    parallel_scale: float | None = None
    perspective_angle: float | None = None
    standard_view: StandardView | str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Build the JSON body for a camera PUT — drops ``None`` fields."""
        body: dict[str, Any] = {}
        if self.position is not None:
            body["position"] = list(self.position)
        if self.focal_point is not None:
            body["focal_point"] = list(self.focal_point)
        if self.view_up is not None:
            body["view_up"] = list(self.view_up)
        if self.parallel_scale is not None:
            body["parallel_scale"] = self.parallel_scale
        if self.perspective_angle is not None:
            body["perspective_angle"] = self.perspective_angle
        if self.standard_view is not None:
            body["standard_view"] = (
                self.standard_view.value
                if isinstance(self.standard_view, StandardView)
                else self.standard_view
            )
        return body

    def replace(self, **fields: Any) -> Camera:
        """Return a new Camera with the given fields overridden."""
        return dataclasses.replace(self, **fields)

    def diff(self, other: Camera) -> Camera:
        """Return a Camera carrying only fields that differ from ``other``.

        ``kind`` is preserved from ``self`` (it is local metadata, never
        diffed). A field is "different" iff it is non-``None`` on ``self``
        and not equal to the value on ``other``. When ``self.kind`` is set,
        the scalar that does not apply to that kind (``perspective_angle``
        for 2D, ``parallel_scale`` for 3D) is dropped from the result so
        ``to_payload()`` cannot produce a body the server rejects.
        """
        kept: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name == "kind":
                continue
            mine = getattr(self, f.name)
            theirs = getattr(other, f.name)
            if mine is not None and mine != theirs:
                kept[f.name] = mine
        if self.kind == WindowKind.TWO_D:
            kept.pop("perspective_angle", None)
        elif self.kind == WindowKind.THREE_D:
            kept.pop("parallel_scale", None)
        return Camera(kind=self.kind, **kept)


# ---------------------------------------------------------------------------
# JSON parsers (module-private)
# ---------------------------------------------------------------------------


def _xyz(seq: Any) -> tuple[float, float, float]:
    x, y, z = seq
    return (float(x), float(y), float(z))


def _opt_xyz(seq: Any) -> tuple[float, float, float] | None:
    return None if seq is None else _xyz(seq)


def _opt_view_direction(value: Any) -> ViewDirection | None:
    return None if value is None else ViewDirection(value)


def _editor_descriptor_from_json(body: Mapping[str, Any]) -> EditorDescriptor:
    return EditorDescriptor(
        alias=body["alias"],
        plugin_id=body["plugin_id"],
        active=body["active"],
    )


def _editor_info_from_json(body: Mapping[str, Any]) -> EditorInfo:
    return EditorInfo(
        alias=body["alias"],
        plugin_id=body["plugin_id"],
        active=body["active"],
        windows=tuple(body.get("windows") or ()),
    )


def _window_summary_from_json(body: Mapping[str, Any]) -> WindowSummary:
    return WindowSummary(
        id=body["id"],
        kind=WindowKind(body["kind"]),
        view_direction=_opt_view_direction(body.get("view_direction")),
        has_camera=body["has_camera"],
        has_selected_slice=body["has_selected_slice"],
    )


def _mxn_window_summary_from_json(body: Mapping[str, Any]) -> MxNWindowSummary:
    return MxNWindowSummary(
        id=body["id"],
        kind=WindowKind(body["kind"]),
        view_direction=_opt_view_direction(body.get("view_direction")),
        has_camera=body["has_camera"],
        has_selected_slice=body["has_selected_slice"],
        display_name=body.get("name"),
        links=dict(body.get("links") or {}),
        has_selected_position=body["has_selected_position"],
    )


def _camera_from_json(body: Mapping[str, Any], *, kind: WindowKind | None) -> Camera:
    return Camera(
        kind=kind,
        position=_opt_xyz(body.get("position")),
        focal_point=_opt_xyz(body.get("focal_point")),
        view_up=_opt_xyz(body.get("view_up")),
        parallel_scale=body.get("parallel_scale"),
        perspective_angle=body.get("perspective_angle"),
    )


def _slice_state_from_json(body: Mapping[str, Any]) -> SelectedSlice:
    raw_bounds = body["bounds"]
    return SelectedSlice(
        step=int(body["step"]),
        position=_xyz(body["position"]),
        bounds=SliceBounds(
            steps=int(raw_bounds["steps"]),
            min_position=_opt_xyz(raw_bounds.get("min_position")),
            max_position=_opt_xyz(raw_bounds.get("max_position")),
        ),
    )


# ---------------------------------------------------------------------------
# Screenshot helper (module-private)
# ---------------------------------------------------------------------------


def _screenshot(
    transport: RestTransport,
    path_url: str,
    *,
    fmt: ScreenshotFormat | str,
    width: int | None,
    height: int | None,
    out: str | Path | None,
) -> bytes:
    if (width is None) != (height is None):
        raise ValueError("width and height must both be given or both omitted")
    params: dict[str, str | int] = {"format": str(fmt)}
    if width is not None:
        params["width"] = width
    if height is not None:
        params["height"] = height
    resp = transport.get_binary(path_url, params=params)
    data = resp.content
    if out is not None:
        Path(out).write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# Editor and render-window base classes
# ---------------------------------------------------------------------------


class EditorBase:
    """Common behaviour for editor handles.

    Subclasses set the ``ALIAS`` class variable to the editor's URL alias
    (e.g. ``"stdmulti"`` or ``"mxn"``). Editor handles hold only the
    transport — every state read goes to the network.
    """

    ALIAS: ClassVar[str] = ""

    def __init__(self, transport: RestTransport) -> None:
        if not self.ALIAS:
            raise TypeError(f"{type(self).__name__} must override the ALIAS class variable")
        self._transport = transport

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def alias(self) -> str:
        """Editor alias (URL segment)."""
        return self.ALIAS

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alias={self.ALIAS!r})"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_info(self) -> EditorInfo:
        """Fetch the editor's metadata (always live).

        Returns a degraded ``EditorInfo`` with ``active=False`` and empty
        ``windows`` when the editor exists but no instance is open
        (server signals ``EDITOR_NOT_ACTIVE``). A missing Qt render-window
        bridge (``RENDER_WINDOW_NOT_AVAILABLE``) is a deployment-level
        problem and still surfaces as :class:`RenderingError`.
        """
        try:
            body = self._transport.get(f"/rendering/editors/{self.ALIAS}").json()
        except EditorNotActiveError:
            return EditorInfo(alias=self.ALIAS, plugin_id="", active=False, windows=())
        return _editor_info_from_json(body)

    @property
    def is_active(self) -> bool:
        """Whether an editor instance is currently open. Always live."""
        return self.get_info().active

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
        """Capture a screenshot of the editor canvas.

        Args:
            fmt: Image encoding format (``"png"`` or ``"jpeg"``).
            width: Output width in pixels. Must be given together with
                ``height``.
            height: Output height in pixels. Must be given together with
                ``width``.
            path: If given, save the screenshot to this file path.

        Returns:
            Raw image bytes.
        """
        return _screenshot(
            self._transport,
            f"/rendering/editors/{self.ALIAS}/screenshot",
            fmt=fmt,
            width=width,
            height=height,
            out=path,
        )

    # ------------------------------------------------------------------
    # Window listing (subclasses override for richer summary types)
    # ------------------------------------------------------------------

    def _list_window_payloads(self) -> list[dict[str, Any]]:
        return list(self._transport.get(f"/rendering/editors/{self.ALIAS}/windows").json())

    def list_windows(self) -> list[RenderWindow]:
        """List the editor's render windows. Always live."""
        return [self._make_window(item) for item in self._list_window_payloads()]

    def get_window(self, window_id: str) -> RenderWindow:
        """Get a render-window handle by its id (no network call)."""
        return self._make_window({"id": window_id})

    def __getitem__(self, window_id: str) -> RenderWindow:
        return self.get_window(window_id)

    def __iter__(self) -> Iterator[RenderWindow]:
        return iter(self.list_windows())

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _make_window(self, payload: Mapping[str, Any]) -> RenderWindow:
        """Construct a window handle from a list-item payload.

        Subclasses override to return their specialised window type and to
        propagate richer payload fields (``view_direction``, ``links``, ...)
        as eager metadata so that simple usage doesn't need an extra GET.
        """
        return RenderWindow(
            self._transport,
            self.ALIAS,
            payload["id"],
            kind=WindowKind(payload["kind"]) if "kind" in payload else None,
        )


class RenderWindow:
    """Handle to a single render window of an editor.

    Args:
        transport: Configured REST transport.
        editor_alias: Editor alias the window belongs to.
        window_id: Render-window id (URL segment).
        kind: Optional eager :class:`WindowKind` populated by the editor's
            window listing. Cached because the kind of a window is an
            immutable identity field. Falls back to a network call when
            ``None``.

    Note:
        ``view_direction`` is intentionally NOT cached: MxN cells can be
        re-bound to a different anatomical plane at runtime. Reading
        :attr:`view_direction` always issues a fresh ``GET`` against the
        per-window summary so callers see the live server state.
    """

    def __init__(
        self,
        transport: RestTransport,
        editor_alias: str,
        window_id: str,
        *,
        kind: WindowKind | None = None,
    ) -> None:
        self._transport = transport
        self._editor_alias = editor_alias
        self._id = window_id
        self._kind = kind

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Render-window id (URL segment for sub-resources)."""
        return self._id

    @property
    def editor_alias(self) -> str:
        """Alias of the editor this window belongs to."""
        return self._editor_alias

    @property
    def kind(self) -> WindowKind:
        """Window kind. Immutable identity field — cached after first
        resolution. Resolved via ``GET .../windows/{id}`` on first access
        when not pre-populated by the editor's window listing."""
        if self._kind is None:
            self._kind = self.get_summary().kind
        return self._kind

    @property
    def view_direction(self) -> ViewDirection | None:
        """View direction. ``None`` for 3D windows. Always live — MxN cells
        can be re-bound to a different anatomical plane at runtime, so this
        is not cached."""
        return self.get_summary().view_direction

    def __repr__(self) -> str:
        return f"{type(self).__name__}(editor={self._editor_alias!r}, id={self._id!r})"

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _path(self, suffix: str = "") -> str:
        base = f"/rendering/editors/{self._editor_alias}/windows/{self._id}"
        return f"{base}{suffix}"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> WindowSummary:
        """Fetch the per-window summary from the server (always live)."""
        body = self._transport.get(self._path()).json()
        summary = _window_summary_from_json(body)
        # Kind is immutable — refresh the cache. View direction is left
        # uncached because MxN cells can re-bind their plane at runtime.
        self._kind = summary.kind
        return summary

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
        """Capture a screenshot of this single render window."""
        return _screenshot(
            self._transport,
            self._path("/screenshot"),
            fmt=fmt,
            width=width,
            height=height,
            out=path,
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def get_camera(self) -> Camera:
        """Fetch the current camera state."""
        body = self._transport.get(self._path("/camera")).json()
        return _camera_from_json(body, kind=self.kind)

    def set_camera(
        self,
        camera: Camera | None = None,
        /,
        **fields: Any,
    ) -> None:
        """Update the camera state.

        Two call shapes:

        - **Value form:** ``set_camera(Camera(parallel_scale=120))`` — sends
          the camera's non-``None`` fields.
        - **Kwargs form:** ``set_camera(parallel_scale=120, standard_view="anterior")``
          — equivalent to building a one-off Camera.

        Mixing the two raises :class:`TypeError`.
        """
        if camera is not None and fields:
            raise TypeError("set_camera takes either a Camera value or keyword fields, not both")
        if camera is None:
            camera = Camera(**fields)
        body = camera.to_payload()
        if not body:
            raise ValueError("set_camera requires at least one camera field to update")
        self._transport.put(self._path("/camera"), json=body)

    # ------------------------------------------------------------------
    # Selected slice
    # ------------------------------------------------------------------

    def get_selected_slice(self) -> SelectedSlice:
        """Fetch the selected-slice state.

        Raises:
            UnsupportedOperationError: For the StdMulti 3D window.
        """
        body = self._transport.get(self._path("/selected-slice")).json()
        return _slice_state_from_json(body)

    def set_selected_slice(self, step: int) -> None:
        """Step the slice navigator to a target index.

        ``step`` is not range-checked — out-of-range values are clamped or
        snapped by MITK.

        Raises:
            UnsupportedOperationError: For the StdMulti 3D window.
        """
        self._transport.put(self._path("/selected-slice"), json={"step": int(step)})

    # ------------------------------------------------------------------
    # Per-window node ergonomics
    # ------------------------------------------------------------------

    def _check_node_owner(self, node: DataNode) -> None:
        """Reject nodes whose transport differs from this window's.

        Per-window node-property helpers route through ``DataNode``'s own
        transport. Silently honoring a node from a different ``Workbench``
        would target the wrong server. CLAUDE.md guarantees independent
        ``Workbench`` instances are independent — surface the violation.
        """
        if node.transport is not self._transport:
            raise ValueError(
                "DataNode belongs to a different Workbench transport than this RenderWindow"
            )

    def set_node_visible(self, node: DataNode, visible: bool) -> None:
        """Set this node's per-window visibility property."""
        self._check_node_owner(node)
        node.update_properties(scope=PropertyScope.NODE, context=self._id, visible=visible)

    def is_node_visible(self, node: DataNode) -> bool:
        """Read this node's per-window visibility property."""
        self._check_node_owner(node)
        return bool(node.get_property("visible", scope=PropertyScope.NODE, context=self._id))

    def set_node_layer(self, node: DataNode, layer: int) -> None:
        """Set this node's per-window layer (stack order) property."""
        self._check_node_owner(node)
        node.update_properties(scope=PropertyScope.NODE, context=self._id, layer=int(layer))

    def get_node_layer(self, node: DataNode) -> int:
        """Read this node's per-window layer (stack order) property."""
        self._check_node_owner(node)
        return int(node.get_property("layer", scope=PropertyScope.NODE, context=self._id))
