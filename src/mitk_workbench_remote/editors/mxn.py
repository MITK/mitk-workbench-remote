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

"""MxN multi-widget editor handle and per-cell render-window handle.

Wraps ``GET /rendering/editors/mxn`` and its sub-resources, plus the
layout endpoint (``GET/PUT /rendering/editors/mxn/layout``).

Layout I/O comes in two flavours:

- **Raw-dict** (always available): :meth:`MxNEditor.get_layout_json` and
  ``set_layout(dict)`` move the wire JSON unchanged. No DSL dependency.
- **Typed** (requires the optional ``mitk`` package): :meth:`MxNEditor.get_layout`,
  ``set_layout(MxNLayoutDocument)``, :meth:`MxNEditor.update_layout`,
  :meth:`MxNEditor.apply_grid`, :meth:`MxNEditor.apply_preset` use the
  :mod:`mitk.mxn.layout` DSL for typed construction, validation, and
  bulk transforms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mitk_workbench_remote.editors._base import (
    EditorAlias,
    EditorBase,
    MxNWindowSummary,
    RenderWindow,
    ViewDirection,
    WindowKind,
    _mxn_window_summary_from_json,
    _xyz,
)
from mitk_workbench_remote.workbench import (
    SelectedPosition,
    _position_bounds_from_json,
)

if TYPE_CHECKING:
    from mitk.mxn.layout import Group, MxNLayoutDocument


# ---------------------------------------------------------------------------
# Optional layout-DSL access
# ---------------------------------------------------------------------------


def _import_layout() -> Any:
    """Import :mod:`mitk.mxn.layout` lazily and raise an actionable error
    if the optional ``mitk`` package is not installed.
    """
    try:
        from mitk.mxn import layout
    except ImportError as exc:
        raise ImportError(
            "The typed MxN layout API requires the optional 'mitk' package. "
            "Install with: pip install mitk-workbench-remote[layout]"
        ) from exc
    return layout


# ---------------------------------------------------------------------------
# Per-cell render window
# ---------------------------------------------------------------------------


class MxNRenderWindow(RenderWindow):
    """Handle to a single MxN cell.

    Adds the per-cell selected-position primitive and overrides
    :meth:`get_summary` to return the richer :class:`MxNWindowSummary`.
    """

    @property
    def kind(self) -> WindowKind:
        # v2-only short-circuit: under the v2 schema every MxN cell is 2D
        # (the layout schema's view_direction enum has no 3D value), so a
        # handle obtained via `editor[id]` doesn't need a network round-trip
        # just to learn its kind. WHEN v3 INTRODUCES 3D MxN CELLS, drop this
        # override and let the base class fetch the live summary instead —
        # otherwise this returns the wrong kind silently.
        if self._kind is None:
            self._kind = WindowKind.TWO_D
        return self._kind

    def get_summary(self) -> MxNWindowSummary:
        """Fetch the per-cell summary (always live)."""
        body = self._transport.get(self._path()).json()
        summary = _mxn_window_summary_from_json(body)
        # Refresh the kind cache only — view_direction is intentionally
        # uncached because MxN cells can re-bind their plane at runtime.
        self._kind = summary.kind
        return summary

    # ------------------------------------------------------------------
    # Per-cell selected position
    # ------------------------------------------------------------------

    def get_selected_position(self) -> SelectedPosition:
        """Fetch this cell's 3D world anchor and scene bounds."""
        body = self._transport.get(self._path("/selected-position")).json()
        return SelectedPosition(
            position=_xyz(body["position"]),
            bounds=_position_bounds_from_json(body["bounds"]),
        )

    def set_selected_position(
        self, position: tuple[float, float, float] | Sequence[float]
    ) -> None:
        """Set this cell's 3D world anchor.

        Distinct from the global ``/rendering/selected-position`` resource:
        whether the change propagates to other cells or to the global anchor
        depends on the workbench's interactive coupling toolbar state, which
        is intentionally not exposed via REST.

        Args:
            position: Target world position ``[x, y, z]``.

        Raises:
            ValueError: If ``position`` does not have exactly 3 elements.
        """
        if len(position) != 3:
            raise ValueError(
                f"position must have exactly 3 elements [x, y, z], got {len(position)}"
            )
        self._transport.put(
            self._path("/selected-position"),
            json={"position": [float(c) for c in position]},
        )


# ---------------------------------------------------------------------------
# MxN editor
# ---------------------------------------------------------------------------


class MxNEditor(EditorBase):
    """Handle to the MxN multi-widget editor.

    Cell ids are the canonical fully-qualified form
    ``<editor_name>__<bare>`` (e.g. ``"mxn__widget0"``). The same string
    appears in the URL, the layout document, and the engine — no prefix
    translation happens at any boundary.
    """

    ALIAS = EditorAlias.MXN.value

    # ------------------------------------------------------------------
    # Window construction — return MxNRenderWindow with eager metadata
    # ------------------------------------------------------------------

    def _make_window(self, payload: Mapping[str, Any]) -> MxNRenderWindow:
        kind = WindowKind(payload["kind"]) if "kind" in payload else None
        return MxNRenderWindow(
            self._transport,
            self.ALIAS,
            payload["id"],
            kind=kind,
        )

    def list_windows(self) -> list[MxNRenderWindow]:  # type: ignore[override]
        """List the editor's cells (in pre-order traversal of the layout)."""
        return [self._make_window(item) for item in self._list_window_payloads()]

    def get_window(self, window_id: str) -> MxNRenderWindow:
        return self._make_window({"id": window_id})

    def __getitem__(self, window_id: str) -> MxNRenderWindow:
        return self.get_window(window_id)

    # ------------------------------------------------------------------
    # Layout — raw-dict path (always available)
    # ------------------------------------------------------------------

    def get_layout_json(self) -> dict[str, Any]:
        """Fetch the current layout document as a raw dict.

        Returns the JSON body unchanged — no DSL parsing — so callers can
        round-trip the layout byte-for-byte without a ``mitk`` install.
        """
        return dict(self._transport.get(f"/rendering/editors/{self.ALIAS}/layout").json())

    # ------------------------------------------------------------------
    # Layout — typed path (requires `mitk`)
    # ------------------------------------------------------------------

    def get_layout(self) -> MxNLayoutDocument:
        """Fetch the current layout, parsed into a typed
        :class:`MxNLayoutDocument`. Requires the optional ``mitk`` package.
        """
        layout_mod = _import_layout()
        return layout_mod.MxNLayoutDocument.from_json(self.get_layout_json())

    def set_layout(
        self,
        layout: MxNLayoutDocument | Mapping[str, Any] | str | Path,
    ) -> dict[str, Any]:
        """Apply a layout document to the editor.

        Accepts:

        - :class:`MxNLayoutDocument` — sent as-is (trusted; the DSL
          validates on construction). Requires the optional ``mitk``
          package.
        - ``Mapping`` (raw dict) — sent as-is, **no client-side
          validation**. Always available; the server enforces the schema
          and surfaces 400 ``INVALID_REQUEST`` on a malformed body.
        - JSON ``str`` or :class:`pathlib.Path` — parsed via
          :meth:`MxNLayoutDocument.from_json` and validated client-side
          before PUT (catches typos / missing groups before a wasted
          round-trip). Requires the optional ``mitk`` package.

        Returns the freshly-serialized layout returned by the server in
        the PUT response (same shape as :meth:`get_layout_json`).

        **All existing cells are torn down and rebuilt from the document
        — any cell handle a caller cached prior to the call is invalid
        afterwards.**
        """
        body: Mapping[str, Any]
        if _is_layout_document(layout):
            body = layout.to_json()  # type: ignore[union-attr]
        elif isinstance(layout, Path):
            text = layout.read_text(encoding="utf-8")
            body = _import_layout().MxNLayoutDocument.from_json(text).to_json()
        elif isinstance(layout, str):
            body = _import_layout().MxNLayoutDocument.from_json(layout).to_json()
        elif isinstance(layout, Mapping):
            # Raw-dict path: no DSL involvement; server validates.
            body = layout
        else:
            raise TypeError(
                f"layout must be MxNLayoutDocument, Mapping, str, or Path; "
                f"got {type(layout).__name__}"
            )

        response = self._transport.put(f"/rendering/editors/{self.ALIAS}/layout", json=body)
        return dict(response.json())

    # ------------------------------------------------------------------
    # GET-modify-PUT helper
    # ------------------------------------------------------------------

    def update_layout(
        self,
        transform: Callable[[MxNLayoutDocument], MxNLayoutDocument],
    ) -> MxNLayoutDocument:
        """GET the layout, apply ``transform``, validate, PUT it back.

        ``transform`` receives the freshly-fetched typed
        :class:`MxNLayoutDocument` and returns a new one (DSL operations
        are immutable). The returned document is validated client-side
        before PUT; the parsed PUT response is returned to the caller.

        Idiomatic use::

            from mitk.mxn.layout import MxNWindowSelector, ViewDirection

            def link_axials(doc):
                return (
                    MxNWindowSelector(doc)
                    .by_view(ViewDirection.AXIAL)
                    .link_to("axial_group")
                )

            wb.mxn.update_layout(link_axials)
        """
        layout_mod = _import_layout()
        current = self.get_layout()
        modified = transform(current)
        if not isinstance(modified, layout_mod.MxNLayoutDocument):
            raise TypeError(
                f"transform must return MxNLayoutDocument, got {type(modified).__name__}"
            )
        modified.validate()
        applied = self.set_layout(modified)
        return layout_mod.MxNLayoutDocument.from_json(applied)

    # ------------------------------------------------------------------
    # Convenience preset entry points
    # ------------------------------------------------------------------

    def apply_grid(
        self,
        rows: int,
        cols: int,
        *,
        view_directions: Sequence[ViewDirection] | None = None,
        group: str | Group = "main",
    ) -> MxNLayoutDocument:
        """Apply a fresh ``rows`` x ``cols`` grid layout.

        Returns the freshly-serialized layout from the PUT response,
        parsed back into a typed :class:`MxNLayoutDocument`.
        """
        layout_mod = _import_layout()
        # The local ``ViewDirection`` enum mirrors ``mitk.mxn.layout.ViewDirection``
        # value-for-value, but the layout's builder type-checks against its
        # own enum class — coerce by string value.
        coerced_views: Sequence[Any] | None = (
            None
            if view_directions is None
            else [
                layout_mod.ViewDirection(v.value if isinstance(v, ViewDirection) else v)
                for v in view_directions
            ]
        )
        root = layout_mod.grid(rows, cols, view_directions=coerced_views, group=group)
        doc: MxNLayoutDocument = layout_mod.MxNLayoutDocument.create(root=root)
        applied = self.set_layout(doc)
        return layout_mod.MxNLayoutDocument.from_json(applied)

    def apply_preset(self, name: str) -> MxNLayoutDocument:
        """Apply a named preset from :data:`mitk.mxn.layout.PRESETS`.

        Returns the freshly-serialized layout from the PUT response,
        parsed back into a typed :class:`MxNLayoutDocument`.
        """
        layout_mod = _import_layout()
        doc = layout_mod.preset(name)
        applied = self.set_layout(doc)
        return layout_mod.MxNLayoutDocument.from_json(applied)


def _is_layout_document(value: object) -> bool:
    """Best-effort isinstance check for ``MxNLayoutDocument`` that doesn't
    force-import the optional ``mitk`` package.

    If ``mitk`` isn't importable the user can't have constructed an
    ``MxNLayoutDocument`` in the first place, so a ``False`` here is safe.
    """
    try:
        layout_mod = _import_layout()
    except ImportError:
        return False
    return isinstance(value, layout_mod.MxNLayoutDocument)
