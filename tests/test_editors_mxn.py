# SPDX-FileCopyrightText: 2026, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
#
# SPDX-License-Identifier: Apache-2.0

"""MxNEditor — info, listings, layout I/O (raw + typed), per-cell selected-position."""

from __future__ import annotations

import json
from typing import Any

import pytest
import responses

from mitk_workbench_remote import (
    MxNEditor,
    MxNRenderWindow,
    MxNWindowSummary,
    SelectedPosition,
    ViewDirection,
    WindowKind,
)
from mitk_workbench_remote.transport import RestTransport

BASE = "http://127.0.0.1:8080"


def _api(path: str) -> str:
    return f"{BASE}/api/v1{path}"


def _transport() -> RestTransport:
    return RestTransport(BASE, token="t", transfer_mode="direct")


_THREE_UP_LAYOUT: dict[str, Any] = {
    "version": "2.0",
    "name": "Three Views",
    "groups": {"main": {"select_all": True}},
    "root": {
        "type": "split",
        "orientation": "horizontal",
        "children": [
            {
                "type": "window",
                "id": "mxn__widget0",
                "view_direction": "axial",
                "links": {"selection": "main"},
                "size": 1,
            },
            {
                "type": "window",
                "id": "mxn__widget1",
                "view_direction": "sagittal",
                "links": {"selection": "main"},
                "size": 1,
            },
            {
                "type": "window",
                "id": "mxn__widget2",
                "view_direction": "coronal",
                "links": {"selection": "main"},
                "size": 1,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Window listing — returns MxNRenderWindow handles with eager metadata
# ---------------------------------------------------------------------------


@responses.activate
def test_list_windows_returns_mxn_render_windows() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/mxn/windows"),
        json=[
            {
                "id": "mxn__widget0",
                "name": "Tumor axial",
                "kind": "2d",
                "view_direction": "axial",
                "links": {"selection": "main"},
            },
            {
                "id": "mxn__widget1",
                "kind": "2d",
                "view_direction": "sagittal",
                "links": {"selection": "main"},
            },
        ],
    )
    # view_direction is not cached on MxN cells (re-bindable at runtime),
    # so probing it would issue an additional GET against the per-cell
    # summary endpoint. Mock that here so the test reflects the live read.
    responses.add(
        responses.GET,
        _api("/rendering/editors/mxn/windows/mxn__widget0"),
        json={
            "id": "mxn__widget0",
            "name": "Tumor axial",
            "kind": "2d",
            "view_direction": "axial",
            "links": {"selection": "main"},
            "has_camera": True,
            "has_selected_slice": True,
            "has_selected_position": True,
        },
    )
    windows = MxNEditor(_transport()).list_windows()
    assert all(isinstance(w, MxNRenderWindow) for w in windows)
    assert [w.id for w in windows] == ["mxn__widget0", "mxn__widget1"]
    assert windows[0].view_direction == ViewDirection.AXIAL


def test_get_window_does_not_round_trip() -> None:
    editor = MxNEditor(_transport())
    win = editor["mxn__widget0"]
    assert isinstance(win, MxNRenderWindow)
    assert win.id == "mxn__widget0"
    # Per v2 contract every MxN cell is 2D — eager fast-path.
    assert win.kind == WindowKind.TWO_D


@responses.activate
def test_window_get_summary_returns_mxn_summary() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/mxn/windows/mxn__widget0"),
        json={
            "id": "mxn__widget0",
            "name": "Tumor axial",
            "kind": "2d",
            "view_direction": "axial",
            "links": {"selection": "main"},
            "has_camera": True,
            "has_selected_slice": True,
            "has_selected_position": True,
        },
    )
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    summary = win.get_summary()
    assert isinstance(summary, MxNWindowSummary)
    assert summary.display_name == "Tumor axial"
    assert summary.links == {"selection": "main"}
    assert summary.has_selected_position is True


# ---------------------------------------------------------------------------
# Per-cell selected position
# ---------------------------------------------------------------------------


@responses.activate
def test_get_selected_position_parses_payload() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/mxn/windows/mxn__widget0/selected-position"),
        json={
            "position": [127.5, 83.2, 45.0],
            "bounds": {
                "min_position": [0.0, 0.0, 0.0],
                "max_position": [255.0, 255.0, 90.0],
            },
        },
    )
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    pos = win.get_selected_position()
    assert isinstance(pos, SelectedPosition)
    assert pos.position == (127.5, 83.2, 45.0)
    assert pos.bounds.min_position == (0.0, 0.0, 0.0)
    assert pos.bounds.max_position == (255.0, 255.0, 90.0)


@responses.activate
def test_set_selected_position_sends_put() -> None:
    responses.add(
        responses.PUT,
        _api("/rendering/editors/mxn/windows/mxn__widget0/selected-position"),
        body=b"",
        status=204,
    )
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    win.set_selected_position((1.0, 2.0, 3.0))
    body = json.loads(responses.calls[0].request.body)
    assert body == {"position": [1.0, 2.0, 3.0]}


def test_set_selected_position_rejects_wrong_length() -> None:
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    with pytest.raises(ValueError, match="exactly 3 elements"):
        win.set_selected_position([1.0, 2.0])


# ---------------------------------------------------------------------------
# Layout I/O — raw-dict path (always available)
# ---------------------------------------------------------------------------


@responses.activate
def test_get_layout_json_returns_raw_dict() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/mxn/layout"),
        json=_THREE_UP_LAYOUT,
    )
    body = MxNEditor(_transport()).get_layout_json()
    assert body == _THREE_UP_LAYOUT


@responses.activate
def test_set_layout_with_dict_is_raw_passthrough() -> None:
    responses.add(
        responses.PUT,
        _api("/rendering/editors/mxn/layout"),
        json=_THREE_UP_LAYOUT,
        status=200,
    )
    applied = MxNEditor(_transport()).set_layout(_THREE_UP_LAYOUT)
    sent = json.loads(responses.calls[0].request.body)
    assert sent == _THREE_UP_LAYOUT
    assert applied == _THREE_UP_LAYOUT


# ---------------------------------------------------------------------------
# Layout I/O — typed path (skipped when `mitk` isn't available)
# ---------------------------------------------------------------------------


@responses.activate
def test_get_layout_returns_typed_document() -> None:
    pytest.importorskip("mitk.mxn.layout")
    responses.add(
        responses.GET,
        _api("/rendering/editors/mxn/layout"),
        json=_THREE_UP_LAYOUT,
    )
    doc = MxNEditor(_transport()).get_layout()
    from mitk.mxn.layout import MxNLayoutDocument

    assert isinstance(doc, MxNLayoutDocument)
    assert sorted(w.id for w in doc.root.windows()) == [
        "mxn__widget0",
        "mxn__widget1",
        "mxn__widget2",
    ]


@responses.activate
def test_update_layout_calls_get_then_put_once() -> None:
    pytest.importorskip("mitk.mxn.layout")
    responses.add(
        responses.GET,
        _api("/rendering/editors/mxn/layout"),
        json=_THREE_UP_LAYOUT,
    )
    responses.add(
        responses.PUT,
        _api("/rendering/editors/mxn/layout"),
        json=_THREE_UP_LAYOUT,
        status=200,
    )
    from mitk.mxn.layout import MxNWindowSelector

    editor = MxNEditor(_transport())
    result = editor.update_layout(
        lambda d: MxNWindowSelector(d).by_view(ViewDirection.AXIAL).link_to("axial_group")
    )
    methods = [c.request.method for c in responses.calls]
    assert methods == ["GET", "PUT"]
    # The lambda introduces a new group — the document we sent must
    # reference it under top-level groups (strict mode).
    sent = json.loads(responses.calls[1].request.body)
    assert "axial_group" in sent["groups"]
    # Round-tripped result is a typed document.
    from mitk.mxn.layout import MxNLayoutDocument

    assert isinstance(result, MxNLayoutDocument)


def test_update_layout_rejects_non_document_return() -> None:
    pytest.importorskip("mitk.mxn.layout")
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            _api("/rendering/editors/mxn/layout"),
            json=_THREE_UP_LAYOUT,
        )
        editor = MxNEditor(_transport())
        with pytest.raises(TypeError, match="MxNLayoutDocument"):
            editor.update_layout(lambda _d: {"version": "2.0"})  # type: ignore[arg-type]


@responses.activate
def test_apply_preset_three_up() -> None:
    pytest.importorskip("mitk.mxn.layout")
    responses.add(
        responses.PUT,
        _api("/rendering/editors/mxn/layout"),
        json=_THREE_UP_LAYOUT,
        status=200,
    )
    doc = MxNEditor(_transport()).apply_preset("three-up")
    sent = json.loads(responses.calls[0].request.body)
    assert sent["version"] == "2.0"
    assert {w["view_direction"] for w in sent["root"]["children"]} == {
        "axial",
        "sagittal",
        "coronal",
    }
    from mitk.mxn.layout import MxNLayoutDocument

    assert isinstance(doc, MxNLayoutDocument)


@responses.activate
def test_apply_grid_2x2() -> None:
    pytest.importorskip("mitk.mxn.layout")
    grid_layout = dict(_THREE_UP_LAYOUT)  # placeholder server response shape
    responses.add(
        responses.PUT,
        _api("/rendering/editors/mxn/layout"),
        json=grid_layout,
        status=200,
    )
    MxNEditor(_transport()).apply_grid(2, 2)
    sent = json.loads(responses.calls[0].request.body)
    # Outer vertical of two horizontal rows -> 4 leaf windows total.
    leaves = []

    def collect(node: dict[str, Any]) -> None:
        if node["type"] == "window":
            leaves.append(node)
        else:
            for c in node["children"]:
                collect(c)

    collect(sent["root"])
    assert len(leaves) == 4
