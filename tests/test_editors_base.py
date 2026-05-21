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

"""Shared editor / render-window behaviour (Camera, summaries, error mapping)."""

from __future__ import annotations

import json

import pytest
import responses

from mitk_workbench_remote import (
    Camera,
    EditorNotActiveError,
    RenderWindow,
    RenderWindowNotFoundError,
    SliceBounds,
    StandardView,
    StdMultiEditor,
    UnsupportedOperationError,
    ViewDirection,
    WindowKind,
)
from mitk_workbench_remote.transport import RestTransport

BASE = "http://127.0.0.1:8080"


def _api(path: str) -> str:
    return f"{BASE}/api/v1{path}"


def _transport() -> RestTransport:
    return RestTransport(BASE, token="t", transfer_mode="direct")


# ---------------------------------------------------------------------------
# Camera dataclass
# ---------------------------------------------------------------------------


def test_camera_to_payload_drops_none_fields() -> None:
    cam = Camera(parallel_scale=120.0, standard_view=StandardView.ANTERIOR)
    body = cam.to_payload()
    assert body == {"parallel_scale": 120.0, "standard_view": "anterior"}


def test_camera_to_payload_serialises_pose_arrays() -> None:
    cam = Camera(
        position=(1.0, 2.0, 3.0),
        focal_point=(4.0, 5.0, 6.0),
        view_up=(0.0, 1.0, 0.0),
    )
    assert cam.to_payload() == {
        "position": [1.0, 2.0, 3.0],
        "focal_point": [4.0, 5.0, 6.0],
        "view_up": [0.0, 1.0, 0.0],
    }


def test_camera_to_payload_accepts_str_standard_view() -> None:
    assert Camera(standard_view="left").to_payload() == {"standard_view": "left"}


def test_camera_replace_returns_new_instance_with_overrides() -> None:
    a = Camera(parallel_scale=10.0, standard_view=StandardView.LEFT)
    b = a.replace(parallel_scale=20.0)
    assert a.parallel_scale == 10.0  # immutable
    assert b.parallel_scale == 20.0
    assert b.standard_view == StandardView.LEFT


def test_camera_diff_keeps_only_differing_non_none_fields() -> None:
    base = Camera(parallel_scale=10.0, standard_view=StandardView.LEFT)
    new = Camera(parallel_scale=20.0, standard_view=StandardView.LEFT)
    diff = new.diff(base)
    assert diff.to_payload() == {"parallel_scale": 20.0}


def test_camera_diff_ignores_kind() -> None:
    base = Camera(kind=WindowKind.TWO_D, parallel_scale=10.0)
    new = Camera(kind=WindowKind.THREE_D, parallel_scale=10.0)
    assert new.diff(base).to_payload() == {}


# ---------------------------------------------------------------------------
# RenderWindow camera GET / PUT
# ---------------------------------------------------------------------------


@responses.activate
def test_get_camera_populates_kind_from_window() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/axial/camera"),
        json={
            "position": [0.0, 0.0, 100.0],
            "focal_point": [0.0, 0.0, 0.0],
            "view_up": [0.0, 1.0, 0.0],
            "parallel_scale": 50.0,
        },
    )
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    cam = win.get_camera()
    assert cam.kind == WindowKind.TWO_D
    assert cam.parallel_scale == 50.0
    assert cam.position == (0.0, 0.0, 100.0)


@responses.activate
def test_set_camera_value_form() -> None:
    responses.add(
        responses.PUT,
        _api("/rendering/editors/stdmulti/windows/axial/camera"),
        body=b"",
        status=204,
    )
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    win.set_camera(Camera(parallel_scale=120.0))
    body = json.loads(responses.calls[0].request.body)
    assert body == {"parallel_scale": 120.0}


@responses.activate
def test_set_camera_kwargs_form() -> None:
    responses.add(
        responses.PUT,
        _api("/rendering/editors/stdmulti/windows/axial/camera"),
        body=b"",
        status=204,
    )
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    win.set_camera(standard_view="anterior", parallel_scale=120.0)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"standard_view": "anterior", "parallel_scale": 120.0}


def test_set_camera_mixed_value_and_kwargs_raises() -> None:
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    with pytest.raises(TypeError, match="Camera value or keyword fields"):
        win.set_camera(Camera(parallel_scale=10.0), parallel_scale=20.0)


def test_set_camera_empty_payload_raises() -> None:
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    with pytest.raises(ValueError, match="at least one camera field"):
        win.set_camera()


# ---------------------------------------------------------------------------
# RenderWindow selected-slice
# ---------------------------------------------------------------------------


@responses.activate
def test_get_selected_slice_returns_dataclass() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/axial/selected-slice"),
        json={
            "step": 45,
            "position": [127.5, 83.2, 45.0],
            "bounds": {
                "steps": 90,
                "min_position": [0.0, 0.0, 0.0],
                "max_position": [255.0, 255.0, 90.0],
            },
        },
    )
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    state = win.get_selected_slice()
    assert state.step == 45
    assert state.position == (127.5, 83.2, 45.0)
    assert isinstance(state.bounds, SliceBounds)
    assert state.bounds.steps == 90
    assert state.bounds.min_position == (0.0, 0.0, 0.0)
    assert state.bounds.max_position == (255.0, 255.0, 90.0)


@responses.activate
def test_get_selected_slice_handles_null_bounds() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/axial/selected-slice"),
        json={
            "step": 0,
            "position": [0.0, 0.0, 0.0],
            "bounds": {"steps": 0, "min_position": None, "max_position": None},
        },
    )
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    state = win.get_selected_slice()
    assert state.bounds.min_position is None
    assert state.bounds.max_position is None


@responses.activate
def test_set_selected_slice_sends_step() -> None:
    responses.add(
        responses.PUT,
        _api("/rendering/editors/stdmulti/windows/axial/selected-slice"),
        body=b"",
        status=204,
    )
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    win.set_selected_slice(42)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"step": 42}


# ---------------------------------------------------------------------------
# RenderWindow screenshot
# ---------------------------------------------------------------------------


@responses.activate
def test_window_screenshot_default_format() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/axial/screenshot"),
        body=b"\x89PNG\r\n\x1a\n",
        status=200,
        content_type="image/png",
    )
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    data = win.screenshot()
    assert data.startswith(b"\x89PNG")


def test_window_screenshot_rejects_partial_dimensions() -> None:
    win = RenderWindow(_transport(), "stdmulti", "axial", kind=WindowKind.TWO_D)
    with pytest.raises(ValueError, match="both be given or both omitted"):
        win.screenshot(width=512)


# ---------------------------------------------------------------------------
# Error mapping (3D slice → UnsupportedOperationError, missing window → NotFound)
# ---------------------------------------------------------------------------


@responses.activate
def test_3d_slice_raises_unsupported_operation() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/3d/selected-slice"),
        json={
            "error": {
                "type": "https://docs.mitk.org/api/errors/UNSUPPORTED_OPERATION",
                "code": "UNSUPPORTED_OPERATION",
                "title": "Unsupported Operation",
                "message": "selected-slice is not applicable to the 3D window.",
                "status": 404,
            }
        },
        status=404,
    )
    win = RenderWindow(_transport(), "stdmulti", "3d", kind=WindowKind.THREE_D)
    with pytest.raises(UnsupportedOperationError) as excinfo:
        win.get_selected_slice()
    assert excinfo.value.editor_alias == "stdmulti"
    assert excinfo.value.window_id == "3d"
    assert excinfo.value.operation == "selected-slice"


@responses.activate
def test_unknown_window_raises_render_window_not_found() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/bogus/camera"),
        json={
            "error": {
                "type": "https://docs.mitk.org/api/errors/RENDER_WINDOW_NOT_FOUND",
                "code": "RENDER_WINDOW_NOT_FOUND",
                "title": "Render Window Not Found",
                "message": "No render window named 'bogus' in the addressed editor",
                "status": 404,
            }
        },
        status=404,
    )
    win = RenderWindow(_transport(), "stdmulti", "bogus", kind=WindowKind.TWO_D)
    with pytest.raises(RenderWindowNotFoundError) as excinfo:
        win.get_camera()
    assert excinfo.value.editor_alias == "stdmulti"
    assert excinfo.value.window_id == "bogus"


@responses.activate
def test_inactive_editor_raises_editor_not_active() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/screenshot"),
        json={
            "error": {
                "type": "https://docs.mitk.org/api/errors/EDITOR_NOT_ACTIVE",
                "code": "EDITOR_NOT_ACTIVE",
                "title": "Editor Not Active",
                "message": "StdMultiWidgetEditor is not open",
                "status": 503,
            }
        },
        status=503,
    )
    editor = StdMultiEditor(_transport())
    with pytest.raises(EditorNotActiveError) as excinfo:
        editor.screenshot()
    assert excinfo.value.alias == "stdmulti"


@responses.activate
def test_get_info_returns_degraded_info_when_editor_not_active() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti"),
        json={
            "error": {
                "type": "https://docs.mitk.org/api/errors/EDITOR_NOT_ACTIVE",
                "code": "EDITOR_NOT_ACTIVE",
                "title": "Editor Not Active",
                "message": "StdMultiWidgetEditor is not open",
                "status": 503,
            }
        },
        status=503,
    )
    editor = StdMultiEditor(_transport())
    info = editor.get_info()
    assert info.alias == "stdmulti"
    assert info.active is False
    assert info.windows == ()


@responses.activate
def test_is_active_returns_false_when_editor_not_active() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti"),
        json={
            "error": {
                "code": "EDITOR_NOT_ACTIVE",
                "title": "Editor Not Active",
                "message": "StdMultiWidgetEditor is not open",
                "status": 503,
            }
        },
        status=503,
    )
    editor = StdMultiEditor(_transport())
    assert editor.is_active is False


# ---------------------------------------------------------------------------
# EditorBase: list / iter / __getitem__
# ---------------------------------------------------------------------------


@responses.activate
def test_editor_list_windows_returns_handles() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows"),
        json=[
            {"id": "axial", "kind": "2d", "view_direction": "axial"},
            {"id": "3d", "kind": "3d"},
        ],
    )
    # view_direction is not cached -- the user can tilt a stdmulti plane to a
    # custom orientation, so the live per-window summary is authoritative.
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/axial"),
        json={
            "id": "axial",
            "kind": "2d",
            "view_direction": "axial",
            "has_camera": True,
            "has_selected_slice": True,
        },
    )
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/3d"),
        json={
            "id": "3d",
            "kind": "3d",
            "view_direction": None,
            "has_camera": True,
            "has_selected_slice": False,
        },
    )
    editor = StdMultiEditor(_transport())
    windows = editor.list_windows()
    assert len(windows) == 2
    assert windows[0].id == "axial"
    assert windows[0].kind == WindowKind.TWO_D
    assert windows[0].view_direction == ViewDirection.AXIAL
    assert windows[1].kind == WindowKind.THREE_D
    assert windows[1].view_direction is None


def test_editor_getitem_returns_handle_without_network_for_id_and_kind() -> None:
    # id and kind are derivable without network (kind is engine-fixed for
    # stdmulti's named slots). view_direction is intentionally NOT covered
    # here because it is no longer cached -- see
    # ``test_view_direction_always_hits_network`` below.
    editor = StdMultiEditor(_transport())
    win = editor["sagittal"]
    assert win.id == "sagittal"
    assert win.kind == WindowKind.TWO_D


@responses.activate
def test_view_direction_always_hits_network() -> None:
    # view_direction is not cached: stdmulti planes can be tilted to a
    # custom orientation and MxN cells can re-bind to a different plane,
    # so each access must reflect live server state. Two consecutive reads
    # therefore produce two GETs even when the previous one returned a
    # value, and a later tilt is observed without re-fetching the handle.
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/axial"),
        json={
            "id": "axial",
            "kind": "2d",
            "view_direction": "axial",
            "has_camera": True,
            "has_selected_slice": True,
        },
    )
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/windows/axial"),
        json={
            "id": "axial",
            "kind": "2d",
            "view_direction": "original",
            "has_camera": True,
            "has_selected_slice": True,
        },
    )
    win = StdMultiEditor(_transport())["axial"]
    assert win.view_direction == ViewDirection.AXIAL
    assert win.view_direction == ViewDirection.ORIGINAL  # type: ignore[comparison-overlap]
    assert len(responses.calls) == 2
