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

"""StdMultiEditor — info, screenshot, listings, named-window properties."""

from __future__ import annotations

import responses

from mitk_workbench_remote import (
    StdMultiEditor,
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
# get_info
# ---------------------------------------------------------------------------


@responses.activate
def test_get_info_parses_editor_metadata() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti"),
        json={
            "alias": "stdmulti",
            "plugin_id": "org.mitk.editors.stdmultiwidget",
            "active": True,
            "windows": ["axial", "sagittal", "coronal", "3d"],
        },
    )
    info = StdMultiEditor(_transport()).get_info()
    assert info.alias == "stdmulti"
    assert info.plugin_id == "org.mitk.editors.stdmultiwidget"
    assert info.active is True
    assert info.windows == ("axial", "sagittal", "coronal", "3d")


# ---------------------------------------------------------------------------
# screenshot
# ---------------------------------------------------------------------------


@responses.activate
def test_editor_screenshot_returns_bytes_and_encodes_format() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/editors/stdmulti/screenshot"),
        body=b"\xff\xd8\xff",  # JPEG SOI
        status=200,
        content_type="image/jpeg",
    )
    data = StdMultiEditor(_transport()).screenshot(fmt="jpeg")
    assert data == b"\xff\xd8\xff"
    request = responses.calls[0].request
    assert "format=jpeg" in (request.url or "")


# ---------------------------------------------------------------------------
# Named-window properties — id and kind are derivable without a network call
# (kind is engine-fixed for stdmulti's named slots). view_direction is NOT
# cached: even on stdmulti the user can tilt a plane to a custom orientation,
# so the live per-window summary is authoritative.
# ---------------------------------------------------------------------------


def test_named_window_properties_have_eager_id_and_kind() -> None:
    editor = StdMultiEditor(_transport())
    assert editor.axial.id == "axial"
    assert editor.axial.kind == WindowKind.TWO_D
    assert editor.sagittal.kind == WindowKind.TWO_D
    assert editor.coronal.kind == WindowKind.TWO_D
    assert editor.three_d.id == "3d"
    assert editor.three_d.kind == WindowKind.THREE_D


@responses.activate
def test_named_window_view_direction_hits_per_window_summary() -> None:
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
    assert StdMultiEditor(_transport()).axial.view_direction == ViewDirection.AXIAL


@responses.activate
def test_three_d_view_direction_is_none_from_summary() -> None:
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
    assert StdMultiEditor(_transport()).three_d.view_direction is None
