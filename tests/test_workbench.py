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

"""Tests for workbench.py."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
import responses

from mitk_workbench_remote import errors
from mitk_workbench_remote.image import Image
from mitk_workbench_remote.node import DataNode
from mitk_workbench_remote.storage import DataStorage
from mitk_workbench_remote.transport import RestTransport
from mitk_workbench_remote.workbench import (
    ReinitMode,
    SelectedPosition,
    SelectedTime,
    TimeBounds,
    Workbench,
    WorkbenchInfo,
    connect,
)

BASE = "http://127.0.0.1:8080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api(path: str) -> str:
    return f"{BASE}/api/v1{path}"


def _root_response(
    *,
    name: str = "MITK Workbench",
    api_version: str = "v1",
    mitk_version: str = "2024.06",
    transfer_modes: list[str] | None = None,
) -> dict:
    if transfer_modes is None:
        transfer_modes = ["direct"]
    return {
        "data": {
            "name": name,
            "api_version": api_version,
            "mitk_version": mitk_version,
            "capabilities": {"transfer_modes": transfer_modes},
        }
    }


def _make_transport(transfer_mode: str = "direct") -> RestTransport:
    return RestTransport(BASE, token="test-token", transfer_mode=transfer_mode)


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


def test_connect_returns_workbench() -> None:
    wb = connect(BASE)
    assert isinstance(wb, Workbench)
    wb.close()


@responses.activate
def test_connect_no_http_on_construction() -> None:
    wb = connect(BASE)
    assert len(responses.calls) == 0
    wb.close()


# ---------------------------------------------------------------------------
# url
# ---------------------------------------------------------------------------


def test_url_returns_base_url() -> None:
    t = _make_transport()
    wb = Workbench(t)
    assert wb.url == BASE
    wb.close()


# ---------------------------------------------------------------------------
# is_launched_remotely
# ---------------------------------------------------------------------------


def test_is_launched_remotely_false_when_connected() -> None:
    t = _make_transport()
    wb = Workbench(t)
    assert wb.is_launched_remotely is False
    wb.close()


def test_is_launched_remotely_true_when_process_given() -> None:
    mock_process: MagicMock = MagicMock(spec=subprocess.Popen)
    t = _make_transport()
    wb = Workbench(t, process=mock_process)  # type: ignore[arg-type]
    assert wb.is_launched_remotely is True
    wb.close()


def test_connect_is_not_launched_remotely() -> None:
    wb = connect(BASE)
    assert wb.is_launched_remotely is False
    wb.close()


# ---------------------------------------------------------------------------
# ping()
# ---------------------------------------------------------------------------


@responses.activate
def test_ping_returns_true_when_healthy() -> None:
    responses.add(responses.GET, _api("/health"), json={"data": {"status": "healthy"}}, status=200)
    t = _make_transport()
    wb = Workbench(t)
    assert wb.ping() is True
    wb.close()


@responses.activate
def test_ping_returns_false_on_connection_error() -> None:
    responses.add(
        responses.GET,
        _api("/health"),
        body=requests.exceptions.ConnectionError("refused"),
    )
    t = _make_transport()
    wb = Workbench(t)
    assert wb.ping() is False
    wb.close()


@responses.activate
def test_ping_returns_false_on_server_error() -> None:
    responses.add(
        responses.GET,
        _api("/health"),
        json={"error": {"code": "INTERNAL", "message": "boom"}},
        status=500,
    )
    t = _make_transport()
    wb = Workbench(t)
    assert wb.ping() is False
    wb.close()


@responses.activate
def test_ping_never_raises() -> None:
    # Even unexpected exceptions must be swallowed
    responses.add(
        responses.GET,
        _api("/health"),
        body=ValueError("unexpected"),
    )
    t = _make_transport()
    wb = Workbench(t)
    result = wb.ping()  # Must not raise
    assert result is False
    wb.close()


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------


@responses.activate
def test_is_connected_true() -> None:
    responses.add(responses.GET, _api("/health"), json={"data": {"status": "healthy"}}, status=200)
    t = _make_transport()
    wb = Workbench(t)
    assert wb.is_connected is True
    wb.close()


@responses.activate
def test_is_connected_false() -> None:
    responses.add(
        responses.GET,
        _api("/health"),
        body=requests.exceptions.ConnectionError("refused"),
    )
    t = _make_transport()
    wb = Workbench(t)
    assert wb.is_connected is False
    wb.close()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@responses.activate
def test_info_returns_workbench_info() -> None:
    responses.add(responses.GET, _api("/"), json=_root_response(), status=200)
    t = _make_transport()
    wb = Workbench(t)
    info = wb.info
    assert isinstance(info, WorkbenchInfo)
    assert info.name == "MITK Workbench"
    assert info.api_version == "v1"
    assert info.mitk_version == "2024.06"
    wb.close()


@responses.activate
def test_info_includes_url() -> None:
    responses.add(responses.GET, _api("/"), json=_root_response(), status=200)
    t = _make_transport()
    wb = Workbench(t)
    assert wb.info.url == BASE
    wb.close()


@responses.activate
def test_info_cached_after_first_access() -> None:
    responses.add(responses.GET, _api("/"), json=_root_response(), status=200)
    t = _make_transport()
    wb = Workbench(t)
    _ = wb.info
    _ = wb.info
    _ = wb.info
    root_calls = [c for c in responses.calls if c.request.url.endswith("/api/v1/")]
    assert len(root_calls) == 1
    wb.close()


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


def test_storage_returns_datastorage_instance() -> None:
    t = _make_transport()
    wb = Workbench(t)
    assert isinstance(wb.storage, DataStorage)
    wb.close()


def test_storage_cached() -> None:
    t = _make_transport()
    wb = Workbench(t)
    s1 = wb.storage
    s2 = wb.storage
    assert s1 is s2
    wb.close()


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


def test_shutdown_raises_for_connected_instance() -> None:
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(errors.MitkError):
        wb.shutdown()
    wb.close()


def test_shutdown_terminates_process() -> None:
    mock_process: MagicMock = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 12345
    mock_process.communicate.return_value = (b"", b"")
    t = _make_transport()
    wb = Workbench(t, process=mock_process)  # type: ignore[arg-type]
    with patch("subprocess.run"):
        wb.shutdown()
    mock_process.communicate.assert_called()
    wb.close()


def test_shutdown_kills_if_communicate_times_out() -> None:
    mock_process: MagicMock = MagicMock(spec=subprocess.Popen)
    mock_process.pid = 12345
    mock_process.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="mitk", timeout=10),
        (b"", b""),  # after kill()
    ]
    t = _make_transport()
    wb = Workbench(t, process=mock_process)  # type: ignore[arg-type]
    with patch("subprocess.run"):
        wb.shutdown()
    mock_process.kill.assert_called_once()
    wb.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_closes_transport() -> None:
    t = _make_transport()
    mock_close = MagicMock()
    with patch.object(t, "close", mock_close):
        wb = Workbench(t)
        wb.close()
    mock_close.assert_called_once()


def test_context_manager_closes_on_exit() -> None:
    t = _make_transport()
    mock_close = MagicMock()
    with patch.object(t, "close", mock_close), Workbench(t):
        pass
    mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_includes_url() -> None:
    t = _make_transport()
    wb = Workbench(t)
    assert BASE in repr(wb)
    assert "Workbench" in repr(wb)
    wb.close()


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


@responses.activate
def test_update_posts_to_rendering_update_endpoint() -> None:
    responses.add(responses.POST, _api("/rendering/update"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.update()
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == _api("/rendering/update")
    wb.close()


@responses.activate
def test_update_sends_type_all_by_default() -> None:
    responses.add(responses.POST, _api("/rendering/update"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.update()
    body = json.loads(responses.calls[0].request.body)
    assert body == {"type": "all"}
    wb.close()


@responses.activate
def test_update_sends_custom_windows_type() -> None:
    responses.add(responses.POST, _api("/rendering/update"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.update(windows="2d")
    body = json.loads(responses.calls[0].request.body)
    assert body == {"type": "2d"}
    wb.close()


# ---------------------------------------------------------------------------
# reinit()
# ---------------------------------------------------------------------------


@responses.activate
def test_reinit_global_posts_empty_body() -> None:
    responses.add(responses.POST, _api("/rendering/reinit"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.reinit()
    body = json.loads(responses.calls[0].request.body)
    assert body == {}
    wb.close()


@responses.activate
def test_reinit_with_uid_strings_sends_uids() -> None:
    responses.add(responses.POST, _api("/rendering/reinit"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.reinit(["node_1", "node_2"])
    body = json.loads(responses.calls[0].request.body)
    assert body == {"uids": ["node_1", "node_2"]}
    wb.close()


@responses.activate
def test_reinit_with_data_nodes_extracts_uids() -> None:
    responses.add(responses.POST, _api("/rendering/reinit"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    node = DataNode._from_node_dict(
        {"uid": "node_1", "name": "CT", "path": "/CT", "parent_uid": None, "data_type": "Image"},
        t,
    )
    wb.reinit([node])
    body = json.loads(responses.calls[0].request.body)
    assert body == {"uids": ["node_1"]}
    wb.close()


@responses.activate
def test_reinit_with_mixed_list() -> None:
    responses.add(responses.POST, _api("/rendering/reinit"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    node = DataNode._from_node_dict(
        {"uid": "node_1", "name": "CT", "path": "/CT", "parent_uid": None, "data_type": "Image"},
        t,
    )
    wb.reinit([node, "node_2"])
    body = json.loads(responses.calls[0].request.body)
    assert body == {"uids": ["node_1", "node_2"]}
    wb.close()


def test_reinit_raises_value_error_for_empty_list() -> None:
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(ValueError):
        wb.reinit([])
    wb.close()


@responses.activate
def test_reinit_raises_node_not_found() -> None:
    responses.add(
        responses.POST,
        _api("/rendering/reinit"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "not found"}},
        status=404,
    )
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(errors.NodeNotFoundError):
        wb.reinit(["ghost_node"])
    wb.close()


@responses.activate
def test_reinit_raises_rendering_error() -> None:
    responses.add(
        responses.POST,
        _api("/rendering/reinit"),
        json={"error": {"code": "RENDERING_ERROR", "message": "render pipeline failure"}},
        status=422,
    )
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(errors.RenderingError):
        wb.reinit()
    wb.close()


# ---------------------------------------------------------------------------
# Helpers for show() tests
# ---------------------------------------------------------------------------

_NODE_RESP = {
    "data": {
        "uid": "new-uid",
        "name": "test",
        "path": "/test",
        "parent_uid": None,
        "data_type": None,
        "children_count": 0,
        "timestamp": "2026-01-01T00:00:00Z",
    }
}

_NODE_RESP_WITH_TYPE = {
    "data": {
        "uid": "new-uid",
        "name": "test",
        "path": "/test",
        "parent_uid": None,
        "data_type": "Image",
        "children_count": 0,
        "timestamp": "2026-01-01T00:00:00Z",
    }
}


def _stub_create_and_upload() -> None:
    """Register stubs for node create, data PUT, props PATCH, reinit, refresh."""
    # create node
    responses.add(responses.POST, _api("/datastorage/nodes"), json=_NODE_RESP, status=201)
    # data upload (direct)
    responses.add(responses.PUT, _api("/datastorage/nodes/new-uid/data"), body=b"", status=204)
    # properties PATCH
    responses.add(
        responses.PATCH, _api("/datastorage/nodes/new-uid/properties"), body=b"", status=204
    )
    # reinit
    responses.add(responses.POST, _api("/rendering/reinit"), body=b"", status=204)
    # refresh (for file uploads)
    responses.add(responses.GET, _api("/datastorage/nodes/new-uid"), json=_NODE_RESP_WITH_TYPE)


# ---------------------------------------------------------------------------
# show() tests
# ---------------------------------------------------------------------------


@responses.activate
def test_show_creates_node_uploads_data_sets_properties() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    node = wb.show(img, name="MyImage")

    assert node.uid == "new-uid"
    # Verify create was called
    create_call = responses.calls[0]
    assert create_call.request.url == _api("/datastorage/nodes")
    create_body = json.loads(create_call.request.body)
    assert create_body["name"] == "MyImage"
    wb.close()


@responses.activate
def test_show_infers_name_from_image() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img)

    create_body = json.loads(responses.calls[0].request.body)
    assert create_body["name"] == "Image"
    wb.close()


@responses.activate
def test_show_infers_name_from_ndarray() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    wb.show(np.zeros((2, 3, 4), dtype=np.float32))

    create_body = json.loads(responses.calls[0].request.body)
    assert create_body["name"] == "Array"
    wb.close()


@responses.activate
def test_show_infers_name_from_path(tmp_path: Path) -> None:
    _stub_create_and_upload()
    f = tmp_path / "my_scan.nrrd"
    f.write_bytes(b"fake-nrrd-data")
    t = _make_transport()
    wb = Workbench(t)
    wb.show(f)

    create_body = json.loads(responses.calls[0].request.body)
    assert create_body["name"] == "my_scan"
    wb.close()


@responses.activate
def test_show_cleans_up_orphan_on_unsupported_type() -> None:
    _stub_create_and_upload()
    # show() must DELETE the created node when set_data raises TypeError
    responses.add(responses.DELETE, _api("/datastorage/nodes/new-uid"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)

    with pytest.raises(TypeError):
        wb.show(42)

    create_body = json.loads(responses.calls[0].request.body)
    assert create_body["name"] == "int"
    # Verify the cleanup DELETE was issued
    delete_calls = [c for c in responses.calls if c.request.method == "DELETE"]
    assert len(delete_calls) == 1
    wb.close()


@responses.activate
def test_show_uses_explicit_name() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="CustomName")

    create_body = json.loads(responses.calls[0].request.body)
    assert create_body["name"] == "CustomName"
    wb.close()


@responses.activate
def test_show_hide_others() -> None:
    # list returns two existing nodes
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json={
            "data": [
                {
                    "uid": "a",
                    "name": "A",
                    "path": "/A",
                    "parent_uid": None,
                    "data_type": "Image",
                    "children_count": 0,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
                {
                    "uid": "b",
                    "name": "B",
                    "path": "/B",
                    "parent_uid": None,
                    "data_type": "Image",
                    "children_count": 0,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            ],
            "meta": {"total_count": 2},
        },
    )
    # hide calls: PATCH properties for a and b
    responses.add(responses.PATCH, _api("/datastorage/nodes/a/properties"), body=b"", status=204)
    responses.add(responses.PATCH, _api("/datastorage/nodes/b/properties"), body=b"", status=204)
    _stub_create_and_upload()

    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="New", hide_others=True)

    # Check that visibility was set to False on existing nodes
    hide_a = responses.calls[1]
    assert "/datastorage/nodes/a/properties" in hide_a.request.url
    assert json.loads(hide_a.request.body)["visible"] is False

    hide_b = responses.calls[2]
    assert "/datastorage/nodes/b/properties" in hide_b.request.url
    assert json.loads(hide_b.request.body)["visible"] is False
    wb.close()


@responses.activate
def test_show_reinit_all_by_default() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="Test")

    # Find reinit call
    reinit_calls = [c for c in responses.calls if "/rendering/reinit" in c.request.url]
    assert len(reinit_calls) == 1
    body = json.loads(reinit_calls[0].request.body)
    assert body == {}  # global reinit
    wb.close()


@responses.activate
def test_show_reinit_node_only() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="Test", reinit=ReinitMode.NODE)

    reinit_calls = [c for c in responses.calls if "/rendering/reinit" in c.request.url]
    assert len(reinit_calls) == 1
    body = json.loads(reinit_calls[0].request.body)
    assert body == {"uids": ["new-uid"]}
    wb.close()


@responses.activate
def test_show_no_reinit() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="Test", reinit=ReinitMode.NONE)

    reinit_calls = [c for c in responses.calls if "/rendering/reinit" in c.request.url]
    assert len(reinit_calls) == 0
    wb.close()


@responses.activate
def test_show_passes_parent() -> None:
    # create under parent
    responses.add(
        responses.POST,
        _api("/datastorage/nodes/parent-uid/children"),
        json=_NODE_RESP,
        status=201,
    )
    responses.add(responses.PUT, _api("/datastorage/nodes/new-uid/data"), body=b"", status=204)
    responses.add(
        responses.PATCH, _api("/datastorage/nodes/new-uid/properties"), body=b"", status=204
    )
    responses.add(responses.POST, _api("/rendering/reinit"), body=b"", status=204)

    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="Child", parent="parent-uid")

    assert "/datastorage/nodes/parent-uid/children" in responses.calls[0].request.url
    wb.close()


@responses.activate
def test_show_sets_opacity_and_color() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="Test", opacity=0.5, color=(1.0, 0.0, 0.0))

    # Find PATCH call for the new node
    patch_calls = [
        c
        for c in responses.calls
        if "new-uid/properties" in c.request.url and c.request.method == "PATCH"
    ]
    assert len(patch_calls) == 1
    body = json.loads(patch_calls[0].request.body)
    assert body["visible"] is True
    assert body["opacity"] == 0.5
    assert body["color"] == {"type": "ColorProperty", "value": [1.0, 0.0, 0.0]}
    wb.close()


@responses.activate
def test_show_omits_default_opacity() -> None:
    _stub_create_and_upload()
    t = _make_transport()
    wb = Workbench(t)
    img = Image(np.zeros((2, 3, 4), dtype=np.float32))
    wb.show(img, name="Test")

    patch_calls = [
        c
        for c in responses.calls
        if "new-uid/properties" in c.request.url and c.request.method == "PATCH"
    ]
    assert len(patch_calls) == 1
    body = json.loads(patch_calls[0].request.body)
    assert "opacity" not in body
    assert "color" not in body
    wb.close()


@responses.activate
def test_show_file_path_uploads_raw_bytes(tmp_path: Path) -> None:
    _stub_create_and_upload()
    f = tmp_path / "scan.nrrd"
    f.write_bytes(b"fake-nrrd-bytes")

    t = _make_transport()
    wb = Workbench(t)
    wb.show(f, name="Scan")

    # Find the PUT data call
    put_calls = [
        c for c in responses.calls if "/data" in c.request.url and c.request.method == "PUT"
    ]
    assert len(put_calls) == 1
    assert put_calls[0].request.body == b"fake-nrrd-bytes"
    wb.close()


@responses.activate
def test_show_file_path_file_reference_mode(tmp_path: Path) -> None:
    _stub_create_and_upload()
    f = tmp_path / "scan.nrrd"
    f.write_bytes(b"fake-nrrd-bytes")

    t = _make_transport(transfer_mode="file-reference")
    wb = Workbench(t)
    wb.show(f, name="Scan")

    put_calls = [
        c for c in responses.calls if "/data" in c.request.url and c.request.method == "PUT"
    ]
    assert len(put_calls) == 1
    body = json.loads(put_calls[0].request.body)
    assert body["transfer"]["mode"] == "file-reference"
    assert body["transfer"]["file_path"] == str(f.resolve())
    wb.close()


# ---------------------------------------------------------------------------
# get_position / set_position
# ---------------------------------------------------------------------------


@responses.activate
def test_get_position_returns_dataclass() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/selected-position"),
        json={
            "position": [10.0, 20.0, 30.0],
            "bounds": {"min": [-50.0, -50.0, -50.0], "max": [50.0, 50.0, 50.0]},
        },
    )
    t = _make_transport()
    wb = Workbench(t)
    pos = wb.get_position()
    assert isinstance(pos, SelectedPosition)
    assert pos.position == (10.0, 20.0, 30.0)
    assert pos.bounds.min == (-50.0, -50.0, -50.0)
    assert pos.bounds.max == (50.0, 50.0, 50.0)
    wb.close()


@responses.activate
def test_get_position_with_null_bounds() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/selected-position"),
        json={
            "position": [0.0, 0.0, 0.0],
            "bounds": {"min": None, "max": None},
        },
    )
    t = _make_transport()
    wb = Workbench(t)
    pos = wb.get_position()
    assert pos.bounds.min is None
    assert pos.bounds.max is None
    wb.close()


@responses.activate
def test_set_position_sends_put() -> None:
    responses.add(responses.PUT, _api("/rendering/selected-position"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.set_position((10.0, 20.0, 30.0))
    body = json.loads(responses.calls[0].request.body)
    assert body == {"position": [10.0, 20.0, 30.0]}
    wb.close()


def test_set_position_raises_for_wrong_length() -> None:
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(ValueError, match="exactly 3 elements"):
        wb.set_position([1.0, 2.0])
    wb.close()


# ---------------------------------------------------------------------------
# get_time / set_timepoint / set_timestep
# ---------------------------------------------------------------------------

_TIME_RESPONSE = {
    "timepoint_ms": 1500.0,
    "timestep": 3,
    "bounds": {"min_timepoint_ms": 0.0, "max_timepoint_ms": 4500.0, "steps": 10},
}


@responses.activate
def test_get_time_returns_dataclass() -> None:
    responses.add(responses.GET, _api("/rendering/selected-time"), json=_TIME_RESPONSE)
    t = _make_transport()
    wb = Workbench(t)
    time = wb.get_time()
    assert isinstance(time, SelectedTime)
    assert time.timepoint_ms == 1500.0
    assert time.timestep == 3
    assert isinstance(time.bounds, TimeBounds)
    assert time.bounds.min_timepoint_ms == 0.0
    assert time.bounds.max_timepoint_ms == 4500.0
    assert time.bounds.steps == 10
    wb.close()


@responses.activate
def test_set_timepoint_sends_put() -> None:
    responses.add(responses.PUT, _api("/rendering/selected-time"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.set_timepoint(1500.0)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"timepoint_ms": 1500.0}
    wb.close()


@responses.activate
def test_set_timestep_sends_put() -> None:
    responses.add(responses.PUT, _api("/rendering/selected-time"), body=b"", status=204)
    t = _make_transport()
    wb = Workbench(t)
    wb.set_timestep(3)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"timestep": 3}
    wb.close()


@responses.activate
def test_get_timepoint_returns_float() -> None:
    responses.add(responses.GET, _api("/rendering/selected-time"), json=_TIME_RESPONSE)
    t = _make_transport()
    wb = Workbench(t)
    assert wb.get_timepoint() == 1500.0
    wb.close()


@responses.activate
def test_get_timestep_returns_int() -> None:
    responses.add(responses.GET, _api("/rendering/selected-time"), json=_TIME_RESPONSE)
    t = _make_transport()
    wb = Workbench(t)
    assert wb.get_timestep() == 3
    wb.close()


# ---------------------------------------------------------------------------
# screenshot
# ---------------------------------------------------------------------------


@responses.activate
def test_screenshot_returns_bytes() -> None:
    png_data = b"\x89PNG\r\n\x1a\nfake"
    responses.add(
        responses.GET,
        _api("/rendering/screenshot"),
        body=png_data,
        content_type="image/png",
    )
    t = _make_transport()
    wb = Workbench(t)
    result = wb.screenshot()
    assert result == png_data
    wb.close()


@responses.activate
def test_screenshot_with_dimensions() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/screenshot"),
        body=b"img",
        content_type="image/png",
    )
    t = _make_transport()
    wb = Workbench(t)
    wb.screenshot(width=800, height=600)
    url = responses.calls[0].request.url
    assert "width=800" in url
    assert "height=600" in url
    wb.close()


def test_screenshot_raises_if_only_width() -> None:
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(ValueError, match="width and height must both"):
        wb.screenshot(width=800)
    wb.close()


def test_screenshot_raises_if_only_height() -> None:
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(ValueError, match="width and height must both"):
        wb.screenshot(height=600)
    wb.close()


@responses.activate
def test_screenshot_saves_to_path(tmp_path: Path) -> None:
    png_data = b"\x89PNG\r\n\x1a\nfake"
    responses.add(
        responses.GET,
        _api("/rendering/screenshot"),
        body=png_data,
        content_type="image/png",
    )
    t = _make_transport()
    wb = Workbench(t)
    dest = tmp_path / "shot.png"
    result = wb.screenshot(path=dest)
    assert result == png_data
    assert dest.read_bytes() == png_data
    wb.close()


# ---------------------------------------------------------------------------
# RENDER_WINDOW_NOT_AVAILABLE error mapping
# ---------------------------------------------------------------------------


@responses.activate
def test_render_window_not_available_raises_rendering_error() -> None:
    responses.add(
        responses.GET,
        _api("/rendering/selected-position"),
        json={
            "error": {
                "code": "RENDER_WINDOW_NOT_AVAILABLE",
                "message": "StdMultiWidgetEditor is not open",
            }
        },
        status=503,
    )
    t = _make_transport()
    wb = Workbench(t)
    with pytest.raises(errors.RenderingError):
        wb.get_position()
    wb.close()
