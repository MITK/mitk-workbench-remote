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

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests
import responses

from mitk_workbench_remote import errors
from mitk_workbench_remote.storage import DataStorage
from mitk_workbench_remote.transport import RestTransport
from mitk_workbench_remote.workbench import Workbench, WorkbenchInfo, connect

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
