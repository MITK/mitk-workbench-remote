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

"""Tests for transport.py."""

from unittest.mock import MagicMock, patch

import pytest
import responses

from mitk_workbench_remote import errors
from mitk_workbench_remote.transport import (
    FileAccessConfig,
    FileAccessMode,
    RestTransport,
    ServerInfo,
    TransferMode,
)

BASE = "http://127.0.0.1:8080"
BASE_REMOTE = "http://192.168.1.100:8080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api(path: str, base: str = BASE) -> str:
    return f"{base}/api/v1{path}"


def _root_response(transfer_modes: list[str]) -> dict:
    """Build a minimal GET / response body."""
    return {
        "data": {
            "name": "MITK Workbench",
            "api_version": "v1",
            "mitk_version": "2024.06",
            "capabilities": {"transfer_modes": transfer_modes},
        }
    }


def _file_access_response(
    *,
    mode: FileAccessMode | str = FileAccessMode.UNRESTRICTED,
    restrictions_active: bool = False,
    max_active_temp_dirs_per_ip: int = 5,
    allowed_paths: list[str] | None = None,
) -> dict:
    """Build a minimal GET /config/file-access response body."""
    body: dict = {
        "data": {
            "mode": mode,
            "restrictions_active": restrictions_active,
            "max_active_temp_dirs_per_ip": max_active_temp_dirs_per_ip,
        }
    }
    if allowed_paths is not None:
        body["data"]["allowed_paths"] = allowed_paths
    return body


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_base_url_trailing_slash_stripped() -> None:
    with RestTransport("http://localhost:8080/") as t:
        assert t.base_url == "http://localhost:8080"


def test_auth_header_set_when_token_given() -> None:
    with RestTransport(BASE, token="secret") as t:
        assert t._session.headers["X-MITK-API-Token"] == "secret"


def test_no_auth_header_without_token() -> None:
    with RestTransport(BASE) as t:
        assert "X-MITK-API-Token" not in t._session.headers


def test_default_timeout() -> None:
    with RestTransport(BASE) as t:
        assert t._timeout == 30.0


def test_custom_timeout() -> None:
    with RestTransport(BASE, timeout=5.0) as t:
        assert t._timeout == 5.0


@responses.activate
def test_explicit_transfer_mode_not_auto_detected() -> None:
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        # Accessing the property should NOT call the server
        assert t.transfer_mode == TransferMode.DIRECT
        assert len(responses.calls) == 0


# ---------------------------------------------------------------------------
# HTTP methods — happy path
# ---------------------------------------------------------------------------


@responses.activate
def test_get_success() -> None:
    responses.add(responses.GET, _api("/nodes"), json={"nodes": []}, status=200)
    with RestTransport(BASE) as t:
        r = t.get("/nodes")
        assert r.status_code == 200
        assert r.ok
        assert r.json() == {"nodes": []}


@responses.activate
def test_post_success() -> None:
    responses.add(responses.POST, _api("/nodes"), json={"uid": "abc"}, status=201)
    with RestTransport(BASE) as t:
        r = t.post("/nodes", json={"name": "test"})
        assert r.status_code == 201


@responses.activate
def test_put_success() -> None:
    responses.add(responses.PUT, _api("/nodes/abc"), json={}, status=200)
    with RestTransport(BASE) as t:
        r = t.put("/nodes/abc", json={"visible": True})
        assert r.status_code == 200


@responses.activate
def test_patch_success() -> None:
    responses.add(responses.PATCH, _api("/nodes/abc"), json={}, status=200)
    with RestTransport(BASE) as t:
        r = t.patch("/nodes/abc", json={"name": "new"})
        assert r.status_code == 200


@responses.activate
def test_delete_success() -> None:
    responses.add(responses.DELETE, _api("/nodes/abc"), body=b"", status=204)
    with RestTransport(BASE) as t:
        r = t.delete("/nodes/abc")
        assert r.status_code == 204


# ---------------------------------------------------------------------------
# Binary methods
# ---------------------------------------------------------------------------


@responses.activate
def test_get_binary_returns_rest_response_with_content() -> None:
    payload = b"\x00\x01\x02\x03"
    responses.add(responses.GET, _api("/nodes/abc/data"), body=payload, status=200)
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        r = t.get_binary("/nodes/abc/data")
        assert r.content == payload


@responses.activate
def test_get_binary_sets_transfer_mode_header() -> None:
    responses.add(responses.GET, _api("/nodes/abc/data"), body=b"data", status=200)
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        t.get_binary("/nodes/abc/data")
        assert (
            responses.calls[0].request.headers.get("X-MITK-Transfer-Mode") == TransferMode.DIRECT
        )


@responses.activate
def test_get_binary_transfer_mode_header_can_be_overridden() -> None:
    responses.add(responses.GET, _api("/nodes/abc/data"), body=b"data", status=200)
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        t.get_binary(
            "/nodes/abc/data", headers={"X-MITK-Transfer-Mode": TransferMode.FILE_REFERENCE}
        )
        assert (
            responses.calls[0].request.headers.get("X-MITK-Transfer-Mode")
            == TransferMode.FILE_REFERENCE
        )


@responses.activate
def test_get_binary_does_not_mutate_caller_headers() -> None:
    responses.add(responses.GET, _api("/nodes/abc/data"), body=b"data", status=200)
    caller_headers = {"X-Custom": "value"}
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        t.get_binary("/nodes/abc/data", headers=caller_headers)
    assert "X-MITK-Transfer-Mode" not in caller_headers


@responses.activate
def test_put_binary_sends_bytes() -> None:
    responses.add(responses.PUT, _api("/nodes/abc/data"), json={}, status=200)
    with RestTransport(BASE) as t:
        r = t.put_binary("/nodes/abc/data", b"\xff\xfe")
        assert r.status_code == 200
        assert responses.calls[0].request.body == b"\xff\xfe"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@responses.activate
def test_404_node_not_found_raises_NodeNotFoundError() -> None:
    responses.add(
        responses.GET,
        _api("/nodes/xyz"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "no such node"}},
        status=404,
    )
    with RestTransport(BASE) as t, pytest.raises(errors.NodeNotFoundError):
        t.get("/nodes/xyz")


@responses.activate
def test_404_without_node_not_found_code_raises_ApiError() -> None:
    responses.add(
        responses.GET,
        _api("/config/file-access"),
        json={"error": {"code": "NOT_FOUND", "message": "endpoint missing"}},
        status=404,
    )
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        with pytest.raises(errors.ApiError) as exc_info:
            t.get("/config/file-access")
        assert exc_info.value.status_code == 404


@responses.activate
def test_503_raises_DataStorageNotAvailableError() -> None:
    responses.add(
        responses.GET,
        _api("/nodes"),
        json={"error": {"code": "DATASTORAGE_NOT_AVAILABLE", "message": "storage gone"}},
        status=503,
    )
    with RestTransport(BASE) as t, pytest.raises(errors.DataStorageNotAvailableError):
        t.get("/nodes")


@responses.activate
def test_401_raises_AuthenticationError() -> None:
    responses.add(
        responses.GET,
        _api("/nodes"),
        json={"error": {"code": "UNAUTHORIZED", "message": "token required"}},
        status=401,
    )
    with RestTransport(BASE) as t:
        with pytest.raises(errors.AuthenticationError) as exc_info:
            t.get("/nodes")
        assert exc_info.value.status_code == 401


@responses.activate
def test_403_raises_AuthenticationError() -> None:
    responses.add(
        responses.GET,
        _api("/nodes"),
        json={"error": {"code": "ACCESS_DENIED", "message": "access denied"}},
        status=403,
    )
    with RestTransport(BASE) as t:
        with pytest.raises(errors.AuthenticationError) as exc_info:
            t.get("/nodes")
        assert exc_info.value.status_code == 403


@responses.activate
def test_400_raises_ApiError() -> None:
    responses.add(
        responses.POST,
        _api("/nodes"),
        json={"error": {"code": "INVALID_REQUEST", "message": "bad input"}},
        status=400,
    )
    with RestTransport(BASE) as t:
        with pytest.raises(errors.ApiError) as exc_info:
            t.post("/nodes")
        err = exc_info.value
        assert err.status_code == 400
        assert err.code == "INVALID_REQUEST"
        assert err.message == "bad input"


@responses.activate
def test_415_raises_TransferError() -> None:
    responses.add(
        responses.PUT,
        _api("/nodes/abc/data"),
        json={"error": {"code": "UNSUPPORTED_FORMAT", "message": "format not supported"}},
        status=415,
    )
    with RestTransport(BASE) as t, pytest.raises(errors.TransferError):
        t.put("/nodes/abc/data")


@responses.activate
def test_406_transfer_mode_not_available_raises_TransferError() -> None:
    responses.add(
        responses.GET,
        _api("/nodes/abc/data"),
        json={
            "error": {
                "code": "TRANSFER_MODE_NOT_AVAILABLE",
                "message": "file-reference not supported",
            }
        },
        status=406,
    )
    with (
        RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t,
        pytest.raises(errors.TransferError),
    ):
        t.get_binary("/nodes/abc/data")


@responses.activate
def test_unknown_4xx_raises_ApiError() -> None:
    responses.add(
        responses.GET,
        _api("/nodes"),
        json={"error": {"code": "SOMETHING_WEIRD", "message": "unexpected"}},
        status=422,
    )
    with RestTransport(BASE) as t:
        with pytest.raises(errors.ApiError) as exc_info:
            t.get("/nodes")
        assert exc_info.value.status_code == 422


@responses.activate
def test_422_rendering_error_raises_RenderingError() -> None:
    responses.add(
        responses.POST,
        _api("/rendering/update"),
        json={"error": {"code": "RENDERING_ERROR", "message": "render pipeline failure"}},
        status=422,
    )
    t = RestTransport(BASE)
    with pytest.raises(errors.RenderingError):
        t.post("/rendering/update")
    t.close()


@responses.activate
def test_connection_error_raises_ConnectionError() -> None:
    import requests as req

    responses.add(
        responses.GET,
        _api("/nodes"),
        body=req.exceptions.ConnectionError("refused"),
    )
    with RestTransport(BASE) as t, pytest.raises(errors.MitkConnectionError):
        t.get("/nodes")


@responses.activate
def test_malformed_error_body_raises_ApiError() -> None:
    responses.add(
        responses.GET,
        _api("/nodes"),
        body=b"not json at all",
        status=500,
        content_type="text/plain",
    )
    with RestTransport(BASE) as t:
        with pytest.raises(errors.ApiError) as exc_info:
            t.get("/nodes")
        err = exc_info.value
        assert err.status_code == 500
        assert err.code == "UNKNOWN"


# ---------------------------------------------------------------------------
# ServerInfo — lazy property
# ---------------------------------------------------------------------------


@responses.activate
def test_server_info_parsed_correctly() -> None:
    responses.add(
        responses.GET,
        _api("/"),
        json=_root_response(["direct", "file-reference"]),
        status=200,
    )
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        info = t.server_info
        assert isinstance(info, ServerInfo)
        assert info.name == "MITK Workbench"
        assert info.api_version == "v1"
        assert info.mitk_version == "2024.06"
        assert info.transfer_modes == ("direct", "file-reference")


@responses.activate
def test_server_info_cached_after_first_access() -> None:
    responses.add(
        responses.GET,
        _api("/"),
        json=_root_response(["direct"]),
        status=200,
    )
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        _ = t.server_info
        _ = t.server_info
        _ = t.server_info
        root_calls = [c for c in responses.calls if c.request.url.endswith("/api/v1/")]
        assert len(root_calls) == 1


@responses.activate
def test_server_info_missing_capabilities_defaults_to_empty_tuple() -> None:
    responses.add(
        responses.GET,
        _api("/"),
        json={"data": {"name": "MITK", "api_version": "v1", "mitk_version": "2024"}},
        status=200,
    )
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        info = t.server_info
        assert info.transfer_modes == ()


# ---------------------------------------------------------------------------
# FileAccessConfig — lazy property
# ---------------------------------------------------------------------------


@responses.activate
def test_file_access_config_parsed_unrestricted() -> None:
    responses.add(
        responses.GET,
        _api("/config/file-access"),
        json=_file_access_response(max_active_temp_dirs_per_ip=10),
        status=200,
    )
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        cfg = t.file_access_config
        assert isinstance(cfg, FileAccessConfig)
        assert cfg.mode == FileAccessMode.UNRESTRICTED
        assert cfg.restrictions_active is False
        assert cfg.max_active_temp_dirs_per_ip == 10
        assert cfg.allowed_paths == ()


@responses.activate
def test_file_access_config_parsed_restricted() -> None:
    responses.add(
        responses.GET,
        _api("/config/file-access"),
        json=_file_access_response(
            mode=FileAccessMode.ALLOWED_DIRECTORIES,
            restrictions_active=True,
            max_active_temp_dirs_per_ip=3,
            allowed_paths=["/tmp", "/data"],
        ),
        status=200,
    )
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        cfg = t.file_access_config
        assert cfg.mode == FileAccessMode.ALLOWED_DIRECTORIES
        assert cfg.restrictions_active is True
        assert cfg.max_active_temp_dirs_per_ip == 3
        assert cfg.allowed_paths == ("/tmp", "/data")


@responses.activate
def test_file_access_config_cached_after_first_access() -> None:
    responses.add(
        responses.GET,
        _api("/config/file-access"),
        json=_file_access_response(),
        status=200,
    )
    with RestTransport(BASE, transfer_mode=TransferMode.DIRECT) as t:
        _ = t.file_access_config
        _ = t.file_access_config
        _ = t.file_access_config
        calls = [c for c in responses.calls if "/config/file-access" in c.request.url]
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Transfer mode auto-detection
# ---------------------------------------------------------------------------


@responses.activate
def test_transfer_mode_file_reference_when_localhost_and_supported() -> None:
    responses.add(
        responses.GET,
        _api("/"),
        json=_root_response(["direct", "file-reference"]),
        status=200,
    )
    with RestTransport(BASE) as t:  # 127.0.0.1 → localhost
        assert t.transfer_mode == TransferMode.FILE_REFERENCE


@responses.activate
def test_transfer_mode_direct_when_server_does_not_advertise_file_reference() -> None:
    responses.add(
        responses.GET,
        _api("/"),
        json=_root_response(["direct"]),
        status=200,
    )
    with RestTransport(BASE) as t:
        assert t.transfer_mode == TransferMode.DIRECT


@responses.activate
def test_transfer_mode_remote_host() -> None:
    # Remote hosts short-circuit in _detect_transfer_mode before any server
    # query, so no HTTP stub is needed.  @responses.activate is kept so that
    # any accidental network call raises immediately rather than hanging or
    # hitting a real host.
    with RestTransport(BASE_REMOTE) as t:
        assert t.transfer_mode == TransferMode.DIRECT
        assert len(responses.calls) == 0


@responses.activate
def test_transfer_mode_detection_propagates_auth_error() -> None:
    responses.add(
        responses.GET,
        _api("/"),
        json={"error": {"code": "UNAUTHORIZED", "message": "token required"}},
        status=401,
    )
    with RestTransport(BASE) as t, pytest.raises(errors.AuthenticationError):
        _ = t.transfer_mode


@responses.activate
def test_transfer_mode_cached_after_first_access() -> None:
    responses.add(
        responses.GET,
        _api("/"),
        json=_root_response(["direct", "file-reference"]),
        status=200,
    )
    with RestTransport(BASE) as t:
        _ = t.transfer_mode
        _ = t.transfer_mode
        _ = t.transfer_mode
        # Root endpoint should have been called exactly once (server_info is shared)
        root_calls = [c for c in responses.calls if c.request.url.endswith("/api/v1/")]
        assert len(root_calls) == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_closes_session() -> None:
    t = RestTransport(BASE)
    mock_close = MagicMock()
    with patch.object(t._session, "close", mock_close):
        t.close()
    mock_close.assert_called_once()


@responses.activate
def test_context_manager() -> None:
    responses.add(responses.GET, _api("/health"), json={"status": "ok"}, status=200)
    with RestTransport(BASE) as t:
        r = t.get("/health")
        assert r.ok
