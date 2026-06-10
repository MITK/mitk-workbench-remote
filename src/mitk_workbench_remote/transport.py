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

"""REST transport layer — HTTP abstraction over requests.

All REST calls go through RestTransport. This is the single point of change
when migrating the way we do the REST communication.

Classes:
    ServerInfo: Parsed capabilities from GET /api/v1/info.
    FileAccessConfig: Parsed response from GET /api/v1/config/file-access.
    RestTransport: Thin HTTP client with auth, timeout, and transfer mode support.
    RestResponse: Lightweight response wrapper (not leaked to users).
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any

import requests

from mitk_workbench_remote import errors

_log = logging.getLogger(__name__)


def _parse_editor_url(response: requests.Response) -> tuple[str, str, str]:
    """Extract (editor_alias, window_id, operation) from a failing editor URL.

    Looks for ``/rendering/editors/<alias>[/windows/<id>[/<operation>]]`` in
    the request URL. Missing segments come back as empty strings — callers
    populate exception fields on a best-effort basis (the RFC 7807 message
    already carries the human-readable detail).
    """
    request = response.request
    path = urllib.parse.urlparse(request.url or "").path if request is not None else ""
    parts = path.split("/")
    alias = ""
    window = ""
    operation = ""
    try:
        idx = parts.index("editors")
    except ValueError:
        return alias, window, operation
    if idx + 1 < len(parts):
        alias = parts[idx + 1]
    if idx + 3 < len(parts) and parts[idx + 2] == "windows":
        window = parts[idx + 3]
    if idx + 4 < len(parts):
        operation = parts[idx + 4]
    return alias, window, operation


class TransferMode(str, Enum):
    """Transfer modes supported by the MITK REST server."""

    DIRECT = "direct"
    FILE_REFERENCE = "file-reference"

    def __str__(self) -> str:
        return self.value


class FileAccessMode(str, Enum):
    """File-access restriction modes reported by GET /api/v1/config/file-access."""

    UNRESTRICTED = "unrestricted"
    ALLOWED_DIRECTORIES = "allowed-directories"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ServerInfo:
    """Parsed response from GET /api/v1/info.

    Args:
        name: Human-readable server name.
        api_version: API version string (e.g. ``"v1"``).
        mitk_version: Underlying MITK version string.
        transfer_modes: Transfer modes advertised by the server. Kept as plain
            strings (not ``TransferMode`` enum) so that unknown modes from future
            server versions are preserved rather than causing a ``ValueError``
            at parse time.
    """

    name: str
    api_version: str
    mitk_version: str
    transfer_modes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FileAccessConfig:
    """Parsed response from GET /api/v1/config/file-access.

    Args:
        mode: Access mode — :attr:`FileAccessMode.UNRESTRICTED` or
            :attr:`FileAccessMode.ALLOWED_DIRECTORIES`.
        restrictions_active: Whether path restrictions are enforced.
        max_active_temp_dirs_per_ip: Server-side limit on concurrent temp
            directories per client IP. Used by file-reference transfer logic
            to avoid unexpected server-side eviction.
        allowed_paths: Allowed base paths when ``restrictions_active`` is True.
    """

    mode: FileAccessMode
    restrictions_active: bool
    max_active_temp_dirs_per_ip: int
    allowed_paths: tuple[str, ...] = field(default_factory=tuple)


class RestResponse:
    """Thin wrapper around requests.Response. Never returned to users directly.

    Args:
        response: The underlying requests Response object.
    """

    def __init__(self, response: requests.Response) -> None:
        self._response = response

    @property
    def status_code(self) -> int:
        """HTTP status code."""
        return int(self._response.status_code)

    @property
    def ok(self) -> bool:
        """True for 2xx responses."""
        return bool(self._response.ok)

    @property
    def content(self) -> bytes:
        """Raw response body as bytes."""
        return bytes(self._response.content)

    @property
    def text(self) -> str:
        """Raw response body as text.

        Encoding is determined by ``requests`` from the Content-Type header or
        charset auto-detection.  Use ``.content`` for endpoints that return
        binary data.
        """
        return str(self._response.text)

    @property
    def headers(self) -> dict[str, str]:
        """Response headers as a plain dict."""
        return dict(self._response.headers)

    def json(self) -> Any:
        """Parse and return the JSON response body."""
        return self._response.json()


class RestTransport:
    """HTTP client with auth, timeout, and transfer-mode support.

    All REST calls made by the library go through this class. Nothing outside
    this module imports the underlying used dependency directly.

    Args:
        base_url: Base URL of the MITK Workbench REST server
            (e.g. ``"http://localhost:8080"``). Trailing slash is stripped.
        token: Optional API token sent as the ``Authorization: Bearer <token>`` header.
        timeout: Request timeout in seconds. Defaults to 30.
        transfer_mode: Override transfer mode (:attr:`TransferMode.DIRECT` or
            :attr:`TransferMode.FILE_REFERENCE`). When ``None`` (default) the
            mode is auto-detected on first access via server capabilities.
            A plain string is accepted for convenience; an unrecognised value
            raises ``ValueError`` immediately.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        transfer_mode: TransferMode | str | None = None,
    ) -> None:
        self._session = requests.Session()
        self._base_url = base_url.rstrip("/")
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._timeout = timeout
        self._transfer_mode: TransferMode | None = (
            TransferMode(transfer_mode) if transfer_mode is not None else None
        )
        self._server_info: ServerInfo | None = None
        self._file_access_config: FileAccessConfig | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Base URL (without trailing slash)."""
        return self._base_url

    @property
    def server_info(self) -> ServerInfo:
        """Server capabilities, fetched once from GET /api/v1/info and cached."""
        if self._server_info is None:
            data = self.get("/info").json()["data"]
            caps = data.get("capabilities", {})
            self._server_info = ServerInfo(
                name=data.get("name", ""),
                api_version=data.get("api_version", ""),
                mitk_version=data.get("mitk_version", ""),
                transfer_modes=tuple(caps.get("transfer_modes", [])),
            )
            _log.debug(
                "[%s] Server info: %s %s, transfer modes: %s",
                self._base_url,
                self._server_info.name,
                self._server_info.mitk_version,
                self._server_info.transfer_modes,
            )
        return self._server_info

    @property
    def file_access_config(self) -> FileAccessConfig:
        """File-reference configuration, fetched once from GET /config/file-access and cached."""
        if self._file_access_config is None:
            data = self.get("/config/file-access").json()["data"]
            self._file_access_config = FileAccessConfig(
                mode=FileAccessMode(data["mode"]),
                restrictions_active=data["restrictions_active"],
                max_active_temp_dirs_per_ip=data["max_active_temp_dirs_per_ip"],
                allowed_paths=tuple(data.get("allowed_paths", [])),
            )
            _log.debug(
                "[%s] File-access config: mode=%s, restrictions=%s",
                self._base_url,
                self._file_access_config.mode,
                self._file_access_config.restrictions_active,
            )
        return self._file_access_config

    @property
    def transfer_mode(self) -> TransferMode:
        """Preferred transfer mode for data requests.

        Resolution order:

        1. Explicit value set in the constructor.
        2. :attr:`TransferMode.FILE_REFERENCE` if the server advertises it in
           ``GET /api/v1/info`` capabilities AND the server is on localhost
           (filesystem access required).
        3. :attr:`TransferMode.DIRECT` otherwise.

        Cached after first resolution.
        """
        if self._transfer_mode is None:
            self._transfer_mode = self._detect_transfer_mode()
        return self._transfer_mode

    # ------------------------------------------------------------------
    # HTTP verbs
    # ------------------------------------------------------------------

    def get(self, path: str, **kwargs: Any) -> RestResponse:
        """Send a GET request."""
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> RestResponse:
        """Send a POST request."""
        return self._request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> RestResponse:
        """Send a PUT request."""
        return self._request("PUT", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> RestResponse:
        """Send a PATCH request."""
        return self._request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> RestResponse:
        """Send a DELETE request."""
        return self._request("DELETE", path, **kwargs)

    # ------------------------------------------------------------------
    # Binary helpers
    # ------------------------------------------------------------------

    def get_binary(self, path: str, **kwargs: Any) -> RestResponse:
        """Send a GET request for binary data, advertising the preferred transfer mode."""
        # Copy the caller's dict so setdefault does not mutate it as a side effect.
        headers = {**kwargs.pop("headers", {})}
        headers.setdefault("X-MITK-Transfer-Mode", self.transfer_mode)
        return self._request("GET", path, headers=headers, **kwargs)

    def put_binary(self, path: str, data: bytes, **kwargs: Any) -> RestResponse:
        """Send a PUT request with raw bytes as the body."""
        return self._request("PUT", path, data=data, **kwargs)

    def put_file_reference(self, path: str, *, file_path: str, **kwargs: Any) -> RestResponse:
        """Send a PUT with file-reference transfer mode (JSON body with file path)."""
        body = {"transfer": {"mode": "file-reference", "file_path": file_path}}
        headers = kwargs.pop("headers", {})
        headers["Content-Type"] = "application/json"
        return self._request("PUT", path, json=body, headers=headers, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying requests Session."""
        self._session.close()

    def __enter__(self) -> RestTransport:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base_url}/api/v1{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> RestResponse:
        t0 = time.monotonic()
        try:
            response = self._session.request(
                method, self._url(path), timeout=self._timeout, **kwargs
            )
        except requests.exceptions.ConnectionError as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _log.debug(
                "[%s] %s %s -> ConnectionError (%.0fms)", self._base_url, method, path, elapsed_ms
            )
            raise errors.MitkConnectionError(str(exc)) from exc
        elapsed_ms = (time.monotonic() - t0) * 1000
        _log.debug(
            "[%s] %s %s -> %d (%.0fms)",
            self._base_url,
            method,
            path,
            response.status_code,
            elapsed_ms,
        )
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> RestResponse:
        if response.ok:
            return RestResponse(response)

        # Attempt to parse RFC 7807 error body
        code: str = "UNKNOWN"
        message: str = response.text[:200]
        try:
            body = response.json()
            err = body.get("error", {})
            code = err.get("code", "UNKNOWN")
            message = err.get("message", response.text[:200])
        except (ValueError, KeyError):
            pass

        status = response.status_code

        if code == "NODE_NOT_FOUND":
            raise errors.NodeNotFoundError(uid="")
        if code == "DATASTORAGE_NOT_AVAILABLE":
            raise errors.DataStorageNotAvailableError(message)
        if status == 401 or code == "UNAUTHORIZED":
            raise errors.AuthenticationError(status, message)
        if status == 403 or code == "ACCESS_DENIED":
            raise errors.AuthenticationError(status, message)
        if code == "TRANSFER_MODE_NOT_AVAILABLE":
            raise errors.TransferError(message)
        if code == "UNSUPPORTED_FORMAT":
            raise errors.TransferError(message)
        if code == "RENDERING_ERROR":
            raise errors.RenderingError(message)
        if code == "RENDER_WINDOW_NOT_AVAILABLE":
            raise errors.RenderingError(message)
        if code == "EDITOR_NOT_ACTIVE":
            alias, _window, _op = _parse_editor_url(response)
            raise errors.EditorNotActiveError(alias, message)
        if code == "RENDER_WINDOW_NOT_FOUND":
            alias, window, _op = _parse_editor_url(response)
            raise errors.RenderWindowNotFoundError(alias, window, message)
        if code == "UNSUPPORTED_OPERATION":
            alias, window, op = _parse_editor_url(response)
            raise errors.UnsupportedOperationError(alias, window, op, message)
        raise errors.ApiError(status, code, message)

    def _is_localhost(self) -> bool:
        parsed = urllib.parse.urlparse(self._base_url)
        # urlparse strips brackets from IPv6 literals, so "::1" matches http://[::1]:8080
        return parsed.hostname in {"localhost", "127.0.0.1", "::1"}

    def _detect_transfer_mode(self) -> TransferMode:
        # file-reference requires direct filesystem access, so it is only viable
        # when the server is on localhost.  For remote hosts we short-circuit here
        # and return DIRECT without querying the server at all — the server query
        # would only be needed to learn supported modes, which is irrelevant when
        # the only viable mode is DIRECT regardless.
        if self._is_localhost() and TransferMode.FILE_REFERENCE in self.server_info.transfer_modes:
            _log.info(
                "[%s] Transfer mode: file-reference (localhost + server support)",
                self._base_url,
            )
            return TransferMode.FILE_REFERENCE
        if self._is_localhost():
            _log.warning(
                "[%s] file-reference unavailable (not in server capabilities), "
                "falling back to direct",
                self._base_url,
            )
        else:
            _log.info("[%s] Transfer mode: direct (remote server)", self._base_url)
        return TransferMode.DIRECT
