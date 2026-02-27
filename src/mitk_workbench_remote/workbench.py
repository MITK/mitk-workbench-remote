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

"""Workbench handle — primary entry point for controlling a MITK Workbench instance.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from types import TracebackType

from mitk_workbench_remote import errors
from mitk_workbench_remote.storage import DataStorage
from mitk_workbench_remote.transport import RestTransport


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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Base URL of the Workbench REST server."""
        return self._transport.base_url

    @property
    def is_launched_remotely(self) -> bool:
        """Returns if the workbench was launched remotely (true) or was just connected to but is independent (false)."""
        return self._process is not None

    @property
    def info(self) -> WorkbenchInfo:
        """Metadata about the connected Workbench instance.

        Fetched lazily from ``GET /api/v1/`` on first access and cached thereafter.
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
            return response.ok
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        """Whether the Workbench is currently reachable. Alias for :meth:`ping`."""
        return self.ping()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Terminate the subprocess started by :func:`~mitk_workbench_remote.discovery.launch`.

        Sends ``SIGTERM`` (or ``TerminateProcess`` on Windows) and waits up to
        10 seconds; falls back to forceful termination (``SIGKILL`` on Unix,
        ``TerminateProcess`` on Windows) on timeout. Also closes the underlying
        transport session.

        Raises:
            MitkError: If this instance was not created by ``launch()``. Use property is_launched_remotely to check
            this.
        """
        if self._process is None:
            raise errors.MitkError("shutdown() is only valid for instances started with launch()")
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
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
    transport = RestTransport(url, token=token, timeout=timeout, transfer_mode=transfer_mode)
    return Workbench(transport)
