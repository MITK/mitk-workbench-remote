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

"""Workbench discovery and lifecycle — find or launch MITK Workbench instances.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mitk_workbench_remote import errors
from mitk_workbench_remote.transport import RestTransport
from mitk_workbench_remote.workbench import Workbench

_PREFERENCE_PATCH_FLAG: str = "--patch-preferences"
_EXECUTABLE_ENV_VAR: str = "MITK_WORKBENCH_REMOTE"


def discover(
    ports: Iterable[int] = range(8080, 8100),
    *,
    timeout: float = 0.5,
) -> list[Workbench]:
    """Probe localhost ports for running MITK Workbench instances.

    Probes are executed concurrently. Any port that responds to
    ``GET /api/v1/health`` with a 2xx status is included in the result.

    Args:
        ports: Iterable of port numbers to probe. Defaults to 8080-8099.
        timeout: Per-probe connection timeout in seconds.

    Returns:
        List of :class:`~mitk_workbench_remote.workbench.Workbench` handles,
        sorted by port number (ascending).
    """
    port_list = list(ports)

    def _probe(port: int) -> tuple[int, Workbench | None]:
        transport = RestTransport(f"http://localhost:{port}", timeout=timeout)
        try:
            transport.get("/health")
            return port, Workbench(transport)
        except (errors.MitkError, OSError):
            transport.close()
            return port, None

    results: list[tuple[int, Workbench]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(len(port_list), 64))) as executor:
        futures = {executor.submit(_probe, p): p for p in port_list}
        for future in as_completed(futures):
            port, wb = future.result()
            if wb is not None:
                results.append((port, wb))

    results.sort(key=lambda t: t[0])
    return [wb for _, wb in results]


def _find_free_port(start: int = 8080, end: int = 8100) -> int:
    """Find the first free TCP port in the given range.

    Args:
        start: First port to try (inclusive).
        end: Last port to try (exclusive).

    Returns:
        A port number that could be bound on ``127.0.0.1``.

    Raises:
        MitkError: If no free port is found in the range.

    Note:
        There is an inherent TOCTOU (time-of-check/time-of-use) race between
        releasing the probe socket and the subprocess binding to the port.
        This is unavoidable with the current architecture: MITK requires the
        port number up-front in the preference JSON, so the socket cannot be
        held open across the process boundary.
    """
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise errors.MitkError(f"No free port in {start}-{end}")


def launch(
    executable: str | Path | None = None,
    *,
    port: int | None = None,
    token: str | None = None,
    timeout: float = 30.0,
    extra_args: list[str] | None = None,
) -> Workbench:
    """Start a new MITK Workbench process with the REST API enabled.

    Spawns the process, waits for the REST server to become healthy, and
    returns a connected :class:`~mitk_workbench_remote.workbench.Workbench`
    handle.

    Args:
        executable: Path to the MITK Workbench executable. When ``None``,
            the ``MITK_WORKBENCH`` environment variable is used. Pass either
            this argument or set the env var.
        port: Port for the REST server. When ``None``, the first free port
            in 8080-8099 is chosen automatically.
        token: API token for the new server. When ``None``, a secure random
            32-character hex token is generated.
        timeout: Maximum seconds to wait for the server to become healthy
            before giving up.
        extra_args: Additional command-line arguments appended after the
            preference patch JSON.

    Returns:
        A :class:`~mitk_workbench_remote.workbench.Workbench` handle backed
        by the newly started process.

    Raises:
        MitkError: If ``executable`` is ``None`` and ``MITK_WORKBENCH`` is
            not set, or if no free port is available.
        FileNotFoundError: If the resolved executable path does not exist.
        MitkConnectionError: If the server does not become healthy within
            ``timeout`` seconds.
    """
    # 1. Resolve executable
    if executable is None:
        env_val = os.environ.get(_EXECUTABLE_ENV_VAR)
        if env_val is None:
            raise errors.MitkError(
                f"No executable given and {_EXECUTABLE_ENV_VAR!r} is not set. "
                f"Pass the executable path or set {_EXECUTABLE_ENV_VAR}."
            )
        executable = env_val

    resolved_exe = Path(executable)
    if not resolved_exe.exists():
        which_result = shutil.which(str(resolved_exe))
        if which_result is None:
            raise FileNotFoundError(f"MITK Workbench executable not found: {resolved_exe}")
        resolved_exe = Path(which_result)

    # 2. Build MITK workbench call
    if port is None:
        port = _find_free_port()

    if token is None:
        token = secrets.token_hex(16)

    prefs = {
        "org.mitk.restapi": {
            "enabled": True,
            "port": port,
            "token": token,
            "log_requests": False,
        }
    }
    cmd: list[str] = [str(resolved_exe), _PREFERENCE_PATCH_FLAG, json.dumps(prefs)]
    if extra_args:
        cmd.extend(extra_args)

    # Get MITK subprocess running and connected
    process: subprocess.Popen[bytes] = subprocess.Popen(cmd)

    transport = RestTransport(f"http://localhost:{port}", token=token)

    # Health poll loop — always attempt at least one probe even for timeout=0.
    # On any non-connection exception the process is terminated before re-raising.
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    try:
        while True:
            try:
                transport.get("/health")
                return Workbench(transport, process=process)
            except errors.MitkConnectionError as exc:
                last_exc = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
    except:
        process.terminate()
        transport.close()
        raise

    # Deadline expired — clean up
    process.terminate()
    transport.close()
    raise errors.MitkConnectionError(f"Workbench did not start within {timeout}s") from last_exc
