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

"""Workbench discovery and lifecycle; find or launch MITK Workbench instances."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mitk_workbench_remote import errors
from mitk_workbench_remote._win import _kill_port_listeners
from mitk_workbench_remote.transport import RestTransport
from mitk_workbench_remote.workbench import Workbench

_log = logging.getLogger(__name__)

_PREFERENCE_PATCH_FLAG: str = "--MITK.preferences-override"
_EXECUTABLE_ENV_VAR: str = "MITK_WORKBENCH"


def _build_prefs_xml(port: int, token: str) -> str:
    """Build the MITK XML preferences string to enable the REST API."""
    root = ET.Element("preferences")
    root.set("name", "")
    restapi = ET.SubElement(root, "preferences")
    restapi.set("name", "org.mitk.restapi")
    for prop_name, prop_value in [
        ("enabled", "true"),
        ("autoStart", "true"),
        ("port", str(port)),
        ("requireAuth", "true"),
        ("apiToken", token),
    ]:
        elem = ET.SubElement(restapi, "property")
        elem.set("name", prop_name)
        elem.set("value", prop_value)
    ET.indent(root, space="    ")
    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding="UTF-8", xml_declaration=True)
    return buf.getvalue().decode("UTF-8")


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
    _log.info("Discovering workbenches on %d ports", len(port_list))

    def _probe(port: int) -> tuple[int, Workbench | None]:
        transport = RestTransport(f"http://localhost:{port}", timeout=timeout)
        try:
            transport.get("/health")
            _log.debug("Port %d: found workbench", port)
            return port, Workbench(transport)
        except (errors.MitkError, OSError):
            _log.debug("Port %d: no response", port)
            transport.close()
            return port, None
        except Exception:
            transport.close()
            raise

    results: list[tuple[int, Workbench]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(len(port_list), 64))) as executor:
        futures = {executor.submit(_probe, p): p for p in port_list}
        for future in as_completed(futures):
            port, wb = future.result()
            if wb is not None:
                results.append((port, wb))

    results.sort(key=lambda t: t[0])
    found_ports = [p for p, _ in results]
    _log.info("Discovery complete: found %d instance(s) on ports %s", len(results), found_ports)
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
        port number up-front in the preference XML, so the socket cannot be
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


def _read_launch_stderr(path: str | os.PathLike[str]) -> str:
    """Best-effort read of the launch stderr log (utf-8, errors replaced).

    Returns '' if the file is missing or unreadable.
    """
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


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

    On Windows the shipped launcher is a ``.bat`` wrapper that ``start /B``
    detaches the real ``MitkWorkbench.exe``; the tracked ``cmd.exe`` then
    exits 0 within ~0.1 s, long before the REST server is up. ``launch()``
    therefore tolerates an exit code of 0 and keeps polling ``/health``; only
    a non-zero exit is treated as an immediate failure. The trade-off: a
    direct executable that genuinely exits 0 right away (e.g. ``--help``) is
    not detected until the ``timeout`` deadline rather than instantly.

    The child's stderr is redirected to a temp file (never an inherited PIPE,
    which the detached exe would hold open and deadlock on). On success the
    file is removed by :meth:`~mitk_workbench_remote.workbench.Workbench.shutdown`;
    on a failed launch it is removed before raising. Any failure also kills the
    process listening on ``port`` (the detached exe), since a raised
    ``launch()`` returns no handle for the caller to clean up later.

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
            preference patch XML file reference.

    Returns:
        A :class:`~mitk_workbench_remote.workbench.Workbench` handle backed
        by the newly started process.

    Raises:
        MitkError: If ``executable`` is ``None`` and ``MITK_WORKBENCH`` is
            not set, if no free port is available, or if the Workbench process
            exits immediately with a preference-patch failure.
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

    # 2. Build MITK workbench call & Execute
    if port is None:
        port = _find_free_port()

    if token is None:
        token = secrets.token_hex(16)

    _log.info("Launching workbench: %s on port %d", resolved_exe, port)

    fd, prefs_path = tempfile.mkstemp(suffix=".xml", prefix="mitk_prefs_")
    stderr_path: str | None = None
    process: subprocess.Popen[bytes] | None = None
    transport: RestTransport | None = None

    def _abort() -> None:
        """Tear down a FAILED launch: kill the detached real exe (port-keyed) and the
        (already-dead) wrapper, close the transport, drop the stderr temp file. A raised
        launch() returns no handle, so the caller cannot clean up later."""
        _kill_port_listeners(port)
        if process is not None:
            process.terminate()
        if transport is not None:
            transport.close()
        if stderr_path is not None:
            with contextlib.suppress(OSError):
                os.remove(stderr_path)

    try:
        with os.fdopen(fd, "w", encoding="UTF-8") as f:
            f.write(_build_prefs_xml(port, token))

        # We use the file option and not the direct xml content passing for security
        # reasons to avoid that the API key can be queried via the process informations
        cmd: list[str] = [str(resolved_exe), _PREFERENCE_PATCH_FLAG, f"@{prefs_path}"]
        if extra_args:
            cmd.extend(extra_args)

        # stderr -> temp file, NOT PIPE. `start /B` hands the inherited handle to the
        # detached exe, which holds it for its whole life; an unread PIPE then blocks any
        # parent read AND deadlocks the exe once its 64 KB buffer fills. A file has neither
        # problem and preserves the exe's "Cannot apply preferences" diagnostic.
        stderr_fd, stderr_path = tempfile.mkstemp(suffix=".log", prefix="mitk_stderr_")
        stderr_file = os.fdopen(stderr_fd, "wb")
        try:
            process = subprocess.Popen(cmd, stderr=stderr_file)
        finally:
            # Popen inherited its own copy; close the parent's so we don't leak a
            # descriptor. The child keeps writing through its inherited handle.
            stderr_file.close()

        transport = RestTransport(f"http://localhost:{port}", token=token)

        # The shipped MITK launcher is a .bat that `start /B`-detaches the real exe, so the
        # tracked cmd.exe exits 0 within ~0.1 s — long before the REST server is up. Tolerate
        # exit-0 and keep polling /health; only a NON-ZERO exit is an immediate hard failure
        # (a direct-exe crash). Always attempt at least one probe even for timeout=0.
        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None
        while True:
            returncode = process.poll()
            if returncode is not None and returncode != 0:
                # Genuine failure — only reachable for a direct-exe launch (the .bat wrapper
                # always exits 0). The exe has exited and released the stderr file, so reading
                # it is safe and surfaces real diagnostics.
                stderr_text = _read_launch_stderr(stderr_path)
                if "Cannot apply preferences" in stderr_text:
                    raise errors.MitkError(
                        f"MITK Workbench exited with code {returncode}: failed to apply "
                        "preference overrides. If this is a fresh installation, please start "
                        "the Workbench manually once to initialise its preferences store, "
                        "then try again."
                    )
                raise errors.MitkError(
                    f"MITK Workbench process exited unexpectedly with code {returncode}."
                )
            try:
                transport.get("/health")
                return Workbench(transport, process=process, stderr_log_path=stderr_path)
            except errors.MitkConnectionError as exc:
                last_exc = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)

        # Deadline expired. Name the early-exit case and surface the prefs diagnostic (safe
        # now that stderr is a file, not a still-live PIPE).
        exited = process.poll()
        diag = _read_launch_stderr(stderr_path)
        if "Cannot apply preferences" in diag:
            raise errors.MitkError(
                f"MITK Workbench failed to apply preference overrides (process exited with "
                f"code {exited}). If this is a fresh installation, please start the Workbench "
                "manually once to initialise its preferences store, then try again."
            )
        detail = (
            f"the launched process already exited (code {exited}) and "
            if exited is not None
            else ""
        )
        raise errors.MitkConnectionError(
            f"Workbench did not become reachable on port {port} within {timeout}s; "
            f"{detail}check that the Workbench actually started."
        ) from last_exc
    except BaseException:
        # Any failure after the detached exe may already be listening (incl. KeyboardInterrupt,
        # and the raises above). Kill the real listener too, not just the dead wrapper.
        _abort()
        raise
    finally:
        with contextlib.suppress(OSError):
            os.remove(prefs_path)
