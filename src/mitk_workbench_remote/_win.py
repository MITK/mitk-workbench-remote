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

"""Windows process helpers (locale- and address-family-independent).

Shared by discovery.launch() failure cleanup and Workbench.shutdown(). Kept in
its own private module so both can import it without a circular dependency
(discovery already imports Workbench).
"""

from __future__ import annotations

import subprocess
import sys
import warnings


def _find_listening_pids(port: int) -> set[int]:
    """Return PIDs of processes listening on *port* (Windows only).

    Scans both ``netstat -ano -p TCP`` and ``-p TCPv6``. A listening TCP socket is
    identified locale-independently by its wildcard foreign address (``0.0.0.0:0`` /
    ``[::]:0`` -> ends with ``:0``), avoiding the localized state word
    (``LISTENING`` / ``ABHÖREN`` / ...). Returns the deduped set of owning PIDs
    (a single dual-stack instance appears under both families with the same PID);
    empty if none match or the command fails.
    """
    pids: set[int] = set()
    for proto in ("TCP", "TCPv6"):
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", proto],
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        for line in result.stdout.splitlines():
            parts = line.split()
            # Format: Proto  LocalAddr  ForeignAddr  State  PID
            if len(parts) < 5:
                continue
            local_addr, foreign_addr = parts[1], parts[2]
            if local_addr.endswith(f":{port}") and foreign_addr.endswith(":0"):
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    continue
    return pids


def _kill_pid_tree(pid: int) -> None:
    """``taskkill /F /T`` the given PID, warning (not raising) on failure."""
    result = subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        warnings.warn(
            f"taskkill /PID {pid} failed (code {result.returncode}): {result.stderr.strip()}",
            RuntimeWarning,
            stacklevel=2,
        )


def _kill_port_listeners(port: int) -> None:
    """Kill every process listening on *port* (Windows). No-op off Windows."""
    if sys.platform != "win32":
        return
    for pid in _find_listening_pids(port):
        _kill_pid_tree(pid)


def _workbench_process_running(image_name: str) -> bool:
    """Best-effort check whether a process with *image_name* is running (Windows only).

    MITK Workbench is single-instance: starting it again while an instance is
    already running makes the new process hand off to the running one and exit
    without bringing up a REST server on the requested port. ``launch()`` uses
    this to detect that situation up front and give an actionable error instead
    of waiting out the full timeout on a handoff that can never succeed.

    Returns ``False`` off Windows or if the query cannot be performed or parsed.
    Callers treat an inconclusive result as "not running" and fall back to the
    ordinary launch-then-poll behavior, so a false negative only costs the old
    (worse) error message, never a crash.

    Args:
        image_name: Process image name to look for, e.g. ``"MitkWorkbench.exe"``.
    """
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/NH", "/FO", "CSV"],
            capture_output=True,
        )
    except OSError:
        return False
    # tasklist prints a localized "no tasks match" line to stdout when nothing
    # matches, and CSV rows that quote the image name when it does. Match on the
    # (ASCII) image name so the check is locale- and codepage-independent; decode
    # with replacement so an OEM-codepage info line can never raise.
    raw = result.stdout or b""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    return image_name.lower() in text.lower()
