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

"""Tests for discovery.py."""

import contextlib
import glob
import os
import re
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import responses

from mitk_workbench_remote import errors
from mitk_workbench_remote._win import _find_listening_pids
from mitk_workbench_remote.discovery import (
    _EXECUTABLE_ENV_VAR,
    _PREFERENCE_PATCH_FLAG,
    _find_free_port,
    discover,
    launch,
)
from mitk_workbench_remote.workbench import Workbench


def _api(port: int, path: str) -> str:
    return f"http://localhost:{port}/api/v1{path}"


@pytest.fixture(autouse=True)
def _cleanup_launch_stderr_logs():
    """Remove stderr temp files leaked by launch() success-path tests.

    launch() now writes the child's stderr to a ``mitk_stderr_*.log`` temp file
    that only shutdown() deletes; tests that call close() (or never shut down)
    would otherwise litter the temp dir. Only files appearing during the test
    are removed, so unrelated files are never touched.
    """
    pattern = os.path.join(tempfile.gettempdir(), "mitk_stderr_*.log")
    before = set(glob.glob(pattern))
    yield
    for path in set(glob.glob(pattern)) - before:
        with contextlib.suppress(OSError):
            os.remove(path)


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------


@responses.activate
def test_discover_finds_healthy_port() -> None:
    responses.add(
        responses.GET,
        _api(8080, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )
    result = discover(ports=[8080], timeout=0.5)
    assert len(result) == 1
    assert isinstance(result[0], Workbench)
    assert result[0].url == "http://localhost:8080"
    for wb in result:
        wb.close()


@responses.activate
def test_discover_skips_unreachable_port() -> None:
    responses.add(
        responses.GET,
        _api(8080, "/health"),
        body=requests.exceptions.ConnectionError("refused"),
    )
    result = discover(ports=[8080], timeout=0.5)
    assert result == []


@responses.activate
def test_discover_returns_sorted_by_port() -> None:
    responses.add(
        responses.GET,
        _api(8080, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )
    responses.add(
        responses.GET,
        _api(8082, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )
    result = discover(ports=[8082, 8080], timeout=0.5)
    assert len(result) == 2
    assert result[0].url == "http://localhost:8080"
    assert result[1].url == "http://localhost:8082"
    for wb in result:
        wb.close()


@responses.activate
def test_discover_returns_empty_when_none_healthy() -> None:
    for port in [8080, 8081, 8082]:
        responses.add(
            responses.GET,
            _api(port, "/health"),
            body=requests.exceptions.ConnectionError("refused"),
        )
    result = discover(ports=range(8080, 8083), timeout=0.5)
    assert result == []


@responses.activate
def test_discover_custom_port_range() -> None:
    responses.add(
        responses.GET,
        _api(9000, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )
    result = discover(ports=[9000], timeout=0.5)
    assert len(result) == 1
    assert result[0].url == "http://localhost:9000"
    for wb in result:
        wb.close()


def test_discover_uses_timeout_parameter() -> None:
    captured: list[float] = []

    original_init = __import__(
        "mitk_workbench_remote.transport", fromlist=["RestTransport"]
    ).RestTransport.__init__

    def capturing_init(self, base_url, **kwargs):
        captured.append(kwargs.get("timeout", 30.0))
        original_init(self, base_url, **kwargs)

    with (
        patch(
            "mitk_workbench_remote.discovery.RestTransport.__init__",
            capturing_init,
        ),
        patch("mitk_workbench_remote.discovery.RestTransport.get") as mock_get,
    ):
        mock_get.side_effect = errors.MitkConnectionError("refused")
        discover(ports=[8080], timeout=0.25)

    assert all(t == 0.25 for t in captured)


# ---------------------------------------------------------------------------
# _build_prefs_xml()
# ---------------------------------------------------------------------------


def _restapi_props(xml_str: str) -> dict[str, str]:
    root = ET.fromstring(xml_str)
    restapi = root.find("./preferences[@name='org.mitk.restapi']")
    assert restapi is not None
    return {e.get("name"): e.get("value") for e in restapi.findall("property")}


def test_build_prefs_xml_contains_port_and_token() -> None:
    from mitk_workbench_remote.discovery import _build_prefs_xml

    props = _restapi_props(_build_prefs_xml(8085, "mytoken"))
    assert props["port"] == "8085"
    assert props["apiToken"] == "mytoken"


def test_build_prefs_xml_sets_auth_and_autostart() -> None:
    from mitk_workbench_remote.discovery import _build_prefs_xml

    props = _restapi_props(_build_prefs_xml(8080, "tok"))
    assert props["requireAuth"] == "true"
    assert props["enabled"] == "true"
    assert props["autoStart"] == "true"


def test_build_prefs_xml_is_valid_xml() -> None:
    from mitk_workbench_remote.discovery import _build_prefs_xml

    ET.fromstring(_build_prefs_xml(8080, "tok"))  # must not raise


# ---------------------------------------------------------------------------
# launch() — happy path
# ---------------------------------------------------------------------------


def _make_fake_exe(tmp_path: Path) -> Path:
    exe = tmp_path / "MitkWorkbench"
    exe.touch()
    return exe


@responses.activate
def test_launch_spawns_process_with_correct_args(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8099

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch("subprocess.Popen") as mock_popen,
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(exe, port=port, token="mytoken", timeout=5.0)

    call_args = mock_popen.call_args[0][0]
    assert call_args[0] == str(exe)
    assert _PREFERENCE_PATCH_FLAG in call_args
    flag_idx = call_args.index(_PREFERENCE_PATCH_FLAG)
    assert call_args[flag_idx + 1].startswith("@")
    wb.close()


@responses.activate
def test_launch_prefs_xml_built_with_correct_port_and_token(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8098

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"
        ) as mock_build,
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(exe, port=port, token="tok123", timeout=5.0)

    mock_build.assert_called_once_with(port, "tok123")
    wb.close()


@responses.activate
def test_launch_auto_generates_token(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8097

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"
        ) as mock_build,
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(exe, port=port, timeout=5.0)

    token = mock_build.call_args.args[1]
    assert re.fullmatch(r"[0-9a-f]{32}", token), f"Expected 32-hex token, got: {token!r}"
    wb.close()


@responses.activate
def test_launch_uses_explicit_token(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8096

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"
        ) as mock_build,
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(exe, port=port, token="explicit-token", timeout=5.0)

    mock_build.assert_called_once_with(port, "explicit-token")
    wb.close()


@responses.activate
def test_launch_uses_explicit_port(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8095

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"
        ) as mock_build,
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(exe, port=port, timeout=5.0)

    assert mock_build.call_args.args[0] == port
    wb.close()


@responses.activate
def test_launch_extra_args_appended_to_cmd(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8094

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch("subprocess.Popen") as mock_popen,
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(exe, port=port, token="tok", timeout=5.0, extra_args=["--no-gui", "--debug"])

    call_args = mock_popen.call_args[0][0]
    assert "--no-gui" in call_args
    assert "--debug" in call_args
    assert call_args[-2] == "--no-gui"
    assert call_args[-1] == "--debug"
    wb.close()


@responses.activate
def test_launch_returns_workbench_with_process(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8093

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345
    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="")),
    ):
        wb = launch(exe, port=port, token="tok", timeout=5.0)

    assert isinstance(wb, Workbench)
    # Verify it is a launched instance via public behaviour: shutdown() must not raise
    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="")):
        wb.shutdown()  # also closes transport


# ---------------------------------------------------------------------------
# launch() — wrapper exit-0 tolerance and child stderr handling
# ---------------------------------------------------------------------------


@responses.activate
def test_launch_tolerates_wrapper_exit_zero_then_becomes_healthy(tmp_path: Path) -> None:
    """The shipped .bat ``start /B``-detaches the real exe and exits 0 within
    ~0.1s. launch() must tolerate exit-0 and keep polling /health instead of
    treating the wrapper's exit as a fatal failure."""
    exe = _make_fake_exe(tmp_path)
    port = 8085

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 0  # wrapper has already exited 0
    mock_proc.pid = 12345
    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
    ):
        wb = launch(exe, port=port, token="tok", timeout=5.0)

    assert isinstance(wb, Workbench)
    wb.close()


@responses.activate
def test_launch_redirects_child_stderr_to_closed_file_not_pipe(tmp_path: Path) -> None:
    """stderr must be a real file handle (never PIPE, which the detached exe
    would hold open and deadlock on), and the parent closes its own copy
    immediately after Popen."""
    exe = _make_fake_exe(tmp_path)
    port = 8084

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    captured: dict[str, object] = {}

    def _capture_popen(cmd, **kwargs):
        captured["stderr"] = kwargs.get("stderr")
        # No spec=subprocess.Popen here: inside the patch, subprocess.Popen IS the
        # mock, and a Mock cannot spec another Mock.
        m = MagicMock()
        m.poll.return_value = None
        m.pid = 12345
        return m

    with (
        patch("subprocess.Popen", side_effect=_capture_popen),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
    ):
        wb = launch(exe, port=port, token="tok", timeout=5.0)

    stderr_obj = captured["stderr"]
    assert stderr_obj is not subprocess.PIPE
    assert hasattr(stderr_obj, "write")  # a writable file object, not a pipe sentinel
    assert stderr_obj.closed is True  # parent closed its handle after Popen
    wb.close()


@responses.activate
def test_launch_timeout_zero_still_probes_once(tmp_path: Path) -> None:
    """timeout=0 must still perform exactly one /health probe before giving up."""
    exe = _make_fake_exe(tmp_path)
    port = 8083

    probe_count = 0

    def _refuse(request):
        nonlocal probe_count
        probe_count += 1
        raise requests.exceptions.ConnectionError("refused")

    responses.add_callback(responses.GET, _api(port, "/health"), callback=_refuse)

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch("mitk_workbench_remote.discovery._kill_port_listeners"),
        pytest.raises(errors.MitkConnectionError),
    ):
        launch(exe, port=port, token="tok", timeout=0)

    assert probe_count == 1


@responses.activate
def test_launch_never_healthy_names_early_exit_and_kills_listeners(tmp_path: Path) -> None:
    """A never-healthy launch whose wrapper exited 0 fails at the deadline with a
    message naming the early exit, kills the (detached) port listener so it is
    not orphaned, and removes the stderr temp file."""
    exe = _make_fake_exe(tmp_path)
    port = 8082

    def _refuse(request):
        raise requests.exceptions.ConnectionError("refused")

    responses.add_callback(responses.GET, _api(port, "/health"), callback=_refuse)

    # Transparent spy on mkstemp to capture the real stderr (.log) path so we can
    # assert the failure-path cleanup removed it (the file was opened from an fd,
    # so the file object's .name is the fd number, not a usable path).
    real_mkstemp = tempfile.mkstemp
    stderr_paths: list[str] = []

    def _spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        if kwargs.get("suffix") == ".log":
            stderr_paths.append(path)
        return fd, path

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 0  # exited 0, but /health never comes up
    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch("mitk_workbench_remote.discovery._kill_port_listeners") as mock_kill,
        patch("tempfile.mkstemp", side_effect=_spy_mkstemp),
        pytest.raises(errors.MitkConnectionError, match=r"already exited \(code 0\)"),
    ):
        launch(exe, port=port, token="tok", timeout=0)

    mock_kill.assert_called_once_with(port)
    assert stderr_paths, "stderr temp file was never created"
    assert not os.path.exists(stderr_paths[0]), "stderr temp file leaked on failed launch"


# ---------------------------------------------------------------------------
# launch() — env var fallback
# ---------------------------------------------------------------------------


@responses.activate
def test_launch_uses_env_var_when_no_executable_given(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8092

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch.dict(os.environ, {_EXECUTABLE_ENV_VAR: str(exe)}),
        patch("subprocess.Popen") as mock_popen,
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(port=port, token="tok", timeout=5.0)

    call_args = mock_popen.call_args[0][0]
    assert call_args[0] == str(exe)
    wb.close()


@responses.activate
def test_launch_explicit_path_takes_precedence_over_env_var(tmp_path: Path) -> None:
    exe_explicit = _make_fake_exe(tmp_path)
    exe_env = tmp_path / "EnvWorkbench"
    exe_env.touch()
    port = 8091

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    with (
        patch.dict(os.environ, {_EXECUTABLE_ENV_VAR: str(exe_env)}),
        patch("subprocess.Popen") as mock_popen,
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        wb = launch(exe_explicit, port=port, token="tok", timeout=5.0)

    call_args = mock_popen.call_args[0][0]
    assert call_args[0] == str(exe_explicit)
    wb.close()


def test_launch_raises_when_no_executable_and_no_env_var() -> None:
    with patch.dict(os.environ, {}, clear=True):
        # Ensure the env var is not set
        os.environ.pop(_EXECUTABLE_ENV_VAR, None)
        with pytest.raises(errors.MitkError, match=_EXECUTABLE_ENV_VAR):
            launch()


# ---------------------------------------------------------------------------
# launch() — error paths
# ---------------------------------------------------------------------------


def test_launch_missing_executable_raises_FileNotFoundError(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        launch(nonexistent)


@responses.activate
def test_launch_timeout_terminates_process_and_raises_ConnectionError(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8090

    # Always refuse connection -> every call raises ConnectionError
    def _always_refuse(request):
        raise requests.exceptions.ConnectionError("refused")

    responses.add_callback(responses.GET, _api(port, "/health"), callback=_always_refuse)

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch("mitk_workbench_remote.discovery._kill_port_listeners") as mock_kill,
        patch("time.sleep"),  # Speed up the loop
        pytest.raises(errors.MitkConnectionError, match="did not become reachable"),
    ):
        launch(exe, port=port, token="tok", timeout=0.1)

    mock_proc.terminate.assert_called_once()
    mock_kill.assert_called_once_with(port)


@responses.activate
def test_launch_non_connection_error_propagates_immediately(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    port = 8089

    # Auth error — not a ConnectionError
    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"error": {"code": "UNAUTHORIZED", "message": "bad token"}},
        status=401,
    )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch("mitk_workbench_remote.discovery._kill_port_listeners"),
        pytest.raises(errors.AuthenticationError),
    ):
        launch(exe, port=port, token="tok", timeout=5.0)

    mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# launch() — early process exit
# ---------------------------------------------------------------------------


def test_launch_cannot_apply_preferences_raises_descriptive_error(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1

    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch(
            "mitk_workbench_remote.discovery._read_launch_stderr",
            return_value="Cannot apply preferences for org.mitk.restapi\n",
        ),
        patch("mitk_workbench_remote.discovery._kill_port_listeners"),
        pytest.raises(errors.MitkError, match="manually once"),
    ):
        launch(exe, port=8087, token="tok", timeout=5.0)

    mock_proc.terminate.assert_called_once()


def test_launch_unexpected_process_exit_raises_mitk_error(tmp_path: Path) -> None:
    exe = _make_fake_exe(tmp_path)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1

    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch(
            "mitk_workbench_remote.discovery._read_launch_stderr",
            return_value="Segmentation fault\n",
        ),
        patch("mitk_workbench_remote.discovery._kill_port_listeners"),
        pytest.raises(errors.MitkError, match="unexpectedly"),
    ):
        launch(exe, port=8086, token="tok", timeout=5.0)


# ---------------------------------------------------------------------------
# shutdown() on launched instance
# ---------------------------------------------------------------------------


@responses.activate
def test_shutdown_wrapper_kills_both_listener_and_wrapper(tmp_path: Path) -> None:
    """When MITK is launched via a batch wrapper, the listener PID differs from
    the Popen PID.  shutdown() must taskkill both."""
    exe = _make_fake_exe(tmp_path)
    port = 8088

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345

    # Simulate netstat: listener PID (99999) differs from wrapper PID (12345).
    # The state column is irrelevant to the parser (it keys on the wildcard
    # foreign address), so the same line answers both the TCP and TCPv6 scans.
    netstat_output = (
        f"  TCP    127.0.0.1:{port}         0.0.0.0:0              LISTENING       99999\n"
    )
    mock_netstat = MagicMock(stdout=netstat_output)

    def _run_side_effect(cmd, **kwargs):
        if cmd[0] == "netstat":
            return mock_netstat
        return MagicMock(returncode=0)  # taskkill

    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch("subprocess.run", side_effect=_run_side_effect) as mock_run,
    ):
        wb = launch(exe, port=port, token="tok", timeout=5.0)
        wb.shutdown()

    mock_proc.wait.assert_called()

    if sys.platform == "win32":
        taskkill_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "taskkill"]
        assert len(taskkill_calls) == 2
        killed_pids = [c[0][0][-1] for c in taskkill_calls]
        assert "99999" in killed_pids  # real Workbench
        assert str(mock_proc.pid) in killed_pids  # wrapper
    else:
        mock_proc.terminate.assert_called_once()

    wb.close()


@responses.activate
def test_shutdown_direct_launch_kills_listener_only(tmp_path: Path) -> None:
    """When MITK is started directly (no wrapper), the listener PID equals the
    Popen PID.  shutdown() must only taskkill once."""
    exe = _make_fake_exe(tmp_path)
    port = 8087

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.pid = 55555

    # Simulate netstat: listener PID matches Popen PID (same process). The same
    # line answers both the TCP and TCPv6 scans and dedups to a single PID.
    netstat_output = (
        f"  TCP    127.0.0.1:{port}         0.0.0.0:0              LISTENING       55555\n"
    )
    mock_netstat = MagicMock(stdout=netstat_output)

    def _run_side_effect(cmd, **kwargs):
        if cmd[0] == "netstat":
            return mock_netstat
        return MagicMock(returncode=0)  # taskkill

    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
        patch("subprocess.run", side_effect=_run_side_effect) as mock_run,
    ):
        wb = launch(exe, port=port, token="tok", timeout=5.0)
        wb.shutdown()

    mock_proc.wait.assert_called()

    if sys.platform == "win32":
        taskkill_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "taskkill"]
        # Only one taskkill — no duplicate for the same PID
        assert len(taskkill_calls) == 1
        assert taskkill_calls[0][0][0][-1] == "55555"
    else:
        mock_proc.terminate.assert_called_once()

    wb.close()


@responses.activate
def test_shutdown_removes_stderr_temp_file(tmp_path: Path) -> None:
    """The stderr temp file created by launch() is deleted on shutdown()."""
    exe = _make_fake_exe(tmp_path)
    port = 8086

    responses.add(
        responses.GET,
        _api(port, "/health"),
        json={"data": {"status": "healthy"}},
        status=200,
    )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345
    with (
        patch("subprocess.Popen", return_value=mock_proc),
        patch("mitk_workbench_remote.discovery._build_prefs_xml", return_value="<xml/>"),
    ):
        wb = launch(exe, port=port, token="tok", timeout=5.0)

    stderr_path = wb._stderr_log_path
    assert stderr_path is not None
    assert os.path.exists(stderr_path)

    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="")):
        wb.shutdown()

    assert not os.path.exists(stderr_path)


# ---------------------------------------------------------------------------
# _find_listening_pids() — locale- and address-family-independent lookup
# ---------------------------------------------------------------------------


def _netstat_per_proto(*, tcp: str = "", tcpv6: str = ""):
    """Build a subprocess.run side-effect that answers per ``-p`` proto arg."""

    def _side_effect(cmd, **kwargs):
        proto = cmd[cmd.index("-p") + 1]
        return MagicMock(stdout=tcp if proto == "TCP" else tcpv6)

    return _side_effect


def test_find_listening_pids_locale_independent() -> None:
    # German Windows prints ABHÖREN, not LISTENING. The parser keys on the
    # wildcard foreign address (:0), so the localized state word is irrelevant.
    tcp = "  TCP    0.0.0.0:8080    0.0.0.0:0    ABHÖREN    4242\n"
    with patch("subprocess.run", side_effect=_netstat_per_proto(tcp=tcp)):
        assert _find_listening_pids(8080) == {4242}


def test_find_listening_pids_finds_tcpv6_only_listener() -> None:
    tcpv6 = "  TCP    [::]:8080    [::]:0    LISTENING    4242\n"
    with patch("subprocess.run", side_effect=_netstat_per_proto(tcpv6=tcpv6)):
        assert _find_listening_pids(8080) == {4242}


def test_find_listening_pids_ignores_established_rows() -> None:
    # An established connection whose local port matches has a real foreign
    # address (not :0) and must not be mistaken for a listener.
    tcp = "  TCP    127.0.0.1:8080    93.184.216.34:443    ESTABLISHED    1111\n"
    with patch("subprocess.run", side_effect=_netstat_per_proto(tcp=tcp)):
        assert _find_listening_pids(8080) == set()


def test_find_listening_pids_dedups_dual_stack_instance() -> None:
    # A single dual-stack instance appears under both families with the SAME
    # PID; the two-family merge must collapse to one.
    tcp = "  TCP    0.0.0.0:8080    0.0.0.0:0    LISTENING    7777\n"
    tcpv6 = "  TCP    [::]:8080    [::]:0    LISTENING    7777\n"
    with patch("subprocess.run", side_effect=_netstat_per_proto(tcp=tcp, tcpv6=tcpv6)):
        assert _find_listening_pids(8080) == {7777}


# ---------------------------------------------------------------------------
# _find_free_port()
# ---------------------------------------------------------------------------


def test_find_free_port_returns_bindable_port() -> None:
    port = _find_free_port()
    assert isinstance(port, int)
    # Verify we can actually bind to it
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
