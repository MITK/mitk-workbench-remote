# SPDX-FileCopyrightText: 2026, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
#
# SPDX-License-Identifier: Apache-2.0

"""Live StdMultiEditor integration tests.

Run with: pytest tests/test_integration/test_editors_std_multi.py -v -m integration
"""

from __future__ import annotations

import math
import os

import pytest

import mitk_workbench_remote as mw
from mitk_workbench_remote import Camera, ViewDirection, WindowKind

pytestmark = pytest.mark.integration


@pytest.fixture
def workbench() -> mw.Workbench:
    url = os.environ.get("MITK_WORKBENCH_URL", "http://localhost:8080")
    token = os.environ.get("MITK_WORKBENCH_TOKEN")
    wb = mw.connect(url, token=token)
    if not wb.ping():
        pytest.skip(f"No MITK Workbench reachable at {url}")
    return wb


def test_editor_listing_includes_stdmulti(workbench: mw.Workbench) -> None:
    aliases = [d.alias for d in workbench.editors()]
    assert "stdmulti" in aliases


def test_editor_info_returns_window_names(workbench: mw.Workbench) -> None:
    info = workbench.std_multi.get_info()
    if not info.active:
        pytest.skip("StdMultiEditor is not open")
    assert "axial" in info.windows
    assert "3d" in info.windows


def test_editor_screenshot_returns_non_empty_bytes(workbench: mw.Workbench) -> None:
    if not workbench.std_multi.is_active:
        pytest.skip("StdMultiEditor is not open")
    data = workbench.std_multi.screenshot()
    assert len(data) > 0


def test_camera_get_modify_put_round_trip(workbench: mw.Workbench) -> None:
    if not workbench.std_multi.is_active:
        pytest.skip("StdMultiEditor is not open")
    axial = workbench.std_multi.axial
    original = axial.get_camera()
    try:
        axial.set_camera(parallel_scale=200.0)
        roundtrip = axial.get_camera()
        assert math.isclose(roundtrip.parallel_scale or 0.0, 200.0, rel_tol=1e-3)
    finally:
        # Restore — only forward the explicit pose if it was set.
        restore = Camera(parallel_scale=original.parallel_scale)
        axial.set_camera(restore)


def test_named_window_properties_align_with_window_listing(
    workbench: mw.Workbench,
) -> None:
    if not workbench.std_multi.is_active:
        pytest.skip("StdMultiEditor is not open")
    listed = {w.id: w for w in workbench.std_multi.list_windows()}
    assert workbench.std_multi.axial.id in listed
    assert workbench.std_multi.three_d.kind == WindowKind.THREE_D
    assert workbench.std_multi.axial.view_direction == ViewDirection.AXIAL
