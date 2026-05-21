# SPDX-FileCopyrightText: 2026, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
#
# SPDX-License-Identifier: Apache-2.0

"""Live MxNEditor integration tests.

Run with: pytest tests/test_integration/test_editors_mxn.py -v -m integration
"""

from __future__ import annotations

import os

import pytest

import mitk_workbench_remote as mw
from mitk_workbench_remote import ViewDirection

pytestmark = pytest.mark.integration


@pytest.fixture
def workbench() -> mw.Workbench:
    url = os.environ.get("MITK_WORKBENCH_URL", "http://localhost:8080")
    token = os.environ.get("MITK_WORKBENCH_TOKEN")
    wb = mw.connect(url, token=token)
    if not wb.ping():
        pytest.skip(f"No MITK Workbench reachable at {url}")
    return wb


def test_layout_round_trip_via_dict(workbench: mw.Workbench) -> None:
    if not workbench.mxn.is_active:
        pytest.skip("MxNEditor is not open")
    original = workbench.mxn.get_layout_json()
    applied = workbench.mxn.set_layout(original)
    assert applied["version"] == original["version"]
    assert "root" in applied


def test_apply_preset_three_up(workbench: mw.Workbench) -> None:
    pytest.importorskip("mitk.mxn.layout")
    if not workbench.mxn.is_active:
        pytest.skip("MxNEditor is not open")
    doc = workbench.mxn.apply_preset("three-up")
    ids = sorted(w.id for w in doc.root.windows())
    assert len(ids) == 3


def test_update_layout_links_axial_windows_to_group(workbench: mw.Workbench) -> None:
    pytest.importorskip("mitk.mxn.layout")
    if not workbench.mxn.is_active:
        pytest.skip("MxNEditor is not open")
    from mitk.mxn.layout import MxNWindowSelector  # type: ignore[import-not-found]

    workbench.mxn.apply_preset("three-up")
    result = workbench.mxn.update_layout(
        lambda d: MxNWindowSelector(d).by_view(ViewDirection.AXIAL).link_to("axial_group")
    )
    assert "axial_group" in result.groups


def test_per_cell_selected_position_round_trip(workbench: mw.Workbench) -> None:
    pytest.importorskip("mitk.mxn.layout")
    if not workbench.mxn.is_active:
        pytest.skip("MxNEditor is not open")
    workbench.mxn.apply_preset("three-up")
    cells = workbench.mxn.list_windows()
    assert cells, "preset 'three-up' should produce cells"
    cell = cells[0]
    target = (10.0, 20.0, 30.0)
    cell.set_selected_position(target)
    pos = cell.get_selected_position()
    # MITK may snap to discrete steps — use loose tolerance per axis.
    assert all(abs(a - b) < 5.0 for a, b in zip(pos.position, target, strict=True))
