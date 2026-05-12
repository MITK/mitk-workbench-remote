# SPDX-FileCopyrightText: 2026, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
#
# SPDX-License-Identifier: Apache-2.0

"""RenderWindow per-window node-property helpers route through DataNode + context."""

from __future__ import annotations

import json

import pytest
import responses

from mitk_workbench_remote import MxNRenderWindow
from mitk_workbench_remote.node import DataNode
from mitk_workbench_remote.transport import RestTransport

BASE = "http://127.0.0.1:8080"


def _api(path: str) -> str:
    return f"{BASE}/api/v1{path}"


def _transport() -> RestTransport:
    return RestTransport(BASE, token="t", transfer_mode="direct")


def _make_node(transport: RestTransport) -> DataNode:
    return DataNode(
        uid="node_1",
        name="CT",
        data_type="Image",
        path="/CT",
        parent_uid=None,
        transport=transport,
    )


@responses.activate
def test_set_node_visible_routes_through_update_properties_with_context() -> None:
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        body=b"",
        status=204,
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    win.set_node_visible(_make_node(t), False)
    request = responses.calls[0].request
    assert json.loads(request.body) == {"visible": False}
    assert "context=mxn__widget0" in (request.url or "")


@responses.activate
def test_is_node_visible_reads_with_context() -> None:
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/visible"),
        json={"data": {"visible": True}},
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    assert win.is_node_visible(_make_node(t)) is True
    request = responses.calls[0].request
    assert "context=mxn__widget0" in (request.url or "")


@responses.activate
def test_set_node_layer_routes_through_update_properties() -> None:
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        body=b"",
        status=204,
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    win.set_node_layer(_make_node(t), 5)
    request = responses.calls[0].request
    assert json.loads(request.body) == {"layer": 5}
    assert "context=mxn__widget0" in (request.url or "")


@responses.activate
def test_get_node_layer_reads_with_context() -> None:
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/layer"),
        json={"data": {"layer": 7}},
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    assert win.get_node_layer(_make_node(t)) == 7


# ---------------------------------------------------------------------------
# PropertyScope must be NODE on both write and read so a like-named DATA-scope
# property cannot shadow the value just written. Asymmetric scopes between
# update_properties() (NODE default) and get_property() (ALL default) would
# otherwise let an unrelated DATA-scope ``visible``/``layer`` win the read.
# ---------------------------------------------------------------------------


@responses.activate
def test_set_node_visible_uses_node_scope() -> None:
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        body=b"",
        status=204,
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    win.set_node_visible(_make_node(t), True)
    url = responses.calls[0].request.url or ""
    assert "property_scope=node" in url


@responses.activate
def test_is_node_visible_uses_node_scope() -> None:
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/visible"),
        json={"data": {"visible": True}},
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    win.is_node_visible(_make_node(t))
    url = responses.calls[0].request.url or ""
    assert "property_scope=node" in url


@responses.activate
def test_set_node_layer_uses_node_scope() -> None:
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        body=b"",
        status=204,
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    win.set_node_layer(_make_node(t), 3)
    url = responses.calls[0].request.url or ""
    assert "property_scope=node" in url


@responses.activate
def test_get_node_layer_uses_node_scope() -> None:
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/layer"),
        json={"data": {"layer": 4}},
    )
    t = _transport()
    win = MxNRenderWindow(t, "mxn", "mxn__widget0")
    win.get_node_layer(_make_node(t))
    url = responses.calls[0].request.url or ""
    assert "property_scope=node" in url


# ---------------------------------------------------------------------------
# Cross-Workbench guard: a DataNode whose transport differs from the window's
# would silently target the wrong server. CLAUDE.md guarantees independent
# Workbench instances are independent, so the helpers must reject this.
# ---------------------------------------------------------------------------


def test_set_node_visible_rejects_node_from_different_transport() -> None:
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    foreign = _make_node(_transport())  # distinct transport instance
    with pytest.raises(ValueError, match="different Workbench transport"):
        win.set_node_visible(foreign, True)


def test_is_node_visible_rejects_node_from_different_transport() -> None:
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    foreign = _make_node(_transport())
    with pytest.raises(ValueError, match="different Workbench transport"):
        win.is_node_visible(foreign)


def test_set_node_layer_rejects_node_from_different_transport() -> None:
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    foreign = _make_node(_transport())
    with pytest.raises(ValueError, match="different Workbench transport"):
        win.set_node_layer(foreign, 1)


def test_get_node_layer_rejects_node_from_different_transport() -> None:
    win = MxNRenderWindow(_transport(), "mxn", "mxn__widget0")
    foreign = _make_node(_transport())
    with pytest.raises(ValueError, match="different Workbench transport"):
        win.get_node_layer(foreign)
