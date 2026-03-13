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

"""Tests for storage.py."""

import json

import pytest
import responses

from mitk_workbench_remote import errors
from mitk_workbench_remote.node import DataNode, PropertyScope
from mitk_workbench_remote.storage import DataStorage
from mitk_workbench_remote.transport import RestTransport

BASE = "http://127.0.0.1:8080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api(path: str) -> str:
    return f"{BASE}/api/v1{path}"


def _node_dict(**overrides: object) -> dict:
    base: dict = {
        "uid": "node_1",
        "name": "CT_Scan",
        "path": "/CT_Scan",
        "parent_uid": None,
        "data_type": "Image",
        "children_count": 0,
        "timestamp": 12345,
    }
    base.update(overrides)
    return base


def _list_response(*nodes: dict) -> dict:
    return {
        "data": list(nodes),
        "meta": {"total_count": len(nodes)},
    }


def _make_storage() -> DataStorage:
    transport = RestTransport(BASE)
    return DataStorage(transport)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@responses.activate
def test_list_returns_data_nodes() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict(), _node_dict(uid="node_2", name="Seg")),
        status=200,
    )
    nodes = storage.list()
    assert len(nodes) == 2
    assert all(isinstance(n, DataNode) for n in nodes)
    assert nodes[0].uid == "node_1"
    assert nodes[1].uid == "node_2"


@responses.activate
def test_list_empty_storage() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json={"data": [], "meta": {"total_count": 0}},
        status=200,
    )
    assert storage.list() == []


@responses.activate
def test_list_filters_by_data_type() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.list(data_type="Image")
    assert "data_type=Image" in responses.calls[0].request.url


@responses.activate
def test_list_toplevel_only() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.list(toplevel=True)
    assert "hierarchy=toplevel" in responses.calls[0].request.url


@responses.activate
def test_list_paginates_when_results_exceed_one_page() -> None:
    """list() must issue multiple requests until all nodes are fetched."""
    storage = _make_storage()
    # First page: 2 nodes, total_count says 3
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json={
            "data": [_node_dict(uid="node_1"), _node_dict(uid="node_2")],
            "meta": {"total_count": 3},
        },
        status=200,
    )
    # Second page: remaining 1 node
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json={
            "data": [_node_dict(uid="node_3")],
            "meta": {"total_count": 3},
        },
        status=200,
    )
    nodes = storage.list()
    assert len(nodes) == 3
    assert [n.uid for n in nodes] == ["node_1", "node_2", "node_3"]
    assert len(responses.calls) == 2
    assert "limit=1000" in responses.calls[0].request.url
    assert "offset=0" in responses.calls[0].request.url
    assert "offset=2" in responses.calls[1].request.url


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


@responses.activate
def test_filter_returns_data_nodes() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    nodes = storage.filter(properties={"visible": True})
    assert len(nodes) == 1
    assert isinstance(nodes[0], DataNode)


@responses.activate
def test_filter_by_data_type() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(data_type="Image")
    assert "data_type=Image" in responses.calls[0].request.url


@responses.activate
def test_filter_by_toplevel() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(toplevel=True)
    assert "hierarchy=toplevel" in responses.calls[0].request.url


@responses.activate
def test_filter_by_parent_uid() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(parent_uid="node_0")
    assert "parent_uid=node_0" in responses.calls[0].request.url


@responses.activate
def test_filter_property_bool_serialized_to_lowercase() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(properties={"visible": True})
    assert "filter.visible=true" in responses.calls[0].request.url


@responses.activate
def test_filter_property_string_passes_through_with_wildcard() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(properties={"name": "CT*"})
    assert (
        "filter.name=CT%2A" in responses.calls[0].request.url
        or "filter.name=CT*" in responses.calls[0].request.url
    )


@responses.activate
def test_filter_property_color_serialized_as_json() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(properties={"color": (1.0, 0.0, 0.0)})
    url = responses.calls[0].request.url
    assert "filter.color=" in url


@responses.activate
def test_filter_passes_property_scope_when_properties_given() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(properties={"visible": True}, scope=PropertyScope.NODE)
    assert "property_scope=node" in responses.calls[0].request.url


@responses.activate
def test_filter_omits_property_scope_when_no_properties() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(data_type="Image")
    assert "property_scope=" not in responses.calls[0].request.url


@responses.activate
def test_filter_passes_context_when_properties_given() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(properties={"visible": True}, context="renderer1")
    assert "context=renderer1" in responses.calls[0].request.url


@responses.activate
def test_filter_omits_context_when_none() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(properties={"visible": True})
    assert "context=" not in responses.calls[0].request.url


@responses.activate
def test_filter_omits_context_when_no_properties() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    storage.filter(context="renderer1")
    assert "context=" not in responses.calls[0].request.url


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


@responses.activate
def test_get_returns_data_node() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": _node_dict()},
        status=200,
    )
    node = storage.get("node_1")
    assert isinstance(node, DataNode)
    assert node.uid == "node_1"


@responses.activate
def test_get_raises_node_not_found() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/missing"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "not found"}},
        status=404,
    )
    with pytest.raises(errors.NodeNotFoundError):
        storage.get("missing")


# ---------------------------------------------------------------------------
# create — top-level
# ---------------------------------------------------------------------------


@responses.activate
def test_create_posts_to_nodes_endpoint() -> None:
    storage = _make_storage()
    responses.add(
        responses.POST,
        _api("/datastorage/nodes"),
        json={"data": _node_dict(uid="new_node", name="NewNode")},
        status=201,
    )
    storage.create("NewNode")
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url.endswith("/datastorage/nodes")


@responses.activate
def test_create_returns_data_node() -> None:
    storage = _make_storage()
    responses.add(
        responses.POST,
        _api("/datastorage/nodes"),
        json={"data": _node_dict(uid="new_node", name="NewNode")},
        status=201,
    )
    node = storage.create("NewNode")
    assert isinstance(node, DataNode)
    assert node.uid == "new_node"


@responses.activate
def test_create_sends_name_in_body() -> None:
    storage = _make_storage()
    responses.add(
        responses.POST,
        _api("/datastorage/nodes"),
        json={"data": _node_dict(name="NewNode")},
        status=201,
    )
    storage.create("NewNode")
    body = json.loads(responses.calls[0].request.body)
    assert body["name"] == "NewNode"


# ---------------------------------------------------------------------------
# create — child node
# ---------------------------------------------------------------------------


@responses.activate
def test_create_child_from_datanode_parent() -> None:
    storage = _make_storage()
    transport = RestTransport(BASE)
    parent = DataNode._from_node_dict(_node_dict(uid="parent_1"), transport)
    responses.add(
        responses.POST,
        _api("/datastorage/nodes/parent_1/children"),
        json={"data": _node_dict(uid="child_1", parent_uid="parent_1")},
        status=201,
    )
    child = storage.create("ChildNode", parent=parent)
    assert isinstance(child, DataNode)
    assert child.uid == "child_1"


@responses.activate
def test_create_child_from_uid_string_parent() -> None:
    storage = _make_storage()
    responses.add(
        responses.POST,
        _api("/datastorage/nodes/parent_1/children"),
        json={"data": _node_dict(uid="child_1")},
        status=201,
    )
    child = storage.create("ChildNode", parent="parent_1")
    assert isinstance(child, DataNode)


@responses.activate
def test_create_child_posts_to_children_endpoint() -> None:
    storage = _make_storage()
    responses.add(
        responses.POST,
        _api("/datastorage/nodes/parent_1/children"),
        json={"data": _node_dict(uid="child_1")},
        status=201,
    )
    storage.create("ChildNode", parent="parent_1")
    assert "/datastorage/nodes/parent_1/children" in responses.calls[0].request.url


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


@responses.activate
def test_len_returns_total_count() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict(), _node_dict(uid="node_2")),
        status=200,
    )
    assert len(storage) == 2


@responses.activate
def test_len_returns_zero_when_empty() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json={"data": [], "meta": {"total_count": 0}},
        status=200,
    )
    assert len(storage) == 0


# ---------------------------------------------------------------------------
# __iter__
# ---------------------------------------------------------------------------


@responses.activate
def test_iter_yields_data_nodes() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict(), _node_dict(uid="node_2")),
        status=200,
    )
    nodes = list(storage)
    assert len(nodes) == 2
    assert all(isinstance(n, DataNode) for n in nodes)


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


@responses.activate
def test_contains_returns_true_for_existing_uid() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": _node_dict()},
        status=200,
    )
    assert "node_1" in storage


@responses.activate
def test_contains_returns_false_for_missing_uid() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/missing"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "not found"}},
        status=404,
    )
    assert "missing" not in storage


@responses.activate
def test_contains_returns_true_for_data_node_instance() -> None:
    storage = _make_storage()
    transport = RestTransport(BASE)
    node = DataNode._from_node_dict(_node_dict(), transport)
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": _node_dict()},
        status=200,
    )
    assert node in storage


def test_contains_returns_false_for_non_string_non_node() -> None:
    storage = _make_storage()
    assert (42 in storage) is False
    assert (None in storage) is False
    assert ([] in storage) is False


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


@responses.activate
def test_getitem_returns_data_node() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": _node_dict()},
        status=200,
    )
    node = storage["node_1"]
    assert isinstance(node, DataNode)
    assert node.uid == "node_1"


@responses.activate
def test_getitem_raises_node_not_found() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/missing"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "not found"}},
        status=404,
    )
    with pytest.raises(errors.NodeNotFoundError):
        _ = storage["missing"]


# ---------------------------------------------------------------------------
# repr (no REST calls except _repr_html_)
# ---------------------------------------------------------------------------


def test_repr_is_string() -> None:
    storage = _make_storage()
    assert isinstance(repr(storage), str)


def test_repr_contains_url() -> None:
    storage = _make_storage()
    assert BASE in repr(storage)


@responses.activate
def test_repr_html_returns_string() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    html = storage._repr_html_()
    assert isinstance(html, str)


@responses.activate
def test_repr_html_contains_node_uid() -> None:
    storage = _make_storage()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes"),
        json=_list_response(_node_dict()),
        status=200,
    )
    html = storage._repr_html_()
    assert "node_1" in html
