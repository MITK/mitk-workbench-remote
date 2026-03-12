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

"""Tests for node.py."""

import json

import pytest
import responses

from mitk_workbench_remote import errors
from mitk_workbench_remote.node import DataNode, _deserialize_property, _serialize_property
from mitk_workbench_remote.transport import RestTransport

BASE = "http://127.0.0.1:8080"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api(path: str) -> str:
    return f"{BASE}/api/v1{path}"


def _node() -> dict:
    return {
        "uid": "node_1",
        "name": "CT_Scan",
        "path": "/CT_Scan",
        "parent_uid": None,
        "data_type": "Image",
        "children_count": 0,
        "timestamp": 12345,
    }


def _prop_response(key: str, value: object) -> dict:
    """Build a single-property GET response body."""
    return {"data": {key: value}}


def _props_response(props: dict) -> dict:
    """Build a GET /properties response body."""
    return {"data": {"properties": props}}


def _make_node() -> DataNode:
    transport = RestTransport(BASE)
    return DataNode._from_node_dict(_node(), transport)


# ---------------------------------------------------------------------------
# _deserialize_property / _serialize_property unit tests
# ---------------------------------------------------------------------------


def test_deserialize_simple_string_passthrough() -> None:
    assert _deserialize_property("hello") == "hello"


def test_deserialize_bool_passthrough() -> None:
    assert _deserialize_property(True) is True


def test_deserialize_color_property_to_tuple() -> None:
    raw = {"type": "ColorProperty", "value": [1.0, 0.0, 0.0]}
    assert _deserialize_property(raw) == (1.0, 0.0, 0.0)


def test_deserialize_unknown_complex_type_passthrough() -> None:
    raw = {"type": "DoubleProperty", "value": 3.14}
    assert _deserialize_property(raw) == raw


def test_deserialize_dict_without_type_passthrough() -> None:
    raw = {"value": 3.14}
    assert _deserialize_property(raw) == raw


def test_serialize_color_to_wire_format() -> None:
    result = _serialize_property("color", (1.0, 0.0, 0.5))
    assert result == {"type": "ColorProperty", "value": [1.0, 0.0, 0.5]}


def test_serialize_simple_value_passthrough() -> None:
    assert _serialize_property("visible", True) is True
    assert _serialize_property("opacity", 0.5) == 0.5
    assert _serialize_property("name", "seg") == "seg"


def test_serialize_raw_dict_with_type_key_passthrough() -> None:
    wire = {"type": "DoubleProperty", "value": 3.14}
    assert _serialize_property("myProp", wire) is wire


# ---------------------------------------------------------------------------
# Identity (cached)
# ---------------------------------------------------------------------------


def test_uid_returns_cached_value() -> None:
    node = _make_node()
    assert node.uid == "node_1"


def test_data_type_returns_cached_value() -> None:
    node = _make_node()
    assert node.data_type == "Image"


def test_path_returns_cached_value() -> None:
    node = _make_node()
    assert node.path == "/CT_Scan"


# ---------------------------------------------------------------------------
# name
# ---------------------------------------------------------------------------


@responses.activate
def test_name_getter_calls_get_property_endpoint() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/name"),
        json=_prop_response("name", "CT_Scan"),
        status=200,
    )
    assert node.name == "CT_Scan"
    assert len(responses.calls) == 1


@responses.activate
def test_name_setter_calls_put_property_endpoint() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/name"),
        json={},
        status=200,
    )
    node.name = "NewName"
    assert len(responses.calls) == 1
    body = json.loads(responses.calls[0].request.body)
    assert body == "NewName"


@responses.activate
def test_name_setter_updates_display_cache() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/name"),
        json={},
        status=200,
    )
    node.name = "Updated"
    assert node._name == "Updated"


# ---------------------------------------------------------------------------
# visible
# ---------------------------------------------------------------------------


@responses.activate
def test_visible_getter_returns_bool() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/visible"),
        json=_prop_response("visible", True),
        status=200,
    )
    assert node.visible is True


@responses.activate
def test_visible_setter_sends_bool() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/visible"),
        json={},
        status=200,
    )
    node.visible = False
    body = json.loads(responses.calls[0].request.body)
    assert body is False


# ---------------------------------------------------------------------------
# opacity
# ---------------------------------------------------------------------------


@responses.activate
def test_opacity_getter_returns_float() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/opacity"),
        json=_prop_response("opacity", 0.75),
        status=200,
    )
    assert node.opacity == 0.75


@responses.activate
def test_opacity_setter_sends_float() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/opacity"),
        json={},
        status=200,
    )
    node.opacity = 0.5
    body = json.loads(responses.calls[0].request.body)
    assert body == 0.5


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


@responses.activate
def test_color_getter_deserializes_color_property() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/color"),
        json=_prop_response("color", {"type": "ColorProperty", "value": [1.0, 0.0, 0.0]}),
        status=200,
    )
    assert node.color == (1.0, 0.0, 0.0)


@responses.activate
def test_color_setter_serializes_color_property() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/color"),
        json={},
        status=200,
    )
    node.color = (0.0, 1.0, 0.5)
    body = json.loads(responses.calls[0].request.body)
    assert body == {"type": "ColorProperty", "value": [0.0, 1.0, 0.5]}


# ---------------------------------------------------------------------------
# update_properties
# ---------------------------------------------------------------------------


@responses.activate
def test_update_properties_patches_endpoint() -> None:
    node = _make_node()
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    node.update_properties(visible=True)
    assert len(responses.calls) == 1


@responses.activate
def test_update_properties_serializes_color() -> None:
    node = _make_node()
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    node.update_properties(color=(1.0, 0.0, 0.0))
    body = json.loads(responses.calls[0].request.body)
    assert body == {"color": {"type": "ColorProperty", "value": [1.0, 0.0, 0.0]}}


@responses.activate
def test_update_properties_passes_multiple_keys() -> None:
    node = _make_node()
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    node.update_properties(name="seg", visible=False)
    body = json.loads(responses.calls[0].request.body)
    assert body["name"] == "seg"
    assert body["visible"] is False


@responses.activate
def test_update_properties_updates_name_display_cache() -> None:
    node = _make_node()
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    node.update_properties(name="renamed")
    assert node._name == "renamed"


@responses.activate
def test_update_properties_passes_scope_param() -> None:
    node = _make_node()
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    node.update_properties(scope="data", visible=True)
    url = responses.calls[0].request.url
    assert "property_scope=data" in url


# ---------------------------------------------------------------------------
# get_properties
# ---------------------------------------------------------------------------


@responses.activate
def test_get_properties_returns_deserialized_dict() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties"),
        json=_props_response({"name": "CT_Scan", "visible": True}),
        status=200,
    )
    props = node.get_properties()
    assert props["name"] == "CT_Scan"
    assert props["visible"] is True


@responses.activate
def test_get_properties_passes_scope_param() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties"),
        json=_props_response({}),
        status=200,
    )
    node.get_properties(scope="node")
    assert "property_scope=node" in responses.calls[0].request.url


@responses.activate
def test_get_properties_passes_context_param() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties"),
        json=_props_response({}),
        status=200,
    )
    node.get_properties(context="renderer1")
    assert "context=renderer1" in responses.calls[0].request.url


# ---------------------------------------------------------------------------
# get_property
# ---------------------------------------------------------------------------


@responses.activate
def test_get_property_returns_scalar() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/opacity"),
        json=_prop_response("opacity", 0.8),
        status=200,
    )
    assert node.get_property("opacity") == 0.8


@responses.activate
def test_get_property_deserializes_color_property() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/color"),
        json=_prop_response("color", {"type": "ColorProperty", "value": [0.5, 0.5, 0.5]}),
        status=200,
    )
    assert node.get_property("color") == (0.5, 0.5, 0.5)


@responses.activate
def test_get_property_returns_raw_dict_for_unknown_complex() -> None:
    node = _make_node()
    raw = {"type": "DoubleProperty", "value": 3.14}
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/myProp"),
        json=_prop_response("myProp", raw),
        status=200,
    )
    result = node.get_property("myProp")
    assert result == raw


@responses.activate
def test_get_property_raises_on_404() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties/nonexistent"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "not found"}},
        status=404,
    )
    with pytest.raises(errors.NodeNotFoundError):
        node.get_property("nonexistent")


# ---------------------------------------------------------------------------
# set_property
# ---------------------------------------------------------------------------


@responses.activate
def test_set_property_sends_put_with_body() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/visible"),
        json={},
        status=200,
    )
    node.set_property("visible", True)
    body = json.loads(responses.calls[0].request.body)
    assert body is True


@responses.activate
def test_set_property_passes_scope_param() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/visible"),
        json={},
        status=200,
    )
    node.set_property("visible", True, scope="data")
    assert "property_scope=data" in responses.calls[0].request.url


@responses.activate
def test_set_property_passes_raw_dict_through() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/myProp"),
        json={},
        status=200,
    )
    wire = {"type": "DoubleProperty", "value": 3.14}
    node.set_property("myProp", wire)
    body = json.loads(responses.calls[0].request.body)
    assert body == wire


# ---------------------------------------------------------------------------
# delete_property
# ---------------------------------------------------------------------------


@responses.activate
def test_delete_property_sends_delete_request() -> None:
    node = _make_node()
    responses.add(
        responses.DELETE,
        _api("/datastorage/nodes/node_1/properties/myProp"),
        body=b"",
        status=204,
    )
    node.delete_property("myProp")
    assert len(responses.calls) == 1


@responses.activate
def test_delete_property_passes_scope_param() -> None:
    node = _make_node()
    responses.add(
        responses.DELETE,
        _api("/datastorage/nodes/node_1/properties/myProp"),
        body=b"",
        status=204,
    )
    node.delete_property("myProp", scope="data")
    assert "property_scope=data" in responses.calls[0].request.url


# ---------------------------------------------------------------------------
# children
# ---------------------------------------------------------------------------


@responses.activate
def test_children_returns_list_of_data_nodes() -> None:
    node = _make_node()
    child_dict = {
        "uid": "node_2",
        "name": "seg",
        "path": "/CT_Scan/seg",
        "parent_uid": "node_1",
        "data_type": "Image",
        "children_count": 0,
        "timestamp": 12346,
    }
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/children"),
        json={"data": [child_dict]},
        status=200,
    )
    children = node.children
    assert len(children) == 1
    assert isinstance(children[0], DataNode)
    assert children[0].uid == "node_2"


@responses.activate
def test_children_returns_empty_list() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/children"),
        json={"data": []},
        status=200,
    )
    assert node.children == []


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


@responses.activate
def test_remove_sends_delete() -> None:
    node = _make_node()
    responses.add(
        responses.DELETE,
        _api("/datastorage/nodes/node_1"),
        body=b"",
        status=204,
    )
    node.remove()
    assert len(responses.calls) == 1


@responses.activate
def test_remove_recursive_passes_query_param() -> None:
    node = _make_node()
    responses.add(
        responses.DELETE,
        _api("/datastorage/nodes/node_1"),
        body=b"",
        status=204,
    )
    node.remove(recursive=True)
    assert "recursive=true" in responses.calls[0].request.url


@responses.activate
def test_remove_raises_on_node_not_found() -> None:
    node = _make_node()
    responses.add(
        responses.DELETE,
        _api("/datastorage/nodes/node_1"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "not found"}},
        status=404,
    )
    with pytest.raises(errors.NodeNotFoundError):
        node.remove()


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


@responses.activate
def test_refresh_updates_data_type() -> None:
    node = _make_node()
    assert node.data_type == "Image"
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": {**_node(), "data_type": "Surface"}},
        status=200,
    )
    node.refresh()
    assert node.data_type == "Surface"


@responses.activate
def test_refresh_updates_path() -> None:
    node = _make_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": {**_node(), "path": "/new/path"}},
        status=200,
    )
    node.refresh()
    assert node.path == "/new/path"


# ---------------------------------------------------------------------------
# repr (no REST calls)
# ---------------------------------------------------------------------------


def test_repr_includes_uid() -> None:
    node = _make_node()
    assert "node_1" in repr(node)


def test_repr_includes_name() -> None:
    node = _make_node()
    assert "CT_Scan" in repr(node)


def test_repr_includes_path() -> None:
    node = _make_node()
    assert "/CT_Scan" in repr(node)


def test_repr_html_returns_string() -> None:
    node = _make_node()
    assert isinstance(node._repr_html_(), str)


def test_repr_html_contains_uid() -> None:
    node = _make_node()
    assert "node_1" in node._repr_html_()


def test_repr_html_contains_parent_uid() -> None:
    transport = RestTransport(BASE)
    node = DataNode._from_node_dict({**_node(), "parent_uid": "parent_node"}, transport)
    assert "parent_node" in node._repr_html_()


# ---------------------------------------------------------------------------
# reinit
# ---------------------------------------------------------------------------


@responses.activate
def test_reinit_posts_to_rendering_reinit_endpoint() -> None:
    node = _make_node()
    responses.add(
        responses.POST,
        _api("/rendering/reinit"),
        body=b"",
        status=204,
    )
    node.reinit()
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == _api("/rendering/reinit")


@responses.activate
def test_reinit_sends_own_uid_in_body() -> None:
    node = _make_node()
    responses.add(
        responses.POST,
        _api("/rendering/reinit"),
        body=b"",
        status=204,
    )
    node.reinit()
    body = json.loads(responses.calls[0].request.body)
    assert body == {"uids": ["node_1"]}
