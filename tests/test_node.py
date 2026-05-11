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
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import responses

from mitk_workbench_remote import errors
from mitk_workbench_remote.errors import UnsupportedDataTypeError
from mitk_workbench_remote.image import Image
from mitk_workbench_remote.node import (
    DataNode,
    DataRepresentation,
    PropertyScope,
    _deserialize_property,
    _serialize_property,
)
from mitk_workbench_remote.transport import (
    FileAccessConfig,
    FileAccessMode,
    RestTransport,
    TransferMode,
)

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
    node.update_properties(scope=PropertyScope.DATA, visible=True)
    url = responses.calls[0].request.url
    assert "property_scope=data" in url


@responses.activate
def test_update_properties_passes_context_param() -> None:
    node = _make_node()
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    node.update_properties(context="renderer1", visible=True)
    assert "context=renderer1" in responses.calls[0].request.url


@responses.activate
def test_update_properties_omits_context_when_none() -> None:
    node = _make_node()
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    node.update_properties(visible=True)
    assert "context=" not in responses.calls[0].request.url


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
    node.get_properties(scope=PropertyScope.NODE)
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
    node.set_property("visible", True, scope=PropertyScope.DATA)
    assert "property_scope=data" in responses.calls[0].request.url


@responses.activate
def test_set_property_passes_context_param() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/opacity"),
        json={},
        status=200,
    )
    node.set_property("opacity", 0.5, context="renderer2")
    assert "context=renderer2" in responses.calls[0].request.url


@responses.activate
def test_set_property_omits_context_when_none() -> None:
    node = _make_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/properties/opacity"),
        json={},
        status=200,
    )
    node.set_property("opacity", 0.5)
    assert "context=" not in responses.calls[0].request.url


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
# remove_property
# ---------------------------------------------------------------------------


@responses.activate
def test_remove_property_sends_delete_request() -> None:
    node = _make_node()
    responses.add(
        responses.DELETE,
        _api("/datastorage/nodes/node_1/properties/myProp"),
        body=b"",
        status=204,
    )
    node.remove_property("myProp")
    assert len(responses.calls) == 1


@responses.activate
def test_remove_property_passes_scope_param() -> None:
    node = _make_node()
    responses.add(
        responses.DELETE,
        _api("/datastorage/nodes/node_1/properties/myProp"),
        body=b"",
        status=204,
    )
    node.remove_property("myProp", scope=PropertyScope.DATA)
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
        json={"data": [child_dict], "meta": {"total_count": 1}},
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
        json={"data": [], "meta": {"total_count": 0}},
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


# ---------------------------------------------------------------------------
# Helpers for data transfer tests
# ---------------------------------------------------------------------------


def _make_nrrd_bytes() -> bytes:
    """Create minimal NRRD bytes from a small array."""
    from mitk_workbench_remote._io import write_nrrd

    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    img = Image(arr, spacing=(0.5, 1.0, 2.0), origin=(1.0, 2.0, 3.0))
    return write_nrrd(img)


def _make_direct_node(data_type: str = "Image") -> DataNode:
    """Create a DataNode with transfer_mode=DIRECT."""
    transport = RestTransport(BASE, transfer_mode=TransferMode.DIRECT)
    return DataNode._from_node_dict({**_node(), "data_type": data_type}, transport)


def _make_file_reference_node(data_type: str = "Image") -> DataNode:
    """Create a DataNode with transfer_mode=FILE_REFERENCE."""
    transport = RestTransport(BASE, transfer_mode=TransferMode.FILE_REFERENCE)
    return DataNode._from_node_dict({**_node(), "data_type": data_type}, transport)


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------


@responses.activate
def test_get_data_direct_mode() -> None:
    node = _make_direct_node()
    nrrd_bytes = _make_nrrd_bytes()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    image = node.get_data(as_type=DataRepresentation.REMOTE)
    assert isinstance(image, Image)
    assert image.shape == (2, 3, 4)
    assert np.allclose(image.spacing, (0.5, 1.0, 2.0))


@responses.activate
def test_get_data_file_reference_mode(tmp_path: Path) -> None:
    node = _make_file_reference_node()
    nrrd_bytes = _make_nrrd_bytes()
    nrrd_file = tmp_path / "data.nrrd"
    nrrd_file.write_bytes(nrrd_bytes)

    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        json={"transfer": {"file_path": str(nrrd_file)}},
        status=200,
        content_type="application/json",
    )
    image = node.get_data(as_type=DataRepresentation.REMOTE)
    assert isinstance(image, Image)
    assert image.shape == (2, 3, 4)


@responses.activate
def test_get_data_include_properties_false() -> None:
    node = _make_direct_node()
    nrrd_bytes = _make_nrrd_bytes()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    image = node.get_data(include_properties=False)
    # properties should not contain server-fetched properties
    # (may contain NRRD custom fields, but not node properties)
    assert "name" not in image.properties


@responses.activate
def test_get_data_include_properties_true() -> None:
    node = _make_direct_node()
    nrrd_bytes = _make_nrrd_bytes()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/properties"),
        json=_props_response({"imagescalar.min": 0.0, "imagescalar.max": 255.0}),
        status=200,
    )
    image = node.get_data(include_properties=True, as_type=DataRepresentation.REMOTE)
    assert image.get_property("imagescalar.min") == 0.0
    assert image.get_property("imagescalar.max") == 255.0


@responses.activate
def test_get_data_no_data_raises() -> None:
    node = _make_direct_node()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        json={"error": {"code": "NODE_NOT_FOUND", "message": "no data"}},
        status=404,
    )
    with pytest.raises(errors.NodeNotFoundError):
        node.get_data()


def test_get_data_unsupported_data_type_raises() -> None:
    node = _make_direct_node(data_type="Surface")
    with pytest.raises(UnsupportedDataTypeError, match="Surface"):
        node.get_data()


def test_get_data_none_data_type_raises() -> None:
    transport = RestTransport(BASE, transfer_mode=TransferMode.DIRECT)
    node = DataNode._from_node_dict({**_node(), "data_type": None}, transport)
    with pytest.raises(UnsupportedDataTypeError, match="None"):
        node.get_data()


def test_get_data_unsupported_type_does_not_make_http_call() -> None:
    """Early validation must reject before any HTTP request."""
    node = _make_direct_node(data_type="PointSet")
    # No responses registered -- if HTTP were attempted, responses library
    # would raise ConnectionError.
    with pytest.raises(UnsupportedDataTypeError):
        node.get_data()


@responses.activate
def test_get_data_multilabel_segmentation_direct_mode() -> None:
    from mitk_workbench_remote._io import write_multilabel_nrrd
    from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

    seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
    g = seg.add_group("Organs")
    seg.add_label(Label(1, "Liver"), group=g)
    nrrd_bytes = write_multilabel_nrrd(seg)

    node = _make_direct_node(data_type="MultiLabelSegmentation")
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    # Use REMOTE to always get mw type regardless of mitk availability.
    result = node.get_data(as_type=DataRepresentation.REMOTE)
    assert isinstance(result, MultiLabelSegmentation)
    assert len(result.groups) == 1
    assert result.get_label(1) is not None


# ---------------------------------------------------------------------------
# save_data
# ---------------------------------------------------------------------------


@responses.activate
def test_save_data_direct_mode(tmp_path: Path) -> None:
    node = _make_direct_node(data_type="Surface")
    nrrd_bytes = _make_nrrd_bytes()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    dest = tmp_path / "output.nrrd"
    result = node.save_data(dest)
    assert result == dest
    assert dest.read_bytes() == nrrd_bytes


@responses.activate
def test_save_data_file_reference_mode(tmp_path: Path) -> None:
    node = _make_file_reference_node(data_type="Surface")
    nrrd_bytes = _make_nrrd_bytes()
    source_file = tmp_path / "server_data.nrrd"
    source_file.write_bytes(nrrd_bytes)

    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        json={"transfer": {"file_path": str(source_file)}},
        status=200,
        content_type="application/json",
    )
    dest = tmp_path / "output.nrrd"
    result = node.save_data(dest)
    assert result == dest
    assert dest.read_bytes() == nrrd_bytes


@responses.activate
def test_save_data_works_for_unsupported_data_type(tmp_path: Path) -> None:
    """save_data must work even for data types that get_data cannot handle."""
    node = _make_direct_node(data_type="PointSet")
    fake_bytes = b"NRRD0004\nfake pointset data"
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=fake_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    dest = tmp_path / "pointset.nrrd"
    node.save_data(dest)
    assert dest.read_bytes() == fake_bytes


# ---------------------------------------------------------------------------
# set_data
# ---------------------------------------------------------------------------


@responses.activate
def test_set_data_direct_mode_image() -> None:
    node = _make_direct_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/data"),
        json={},
        status=200,
    )
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    img = Image(arr)
    node.set_data(img)
    assert len(responses.calls) == 1
    req = responses.calls[0].request
    assert req.headers.get("Content-Type") == "application/octet-stream"
    assert 'filename="data.nrrd"' in req.headers.get("Content-Disposition", "")


@responses.activate
def test_set_data_direct_mode_ndarray() -> None:
    node = _make_direct_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/data"),
        json={},
        status=200,
    )
    # numpy arrays go through the converter registry, so set_data calls refresh()
    # to determine the resulting data_type from the server.
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": _node()},
        status=200,
    )
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    node.set_data(arr)
    # Body should be NRRD bytes (starts with NRRD magic)
    assert responses.calls[0].request.body[:4] == b"NRRD"


def _patch_file_access(node: DataNode, config: FileAccessConfig):  # type: ignore[type-arg]
    """Context manager to mock file_access_config on a node's transport."""
    return patch.object(
        type(node._transport),
        "file_access_config",
        new_callable=PropertyMock,
        return_value=config,
    )


_UNRESTRICTED = FileAccessConfig(
    mode=FileAccessMode.UNRESTRICTED,
    restrictions_active=False,
    max_active_temp_dirs_per_ip=5,
)


@responses.activate
def test_set_data_file_reference_mode_image(tmp_path: Path) -> None:
    node = _make_file_reference_node()
    with _patch_file_access(node, _UNRESTRICTED):
        responses.add(
            responses.PUT,
            _api("/datastorage/nodes/node_1/data"),
            json={},
            status=200,
        )
        arr = np.zeros((3, 4, 5), dtype=np.float32)
        img = Image(arr)
        node.set_data(img)
        body = json.loads(responses.calls[0].request.body)
        assert body["transfer"]["mode"] == "file-reference"
        # Temp file should have been cleaned up
        file_path = body["transfer"]["file_path"]
        assert not Path(file_path).exists()


def test_set_data_unsupported_type() -> None:
    node = _make_direct_node()
    with pytest.raises(TypeError, match="No converter found"):
        node.set_data(42)


def test_set_data_rejects_file_path() -> None:
    node = _make_direct_node()
    with pytest.raises(TypeError, match="No converter found"):
        node.set_data("/some/path/data.nrrd")


@responses.activate
def test_set_data_include_properties() -> None:
    node = _make_direct_node()
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/data"),
        json={},
        status=200,
    )
    responses.add(
        responses.PATCH,
        _api("/datastorage/nodes/node_1/properties"),
        json={},
        status=200,
    )
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    img = Image(arr, properties={"my_prop": "my_value"})
    node.set_data(img, include_properties=True)
    # Should have made 2 calls: PUT data + PATCH properties
    assert len(responses.calls) == 2
    patch_body = json.loads(responses.calls[1].request.body)
    assert patch_body.get("my_prop") == "my_value"


# ---------------------------------------------------------------------------
# set_data — data_type cache update
# ---------------------------------------------------------------------------


@responses.activate
def test_set_data_updates_data_type_to_image() -> None:
    """set_data with an Image sets data_type locally — no refresh call."""
    node = _make_direct_node(data_type=None)
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/data"),
        json={},
        status=200,
    )
    img = Image(np.zeros((3, 4, 5), dtype=np.float32))
    node.set_data(img)
    assert node.data_type == "Image"
    assert len(responses.calls) == 1  # only the PUT, no refresh


@responses.activate
def test_set_data_updates_data_type_to_multilabel_segmentation() -> None:
    """set_data with a MultiLabelSegmentation sets data_type locally — no refresh call."""
    from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

    node = _make_direct_node(data_type=None)
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/data"),
        json={},
        status=200,
    )
    seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
    seg.add_label(Label(1, "Organ"), group=seg.add_group("Group1"))
    node.set_data(seg)
    assert node.data_type == "MultiLabelSegmentation"
    assert len(responses.calls) == 1  # only the PUT, no refresh


@responses.activate
def test_set_data_refreshes_data_type_for_ndarray() -> None:
    """set_data with a numpy array calls refresh() to get the data_type from the server."""
    node = _make_direct_node(data_type=None)
    responses.add(
        responses.PUT,
        _api("/datastorage/nodes/node_1/data"),
        json={},
        status=200,
    )
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1"),
        json={"data": {**_node(), "data_type": "Image"}},
        status=200,
    )
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    node.set_data(arr)
    assert node.data_type == "Image"
    assert len(responses.calls) == 2  # PUT data + GET refresh


# ---------------------------------------------------------------------------
# get_data -- DataRepresentation (WP-8)
# ---------------------------------------------------------------------------


@responses.activate
def test_get_data_image_remote_returns_mw_image() -> None:
    """as_type=REMOTE always returns mw.Image regardless of mitk availability."""
    node = _make_direct_node()
    nrrd_bytes = _make_nrrd_bytes()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    result = node.get_data(as_type=DataRepresentation.REMOTE)
    assert isinstance(result, Image)


@responses.activate
def test_get_data_image_auto_without_mitk_returns_mw_image() -> None:
    """AUTO falls back to mw.Image when mitk is not available."""
    import sys
    from unittest.mock import patch

    node = _make_direct_node()
    nrrd_bytes = _make_nrrd_bytes()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )

    with patch.dict(sys.modules, {"mitk": None}):
        result = node.get_data(as_type=DataRepresentation.AUTO)

    assert isinstance(result, Image)


@responses.activate
def test_get_data_image_mitk_raises_when_mitk_absent() -> None:
    """as_type=MITK propagates ImportError when mitk is not installed."""
    from unittest.mock import patch

    node = _make_direct_node()
    nrrd_bytes = _make_nrrd_bytes()
    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )

    with (
        patch.object(Image, "to_mitk", side_effect=ImportError("mitk not available")),
        pytest.raises(ImportError),
    ):
        node.get_data(as_type=DataRepresentation.MITK)


@responses.activate
def test_get_data_multilabel_remote_returns_mw_mls() -> None:
    """as_type=REMOTE always returns mw.MultiLabelSegmentation."""
    from mitk_workbench_remote._io import write_multilabel_nrrd
    from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

    node = _make_direct_node(data_type="MultiLabelSegmentation")
    seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
    seg.add_label(Label(1, "Organ"), group=seg.add_group("Group1"))
    nrrd_bytes = write_multilabel_nrrd(seg)

    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    result = node.get_data(as_type=DataRepresentation.REMOTE)
    assert isinstance(result, MultiLabelSegmentation)
    assert result.get_label(1) is not None


@responses.activate
def test_get_data_multilabel_auto_without_mitk_returns_mw_mls() -> None:
    """AUTO falls back to mw.MultiLabelSegmentation when mitk is not available."""
    import sys
    from unittest.mock import patch

    from mitk_workbench_remote._io import write_multilabel_nrrd
    from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

    node = _make_direct_node(data_type="MultiLabelSegmentation")
    seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
    seg.add_label(Label(1, "Organ"), group=seg.add_group("Group1"))
    nrrd_bytes = write_multilabel_nrrd(seg)

    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    with patch.dict(sys.modules, {"mitk": None}):
        result = node.get_data(as_type=DataRepresentation.AUTO)

    assert isinstance(result, MultiLabelSegmentation)
    assert result.get_label(1) is not None


@responses.activate
def test_get_data_multilabel_mitk_raises_when_mitk_absent() -> None:
    """as_type=MITK raises ImportError for MLS when mitk is not installed."""
    import sys
    from unittest.mock import patch

    from mitk_workbench_remote._io import write_multilabel_nrrd
    from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

    node = _make_direct_node(data_type="MultiLabelSegmentation")
    seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
    seg.add_label(Label(1, "Organ"), group=seg.add_group("Group1"))
    nrrd_bytes = write_multilabel_nrrd(seg)

    responses.add(
        responses.GET,
        _api("/datastorage/nodes/node_1/data"),
        body=nrrd_bytes,
        status=200,
        content_type="application/octet-stream",
    )
    with patch.dict(sys.modules, {"mitk": None}), pytest.raises(ImportError):
        node.get_data(as_type=DataRepresentation.MITK)


# ---------------------------------------------------------------------------
# get_data -- DataRepresentation with mitk installed
# ---------------------------------------------------------------------------


class TestGetDataWithMitk:
    """Tests that require the ``mitk`` package to be installed."""

    @pytest.fixture(autouse=True)
    def require_mitk(self) -> None:
        pytest.importorskip("mitk")

    @responses.activate
    def test_get_data_image_auto_with_mitk_returns_mitk_image(self) -> None:
        import mitk

        node = _make_direct_node()
        nrrd_bytes = _make_nrrd_bytes()
        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/data"),
            body=nrrd_bytes,
            status=200,
            content_type="application/octet-stream",
        )
        result = node.get_data(as_type=DataRepresentation.AUTO)
        assert isinstance(result, mitk.Image)

    @responses.activate
    def test_get_data_image_include_properties_routes_to_mitk_auto_wrap(self) -> None:
        import mitk

        node = _make_direct_node()
        nrrd_bytes = _make_nrrd_bytes()

        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/data"),
            body=nrrd_bytes,
            status=200,
            content_type="application/octet-stream",
        )
        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/properties"),
            json={"data": {"properties": {"name": "X"}}},
            status=200,
        )

        result = node.get_data(as_type=DataRepresentation.MITK, include_properties=True)
        assert isinstance(result, mitk.Image)

        # Default (raw=False): coerced to plain Python str
        assert result.get_property("name") == "X"

        # Explicit raw=False: same coerced value
        assert result.get_property("name", raw=False) == "X"

        # raw=True: returns the underlying mitk.StringProperty object
        raw_prop = result.get_property("name", raw=True)
        assert isinstance(raw_prop, mitk.StringProperty)
        assert raw_prop.value == "X"

    @responses.activate
    def test_get_data_image_include_properties_dict_form_via_from_json(self) -> None:
        """Cover the ``from_json`` branch in ``_apply_remote_properties_to_mitk``.

        Uses ``TemporoSpatialStringProperty`` because it is *not* listed in
        ``properties._COMPLEX_DESERIALIZERS`` -- so its dict-form wire payload
        survives ``_deserialize_property`` unchanged and reaches
        ``_apply_remote_properties_to_mitk`` as a raw dict. That is the only
        input shape that triggers the ``mitk.BaseProperty.from_json()`` call.
        ``ColorProperty`` would not exercise this branch because the
        deserializer coerces it to a tuple first (auto-wrap path).
        """
        import mitk

        node = _make_direct_node()
        nrrd_bytes = _make_nrrd_bytes()

        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/data"),
            body=nrrd_bytes,
            status=200,
            content_type="application/octet-stream",
        )
        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/properties"),
            json={
                "data": {
                    "properties": {
                        "tsstring": {
                            "type": "TemporoSpatialStringProperty",
                            "value": {"values": [{"t": 0, "z": 0, "value": "hello"}]},
                        },
                    }
                }
            },
            status=200,
        )

        result = node.get_data(as_type=DataRepresentation.MITK, include_properties=True)
        assert isinstance(result, mitk.Image)

        raw_prop = result.get_property("tsstring", raw=True)
        assert isinstance(raw_prop, mitk.TemporoSpatialStringProperty)
        assert raw_prop.value == "hello"

    # ------------------------------------------------------------------
    # MLS + mitk
    # ------------------------------------------------------------------

    @responses.activate
    def test_get_data_multilabel_auto_with_mitk_returns_mitk_mls(self) -> None:
        import mitk

        from mitk_workbench_remote._io import write_multilabel_nrrd
        from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

        seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
        seg.add_label(Label(1, "Liver"), group=seg.add_group("Anatomy"))
        nrrd_bytes = write_multilabel_nrrd(seg)

        node = _make_direct_node(data_type="MultiLabelSegmentation")
        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/data"),
            body=nrrd_bytes,
            status=200,
            content_type="application/octet-stream",
        )
        result = node.get_data(as_type=DataRepresentation.AUTO)
        assert isinstance(result, mitk.MultiLabelSegmentation)

    @responses.activate
    def test_get_data_multilabel_mitk_returns_mitk_mls(self) -> None:
        import mitk

        from mitk_workbench_remote._io import write_multilabel_nrrd
        from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

        seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
        seg.add_label(Label(1, "Liver"), group=seg.add_group("Anatomy"))
        seg.add_label(Label(2, "Spleen"), group=0)
        nrrd_bytes = write_multilabel_nrrd(seg)

        node = _make_direct_node(data_type="MultiLabelSegmentation")
        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/data"),
            body=nrrd_bytes,
            status=200,
            content_type="application/octet-stream",
        )
        result = node.get_data(as_type=DataRepresentation.MITK)
        assert isinstance(result, mitk.MultiLabelSegmentation)
        values = sorted(int(v) for v in result.label_values if int(v) != 0)
        assert values == [1, 2]

    @responses.activate
    def test_get_data_multilabel_mitk_include_properties(self) -> None:
        import mitk

        from mitk_workbench_remote._io import write_multilabel_nrrd
        from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

        seg = MultiLabelSegmentation.create(shape=(3, 4, 5), spacing=(1.0, 1.0, 1.0))
        seg.add_label(Label(1, "Liver"), group=seg.add_group("Anatomy"))
        nrrd_bytes = write_multilabel_nrrd(seg)

        node = _make_direct_node(data_type="MultiLabelSegmentation")
        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/data"),
            body=nrrd_bytes,
            status=200,
            content_type="application/octet-stream",
        )
        responses.add(
            responses.GET,
            _api("/datastorage/nodes/node_1/properties"),
            json={"data": {"properties": {"name": "MySeg"}}},
            status=200,
        )
        result = node.get_data(as_type=DataRepresentation.MITK, include_properties=True)
        assert isinstance(result, mitk.MultiLabelSegmentation)
        assert result.get_property("name") == "MySeg"

    @responses.activate
    def test_set_data_mitk_mls_uploads(self) -> None:
        import mitk

        node = _make_direct_node(data_type="MultiLabelSegmentation")
        responses.add(
            responses.PUT,
            _api("/datastorage/nodes/node_1/data"),
            json={},
            status=200,
        )
        ref = mitk.Image.from_numpy(np.zeros((3, 4, 5), dtype=np.uint16), spacing=(1.0, 1.0, 1.0))
        mitk_seg = mitk.MultiLabelSegmentation(ref)
        lbl = mitk.Label(1, "Organ")
        mitk_seg.add_group(None, [lbl])

        node.set_data(mitk_seg)
        assert len(responses.calls) == 1
        assert responses.calls[0].request.method == "PUT"

    @responses.activate
    def test_set_data_mitk_mls_sets_data_type(self) -> None:
        import mitk

        node = _make_direct_node(data_type="MultiLabelSegmentation")
        responses.add(
            responses.PUT,
            _api("/datastorage/nodes/node_1/data"),
            json={},
            status=200,
        )
        ref = mitk.Image.from_numpy(np.zeros((3, 4, 5), dtype=np.uint16), spacing=(1.0, 1.0, 1.0))
        mitk_seg = mitk.MultiLabelSegmentation(ref)

        node.set_data(mitk_seg)
        # data_type must be set locally without a refresh() round-trip.
        assert node.data_type == "MultiLabelSegmentation"
        assert len(responses.calls) == 1  # only the PUT, no GET for refresh
