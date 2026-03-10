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

"""Tests for _io/multilabel_nrrd.py."""

from pathlib import Path

import numpy as np

from mitk_workbench_remote._io.multilabel_nrrd import (
    _MODALITY_VALUE,
    is_multilabel_nrrd,
    parse_labelgroups_json,
    read_multilabel_nrrd_raw,
    serialize_labelgroups_json,
    write_multilabel_nrrd_raw,
)

# ---------------------------------------------------------------------------
# is_multilabel_nrrd
# ---------------------------------------------------------------------------


def test_is_multilabel_nrrd_true() -> None:
    header = {"modality": _MODALITY_VALUE}
    assert is_multilabel_nrrd(header) is True


def test_is_multilabel_nrrd_false_missing_modality() -> None:
    assert is_multilabel_nrrd({}) is False


def test_is_multilabel_nrrd_false_wrong_modality() -> None:
    assert is_multilabel_nrrd({"modality": "MR"}) is False


# ---------------------------------------------------------------------------
# JSON parse / serialize
# ---------------------------------------------------------------------------


def test_parse_labelgroups_json() -> None:
    json_str = '[{"labels":[{"name":"bg","value":0}]}]'
    result = parse_labelgroups_json(json_str)
    assert len(result) == 1
    assert result[0]["labels"][0]["name"] == "bg"


def test_serialize_labelgroups_json_compact() -> None:
    groups = [{"labels": [{"name": "bg", "value": 0}]}]
    result = serialize_labelgroups_json(groups)
    assert " " not in result  # compact, no spaces
    assert parse_labelgroups_json(result) == groups


def test_json_roundtrip() -> None:
    groups = [
        {"labels": [{"name": "Background", "value": 0}, {"name": "Tumor", "value": 1}]},
        {"labels": [{"name": "Background", "value": 0}]},
    ]
    json_str = serialize_labelgroups_json(groups)
    result = parse_labelgroups_json(json_str)
    assert result == groups


# ---------------------------------------------------------------------------
# read/write roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_bytes() -> None:
    arr = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    arr[0, 1, 2, 3] = 1
    arr[1, 0, 0, 0] = 2
    groups = [
        {"labels": [{"name": "Background", "value": 0}, {"name": "Tumor", "value": 1}]},
        {"labels": [{"name": "Background", "value": 0}, {"name": "Organ", "value": 2}]},
    ]
    nrrd_bytes = write_multilabel_nrrd_raw(
        arr,
        groups,
        spacing=(0.5, 1.0, 2.0),
        origin=(10.0, 20.0, 30.0),
        direction=np.eye(3),
    )
    data, groups2, spatial = read_multilabel_nrrd_raw(nrrd_bytes)
    np.testing.assert_array_equal(data, arr)
    assert groups2 == groups
    assert np.allclose(spatial["spacing"], (0.5, 1.0, 2.0))
    assert np.allclose(spatial["origin"], (10.0, 20.0, 30.0))


def test_roundtrip_file(tmp_path: Path) -> None:
    arr = np.ones((1, 4, 5, 6), dtype=np.uint16)
    groups = [{"labels": []}]
    file_path = tmp_path / "seg.nrrd"
    write_multilabel_nrrd_raw(
        arr,
        groups,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        path=file_path,
    )
    assert file_path.exists()
    data, _groups2, _spatial = read_multilabel_nrrd_raw(file_path)
    np.testing.assert_array_equal(data, arr)


def test_4d_space_directions_has_nan_for_vector_axis() -> None:
    """The first space direction entry should be NaN (vector/group axis)."""
    import io

    import nrrd

    arr = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    nrrd_bytes = write_multilabel_nrrd_raw(
        arr,
        [],
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
    )
    buf = io.BytesIO(nrrd_bytes)
    header = nrrd.read_header(buf)
    space_dirs = header["space directions"]
    # First entry should be nan-vector
    assert np.all(np.isnan(space_dirs[0]))
    # Remaining should be spatial directions
    for i in range(1, 4):
        assert not np.any(np.isnan(space_dirs[i]))
