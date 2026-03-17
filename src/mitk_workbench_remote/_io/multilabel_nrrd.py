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

"""MultiLabel NRRD parsing and serialization.

Handles the 4D NRRD format used by MITK MultiLabelSegmentation. Label group properties
are stored in the NRRD header key 'org.mitk.multilabel.segmentation.labelgroups' as JSON.

High-level functions using LabelGroup/Label types are deferred to T9.
This module provides raw dict-based functions.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import nrrd
import numpy as np

from mitk_workbench_remote._io.nrrd import _MITK_SPACE, _extract_spatial

_LABELGROUPS_KEY = "org.mitk.multilabel.segmentation.labelgroups"
_MODALITY_KEY = "modality"
_MODALITY_VALUE = "org.mitk.multilabel.segmentation"


def is_multilabel_nrrd(header: dict[str, Any]) -> bool:
    """Check whether an NRRD header describes a multilabel segmentation.

    Args:
        header: Parsed NRRD header dict.

    Returns:
        True if the modality field indicates a multilabel segmentation.
    """
    return header.get(_MODALITY_KEY) == _MODALITY_VALUE


def parse_labelgroups_json(json_str: str) -> list[dict[str, Any]]:
    """Parse the label groups JSON string from the NRRD header.

    Args:
        json_str: JSON string from the ``org.mitk.multilabel.segmentation.labelgroups``
            header field.

    Returns:
        List of label group dicts, each containing a ``"labels"`` list.
    """
    return json.loads(json_str)  # type: ignore[no-any-return]


def serialize_labelgroups_json(groups_data: list[dict[str, Any]]) -> str:
    """Serialize label groups to a compact JSON string.

    Args:
        groups_data: List of label group dicts.

    Returns:
        Compact JSON string for storage in the NRRD header.
    """
    return json.dumps(groups_data, separators=(",", ":"))


def read_multilabel_nrrd_raw(
    source: bytes | str | Path,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Read a multilabel NRRD and return raw data.

    Args:
        source: NRRD data as raw bytes, a file path string, or a Path object.

    Returns:
        Tuple of:
        - 4D numpy array with shape ``[num_groups, x, y, z]``
        - Parsed label groups list (list of dicts)
        - Spatial dict with keys ``spacing``, ``origin``, ``direction``, and
          any custom properties
    """
    if isinstance(source, bytes):
        buf = io.BytesIO(source)
        header = nrrd.read_header(buf)
        data = nrrd.read_data(header, buf, None)
    else:
        data, header = nrrd.read(str(source))

    # Determine spatial ndim (excluding vector axis)
    kinds = header.get("kinds", [])
    spatial_ndim = sum(1 for k in kinds if k not in ("vector", "list", "???"))
    if spatial_ndim == 0:
        spatial_ndim = data.ndim - 1  # assume first axis is group axis

    spacing, origin, direction = _extract_spatial(header, spatial_ndim)

    # Parse label groups JSON
    groups_json = header.get(_LABELGROUPS_KEY, "[]")
    groups_data = parse_labelgroups_json(groups_json)

    spatial_info: dict[str, Any] = {
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
    }

    return data, groups_data, spatial_info


def write_multilabel_nrrd_raw(
    data: np.ndarray,
    groups_data: list[dict[str, Any]],
    *,
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: np.ndarray,
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> bytes:
    """Write a 4D multilabel NRRD.

    Args:
        data: 4D numpy array with shape ``[num_groups, x, y, z]``.
        groups_data: Label groups properties (list of dicts).
        spacing: Voxel spacing for the spatial dimensions.
        origin: World-space origin.
        direction: Direction cosine matrix for spatial dims.
        metadata: Additional custom properties for the header.
        path: If given, writes to this file path. Always returns bytes.

    Returns:
        The NRRD file content as bytes.
    """
    spatial_ndim = len(spacing)
    header: dict[str, Any] = {
        "space": _MITK_SPACE,
        "encoding": "gzip",
    }

    # kinds: vector for group axis, domain for spatial axes
    header["kinds"] = ["vector"] + ["domain"] * spatial_ndim

    # space directions: nan-vector for non-spatial axis, then direction * spacing
    space_dirs: list[Any] = [np.full(spatial_ndim, np.nan).tolist()]
    for i in range(spatial_ndim):
        space_dirs.append((direction[i] * spacing[i]).tolist())
    header["space directions"] = space_dirs

    header["space origin"] = list(origin)

    # Label groups JSON
    header[_LABELGROUPS_KEY] = serialize_labelgroups_json(groups_data)
    header[_MODALITY_KEY] = _MODALITY_VALUE

    if metadata:
        for key, value in metadata.items():
            header[key] = value

    if path is not None:
        nrrd.write(str(path), data, header)
        return Path(path).read_bytes()

    buf = io.BytesIO()
    nrrd.write(buf, data, header)
    return buf.getvalue()
