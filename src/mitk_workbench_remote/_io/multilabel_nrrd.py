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

Raw dict-based functions (``read_multilabel_nrrd_raw``, ``write_multilabel_nrrd_raw``) do
not depend on the domain classes. High-level functions (``read_multilabel_nrrd``,
``write_multilabel_nrrd``) use deferred imports to avoid circular dependencies.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mitk_workbench_remote.multilabel import Label, LabelGroup, MultiLabelSegmentation

import nrrd
import numpy as np

from mitk_workbench_remote._io.nrrd import _INDEX_ORDER, _MITK_SPACE, _extract_spatial

_log = logging.getLogger(__name__)

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
        data = nrrd.read_data(header, buf, None, index_order=_INDEX_ORDER)
    else:
        data, header = nrrd.read(str(source), index_order=_INDEX_ORDER)

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
    properties: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> bytes:
    """Write a 4D multilabel NRRD.

    Args:
        data: 4D numpy array with shape ``[num_groups, x, y, z]``.
        groups_data: Label groups properties (list of dicts).
        spacing: Voxel spacing for the spatial dimensions.
        origin: World-space origin.
        direction: Direction cosine matrix for spatial dims.
        properties: Additional custom properties for the header.
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

    if properties:
        for key, value in properties.items():
            header[key] = value

    if path is not None:
        nrrd.write(str(path), data, header, index_order=_INDEX_ORDER)
        return Path(path).read_bytes()

    buf = io.BytesIO()
    nrrd.write(buf, data, header, index_order=_INDEX_ORDER)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Dict <-> domain object conversion helpers (use deferred imports)
# ---------------------------------------------------------------------------


def _label_from_dict(d: dict[str, Any]) -> Label:
    """Build a Label from a raw label dict.

    Skips value=0 check -- callers must filter out UNLABELED_VALUE before calling.

    Args:
        d: Raw label dict from the NRRD header JSON.

    Returns:
        A :class:`~mitk_workbench_remote.multilabel.Label`.
    """
    from mitk_workbench_remote.multilabel import Label

    color_raw = d.get("color", [1.0, 1.0, 1.0])
    return Label(
        value=d["value"],
        name=d.get("name", ""),
        color=tuple(color_raw),
        opacity=d.get("opacity", 1.0),
        visible=d.get("visible", True),
        locked=d.get("locked", False),
        tracking_id=d.get("tracking_id"),
        tracking_uid=d.get("tracking_uid"),
        description=d.get("description"),
    )


def _labelgroup_from_dict(g: dict[str, Any]) -> tuple[LabelGroup, list[Label]]:
    """Build a LabelGroup and its Labels from a raw group dict.

    Labels with value=0 (UNLABELED_VALUE) are silently skipped.

    Args:
        g: Raw group dict from the NRRD header JSON.

    Returns:
        Tuple of (LabelGroup with label_ids populated, list of Label objects).
    """
    from mitk_workbench_remote.multilabel import LabelGroup

    group = LabelGroup(name=g.get("name"))
    labels: list[Label] = []
    for label_dict in g.get("labels", []):
        if label_dict.get("value", 0) == 0:
            continue  # skip UNLABELED_VALUE
        label = _label_from_dict(label_dict)
        if label.value is None:
            raise ValueError("Label loaded from NRRD has no value assigned")
        group._label_ids.append(label.value)
        labels.append(label)
    return group, labels


def _label_to_dict(label: Label) -> dict[str, Any]:
    """Serialize a Label to a raw dict for NRRD header storage.

    Args:
        label: A :class:`~mitk_workbench_remote.multilabel.Label`.

    Returns:
        Dict suitable for inclusion in the labelgroups JSON.
    """
    d: dict[str, Any] = {
        "value": label.value,
        "name": label.name,
        "color": list(label.color),
        "opacity": label.opacity,
        "visible": label.visible,
        "locked": label.locked,
    }
    if label.tracking_id is not None:
        d["tracking_id"] = label.tracking_id
    if label.tracking_uid is not None:
        d["tracking_uid"] = label.tracking_uid
    if label.description is not None:
        d["description"] = label.description
    return d


def _labelgroup_to_dict(group: LabelGroup, labels_dict: dict[int, Label]) -> dict[str, Any]:
    """Serialize a LabelGroup to a raw dict for NRRD header storage.

    Args:
        group: A :class:`~mitk_workbench_remote.multilabel.LabelGroup`.
        labels_dict: Global label registry from the parent segmentation.

    Returns:
        Dict suitable for inclusion in the labelgroups JSON.
    """
    d: dict[str, Any] = {
        "labels": [_label_to_dict(labels_dict[v]) for v in group.label_ids if v in labels_dict]
    }
    if group.name is not None:
        d["name"] = group.name
    return d


# ---------------------------------------------------------------------------
# High-level I/O
# ---------------------------------------------------------------------------


def read_multilabel_nrrd(source: bytes | str | Path) -> MultiLabelSegmentation:
    """Read a multilabel NRRD and return a MultiLabelSegmentation.

    Args:
        source: NRRD data as raw bytes, a file path string, or a Path object.

    Returns:
        A :class:`~mitk_workbench_remote.multilabel.MultiLabelSegmentation`.
    """
    from mitk_workbench_remote.image import Image
    from mitk_workbench_remote.multilabel import MultiLabelSegmentation

    array, groups_data, spatial_info = read_multilabel_nrrd_raw(source)
    _log.debug(
        "read_multilabel_nrrd: shape=%s dtype=%s groups=%d",
        array.shape,
        array.dtype,
        len(groups_data),
    )

    groups: list[LabelGroup] = []
    labels_dict: dict[int, Label] = {}
    group_images: list[Image] = []

    for i, group_dict in enumerate(groups_data):
        group, group_labels = _labelgroup_from_dict(group_dict)
        groups.append(group)
        for label in group_labels:
            # _labelgroup_from_dict guarantees label.value is not None.
            assert label.value is not None
            labels_dict[label.value] = label
        group_images.append(
            Image(
                array[i],
                spacing=spatial_info["spacing"],
                origin=spatial_info["origin"],
                direction=spatial_info["direction"],
            )
        )

    return MultiLabelSegmentation(
        groups=groups,
        labels=labels_dict,
        group_images=group_images,
        spacing=spatial_info["spacing"],
        origin=spatial_info["origin"],
        direction=spatial_info["direction"],
    )


def write_multilabel_nrrd(seg: MultiLabelSegmentation, *, path: str | Path | None = None) -> bytes:
    """Serialize a MultiLabelSegmentation to NRRD bytes.

    Args:
        seg: A :class:`~mitk_workbench_remote.multilabel.MultiLabelSegmentation`.
        path: If given, also writes the NRRD to this file path.

    Returns:
        The NRRD file content as bytes.
    """
    groups_data = [_labelgroup_to_dict(g, seg._labels) for g in seg._groups]
    array = seg._compose_array()
    _log.debug(
        "write_multilabel_nrrd: shape=%s dtype=%s groups=%d",
        array.shape,
        array.dtype,
        len(groups_data),
    )
    return write_multilabel_nrrd_raw(
        array,
        groups_data,
        spacing=seg.spacing,
        origin=seg.origin,
        direction=seg.direction,
        properties=seg.properties if seg.properties else None,
        path=path,
    )
