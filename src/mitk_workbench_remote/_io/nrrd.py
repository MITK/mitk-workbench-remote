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

"""NRRD read/write via pynrrd.

Converts between raw NRRD bytes and Image objects, handling spatial properties
(space, space directions, space origin) from the NRRD header.

Axis ordering
-------------
pynrrd's ``index_order`` parameter controls how numpy array axes map to NRRD
dimensions.  We use ``index_order='C'`` throughout this module so that numpy
axis 0 is the slowest-varying dimension (z in a 3-D image) -- the standard
C-order / row-major convention.

This matches the axis convention used by ``mitk.Image.from_numpy`` /
``mitk.Image.as_numpy`` in the MITK Python bindings (pybind11), which also
treat numpy axis 0 as the slowest-varying spatial dimension.  Without this
alignment, NRRD files written by ``mitk.IOUtil.save`` would produce transposed
arrays when read back through pynrrd (and vice-versa), because pynrrd's
default ``index_order='F'`` treats axis 0 as the *fastest*-varying dimension.

The ``index_order`` must be the same for reading and writing; mixing
conventions silently transposes data.  All call sites in this module and in
``multilabel_nrrd.py`` therefore use the shared ``_INDEX_ORDER`` constant.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Literal

import nrrd
import numpy as np

_log = logging.getLogger(__name__)

# NRRD header keys used for spatial properties
_SPACE = "space"
_SPACE_DIRECTIONS = "space directions"
_SPACE_ORIGIN = "space origin"
_KINDS = "kinds"

# MITK convention: left-posterior-superior
_MITK_SPACE = "left-posterior-superior"

# Axis ordering for pynrrd read/write -- see module docstring for rationale.
_INDEX_ORDER: Literal["C", "F"] = "C"


def read_nrrd(source: bytes | str | Path) -> Any:
    """Read an NRRD file/bytes and return an Image.

    Args:
        source: NRRD data as raw bytes, a file path string, or a Path object.

    Returns:
        An Image with pixel data and spatial properties extracted from the header.
    """
    from mitk_workbench_remote.image import Image

    if isinstance(source, bytes):
        buf = io.BytesIO(source)
        header = nrrd.read_header(buf)
        data = nrrd.read_data(header, buf, None, index_order=_INDEX_ORDER)
    else:
        data, header = nrrd.read(str(source), index_order=_INDEX_ORDER)

    ndim = _spatial_ndim(header, data.ndim)
    spacing, origin, direction = _extract_spatial(header, ndim)
    metadata = _extract_custom_properties(header)

    _log.debug("read_nrrd: shape=%s dtype=%s", data.shape, data.dtype)
    return Image(
        data,
        spacing=spacing,
        origin=origin,
        direction=direction,
        properties=metadata,
    )


def write_nrrd(image: Any, path: str | Path | None = None) -> bytes:
    """Write an Image to NRRD format.

    Args:
        image: An Image object with array, spacing, origin, direction, properties.
        path: If given, writes to this file path and returns the bytes.
            If None, returns the NRRD bytes without writing to disk.

    Returns:
        The NRRD file content as bytes.
    """
    _log.debug("write_nrrd: shape=%s dtype=%s", image.array.shape, image.array.dtype)
    header = _build_header(image)
    _write_custom_properties(header, image.properties)
    arr = image.array

    if path is not None:
        nrrd.write(str(path), arr, header, index_order=_INDEX_ORDER)
        return Path(path).read_bytes()

    # Write to memory buffer
    buf = io.BytesIO()
    nrrd.write(buf, arr, header, index_order=_INDEX_ORDER)
    return buf.getvalue()


def _spatial_ndim(header: dict[str, Any], array_ndim: int) -> int:
    """Determine the number of spatial dimensions from the header.

    For standard images, spatial ndim == array ndim.
    For vector/4D images with a ``kinds`` field containing ``"vector"`` or
    ``"list"``, the non-spatial axis is excluded.
    """
    kinds = header.get(_KINDS)
    if kinds is not None:
        spatial_kinds = [k for k in kinds if k not in ("vector", "list", "???")]
        if len(spatial_kinds) < len(kinds):
            return len(spatial_kinds)
    return array_ndim


def _extract_spatial(
    header: dict[str, Any], ndim: int
) -> tuple[tuple[float, ...], tuple[float, ...], np.ndarray]:
    """Extract spacing, origin, and direction from NRRD header.

    Args:
        header: Parsed NRRD header dict.
        ndim: Number of spatial dimensions.

    Returns:
        Tuple of (spacing, origin, direction).
    """
    direction = np.eye(ndim, dtype=np.float64)
    spacing = tuple(1.0 for _ in range(ndim))

    space_directions = header.get(_SPACE_DIRECTIONS)
    if space_directions is not None:
        # Filter out None entries and nan-vector entries (non-spatial axes)
        spatial_dirs = [
            d
            for d in space_directions
            if d is not None and not (hasattr(d, "__len__") and np.all(np.isnan(d)))
        ]
        if len(spatial_dirs) == ndim:
            dir_matrix = np.array(spatial_dirs, dtype=np.float64)
            norms = np.linalg.norm(dir_matrix, axis=1)
            # Avoid division by zero
            safe_norms = np.where(norms > 0, norms, 1.0)
            spacing = tuple(float(n) for n in norms)
            direction = dir_matrix / safe_norms[:, np.newaxis]

    origin = tuple(0.0 for _ in range(ndim))
    space_origin = header.get(_SPACE_ORIGIN)
    if space_origin is not None:
        origin = tuple(float(o) for o in space_origin[:ndim])

    return spacing, origin, direction


def _build_header(image: Any) -> dict[str, Any]:
    """Build an NRRD header dict from an Image.

    Args:
        image: Image object with spacing, origin, direction, ndim.

    Returns:
        NRRD header dict ready for nrrd.write().
    """
    ndim = image.ndim
    header: dict[str, Any] = {
        _SPACE: _MITK_SPACE,
        "encoding": "gzip",
    }

    # Build space directions: direction * spacing
    direction = image.direction
    spacing = image.spacing
    space_dirs = np.zeros((ndim, ndim), dtype=np.float64)
    for i in range(ndim):
        space_dirs[i] = direction[i] * spacing[i]
    header[_SPACE_DIRECTIONS] = space_dirs.tolist()

    header[_SPACE_ORIGIN] = list(image.origin)

    return header


def _extract_custom_properties(header: dict[str, Any]) -> dict[str, Any]:
    """Extract custom key-value fields from the NRRD header.

    pynrrd stores custom `:=` fields in the header alongside standard fields.
    We extract all fields that are not standard NRRD header keys.
    """
    standard_keys = {
        "type",
        "dimension",
        "space",
        "space dimension",
        "sizes",
        "space directions",
        "kinds",
        "encoding",
        "endian",
        "space origin",
        "data file",
        "data_file",
        "content",
        "block size",
        "blocksize",
        "sample units",
        "sampleunits",
        "min",
        "max",
        "old min",
        "oldmin",
        "old max",
        "oldmax",
        "line skip",
        "lineskip",
        "byte skip",
        "byteskip",
        "number",
        "thicknesses",
        "axis mins",
        "axismins",
        "axis maxs",
        "axismaxs",
        "centers",
        "centerings",
        "labels",
        "units",
        "space units",
        "measurement frame",
    }
    properties: dict[str, Any] = {}
    for key, value in header.items():
        if key.lower() not in standard_keys:
            properties[key] = value
    return properties


def _write_custom_properties(header: dict[str, Any], properties: dict[str, Any]) -> None:
    """Write properties dict as custom fields in the NRRD header.

    Args:
        header: NRRD header dict to modify in-place.
        properties: Custom key-value pairs to add.

    Raises:
        ValueError: If any properties key conflicts with a standard NRRD header field
            already present in *header*.
    """
    conflicts = [k for k in properties if k in header]
    if conflicts:
        raise ValueError(
            f"Properties keys conflict with standard NRRD header fields: {conflicts!r}."
            " Remove these keys from the image properties before writing."
        )
    for key, value in properties.items():
        header[key] = value
