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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mitk_workbench_remote._spatial import (
    _normalize_direction,
    _normalize_origin,
    _normalize_spacing,
)


class Image:
    """Spatial image wrapper backed by a numpy array with spacing, origin, and direction.

    Accepts any type handled by the converter registry (ndarray, SimpleITK.Image,
    mlarray.MLArray, file path). Pixel data conversion is lazy -- the array is
    only materialized on first access to :attr:`array`.

    Args:
        data: Pixel data or an object convertible via the converter registry.
        spacing: Voxel spacing. Overrides converter-extracted values. Defaults
            to ``(1.0, ...)`` per dimension.
        origin: World-space origin. Overrides converter-extracted values.
            Defaults to ``(0.0, ...)``.
        direction: Direction cosine matrix. Overrides converter-extracted values.
            Defaults to the identity matrix.
        properties: Data-scope properties/metadata dict. If ``None`` and a converter
            is used, metadata is extracted from the source object.
    """

    def __init__(
        self,
        data: Any,
        *,
        spacing: Sequence[float] | np.ndarray | None = None,
        origin: Sequence[float] | np.ndarray | None = None,
        direction: Sequence[Any] | np.ndarray | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(data, np.ndarray):
            self._array: np.ndarray | None = data
            self._source_data: Any = None
            self._converter: Any = None
            ndim = data.ndim
            geo_defaults: dict[str, Any] = {}
            properties_defaults: dict[str, Any] = {}
        else:
            from mitk_workbench_remote.converters import find_image_converter

            converter = find_image_converter(data)
            if converter is None:
                raise TypeError(
                    f"No converter found for {type(data).__name__}. "
                    f"Pass a numpy array or register a converter."
                )
            self._array = None
            self._source_data = data
            self._converter = converter
            geo_defaults = converter.extract_geometry(data)
            properties_defaults = converter.extract_metadata(data)
            # Need ndim: try to get from geometry, else materialize array
            if "spacing" in geo_defaults:
                ndim = len(geo_defaults["spacing"])
            elif "origin" in geo_defaults:
                ndim = len(geo_defaults["origin"])
            else:
                ndim = self.array.ndim

        self._spacing = _normalize_spacing(
            spacing if spacing is not None else geo_defaults.get("spacing"),
            ndim=ndim,
        )
        self._origin = _normalize_origin(
            origin if origin is not None else geo_defaults.get("origin"),
            ndim=ndim,
        )
        self._direction = _normalize_direction(
            direction if direction is not None else geo_defaults.get("direction"),
            ndim=ndim,
        )
        self._properties = properties if properties is not None else properties_defaults

    def __repr__(self) -> str:
        return f"Image(shape={self.shape}, dtype={self.dtype}, spacing={self._spacing})"

    def _repr_html_(self) -> str:
        import html as _html

        e = _html.escape

        dir_rows = " | ".join(
            "  ".join(f"{v:.4g}" for v in row) for row in self._direction.tolist()
        )

        rows = [
            f"<tr><th>shape</th><td>{e(str(self.shape))}</td></tr>",
            f"<tr><th>dtype</th><td>{e(str(self.dtype))}</td></tr>",
            f"<tr><th>spacing</th><td>{e(str(self._spacing))}</td></tr>",
            f"<tr><th>origin</th><td>{e(str(self._origin))}</td></tr>",
            f"<tr><th>direction</th><td>{e(dir_rows)}</td></tr>",
        ]

        if self._properties:
            rows.append(
                '<tr><th colspan="2">Fetched properties'
                " <small><em>(snapshot at download time &mdash; additional or changed"
                " properties on the Workbench side are not reflected here)</em></small>"
                "</th></tr>"
            )
            for key, value in self._properties.items():
                rows.append(f"<tr><th>{e(str(key))}</th><td>{e(str(value))}</td></tr>")

        return "<table>" + "".join(rows) + "</table>"

    def __eq__(self, other: object) -> bool:
        """Return True if both images have identical pixel data and geometry.

        Comparison is exact (bit-for-bit) for all spatial properties. Images
        that round-trip through NRRD serialization may not compare equal due
        to floating-point representation differences.
        """
        if not isinstance(other, Image):
            return NotImplemented
        return (
            np.array_equal(self.array, other.array)
            and self._spacing == other._spacing
            and self._origin == other._origin
            and np.array_equal(self._direction, other._direction)
        )

    # ------------------------------------------------------------------
    # Lazy array access
    # ------------------------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        """Pixel data as a numpy array. Materialized lazily on first access."""
        if self._array is None:
            self._array = self._converter.to_ndarray(self._source_data)
        return self._array

    # ------------------------------------------------------------------
    # Spatial properties
    # ------------------------------------------------------------------

    @property
    def spacing(self) -> tuple[float, ...]:
        """Voxel spacing per dimension."""
        return self._spacing

    @property
    def origin(self) -> tuple[float, ...]:
        """World-space origin."""
        return self._origin

    @property
    def direction(self) -> np.ndarray:
        """Direction cosine matrix (ndim x ndim, float64)."""
        return self._direction

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions.

        For vector or multichannel images, ``ndim`` is the number of *spatial*
        axes (e.g. 3 for a 3-D volume), while :attr:`shape` may have an
        additional channel axis. ``ndim`` equals ``len(spacing)`` and always
        matches the geometry, not the array rank.
        """
        return len(self._spacing)

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        return self.array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """Array data type."""
        return self.array.dtype

    @property
    def metadata(self) -> dict[str, Any]:
        """Data-scope metadata."""
        return self._properties

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return pixel data as a numpy array."""
        return self.array

    def to_simpleitk(self) -> Any:
        """Convert to a SimpleITK Image.

        Raises:
            ImportError: If SimpleITK is not installed.
        """
        try:
            import SimpleITK
        except ImportError:
            raise ImportError(
                "SimpleITK is required for to_simpleitk(). Install it with: pip install SimpleITK"
            ) from None

        from mitk_workbench_remote.converters import find_converter_for_type

        converter = find_converter_for_type(SimpleITK.Image)
        if converter is None:
            raise ImportError("SitkConverter is not registered")
        return converter.from_image(self)

    def to_mlarray(self) -> Any:
        """Convert to an mlarray.MLArray.

        Raises:
            ImportError: If mlarray is not installed.
        """
        try:
            import mlarray
        except ImportError:
            raise ImportError(
                "mlarray is required for to_mlarray(). Install it with: pip install mlarray"
            ) from None

        from mitk_workbench_remote.converters import find_converter_for_type

        converter = find_converter_for_type(mlarray.MLArray)
        if converter is None:
            raise ImportError("MLArrayConverter is not registered")
        return converter.from_image(self)
