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

"""MultiLabelSegmentation -- 4D label volume with group semantics.

Classes:
    MultiLabelSegmentation: Container of label groups backed by a 4D numpy array.
        Label values are globally unique across all groups; value 0 is reserved.
        Supports group-level image access, label properties editing, and value remapping.
    LabelGroup: A named collection of labels within a MultiLabelSegmentation.
    Label: A single label with value, name, color, opacity, visibility, and lock state.
"""

from __future__ import annotations

import html as _html
from collections.abc import Sequence
from typing import Any

import numpy as np

from mitk_workbench_remote._spatial import (
    _normalize_direction,
    _normalize_origin,
    _normalize_spacing,
)

# MITK's LabelSetImage requires this pixel type for all group images.
# mitk::LabelSetImage::LabelValueType = unsigned short = uint16.
LABEL_DTYPE: np.dtype = np.dtype(np.uint16)

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


class Label:
    """A single label within a MultiLabelSegmentation.

    Label value 0 is reserved as UNLABELED_VALUE and is rejected.
    When created with ``value=None``, the value is assigned by
    :meth:`MultiLabelSegmentation.add_label` via :meth:`_assign_value`.

    Args:
        value: Integer label value (>= 1), or ``None`` for deferred assignment.
        name: Human-readable label name.
        color: RGB color tuple with components in [0.0, 1.0].
        opacity: Opacity in [0.0, 1.0].
        visible: Whether the label is visible.
        locked: Whether the label is locked for editing.
        tracking_id: Optional external tracking identifier.
        tracking_uid: Optional external tracking UID.
        description: Optional free-text description.

    Raises:
        ValueError: If ``value == 0`` (reserved) or ``color`` components out of range.
    """

    def __init__(
        self,
        value: int | None = None,
        name: str = "",
        *,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        opacity: float = 1.0,
        visible: bool = True,
        locked: bool = False,
        tracking_id: str | None = None,
        tracking_uid: str | None = None,
        description: str | None = None,
    ) -> None:
        if value == 0:
            raise ValueError("value 0 is reserved as UNLABELED_VALUE")
        self._value: int | None = value
        self._name: str = str(name)
        self._opacity: float = float(opacity)
        self._visible: bool = bool(visible)
        self._locked: bool = bool(locked)
        self._tracking_id: str | None = tracking_id
        self._tracking_uid: str | None = tracking_uid
        self._description: str | None = description
        self._color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.color = color  # go through setter for validation

    def _assign_value(self, v: int) -> None:
        """Assign the label value. Package-private; called only by MultiLabelSegmentation.

        Args:
            v: Integer value to assign (must be >= 1).

        Raises:
            RuntimeError: If the value is already set.
        """
        if self._value is not None:
            raise RuntimeError(
                f"Label value is already assigned ({self._value}) and cannot be changed"
            )
        self._value = v

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def value(self) -> int | None:
        """Integer label value, or ``None`` if not yet assigned."""
        return self._value

    @property
    def name(self) -> str:
        """Human-readable label name."""
        return self._name

    @name.setter
    def name(self, v: str) -> None:
        self._name = str(v)

    @property
    def color(self) -> tuple[float, float, float]:
        """RGB color tuple with components in [0.0, 1.0]."""
        return self._color

    @color.setter
    def color(self, v: Any) -> None:
        c = tuple(float(x) for x in v)
        if len(c) != 3:
            raise ValueError(f"color must have 3 components, got {len(c)}")
        for x in c:
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"color components must be in [0.0, 1.0], got {x}")
        self._color = c  # type: ignore[assignment]

    @property
    def opacity(self) -> float:
        """Opacity in [0.0, 1.0]."""
        return self._opacity

    @opacity.setter
    def opacity(self, v: float) -> None:
        self._opacity = float(v)

    @property
    def visible(self) -> bool:
        """Whether this label is visible."""
        return self._visible

    @visible.setter
    def visible(self, v: bool) -> None:
        self._visible = bool(v)

    @property
    def locked(self) -> bool:
        """Whether this label is locked for editing."""
        return self._locked

    @locked.setter
    def locked(self, v: bool) -> None:
        self._locked = bool(v)

    @property
    def tracking_id(self) -> str | None:
        """Optional external tracking identifier."""
        return self._tracking_id

    @tracking_id.setter
    def tracking_id(self, v: str | None) -> None:
        self._tracking_id = v

    @property
    def tracking_uid(self) -> str | None:
        """Optional external tracking UID."""
        return self._tracking_uid

    @tracking_uid.setter
    def tracking_uid(self, v: str | None) -> None:
        self._tracking_uid = v

    @property
    def description(self) -> str | None:
        """Optional free-text description."""
        return self._description

    @description.setter
    def description(self, v: str | None) -> None:
        self._description = v

    def __repr__(self) -> str:
        return f"<Label value={self._value!r} {self._name!r} color={self._color}>"


# ---------------------------------------------------------------------------
# LabelGroup
# ---------------------------------------------------------------------------


class LabelGroup:
    """A named collection of label values within a MultiLabelSegmentation.

    Label objects themselves are stored in the parent
    :class:`MultiLabelSegmentation._labels` registry; this class only tracks
    the ordered list of label values belonging to the group.

    Args:
        name: Optional group name.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name: str | None = name
        self._label_ids: list[int] = []

    @property
    def name(self) -> str | None:
        """Group name, or ``None`` if unnamed."""
        return self._name

    @name.setter
    def name(self, v: str | None) -> None:
        self._name = v

    @property
    def label_ids(self) -> list[int]:
        """Ordered list of label values in this group. Returns a copy."""
        return list(self._label_ids)

    def __repr__(self) -> str:
        return f"<LabelGroup name={self._name!r} labels={len(self._label_ids)}>"


# ---------------------------------------------------------------------------
# MultiLabelSegmentation
# ---------------------------------------------------------------------------


class MultiLabelSegmentation:
    """Container of label groups backed by per-group spatial images.

    Label values are globally unique across all groups. Value 0 is reserved as
    :attr:`UNLABELED_VALUE`. Group pixel data is stored as individual
    :class:`~mitk_workbench_remote.image.Image` objects (``None`` until set or
    lazily allocated by :meth:`get_group_image`).

    Use :meth:`create` to build a new segmentation from scratch. The constructor
    is the primary path for I/O deserialization.

    Args:
        groups: Ordered list of label groups (each pre-populated with label IDs).
        labels: Global label registry mapping value -> Label.
        group_images: Per-group pixel data. Filled with ``None`` if not given.
        spacing: Voxel spacing (3D).
        origin: World-space origin (3D).
        direction: Direction cosine matrix (3x3).
        properties: Data-scope metadata dict.

    Raises:
        ValueError: On shape/geometry inconsistency or duplicate/missing label IDs.
    """

    UNLABELED_VALUE: int = 0

    def __init__(
        self,
        *,
        groups: list[LabelGroup],
        labels: dict[int, Label],
        group_images: list[Any | None] | None = None,
        spacing: Sequence[float] | np.ndarray | None = None,
        origin: Sequence[float] | np.ndarray | None = None,
        direction: Sequence[Any] | np.ndarray | None = None,
        properties: dict[str, Any] | None = None,
        _shape: tuple[int, ...] | None = None,
        _dtype: Any = LABEL_DTYPE,
    ) -> None:
        # Resolve group images list
        if group_images is None:
            imgs: list[Any] = [None] * len(groups)
        else:
            if len(group_images) != len(groups):
                raise ValueError(
                    f"len(group_images)={len(group_images)} != len(groups)={len(groups)}"
                )
            imgs = list(group_images)

        # Validate label IDs: no duplicates across groups, all present in labels dict
        seen: set[int] = set()
        for g in groups:
            for lid in g._label_ids:
                if lid in seen:
                    raise ValueError(f"Duplicate label ID {lid} across groups")
                seen.add(lid)
                if lid not in labels:
                    raise ValueError(
                        f"Label ID {lid} referenced by group is not in the labels dict"
                    )

        self._groups: list[LabelGroup] = list(groups)
        self._labels: dict[int, Label] = dict(labels)
        self._group_images: list[Any] = imgs
        self._properties: dict[str, Any] = properties if properties is not None else {}
        self._dtype: np.dtype = np.dtype(_dtype)

        # Normalize geometry (always 3D spatial)
        self._spacing = _normalize_spacing(spacing, ndim=3)
        self._origin = _normalize_origin(origin, ndim=3)
        self._direction = _normalize_direction(direction, ndim=3)

        # Infer/validate shape from group images
        canonical_shape: tuple[int, ...] | None = _shape
        for i, img in enumerate(self._group_images):
            if img is None:
                continue
            img_shape: tuple[int, ...] = img.shape
            if canonical_shape is None:
                canonical_shape = img_shape
            elif img_shape != canonical_shape:
                raise ValueError(
                    f"Group image {i} has shape {img_shape}, expected {canonical_shape}"
                )
            # Geometry consistency (approximate for floats)
            if not np.allclose(img.spacing, self._spacing):
                raise ValueError(f"Group image {i} spacing {img.spacing} != {self._spacing}")
            if not np.allclose(img.origin, self._origin):
                raise ValueError(f"Group image {i} origin {img.origin} != {self._origin}")
            if not np.allclose(img.direction, self._direction):
                raise ValueError(f"Group image {i} direction mismatch")
        self._shape: tuple[int, ...] | None = canonical_shape

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        groups: list[LabelGroup] | None = None,
        reference: Any | None = None,
        shape: tuple[int, ...] | None = None,
        spacing: Sequence[float] | None = None,
        origin: Sequence[float] | None = None,
        direction: np.ndarray | None = None,
        dtype: Any = LABEL_DTYPE,
    ) -> MultiLabelSegmentation:
        """Create a new empty segmentation.

        Either ``reference`` or ``shape`` must be supplied.

        Args:
            groups: Initial groups. Defaults to an empty list.
            reference: An :class:`~mitk_workbench_remote.image.Image` from which geometry
                is inherited (overrideable via spacing/origin/direction).
            shape: Spatial shape ``(x, y, z)`` used when ``reference`` is not given.
            spacing: Override spacing (defaults to reference geometry or (1, 1, 1)).
            origin: Override origin (defaults to reference geometry or (0, 0, 0)).
            direction: Override direction (defaults to reference geometry or identity).
            dtype: Numpy dtype for lazily allocated group images. Defaults to
                ``LABEL_DTYPE`` (``uint16``), which is required by MITK.

        Returns:
            A new :class:`MultiLabelSegmentation` with no pixel data.

        Raises:
            ValueError: If neither ``reference`` nor ``shape`` is given.
        """
        if reference is None and shape is None:
            raise ValueError("Either 'reference' or 'shape' must be provided")

        if reference is not None:
            actual_shape: tuple[int, ...] = reference.shape
            eff_spacing = spacing if spacing is not None else reference.spacing
            eff_origin = origin if origin is not None else reference.origin
            eff_direction = direction if direction is not None else reference.direction
        else:
            actual_shape = tuple(shape)  # type: ignore[arg-type]
            eff_spacing = spacing
            eff_origin = origin
            eff_direction = direction

        init_groups: list[LabelGroup] = list(groups) if groups is not None else []
        return cls(
            groups=init_groups,
            labels={},
            group_images=None,
            spacing=eff_spacing,
            origin=eff_origin,
            direction=eff_direction,
            _shape=actual_shape,
            _dtype=np.dtype(dtype),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def groups(self) -> list[LabelGroup]:
        """Ordered list of label groups (copy)."""
        return list(self._groups)

    @property
    def labels(self) -> list[Label]:
        """All Label objects across all groups (unordered)."""
        return list(self._labels.values())

    @property
    def spacing(self) -> tuple[float, ...]:
        """Voxel spacing (3D)."""
        return self._spacing

    @property
    def origin(self) -> tuple[float, ...]:
        """World-space origin (3D)."""
        return self._origin

    @property
    def direction(self) -> np.ndarray:
        """Direction cosine matrix (3x3, float64)."""
        return self._direction

    @property
    def metadata(self) -> dict[str, Any]:
        """Data-scope properties/metadata dict."""
        return self._properties

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_label(self, value: int) -> Label | None:
        """Look up a label by value.

        Args:
            value: Integer label value.

        Returns:
            The :class:`Label`, or ``None`` if not found.
        """
        return self._labels.get(value)

    def get_group(self, index: int) -> LabelGroup:
        """Return the group at the given index.

        Args:
            index: Group index.

        Returns:
            The :class:`LabelGroup`.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if index < 0 or index >= len(self._groups):
            raise IndexError(f"Group index {index} out of range (have {len(self._groups)} groups)")
        return self._groups[index]

    def get_group_by_name(self, name: str) -> LabelGroup | None:
        """Find a group by name using a linear search.

        Args:
            name: Group name to search for.

        Returns:
            The first matching :class:`LabelGroup`, or ``None``.
        """
        for g in self._groups:
            if g.name == name:
                return g
        return None

    def get_group_labels(self, index: int) -> list[Label]:
        """Return all Label objects belonging to a group.

        Args:
            index: Group index.

        Returns:
            Ordered list of :class:`Label` objects.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        group = self.get_group(index)
        return [self._labels[v] for v in group._label_ids if v in self._labels]

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def add_label(self, label: Label, group: int) -> Label:
        """Add a label to the specified group.

        If ``label.value`` is ``None``, the next free integer >= 1 is assigned.

        Args:
            label: The label to add.
            group: Target group index.

        Returns:
            The label (with its value assigned).

        Raises:
            ValueError: If the value is 0 or already exists in any group.
            IndexError: If ``group`` is out of range.
        """
        if group < 0 or group >= len(self._groups):
            raise IndexError(f"Group index {group} out of range (have {len(self._groups)} groups)")

        if label.value is None:
            v = self._next_free_value()
            label._assign_value(v)
        else:
            v = label.value

        if v == self.UNLABELED_VALUE:
            raise ValueError("value 0 is reserved as UNLABELED_VALUE")
        if v in self._labels:
            raise ValueError(f"Label value {v} already exists in this segmentation")

        self._labels[v] = label
        self._groups[group]._label_ids.append(v)
        return label

    def remove_label(self, value: int, *, clear_pixels: bool = True) -> None:
        """Remove a label by value.

        Args:
            value: Label value to remove.
            clear_pixels: If ``True`` (default), zero out pixels equal to ``value``
                in the group's image before removing the label.

        Raises:
            ValueError: If ``value`` is not found in any group.
        """
        group_idx: int | None = None
        for i, g in enumerate(self._groups):
            if value in g._label_ids:
                group_idx = i
                break
        if group_idx is None:
            raise ValueError(f"Label value {value} not found in any group")

        if clear_pixels and self._group_images[group_idx] is not None:
            arr = self._group_images[group_idx].array
            arr[arr == value] = 0

        self._groups[group_idx]._label_ids.remove(value)
        del self._labels[value]

    def add_group(self, name: str | None = None) -> int:
        """Append a new empty group and return its index.

        Args:
            name: Optional group name.

        Returns:
            Index of the newly added group.
        """
        self._groups.append(LabelGroup(name=name))
        self._group_images.append(None)
        return len(self._groups) - 1

    def _next_free_value(self) -> int:
        """Return the smallest integer >= 1 not already used as a label value."""
        v = 1
        while v in self._labels:
            v += 1
        return v

    # ------------------------------------------------------------------
    # Group image access
    # ------------------------------------------------------------------

    def get_group_image(self, index: int) -> Any:
        """Return the pixel data for a group as an Image.

        If no data has been set for this group, a zero-filled Image is created
        and cached. Edits to the returned Image's array are reflected in
        subsequent calls.

        Args:
            index: Group index.

        Returns:
            An :class:`~mitk_workbench_remote.image.Image` with this group's pixels.

        Raises:
            IndexError: If ``index`` is out of range.
            ValueError: If shape is unknown and no pixel data has been set yet.
        """
        from mitk_workbench_remote.image import Image

        if index < 0 or index >= len(self._groups):
            raise IndexError(f"Group index {index} out of range")
        if self._group_images[index] is None:
            if self._shape is None:
                raise ValueError(
                    "Cannot create group image: shape is unknown. "
                    "Use set_group_image() first, or create with shape= or reference=."
                )
            arr = np.zeros(self._shape, dtype=self._dtype)
            self._group_images[index] = Image(
                arr,
                spacing=self._spacing,
                origin=self._origin,
                direction=self._direction,
            )
        return self._group_images[index]

    def set_group_image(self, index: int, data: Any) -> None:
        """Set pixel data for a group.

        Args:
            index: Group index.
            data: An :class:`~mitk_workbench_remote.image.Image`, ndarray, or any
                type handled by the converter registry.

        Raises:
            IndexError: If ``index`` is out of range.
            ValueError: If the array shape does not match the segmentation's shape,
                ndim < 3, or dtype is not integer-compatible.
        """
        from mitk_workbench_remote.image import Image

        if index < 0 or index >= len(self._groups):
            raise IndexError(f"Group index {index} out of range")

        arr = self._resolve_to_ndarray(data)

        if arr.ndim < 3:
            raise ValueError(f"Group image must have ndim >= 3, got {arr.ndim}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Group image dtype must be integer-compatible, got {arr.dtype}")
        if self._shape is not None and arr.shape != self._shape:
            raise ValueError(f"Shape mismatch: expected {self._shape}, got {arr.shape}")

        # MITK requires LabelValueType (uint16) for all group images.
        if arr.dtype != LABEL_DTYPE:
            arr = arr.astype(LABEL_DTYPE)

        self._group_images[index] = Image(
            arr,
            spacing=self._spacing,
            origin=self._origin,
            direction=self._direction,
        )
        if self._shape is None:
            self._shape = arr.shape

    def import_group_image(
        self, index: int, data: Any, *, value_map: dict[int, int | str]
    ) -> None:
        """Import and remap pixel data into a group.

        Source pixel values are remapped according to ``value_map``. Unmapped
        values pass through unchanged. Value 0 always maps to 0.

        Args:
            index: Target group index.
            data: Source pixel data (any type accepted by :meth:`set_group_image`).
            value_map: Mapping from source integer values to target label values.
                Targets may be an integer label value or a label name string.

        Raises:
            IndexError: If ``index`` is out of range.
            ValueError: If a target label name does not exist in the group, or a
                target integer value does not belong to the group.
        """
        arr = self._resolve_to_ndarray(data)

        # Resolve string targets to integer values, validate group membership
        group_label_ids: set[int] = set(self._groups[index]._label_ids)
        group_labels_by_name: dict[str, Label] = {
            lbl.name: lbl for lbl in self.get_group_labels(index)
        }

        resolved: dict[int, int] = {}
        for src_val, target in value_map.items():
            if isinstance(target, str):
                lbl = group_labels_by_name.get(target)
                if lbl is None:
                    raise ValueError(f"No label named {target!r} in group {index}")
                resolved[src_val] = lbl.value  # type: ignore[assignment]
            else:
                if target not in group_label_ids:
                    raise ValueError(f"Label value {target} does not belong to group {index}")
                resolved[src_val] = target

        remapped = arr.copy()
        for src_val, dst_val in resolved.items():
            remapped[arr == src_val] = dst_val

        self.set_group_image(index, remapped)

    def _resolve_to_ndarray(self, data: Any) -> np.ndarray:
        """Resolve data to a numpy ndarray."""
        if isinstance(data, np.ndarray):
            return data
        from mitk_workbench_remote.image import Image

        if isinstance(data, Image):
            return data.array
        from mitk_workbench_remote.converters import find_image_converter

        converter = find_image_converter(data)
        if converter is not None:
            return converter.to_ndarray(data)
        raise TypeError(
            f"Cannot resolve {type(data).__name__} to ndarray. "
            f"Pass an ndarray, Image, or a type with a registered converter."
        )

    # ------------------------------------------------------------------
    # I/O helpers (used by _io layer)
    # ------------------------------------------------------------------

    def _compose_array(self) -> np.ndarray:
        """Compose all group images into a 4D array [num_groups, x, y, z].

        Raises:
            ValueError: If there are no groups or if shape is unknown.
        """
        if not self._groups:
            raise ValueError("Cannot compose: segmentation has no groups")

        shape = self._shape
        if shape is None:
            for img in self._group_images:
                if img is not None:
                    shape = img.shape
                    break
        if shape is None:
            raise ValueError(
                "Cannot compose: no shape information available. "
                "Set pixel data on at least one group first."
            )

        arrays: list[np.ndarray] = []
        for img in self._group_images:
            if img is None:
                arrays.append(np.zeros(shape, dtype=LABEL_DTYPE))
            else:
                arrays.append(img.array)
        return np.stack(arrays, axis=0)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check internal consistency between pixel data and label metadata.

        Returns:
            A list of warning strings. An empty list means the segmentation
            is internally consistent.
        """
        warnings: list[str] = []

        if len(self._groups) != len(self._group_images):
            warnings.append(
                f"Group count mismatch: {len(self._groups)} groups but "
                f"{len(self._group_images)} group images"
            )
            return warnings

        for i, (group, img) in enumerate(zip(self._groups, self._group_images, strict=True)):
            if img is None:
                continue
            arr = img.array
            pixel_values: set[int] = set(int(v) for v in np.unique(arr))
            pixel_values.discard(0)  # 0 is always UNLABELED_VALUE

            declared: set[int] = set(group._label_ids)

            for v in sorted(pixel_values - declared):
                warnings.append(
                    f"Group {i}: pixel value {v} is present in the image "
                    f"but not declared as a label"
                )

            for v in sorted(declared - pixel_values):
                label = self._labels.get(v)
                label_name = label.name if label is not None else "?"
                warnings.append(
                    f"Group {i}: label {v} ({label_name!r}) is declared "
                    f"but not present in the pixel data"
                )

        return warnings

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<MultiLabelSegmentation groups={len(self._groups)} labels={len(self._labels)}>"

    def _repr_html_(self) -> str:
        e = _html.escape
        header = (
            "<tr><th>Value</th><th>Name</th><th>Color</th><th>Visible</th><th>Locked</th></tr>"
        )
        rows: list[str] = []
        for i, group in enumerate(self._groups):
            group_label = e(group.name if group.name is not None else f"Group {i}")
            rows.append(f"<tr><td colspan='5'><b>{group_label}</b></td></tr>")
            for label in self.get_group_labels(i):
                r, g, b = label.color
                swatch = (
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"></span>'
                )
                rows.append(
                    f"<tr>"
                    f"<td style='padding-left:16px'>{label.value}</td>"
                    f"<td>{e(label.name)}</td>"
                    f"<td>{swatch}</td>"
                    f"<td>{label.visible}</td>"
                    f"<td>{label.locked}</td>"
                    f"</tr>"
                )
        return f"<table>{header}{''.join(rows)}</table>"
