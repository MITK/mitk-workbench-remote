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

This module provides:

- :class:`MultiLabelSegmentation`: Container of label groups backed by a 4D
  numpy array. Label values are globally unique across all groups; value 0 is
  reserved. Supports group-level image access, label properties editing, and
  value remapping.
- :class:`LabelGroup`: A named collection of labels within a
  ``MultiLabelSegmentation``, exposed as a live view (its ``labels`` and
  ``image`` reflect the segmentation's current state at access time).
- :class:`Label`: A single label with value, name, color, opacity, visibility,
  and lock state.
"""

from __future__ import annotations

import html as _html
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, overload

import numpy as np

from mitk_workbench_remote._spatial import (
    _normalize_direction,
    _normalize_origin,
    _normalize_spacing,
)
from mitk_workbench_remote.errors import MitkApiDivergenceWarning

# MITK's LabelSetImage requires this pixel type for all group images.
# mitk::LabelSetImage::LabelValueType = unsigned short = uint16.
LABEL_DTYPE: np.dtype = np.dtype(np.uint16)

# Sentinel for add_label's second positional argument. The (label, group) and
# (name, color, group) overloads share one implementation, so an unset second
# positional cannot be distinguished from a legitimate ``group=0``/``color`` by
# value alone.
_MISSING: Any = object()

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
        self._color = c

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
    """One group of a MultiLabelSegmentation.

    Returned by :attr:`MultiLabelSegmentation.groups` and
    :meth:`MultiLabelSegmentation.get_group`. A LabelGroup is a *live view* over
    its owning segmentation, not a point-in-time copy: :attr:`labels` and
    :attr:`image` reflect the segmentation's current state at access time.

    It also remains the internal membership record: the ordered label values
    (:attr:`label_ids`) are read directly by the segmentation and the NRRD I/O
    layer. Label objects themselves live in the parent's
    :class:`MultiLabelSegmentation` ``_labels`` registry.

    Args:
        name: Optional group name.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name: str | None = name
        self._label_ids: list[int] = []
        # Back-reference to the owning segmentation, populated when the group is
        # handed to a MultiLabelSegmentation (its __init__ or add_group). Without
        # it, index/labels/image cannot be resolved.
        self._seg: MultiLabelSegmentation | None = None
        self._index: int | None = None

    def _owner(self) -> tuple[MultiLabelSegmentation, int]:
        if self._seg is None or self._index is None:
            raise RuntimeError(
                "LabelGroup is not attached to a MultiLabelSegmentation; "
                "index/labels/image are only available for groups obtained via "
                "seg.groups or seg.get_group(i)."
            )
        return self._seg, self._index

    @property
    def name(self) -> str | None:
        """Group name, or ``None`` if unnamed.

        Read-only, matching native ``mitk.LabelGroup`` (a read-only snapshot).
        Rename via :meth:`MultiLabelSegmentation.set_group_name`.
        """
        return self._name

    @property
    def index(self) -> int:
        """0-based position of this group within the owning segmentation.

        Raises:
            RuntimeError: If this group is not attached to a segmentation (e.g. a
                bare ``LabelGroup(...)`` obtained other than via ``seg.groups`` or
                ``seg.get_group(i)``).
        """
        _, index = self._owner()
        return index

    @property
    def labels(self) -> list[Label]:
        """Label objects in this group, current at access time.

        Raises:
            RuntimeError: If this group is not attached to a segmentation.
        """
        seg, index = self._owner()
        return seg.get_group_labels(index)

    @property
    def image(self) -> Any:
        """The group's live pixel :class:`~mitk_workbench_remote.image.Image`.

        Delegates to :meth:`MultiLabelSegmentation.get_group_image`, so accessing
        it lazily zero-allocates the image exactly as that method does, and edits
        via ``.image.array`` persist on the segmentation.

        Raises:
            RuntimeError: If this group is not attached to a segmentation.
        """
        seg, index = self._owner()
        return seg.get_group_image(index)

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
        properties: Data-scope properties dict.

    Raises:
        ValueError: On shape/geometry inconsistency or duplicate/missing label IDs.
    """

    UNLABELED_VALUE: int = 0

    def __init__(
        self,
        *,
        groups: list[LabelGroup],
        labels: dict[int, Label],
        group_images: Sequence[Any | None] | None = None,
        spacing: Sequence[float] | np.ndarray | None = None,
        origin: Sequence[float] | np.ndarray | None = None,
        direction: Sequence[Any] | np.ndarray | None = None,
        properties: dict[str, Any] | None = None,
        _shape: tuple[int, ...] | None = None,
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
        self._max_value: int = max(self._labels.keys(), default=0)

        # Single choke point for the group back-references: create() delegates
        # here and _io.read_multilabel_nrrd() calls the constructor directly, so
        # every user-reachable LabelGroup gets its owner/index set exactly once.
        self._bind_groups()

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
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def groups(self) -> list[LabelGroup]:
        """Ordered list of label groups (copy)."""
        return list(self._groups)

    @property
    def num_groups(self) -> int:
        """Number of groups (alias for ``len(self.groups)``)."""
        return len(self._groups)

    @property
    def labels(self) -> list[Label]:
        """All Label objects across all groups, sorted by value.

        Raises:
            ValueError: If any label still has ``value is None``. By
                construction, ``add_label`` assigns a value before storing,
                so a ``None`` here indicates internal-state corruption —
                aliasing it onto the reserved ``UNLABELED_VALUE`` (0) would
                hide a real bug.
        """

        def _key(label: Label) -> int:
            if label.value is None:
                raise ValueError(
                    f"Label {label!r} has value=None — labels stored on a "
                    "MultiLabelSegmentation must have a non-None value (>= 1)."
                )
            return label.value

        return sorted(self._labels.values(), key=_key)

    @property
    def label_values(self) -> list[int]:
        """All label values across all groups, sorted ascending."""
        return sorted(self._labels.keys())

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
    def properties(self) -> dict[str, Any]:
        """Data-scope properties dict."""
        return dict(self._properties)

    @property
    def property_keys(self) -> list[str]:
        """List of property keys."""
        return list(self._properties.keys())

    def get_property(self, key: str) -> Any:
        """Get a property by key.

        Args:
            key: Property key.

        Returns:
            The property value, or ``None`` if not found.
        """
        return self._properties.get(key)

    def set_property(self, key: str, value: Any) -> None:
        """Set a property by key.

        Args:
            key: Property key.
            value: Property value.
        """
        self._properties[key] = value

    def remove_property(self, key: str) -> None:
        """Remove a property by key.

        Args:
            key: Property key.

        Raises:
            KeyError: If the key does not exist.
        """
        del self._properties[key]

    # ------------------------------------------------------------------
    # MITK interop
    # ------------------------------------------------------------------

    def to_mitk(self) -> Any:
        """Convert to a ``mitk.MultiLabelSegmentation`` (native MITK Python binding).

        Requires the ``mitk`` package, which is typically only available inside
        a MITK-provided Python environment. Geometry, pixel data, and all label
        metadata (groups, labels, colors, lock state, etc.) are transferred via
        NRRD round-trip.

        Note: data-scope properties stored in :attr:`properties` are **not**
        transferred, because the remote library's NRRD writer does not embed
        arbitrary properties. For a full round-trip including data-scope
        properties, use :meth:`DataNode.get_data` /
        :meth:`DataNode.set_data` with ``as_type=DataRepresentation.MITK``.

        Returns:
            A ``mitk.MultiLabelSegmentation`` instance.

        Raises:
            ImportError: If ``mitk`` is not installed.
        """
        try:
            import mitk  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'mitk' package is required for to_mitk(). "
                "It is available when using MITK's Python environment."
            ) from None
        from mitk_workbench_remote.converters._mitk_seg import MitkSegmentationConverter

        return MitkSegmentationConverter().from_segmentation(self)

    @classmethod
    def from_mitk(cls, mitk_seg: Any) -> MultiLabelSegmentation:
        """Create from a ``mitk.MultiLabelSegmentation`` (native MITK Python binding).

        Geometry, pixel data, and all label metadata are transferred via NRRD
        round-trip.

        Args:
            mitk_seg: A native MITK MultiLabelSegmentation object.

        Returns:
            A new :class:`MultiLabelSegmentation` instance.

        Raises:
            ImportError: If ``mitk`` is not installed.
        """
        try:
            import mitk  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'mitk' package is required for from_mitk(). "
                "It is available when using MITK's Python environment."
            ) from None
        from mitk_workbench_remote.converters._mitk_seg import MitkSegmentationConverter

        result: MultiLabelSegmentation = MitkSegmentationConverter().to_segmentation(mitk_seg)
        return result

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_label(self, value: int) -> Label:
        """Look up a label by value.

        Labels are keyed by value, so a miss raises ``KeyError`` (dict-like),
        matching native ``mitk.MultiLabelSegmentation.get_label``. Use
        :meth:`has_label` for an existence check that never raises.

        Args:
            value: Integer label value.

        Returns:
            The :class:`Label` with the given value.

        Raises:
            KeyError: If no label with ``value`` exists.
        """
        return self._labels[value]

    def has_label(self, value: int) -> bool:
        """Return whether a label with the given value exists.

        Args:
            value: Integer label value.

        Returns:
            ``True`` if a label with ``value`` exists, else ``False``. Never
            raises (including for the reserved :attr:`UNLABELED_VALUE`).
        """
        return value in self._labels

    def get_group_of_label(self, value: int) -> int:
        """Return the index of the group that owns a label.

        Args:
            value: Integer label value.

        Returns:
            The 0-based index of the owning group.

        Raises:
            KeyError: If no label with ``value`` exists in any group.
        """
        for i, g in enumerate(self._groups):
            if value in g._label_ids:
                return i
        raise KeyError(value)

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

    def set_group_name(self, index: int, name: str | None) -> None:
        """Set a group's display name.

        The group's :attr:`LabelGroup.name` is read-only; this is the supported
        way to rename it, mirroring native ``set_group_name``. Passing ``None`` to
        clear the name is a remote-only allowance -- native ``mitk`` requires a
        string and rejects ``None``.

        Args:
            index: Group index.
            name: New group name, or ``None`` to clear it (remote-only).

        Raises:
            IndexError: If ``index`` is out of range.
        """
        self.get_group(index)._name = name

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

    def get_labels(self, values: Iterable[int]) -> list[Label]:
        """Return the labels for the given values, in input order.

        Missing values are skipped silently (matching native ``get_labels`` and
        preserving cross-mode parity). For a strict single lookup that raises,
        use :meth:`get_label`.

        Args:
            values: Label values to look up.

        Returns:
            The matching :class:`Label` objects, in the order of ``values``.
        """
        return [self._labels[v] for v in values if v in self._labels]

    def get_label_values_by_name(self, name: str, group: int | None = None) -> list[int]:
        """Return the values of all labels whose name equals ``name``.

        Args:
            name: Label name to match exactly.
            group: If given, restrict the search to that group.

        Returns:
            Matching label values, sorted ascending (consistent with
            :attr:`label_values`); an empty list if none match.

        Raises:
            IndexError: If ``group`` is out of range.
        """
        candidates = self._labels.values() if group is None else self.get_group_labels(group)
        matches = [lbl.value for lbl in candidates if lbl.name == name and lbl.value is not None]
        return sorted(matches)

    def get_group_label_values(self, index: int) -> list[int]:
        """Return the label values belonging to a group, in group order.

        Args:
            index: Group index.

        Returns:
            The group's label values, in group order.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        return self.get_group(index).label_ids

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    @overload
    def add_label(self, label: Label, /, group: int = 0) -> Label: ...

    @overload
    def add_label(
        self, name: str, /, color: tuple[float, float, float], group: int = 0
    ) -> Label: ...

    # The two overloads put different parameters at position 2 (group vs color),
    # which no single named implementation signature can express; mypy's
    # overload/impl compatibility check flags this even though every concrete
    # call is accepted at runtime (positional group works in both spellings).
    def add_label(  # type: ignore[misc]
        self, label_or_name: Label | str, /, color: Any = _MISSING, group: int = 0
    ) -> Label:
        """Add a label to a group and return it (with its value assigned).

        Two spellings, dispatched on the first argument's type:

        - ``add_label(label, group=0)`` -- add an existing :class:`Label`.
        - ``add_label(name, color, group=0)`` -- build and add a label from a
          name and RGB color.

        If ``label.value`` is ``None`` the next free integer >= 1 is assigned.

        Unlike native mitk -- which clones the label and reassigns its value on a
        collision -- this stores the given object and raises on collision.

        Args:
            label_or_name: A :class:`Label`, or a label name (str) for the
                convenience spelling.
            color: RGB color, required only in the ``(name, color)`` spelling.
            group: Target group index (default 0).

        Returns:
            The added label.

        Raises:
            TypeError: If the first argument is neither a Label nor a str, or the
                str spelling is used without a color.
            ValueError: If the value is 0 (reserved) or already exists.
            IndexError: If ``group`` is out of range.
        """
        if isinstance(label_or_name, Label):
            label = label_or_name
            # The second positional argument is the group index here; it binds to
            # `color` because both overloads share one implementation.
            if color is not _MISSING:
                group = color
        elif isinstance(label_or_name, str):
            if color is _MISSING:
                raise TypeError("add_label(name, color, group=0) requires a color")
            label = Label(name=label_or_name, color=color)
        else:
            raise TypeError(
                f"first argument must be a Label or a name (str), got "
                f"{type(label_or_name).__name__}"
            )

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
        self._max_value = max(self._max_value, v)
        return label

    def remove_labels(self, values: Iterable[int]) -> None:
        """Remove several labels (and their pixels).

        Atomic: every value is validated to exist first (``KeyError`` naming the
        first missing value) before any label is removed.

        Args:
            values: Label values to remove.

        Raises:
            KeyError: If any value is not present in any group.
        """
        for v in self._validated_unique(values):
            self.remove_label(v)

    def erase_label(self, value: int) -> None:
        """Zero every pixel equal to ``value`` in its group image; keep the label.

        Unlike :meth:`remove_label`, the label metadata is retained. If the
        group's image has not been allocated yet this is a no-op on pixels (it
        does not force allocation). After erasing, :meth:`validate` reports the
        label as declared-but-unpainted, which is the intended state.

        Args:
            value: Label value whose pixels to clear.

        Raises:
            KeyError: If ``value`` is not present in any group.
        """
        group_idx = self.get_group_of_label(value)
        img = self._group_images[group_idx]
        if img is not None:
            arr = img.array
            arr[arr == value] = 0

    def erase_labels(self, values: Iterable[int]) -> None:
        """:meth:`erase_label` for several values, with the same atomic validation.

        Args:
            values: Label values whose pixels to clear.

        Raises:
            KeyError: If any value is not present in any group.
        """
        for v in self._validated_unique(values):
            self.erase_label(v)

    def rename_label(self, value: int, name: str, color: tuple[float, float, float]) -> None:
        """Set a label's name and color in one call.

        Both ``name`` and ``color`` are required, matching native ``rename_label``.

        Args:
            value: Label value to rename.
            name: New label name.
            color: New RGB color, components in [0.0, 1.0].

        Raises:
            KeyError: If ``value`` is not present.
            ValueError: If any color component is out of [0.0, 1.0].
        """
        label = self.get_label(value)
        # Set color first: its setter validates the range and raises before
        # mutating, so a bad color leaves the label untouched (atomic rename).
        label.color = color
        label.name = name

    def merge_labels(self, target: int, sources: Iterable[int]) -> None:
        """Reassign each source label's pixels to ``target``, then remove the sources.

        Simplified vs native ``mitk.MultiLabelSegmentation.merge_labels``: it does
        not implement lock handling, merge/overwrite styles, or native's
        ``TransferLabelContent`` semantics, and -- unlike native, which retains
        the source labels -- it removes them. On use it emits
        :class:`~mitk_workbench_remote.errors.MitkApiDivergenceWarning`. It exists
        so callers do not break under ``get_data(as_type=AUTO)``; for full
        fidelity use the mitk bindings.

        In the target group, overlaps are best-effort (the target value wins),
        which can overwrite another target-group label's last pixels and leave it
        declared but unpainted -- part of the documented divergence.

        Args:
            target: Label value that the sources are merged into.
            sources: Label values to merge into ``target`` and then remove.

        Raises:
            KeyError: If ``target`` or any source is not present.
        """
        if not self.has_label(target):
            raise KeyError(target)
        # Drop the target from the sources (dedup preserves order) so a self-merge
        # cannot remove the target label.
        source_values = [s for s in dict.fromkeys(sources) if s != target]
        for s in source_values:
            if not self.has_label(s):
                raise KeyError(s)

        warnings.warn(
            "merge_labels is a simplified stand-in for native mitk merge_labels: "
            "it does not support lock handling or merge/overwrite styles, and it "
            "removes the source labels (native retains them). Use "
            "mitk.MultiLabelSegmentation.merge_labels for full fidelity.",
            MitkApiDivergenceWarning,
            stacklevel=2,
        )

        target_img = self.get_group_image(self.get_group_of_label(target))
        for s in source_values:
            src_img = self.get_group_image(self.get_group_of_label(s))
            target_img.array[src_img.array == s] = target
            # remove_label zeroes any residual source pixels and drops the label.
            self.remove_label(s)

    def _validated_unique(self, values: Iterable[int]) -> list[int]:
        """Validate every value exists, then return them de-duplicated, in order.

        Deduping is required for atomicity: a repeated value would miss on its
        second pass once the first pass removed it, leaving a half-applied state.

        Raises:
            KeyError: naming the first value not present in any group.
        """
        materialized = list(values)
        for v in materialized:
            if not self.has_label(v):
                raise KeyError(v)
        return list(dict.fromkeys(materialized))

    def remove_label(self, value: int) -> None:
        """Remove a label and zero its pixels in its group image.

        Matches native ``mitk.MultiLabelSegmentation.remove_label``: MITK does
        not model a "remove the label but keep its orphan pixels" state. To
        clear a label's pixels while keeping the label, use :meth:`erase_label`;
        to reassign pixels to another label, remap them first, then remove.

        Args:
            value: Label value to remove.

        Raises:
            KeyError: If ``value`` is not present in any group.
        """
        group_idx = self.get_group_of_label(value)

        if self._group_images[group_idx] is not None:
            arr = self._group_images[group_idx].array
            arr[arr == value] = 0

        self._groups[group_idx]._label_ids.remove(value)
        del self._labels[value]

    def _bind_groups(self) -> None:
        """Attach each group's owner back-reference and positional index."""
        for i, g in enumerate(self._groups):
            g._seg = self
            g._index = i

    def add_group(
        self,
        name: str | None = None,
        image: Any | None = None,
        labels: list[Label] | None = None,
    ) -> int:
        """Append a new group and return its index.

        Optionally seed the group with pixel ``image`` and/or a list of
        ``labels``. ``image`` and ``labels`` are positional-or-keyword to mirror
        native ``add_group(name, image, labels)``, so the same call works in
        both modes.

        Args:
            name: Optional group name.
            image: Optional pixel data (ndarray / Image / converter type, same
                acceptance as :meth:`set_group_image`).
            labels: Optional labels to add to the new group (each via
                :meth:`add_label`; value assignment and collision handling are
                identical).

        Returns:
            Index of the newly added group.

        Raises:
            TypeError: If ``image`` cannot be resolved to an ndarray (not an
                ndarray, :class:`~mitk_workbench_remote.image.Image`, or a type
                with a registered converter).
            ValueError: On shape/geometry mismatch of ``image`` or a duplicate
                label value.

        On any failure the half-seeded group is rolled back, so a raised
        ``add_group`` never leaves a dangling group behind.
        """
        group = LabelGroup(name=name)
        self._groups.append(group)
        self._group_images.append(None)
        index = len(self._groups) - 1
        group._seg = self
        group._index = index

        # Seeding needs the group to exist (set_group_image/add_label take an
        # index), so it happens after the append; roll back if it raises.
        # set_group_image may set self._shape on a previously-shapeless
        # segmentation, so capture and restore it alongside the labels.
        labels_before = set(self._labels)
        shape_before = self._shape
        try:
            if image is not None:
                self.set_group_image(index, image)
            if labels is not None:
                for label in labels:
                    self.add_label(label, index)
        except Exception:
            del self._groups[index]
            del self._group_images[index]
            for v in set(self._labels) - labels_before:
                del self._labels[v]
            self._shape = shape_before
            raise
        return index

    def _next_free_value(self) -> int:
        """Return the next available label value as a high-water mark plus one.

        This is O(1) and matches MITK C++'s own strategy. Gaps left by
        :meth:`remove_label` are intentionally not reused.
        """
        return self._max_value + 1

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
            arr = np.zeros(self._shape, dtype=LABEL_DTYPE)
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
        if index < 0 or index >= len(self._groups):
            raise IndexError(f"Group index {index} out of range (have {len(self._groups)} groups)")

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
        """Compose all group images into a 4D array ``[num_groups, Z, Y, X]``.

        Per-group spatial arrays follow the same ``(Z, Y, X)`` numpy
        layout :class:`~mitk_workbench_remote.image.Image` uses, so a
        voxel at ``arr[k, j, i]`` lives at world ``(i*sx, j*sy, k*sz)``.
        The multilabel NRRD writer transposes to MITK's wire layout at
        the I/O boundary; callers do not see the F-order quirk.

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
