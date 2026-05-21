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

"""Synthetic phantoms for the example notebooks.

Functions here return geometrically consistent :class:`Image` and
:class:`MultiLabelSegmentation` objects so notebook cells can stay focused
on demonstrating the library API instead of reconstructing valid test
data inline.

Two image families are provided:

- :func:`make_ct_like` -- concentric cubes with HU-like intensities.
- :func:`make_mr_like` -- concentric spheres with T1-weighted-like intensities.

Plus higher-level helpers for the storyline notebooks:

- :func:`make_body_with_tumor` -- a snowman-shaped body region (large
  trunk sphere, smaller head sphere stacked on top along ``+Z``, a small
  nose protruding along ``+Y`` from the head) with an irregular tumor
  built from 3-5 perturbed sub-spheres. The asymmetry gives an obvious
  orientation cue in axial / coronal / sagittal views; the irregular
  tumor shape distinguishes per-case appearances when iterating over a
  cohort. The modality flag controls intensities only, not geometry.
- :func:`make_segmentation` -- a single-group MultiLabelSegmentation
  whose labels are randomly placed sphere blobs co-registered with a
  reference image.
- :func:`make_cohort` -- a list of :class:`CaseBundle` (image +
  segmentation + metadata) for cohort-style examples.

The intensity ranges are chosen to look CT-ish or MR-ish for level/window
purposes; they are not physically calibrated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mitk_workbench_remote.image import Image
from mitk_workbench_remote.multilabel import (
    LABEL_DTYPE,
    Label,
    LabelGroup,
    MultiLabelSegmentation,
)

Modality = Literal["ct", "mr"]

# Default geometry. The non-uniform spacing is deliberate: clinical
# axial scans typically have fine in-plane resolution (X/Y) and a
# thicker between-slice (Z) spacing. The numbers below mimic that --
# 96x96 voxels in-plane at 1 mm, 32 slices at 2.5 mm thickness -- so
# the notebooks have to honour anisotropy rather than treat voxel
# indices as world coordinates.
#
# Axis convention: this module follows the SimpleITK / pynrrd
# convention used throughout the library -- the numpy array is indexed
# ``arr[k, j, i]`` where ``k`` is the slowest spatial axis (Z, the
# slice axis) and ``i`` the fastest (X, an in-plane axis).
# ``spacing`` and ``origin`` tuples, however, are in world-axis order
# ``(sx, sy, sz)``. ``DEFAULT_SHAPE = (32, 96, 96)`` therefore means
# ``Z=32 slices, Y=96, X=96`` voxels. Mask helpers that build snowman
# / tumor geometry below operate in **physical millimetres** so that
# spheres come out spherical even with anisotropic spacing.
DEFAULT_SHAPE: tuple[int, int, int] = (32, 96, 96)
DEFAULT_SPACING: tuple[float, float, float] = (1.0, 1.0, 2.5)
DEFAULT_ORIGIN: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Soft, distinguishable label colors. Reused across notebooks.
_PALETTE: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.20, 0.20),
    (0.20, 1.00, 0.20),
    (0.20, 0.40, 1.00),
    (1.00, 0.85, 0.20),
    (0.80, 0.20, 1.00),
)


# ---------------------------------------------------------------------------
# Voxel-index mask primitives
# ---------------------------------------------------------------------------


def _sphere_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """Return a boolean mask of a sphere in voxel-index coordinates."""
    idx = np.indices(shape, dtype=np.float32)
    dx = idx[0] - center[0]
    dy = idx[1] - center[1]
    dz = idx[2] - center[2]
    mask: np.ndarray = (dx * dx + dy * dy + dz * dz) <= float(radius) ** 2
    return mask


def _cube_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    half_extent: tuple[float, float, float],
) -> np.ndarray:
    """Return a boolean mask of an axis-aligned cube in voxel-index coordinates."""
    idx = np.indices(shape, dtype=np.float32)
    mask: np.ndarray = (
        (np.abs(idx[0] - center[0]) <= float(half_extent[0]))
        & (np.abs(idx[1] - center[1]) <= float(half_extent[1]))
        & (np.abs(idx[2] - center[2]) <= float(half_extent[2]))
    )
    return mask


def _center_voxel(shape: tuple[int, int, int]) -> tuple[float, float, float]:
    return (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)


def _phys_sphere_mask(
    shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    center_zyx_mm: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    """Sphere mask in physical (millimetre) coordinates.

    Voxel ``(k, j, i)`` sits at world position ``(k*sz, j*sy, i*sx)``
    so a sphere defined in world units stays spherical regardless of
    anisotropic spacing -- unlike a voxel-radius mask, which would be
    an ellipsoid in world space.

    Args:
        shape: Numpy array shape ``(Z, Y, X)``.
        spacing_xyz: Voxel spacing in world-axis order ``(sx, sy, sz)``.
        center_zyx_mm: Sphere centre in array-axis order ``(z, y, x)``,
            in millimetres.
        radius_mm: Sphere radius in millimetres.
    """
    sx, sy, sz = spacing_xyz
    cz, cy, cx = center_zyx_mm
    idx = np.indices(shape, dtype=np.float32)
    dz = idx[0] * sz - cz
    dy = idx[1] * sy - cy
    dx = idx[2] * sx - cx
    mask: np.ndarray = (dx * dx + dy * dy + dz * dz) <= float(radius_mm) ** 2
    return mask


# ---------------------------------------------------------------------------
# Image phantoms
# ---------------------------------------------------------------------------


def make_ct_like(
    *,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
    origin: tuple[float, float, float] = DEFAULT_ORIGIN,
) -> Image:
    """Return a CT-like volume of concentric cubes with HU-ish intensities.

    Intensity layout (Hounsfield-like):

    - background: ``-1000`` (air),
    - body cube: ``40`` (soft tissue),
    - inner cube: ``200`` (contrast).

    Args:
        shape: Spatial array shape ``(Z, Y, X)`` (slowest axis first; see
            module-level note).
        spacing: Voxel spacing per dimension (mm).
        origin: World-space origin (mm).

    Returns:
        :class:`Image` of float32 pixels with the requested geometry.
    """
    arr = np.full(shape, -1000.0, dtype=np.float32)
    center = _center_voxel(shape)
    smin = float(min(shape))
    body_h = (0.30 * smin, 0.30 * smin, 0.30 * smin)
    inner_h = (0.12 * smin, 0.12 * smin, 0.12 * smin)
    arr[_cube_mask(shape, center, body_h)] = 40.0
    arr[_cube_mask(shape, center, inner_h)] = 200.0
    return Image(arr, spacing=spacing, origin=origin)


def make_mr_like(
    *,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
    origin: tuple[float, float, float] = DEFAULT_ORIGIN,
) -> Image:
    """Return an MR-like volume of concentric spheres with T1-ish intensities.

    Intensity layout (T1-weighted-like, arbitrary units):

    - background: ``0``,
    - body sphere: ``600``,
    - inner sphere: ``1200``.

    Args:
        shape: Spatial array shape ``(Z, Y, X)`` (slowest axis first; see
            module-level note).
        spacing: Voxel spacing per dimension (mm).
        origin: World-space origin (mm).

    Returns:
        :class:`Image` of float32 pixels with the requested geometry.
    """
    arr = np.zeros(shape, dtype=np.float32)
    center = _center_voxel(shape)
    smin = float(min(shape))
    body_r = 0.30 * smin
    inner_r = 0.12 * smin
    arr[_sphere_mask(shape, center, body_r)] = 600.0
    arr[_sphere_mask(shape, center, inner_r)] = 1200.0
    return Image(arr, spacing=spacing, origin=origin)


def _snowman_geometry(
    shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> tuple[
    tuple[tuple[float, float, float], float],
    tuple[tuple[float, float, float], float],
    tuple[tuple[float, float, float], float],
]:
    """Compute trunk/head/nose centres and radii in millimetres.

    The snowman is sized to fill the volume: trunk + head occupy ~80%
    of the Z extent, with the trunk diameter ~half the smaller of the
    in-plane physical extents so the body is large but doesn't crowd
    the volume edges.

    Returns:
        Three ``(center_zyx_mm, radius_mm)`` tuples for trunk, head,
        nose -- in that order.
    """
    sx, sy, sz = spacing_xyz
    z_mm = shape[0] * sz
    y_mm = shape[1] * sy
    x_mm = shape[2] * sx

    # Trunk diameter ~ half the smallest in-plane physical extent --
    # leaves a comfortable margin in Y/X for the nose protrusion.
    in_plane_min = min(y_mm, x_mm)
    body_r = 0.25 * in_plane_min
    head_r = 0.62 * body_r
    nose_r = 0.30 * head_r

    # Place the snowman vertically along Z: bottom of trunk just above
    # the volume floor, head touching the trunk, nose on the head.
    trunk_z = body_r + 0.05 * z_mm  # ~5% margin at the bottom
    head_z = trunk_z + body_r + 0.7 * head_r
    cy = y_mm / 2.0
    cx = x_mm / 2.0

    body_center = (trunk_z, cy, cx)
    head_center = (head_z, cy, cx)
    # Push the nose past the trunk's +Y silhouette so it's an
    # unambiguous feature in the sagittal view.
    nose_center = (head_z, cy + head_r + 1.5 * nose_r, cx)

    return (body_center, body_r), (head_center, head_r), (nose_center, nose_r)


def _snowman_mask(
    shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[tuple[float, float, float], float]]:
    """Build a snowman-shaped body mask in physical units.

    Three stacked physical-mm spheres -- a large trunk, a smaller head
    along ``+Z`` and a small nose protruding along ``+Y`` from the head
    -- make all three orthogonal views distinct.

    Returns:
        ``(mask, (trunk_center_zyx_mm, trunk_radius_mm))`` -- callers
        use the trunk pose to anchor the tumor and the segmentation
        labels.
    """
    (body_center, body_r), (head_center, head_r), (nose_center, nose_r) = _snowman_geometry(
        shape, spacing_xyz
    )
    mask: np.ndarray = (
        _phys_sphere_mask(shape, spacing_xyz, body_center, body_r)
        | _phys_sphere_mask(shape, spacing_xyz, head_center, head_r)
        | _phys_sphere_mask(shape, spacing_xyz, nose_center, nose_r)
    )
    return mask, (body_center, body_r)


def _irregular_tumor_mask(
    shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    center_zyx_mm: tuple[float, float, float],
    radius_mm: float,
    *,
    rng: np.random.Generator,
    n_blobs_range: tuple[int, int] = (3, 5),
) -> np.ndarray:
    """Compose 3-5 sub-spheres at perturbed positions in millimetres.

    Perturbations are bounded by ``radius_mm`` so the union stays
    cohesive (single connected blob) while still looking irregular.
    Sub-radii vary between 0.55 and 1.0 of ``radius_mm``.
    """
    n_blobs = int(rng.integers(n_blobs_range[0], n_blobs_range[1] + 1))
    mask = np.zeros(shape, dtype=bool)
    cz, cy, cx = center_zyx_mm
    for _ in range(n_blobs):
        sub_center = (
            cz + float(rng.uniform(-0.7, 0.7)) * radius_mm,
            cy + float(rng.uniform(-0.7, 0.7)) * radius_mm,
            cx + float(rng.uniform(-0.7, 0.7)) * radius_mm,
        )
        sub_radius = max(0.5, float(rng.uniform(0.55, 1.0)) * radius_mm)
        mask |= _phys_sphere_mask(shape, spacing_xyz, sub_center, sub_radius)
    return mask


def make_body_with_tumor(
    modality: Modality,
    *,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
    origin: tuple[float, float, float] = DEFAULT_ORIGIN,
    tumor_offset_mm: tuple[float, float, float] = (3.0, 2.0, 0.0),
    tumor_radius_mm: float = 5.0,
    tumor_seed: int = 0,
) -> Image:
    """Return a snowman-shaped body region with an irregular tumor.

    The body is the same snowman geometry for both modalities (trunk
    sphere + head sphere stacked along ``+Z`` + nose protruding along
    ``+Y``). Only the intensities differ:

    - ``"ct"`` -> body ``40`` HU (soft tissue), tumor ``200`` HU
      (contrast), background ``-1000`` HU (air).
    - ``"mr"`` -> body ``600`` (T1-ish), tumor ``1200`` (hyperintense),
      background ``0``.

    The tumor itself is a union of 3-5 sub-spheres placed at small
    pseudo-random perturbations around the tumor centre, giving each
    seed a recognisable irregular silhouette.

    Args:
        modality: ``"ct"`` or ``"mr"``.
        shape: Spatial array shape ``(Z, Y, X)`` (slowest axis first; see
            module-level note).
        spacing: Voxel spacing per dimension (mm).
        origin: World-space origin (mm).
        tumor_offset_mm: Tumor centre offset from the trunk centre, in
            millimetres, in array-axis order ``(dz, dy, dx)``.
        tumor_radius_mm: Nominal tumor radius in millimetres; the
            irregular composition uses sub-radii scaled around it.
        tumor_seed: RNG seed for the tumor's sub-sphere placement.
            Passing different seeds across cases yields visibly distinct
            tumor silhouettes (used by :func:`make_cohort`).

    Returns:
        :class:`Image` of float32 pixels.

    Raises:
        ValueError: If ``modality`` is not ``"ct"`` or ``"mr"``.
    """
    if modality == "ct":
        arr = np.full(shape, -1000.0, dtype=np.float32)
        body_value, tumor_value = 40.0, 200.0
    elif modality == "mr":
        arr = np.zeros(shape, dtype=np.float32)
        body_value, tumor_value = 600.0, 1200.0
    else:
        raise ValueError(f"modality must be 'ct' or 'mr', got {modality!r}")

    body_mask, (trunk_center, _trunk_r) = _snowman_mask(shape, spacing)
    arr[body_mask] = body_value

    tumor_center = (
        trunk_center[0] + tumor_offset_mm[0],
        trunk_center[1] + tumor_offset_mm[1],
        trunk_center[2] + tumor_offset_mm[2],
    )
    rng = np.random.default_rng(tumor_seed)
    tumor_mask = _irregular_tumor_mask(shape, spacing, tumor_center, tumor_radius_mm, rng=rng)
    arr[tumor_mask] = tumor_value

    return Image(arr, spacing=spacing, origin=origin)


# ---------------------------------------------------------------------------
# Segmentation phantom
# ---------------------------------------------------------------------------


def make_segmentation(
    reference: Image,
    *,
    label_names: Sequence[str] = ("tumor", "edema", "necrosis"),
    seed: int = 0,
    group_name: str = "main",
) -> MultiLabelSegmentation:
    """Return a single-group segmentation co-registered with ``reference``.

    Each label is an irregular physical-mm blob built from 3-5
    perturbed sub-spheres -- the same recipe used by
    :func:`make_body_with_tumor` for the tumor -- so labels have a
    lumpy, recognisable silhouette rather than looking like a uniform
    grid of spheres. Blob centres are picked from the trunk eroded by
    the nominal label radius, then clipped to the trunk and to voxels
    not yet claimed by an earlier label, so the result is a partition
    fully contained in the body.

    The trunk pose is recovered from ``reference``'s geometry via the
    same recipe :func:`make_body_with_tumor` uses, so this helper is
    only meaningful when ``reference`` is one of those phantoms.

    Args:
        reference: Image whose geometry the segmentation inherits and
            whose snowman trunk hosts the label blobs.
        label_names: Names for each label, in order. Determines the
            label count.
        seed: RNG seed for blob placement (reproducible across runs).
        group_name: Name of the single label group.

    Returns:
        :class:`MultiLabelSegmentation` with one group.

    Raises:
        ValueError: If ``label_names`` is empty.
    """
    if not label_names:
        raise ValueError("label_names must not be empty")

    rng = np.random.default_rng(seed)
    shape: tuple[int, int, int] = (
        int(reference.shape[0]),
        int(reference.shape[1]),
        int(reference.shape[2]),
    )
    spacing: tuple[float, float, float] = (
        float(reference.spacing[0]),
        float(reference.spacing[1]),
        float(reference.spacing[2]),
    )

    (trunk_center, trunk_r), _, _ = _snowman_geometry(shape, spacing)
    label_radius_mm = min(3.5, 0.45 * trunk_r)
    safe_zone = _phys_sphere_mask(
        shape, spacing, trunk_center, max(0.5, trunk_r - label_radius_mm)
    )
    safe_indices = np.argwhere(safe_zone)
    if len(safe_indices) == 0:
        raise ValueError("reference image is too small to host labels with the snowman recipe")
    trunk_mask = _phys_sphere_mask(shape, spacing, trunk_center, trunk_r)

    seg = MultiLabelSegmentation.create(
        groups=[LabelGroup(name=group_name)],
        reference=reference,
    )

    arr = np.zeros(shape, dtype=LABEL_DTYPE)
    sx, sy, sz = spacing
    for idx, name in enumerate(label_names):
        ck, cj, ci = safe_indices[int(rng.integers(0, len(safe_indices)))]
        center_mm = (float(ck) * sz, float(cj) * sy, float(ci) * sx)
        radius_mm = float(rng.uniform(0.7, 1.0)) * label_radius_mm
        # Irregular blob: same composition recipe as the tumor.
        # Sub-spheres can extend past the eroded safe zone, but the
        # AND with trunk_mask clips them; the centre voxel is at
        # least label_radius_mm inside the trunk so the blob always
        # leaves at least one voxel inside the body.
        mask = (
            _irregular_tumor_mask(shape, spacing, center_mm, radius_mm, rng=rng)
            & trunk_mask
            & (arr == 0)
        )
        value = idx + 1
        arr[mask] = value
        seg.add_label(
            Label(value=value, name=name, color=_PALETTE[idx % len(_PALETTE)]),
            group=0,
        )

    seg.set_group_image(0, arr)
    return seg


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseBundle:
    """One synthetic case: image + segmentation + metadata.

    Attributes:
        case_id: Stable identifier (e.g. ``"case_001"``).
        image: The case's spatial image.
        segmentation: A single-group segmentation co-registered with the
            image.
        metadata: Free-form per-case info (modality flag, generator
            parameters, ...).
    """

    case_id: str
    image: Image
    segmentation: MultiLabelSegmentation
    metadata: Mapping[str, Any] = field(default_factory=dict)


def make_cohort(
    n: int = 3,
    *,
    seed: int = 0,
    modality: Modality = "mr",
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
) -> list[CaseBundle]:
    """Return ``n`` synthetic cases with consistent per-case geometry.

    Tumor location and label count vary per case so the cohort loop in
    notebook 08 has visible variation.

    Args:
        n: Number of cases.
        seed: Master RNG seed; derives a per-case seed.
        modality: Modality flag forwarded to :func:`make_body_with_tumor`.
        shape: Spatial array shape ``(Z, Y, X)`` per case (slowest axis
            first; see module-level note).
        spacing: Voxel spacing per case.

    Returns:
        List of :class:`CaseBundle`.

    Raises:
        ValueError: If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    master = np.random.default_rng(seed)
    label_pool: tuple[str, ...] = ("tumor", "edema", "necrosis", "lesion")

    cases: list[CaseBundle] = []
    for i in range(n):
        sub_seed = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        # Offsets are in millimetres around the trunk centre. Keep them
        # comfortably inside the trunk so the tumor stays surrounded by
        # body tissue.
        offset_mm = (
            float(rng.uniform(-6.0, 6.0)),
            float(rng.uniform(-6.0, 6.0)),
            float(rng.uniform(-6.0, 6.0)),
        )
        image = make_body_with_tumor(
            modality,
            shape=shape,
            spacing=spacing,
            tumor_offset_mm=offset_mm,
            tumor_seed=sub_seed,
        )
        n_labels = int(rng.integers(2, len(label_pool) + 1))
        seg = make_segmentation(
            image,
            label_names=label_pool[:n_labels],
            seed=sub_seed,
        )
        cases.append(
            CaseBundle(
                case_id=f"case_{i + 1:03d}",
                image=image,
                segmentation=seg,
                metadata={
                    "modality": modality,
                    "tumor_offset_mm": offset_mm,
                    "seed": sub_seed,
                },
            )
        )

    return cases
