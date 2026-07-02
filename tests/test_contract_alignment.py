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

"""Cross-surface contract for MultiLabelSegmentation (drift detector).

One shared set of behavioral assertions is run against **both** segmentation
surfaces: the remote ``mitk_workbench_remote.MultiLabelSegmentation`` and, when
the ``mitk`` package is importable, the native ``mitk.MultiLabelSegmentation``.
The point is that ``DataNode.get_data(as_type=AUTO)`` returns one type or the
other, so the same user code must behave the same way on both.

Unlike the by-mode notebook test (``tests/test_integration/test_notebooks.py``),
this needs **no running Workbench**: it constructs each object in-process and
drives it through the shared API. It therefore runs in plain CI and catches
signature/behavior drift (such as ``add_group`` accepting ``image``/``labels``
positionally) that a notebook only exercises if it happens to make the call.

The ``mitk`` parametrization auto-skips (via ``importorskip``) when the wheel is
not installed; the remote parametrization always runs.

Deliberately NOT asserted here, because the two surfaces diverge (each covered
by its own tests, see ``SEG_API_ALIGNMENT_*``):

- ``merge_labels``: the remote version is a simplified stand-in that removes the
  sources and emits ``MitkApiDivergenceWarning``; native retains sources and
  supports an ``overwrite_style``.
- ``add_label`` on a value collision: remote raises; native clones and reassigns.
- ``set_group_name(index, None)``: accepted by remote, rejected by native.
- ``remove_labels``/``erase_labels`` with a **missing** value: remote is atomic
  and raises ``LookupError`` (touching nothing). Native is inconsistent here --
  ``remove_labels`` silently skips the missing value (partial removal) while
  ``erase_labels`` raises a non-``LookupError`` exception. Remote keeps the
  stricter, consistent behavior (CLAUDE.md: throw, don't silently skip). The
  all-valid happy path is identical on both and IS asserted below; convergence
  of the missing-value case is tracked in MITK issue #868
  (https://git.dkfz.de/mic/mitk/-/work_items/868). Remote-side atomicity is
  covered in ``tests/test_multilabel.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mitk_workbench_remote.multilabel import Label, MultiLabelSegmentation

# Shared geometry for every constructed segmentation.
SHAPE = (4, 5, 6)
SPACING = (1.0, 2.0, 3.0)


class _Backend:
    """Constructs segmentations and labels for one surface under test.

    ``empty_seg()`` returns a segmentation with a single empty group at index 0
    and the shared geometry, built through that surface's own construction path.
    ``label()`` builds a label of the matching type.
    """

    name: str

    def empty_seg(self) -> Any:
        raise NotImplementedError

    def label(
        self, value: int, name: str, color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Any:
        raise NotImplementedError


class _RemoteBackend(_Backend):
    name = "remote"

    def empty_seg(self) -> Any:
        seg = MultiLabelSegmentation.create(shape=SHAPE, spacing=SPACING)
        seg.add_group("G0")
        return seg

    def label(
        self, value: int, name: str, color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Any:
        return Label(value, name, color=color)


class _MitkBackend(_Backend):
    name = "mitk"

    def __init__(self, mitk: Any) -> None:
        self._mitk = mitk

    def empty_seg(self) -> Any:
        ref = self._mitk.Image.from_numpy(np.zeros(SHAPE, dtype=np.uint16), spacing=SPACING)
        # The native constructor always creates exactly one empty group.
        return self._mitk.MultiLabelSegmentation(ref)

    def label(
        self, value: int, name: str, color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Any:
        lbl = self._mitk.Label(value, name)
        lbl.color = color
        return lbl


@pytest.fixture(params=["remote", "mitk"])
def backend(request: pytest.FixtureRequest) -> _Backend:
    if request.param == "mitk":
        mitk = pytest.importorskip("mitk")
        return _MitkBackend(mitk)
    return _RemoteBackend()


# ---------------------------------------------------------------------------
# Existence and lookup
# ---------------------------------------------------------------------------


def test_has_label_never_raises(backend: _Backend) -> None:
    seg = backend.empty_seg()
    assert seg.has_label(999) is False
    assert seg.has_label(0) is False  # UNLABELED_VALUE is never a registered label
    seg.add_label(backend.label(1, "A"), 0)
    assert seg.has_label(1) is True


def test_get_label_raises_lookuperror_on_miss(backend: _Backend) -> None:
    seg = backend.empty_seg()
    with pytest.raises(LookupError):
        seg.get_label(999)


def test_get_group_raises_lookuperror_on_miss(backend: _Backend) -> None:
    seg = backend.empty_seg()
    with pytest.raises(LookupError):
        seg.get_group(999)


def test_get_group_of_label(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(1, "A"), 0)
    assert seg.get_group_of_label(1) == 0
    with pytest.raises(LookupError):
        seg.get_group_of_label(999)


# ---------------------------------------------------------------------------
# Label lifecycle
# ---------------------------------------------------------------------------


def test_remove_label_takes_no_clear_pixels_kwarg(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(1, "A"), 0)
    with pytest.raises(TypeError):
        seg.remove_label(1, clear_pixels=True)


def test_remove_label_drops_the_label(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(1, "A"), 0)
    seg.remove_label(1)
    assert seg.has_label(1) is False


def test_remove_labels_batch(backend: _Backend) -> None:
    # Happy path (all values present) is identical on both surfaces. The
    # missing-value case diverges by design and is documented in the module
    # docstring (MITK issue #868), not asserted as a shared contract.
    seg = backend.empty_seg()
    seg.add_label(backend.label(1, "A"), 0)
    seg.add_label(backend.label(2, "B"), 0)
    seg.remove_labels([1, 2])
    assert seg.has_label(1) is False
    assert seg.has_label(2) is False


def test_erase_label_keeps_label(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(1, "A"), 0)
    seg.erase_label(1)
    assert seg.has_label(1) is True  # metadata retained, only pixels cleared
    with pytest.raises(LookupError):
        seg.erase_label(999)


def test_rename_label(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(1, "Old"), 0)
    seg.rename_label(1, "New", (0.5, 0.6, 0.7))
    assert seg.get_label(1).name == "New"
    with pytest.raises(LookupError):
        seg.rename_label(999, "X", (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# add_label overloads
# ---------------------------------------------------------------------------


def test_add_label_name_color_overload(backend: _Backend) -> None:
    seg = backend.empty_seg()
    lbl = seg.add_label("Liver", (0.8, 0.2, 0.1))  # (name, color), default group 0
    assert int(lbl.value) == 1
    assert seg.get_group_of_label(1) == 0


def test_add_label_object_with_positional_group(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(3, "C"), 0)  # (label, group) positional
    assert seg.has_label(3) is True


# ---------------------------------------------------------------------------
# Group management (the add_group signature parity this MR fixed)
# ---------------------------------------------------------------------------


def test_add_group_accepts_positional_image_and_labels(backend: _Backend) -> None:
    seg = backend.empty_seg()
    before = seg.num_groups
    # Native is add_group(name, image, labels); labels must be passable
    # positionally on both surfaces. image=None sidesteps the ndarray-vs-
    # mitk.Image type difference while still exercising the positional shape.
    idx = seg.add_group("G1", None, [backend.label(2, "B")])
    assert seg.num_groups == before + 1
    assert 2 in seg.get_group_label_values(idx)


def test_set_group_name(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.set_group_name(0, "Renamed")
    assert seg.get_group(0).name == "Renamed"
    with pytest.raises(LookupError):
        seg.set_group_name(999, "X")


# ---------------------------------------------------------------------------
# Properties (attribute access, not method calls) and sorted lookups
# ---------------------------------------------------------------------------


def test_num_groups_is_a_property(backend: _Backend) -> None:
    seg = backend.empty_seg()
    assert isinstance(seg.num_groups, int)
    assert seg.num_groups == 1  # the single starting group


def test_label_values_is_sorted_ascending(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(3, "C"), 0)
    seg.add_label(backend.label(1, "A"), 0)
    seg.add_label(backend.label(2, "B"), 0)
    values = [int(v) for v in seg.label_values if int(v) != 0]
    assert values == [1, 2, 3]


def test_get_label_values_by_name_sorted_ascending(backend: _Backend) -> None:
    seg = backend.empty_seg()
    seg.add_label(backend.label(5, "Dup"), 0)
    seg.add_label(backend.label(2, "Dup"), 0)
    seg.add_label(backend.label(9, "Other"), 0)
    assert [int(v) for v in seg.get_label_values_by_name("Dup")] == [2, 5]
    assert list(seg.get_label_values_by_name("Missing")) == []
