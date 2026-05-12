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

"""Converter between mw.MultiLabelSegmentation and mitk.MultiLabelSegmentation.

Used by MultiLabelSegmentation.to_mitk() and MultiLabelSegmentation.from_mitk().
Not used by node._resolve_serialized_bytes or node._get_data_multilabel, which
call mitk.IOUtil directly.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mitk_workbench_remote.multilabel import MultiLabelSegmentation


class MitkSegmentationConverter:
    """Converter between mw.MultiLabelSegmentation and mitk.MultiLabelSegmentation.

    Both conversion directions use a NRRD round-trip, with each side's own I/O
    stack handling the serialization it is authoritative for:

    * ``from_segmentation``: mw writes NRRD via pynrrd; mitk reads via the path
      constructor (C++ ``LabelSetImageIO``).
    * ``to_segmentation``: mitk writes NRRD via ``IOUtil.save``; mw reads via
      pynrrd.

    This converter is only used by :meth:`MultiLabelSegmentation.to_mitk` and
    :meth:`MultiLabelSegmentation.from_mitk`.  The ``node.py`` upload and
    download paths call ``mitk.IOUtil`` directly to avoid an unnecessary
    abstraction layer.
    """

    def from_segmentation(self, seg: Any) -> Any:
        """Convert an ``mw.MultiLabelSegmentation`` to a ``mitk.MultiLabelSegmentation``.

        Serializes the mw object to NRRD via the pynrrd writer, then loads it
        with MITK's C++ ``LabelSetImageIO`` via the path constructor.

        Args:
            seg: An ``mw.MultiLabelSegmentation`` instance.

        Returns:
            A ``mitk.MultiLabelSegmentation`` instance.

        Raises:
            ImportError: If the ``mitk`` package is not installed.
        """
        import mitk

        from mitk_workbench_remote import _io

        nrrd_bytes = _io.write_multilabel_nrrd(seg)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            tmp_path.write_bytes(nrrd_bytes)
            return mitk.IOUtil.load(str(tmp_path))[0]
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def to_segmentation(self, mitk_seg: Any) -> MultiLabelSegmentation:
        """Convert a ``mitk.MultiLabelSegmentation`` to an ``mw.MultiLabelSegmentation``.

        Serializes the mitk object to NRRD via ``mitk.IOUtil.save()`` (C++
        authoritative path), then reads it with the mw pynrrd reader.

        Args:
            mitk_seg: A ``mitk.MultiLabelSegmentation`` instance.

        Returns:
            An ``mw.MultiLabelSegmentation`` instance.

        Raises:
            ImportError: If the ``mitk`` package is not installed.
        """
        import mitk

        from mitk_workbench_remote import _io

        # MITK resolves the MultiLabelSegmentation writer via a MIME type whose
        # AppliesTo() inspects the file header when the path exists. We need a
        # path that does *not* exist yet; create a private directory and let
        # mitk.IOUtil.save populate a deterministic name inside it. This avoids
        # the TOCTOU window of unlink-then-save on a shared /tmp.
        tmp_dir = Path(tempfile.mkdtemp(prefix="mw_mitk_seg_"))
        try:
            tmp_path = tmp_dir / "data.nrrd"
            mitk.IOUtil.save(mitk_seg, str(tmp_path))
            return _io.read_multilabel_nrrd(tmp_path.read_bytes())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
