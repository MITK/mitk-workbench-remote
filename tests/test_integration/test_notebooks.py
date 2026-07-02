# SPDX-FileCopyrightText: 2026, German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC)
#
# SPDX-License-Identifier: Apache-2.0

"""By-mode notebook smoke tests.

Executes every example notebook (except notebook 05, the launch notebook, which
is environment-specific and unrelated to ``mitk-python``) end-to-end against a
running MITK Workbench, in two modes:

- ``remote``: the ``mitk`` package is forced *unimportable* inside the kernel,
  so ``DataNode.get_data()`` returns the remote ``mw.*`` types (the AUTO
  fallback). This always runs (given a Workbench).
- ``mitk``: the ``mitk`` package is required to be importable; ``get_data()``
  then returns native ``mitk.*`` types. When ``mitk`` is not importable in the
  kernel the leg skips itself.

This is a drift detector: the *same* notebook cells must succeed in *both*
modes. It catches the class of bug where an optional dependency becoming
importable silently changes return types and breaks otherwise-working code
(see ``SEG_API_ALIGNMENT_*.md``).

Run with::

    pytest tests/test_integration/test_notebooks.py -v -m integration

Design notes:

- **The kernel is the single source of truth for mode.** Both the "block mitk"
  and "require mitk" decisions are enforced by a cell prepended into the kernel,
  not by probing the pytest interpreter. The Jupyter ``python3`` kernel may
  resolve to a different interpreter than the one running pytest; probing in the
  kernel avoids a false *green* (mitk present in pytest but absent in the
  kernel would otherwise run remote types under the ``mitk`` label and pass).
- **Preconditions skip; real errors fail.** The notebooks raise ``SystemExit``
  for unmet preconditions (no Workbench discovered, a required editor not open,
  mitk-only DSL unavailable). Those are turned into skips. Only other exceptions
  (``TypeError``, ``AttributeError``, ...) — the actual API drift — fail.
  Caveat: a ``SystemExit`` skip is **not** a pass. A ``discover()``-based
  notebook can flaky-skip on a missed UDP beacon; a skipped ``mitk-03`` leg
  does not prove the drift is fixed — only a green run does.
- **Coverage gap (intentional):** cells the author marked ``# doctest: +SKIP``
  (the ``as_type=MITK`` / ``to_mitk()`` interop demos in nb02) are dropped in
  both modes. They are covered by the converter/node unit tests; this smoke test
  targets the AUTO ``get_data()`` path, which is the actual footgun.
- Notebooks execute in a throwaway temp directory, so files they write
  (``*.nrrd``, ``*.png``) never touch the repo. Committed notebook outputs are
  not modified — execution happens on an in-memory copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# nbclient/nbformat are dev-only deps (see the [dev] extra). Guard the imports so
# a plain `pytest` run on an environment without them skips this module at
# collection instead of erroring.
nbformat = pytest.importorskip("nbformat")
_nbclient = pytest.importorskip("nbclient")
from nbclient import NotebookClient  # noqa: E402
from nbclient.exceptions import CellExecutionError  # noqa: E402

import mitk_workbench_remote as mw  # noqa: E402

pytestmark = pytest.mark.integration

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "examples"

# Notebook 05 (discovery & launch) is excluded: launching a Workbench is a
# trickier, environment-specific concern and is not bound to mitk-python.
EXCLUDED = {"05_discovery_and_launch.ipynb"}

NOTEBOOKS = sorted(p for p in EXAMPLES_DIR.glob("*.ipynb") if p.name not in EXCLUDED)

MODES = ("remote", "mitk")

# Per-cell execution timeout (seconds). Generous because some notebooks render
# screenshots and transfer image data over the network.
CELL_TIMEOUT = 300

# Prepended (as the first cell) in "remote" mode. Makes `import mitk` fail inside
# the kernel even if mitk-python is installed, and must run before the notebook
# imports mitk_workbench_remote so the optional mitk converters are not
# registered and get_data() takes its remote fallback path.
_BLOCK_MITK_SRC = '''\
import sys as _sys


class _MitkImportBlocker:
    """meta_path finder that refuses every `mitk`/`mitk.*` import."""

    def find_spec(self, name, path=None, target=None):
        if name == "mitk" or name.startswith("mitk."):
            raise ModuleNotFoundError(
                "mitk import blocked by the by-mode notebook test (remote mode)"
            )
        return None


for _m in [k for k in list(_sys.modules) if k == "mitk" or k.startswith("mitk.")]:
    del _sys.modules[_m]
_sys.meta_path.insert(0, _MitkImportBlocker())
del _sys
'''

# Prepended (as the first cell) in "mitk" mode. Asserts — inside the kernel, the
# single source of truth — that mitk is genuinely importable. If it is not, the
# SystemExit is converted to a skip by _execute() rather than producing a false
# pass on remote types.
_REQUIRE_MITK_SRC = """\
try:
    import mitk as _mitk  # noqa: F401

    del _mitk
except Exception as _exc:  # noqa: BLE001
    raise SystemExit(
        f"mitk package not importable in this kernel; mitk-mode not exercised: {_exc!r}"
    )
"""

_PRELUDE = {"remote": _BLOCK_MITK_SRC, "mitk": _REQUIRE_MITK_SRC}


def _workbench_reachable() -> bool:
    url = os.environ.get("MITK_WORKBENCH_URL", "http://localhost:8080")
    token = os.environ.get("MITK_WORKBENCH_TOKEN")
    try:
        return mw.connect(url, token=token).ping()
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_workbench() -> None:
    if not _workbench_reachable():
        pytest.skip(
            "No MITK Workbench reachable (set MITK_WORKBENCH_URL, default http://localhost:8080)"
        )


def _is_skip_cell(cell: nbformat.NotebookNode) -> bool:
    """True for code cells the author marked non-executable (``# doctest: +SKIP``).

    Used in the example notebooks to flag interop cells that require the ``mitk``
    package (``as_type=MITK``, ``to_mitk()``). They are demos covered by unit
    tests, not part of the AUTO get_data() path this smoke test guards, so they
    are dropped before execution in every mode.
    """
    if cell.cell_type != "code":
        return False
    src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
    return "doctest:+skip" in src.lower().replace(" ", "")


def _execute(nb_path: Path, run_dir: Path, *, mode: str) -> None:
    """Execute a notebook on an in-memory copy; raises on any genuine cell error."""
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells = [c for c in nb.cells if not _is_skip_cell(c)]
    prelude = nbformat.v4.new_code_cell(source=_PRELUDE[mode])
    prelude.id = f"mw_mode_{mode}"  # set id to avoid MissingIDFieldWarning
    nb.cells.insert(0, prelude)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(run_dir)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        # Unmet precondition (no Workbench, editor not open, mitk-only DSL, or the
        # mitk-mode probe above) -> skip. Anything else (TypeError, ...) is the
        # API drift this test exists to catch -> fail.
        if exc.ename == "SystemExit":
            pytest.skip(f"notebook precondition not met: {exc.evalue}")
        raise


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("mode", MODES)
def test_notebook_runs(nb_path: Path, mode: str, require_workbench: None, tmp_path: Path) -> None:
    _execute(nb_path, tmp_path, mode=mode)
