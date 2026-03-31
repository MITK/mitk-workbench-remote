# Configuration file for the Sphinx documentation builder.

import os
import sys

# Make the package importable for autodoc
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# Ensure pandoc from pypandoc_binary is on PATH for nbsphinx/nbconvert
try:
    import pypandoc

    pandoc_dir = os.path.dirname(pypandoc.get_pandoc_path())
    os.environ["PATH"] = pandoc_dir + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

from mitk_workbench_remote._version import __version__  # noqa: E402

project = "mitk-workbench-remote"
copyright = "2026, German Cancer Research Center (DKFZ)"
author = "MITK Development Team"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

# -- General -----------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc -----------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

# -- Napoleon (Google-style docstrings) --------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Type hints --------------------------------------------------------

always_document_param_types = False
typehints_defaults = "braces"

# Suppress duplicate object warnings from sphinx-autodoc-typehints
# re-documenting dataclass attributes that autodoc already emitted.
suppress_warnings = ["autodoc.duplicate_object"]

# -- Intersphinx -------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- nbsphinx ---------------------------------------------------------

nbsphinx_execute = "never"

# -- HTML output -------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []
