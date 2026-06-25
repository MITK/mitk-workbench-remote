# mitk-workbench-remote

**Remote control for MITK Workbench from Python**

[![Python 3.11](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![SemVer](https://img.shields.io/badge/semver-2.0.0-blue.svg)](https://semver.org/)
[![Documentation](https://readthedocs.org/projects/mitk-workbench-remote/badge/?version=latest)](https://mitk-workbench-remote.readthedocs.io/en/latest/)

`mitk-workbench-remote` is a Python library for controlling running [MITK Workbench](https://www.mitk.org) instances programmatically via their REST API. It is designed for researchers and developers who want to drive the Workbench from Python scripts, Jupyter notebooks, and ML pipelines — loading images, editing segmentations, and managing data nodes without touching the GUI.

## Quick start

```python
import mitk_workbench_remote as mw

# Connect to a running Workbench
wb = mw.connect("http://localhost:8080")

# Show a file, an Image object or any data format directly supported
# (e.g. numpy array, sitk, mlarray)
wb.show("scan.nrrd", opacity=0.8)
wb.show(my_array, name="Prediction", color=(1.0, 0.5, 0.0))
```

Connections to a loopback host (`localhost`, `127.0.0.1`, `::1`) ignore the
environment proxy settings by default, since a forward/corporate proxy cannot
route to your own machine. Pass `trust_env=True` (or `False`) to `connect()` to
override this.

## Features

- **`wb.show()` one-liner** for files, numpy arrays, SimpleITK images, and mlarray objects
- **DataStorage CRUD** - list, filter, create, and remove data nodes
- **Image transfer** with automatic transfer mode negotiation (direct HTTP or file-reference)
- **MultiLabel segmentation editing** - groups, labels, pixel data, and full round-trip
- **Rendering control** - crosshair position, reinit, and screenshots
- **Discovery and launch** - find running instances or start new ones
- **Converter registry** for custom image types
- **Structured logging** via Python's `logging` module

## Requirements

A running MITK Workbench instance with REST API enabled.

## Installation

```bash
pip install mitk-workbench-remote
```

With optional image format support ([SimpleITK](https://simpleitk.org/) and [MLArray](https://github.com/MIC-DKFZ/mlarray/)):

```bash
pip install mitk-workbench-remote[all]
```

`[all]` covers the image-format converters and installs on every supported Python
(3.11+). The MITK interop and the typed MxN layout DSL need the `mitk` / `layout`
extra, which depends on `mitk-python`; that package currently ships CPython 3.12
wheels only, so those extras require Python 3.12. To install everything on 3.12:

```bash
pip install "mitk-workbench-remote[all,layout]"
```

## Development

```bash
git clone https://github.com/MITK/mitk-workbench-remote.git
cd mitk-workbench-remote
pip install -e ".[dev]"
pytest
```

## Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See [docs/examples/](docs/examples/) for Jupyter notebooks covering common workflows.

## Documentation

Full documentation at [mitk-workbench-remote.readthedocs.io](https://mitk-workbench-remote.readthedocs.io).

## Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) for stable
releases. In short: `MAJOR` bumps mean breaking API changes, `MINOR` bumps add functionality in a
backwards-compatible way, and `PATCH` bumps are backwards-compatible bug fixes.

Pre-release and development builds use [PEP 440](https://peps.python.org/pep-0440/) notation
(e.g. `1.0.0a1`, `0.1.0.dev0`) as required by the Python packaging ecosystem. The semantics are
equivalent to SemVer pre-release identifiers, only the syntax differs.

## License
Please ensure your usage complies with the code license.
Apache-2.0 — see [LICENSE](LICENSE).

## 🆘 Support & Contributing

- 📖 Documentation: Full API documentation available
- 🐛 Issues: Report bugs and feature requests
- 💬 Discussion: Join our community for questions and tips
- 🔧 Contributing: We welcome contributions. More details can be found in the dedicated Contribution Guide.

## Copyright & License

Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC).