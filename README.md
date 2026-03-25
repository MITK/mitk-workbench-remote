# mitk-workbench-remote

**Remote control for MITK Workbench from Python**

[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![SemVer](https://img.shields.io/badge/semver-2.0.0-blue.svg)](https://semver.org/)
[![Documentation](https://readthedocs.org/projects/mitk-workbench-remote/badge/?version=latest)](https://mitk-workbench-remote.readthedocs.io/en/latest/)

`mitk-workbench-remote` is a Python library for controlling running [MITK Workbench](https://www.mitk.org) instances programmatically via their REST API. It is designed for researchers and developers who want to drive the Workbench from Python scripts, Jupyter notebooks, and ML pipelines — loading images, editing segmentations, and managing data nodes without touching the GUI.

## Quick start

```python
import mitk_workbench_remote as mw

# Connect to a running Workbench
wb = mw.connect(port=8080)

# Show a file, a numpy array, or an Image object
wb.show("scan.nrrd", opacity=0.8)
wb.show(my_array, name="Prediction", color=(1.0, 0.5, 0.0))
```

## Requirements

A running MITK Workbench instance with REST API enabled.

## Installation

```bash
pip install mitk-workbench-remote
```

With optional image format support (SimpleITK and mlarray):

```bash
pip install mitk-workbench-remote[all]
```

## Development

```bash
git clone <repo>
cd mitk-workbench-remote
pip install -e ".[dev]"
pytest
```

## Documentation

Full documentation at [docs.mitk.org/mitk-workbench-remote](https://docs.mitk.org/mitk-workbench-remote).

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