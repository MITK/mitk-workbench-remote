# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

> **Note on pre-release versions:** Stable releases follow SemVer 2.0 exactly (`MAJOR.MINOR.PATCH`).
> Pre-release and development versions use [PEP 440](https://peps.python.org/pep-0440/) notation
> (e.g. `1.0.0a1`, `1.0.0.dev0`) as required by the Python packaging ecosystem, which differs
> from SemVer pre-release syntax (`1.0.0-alpha.1`, `1.0.0-dev.0`).

## [Unreleased]

## [0.1.0]

### Added
- `connect()` to attach to a running MITK Workbench, and a `Workbench.show()`
  one-liner for files, numpy arrays, SimpleITK images, and mlarray objects.
- `DataStorage` CRUD: list, filter, create, and remove data nodes.
- `Image` universal spatial image type, with automatic transfer-mode negotiation
  (direct HTTP or file-reference).
- MultiLabel segmentation editing: groups, labels, pixel data, and full round-trip.
- Editor and render-window handles for the StdMultiWidget and MxN editors, with
  rendering control: crosshair position, reinit, and screenshots.
- Typed MxN layout DSL (via the optional `mitk` package) alongside raw-dict
  layout I/O that needs no optional dependency.
- Converter registry for image types: numpy always available, SimpleITK and
  mlarray auto-registered when installed, plus user-registerable converters.
- Discovery and launch helpers to find running instances or start new ones.
- Structured logging via Python's `logging` module.
