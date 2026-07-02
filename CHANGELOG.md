# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

> **Note on pre-release versions:** Stable releases follow SemVer 2.0 exactly (`MAJOR.MINOR.PATCH`).
> Pre-release and development versions use [PEP 440](https://peps.python.org/pep-0440/) notation
> (e.g. `1.0.0a1`, `1.0.0.dev0`) as required by the Python packaging ecosystem, which differs
> from SemVer pre-release syntax (`1.0.0-alpha.1`, `1.0.0-dev.0`).

## [Unreleased]

### Added
- Segmentation API parity with native `mitk.MultiLabelSegmentation`:
  `has_label`, `get_group_of_label`, `num_groups`, `set_group_name`,
  `erase_label`/`erase_labels`, `remove_labels`, `rename_label`, a
  `(name, color)` overload for `add_label` (with a default `group=0`),
  `add_group(image=, labels=)`, and the lookups `label_values`, `get_labels`,
  `get_label_values_by_name`, `get_group_label_values`. `LabelGroup` now exposes
  `index`, `labels`, and `image` as a live view over its segmentation.
- `merge_labels(target, sources)` as a simplified, best-effort operation: it
  reassigns source pixels to the target and removes the sources. It diverges
  from native MITK (no lock handling or merge/overwrite styles; native retains
  the sources) and emits the new `MitkApiDivergenceWarning` on use.
- `MitkApiDivergenceWarning`, emitted when a remote method knowingly diverges in
  behavior from native MITK.

### Changed (breaking)
- `MultiLabelSegmentation.get_label(value)` now raises `KeyError` when the value
  is absent instead of returning `None`. Use `has_label(value)` for existence
  checks.
- `MultiLabelSegmentation.remove_label(value)` no longer accepts `clear_pixels`;
  it always removes the label and zeroes its pixels. To clear pixels but keep
  the label, use `erase_label(value)`.
- `LabelGroup.name` is now read-only; rename a group via
  `MultiLabelSegmentation.set_group_name(index, name)`.
- Label-value lookups that miss now raise `KeyError` (previously `ValueError`
  for `remove_label`); group-index misses raise `IndexError`. Both subclass
  `LookupError`.

### Changed
- `Image(...)` raises a clearer `TypeError` for unconvertible inputs: it now
  enumerates the accepted inputs (numpy ndarray, `SimpleITK.Image`, `mitk.Image`,
  `mlarray.MLArray`, or a `register_converter()` type) and notes that native
  non-image `mitk.*` types are not spatial images.

### Fixed
- Example notebook `02_image_data.ipynb`: the SimpleITK section now fetches the
  remote image explicitly with `get_data(as_type=REMOTE)` before
  `to_simpleitk()`, so it runs whether or not the `mitk` package is installed
  (`get_data()` under `AUTO` returns a native `mitk.Image`, which has no
  `to_simpleitk()`). The AUTO return-type footgun is now documented in the
  interop guide.

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
- `InsecureTransportWarning` when an API token is configured for a non-local
  `http://` host, since the `Authorization` header would travel in cleartext.
- Automatic environment-proxy bypass for loopback connections (a forward proxy
  cannot route to the caller's own machine), with a `trust_env` parameter on
  `connect()` to override it.

### Fixed
- `launch()` no longer mistakes the Windows `.bat` wrapper's immediate exit-0
  for a failure: it tolerates exit-0 and keeps polling `/health` (only a
  non-zero exit fails fast).
- `shutdown()` finds the listening process locale- and address-family-
  independently (no longer keyed on the English word `LISTENING`), so it kills
  the real Workbench on non-English Windows and for dual-stack listeners.
- The launched child's stderr is redirected to a temp file instead of an
  inherited PIPE, removing a deadlock risk over long sessions; the file is
  cleaned up on `shutdown()` and on failed launches.
