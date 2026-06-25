# Getting Started

## Prerequisites

- **Python 3.11+**
- A running **MITK Workbench** instance with the REST API enabled.
  In the Workbench, go to *Window > Preferences > REST API*, check *Enable*
  and *Auto-start*, then restart.

## Installation

Base install (numpy, requests, pynrrd):

```bash
pip install mitk-workbench-remote
```

With optional SimpleITK and mlarray converters:

```bash
pip install mitk-workbench-remote[all]
```

`[all]` (SimpleITK + mlarray) installs on all supported Python versions. The MITK
interop and layout DSL require the `layout` (or `mitk`) extra and Python 3.12 — the
`mitk-python` wheels are 3.12-only. Install everything on 3.12 with
`pip install "mitk-workbench-remote[all,layout]"`.

## Connecting to a Workbench

```python
import mitk_workbench_remote as mw

wb = mw.connect("http://localhost:8080")

# Verify the connection
print(wb.ping())       # True
print(wb.info.name)    # "MITK Workbench REST API"
print(wb.info.mitk_version)
```

`connect()` does not make any HTTP calls — the connection is established
lazily on first use.  Use `wb.ping()` to check reachability.

## Showing Data

`wb.show()` is the primary one-liner for displaying data in the Workbench.
It accepts numpy arrays, `Image` objects, file paths, and any type handled
by the converter registry (SimpleITK, mlarray).

```python
import numpy as np

# Show a numpy array
arr = np.random.random((64, 64, 64)).astype(np.float32)
node = wb.show(arr, name="Random Volume")

# Show with visual properties
wb.show(arr, name="Red Volume", color=(1.0, 0.0, 0.0), opacity=0.5)

# Show a file from disk (no deserialization needed)
wb.show("scan.nrrd")
```

## Browsing the DataStorage

```python
# Iterate over all nodes
for node in wb.storage:
    print(f"{node.uid}  {node.name}  ({node.data_type})")

# Get a node by UID
node = wb.storage["node_1"]

# Filter by data type
images = wb.storage.list(data_type="Image")

# Count nodes
print(len(wb.storage))
```

In Jupyter notebooks, `wb.storage` renders as an interactive HTML table.

## Downloading and Uploading Data

```python
# Download image data
node = wb.storage["node_1"]
image = node.get_data()  # returns mw.Image
print(image.shape, image.spacing)

# Access the numpy array
arr = image.array

# Upload modified data
node.set_data(mw.Image(arr * 2, spacing=image.spacing))
```

For data types that cannot be loaded into Python (e.g. Surface, PointSet),
use `node.save_data("output.nrrd")` to save raw bytes to disk.

## Working with Segmentations

Multi-label segmentations are first-class objects with group and label
management.  See the
[MultiLabel Segmentation notebook](examples/03_multilabel_segmentation.ipynb)
for a full walkthrough.

```python
seg = node.get_data()  # returns MultiLabelSegmentation
for group in seg.groups:
    for label in seg.get_group_labels(group):
        print(f"{label.value}: {label.name}")
```

## Configuring Logging

`mitk-workbench-remote` uses Python's standard `logging` module.  By default
all output is suppressed.  Enable it to see HTTP requests, transfer mode
negotiation, and other internals:

```python
import logging

# Enable all debug output
logging.basicConfig(level=logging.DEBUG)

# Or target just the transport layer
logging.getLogger("mitk_workbench_remote.transport").setLevel(logging.DEBUG)
```

## Next Steps

- [Example notebooks](examples/index.rst) — hands-on workflows
- [API Reference](api/index.rst) — full public API documentation
