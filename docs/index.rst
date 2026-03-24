mitk-workbench-remote
=====================

**Remote control for MITK Workbench from Python.**

``mitk-workbench-remote`` is a Python library for controlling running
`MITK Workbench <https://www.mitk.org>`_ instances programmatically via their
REST API.  It is designed for researchers and developers who want to drive
the Workbench from Python scripts, Jupyter notebooks, and ML pipelines --
loading images, editing segmentations, and managing data nodes without
touching the GUI.

Quick example::

   import mitk_workbench_remote as mw

   wb = mw.connect(port=8080)
   wb.show("scan.nrrd", opacity=0.8)
   wb.show(my_array, name="Prediction", color=(1.0, 0.5, 0.0))

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   api/index
   examples/index
   changelog
