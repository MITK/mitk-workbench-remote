Converters
==========

.. automodule:: mitk_workbench_remote.converters
   :members:
   :show-inheritance:

Interop with the ``mitk`` package
----------------------------------

``MitkImageConverter`` is auto-registered whenever the ``mitk`` package is
importable.  It enables transparent use of ``mitk.Image`` objects with
``DataNode.get_data()`` / ``DataNode.set_data()`` and ``Image.to_mitk()``.
See :doc:`../interop-with-mitk` for the full guide.

.. autoclass:: mitk_workbench_remote.converters._mitk.MitkImageConverter
   :members:
