# -*- coding: utf-8 -*-
"""Headless business logic of the BAMBI plugin.

Modules in this package must be importable without a QGIS installation:

* Allowed module-level imports: stdlib, numpy/scipy/pyproj and
  ``qgis.PyQt.QtCore`` (covered by the test stub).
* Never import ``qgis.PyQt.QtWidgets``, ``qgis.PyQt.QtGui``, ``qgis.core``
  or ``qgis.gui`` at module level.
* Heavy runtime dependencies (alfspy, cv2, rasterio, bambi, trimesh, …)
  are imported lazily inside functions.

The boundary is enforced by ``tests/test_architecture.py``. It is what lets
the logic be tested without QGIS, and run outside it: the dock widget and the
dialogs (``bambi_*.py``) collect settings and show results; everything that
computes lives here.
"""
