# -*- coding: utf-8 -*-
"""Headless business logic of the BAMBI plugin.

Modules in this package must be importable without a QGIS installation:

* Allowed module-level imports: stdlib, numpy/scipy/pyproj and
  ``qgis.PyQt.QtCore`` (covered by the test stub).
* Never import ``qgis.PyQt.QtWidgets``, ``qgis.PyQt.QtGui``, ``qgis.core``
  or ``qgis.gui`` at module level.
* Heavy runtime dependencies (alfspy, cv2, rasterio, bambi, trimesh, …)
  are imported lazily inside functions.

The boundary is enforced by ``tests/test_architecture.py``.
See ``DECOUPLING_PLAN.md`` for the roadmap.
"""
