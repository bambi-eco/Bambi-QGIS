# -*- coding: utf-8 -*-
"""Qt names for the viewer, from QGIS' PyQt when running inside QGIS.

``QOpenGLWidget`` lives in ``QtWidgets`` on Qt5 and in ``QtOpenGLWidgets`` on
Qt6. Outside QGIS (headless checks) the bindings are taken from pyqtgraph,
which picks whichever PyQt is installed.
"""

try:
    from qgis.PyQt import QtCore, QtGui, QtWidgets  # noqa: F401
except ImportError:                                  # pragma: no cover
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: F401

QOpenGLWidget = getattr(QtWidgets, "QOpenGLWidget", None)
if QOpenGLWidget is None:                             # Qt6
    try:
        from qgis.PyQt.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401
    except ImportError:
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401

Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal")
