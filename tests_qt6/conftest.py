# -*- coding: utf-8 -*-
"""The plugin's widgets under Qt 6, without QGIS.

QGIS 4 runs on Qt 6. What Qt 6 no longer has is mostly found by the checker
that plugins.qgis.org applies (``run_qt6_check.sh``) - enums, moved classes.
It does not find calls that only fail when they happen: ``event.x()`` in a
mouse handler is an ``AttributeError`` at the first mouse move.

These tests import the plugin's GUI modules with ``qgis.PyQt`` standing for
the real PyQt6 and send its widgets real events. ``qgis.core`` / ``qgis.gui``
are stand-ins: what needs a running QGIS is not covered here, and has its
tests in ``tests_qgis`` (on Qt 5, until there is a QGIS 4 to run them in).

    pip install PyQt6 PyQt6-QScintilla
    QT_QPA_PLATFORM=offscreen python -m pytest tests_qt6 -p no:cacheprovider

Skipped where PyQt6 is not installed.
"""
import importlib
import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PyQt6 = pytest.importorskip("PyQt6")
pytest.importorskip("PyQt6.QtWidgets")

_QT_MODULES = ("QtCore", "QtGui", "QtWidgets", "QtTest", "QtNetwork",
               "QtOpenGLWidgets", "QtSvg", "QtXml", "sip")


class _Anything(type):
    """A class any attribute of which is such a class again: what a
    ``qgis.core`` name is here."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stand_in(name)


def _stand_in(name):
    return _Anything(name, (), {
        "__init__": lambda self, *args, **kwargs: None,
        "__getattr__": lambda self, attribute: _stand_in(attribute)(),
        "__call__": lambda self, *args, **kwargs: None,
    })


class _Signal:
    def connect(self, *args, **kwargs):
        pass

    def disconnect(self, *args, **kwargs):
        pass

    def emit(self, *args, **kwargs):
        pass


class _Crs:
    def isValid(self):  # noqa: N802 - QGIS' name
        return False

    def authid(self):
        return ""


class _Tree:
    def children(self):
        return []

    def findGroup(self, name):  # noqa: N802
        return None

    def findLayer(self, layer_id):  # noqa: N802
        return None


class _Project:
    """As much of ``QgsProject`` as building the plugin's windows asks for:
    an empty project that keeps what is written to it."""

    _instance = None

    def __init__(self):
        self._entries = {}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[:1].islower() and (name.endswith("ed") or name.endswith("Changed")
                                   or name in ("readProject", "writeProject",
                                               "cleared", "layersAdded")):
            return _Signal()
        return lambda *args, **kwargs: None

    def readEntry(self, scope, key, default=""):  # noqa: N802
        value = self._entries.get((scope, key))
        return (default, False) if value is None else (str(value), True)

    def readNumEntry(self, scope, key, default=0):  # noqa: N802
        value = self._entries.get((scope, key))
        return (default, False) if value is None else (int(value), True)

    def readDoubleEntry(self, scope, key, default=0.0):  # noqa: N802
        value = self._entries.get((scope, key))
        return (default, False) if value is None else (float(value), True)

    def readBoolEntry(self, scope, key, default=False):  # noqa: N802
        value = self._entries.get((scope, key))
        return (default, False) if value is None else (bool(value), True)

    def writeEntry(self, scope, key, value):  # noqa: N802
        self._entries[(scope, key)] = value
        return True

    def removeEntry(self, scope, key):  # noqa: N802
        self._entries.pop((scope, key), None)
        return True

    def entryList(self, scope, key):  # noqa: N802
        return []

    def mapLayers(self):  # noqa: N802
        return {}

    def mapLayersByName(self, name):  # noqa: N802
        return []

    def crs(self):
        return _Crs()

    def layerTreeRoot(self):  # noqa: N802
        return _Tree()

    def homePath(self):  # noqa: N802
        return ""

    def fileName(self):  # noqa: N802
        return ""

    def absoluteFilePath(self):  # noqa: N802
        return ""


class Anything:
    """An ``iface``: whatever is asked of it is there and does nothing."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Anything()

    def __call__(self, *args, **kwargs):
        return Anything()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _Module(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _stand_in(name)
        setattr(self, name, value)
        return value


def _install():
    if "qgis" in sys.modules and not getattr(sys.modules["qgis"], "_qt6_shim", False):
        raise pytest.UsageError(
            "tests_qt6 must run in a process of its own: a real or stubbed "
            "qgis is already imported.")
    qgis = types.ModuleType("qgis")
    qgis._qt6_shim = True
    qgis.__path__ = []
    pyqt = types.ModuleType("qgis.PyQt")
    pyqt.__path__ = []
    sys.modules.update({"qgis": qgis, "qgis.PyQt": pyqt})
    qgis.PyQt = pyqt
    for name in _QT_MODULES:
        try:
            module = importlib.import_module(f"PyQt6.{name}")
        except ImportError:
            continue
        sys.modules[f"qgis.PyQt.{name}"] = module
        setattr(pyqt, name, module)
    # as QGIS' own qgis.PyQt does for Qt 6: classes that moved to QtGui
    from PyQt6 import QtGui, QtWidgets
    for moved in ("QAction", "QActionGroup", "QShortcut"):
        if not hasattr(QtWidgets, moved):
            setattr(QtWidgets, moved, getattr(QtGui, moved))
    for name in ("core", "gui", "utils", "analysis", "processing"):
        module = _Module(f"qgis.{name}")
        sys.modules[f"qgis.{name}"] = module
        setattr(qgis, name, module)
    sys.modules["qgis.core"].QgsProject = _Project


_install()


@pytest.fixture(scope="session")
def app():
    from PyQt6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def shown(app):
    """Show a widget for the test; it is deleted afterwards."""
    widgets = []

    def show(widget, width=400, height=300):
        widgets.append(widget)
        widget.resize(width, height)
        widget.show()
        app.processEvents()
        return widget

    yield show
    for widget in widgets:
        widget.hide()
        widget.deleteLater()
    app.processEvents()
