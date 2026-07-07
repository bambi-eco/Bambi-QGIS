# -*- coding: utf-8 -*-
"""Minimal ``qgis`` package stub for running plugin code without QGIS.

The plugin modules only need ``qgis.PyQt.QtCore`` (Qt base classes) at import
time; all heavier QGIS APIs are imported lazily inside functions. Installing
this stub into ``sys.modules`` before any plugin import makes the pure-logic
code paths usable in any plain Python environment.
"""
import sys
import types


def install_qgis_stubs():
    """Install the stub unless real QGIS (or a previous stub) is available."""
    try:
        import qgis.PyQt.QtCore  # noqa: F401
        return
    except ImportError:
        pass

    class _BoundSignal:
        """No-op stand-in for a connected/bound Qt signal."""

        def connect(self, *args, **kwargs):
            pass

        def disconnect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    def pyqtSignal(*args, **kwargs):  # noqa: N802 - mimics PyQt naming
        return _BoundSignal()

    def pyqtSlot(*args, **kwargs):  # noqa: N802 - mimics PyQt naming
        def decorator(func):
            return func
        return decorator

    class QObject:
        def __init__(self, parent=None):
            self._parent = parent

    class QThread(QObject):
        def start(self, *args, **kwargs):
            pass

        def quit(self):
            pass

        def wait(self, *args, **kwargs):
            return True

        def isRunning(self):  # noqa: N802 - mimics Qt naming
            return False

        def run(self):
            pass

    qgis_pkg = types.ModuleType("qgis")
    qgis_pkg.__path__ = []  # mark as package
    pyqt = types.ModuleType("qgis.PyQt")
    pyqt.__path__ = []
    qtcore = types.ModuleType("qgis.PyQt.QtCore")
    qtcore.QObject = QObject
    qtcore.QThread = QThread
    qtcore.pyqtSignal = pyqtSignal
    qtcore.pyqtSlot = pyqtSlot
    # ``qgis.core`` is imported lazily inside try/except blocks in the plugin;
    # an empty module makes those imports fail cleanly (attribute import
    # error) so the code takes its documented non-QGIS fallback path.
    qgis_core = types.ModuleType("qgis.core")

    qgis_pkg.PyQt = pyqt
    pyqt.QtCore = qtcore
    qgis_pkg.core = qgis_core

    sys.modules["qgis"] = qgis_pkg
    sys.modules["qgis.PyQt"] = pyqt
    sys.modules["qgis.PyQt.QtCore"] = qtcore
    sys.modules["qgis.core"] = qgis_core
