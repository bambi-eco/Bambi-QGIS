# -*- coding: utf-8 -*-
"""The inspector is one window for the whole plugin - and stays usable when
the window it was made in is gone (bambi_feature_viewer.py,
dock/analytics_tab.py).

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.PyQt.QtCore import QCoreApplication, QEvent
from qgis.PyQt.QtWidgets import QMainWindow

from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog


def _destroy(widget) -> None:
    """Delete *widget* - and with it every window it is the parent of -
    now, as Qt does when a main window goes."""
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def no_viewer(monkeypatch):
    """Start without an inspector, and leave none behind."""
    monkeypatch.setattr(FeatureViewerDialog, "_instance", None)
    yield
    viewer = FeatureViewerDialog._instance
    if viewer is not None:
        _destroy(viewer)


def test_it_is_one_window(no_viewer):
    window = QMainWindow()

    first = FeatureViewerDialog.get_instance(window)

    assert FeatureViewerDialog.get_instance(window) is first
    assert FeatureViewerDialog.get_instance() is first


def test_an_inspector_whose_window_is_gone_is_made_anew(no_viewer):
    """The inspector is a child of the window it was first asked for in.
    When that window goes, Qt deletes the inspector with it - and the next
    one to ask used to be handed the dead one ("wrapped C/C++ object of
    type FeatureViewerDialog has been deleted")."""
    gone = QMainWindow()
    first = FeatureViewerDialog.get_instance(gone)
    _destroy(gone)

    window = QMainWindow()
    second = FeatureViewerDialog.get_instance(window)

    assert second is not first
    assert second.windowTitle() == "BAMBI Feature Viewer"      # it is alive
    assert FeatureViewerDialog.get_instance(window) is second


def test_the_dock_follows_the_inspector_that_is_there(dock, no_viewer,
                                                      monkeypatch):
    """What is approved or deleted in the inspector reaches the dock - also
    in an inspector made after the first one went."""
    told = []
    monkeypatch.setattr(
        dock, "_on_viewer_track_approved",
        lambda folder, modality, track, approved: told.append(track))
    gone = QMainWindow()
    dock.iface.mainWindow.return_value = gone
    dock._connect_viewer_signals()
    FeatureViewerDialog.get_instance().trackApproved.emit("/f", "t", 1, True)
    _destroy(gone)

    window = QMainWindow()
    dock.iface.mainWindow.return_value = window
    dock._connect_viewer_signals()
    dock._connect_viewer_signals()                 # asked twice, told once
    FeatureViewerDialog.get_instance().trackApproved.emit("/f", "t", 2, True)

    assert told == [1, 2]
