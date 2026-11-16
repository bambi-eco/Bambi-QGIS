# -*- coding: utf-8 -*-
"""The plugin's windows and widgets take real Qt 6 events.

Up to 9.0 three tools read the mouse position with ``event.x()`` and
``event.y()``, which Qt 6 does not have: the Thermal Viewer failed at the
first mouse move over an image, the Correction Wizard at the first click and
the inspector tools at the first click on the map.
"""
import importlib

import numpy as np
import pytest

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget

from .conftest import Anything

PACKAGE = "bambi_wildlife_detection"


@pytest.fixture(autouse=True)
def failures(monkeypatch):
    """What goes wrong inside an event handler. PyQt6 ends the process for
    an exception there unless somebody else looks at it - this does, and
    the test fails with it."""
    import sys

    seen = []
    monkeypatch.setattr(
        sys, "excepthook",
        lambda kind, value, trace: seen.append(f"{kind.__name__}: {value}"))
    yield seen
    assert seen == []


def _class(module, name):
    return getattr(importlib.import_module(f"{PACKAGE}.{module}"), name)


# ---------------------------------------------------------------------------
# The three tools that read the mouse the Qt 5 way
# ---------------------------------------------------------------------------

def test_the_thermal_viewer_follows_the_mouse(shown):
    label = _class("bambi_thermal_viewer", "_ThermalLabel")()
    temperatures = np.arange(120, dtype=np.float32).reshape(10, 12)
    label.set_temperature_array(temperatures)
    under = []
    label.set_hover_callback(lambda *found: under.append(found))
    label.setPixmap(QPixmap(120, 100))
    shown(label, 400, 300)

    centre = QPoint(label.width() // 2, label.height() // 2)
    QTest.mouseMove(label, centre)
    QTest.mouseMove(label, QPoint(2, 2))            # beside the image

    assert len(under) == 2
    row, column, value = under[0]
    assert (row, column) == (5, 6) and value == temperatures[5, 6]
    assert under[1] == (None, None, None)


def test_the_correction_wizard_takes_a_click_on_the_image(shown):
    label = _class("bambi_correction_wizard", "ClickableImageLabel")()
    label.set_image(np.full((100, 200, 3), 90, dtype=np.uint8))
    placed = []
    label.pointSet.connect(lambda x, y: placed.append((x, y)))
    shown(label, 400, 300)

    QTest.mouseMove(label, QPoint(200, 150))
    QTest.mouseClick(label, Qt.MouseButton.LeftButton,
                     pos=QPoint(label.width() // 2, label.height() // 2))

    assert len(placed) == 1
    assert placed[0] == pytest.approx((0.5, 0.5), abs=0.02)


def test_the_correction_wizards_plot_takes_the_mouse(shown):
    plot = shown(_class("bambi_correction_wizard", "CirclePlotWidget")(),
                 300, 300)

    QTest.mouseMove(plot, QPoint(100, 120))
    QTest.mousePress(plot, Qt.MouseButton.LeftButton, pos=QPoint(100, 120))
    QTest.mouseMove(plot, QPoint(140, 160))
    QTest.mouseRelease(plot, Qt.MouseButton.LeftButton, pos=QPoint(140, 160))


def test_the_inspector_tools_take_a_click_on_the_map(app, monkeypatch):
    tool_class = _class("bambi_click_tool", "BambiClickTool")
    monkeypatch.setattr(tool_class, "_get_bambi_layers",
                        lambda self: ["a layer"])
    asked = []
    monkeypatch.setattr(
        tool_class, "identify",
        lambda self, x, y, layers, mode: asked.append((x, y)) or [],
        raising=False)
    tool = tool_class(Anything())

    tool.canvasReleaseEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, QPointF(31.0, 47.0),
        QPointF(31.0, 47.0), Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))

    assert asked == [(31, 47)]


def test_where_the_mouse_is_whichever_qt(app):
    from bambi_wildlife_detection.gui_utils import mouse_xy

    event = QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(12.4, 7.6), QPointF(12.4, 7.6),
        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier)
    assert not hasattr(event, "x")                  # what Qt 6 took away
    assert mouse_xy(event) == (12, 8)

    class Qt5Event:
        def pos(self):
            return QPoint(3, 4)

    assert mouse_xy(Qt5Event()) == (3, 4)


# ---------------------------------------------------------------------------
# Widgets with something in them
# ---------------------------------------------------------------------------

def test_the_calibration_image_takes_a_point_with_the_magnifier_on(shown):
    label = _class("bambi_camera_calibration", "_AnnotImageLabel")()
    label.load_image(np.full((200, 200, 3), 120, dtype=np.uint8))
    label.set_accepting(True)
    label.set_magnifier_enabled(True)
    placed = []
    label.pointPlaced.connect(lambda x, y: placed.append((x, y)))
    shown(label, 600, 200)

    spot = QPoint(200 + 50, 120)                 # 200 px of border each side
    QTest.mouseMove(label, spot)
    label.repaint()                                 # draws the magnifier
    QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=spot)

    assert placed == [(50, 120)]


def test_the_curve_editor_takes_a_new_point_and_a_drag(shown):
    curve = shown(_class("bambi_curve_widget", "CurveWidget")(), 300, 200)
    before = len(curve.points())

    QTest.mouseClick(curve, Qt.MouseButton.LeftButton, pos=QPoint(150, 60))
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, pos=QPoint(150, 60))
    QTest.mouseMove(curve, QPoint(160, 90))
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, pos=QPoint(160, 90))
    curve.repaint()

    assert len(curve.points()) >= before


def test_the_altitude_profile_says_what_is_under_the_mouse(shown):
    canvas = _class("bambi_agl_tool", "ProfileCanvas")()
    canvas.set_points([
        {"frame": index, "distance_m": 10.0 * index, "time_s": float(index),
         "altitude": 500.0 + index, "ground": 420.0, "agl": 80.0 + index,
         "surface": None, "agl_surface": None}
        for index in range(20)])
    under = []
    canvas.set_hover_callback(under.append)
    shown(canvas, 600, 300)

    QTest.mouseMove(canvas, QPoint(300, 150))
    canvas.repaint()

    assert under and under[-1] is not None and "frame" in under[-1]


# ---------------------------------------------------------------------------
# Every window: built, shown, and poked
# ---------------------------------------------------------------------------

def _dataset_flights():
    """The list of flights that ships with the plugin."""
    from bambi_wildlife_detection.core import bambi_dataset

    return bambi_dataset.flights()


WINDOWS = [
    ("bambi_thermal_viewer", "ThermalViewerDialog", lambda c: c()),
    ("bambi_camera_calibration", "CameraCalibrationWizard", lambda c: c()),
    ("bambi_feature_viewer", "FeatureViewerDialog", lambda c: c()),
    ("bambi_dependency_manager", "DependencyManagerDialog", lambda c: c()),
    ("bambi_curve_widget", "CurveEditorDialog", lambda c: c()),
    ("bambi_flight_planner", "FlightPlannerDialog", lambda c: c(Anything())),
    ("bambi_video_creator", "VideoCreatorDialog", lambda c: c(Anything(), None)),
    ("bambi_labelling_tool", "LabellingToolDialog", lambda c: c(Anything(), None)),
    ("bambi_transect_tool", "TransectToolDialog", lambda c: c(Anything(), None)),
    ("bambi_segmentation_tool", "SegmentationToolDialog",
     lambda c: c(Anything(), None)),
    ("bambi_agl_tool", "AglToolDialog", lambda c: c(Anything(), None)),
    ("bambi_dataset_dialog", "DatasetFlightsDialog",
     lambda c: c(_dataset_flights(), folder="/data/flights")),
    ("bambi_dock_widget", "BambiDockWidget", lambda c: c(Anything())),
]


def _own_widgets(window):
    """The window's visible widgets of classes the plugin defines."""
    found = [window] if type(window).__module__.startswith(PACKAGE) else []
    for child in window.findChildren(QWidget):
        if type(child).__module__.startswith(PACKAGE) and child.isVisible():
            found.append(child)
    return found


def _poke(widget, app):
    """Move over it, click, drag, right-click, turn the wheel."""
    width, height = max(widget.width(), 4), max(widget.height(), 4)
    points = [QPoint(width // 2, height // 2), QPoint(width // 4, height // 3),
              QPoint(2, 2), QPoint(width - 3, height - 3)]
    for point in points:
        QTest.mouseMove(widget, point)
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=points[0])
    QTest.mouseMove(widget, points[1])
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=points[1])
    QTest.mouseClick(widget, Qt.MouseButton.RightButton, pos=points[0])
    QTest.mouseClick(widget, Qt.MouseButton.MiddleButton, pos=points[1])
    QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, pos=points[0])
    position = QPointF(points[0])
    app.sendEvent(widget, QWheelEvent(
        position, widget.mapToGlobal(position), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False))
    app.processEvents()


@pytest.mark.parametrize("module, name, make", WINDOWS,
                         ids=[name for _module, name, _make in WINDOWS])
def test_a_window_is_built_shown_and_used(module, name, make, app, shown,
                                          monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    # nothing may wait for an answer
    for box in ("information", "warning", "critical", "question", "about"):
        monkeypatch.setattr(QMessageBox, box,
                            lambda *args, **kwargs: QMessageBox.StandardButton.No)
    for dialog in ("getOpenFileName", "getSaveFileName", "getOpenFileNames"):
        monkeypatch.setattr(QFileDialog, dialog,
                            lambda *args, **kwargs: ("", ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: "")

    window = shown(make(_class(module, name)), 1100, 800)

    widgets = _own_widgets(window)
    for widget in widgets:
        _poke(widget, app)
    for key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_PageDown,
                Qt.Key.Key_Delete, Qt.Key.Key_M, Qt.Key.Key_Space):
        QTest.keyClick(window, key)
    app.processEvents()
    window.close()


# ---------------------------------------------------------------------------
# What a layer's fields are made with
# ---------------------------------------------------------------------------

def test_field_types_exist_without_qvariants_qt5_names(app):
    """Qt 6 has no ``QVariant.Int``; every layer the plugin makes was made
    with it."""
    from PyQt6.QtCore import QMetaType, QVariant

    from bambi_wildlife_detection.qgis_compat import FieldType

    assert not hasattr(QVariant, "Int")
    assert FieldType.Int == QMetaType.Type.Int
    assert FieldType.LongLong == QMetaType.Type.LongLong
    assert FieldType.Double == QMetaType.Type.Double
    assert FieldType.String == QMetaType.Type.QString
    assert FieldType.Bool == QMetaType.Type.Bool
