# -*- coding: utf-8 -*-
"""The compass over the 3D Flight Viewer's view (viewer3d/compass.py).

The bearing itself is covered without QGIS in tests/test_view_camera.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.PyQt.QtCore import Qt

from bambi_wildlife_detection.viewer3d.compass import Compass


@pytest.fixture
def compass():
    made = Compass()
    yield made
    made.deleteLater()


def test_it_says_which_way_the_view_looks(compass):
    assert compass.text() == "N 0°"

    compass.set_bearing(47.2)

    assert compass.bearing() == pytest.approx(47.2)
    assert compass.text() == "NE 47°"
    compass.set_bearing(-90.0)
    assert compass.text() == "W 270°"


def test_north_is_where_north_is_on_the_screen(compass):
    # looking north: north is up
    assert compass.north_on_screen() == pytest.approx((0.0, -1.0))
    # looking east: north is to the left
    compass.set_bearing(90.0)
    assert compass.north_on_screen() == pytest.approx((-1.0, 0.0))
    # looking south: north is behind, at the bottom
    compass.set_bearing(180.0)
    assert compass.north_on_screen() == pytest.approx((0.0, 1.0))


def test_the_needle_is_drawn_where_north_is(compass):
    def reddest(image):
        """The middle of the needle's red half - inside the ring of
        letters."""
        middle = image.width() // 2
        red = []
        for x in range(middle - 24, middle + 25):
            for y in range(middle - 24, middle + 25):
                colour = image.pixelColor(x, y)
                if colour.red() - colour.green() - colour.blue() > 60:
                    red.append((x, y))
        assert red
        return (sum(x for x, _y in red) / len(red),
                sum(y for _x, y in red) / len(red))

    compass.set_bearing(0.0)
    x, y = reddest(compass.grab().toImage())
    assert y < compass.width() / 2 - 4           # above the middle

    compass.set_bearing(90.0)
    x, y = reddest(compass.grab().toImage())
    assert x < compass.width() / 2 - 4           # left of the middle


def test_it_lets_the_mouse_through_to_the_view(compass):
    assert compass.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_the_viewer_can_hide_it(iface):
    from bambi_wildlife_detection.bambi_flight_viewer import (
        FlightViewerDialog)

    dialog = FlightViewerDialog(iface)
    try:
        assert dialog.compass_check.text() == "Show compass"
        assert dialog.compass_check.isChecked()
        assert "compass_check" in dialog._SAVED
        if dialog.viewport is not None:
            assert not dialog.viewport.compass.isHidden()
            dialog.compass_check.setChecked(False)
            assert dialog.viewport.compass.isHidden()
            dialog.compass_check.setChecked(True)
            assert not dialog.viewport.compass.isHidden()
    finally:
        dialog.compass_check.setChecked(True)
        dialog.close()
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# Over the view
# ---------------------------------------------------------------------------

@pytest.fixture
def viewport():
    pytest.importorskip("OpenGL")
    from bambi_wildlife_detection.viewer3d.viewport import FlightViewport

    made = FlightViewport()
    made.resize(640, 420)
    yield made
    made.deleteLater()


def test_it_sits_in_the_top_right_corner_and_stays_there(viewport):
    from bambi_wildlife_detection.viewer3d.compass import MARGIN
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QResizeEvent

    compass = viewport.compass
    assert compass.parent() is viewport
    assert (compass.x(), compass.y()) == (
        640 - compass.width() - MARGIN, MARGIN)

    viewport.resize(900, 500)
    viewport.resizeEvent(QResizeEvent(QSize(900, 500), QSize(640, 420)))

    assert compass.x() == 900 - compass.width() - MARGIN


def test_it_follows_the_view(viewport):
    # as the viewer opens: the eye south of the scene, looking north
    assert viewport.bearing() == pytest.approx(0.0, abs=1e-6)

    viewport.orbit.orbit(90.0, 0.0)

    assert viewport.bearing() == pytest.approx(270.0)


def test_a_saved_view_has_the_compass_in_it(viewport, monkeypatch):
    from qgis.PyQt.QtGui import QColor, QImage

    def scene():
        image = QImage(640, 420, QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 255))
        return image

    monkeypatch.setattr(viewport, "grabFramebuffer", scene)
    viewport.orbit.orbit(90.0, 0.0)

    saved = viewport.grab_view()

    centre = viewport.compass.geometry().center()
    assert saved.pixelColor(20, 400) == QColor(0, 0, 255)       # the scene
    assert saved.pixelColor(centre.x() - 12, centre.y() - 12) != QColor(
        0, 0, 255)                                              # the rose
    assert viewport.compass.text() == "W 270\u00b0"

    viewport.set_compass_visible(False)
    plain = viewport.grab_view()
    assert plain.pixelColor(centre.x() - 12, centre.y() - 12) == QColor(
        0, 0, 255)
