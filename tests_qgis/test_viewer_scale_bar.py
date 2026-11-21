# -*- coding: utf-8 -*-
"""The scale bar over the 3D Flight Viewer's view (viewer3d/scale_bar.py).

The lengths themselves are covered without QGIS in
tests/test_view_camera.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import numpy as np
import pytest

from bambi_wildlife_detection.viewer3d.scale_bar import WIDEST, ScaleBar


@pytest.fixture
def bar():
    made = ScaleBar()
    yield made
    made.deleteLater()


def test_it_says_a_round_length_and_is_as_wide_as_that_is(bar):
    bar.set_scale(0.3)

    assert bar.length() == pytest.approx(20.0)
    assert bar.text() == "20 m"
    assert bar.pixels() == pytest.approx(20.0 / 0.3)
    assert bar.pixels() <= WIDEST


def test_it_moves_on_to_the_next_length_as_the_view_zooms(bar):
    said = []
    per_pixel = 40.0
    while per_pixel > 0.01:
        bar.set_scale(per_pixel)
        assert 0.4 * WIDEST - 1e-6 <= bar.pixels() <= WIDEST + 1e-6
        if not said or said[-1] != bar.text():
            said.append(bar.text())
        per_pixel *= 0.85

    assert said == ["5 km", "2 km", "1 km", "500 m", "200 m", "100 m",
                    "50 m", "20 m", "10 m", "5 m", "2 m", "1 m"]


def test_without_a_scale_it_shows_nothing(bar):
    from qgis.PyQt.QtGui import QColor, QImage

    assert (bar.length(), bar.pixels(), bar.text()) == (0.0, 0.0, "")

    image = QImage(bar.size(), QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 255))
    bar.render(image)
    assert image.pixelColor(bar.width() // 2, 20) == QColor(0, 0, 255)

    bar.set_scale(1.0)
    bar.render(image)
    assert image.pixelColor(bar.width() // 2, 20) != QColor(0, 0, 255)


# ---------------------------------------------------------------------------
# Over the view
# ---------------------------------------------------------------------------

@pytest.fixture
def viewport():
    pytest.importorskip("OpenGL")
    from bambi_wildlife_detection.viewer3d.viewport import FlightViewport

    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QResizeEvent

    made = FlightViewport()
    size = made.size()
    made.resize(640, 420)
    # not shown, so told of its size by hand
    made.resizeEvent(QResizeEvent(QSize(640, 420), size))
    yield made
    made.deleteLater()


def flat_ground(origin, direction):
    """Where a ray meets the plane z = 0."""
    if direction[2] >= 0.0:
        return None
    return origin - direction * (origin[2] / direction[2])


def test_it_sits_at_the_bottom_in_the_middle(viewport):
    from bambi_wildlife_detection.viewer3d.compass import MARGIN

    bar = viewport.scale_bar
    assert bar.parent() is viewport
    assert bar.x() == (640 - bar.width()) // 2
    assert bar.y() == 420 - bar.height() - MARGIN


def test_it_measures_the_terrain_in_the_middle_of_the_view(viewport):
    orbit = viewport.orbit
    orbit.target = np.array([0.0, 0.0, 0.0])
    orbit.distance, orbit.elevation, orbit.fovy = 420.0, 89.0, 90.0

    # nothing to hit: what the view turns about
    assert viewport.view_distance() == pytest.approx(420.0)
    assert viewport.metres_per_pixel() == pytest.approx(2.0)

    # what the view turns about hangs 100 m over the ground
    viewport.set_pick_function(flat_ground)
    orbit.target = np.array([0.0, 0.0, 100.0])

    assert viewport.view_distance() == pytest.approx(
        420.0 + 100.0 / np.sin(np.radians(89.0)))


def test_zooming_in_shortens_what_the_bar_stands_for(viewport):
    viewport.set_pick_function(flat_ground)
    viewport.scale_bar.set_scale(viewport.metres_per_pixel())
    far = viewport.scale_bar.length()

    viewport.orbit.zoom(0.1)
    viewport.scale_bar.set_scale(viewport.metres_per_pixel())

    assert 0.0 < viewport.scale_bar.length() < far


def test_a_saved_view_has_the_scale_in_it(viewport, monkeypatch):
    from qgis.PyQt.QtGui import QColor, QImage

    def scene():
        image = QImage(640, 420, QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 255))
        return image

    monkeypatch.setattr(viewport, "grabFramebuffer", scene)
    centre = viewport.scale_bar.geometry().center()

    saved = viewport.grab_view()
    assert saved.pixelColor(centre.x(), centre.y()) != QColor(0, 0, 255)
    assert viewport.scale_bar.text() != ""

    viewport.set_scale_visible(False)
    plain = viewport.grab_view()
    assert plain.pixelColor(centre.x(), centre.y()) == QColor(0, 0, 255)
