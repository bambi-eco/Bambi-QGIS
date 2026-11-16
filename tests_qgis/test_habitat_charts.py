# -*- coding: utf-8 -*-
"""The charts of the Habitat Context window (bambi_habitat_dialog.py):
every one says what its axes are of.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.PyQt.QtGui import QImage, QPainter

from bambi_wildlife_detection import bambi_habitat_dialog as module

LAYER = {"title": "Tree cover", "unit": "%", "description": "",
         "bin_edges": [0, 20, 40, 60, 80, 100],
         "available_counts": [5, 10, 30, 40, 15],
         "used_counts": [0, 2, 3, 9, 6],
         "available": {"mean": 58.2}, "used": {"mean": 71.0, "n": 20}}
CLASSES = [
    {"name": "Tree cover", "available_share": 0.6, "used_share": 0.8,
     "n_tracks": 16, "colour": [0, 100, 0]},
    {"name": "Grassland", "available_share": 0.4, "used_share": 0.2,
     "n_tracks": 4, "colour": [255, 255, 76]}]


def drawn(widget, monkeypatch):
    """Every text the widget draws."""
    said = []
    image = QImage(760, widget.minimumHeight(), QImage.Format.Format_RGB32)

    class Words(QPainter):
        """A painter that notes the text it is asked to draw - on a
        picture, since a widget is only painted on from its own event."""

        def __init__(self, _device):
            super().__init__(image)

        def drawText(self, *args):
            said.append(args[-1])
            super().drawText(*args)

    widget.resize(760, widget.minimumHeight())
    monkeypatch.setattr(module, "QPainter", Words)
    widget.paintEvent(None)
    return said


def test_a_histogram_names_its_quantity_with_the_unit(monkeypatch):
    chart = module.HistogramChart(LAYER)

    assert chart.axis_labels() == ("Tree cover (%)", "Share")
    said = drawn(chart, monkeypatch)
    assert "Tree cover (%)" in said and "Share" in said


def test_a_quantity_without_a_unit_is_named_as_it_is(monkeypatch):
    chart = module.HistogramChart(dict(LAYER, title="NDVI", unit=""))

    assert chart.axis_labels() == ("NDVI", "Share")
    assert "NDVI" in drawn(chart, monkeypatch)


def test_a_layer_without_data_still_has_a_name():
    chart = module.HistogramChart({"bin_edges": []})

    assert chart.axis_labels() == ("Value", "Share")


def test_the_land_cover_bars_have_a_scale_and_say_what_of(monkeypatch):
    chart = module.LandCoverChart(CLASSES)

    assert chart.axis_labels() == ("Share of ground / of animals (%)",
                                   "Land-cover class")
    said = drawn(chart, monkeypatch)
    assert "Share of ground / of animals (%)" in said
    assert "Land-cover class" in said
    # the scale runs from nought to the longest bar: 80 %
    assert {"0", "20", "40", "60", "80"} <= set(said)


def test_there_is_room_for_the_labels():
    assert module.HistogramChart(LAYER).minimumHeight() >= 200
    assert module.LandCoverChart(CLASSES).minimumHeight() == (
        module.LandCoverChart.HEAD + module.LandCoverChart.FOOT + 2 * 30)


@pytest.mark.parametrize("chart", [
    lambda: module.HistogramChart(LAYER),
    lambda: module.LandCoverChart(CLASSES)])
def test_a_chart_draws(chart):
    widget = chart()
    widget.resize(760, widget.minimumHeight())

    assert not widget.grab().isNull()
    widget.deleteLater()
