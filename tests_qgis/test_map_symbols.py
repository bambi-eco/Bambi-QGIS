# -*- coding: utf-8 -*-
"""Animal symbology inside a real QGIS: boxes by species up close, pins with
a sex / age indicator when zoomed out, colours from the Project Schema.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.core import (
    QgsCentroidFillSymbolLayer, QgsFeature, QgsFontMarkerSymbolLayer,
    QgsGeometry, QgsMapRendererParallelJob, QgsMapSettings, QgsProject,
    QgsRectangle, QgsRuleBasedRenderer, QgsSimpleFillSymbolLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QColorDialog, QDialog

from bambi_wildlife_detection import bambi_map_symbols as symbols
from bambi_wildlife_detection.core import map_symbols

ANIMALS = [("roe deer", "female", "juvenile"), ("red deer", "male", "adult"),
           ("hare", "", ""), ("", "", "")]


def _layer(animals=ANIMALS):
    layer = QgsVectorLayer("Polygon?crs=EPSG:32633", "boxes", "memory")
    layer.dataProvider().addAttributes(symbols.animal_fields())
    layer.updateFields()
    features = []
    for index, (species, sex, age) in enumerate(animals):
        x = 447000.0 + index * 10.0
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromRect(
            QgsRectangle(x, 5350500.0, x + 1.5, 5350500.8)))
        feature.setAttributes(symbols.animal_values(
            map_symbols.attributes(species, sex, age)))
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()
    return layer


def _rules(layer):
    renderer = layer.renderer()
    assert isinstance(renderer, QgsRuleBasedRenderer)
    return renderer.rootRule().children()


def test_boxes_up_close_pins_beyond_the_switch_scale():
    layer = _layer()
    symbols.style_boxes(layer, {}, 500)
    boxes, pins = _rules(layer)
    assert boxes.label().startswith("Boxes") and "1:500" in boxes.label()
    assert pins.label().startswith("Pins")
    assert boxes.isScaleOK(100) and boxes.isScaleOK(500)
    assert not boxes.isScaleOK(501)
    assert pins.isScaleOK(501) and pins.isScaleOK(50000)
    assert not pins.isScaleOK(499)


def test_one_rule_per_species_present_and_one_for_the_rest():
    layer = _layer()
    symbols.style_boxes(layer, {}, 500)
    for group in _rules(layer):
        labels = [rule.label() for rule in group.children()]
        assert labels == ["hare", "red deer", "roe deer", "undetermined"]
        assert group.children()[-1].isElse()
        assert group.children()[0].filterExpression() == "\"species\" = 'hare'"


def test_the_colour_is_the_species(tmp_path):
    layer = _layer()
    symbols.style_boxes(layer, {"hare": "#010203"}, 500)
    boxes, pins = _rules(layer)
    hare_box = boxes.children()[0].symbol().symbolLayer(0)
    assert isinstance(hare_box, QgsSimpleFillSymbolLayer)
    assert hare_box.strokeColor().name() == "#010203"
    deer_box = boxes.children()[1].symbol().symbolLayer(0)
    assert deer_box.strokeColor().name() == \
        map_symbols.SEEDED_COLOURS["red deer"]
    hare_pin = pins.children()[0].symbol().symbolLayer(0)
    assert isinstance(hare_pin, QgsCentroidFillSymbolLayer)
    head = hare_pin.subSymbol().symbolLayer(1)
    assert head.color().name() == "#010203"


def test_the_pin_carries_the_indicator_field():
    layer = _layer()
    symbols.style_boxes(layer, {}, 500)
    _boxes, pins = _rules(layer)
    marker = pins.children()[0].symbol().symbolLayer(0).subSymbol()
    badges = [marker.symbolLayer(i) for i in range(marker.symbolLayerCount())
              if isinstance(marker.symbolLayer(i), QgsFontMarkerSymbolLayer)]
    assert len(badges) == 2          # halo and text
    for badge in badges:
        properties = badge.dataDefinedProperties()
        assert properties.property(
            symbols._property("PropertyCharacter")).field() == "indicator"
        assert badge.fontFamily() != "Dingbats"
    values = {f["species"]: f["indicator"] for f in layer.getFeatures()}
    assert values["roe deer"] == "♀ juv" and values["red deer"] == "♂ ad"
    assert values["hare"] == ""


def test_pins_can_be_switched_off():
    layer = _layer()
    symbols.style_boxes(layer, {}, 500, pins=False)
    rules = _rules(layer)
    assert [rule.label() for rule in rules] == [
        "hare", "red deer", "roe deer", "undetermined"]
    assert all(rule.isScaleOK(1_000_000) for rule in rules)


def test_a_quote_in_a_species_name_does_not_break_the_filter():
    layer = _layer([("grant's gazelle", "", "")])
    symbols.style_boxes(layer, {}, 500)
    rule = _rules(layer)[0].children()[0]
    assert rule.filterExpression() == "\"species\" = 'grant''s gazelle'"
    assert rule.filter().isValid() if rule.filter() else True


def _render(layer, scale_factor):
    extent = layer.extent()
    extent.scale(scale_factor)
    settings = QgsMapSettings()
    settings.setLayers([layer])
    settings.setExtent(extent)
    settings.setOutputSize(QSize(400, 300))
    settings.setDestinationCrs(layer.crs())
    settings.setBackgroundColor(QColor(255, 255, 255))
    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()
    return settings.scale(), job.renderedImage()


def _has_colour(image, colour):
    wanted = QColor(colour)
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if abs(pixel.red() - wanted.red()) < 12 and \
                    abs(pixel.green() - wanted.green()) < 12 and \
                    abs(pixel.blue() - wanted.blue()) < 12:
                return True
    return False


def test_it_really_draws_at_both_scales():
    """Rendered through QGIS, not just configured: the species colour is on
    the map zoomed in (a box) and zoomed far out (a pin), where the box
    itself would be less than a pixel."""
    layer = _layer([("red deer", "male", "adult")])
    symbols.style_boxes(layer, {}, 500)
    colour = map_symbols.SEEDED_COLOURS["red deer"]
    near_scale, near = _render(layer, 3.0)
    far_scale, far = _render(layer, 4000.0)
    assert near_scale < 500 < far_scale
    assert _has_colour(near, colour)
    assert _has_colour(far, colour)


def test_a_path_takes_its_species_colour():
    layer = QgsVectorLayer("LineString?crs=EPSG:32633", "path", "memory")
    layer.dataProvider().addAttributes(symbols.animal_fields())
    layer.updateFields()
    feature = QgsFeature(layer.fields())
    feature.setGeometry(QgsGeometry.fromWkt("LineString(0 0, 5 5)"))
    feature.setAttributes(symbols.animal_values(
        map_symbols.attributes("fox")))
    layer.dataProvider().addFeatures([feature])
    symbols.style_path(layer, {})
    assert layer.renderer().symbol().color().name() == \
        map_symbols.SEEDED_COLOURS["fox"]


def test_restyle_touches_only_what_was_styled():
    styled, path, other = _layer(), _layer(), _layer()
    symbols.style_boxes(styled, {}, 500)
    assert symbols.restyle([styled, other], {"hare": "#010203"}, 2000) == 1
    boxes, _pins = _rules(styled)
    assert "1:2,000" in boxes.label()
    assert boxes.children()[0].symbol().symbolLayer(0).strokeColor().name() \
        == "#010203"
    assert not isinstance(other.renderer(), QgsRuleBasedRenderer)
    assert path is not None


# --------------------------------------------------------------------------- #
# In the dock
# --------------------------------------------------------------------------- #

@pytest.fixture
def flight(dock, tmp_path, monkeypatch):
    from bambi_wildlife_detection.core import (
        detection_store, label_store, store, track_store)
    from qgis.PyQt.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    root = str(tmp_path / "meadow")
    os.makedirs(root)
    label_store.vocabulary(root, create=True)
    names = label_store.vocabulary(root)["species_by_name"]
    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0,
         "confidence": 0.9, "source_class": "0"} for frame in range(3)])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": det, "gx1": 447000.0 + 3 * i, "gy1": 5350500.0,
         "gz1": 0.0, "gx2": 447001.5 + 3 * i, "gy2": 5350500.8, "gz2": 0.0}
        for i, det in enumerate(ids)])
    species = {1: names["red deer"], 2: names["hare"]}
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 2, "detection_id": ids[2]}],
        species_of=lambda legacy: species[legacy])
    os.makedirs(os.path.join(root, "tracks_t"))
    open(os.path.join(root, "tracks_t", "tracks.csv"), "w").close()
    dock.target_folder_edit.setText(root)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    return root


def _accept_frame_range(monkeypatch):
    from bambi_wildlife_detection import bambi_dock_widget
    monkeypatch.setattr(bambi_dock_widget.FrameRangeDialog, "exec",
                        lambda self: QDialog.DialogCode.Accepted)


def _animal_layers(kind):
    return [layer for layer in QgsProject.instance().mapLayers().values()
            if layer.customProperty(symbols.STYLE_PROPERTY, "") == kind]


def test_the_settings_reach_the_config_and_the_project(dock):
    assert dock.get_config()["map_pins_enabled"] is True
    assert dock.get_config()["map_pin_scale"] == 500
    dock.map_pins_check.setChecked(False)
    dock.map_pin_scale_spin.setValue(1500)
    assert not dock.map_pin_scale_spin.isEnabled()
    dock.save_config_to_project()
    dock.map_pins_check.setChecked(True)
    dock.map_pin_scale_spin.setValue(200)
    dock.load_config_from_project()
    assert not dock.map_pins_check.isChecked()
    assert dock.map_pin_scale_spin.value() == 1500


def test_track_layers_are_drawn_by_species(dock, flight, monkeypatch):
    _accept_frame_range(monkeypatch)
    dock.map_pin_scale_spin.setValue(800)
    dock.add_tracks_to_qgis()
    boxes = _animal_layers(symbols.BOX)
    assert len(boxes) == 2
    by_track = {next(layer.getFeatures())["track_id"]: layer
                for layer in boxes}
    deer = by_track[1]
    assert next(deer.getFeatures())["species"] == "red deer"
    close, far = _rules(deer)
    assert "1:800" in close.label()
    assert [rule.label() for rule in close.children()] == [
        "red deer", "undetermined"]
    assert close.children()[0].symbol().symbolLayer(0).strokeColor().name() \
        == map_symbols.SEEDED_COLOURS["red deer"]
    assert isinstance(far.children()[0].symbol().symbolLayer(0),
                      QgsCentroidFillSymbolLayer)
    # the path of the two-box track, in the same colour
    (path,) = _animal_layers(symbols.PATH)
    assert path.renderer().symbol().color().name() == \
        map_symbols.SEEDED_COLOURS["red deer"]


def test_detection_layers_are_drawn_by_species(dock, flight, monkeypatch):
    _accept_frame_range(monkeypatch)
    dock.add_frame_detections_to_qgis()
    layers = _animal_layers(symbols.BOX)
    assert len(layers) == 3                         # one per frame
    species = sorted(f["species"] for layer in layers
                     for f in layer.getFeatures())
    assert species == ["hare", "red deer", "red deer"]
    assert all(len(_rules(layer)) == 2 for layer in layers)


def test_a_new_colour_in_the_schema_reaches_the_layers(dock, flight,
                                                       monkeypatch):
    from bambi_wildlife_detection.bambi_schema_dialog import BambiSchemaDialog

    _accept_frame_range(monkeypatch)
    dock.add_tracks_to_qgis()

    monkeypatch.setattr(QColorDialog, "getColor",
                        lambda *a, **k: QColor("#010203"))

    def choose_and_accept(dialog):
        for row in range(dialog.species_list.count()):
            if "red deer" in dialog.species_list.item(row).text():
                dialog.species_list.setCurrentRow(row)
        assert not dialog.species_colour_reset_btn.isEnabled()
        dialog.species_colour_btn.click()
        assert dialog.species_colour_reset_btn.isEnabled()
        icon = dialog.species_list.currentItem().icon()
        assert icon.pixmap(14, 14).toImage().pixelColor(7, 7).name() \
            == "#010203"
        dialog.accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(BambiSchemaDialog, "exec", choose_and_accept)
    assert dock.open_schema_dialog()
    assert map_symbols.species_colours(flight)["red deer"] == "#010203"
    deer = next(layer for layer in _animal_layers(symbols.BOX)
                if next(layer.getFeatures())["species"] == "red deer")
    close, _far = _rules(deer)
    assert close.children()[0].symbol().symbolLayer(0).strokeColor().name() \
        == "#010203"


def test_a_cancelled_schema_dialog_keeps_the_colours(dock, flight,
                                                     monkeypatch):
    from bambi_wildlife_detection.bambi_schema_dialog import BambiSchemaDialog

    monkeypatch.setattr(QColorDialog, "getColor",
                        lambda *a, **k: QColor("#010203"))

    def choose_and_cancel(dialog):
        dialog.species_list.setCurrentRow(0)
        dialog.species_colour_btn.click()
        dialog.reject()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(BambiSchemaDialog, "exec", choose_and_cancel)
    assert not dock.open_schema_dialog()
    assert "#010203" not in map_symbols.species_colours(flight).values()


def test_apply_to_layers_takes_the_new_scale(dock, flight, monkeypatch):
    _accept_frame_range(monkeypatch)
    dock.add_tracks_to_qgis()
    dock.map_pin_scale_spin.setValue(2500)
    dock.map_restyle_btn.click()
    close, _far = _rules(_animal_layers(symbols.BOX)[0])
    assert "1:2,500" in close.label()
    dock.map_pins_check.setChecked(False)
    assert dock.restyle_animal_layers() == 3
    assert all(rule.symbol() is not None
               for rule in _rules(_animal_layers(symbols.BOX)[0]))
