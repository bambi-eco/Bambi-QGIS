# -*- coding: utf-8 -*-
"""Animal symbology inside a real QGIS: boxes in the individual's colour up
close; when zoomed out, pins in the species' colour with the sex inside
them and the age as a badge. Species colours come from the Project Schema.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.core import (
    QgsCentroidFillSymbolLayer, QgsFeature, QgsFontMarkerSymbolLayer,
    QgsGeometry, QgsMapRendererParallelJob, QgsMapSettings, QgsProject,
    QgsRectangle, QgsRuleBasedRenderer, QgsSimpleFillSymbolLayer,
    QgsSvgMarkerSymbolLayer, QgsVectorLayer,
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
        feature.setAttributes(symbols.animal_values(map_symbols.attributes(
            species, sex, age, track_id=index + 1 if species else None)))
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()
    return layer


def _rules(layer):
    renderer = layer.renderer()
    assert isinstance(renderer, QgsRuleBasedRenderer)
    return renderer.rootRule().children()


def _pin_colour(rule):
    """Fill colour of the teardrop of a pin rule."""
    return _pin_layers(rule)[0].fillColor().name()


def _pin_layers(rule):
    marker = rule.symbol().symbolLayer(0).subSymbol()
    return [marker.symbolLayer(i) for i in range(marker.symbolLayerCount())]


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


def test_a_box_takes_its_individuals_colour():
    """One symbol for every box, reading the colour from the feature - so
    two animals of one species are still two colours."""
    layer = _layer()
    symbols.style_boxes(layer, {}, 500)
    boxes, _pins = _rules(layer)
    assert boxes.children() == []
    fill = boxes.symbol().symbolLayer(0)
    assert isinstance(fill, QgsSimpleFillSymbolLayer)
    stroke = fill.dataDefinedProperties().property(
        symbols._property("PropertyStrokeColor"))
    assert "box_colour" in stroke.expressionString()
    colours = {f["species"]: f["box_colour"] for f in layer.getFeatures()}
    assert colours["roe deer"] == map_symbols.individual_colour(1)
    assert colours["red deer"] == map_symbols.individual_colour(2)
    assert colours[""] == map_symbols.UNTRACKED_COLOUR
    assert len(set(colours.values())) == 4


def test_one_pin_rule_per_species_present_and_one_for_the_rest():
    layer = _layer()
    symbols.style_boxes(layer, {}, 500)
    _boxes, pins = _rules(layer)
    labels = [rule.label() for rule in pins.children()]
    assert labels == ["hare", "red deer", "roe deer", "undetermined"]
    assert pins.children()[-1].isElse()
    assert pins.children()[0].filterExpression() == "\"species\" = 'hare'"


def test_a_pin_is_the_colour_of_its_species():
    layer = _layer()
    symbols.style_boxes(layer, {"hare": "#010203"}, 500)
    _boxes, pins = _rules(layer)
    hare, deer = pins.children()[0], pins.children()[1]
    assert isinstance(hare.symbol().symbolLayer(0),
                      QgsCentroidFillSymbolLayer)
    assert _pin_colour(hare) == "#010203"
    assert _pin_colour(deer) == map_symbols.SEEDED_COLOURS["red deer"]


def test_sex_and_age_are_part_of_the_pin():
    """Nothing is drawn beside the pin: the sex mark sits in the teardrop's
    round part and the age badge overlaps its lower right, so with pins
    close together neither can be read as a neighbour's."""
    layer = _layer()
    symbols.style_boxes(layer, {}, 500)
    _boxes, pins = _rules(layer)
    body, sex, badge, age = _pin_layers(pins.children()[0])
    assert isinstance(body, QgsSvgMarkerSymbolLayer)
    half_width = body.size() / 2.0

    # the sex mark: centred, inside the pin, "?" where the sex is unknown
    assert isinstance(sex, QgsFontMarkerSymbolLayer)
    assert sex.offset().x() == 0 and sex.offset().y() < 0
    character = sex.dataDefinedProperties().property(
        symbols._property("PropertyCharacter")).expressionString()
    assert "sex_mark" in character and "'?'" in character
    assert sex.fontFamily() != "Dingbats"

    # the age badge: lower right, overlapping the pin, ringed in its colour
    assert 0 < badge.offset().x() < half_width
    assert badge.offset().x() + badge.size() / 2.0 > half_width
    assert badge.strokeColor().name() == body.fillColor().name()
    assert badge.color().name() == "#ffffff"
    assert age.offset() == badge.offset()
    assert age.dataDefinedProperties().property(
        symbols._property("PropertyCharacter")).field() == "age_mark"

    marks = {f["species"]: (f["sex_mark"], f["age_mark"])
             for f in layer.getFeatures()}
    assert marks["roe deer"] == ("♀", "J")
    assert marks["red deer"] == ("♂", "A")
    assert marks["hare"] == ("", "")


def test_the_mark_is_dark_on_a_light_pin():
    layer = _layer()
    symbols.style_boxes(layer, {"hare": "#ffffcc", "red deer": "#400000"},
                        500)
    _boxes, pins = _rules(layer)
    hare, deer = pins.children()[0], pins.children()[1]
    assert _pin_layers(hare)[1].color().name() == "#202020"
    assert _pin_layers(deer)[1].color().name() == "#ffffff"


def test_the_pin_drawing_travels_with_the_style():
    """Embedded, not a path into this plugin installation: the project
    still shows pins on another computer."""
    path = symbols.pin_svg_path()
    assert path.startswith("base64:")
    layer = _layer()
    symbols.style_boxes(layer, {}, 500)
    _boxes, pins = _rules(layer)
    assert _pin_layers(pins.children()[0])[0].path().startswith("base64:")


def test_pins_can_be_switched_off():
    layer = _layer()
    symbols.style_boxes(layer, {}, 500, pins=False)
    (boxes,) = _rules(layer)
    assert boxes.label() == "Boxes"
    assert boxes.isScaleOK(1_000_000)
    assert isinstance(boxes.symbol().symbolLayer(0), QgsSimpleFillSymbolLayer)


def test_a_quote_in_a_species_name_does_not_break_the_filter():
    layer = _layer([("grant's gazelle", "", "")])
    symbols.style_boxes(layer, {}, 500)
    rule = _rules(layer)[1].children()[0]
    assert rule.filterExpression() == "\"species\" = 'grant''s gazelle'"


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
    """Rendered through QGIS, not just configured: zoomed in the box is on
    the map in the individual's colour; zoomed far out, where the box would
    be less than a pixel, the pin is there in the species' colour."""
    layer = _layer([("red deer", "male", "adult")])
    symbols.style_boxes(layer, {}, 500)
    species = map_symbols.SEEDED_COLOURS["red deer"]
    individual = map_symbols.individual_colour(1)
    near_scale, near = _render(layer, 3.0)
    far_scale, far = _render(layer, 4000.0)
    assert near_scale < 500 < far_scale
    assert _has_colour(near, individual) and not _has_colour(near, species)
    assert _has_colour(far, species) and not _has_colour(far, individual)


def test_a_path_takes_its_individuals_colour():
    layer = QgsVectorLayer("LineString?crs=EPSG:32633", "path", "memory")
    layer.dataProvider().addAttributes(symbols.animal_fields())
    layer.updateFields()
    feature = QgsFeature(layer.fields())
    feature.setGeometry(QgsGeometry.fromWkt("LineString(0 0, 5 5)"))
    feature.setAttributes(symbols.animal_values(
        map_symbols.attributes("fox", track_id=4)))
    layer.dataProvider().addFeatures([feature])
    symbols.style_path(layer)
    assert layer.renderer().symbol().color().name() == \
        map_symbols.individual_colour(4)


def test_restyle_touches_only_what_was_styled():
    styled, other = _layer(), _layer()
    symbols.style_boxes(styled, {}, 500)
    assert symbols.restyle([styled, other], {"hare": "#010203"}, 2000) == 1
    boxes, pins = _rules(styled)
    assert "1:2,000" in boxes.label()
    assert _pin_colour(pins.children()[0]) == "#010203"
    assert not isinstance(other.renderer(), QgsRuleBasedRenderer)


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


def test_tracks_differ_up_close_and_agree_by_species_far_out(
        dock, flight, monkeypatch):
    _accept_frame_range(monkeypatch)
    dock.map_pin_scale_spin.setValue(800)
    dock.add_tracks_to_qgis()
    boxes = _animal_layers(symbols.BOX)
    assert len(boxes) == 2
    by_track = {next(layer.getFeatures())["track_id"]: layer
                for layer in boxes}
    deer = by_track[1]
    feature = next(deer.getFeatures())
    assert feature["species"] == "red deer"
    assert feature["box_colour"] == map_symbols.individual_colour(1)
    assert next(by_track[2].getFeatures())["box_colour"] == \
        map_symbols.individual_colour(2)

    close, far = _rules(deer)
    assert "1:800" in close.label()
    assert [rule.label() for rule in far.children()] == [
        "red deer", "undetermined"]
    assert _pin_colour(far.children()[0]) == \
        map_symbols.SEEDED_COLOURS["red deer"]
    # the path of the two-box track, in the individual's colour like its box
    (path,) = _animal_layers(symbols.PATH)
    assert path.renderer().symbol().color().name() == \
        map_symbols.individual_colour(1)


def test_detections_share_their_tracks_colour(dock, flight, monkeypatch):
    _accept_frame_range(monkeypatch)
    dock.add_frame_detections_to_qgis()
    layers = _animal_layers(symbols.BOX)
    assert len(layers) == 3                         # one per frame
    features = [f for layer in layers for f in layer.getFeatures()]
    assert sorted(f["species"] for f in features) == [
        "hare", "red deer", "red deer"]
    deer = {f["box_colour"] for f in features if f["species"] == "red deer"}
    assert deer == {map_symbols.individual_colour(1)}
    assert all(len(_rules(layer)) == 2 for layer in layers)


def test_the_settings_are_read_once_for_all_the_layers(dock, flight,
                                                       monkeypatch):
    """Reading them means reading the whole form (which re-parses the
    flight log for the time zone) and the flight's store: once per add, not
    once per frame layer."""
    _accept_frame_range(monkeypatch)
    calls = []
    original = type(dock)._animal_style_settings

    def counted(self, config=None):
        calls.append(1)
        return original(self, config)

    monkeypatch.setattr(type(dock), "_animal_style_settings", counted)
    dock.add_frame_detections_to_qgis()
    assert len(_animal_layers(symbols.BOX)) == 3
    assert len(calls) == 1
    calls.clear()
    dock.add_tracks_to_qgis()
    assert len(calls) == 1


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
    _close, far = _rules(deer)
    assert _pin_colour(far.children()[0]) == "#010203"


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
    # Restyling replaces the renderer and with it these rules; a wrapper
    # kept past that point is a dangling pointer when Python cleans up.
    del close, _far
    dock.map_pins_check.setChecked(False)
    assert dock.restyle_animal_layers() == 3
    labels = [rule.label() for rule in _rules(_animal_layers(symbols.BOX)[0])]
    assert labels == ["Boxes"]
