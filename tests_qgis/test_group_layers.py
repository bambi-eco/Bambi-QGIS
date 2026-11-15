# -*- coding: utf-8 -*-
"""The groups on the map, and in the census table (dock/group_layers.py).

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import json
import os

import pytest

from qgis.core import QgsProject, QgsWkbTypes
from qgis.PyQt.QtWidgets import QMessageBox


def _row(group, species, positions, flight="Meadow"):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return {"flight": flight, "group": group, "species": species,
            "size": len(positions), "counted": len(positions),
            "x": sum(xs) / len(xs), "y": sum(ys) / len(ys), "radius_m": 12.5,
            "first_time": "2024-05-01T05:00:30Z",
            "last_time": "2024-05-01T05:00:55Z",
            "members": list(range(group, group + len(positions))),
            "positions": [list(p) for p in positions]}


GROUPS = [
    _row(1, "red deer", [(500010.0, 5300020.0), (500040.0, 5300020.0),
                         (500040.0, 5300060.0)]),
    _row(6, "roe deer", [(500300.0, 5300000.0), (500330.0, 5300000.0)]),
    _row(8, "red deer", [(500700.0, 5300000.0)]),          # alone
    dict(_row(9, "lynx", [(0.0, 0.0)]), x=None, y=None, positions=[]),
]


@pytest.fixture
def survey_with_groups(dock, tmp_path, monkeypatch):
    flight = str(tmp_path / "meadow")
    os.makedirs(flight)
    dock.target_folder_edit.setText(flight)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    folder = tmp_path / "survey"
    dock.survey_panel.folder_edit.setText(str(folder))
    out = folder / "analytics_t"
    out.mkdir(parents=True)
    (out / "groups_tracks.json").write_text(json.dumps({
        "source": "tracks", "camera": "Thermal", "crs": "EPSG:32633",
        "groups": GROUPS}))
    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *a, **k: shown.append(a[1:]))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[1:]))
    return shown


def _groups_layer():
    found = [layer for layer in QgsProject.instance().mapLayers().values()
             if layer.customProperty("bambi_layer_type", "") == "groups"]
    assert len(found) == 1, [layer.name() for layer in found]
    return found[0]


def test_the_census_box_has_the_button(dock):
    assert dock.add_groups_btn.text() == "→ Add Groups to QGIS"
    assert dock.add_groups_status.text() == "⚪"
    dock.update_status("add_groups", "🟢 Added")
    assert "Added" in dock.add_groups_status.text()


def test_groups_of_two_or_more_are_outlined(dock, survey_with_groups):
    dock.add_groups_to_qgis()

    assert survey_with_groups == []
    layer = _groups_layer()
    assert layer.name() == "BAMBI Groups (Thermal)"
    assert layer.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry
    assert layer.crs().authid() == "EPSG:32633"
    rows = {feature["group"]: feature for feature in layer.getFeatures()}
    assert sorted(rows) == [1, 6]                 # the two that were alone: no
    herd = rows[1]
    assert (herd["flight"], herd["species"], herd["size"], herd["counted"],
            herd["members"]) == ("Meadow", "red deer", 3, 3, "1 2 3")
    assert herd["first_time"] == "2024-05-01T05:00:30Z"
    assert "Added" in dock.add_groups_status.text()
    assert "Added 2 group(s) of two or more animals; 2 animal(s) were alone" \
        in dock.log_text.toPlainText()


def test_an_outline_goes_around_the_members_a_little_way_off(
        dock, survey_with_groups):
    from qgis.core import QgsGeometry, QgsPointXY

    dock.add_groups_to_qgis()
    rows = {feature["group"]: feature for feature in
            _groups_layer().getFeatures()}

    for group in (1, 6):
        outline = rows[group].geometry()
        for x, y in next(row for row in GROUPS
                         if row["group"] == group)["positions"]:
            member = QgsGeometry.fromPointXY(QgsPointXY(x, y))
            assert outline.contains(member)
            # 3 m off, give or take the corners of a 12-sided round
            assert 2.8 < QgsGeometry.fromPolylineXY(
                outline.asPolygon()[0]).distance(member) < 3.01
    # two animals are a line: its outline still has an inside
    assert rows[6].geometry().area() > 30 * 6 * 0.99


def test_every_species_has_its_colour_and_the_size_is_written_in(
        dock, survey_with_groups):
    from bambi_wildlife_detection.core import map_symbols

    dock.add_groups_to_qgis()
    layer = _groups_layer()

    renderer = layer.renderer()
    assert renderer.classAttribute() == "species"
    colours = {category.value(): category.symbol().symbolLayer(0)
               .strokeColor().name() for category in renderer.categories()}
    assert colours == {
        "red deer": map_symbols.default_colour("red deer"),
        "roe deer": map_symbols.default_colour("roe deer")}
    assert layer.labelsEnabled()
    assert layer.labeling().settings().fieldName == "size"


def test_the_layer_goes_into_the_surveys_group(dock, survey_with_groups):
    dock.add_groups_to_qgis()

    group = QgsProject.instance().layerTreeRoot().findGroup(dock.SURVEY_GROUP)
    assert group is not None
    assert "BAMBI Groups (Thermal)" in [
        node.name() for node in group.findLayers()]


def test_adding_again_replaces_the_layer(dock, survey_with_groups):
    dock.add_groups_to_qgis()
    dock.add_groups_to_qgis()

    assert _groups_layer().featureCount() == 2


def test_without_a_census_there_is_nothing_to_add(dock, tmp_path,
                                                  monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[1:]))
    dock.survey_panel.folder_edit.setText(str(tmp_path / "empty"))

    dock.add_groups_to_qgis()

    assert warned and "Build Census Summary" in warned[0][1]
    assert dock.add_groups_status.text() == "⚪"


def test_the_census_table_shows_the_groups(dock):
    from bambi_wildlife_detection.bambi_survey_panel import CensusTableDialog
    from bambi_wildlife_detection.core import census

    def record(group):
        return {"species": "red deer", "group": group, "x": 1.0, "y": 1.0,
                "sex": None, "age": None, "approved": False}

    result = census.build([{"name": "Meadow", "covered_m2": 1e4, "records": [
        record(1), record(1), record(1), record(4)]}])
    result.update({"camera": "Thermal", "source": "tracks", "notes": ""})
    dialog = CensusTableDialog(result)
    table = dialog._table
    headers = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]

    assert headers[11:14] == ["Groups", "Mean group size", "Largest group"]
    assert [table.item(0, c).text() for c in (11, 12, 13)] == [
        "2", "2.00", "3"]
