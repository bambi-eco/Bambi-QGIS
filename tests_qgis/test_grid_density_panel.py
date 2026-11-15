# -*- coding: utf-8 -*-
"""Survey Analytics > Grid Density: the settings, the result, the layer
(bambi_grid_panel.py, dock/grid_density.py).

The analysis itself is covered without QGIS in tests/test_grid_density.py
and tests/test_grid_step.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import json
import os

import pytest

from qgis.core import QgsProject, QgsWkbTypes
from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.bambi_grid_panel import (
    ENTRY_GRID, GridSettingsPanel, GridStatisticsDialog, statistics_rows)
from bambi_wildlife_detection.core import grid_density, survey_grid
from bambi_wildlife_detection.dock.grid_density import (
    STYLE_DENSITY, STYLE_HOT_SPOTS, density_breaks, density_class,
    density_classes, spot_class)
from bambi_wildlife_detection.gui_utils import PLUGIN_SCOPE


# ---------------------------------------------------------------------------
# The settings
# ---------------------------------------------------------------------------

@pytest.fixture
def panel():
    made = GridSettingsPanel()
    made.show()
    yield made
    made.deleteLater()


def _select(combo, data):
    combo.setCurrentIndex(combo.findData(data))


class TestSettings:
    def test_a_hexagon_of_a_hectare_half_seen_averaged(self, panel):
        assert panel.values() == {
            "grid_shape": "hexagon", "grid_size_mode": "area",
            "grid_cell_area_ha": 1.0, "grid_cell_edge_m": 100.0,
            "grid_cell_width_m": 100.0, "grid_cell_height_m": 100.0,
            "grid_min_seen_pct": 50.0, "grid_repeat": "average"}
        assert [panel.shape_combo.itemText(i)
                for i in range(panel.shape_combo.count())] == [
            "Hexagons", "Squares", "Triangles", "Rectangles"]
        assert [panel.repeat_combo.itemText(i)
                for i in range(panel.repeat_combo.count())] == [
            "Average over the flights", "Highest flight"]

    def test_the_panel_says_what_the_grid_is(self, panel):
        assert panel.summary_label.text().startswith(
            "Hexagons of 1 ha (edge 62.0 m). A cell counts when at least "
            "50 % of it was seen")

        _select(panel.shape_combo, "square")
        panel.area_spin.setValue(0.25)
        panel.min_seen_spin.setValue(80)

        assert panel.summary_label.text().startswith(
            "Squares of 0.25 ha (edge 50.0 m). A cell counts when at least "
            "80 % of it was seen")

    def test_size_by_area_or_by_edge_shows_the_one_that_applies(self, panel):
        assert panel.area_spin.isVisible() and not panel.edge_spin.isVisible()

        _select(panel.size_mode_combo, "edge")

        assert panel.edge_spin.isVisible() and not panel.area_spin.isVisible()

    def test_switching_keeps_the_cells_as_they_are(self, panel):
        """Area and edge are two ways to say the same."""
        _select(panel.size_mode_combo, "edge")

        assert panel.edge_spin.value() == pytest.approx(62.04, abs=0.01)
        assert panel.spec().cell_area_m2 == pytest.approx(10000.0, rel=1e-3)

        panel.edge_spin.setValue(100.0)
        _select(panel.size_mode_combo, "area")

        assert panel.area_spin.value() == pytest.approx(2.5981, abs=1e-4)
        assert panel.spec().edge == pytest.approx(100.0, rel=1e-3)

    def test_another_shape_keeps_the_size_that_is_shown(self, panel):
        panel.area_spin.setValue(2.0)
        _select(panel.shape_combo, "triangle")

        assert panel.area_spin.value() == 2.0
        assert panel.spec().cell_area_m2 == pytest.approx(20000.0)

    def test_a_rectangle_has_a_width_and_a_height_instead(self, panel):
        _select(panel.shape_combo, "rectangle")
        panel.width_spin.setValue(200.0)
        panel.height_spin.setValue(50.0)

        assert panel.width_spin.isVisible() and panel.height_spin.isVisible()
        assert not panel.size_mode_combo.isVisible()
        assert not panel.area_spin.isVisible()
        assert not panel.edge_spin.isVisible()
        assert panel.spec().cell_area_m2 == 10000.0
        assert panel.summary_label.text().startswith(
            "Rectangles of 200 m x 50 m (1 ha)")

    def test_a_change_by_the_user_is_reported_a_loaded_one_is_not(self, panel):
        said = []
        panel.changed.connect(lambda: said.append(True))

        panel.set_values({"grid_shape": "square", "grid_min_seen_pct": 70})
        assert said == []
        assert panel.values()["grid_shape"] == "square"
        assert panel.values()["grid_min_seen_pct"] == 70.0

        panel.min_seen_spin.setValue(60)
        _select(panel.repeat_combo, "highest")
        assert len(said) == 2

    def test_what_is_missing_or_wrong_falls_back_on_the_default(self, panel):
        panel.set_values({"grid_shape": "octagon", "grid_cell_area_ha": "x",
                          "grid_repeat": None, "something": 1})

        assert panel.values()["grid_shape"] == "hexagon"
        assert panel.values()["grid_cell_area_ha"] == 1.0
        assert panel.values()["grid_repeat"] == "average"

    def test_the_settings_are_kept_with_the_project(self, panel):
        project = QgsProject.instance()
        _select(panel.shape_combo, "triangle")
        _select(panel.repeat_combo, "highest")
        panel.min_seen_spin.setValue(75)
        panel.save(project)

        stored, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_GRID, "")
        assert ok and json.loads(stored)["grid_shape"] == "triangle"

        panel.reset()
        assert panel.values()["grid_shape"] == "hexagon"
        panel.load(project)
        assert panel.values()["grid_shape"] == "triangle"
        assert panel.values()["grid_repeat"] == "highest"
        assert panel.values()["grid_min_seen_pct"] == 75.0
        project.removeEntry(PLUGIN_SCOPE, ENTRY_GRID)

    def test_a_project_without_grid_settings_has_the_defaults(self, panel):
        project = QgsProject.instance()
        project.removeEntry(PLUGIN_SCOPE, ENTRY_GRID)
        panel.set_values({"grid_shape": "square"})

        panel.load(project)

        assert panel.values()["grid_shape"] == "hexagon"
        project.writeEntry(PLUGIN_SCOPE, ENTRY_GRID, "{not json")
        panel.load(project)
        assert panel.values()["grid_shape"] == "hexagon"
        project.removeEntry(PLUGIN_SCOPE, ENTRY_GRID)


# ---------------------------------------------------------------------------
# A result to show
# ---------------------------------------------------------------------------

def _document():
    from shapely.geometry import box

    region = box(500000, 5300000, 500800, 5300800)
    animals = []
    for col in range(3):
        for row in range(3):
            for k in range(6):
                animals.append({
                    "species": "red deer", "sex": None, "age": None,
                    "group": col * 10 + row,
                    "x": 500000 + 100 * col + 10 + 12 * k,
                    "y": 5300000 + 100 * row + 50})
    for col, row in ((5, 1), (6, 6), (2, 6)):
        animals.append({"species": "roe deer", "sex": "female", "age": None,
                        "group": 100 + col, "x": 500000 + 100 * col + 50,
                        "y": 5300000 + 100 * row + 50})
    spec = survey_grid.GridSpec(shape=survey_grid.SQUARE,
                                size_mode=survey_grid.BY_EDGE, edge_m=100.0)
    document = grid_density.analyse(
        spec, [{"name": "Meadow", "records": animals,
                "seen": box(500000, 5300000, 500800, 5300640)}], region,
        plot_m2=float(region.area))
    document.update({"source": "tracks", "camera": "Thermal",
                     "crs": "EPSG:32633", "notes": "What the figures mean."})
    return document


@pytest.fixture
def survey_with_grid(dock, tmp_path, monkeypatch):
    flight = str(tmp_path / "meadow")
    os.makedirs(flight)
    dock.target_folder_edit.setText(flight)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    folder = tmp_path / "survey"
    dock.survey_panel.folder_edit.setText(str(folder))
    out = folder / "analytics_t"
    out.mkdir(parents=True)
    grid_density.write_json(
        _document(), str(out / "grid_density_tracks.json"))
    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *a, **k: shown.append(a[1:]))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[1:]))
    return shown


# ---------------------------------------------------------------------------
# The group on the tab
# ---------------------------------------------------------------------------

class TestGroup:
    def test_the_tab_has_the_group(self, dock):
        assert isinstance(dock.grid_panel, GridSettingsPanel)
        assert dock.grid_btn.text() == "→ Analyse Grid Density"
        assert dock.grid_show_btn.text() == "→ Show Grid Statistics"
        assert dock.add_grid_btn.text() == "→ Add Grid to QGIS"
        assert dock.grid_status.text() == "⚪"
        assert "grid_density" in dock._SURVEY_STEPS
        dock.update_status("grid_density", "🟢 Completed")
        dock.update_status("add_grid", "🟢 Added")
        assert "Completed" in dock.grid_status.text()
        assert "Added" in dock.add_grid_status.text()

    def test_the_settings_reach_the_step(self, dock):
        _select(dock.grid_panel.shape_combo, "square")
        dock.grid_panel.min_seen_spin.setValue(65)
        config = dock.get_config()

        assert config["grid_shape"] == "square"
        assert config["grid_min_seen_pct"] == 65.0
        assert config["grid_repeat"] == "average"
        assert config["grid_source"] == config["census_source"]
        assert survey_grid.GridSpec.from_config(config).shape == "square"

    def test_the_settings_are_saved_and_restored_with_the_project(self, dock):
        _select(dock.grid_panel.shape_combo, "triangle")
        dock.grid_panel.area_spin.setValue(0.5)
        dock.save_config_to_project()

        dock.grid_panel.reset()
        dock.load_config_from_project()

        assert dock.grid_panel.values()["grid_shape"] == "triangle"
        assert dock.grid_panel.values()["grid_cell_area_ha"] == 0.5

    def test_a_new_project_starts_with_the_default_grid(self, dock):
        _select(dock.grid_panel.shape_combo, "square")

        dock.reset_config_to_defaults()

        assert dock.grid_panel.values()["grid_shape"] == "hexagon"

    def test_the_button_starts_the_step(self, dock, monkeypatch):
        started = []
        survey = dock.current_survey()
        monkeypatch.setattr(dock, "_checked_survey", lambda: survey)
        monkeypatch.setattr(dock, "_survey_prerequisites",
                            lambda survey, checks: True)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.grid_btn.click()

        assert started == ["grid_density"]

    def test_nothing_starts_without_a_survey(self, dock, monkeypatch):
        started = []
        monkeypatch.setattr(dock, "_checked_survey", lambda: None)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.run_grid_density()

        assert started == []

    def test_when_the_step_has_ended_the_statistics_are_shown(
            self, dock, survey_with_grid):
        dock.on_worker_finished("grid_density", True)

        assert dock._grid_dialog.isVisible()
        assert "Completed" in dock.grid_status.text()
        dock._grid_dialog.close()


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_a_row_for_all_animals_and_one_per_species(self):
        headers, rows = statistics_rows(_document())

        assert headers[:4] == ["", "Animals", "Density /km² (95 %)",
                               "In the survey area"]
        assert [row[0] for row in rows] == ["All animals", "red deer",
                                            "roe deer"]
        red = rows[1]
        assert red[1] == "54"
        assert red[2].startswith("112.5 (")           # 54 on 48 ha counted
        assert red[3].startswith("72 (")              # on 64 ha of plot
        assert "p < 0.001" in red[4] and red[5] == "clumped"
        assert red[7].startswith("9 of 48 (19 %, ")
        assert int(red[8]) >= 4
        assert rows[2][5] == "scattered at random"

    def test_the_dialog_shows_them(self, dock):
        dialog = GridStatisticsDialog(_document(), json_path="/x/grid.json",
                                      parent=dock)
        table = dialog._table

        assert dialog.windowTitle() == "Grid Density (Thermal, tracks)"
        assert table.rowCount() == 3 and table.columnCount() == 10
        assert table.item(1, 0).text() == "red deer"
        assert table.item(0, 0).font().bold()
        dialog.copy_table()
        from qgis.PyQt.QtWidgets import QApplication
        copied = QApplication.clipboard().text().splitlines()
        assert copied[0].split("\t")[1] == "Animals"
        assert copied[2].split("\t")[0] == "red deer"

    def test_show_reads_the_surveys_result(self, dock, survey_with_grid):
        dock.show_grid_statistics()

        assert survey_with_grid == []
        assert dock._grid_dialog.isVisible()
        dock._grid_dialog.close()

    def test_without_a_result_there_is_nothing_to_show(self, dock, tmp_path,
                                                       monkeypatch):
        warned = []
        monkeypatch.setattr(QMessageBox, "warning",
                            lambda *a, **k: warned.append(a[1:]))
        dock.survey_panel.folder_edit.setText(str(tmp_path / "empty"))

        dock.show_grid_statistics()
        dock.add_grid_to_qgis()

        assert len(warned) == 2
        assert all("Analyse Grid Density" in text for _title, text in warned)
        assert dock.add_grid_status.text() == "⚪"

    def test_a_result_without_figures_is_all_dashes(self):
        document = {"statistics": {"all": {
            "animals": 0, "density_per_km2": {"estimate": None},
            "total_in_survey_area": None, "dispersion": {}, "morans_i": {},
            "occupancy": {}}}}

        _headers, rows = statistics_rows(document)

        assert rows == [["All animals", "0", "–", "–", "–", "–", "–", "–",
                         "0", "0"]]


# ---------------------------------------------------------------------------
# The grid on the map
# ---------------------------------------------------------------------------

def _grid_layer():
    found = [layer for layer in QgsProject.instance().mapLayers().values()
             if layer.customProperty("bambi_layer_type", "") == "grid_density"]
    assert len(found) == 1, [layer.name() for layer in found]
    return found[0]


class TestLayer:
    def test_every_cell_is_a_polygon_with_its_figures(self, dock,
                                                      survey_with_grid):
        dock.add_grid_to_qgis()

        assert survey_with_grid == []
        layer = _grid_layer()
        assert layer.name() == "BAMBI Grid Density (Thermal)"
        assert layer.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry
        assert layer.crs().authid() == "EPSG:32633"
        assert layer.featureCount() == 64
        cells = {feature["cell"]: feature for feature in layer.getFeatures()}
        herd = cells["c5000r53000"]
        assert (herd["status"], herd["animals"], herd["n_red_deer"]) == (
            "counted", 6, 6)
        assert herd["density_per_km2"] == pytest.approx(600.0)
        assert herd["spot"] == "hot spot (99 %)"
        assert herd.geometry().area() == pytest.approx(10000.0)
        assert cells["c5000r53007"]["status"] == "not seen"
        assert cells["c5000r53006"]["status"] == "too little seen"   # 40 %
        names = [field.name() for field in layer.fields()]
        for column in ("seen_share", "groups", "mean_group_size",
                       "per_km2_roe_deer", "spot_red_deer", "sex_female"):
            assert column in names
        assert "Added" in dock.add_grid_status.text()
        assert "Added the grid: 64 cell(s), 48 counted" in \
            dock.log_text.toPlainText()

    def test_the_layer_comes_with_two_styles(self, dock, survey_with_grid):
        dock.add_grid_to_qgis()
        layer = _grid_layer()
        manager = layer.styleManager()

        assert STYLE_DENSITY in manager.styles()
        assert STYLE_HOT_SPOTS in manager.styles()
        assert manager.currentStyle() == STYLE_DENSITY

        def classes():
            renderer = layer.renderer()
            return renderer.classAttribute(), [
                category.label() for category in renderer.categories()]

        attribute, density = classes()
        assert attribute == "density_class"
        assert density[0] == "no animals"
        assert density[-2:] == ["too little seen", "not seen"]
        assert density[1:-2] == ["0 - 600 /km²"]

        manager.setCurrentStyle(STYLE_HOT_SPOTS)
        attribute, spots = classes()
        assert attribute == "spot_class"
        assert spots[0] == "hot spot (99 %)" and "cold spot (99 %)" in spots
        assert spots[-3:] == ["nothing stands out", "too little seen",
                              "not seen"]
        manager.setCurrentStyle(STYLE_DENSITY)
        assert classes()[0] == "density_class"

    def test_every_cell_has_its_class_in_both_styles(self, dock,
                                                     survey_with_grid):
        dock.add_grid_to_qgis()
        layer = _grid_layer()
        manager = layer.styleManager()

        for style, attribute in ((STYLE_DENSITY, "density_class"),
                                 (STYLE_HOT_SPOTS, "spot_class")):
            manager.setCurrentStyle(style)
            known = {category.value()
                     for category in layer.renderer().categories()}
            found = {feature[attribute] for feature in layer.getFeatures()}
            assert found <= known, (style, found - known)
        manager.setCurrentStyle(STYLE_DENSITY)
        cells = {feature["cell"]: feature for feature in layer.getFeatures()}
        assert cells["c5000r53000"]["density_class"] == "0 - 600 /km²"
        assert cells["c5000r53000"]["spot_class"] == "hot spot (99 %)"
        assert cells["c5007r53000"]["density_class"] == "no animals"
        assert cells["c5007r53000"]["spot_class"] in (
            "nothing stands out", "cold spot (90 %)", "cold spot (95 %)")
        assert cells["c5000r53007"]["density_class"] == "not seen"
        assert cells["c5000r53006"]["spot_class"] == "too little seen"

    def test_cells_with_animals_say_how_many(self, dock, survey_with_grid):
        dock.add_grid_to_qgis()
        layer = _grid_layer()

        assert layer.labelsEnabled()
        settings = layer.labeling().settings()
        assert settings.isExpression and "animals" in settings.fieldName

    def test_the_layer_goes_into_the_surveys_group_once(self, dock,
                                                        survey_with_grid):
        dock.add_grid_to_qgis()
        dock.add_grid_to_qgis()

        group = QgsProject.instance().layerTreeRoot().findGroup(
            dock.SURVEY_GROUP)
        names = [node.name() for node in group.findLayers()]
        assert names.count("BAMBI Grid Density (Thermal)") == 1
        assert _grid_layer().featureCount() == 64


class TestBreaks:
    def test_classes_hold_about_as_many_cells_each(self):
        limits = density_breaks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        assert limits == [20.0, 40.0, 60.0, 80.0, 100.0]

    def test_nought_and_nothing_are_no_class(self):
        assert density_breaks([0, 0, None]) == []
        assert density_breaks([]) == []

    def test_the_last_class_reaches_the_highest_cell(self):
        limits = density_breaks([1.234, 5.678, 987.654])

        assert limits[-1] >= 987.654
        assert limits == sorted(set(limits))

    def test_few_values_few_classes(self):
        assert density_breaks([200.0, 200.0, 200.0]) == [200.0]

    def test_limits_can_be_read(self):
        assert density_breaks([123.456] * 3 + [987.6])[0] == 120.0

    def test_a_cell_is_in_the_class_its_density_falls_in(self):
        limits = [20.0, 40.0, 100.0]

        assert density_classes(limits) == [
            "0 - 20 /km²", "20 - 40 /km²", "40 - 100 /km²"]

        def cell(density, status="counted"):
            return {"status": status, "density_per_km2": density}

        assert density_class(cell(0.0), limits) == "no animals"
        assert density_class(cell(20.0), limits) == "0 - 20 /km²"
        assert density_class(cell(20.1), limits) == "20 - 40 /km²"
        assert density_class(cell(100.0), limits) == "40 - 100 /km²"
        assert density_class(cell(250.0), limits) == "40 - 100 /km²"
        assert density_class(cell(250.0, "too little seen"), limits) == \
            "too little seen"
        assert density_class(cell(None, "not seen"), limits) == "not seen"
        assert density_class(cell(5.0), []) == "no animals"

    def test_a_cell_is_in_the_class_of_what_stands_out(self):
        assert spot_class({"status": "counted", "spot": ""}) == \
            "nothing stands out"
        assert spot_class({"status": "counted",
                           "spot": "hot spot (95 %)"}) == "hot spot (95 %)"
        assert spot_class({"status": "not seen", "spot": ""}) == "not seen"

