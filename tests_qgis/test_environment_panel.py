# -*- coding: utf-8 -*-
"""The environmental-context UI inside a real QGIS: the four sub-steps and
their statuses, the layer group, the settings round trip and the habitat
charts.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import json
import os

import numpy as np
import pytest

from qgis.core import (
    QgsLayerTree, QgsPalettedRasterRenderer, QgsProject,
    QgsSingleBandPseudoColorRenderer,
)
from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.bambi_environment import ENVIRONMENT_STEPS
from bambi_wildlife_detection.bambi_environment_panel import ENVIRONMENT_GROUP
from bambi_wildlife_detection.core import environment as env


@pytest.fixture
def quiet(monkeypatch):
    shown = {"warn": [], "critical": []}
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *a, **k: shown["critical"].append(a))
    return shown


@pytest.fixture
def flight(dock, quiet, tmp_path):
    """A flight with poses, a DEM origin and three environment sources."""
    root = str(tmp_path / "meadow")
    os.makedirs(root)
    with open(os.path.join(root, "poses_t.json"), "w") as fh:
        json.dump({"images": [
            {"imagefile": f"{i}.jpg", "location": [i * 20.0, 0.0, 80.0],
             "timestamp": f"2026-06-10T03:1{i}:00"} for i in range(5)]}, fh)
    with open(os.path.join(root, "dem.json"), "w") as fh:
        json.dump({"origin": [447000.0, 5350500.0, 260.0]}, fh)
    grid = env.make_grid((447000.0, 5350500.0, 447200.0, 5350600.0),
                         32633, 0.0)
    env.write_raster(env.layer_path(root, "ndvi"),
                     np.full(grid.shape, 0.6, dtype=np.float32), grid)
    classes = np.full(grid.shape, 30, dtype=np.uint8)
    classes[:, :5] = 10
    env.write_raster(env.layer_path(root, "landcover"), classes, grid,
                     nodata=0)
    env.write_raster(env.layer_path(root, "twi"), np.linspace(
        5.0, 12.0, grid.width * grid.height,
        dtype=np.float32).reshape(grid.shape), grid)
    env.write_metadata(root, env.SENTINEL2, {
        "files": ["ndvi.tif"], "scene": "S2X", "offset_days": -1.7,
        "scene_time": "2026-06-08T10:17:35Z", "clear_fraction": 0.96})
    env.write_metadata(root, env.LANDCOVER, {"files": ["landcover.tif"]})
    env.write_metadata(root, env.TERRAIN, {"files": ["twi.tif"]})

    dock.target_folder_edit.setText(root)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    return root


def test_one_button_and_one_status_per_source(dock):
    assert list(dock._environment_buttons) == list(ENVIRONMENT_STEPS)
    assert set(dock._environment_status_labels) == set(ENVIRONMENT_STEPS)
    for label in dock._environment_status_labels.values():
        assert label.text() == "⚪ Not started"
    assert dock.habitat_status.text() == "⚪ Not started"


def test_the_settings_reach_the_processor_config(dock):
    dock.environment_window_spin.setValue(21)
    dock.environment_clear_spin.setValue(60)
    dock.environment_buffer_spin.setValue(500)
    dock.environment_area_check.setChecked(False)
    dock.habitat_radius_spin.setValue(50.0)
    config = dock.get_config()
    assert config["environment_window_days"] == 21
    assert config["environment_min_clear"] == pytest.approx(0.6)
    assert config["environment_buffer_m"] == 500
    assert config["environment_include_survey_area"] is False
    assert config["environment_radius_m"] == 50.0


def test_the_settings_are_saved_with_the_project(dock):
    dock.environment_window_spin.setValue(21)
    dock.environment_clear_spin.setValue(60)
    dock.environment_area_check.setChecked(False)
    dock.habitat_radius_spin.setValue(50.0)
    dock.save_config_to_project()
    dock.environment_window_spin.setValue(3)
    dock.environment_clear_spin.setValue(99)
    dock.environment_area_check.setChecked(True)
    dock.habitat_radius_spin.setValue(10.0)
    dock.load_config_from_project()
    assert dock.environment_window_spin.value() == 21
    assert dock.environment_clear_spin.value() == 60
    assert not dock.environment_area_check.isChecked()
    assert dock.habitat_radius_spin.value() == 50.0


def test_statuses_follow_the_flights_folder(dock, flight):
    labels = dock._environment_status_labels
    assert labels["env_sentinel2"].text() == "🟢 2026-06-08 (-1.7 d)"
    assert "96% clear" in labels["env_sentinel2"].toolTip()
    assert labels["env_landcover"].text() == "🟢 Completed"
    assert labels["env_terrain"].text() == "🟢 Completed"
    assert labels["env_weather"].text() == "⚪ Not started"

    env.clear(flight, env.SENTINEL2)
    dock._refresh_environment_status(flight)
    assert labels["env_sentinel2"].text() == "⚪ Not started"
    assert labels["env_sentinel2"].toolTip() == ""


def test_a_step_that_cannot_run_says_why(dock, quiet, tmp_path):
    assert not dock.run_environment_step("env_weather")
    assert "target folder" in quiet["warn"][-1][2]

    root = str(tmp_path / "empty")
    os.makedirs(root)
    dock.target_folder_edit.setText(root)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    assert not dock.run_environment_step("env_sentinel2")
    message = quiet["warn"][-1][2]
    assert "extent is unknown" in message and "date is unknown" in message
    assert dock.worker is None


def test_run_all_goes_on_past_a_source_that_cannot_start(dock, flight,
                                                         monkeypatch):
    started = []
    monkeypatch.setattr(type(dock), "start_worker",
                        lambda self, step: started.append(step))
    dock.run_environment_all()
    # start_worker is stubbed, so no worker ever exists: every runnable
    # source is tried in turn and none blocks the next.
    assert started == ["env_sentinel2", "env_landcover", "env_weather"]
    assert dock._environment_status_labels["env_terrain"].text() == \
        "⚪ Skipped"      # no DEM mesh selected
    assert dock._environment_queue == []


def test_a_finished_source_starts_the_next_queued_one(dock, flight,
                                                      monkeypatch):
    started = []
    monkeypatch.setattr(type(dock), "run_environment_step",
                        lambda self, step: started.append(step) or True)
    from qgis.PyQt.QtCore import QCoreApplication

    dock._environment_queue = ["env_landcover", "env_weather"]
    dock.on_worker_finished("env_sentinel2", False)     # it failed
    # not from inside the finished signal: the worker's thread has to stop
    assert started == []
    QCoreApplication.processEvents()
    assert started == ["env_landcover"]
    assert dock._environment_queue == ["env_weather"]
    dock.on_worker_finished("env_landcover", True)
    QCoreApplication.processEvents()
    assert started == ["env_landcover", "env_weather"]
    # the label shows what is on disk, not just "completed"
    assert dock._environment_status_labels["env_sentinel2"].text() == \
        "🟢 2026-06-08 (-1.7 d)"


def test_the_layers_go_into_one_group_below_the_cameras(dock, flight):
    flight_group = dock._flight_group()
    dock._insert_group(flight_group, "Thermal")
    dock._insert_group(flight_group, "RGB")
    dock.add_environment_to_qgis()

    names = [child.name() for child in flight_group.children()]
    assert names == ["Thermal", "RGB", ENVIRONMENT_GROUP]
    group = flight_group.findGroup(ENVIRONMENT_GROUP)
    assert group.isMutuallyExclusive()
    nodes = [node for node in group.children() if QgsLayerTree.isLayer(node)]
    assert [node.name() for node in nodes] == [
        "NDVI (greenness)", "Land cover", "Topographic wetness index"]
    assert [node.itemVisibilityChecked() for node in nodes] == [
        True, False, False]
    layers = {node.name(): node.layer() for node in nodes}
    assert all(layer.isValid() for layer in layers.values())
    assert isinstance(layers["NDVI (greenness)"].renderer(),
                      QgsSingleBandPseudoColorRenderer)
    assert isinstance(layers["Land cover"].renderer(),
                      QgsPalettedRasterRenderer)
    labels = {c.label for c in layers["Land cover"].renderer().classes()}
    assert {"Tree cover", "Grassland"} <= labels
    assert dock.add_environment_status.text() == "🟢 Added"

    # adding again replaces the group instead of doubling it
    dock.add_environment_to_qgis()
    assert [child.name() for child in flight_group.children()].count(
        ENVIRONMENT_GROUP) == 1
    assert len(QgsProject.instance().mapLayers()) == 3


def test_adding_without_layers_is_refused(dock, quiet, tmp_path):
    dock.add_environment_to_qgis()
    assert "no environment layers" in quiet["warn"][-1][2]


def _habitat_files(folder):
    from bambi_wildlife_detection.core import habitat

    tracks = [{"flight": "A", "track_id": i, "species": name,
               "x": 447050.0 + i, "y": 5350550.0, "crs": "EPSG:32633",
               "ndvi": 0.3 + 0.1 * i, "landcover": "Grassland",
               "landcover_share": 1.0, "air_temperature_c": 9.5}
              for i, name in enumerate(["roe deer", "roe deer", "hare"])]
    edges = [round(-0.2 + 0.06 * i, 4) for i in range(21)]
    summary = {
        "n_tracks": 3, "radius_m": 30.0, "modality": "t",
        "flights": [{"flight": "A", "n_tracks": 3, "missing": [],
                     "sources": {"sentinel2": {
                         "scene_time": "2026-06-08T10:17:35Z",
                         "offset_days": -1.7, "clear_fraction": 0.96},
                         "weather": {"at_flight": {"temperature_2m": 9.5},
                                     "previous_24h": {}}}}],
        "layers": {"ndvi": {
            "title": "NDVI (greenness)", "unit": "", "description": "",
            "bin_edges": edges, "available_counts": [5] * 20,
            "used_counts": habitat.used_histogram(tracks, "ndvi", edges),
            "available": {"n": 100, "mean": 0.4},
            "used": {"n": 3, "mean": 0.4}}},
        "landcover": {"n_tracks": 3, "classes": [
            {"name": "Grassland", "colour": [255, 255, 76],
             "available_share": 0.5, "n_tracks": 3, "used_share": 1.0,
             "selection_ratio": 2.0}]},
    }
    os.makedirs(folder, exist_ok=True)
    habitat.write_json(summary, os.path.join(folder, habitat.SUMMARY_JSON))
    habitat.write_json(tracks, os.path.join(folder, habitat.TRACKS_JSON))
    return summary, tracks


def test_the_charts_redraw_per_species(dock, flight):
    folder = dock.current_survey().analytics_folder
    _habitat_files(folder)
    dock.show_habitat_charts()
    dialog = dock._habitat_dialog
    assert dialog.chart_keys() == ["ndvi"]
    assert dialog.flight_table.rowCount() == 1
    assert dialog.flight_table.item(0, 2).text() == "2026-06-08 (-1.7)"
    assert dialog.flight_table.item(0, 4).text() == "9.5"
    assert dialog.species_note.text() == "3 track(s)"
    dialog.species_combo.setCurrentIndex(
        dialog.species_combo.findData("hare"))
    assert dialog.species_note.text() == "1 track(s)"
    chart = dialog._charts["ndvi"]
    assert sum(chart._used_counts) == 1
    assert chart._used_mean == pytest.approx(0.5)
    dialog.grab()       # paints every chart once
    dialog.close()


def test_charts_without_a_summary_are_refused(dock, flight, quiet):
    dock.show_habitat_charts()
    assert "Summarise Habitat" in quiet["warn"][-1][2]
    dock.add_habitat_to_qgis()
    assert "Summarise Habitat" in quiet["warn"][-1][2]


def test_habitat_tracks_become_a_survey_layer(dock, flight):
    from bambi_wildlife_detection.core import habitat

    _habitat_files(dock.current_survey().analytics_folder)
    dock._refresh_habitat_status()
    assert dock.habitat_status.text() == "🟢 Completed"
    dock.add_habitat_to_qgis()
    assert dock.add_habitat_status.text() == "🟢 Added"
    group = QgsProject.instance().layerTreeRoot().findGroup(dock.SURVEY_GROUP)
    (node,) = [n for n in group.findLayers()
               if "Habitat Tracks" in n.name()]
    layer = node.layer()
    assert layer.featureCount() == 3
    assert [f.name() for f in layer.fields()][-len(habitat.TRACK_COLUMNS):] \
        == list(habitat.TRACK_COLUMNS)
    values = sorted(f["ndvi"] for f in layer.getFeatures())
    assert values == pytest.approx([0.3, 0.4, 0.5])


def test_run_all_runs_the_sources_on_real_worker_threads(dock, flight,
                                                         monkeypatch):
    """The next source must not start before the previous worker's thread
    has stopped: replacing a QThread that is still running destroys it and
    takes QGIS down with it (9.1.0: a crash near the end of the first
    source of "Run All Four")."""
    from qgis.PyQt.QtCore import QCoreApplication, QEventLoop, QTimer

    ran = []

    def source(name):
        def run(self, config, progress_fn=None, log_fn=None,
                cancel_check=None):
            ran.append(name)
            if progress_fn:
                progress_fn(95)
        return run

    processor = type(dock.processor)
    for step in ENVIRONMENT_STEPS:
        monkeypatch.setattr(processor, f"run_{step}", source(step))

    dock.run_environment_all()
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(
        lambda: (dock.worker is None and not dock._environment_queue
                 and loop.quit()))
    timer.start(20)
    QTimer.singleShot(15000, loop.quit)
    loop.exec()
    timer.stop()
    QCoreApplication.processEvents()

    assert ran == ["env_sentinel2", "env_landcover", "env_weather"]
    assert dock.worker is None and dock._environment_queue == []
