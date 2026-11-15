# -*- coding: utf-8 -*-
"""What has been done is what the store says.

Three steps and "Add Tracks to QGIS" still asked whether the folder of a
legacy text file existed - and "Add Tracks" whether ``tracks.csv`` did, a
file it read nothing from. The text files are only written while "Also
write legacy text outputs" is on, and never for animals that were labelled
by hand; a flight copied with its stores alone has none of the folders.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.core import QgsProject
from qgis.PyQt.QtWidgets import QDialog, QMessageBox

from bambi_wildlife_detection import bambi_dock_widget
from bambi_wildlife_detection.core import detection_store, track_store


@pytest.fixture
def stores_only(dock, tmp_path):
    """A flight with detections, geo-referencing and two tracks in the
    stores - and none of the legacy folders."""
    root = str(tmp_path / "flight")
    os.makedirs(root)
    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 10.0 + 40 * n, "y1": 20.0, "x2": 30.0 + 40 * n,
         "y2": 40.0, "confidence": 0.9, "source_class": "0"}
        for n in range(2) for frame in range(3)])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": det, "gx1": 500000.0 + 5 * i, "gy1": 5300000.0,
         "gz1": 0.0, "gx2": 500002.0 + 5 * i, "gy2": 5300002.0, "gz2": 0.0}
        for i, det in enumerate(ids)])
    track_store.record_tracks(root, "t", [
        {"track_id": 1 + i // 3, "detection_id": det}
        for i, det in enumerate(ids)])
    assert not any(name.startswith(("detections_", "georeferenced_",
                                    "tracks_")) for name in os.listdir(root))
    dock.target_folder_edit.setText(root)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    return root


@pytest.fixture
def quiet(monkeypatch):
    shown = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, **k: shown.append(a))
    return shown


class _WholeRange:
    """The frame-range dialog, answered with "everything"."""

    def __init__(self, _parent, first, last, *_rest):
        self._range = (first, last)

    def exec(self):
        return QDialog.DialogCode.Accepted

    def selected_range(self):
        return self._range


def _track_groups():
    root = QgsProject.instance().layerTreeRoot()
    return sorted(group.name() for group in root.findGroups(True)
                  if group.name().startswith("Track "))


def test_tracks_are_added_without_any_legacy_file(dock, stores_only, quiet,
                                                  monkeypatch):
    monkeypatch.setattr(bambi_dock_widget, "FrameRangeDialog", _WholeRange)
    dock.add_tracks_to_qgis()

    assert quiet == []
    assert _track_groups() == ["Track 1", "Track 2"]


def test_csv_files_in_the_legacy_folder_do_not_multiply_the_tracks(
        dock, stores_only, quiet, monkeypatch):
    """Every .csv there stood for the whole store."""
    folder = os.path.join(stores_only, "tracks_t")
    os.makedirs(folder)
    for name in ("tracks.csv", "tracks_backup.csv"):
        with open(os.path.join(folder, name), "w") as handle:
            handle.write("# old\n")
    monkeypatch.setattr(bambi_dock_widget, "FrameRangeDialog", _WholeRange)
    dock.add_tracks_to_qgis()

    assert _track_groups() == ["Track 1", "Track 2"]


def test_without_tracks_the_action_says_so(dock, quiet, tmp_path):
    root = str(tmp_path / "empty")
    os.makedirs(os.path.join(root, "tracks_t"))    # the folder alone is not it
    dock.target_folder_edit.setText(root)
    dock._on_target_folder_changed()
    dock.add_tracks_to_qgis()
    assert quiet and "tracking has not been completed" in str(quiet[0])


@pytest.mark.parametrize("action, step", [
    ("run_georeference", "georeference"),
    ("run_tracking", "tracking"),
])
def test_a_step_starts_from_the_stores_alone(dock, stores_only, quiet,
                                             monkeypatch, action, step):
    started = []
    monkeypatch.setattr(type(dock), "start_worker",
                        lambda self, name, *a, **k: started.append(name))
    monkeypatch.setattr(type(dock), "validate_inputs",
                        lambda self, *a, **k: True)
    getattr(dock, action)()

    assert quiet == []
    assert started == [step]


def test_the_track_distances_ask_for_tracks_in_the_store(dock, stores_only,
                                                         quiet, monkeypatch):
    """... and still for the flight route, which this flight has not."""
    started = []
    monkeypatch.setattr(type(dock), "start_worker",
                        lambda self, name, *a, **k: started.append(name))
    dock.run_track_perpendicular()

    assert started == []
    text = str(quiet[0])
    assert "Flight route" in text and "Tracked animals" not in text


# ---------------------------------------------------------------------------
# "Add ... to QGIS" twice shows the product once
# ---------------------------------------------------------------------------

def _tree(node=None, depth=0):
    """The layer tree as indented names, "G" for a group and "L" for a
    layer."""
    from qgis.core import QgsLayerTree

    node = node or QgsProject.instance().layerTreeRoot()
    lines = []
    for child in node.children():
        is_group = QgsLayerTree.isGroup(child)
        lines.append("  " * depth + ("G " if is_group else "L ") + child.name())
        if is_group:
            lines.extend(_tree(child, depth + 1))
    return lines


def _raster(path):
    """A small GeoTIFF QGIS can open."""
    from osgeo import gdal, osr

    dataset = gdal.GetDriverByName("GTiff").Create(path, 4, 4, 1,
                                                   gdal.GDT_Byte)
    dataset.SetGeoTransform((500000.0, 1.0, 0.0, 5300004.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).Fill(7)
    dataset.FlushCache()
    dataset = None
    return path


def test_adding_the_tracks_twice_shows_them_once(dock, stores_only, quiet,
                                                 monkeypatch):
    """A second group of the same name appeared - and the first one stood
    empty beside it, its layers gone with the files that were rewritten."""
    monkeypatch.setattr(bambi_dock_widget, "FrameRangeDialog", _WholeRange)
    dock.add_tracks_to_qgis()
    once = _tree()
    dock.add_tracks_to_qgis()

    assert _tree() == once
    assert once.count("    G BAMBI Wildlife Tracks (Thermal)") == 1
    assert "        L Path" in once


def test_adding_the_frame_detections_twice_shows_them_once(
        dock, stores_only, quiet, monkeypatch):
    monkeypatch.setattr(bambi_dock_widget, "FrameRangeDialog", _WholeRange)
    dock.add_frame_detections_to_qgis()
    once = _tree()
    dock.add_frame_detections_to_qgis()

    assert _tree() == once
    assert sum("BAMBI Frame Detections" in line for line in once) == 1
    assert sum(line.strip().startswith("L Detections_Frame")
               for line in once) == 3


def test_another_range_replaces_the_one_shown(dock, stores_only, quiet,
                                              monkeypatch):
    """The group shows what was asked for last."""
    class FirstFrameOnly(_WholeRange):
        def selected_range(self):
            return (self._range[0], self._range[0])

    monkeypatch.setattr(bambi_dock_widget, "FrameRangeDialog", _WholeRange)
    dock.add_frame_detections_to_qgis()
    monkeypatch.setattr(bambi_dock_widget, "FrameRangeDialog", FirstFrameOnly)
    dock.add_frame_detections_to_qgis()

    shown = [line.strip() for line in _tree()
             if line.strip().startswith("L Detections_Frame")]
    assert shown == ["L Detections_Frame_000000"]


def test_a_layer_added_again_replaces_itself(dock, stores_only, tmp_path):
    from qgis.core import QgsRasterLayer

    path = _raster(str(tmp_path / "ortho.tif"))
    first = QgsRasterLayer(path, "BAMBI Orthomosaic (Thermal)")
    assert first.isValid()
    dock._add_layer_to_flight(first)
    once = _tree()
    dock._add_layer_to_flight(
        QgsRasterLayer(path, "BAMBI Orthomosaic (Thermal)"))

    assert _tree() == once
    assert once.count("    L BAMBI Orthomosaic (Thermal)") == 1
    assert len(QgsProject.instance().mapLayers()) == 1


def test_the_same_layer_object_is_not_removed_from_under_itself(
        dock, stores_only, tmp_path):
    from qgis.core import QgsRasterLayer

    layer = QgsRasterLayer(_raster(str(tmp_path / "ortho.tif")),
                           "BAMBI Orthomosaic (Thermal)")
    dock._add_layer_to_flight(layer)
    assert len(QgsProject.instance().mapLayers()) == 1
    assert layer.isValid()


def test_the_other_cameras_layer_of_the_same_kind_stays(
        dock, stores_only, tmp_path):
    from qgis.core import QgsRasterLayer

    path = _raster(str(tmp_path / "ortho.tif"))
    dock._add_layer_to_flight(
        QgsRasterLayer(path, "BAMBI Orthomosaic (Thermal)"))
    dock._add_layer_to_flight(QgsRasterLayer(path, "BAMBI Orthomosaic (RGB)"))
    dock._add_layer_to_flight(
        QgsRasterLayer(path, "BAMBI Orthomosaic (Thermal)"))

    names = sorted(layer.name()
                   for layer in QgsProject.instance().mapLayers().values())
    assert names == ["BAMBI Orthomosaic (RGB)", "BAMBI Orthomosaic (Thermal)"]


def test_a_survey_layer_added_again_replaces_itself(dock, stores_only,
                                                    tmp_path):
    from qgis.core import QgsRasterLayer

    path = _raster(str(tmp_path / "density.tif"))
    for _ in range(2):
        dock._add_layer_to_survey(
            QgsRasterLayer(path, "BAMBI Density Tracks (Thermal)"))

    assert sum("BAMBI Density Tracks (Thermal)" in line
               for line in _tree()) == 1
    assert len(QgsProject.instance().mapLayers()) == 1


def test_a_users_own_layer_of_another_name_is_left_alone(
        dock, stores_only, quiet, monkeypatch, tmp_path):
    from qgis.core import QgsRasterLayer

    mine = QgsRasterLayer(_raster(str(tmp_path / "mine.tif")), "my notes")
    QgsProject.instance().addMapLayer(mine)
    monkeypatch.setattr(bambi_dock_widget, "FrameRangeDialog", _WholeRange)
    dock.add_tracks_to_qgis()
    dock.add_tracks_to_qgis()

    assert "L my notes" in [line.strip() for line in _tree()]


# ---------------------------------------------------------------------------
# The coverage map: a gap is a value, and has a colour of its own
# ---------------------------------------------------------------------------

def _coverage_raster(path, rows, nodata=65535):
    from osgeo import gdal, osr
    import numpy as np

    data = np.asarray(rows, dtype=np.uint16)
    dataset = gdal.GetDriverByName("GTiff").Create(
        path, data.shape[1], data.shape[0], 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform((500000.0, 1.0, 0.0, 5300000.0 + data.shape[0],
                             0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(data)
    dataset.FlushCache()
    dataset = None
    return path


def test_the_coverage_layer_shows_a_gap_as_not_seen(dock, stores_only,
                                                    tmp_path):
    from qgis.core import QgsColorRampShader, QgsRasterLayer

    path = _coverage_raster(str(tmp_path / "coverage_map.tif"),
                            [[0, 1, 2], [65535, 1, 3]])
    layer = QgsRasterLayer(path, "BAMBI Coverage Map (Thermal)")
    assert layer.isValid()

    dock._apply_coverage_style(layer)

    shader = layer.renderer().shader().rasterShaderFunction()
    assert shader.colorRampType() == QgsColorRampShader.Type.Discrete
    items = [(item.value, item.label) for item in shader.colorRampItemList()]
    assert items == [(0.0, "not seen"), (1.0, "1"), (2.0, "2"), (3.0, "3")]


def test_a_coverage_layer_without_gaps_has_no_such_class(dock, stores_only,
                                                         tmp_path):
    from qgis.core import QgsRasterLayer

    path = _coverage_raster(str(tmp_path / "coverage_map.tif"),
                            [[65535, 1, 2], [65535, 1, 2]])
    layer = QgsRasterLayer(path, "BAMBI Coverage Map (Thermal)")
    dock._apply_coverage_style(layer)

    labels = [item.label for item in layer.renderer().shader()
              .rasterShaderFunction().colorRampItemList()]
    assert labels == ["1", "2"]
