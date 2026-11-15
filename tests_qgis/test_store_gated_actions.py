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
