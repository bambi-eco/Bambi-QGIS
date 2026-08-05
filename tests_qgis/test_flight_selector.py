# -*- coding: utf-8 -*-
"""The flight selector and flight-scoped configuration (§10.2).

The headless rules live in ``tests/test_flights.py``; what needs a real QGIS is
the wiring — that switching flights swaps the whole form, that configuration
lands in the flight's own store rather than the ``.qgz``, and that renaming a
flight renames its layer group instead of orphaning it.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.core import QgsProject
from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from bambi_wildlife_detection.core import flights


@pytest.fixture
def two_flights(tmp_path):
    a = str(tmp_path / "meadow" / "qgis")
    b = str(tmp_path / "forest" / "qgis")
    os.makedirs(a, exist_ok=True)
    os.makedirs(b, exist_ok=True)
    return a, b


@pytest.fixture
def quiet(monkeypatch):
    shown = {"info": [], "warn": []}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    return shown


def _add(dock, monkeypatch, folder, name):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: folder)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (name, True))
    dock.add_flight()


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------

def test_the_widgets_exist(dock):
    assert dock.flight_combo is not None
    assert dock.flight_add_btn is not None
    assert dock.flight_rename_btn is not None


def test_setting_a_target_folder_creates_the_first_flight(dock, two_flights):
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    assert len(dock._flights()) == 1
    assert dock._active_flight()["target_folder"] == a


def test_the_default_name_comes_from_the_parent_folder(dock, two_flights):
    """The default target folder is <input>/qgis, so 'qgis' names nothing."""
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    assert dock._active_flight()["name"] == "meadow"


def test_adding_a_second_flight_switches_to_it(dock, quiet, monkeypatch,
                                               two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    _add(dock, monkeypatch, b, "Forest")
    assert len(dock._flights()) == 2
    assert dock._active_flight()["name"] == "Forest"
    assert dock.target_folder_edit.text() == b


def test_two_flights_cannot_share_a_folder(dock, quiet, monkeypatch,
                                           two_flights):
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    _add(dock, monkeypatch, a, "Duplicate")
    assert len(dock._flights()) == 1
    assert quiet["warn"], "the clash must be explained, not silently accepted"


def test_the_selector_is_enabled_only_with_several_flights(dock, quiet,
                                                           monkeypatch,
                                                           two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    assert not dock.flight_combo.isEnabled()

    _add(dock, monkeypatch, b, "Forest")
    assert dock.flight_combo.isEnabled()


# ---------------------------------------------------------------------------
# Switching swaps the whole form
# ---------------------------------------------------------------------------

def test_switching_flights_swaps_the_configuration(dock, quiet, monkeypatch,
                                                   two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock.confidence_spin.setValue(0.25)

    _add(dock, monkeypatch, b, "Forest")
    dock.confidence_spin.setValue(0.75)

    dock.flight_combo.setCurrentIndex(0)
    assert dock.confidence_spin.value() == pytest.approx(0.25)
    assert dock.target_folder_edit.text() == a

    dock.flight_combo.setCurrentIndex(1)
    assert dock.confidence_spin.value() == pytest.approx(0.75)


def test_configuration_lands_in_the_flights_own_store(dock, two_flights):
    """The folder becomes self-describing (§10.2)."""
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock.confidence_spin.setValue(0.42)
    dock.save_config_to_project()

    stored = flights.load_config(a)
    assert stored["Detection/Confidence"] == "0.42"


def test_switching_is_blocked_while_a_step_runs(dock, quiet, monkeypatch,
                                                two_flights):
    """Otherwise the log would describe a flight you are no longer looking at."""
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")

    dock.current_worker = object()
    try:
        dock.flight_combo.setCurrentIndex(0)
        assert dock._active_flight()["name"] == "Forest"   # unchanged
        assert quiet["info"]
    finally:
        dock.current_worker = None


# ---------------------------------------------------------------------------
# Layer groups follow the flight
# ---------------------------------------------------------------------------

def test_layer_groups_nest_under_the_flight(dock, two_flights):
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    group = dock._create_layer_group("Detections (Thermal)")
    parent = group.parent()
    assert parent is not None
    assert parent.name() == "meadow"


def test_renaming_a_flight_renames_its_group(dock, quiet, monkeypatch,
                                             two_flights):
    """A stale group per rename would accumulate quickly (§10.2)."""
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock._create_layer_group("Detections (Thermal)")

    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *a, **k: ("North Meadow", True))
    dock.rename_flight()

    root = QgsProject.instance().layerTreeRoot()
    assert root.findGroup("North Meadow") is not None
    assert root.findGroup("meadow") is None


def test_the_renamed_group_keeps_its_children(dock, quiet, monkeypatch,
                                              two_flights):
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock._create_layer_group("Detections (Thermal)")

    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *a, **k: ("North Meadow", True))
    dock.rename_flight()

    group = QgsProject.instance().layerTreeRoot().findGroup("North Meadow")
    assert group.findGroup("Detections (Thermal)") is not None


def test_each_flight_gets_its_own_group(dock, quiet, monkeypatch, two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock._create_layer_group("Detections (Thermal)")

    _add(dock, monkeypatch, b, "Forest")
    dock._create_layer_group("Detections (Thermal)")

    root = QgsProject.instance().layerTreeRoot()
    assert root.findGroup("meadow") is not None
    assert root.findGroup("Forest") is not None


def test_renaming_onto_an_existing_name_is_refused(dock, quiet, monkeypatch,
                                                   two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("meadow", True))
    dock.rename_flight()
    assert quiet["warn"]
    assert dock._active_flight()["name"] == "Forest"


# ---------------------------------------------------------------------------
# Adding a flight to a multi-project analysis
# ---------------------------------------------------------------------------

def _flight_entries(dock, prefix):
    from qgis.PyQt.QtCore import Qt

    lst = getattr(dock, f"{prefix}_projects_list")
    return [lst.item(i).data(Qt.ItemDataRole.UserRole) or {}
            for i in range(lst.count())]


def test_the_add_flight_button_exists_on_both_tools(dock):
    assert dock.ds_add_flight_btn is not None
    assert dock.pop_add_flight_btn is not None


def test_adding_a_flight_needs_no_path(dock, quiet, monkeypatch, two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")
    dock.flight_combo.setCurrentIndex(0)          # back to the meadow

    monkeypatch.setattr(QInputDialog, "getItem",
                        lambda *a_, **k: ("Forest  —  " + b, True))
    dock._add_project_from_flight("ds", with_dem=False)

    assert [e.get("target") for e in _flight_entries(dock, "ds")] == [b]


def test_the_active_flight_is_not_offered(dock, quiet, monkeypatch,
                                          two_flights):
    """It is added separately by 'Add current project'."""
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    dock._add_project_from_flight("ds", with_dem=False)
    assert quiet["info"], "with no other flights the user should be told"


def test_the_dem_comes_from_the_flights_own_config(dock, quiet, monkeypatch,
                                                   two_flights, tmp_path):
    a, b = two_flights
    dem = str(tmp_path / "dem.json")
    open(dem, "w").close()

    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")
    dock.dem_metadata_path_edit.setText(dem)
    dock.save_config_to_project()
    dock.flight_combo.setCurrentIndex(0)

    monkeypatch.setattr(QInputDialog, "getItem",
                        lambda *a_, **k: ("Forest  —  " + b, True))
    dock._add_project_from_flight("pop", with_dem=True)

    entries = _flight_entries(dock, "pop")
    assert entries and entries[0].get("dem") == dem


def test_a_flight_without_a_dem_is_refused_where_one_is_needed(
        dock, quiet, monkeypatch, two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")
    dock.flight_combo.setCurrentIndex(0)

    monkeypatch.setattr(QInputDialog, "getItem",
                        lambda *a_, **k: ("Forest  —  " + b, True))
    dock._add_project_from_flight("pop", with_dem=True)

    assert quiet["warn"]
    assert not _flight_entries(dock, "pop")
