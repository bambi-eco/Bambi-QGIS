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


def _add(dock, monkeypatch, folder, name, copy_configuration=True):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: folder)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (name, True))
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: copy_configuration)
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


# ---------------------------------------------------------------------------
# What a new flight starts with
# ---------------------------------------------------------------------------

def test_a_new_flight_clears_the_recordings(dock, quiet, monkeypatch,
                                            two_flights):
    """Otherwise the previous flight's data is processed again (§10.2)."""
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock.airdata_path_edit.setText("/previous/airdata.csv")
    dock.dem_path_edit.setText("/previous/dem.glb")

    _add(dock, monkeypatch, b, "Forest")

    assert dock.airdata_path_edit.text() == ""
    assert dock.dem_path_edit.text() == ""


def test_copying_keeps_the_processing_settings(dock, quiet, monkeypatch,
                                               two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock.confidence_spin.setValue(0.33)

    _add(dock, monkeypatch, b, "Forest", copy_configuration=True)
    assert dock.confidence_spin.value() == pytest.approx(0.33)


def test_defaults_reset_the_processing_settings(dock, quiet, monkeypatch,
                                                two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock.confidence_spin.setValue(0.33)

    _add(dock, monkeypatch, b, "Forest", copy_configuration=False)
    assert dock.confidence_spin.value() != pytest.approx(0.33)


def test_cancelling_the_configuration_prompt_adds_nothing(dock, quiet,
                                                          monkeypatch,
                                                          two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    _add(dock, monkeypatch, b, "Forest", copy_configuration=None)
    assert len(dock._flights()) == 1
    assert dock.target_folder_edit.text() == a


def test_the_new_flight_keeps_its_own_target_folder(dock, quiet, monkeypatch,
                                                    two_flights):
    """Clearing the inputs must not clear the folder that was just chosen."""
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    _add(dock, monkeypatch, b, "Forest")
    assert dock.target_folder_edit.text() == b


def test_the_prompt_offers_three_choices(dock):
    """Copy, default, cancel — cancel must not read as "use defaults"."""
    import inspect

    source = inspect.getsource(type(dock)._ask_new_flight_configuration)
    assert "copy configurations" in source
    assert "default configurations" in source
    assert "Cancel" in source


# ---------------------------------------------------------------------------
# Removing a flight
# ---------------------------------------------------------------------------

def _confirm(monkeypatch, dock, answer):
    monkeypatch.setattr(type(dock), "_confirm_remove_flight",
                        lambda self, flight: answer)


def test_removing_takes_the_flight_out_of_the_project(dock, quiet, monkeypatch,
                                                      two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")

    _confirm(monkeypatch, dock, True)
    dock.remove_flight()

    assert [f["name"] for f in dock._flights()] == ["meadow"]


def test_removing_needs_the_confirmation(dock, quiet, monkeypatch,
                                         two_flights):
    """The pop-up is the whole safeguard, so declining must change nothing."""
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")

    _confirm(monkeypatch, dock, False)
    dock.remove_flight()

    assert len(dock._flights()) == 2
    assert dock.target_folder_edit.text() == b


def test_removing_leaves_the_target_folder_on_disk(dock, quiet, monkeypatch,
                                                   two_flights):
    """A flight is days of processing; removing it from a project says
    nothing about the outputs."""
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")
    dock.save_config_to_project()
    assert flights.has_config(b)

    _confirm(monkeypatch, dock, True)
    dock.remove_flight()

    assert os.path.isdir(b)
    assert flights.has_config(b)


def test_the_removed_flights_folder_can_be_added_again(dock, quiet, monkeypatch,
                                                       two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")
    _confirm(monkeypatch, dock, True)
    dock.remove_flight()

    _add(dock, monkeypatch, b, "Forest again")
    assert len(dock._flights()) == 2


def test_removing_switches_to_the_remaining_flight(dock, quiet, monkeypatch,
                                                   two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")

    _confirm(monkeypatch, dock, True)
    dock.remove_flight()

    assert dock._active_flight()["target_folder"] == a
    assert dock.target_folder_edit.text() == a


def test_removing_takes_the_layer_group_with_it(dock, quiet, monkeypatch,
                                                two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")
    root = QgsProject.instance().layerTreeRoot()
    root.addGroup("Forest")

    _confirm(monkeypatch, dock, True)
    dock.remove_flight()

    assert root.findGroup("Forest") is None


def test_the_other_flights_group_is_left_alone(dock, quiet, monkeypatch,
                                               two_flights):
    a, b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    _add(dock, monkeypatch, b, "Forest")
    root = QgsProject.instance().layerTreeRoot()
    root.addGroup("meadow")
    root.addGroup("Forest")

    _confirm(monkeypatch, dock, True)
    dock.remove_flight()

    assert root.findGroup("meadow") is not None


def test_removing_the_last_flight_clears_the_target_folder(dock, quiet,
                                                           monkeypatch,
                                                           two_flights):
    """Otherwise the next save would write straight back into the folder
    that was just removed."""
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()

    _confirm(monkeypatch, dock, True)
    dock.remove_flight()

    assert dock._flights() == []
    assert dock.target_folder_edit.text() == ""


def test_removing_is_refused_while_a_step_runs(dock, quiet, monkeypatch,
                                               two_flights):
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    dock.current_worker = object()
    _confirm(monkeypatch, dock, True)
    try:
        dock.remove_flight()
    finally:
        dock.current_worker = None

    assert len(dock._flights()) == 1
    assert quiet["info"]


def test_removing_nothing_says_so(dock, quiet, monkeypatch):
    _confirm(monkeypatch, dock, True)
    dock.remove_flight()
    assert quiet["info"]


def test_the_button_is_disabled_without_a_flight(dock):
    assert dock.flight_remove_btn.isEnabled() is False


def test_the_button_is_enabled_once_a_flight_exists(dock, two_flights):
    a, _b = two_flights
    dock.target_folder_edit.setText(a)
    dock._on_target_folder_changed()
    assert dock.flight_remove_btn.isEnabled() is True


def test_the_confirmation_states_that_nothing_is_deleted(dock):
    """The dialog is the only place a user learns the folder survives."""
    import inspect

    source = inspect.getsource(type(dock)._confirm_remove_flight)
    assert "Nothing on disk is deleted" in source
    assert "Cancel" in source
