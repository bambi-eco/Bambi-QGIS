# -*- coding: utf-8 -*-
"""A folder's stored settings are loaded - never written over.

A flight keeps its configuration in its target folder, so a folder that has
been processed before brings its settings with it. That held for a flight
added with "Browse and Add..." and for one just added by name. It did not
hold where a folder is most often entered:

* the **first flight** of a project, created by typing or browsing the target
  folder (or by "+", with the folder already in the field): the flight kept
  the form's values, and
* a flight **pointed at another folder**: it kept the values of the folder it
  left,

and the next save - which saving the QGIS project is - replaced what the
folder had stored with them. The recordings, the calibration, the correction
and every threshold of the processed flight were gone.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.core import flights, store

STORED = {"Detection/Confidence": "0.42",
          "Input/AirdataPath": "/data/forest/airdata.csv"}


@pytest.fixture
def folders(tmp_path):
    """``a`` is new; ``b`` has been processed and has its settings stored;
    ``c`` is new as well."""
    made = []
    for name in ("meadow", "forest", "lake"):
        folder = str(tmp_path / name / "qgis")
        os.makedirs(folder)
        made.append(folder)
    a, b, c = made
    flights.save_config(b, dict(STORED, **{"Input/TargetFolder": b}))
    return a, b, c


@pytest.fixture
def quiet(monkeypatch):
    shown = {"info": [], "warn": []}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    return shown


@pytest.fixture
def asked(dock, monkeypatch):
    """The question about a processed folder: what it was asked with, and
    the answer to give (``asked.answer``, yes by default)."""
    class Asked(list):
        answer = True

    calls = Asked()

    def confirm(self, folder, existing, leaving=""):
        calls.append((folder, leaving))
        return calls.answer

    monkeypatch.setattr(type(dock), "_confirm_existing_flight", confirm)
    return calls


def _enter(dock, folder):
    dock.target_folder_edit.setText(folder)
    dock._on_target_folder_changed()


def _save_the_qgis_project(dock):
    dock._on_project_write(None)


def _stored(folder):
    values = flights.load_config(folder)
    return {key: values.get(key) for key in STORED}


# ---------------------------------------------------------------------------
# The first flight of a project
# ---------------------------------------------------------------------------

def test_the_first_flight_takes_a_processed_folder_as_it_stands(
        dock, quiet, asked, folders):
    _a, b, _c = folders
    _enter(dock, b)

    assert asked == [(b, "")]
    assert dock._active_flight()["target_folder"] == b
    assert dock.confidence_spin.value() == pytest.approx(0.42)
    assert dock.airdata_path_edit.text() == "/data/forest/airdata.csv"

    _save_the_qgis_project(dock)
    assert _stored(b) == STORED


def test_declining_leaves_no_flight_and_the_folder_as_it_was(
        dock, quiet, asked, folders):
    _a, b, _c = folders
    asked.answer = False
    _enter(dock, b)

    assert dock._flights() == []
    assert dock.target_folder_edit.text() == ""
    _save_the_qgis_project(dock)
    assert _stored(b) == STORED


def test_a_folder_no_flight_has_yet_is_not_written_to(
        dock, quiet, asked, monkeypatch, folders):
    """No name, so no flight - the folder is merely in the field. Saving the
    QGIS project wrote the form into it all the same."""
    _a, b, _c = folders
    monkeypatch.setattr(type(dock), "_prompt_flight_name",
                        lambda self, folder: "")
    _enter(dock, b)
    assert dock._flights() == []
    assert dock.target_folder_edit.text() == b

    _save_the_qgis_project(dock)
    assert _stored(b) == STORED


def test_plus_takes_a_processed_folder_as_it_stands(
        dock, quiet, asked, monkeypatch, folders):
    """"+" with the folder already in the field saved the form into it at
    once."""
    _a, b, _c = folders
    monkeypatch.setattr(type(dock), "_prompt_flight_name",
                        lambda self, folder: "")
    _enter(dock, b)
    dock.flight_name_edit.setText("Forest")
    dock.add_flight()

    assert asked == [(b, "")]
    assert dock._active_flight() == {"name": "Forest", "target_folder": b}
    assert dock.confidence_spin.value() == pytest.approx(0.42)
    assert dock.flight_name_edit.text() == ""
    assert _stored(b) == STORED


def test_plus_declined_adds_nothing(dock, quiet, asked, monkeypatch, folders):
    _a, b, _c = folders
    monkeypatch.setattr(type(dock), "_prompt_flight_name",
                        lambda self, folder: "")
    _enter(dock, b)
    dock.flight_name_edit.setText("Forest")
    asked.answer = False
    dock.add_flight()

    assert dock._flights() == []
    assert dock.flight_name_edit.text() == "Forest"
    assert _stored(b) == STORED


def test_a_new_folder_is_not_asked_about(dock, quiet, asked, folders):
    a, _b, _c = folders
    dock.confidence_spin.setValue(0.9)
    _enter(dock, a)

    assert asked == []
    assert dock._active_flight()["target_folder"] == a
    # the configuration has its home in the flight's folder at once
    assert flights.load_config(a)["Detection/Confidence"] == "0.9"


def test_the_recordings_folder_brings_its_processed_flight(
        dock, quiet, asked, tmp_path):
    """Choosing the videos fills in ``<their folder>/qgis`` as the target
    folder - the commonest way a processed folder gets into the field."""
    videos = str(tmp_path / "flight")
    target = os.path.join(videos, "qgis")
    os.makedirs(target)
    flights.save_config(target, dict(STORED, **{"Input/TargetFolder": target}))

    dock._auto_detect_common_files(videos)

    assert asked == [(target, "")]
    assert dock._active_flight()["target_folder"] == target
    assert dock.confidence_spin.value() == pytest.approx(0.42)
    _save_the_qgis_project(dock)
    assert _stored(target) == STORED


# ---------------------------------------------------------------------------
# Pointing a flight at another folder
# ---------------------------------------------------------------------------

def test_pointing_a_flight_at_a_processed_folder_loads_its_settings(
        dock, quiet, asked, folders):
    a, b, _c = folders
    _enter(dock, a)
    dock.confidence_spin.setValue(0.9)

    _enter(dock, b)

    assert asked == [(b, a)]
    assert dock._active_flight()["target_folder"] == b
    assert dock.target_folder_edit.text() == b
    assert dock.confidence_spin.value() == pytest.approx(0.42)

    _save_the_qgis_project(dock)
    assert _stored(b) == STORED


def test_what_the_form_showed_stays_with_the_folder_that_was_left(
        dock, quiet, asked, folders):
    a, b, _c = folders
    _enter(dock, a)
    dock.confidence_spin.setValue(0.9)     # not saved by anything yet

    _enter(dock, b)

    left = flights.load_config(a)
    assert left["Detection/Confidence"] == "0.9"
    assert left["Input/TargetFolder"] == a


def test_declining_keeps_the_flight_where_it_was(dock, quiet, asked, folders):
    a, b, _c = folders
    _enter(dock, a)
    dock.confidence_spin.setValue(0.9)
    asked.answer = False

    _enter(dock, b)

    assert dock._active_flight()["target_folder"] == a
    assert dock.target_folder_edit.text() == a
    assert dock.confidence_spin.value() == pytest.approx(0.9)
    _save_the_qgis_project(dock)
    assert _stored(b) == STORED
    assert flights.load_config(a)["Detection/Confidence"] == "0.9"


def test_a_flight_moves_to_a_new_folder_with_its_settings(
        dock, quiet, asked, folders):
    a, _b, c = folders
    _enter(dock, a)
    dock.confidence_spin.setValue(0.9)

    _enter(dock, c)

    assert asked == []
    assert dock._active_flight()["target_folder"] == c
    assert dock.confidence_spin.value() == pytest.approx(0.9)
    moved = flights.load_config(c)
    assert moved["Detection/Confidence"] == "0.9"
    assert moved["Input/TargetFolder"] == c


def test_results_without_settings_keep_the_form(dock, quiet, asked, folders):
    """A folder with result stores and no stored configuration - a migrated
    5.x project, say: there is nothing to load and nothing to write over."""
    a, _b, c = folders
    store.open_store(store.stage_path(c, store.TRACKS, "t"),
                     store.TRACKS, "t").close()
    _enter(dock, a)
    dock.confidence_spin.setValue(0.9)

    _enter(dock, c)

    assert asked == [(c, a)]
    assert dock._active_flight()["target_folder"] == c
    assert dock.confidence_spin.value() == pytest.approx(0.9)
    assert flights.load_config(c)["Detection/Confidence"] == "0.9"


def test_browsing_for_the_folder_takes_the_same_path(
        dock, quiet, asked, monkeypatch, folders):
    from qgis.PyQt.QtWidgets import QFileDialog

    a, b, _c = folders
    _enter(dock, a)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: b)
    dock.browse_target_folder()

    assert asked == [(b, a)]
    assert dock.confidence_spin.value() == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# The folder is the flight's identity
# ---------------------------------------------------------------------------

def test_a_loaded_flight_shows_the_folder_it_was_read_from(
        dock, quiet, asked, folders):
    """The path stored with the settings is the one the folder had when
    they were saved - another one once it has been moved or handed on. The
    field showed that one, and every step then ran in the old place."""
    import json

    from qgis.core import QgsProject

    from bambi_wildlife_detection.bambi_dock_widget import PLUGIN_SCOPE

    _a, b, _c = folders
    flights.save_config(b, dict(STORED, **{
        "Input/TargetFolder": "/somewhere/it/was/before"}))
    project = QgsProject.instance()
    project.writeEntry(PLUGIN_SCOPE, "Flights/List", json.dumps(
        [{"name": "Forest", "target_folder": b}]))
    project.writeEntry(PLUGIN_SCOPE, "Flights/Active", "0")

    dock._on_project_read(None)

    assert dock._active_flight()["target_folder"] == b
    assert dock.confidence_spin.value() == pytest.approx(0.42)
    assert dock.target_folder_edit.text() == b
    assert dock.get_config()["target_folder"] == b


def test_switching_shows_each_flights_own_folder(dock, quiet, asked,
                                                 monkeypatch, folders):
    from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog

    a, b, _c = folders
    flights.save_config(b, dict(STORED, **{
        "Input/TargetFolder": "/somewhere/it/was/before"}))
    _enter(dock, a)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: b)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Forest", True))
    dock.browse_and_add_flight()
    assert dock.target_folder_edit.text() == b

    dock.flight_combo.setCurrentIndex(0)
    assert dock.target_folder_edit.text() == a
    dock.flight_combo.setCurrentIndex(1)
    assert dock.target_folder_edit.text() == b


# ---------------------------------------------------------------------------
# A step does not start on settings nobody has looked at
# ---------------------------------------------------------------------------

def test_a_step_stops_when_the_settings_were_just_loaded(
        dock, quiet, asked, monkeypatch, folders):
    """The flight is named - and its folder taken over - when the first step
    is started, if not before."""
    _a, b, _c = folders
    dock.target_folder_edit.setText(b)        # in the field, no flight yet

    assert dock.validate_inputs([]) is False
    assert dock._active_flight()["target_folder"] == b
    assert any("stored configuration" in str(args).lower()
               for args in quiet["info"])
    # the second time there is nothing left to load
    assert dock.validate_inputs([]) is True


def test_a_step_starts_in_a_new_folder(dock, quiet, asked, folders):
    a, _b, _c = folders
    dock.target_folder_edit.setText(a)

    assert dock.validate_inputs([]) is True
    assert dock._active_flight()["target_folder"] == a


def test_a_declined_folder_starts_nothing(dock, quiet, asked, folders):
    _a, b, _c = folders
    asked.answer = False
    dock.target_folder_edit.setText(b)

    assert dock.validate_inputs([]) is False
    assert dock._flights() == []
    assert dock.target_folder_edit.text() == ""


def test_a_refused_folder_starts_nothing(dock, quiet, asked, monkeypatch,
                                         folders):
    """The folder in the field belongs to another flight: the field goes
    back to this flight's own - and the step used to start there."""
    from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog

    a, _b, c = folders
    _enter(dock, a)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: c)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Lake", True))
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: True)
    dock.browse_and_add_flight()
    assert dock._active_flight()["target_folder"] == c

    dock.target_folder_edit.setText(a)        # typed, not yet confirmed

    assert dock.validate_inputs([]) is False
    assert quiet["warn"]
    assert dock._active_flight()["target_folder"] == c
    assert dock.target_folder_edit.text() == c


# ---------------------------------------------------------------------------
# A folder that cannot be read
# ---------------------------------------------------------------------------

def test_an_unreadable_folder_is_left_alone(dock, quiet, asked, folders):
    a, _b, c = folders
    with open(store.project_path(c), "w") as handle:
        handle.write("not a GeoPackage")
    _enter(dock, a)

    _enter(dock, c)

    assert quiet["warn"]
    assert dock._active_flight()["target_folder"] == a
    assert dock.target_folder_edit.text() == a
    with open(store.project_path(c)) as handle:
        assert handle.read() == "not a GeoPackage"


# ---------------------------------------------------------------------------
# The question
# ---------------------------------------------------------------------------

@pytest.fixture
def dialog(monkeypatch):
    """The confirmation as it is shown, answered with its default button."""
    shown = {}

    def run(box):
        shown["title"] = box.windowTitle()
        shown["text"] = box.informativeText()
        shown["buttons"] = [button.text() for button in box.buttons()]
        return 0

    monkeypatch.setattr(QMessageBox, "exec", run)
    return shown


def test_the_question_names_the_folder_being_left(
        dock, dialog, existing_folder_prompt, folders):
    a, b, _c = folders
    found = flights.describe_existing(b)

    existing_folder_prompt(dock, b, found, leaving=a)

    assert dialog["title"] == "Use existing folder"
    assert "Use this folder" in dialog["buttons"]
    assert "stored configuration is loaded" in dialog["text"]
    assert a in dialog["text"]
    assert "nothing is overwritten" in dialog["text"]


def test_the_question_for_a_new_flight_is_unchanged(
        dock, dialog, existing_folder_prompt, folders):
    _a, b, _c = folders
    existing_folder_prompt(dock, b, flights.describe_existing(b))

    assert dialog["title"] == "Add existing flight"
    assert "Add flight" in dialog["buttons"]
    assert "stay with" not in dialog["text"]


# ---------------------------------------------------------------------------
# A flight is never shown with another flight's settings
# ---------------------------------------------------------------------------

@pytest.fixture
def meadow_and_forest(dock, quiet, asked, monkeypatch, folders):
    """Two flights with settings of their own; "meadow" is the active one."""
    from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog

    a, _b, c = folders
    _enter(dock, a)
    dock.confidence_spin.setValue(0.9)
    dock.airdata_path_edit.setText("/data/meadow/airdata.csv")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: c)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Lake", True))
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: False)
    dock.browse_and_add_flight()
    dock.confidence_spin.setValue(0.3)
    dock.airdata_path_edit.setText("/data/lake/airdata.csv")
    dock.save_config_to_project()
    dock.flight_combo.setCurrentIndex(0)
    assert dock.confidence_spin.value() == pytest.approx(0.9)
    return a, c


def _defaults(dock):
    return (dock._pristine_config["Detection/Confidence"],
            dock._pristine_config["Input/AirdataPath"])


def test_a_flight_whose_store_cannot_be_read_shows_the_defaults(
        dock, meadow_and_forest):
    """The switch ended in an error half way: the selector said "Lake", the
    form kept the meadow's folder, flight log and thresholds."""
    _a, c = meadow_and_forest
    with open(store.project_path(c), "w") as handle:
        handle.write("not a GeoPackage")

    dock._on_flight_changed(1)

    assert dock._active_flight()["name"] == "Lake"
    assert dock.target_folder_edit.text() == c
    assert (dock.confidence_spin.value(), dock.airdata_path_edit.text()) == \
        pytest.approx(_defaults(dock))
    assert "could not be read" in dock.log_text.toPlainText()
    with open(store.project_path(c)) as handle:
        assert handle.read() == "not a GeoPackage"         # left alone


def test_a_flight_whose_folder_is_gone_shows_the_defaults(
        dock, meadow_and_forest):
    import shutil

    _a, c = meadow_and_forest
    shutil.rmtree(c)

    dock._on_flight_changed(1)

    assert dock.target_folder_edit.text() == c
    assert (dock.confidence_spin.value(), dock.airdata_path_edit.text()) == \
        pytest.approx(_defaults(dock))
    assert "is not there" in dock.log_text.toPlainText()


def test_the_other_flight_is_untouched_by_it(dock, meadow_and_forest):
    import shutil

    a, c = meadow_and_forest
    shutil.rmtree(c)
    dock._on_flight_changed(1)
    _save_the_qgis_project(dock)

    dock._on_flight_changed(0)

    assert dock.target_folder_edit.text() == a
    assert dock.confidence_spin.value() == pytest.approx(0.9)
    assert dock.airdata_path_edit.text() == "/data/meadow/airdata.csv"
    assert flights.load_config(a)["Detection/Confidence"] == "0.9"


def test_the_settings_are_back_when_the_folder_is(dock, meadow_and_forest,
                                                  tmp_path):
    import shutil

    _a, c = meadow_and_forest
    parked = str(tmp_path / "parked")
    shutil.move(c, parked)                   # the drive is unplugged ...
    dock._on_flight_changed(1)
    _save_the_qgis_project(dock)
    dock._on_flight_changed(0)
    shutil.move(parked, c)                   # ... and connected again

    dock._on_flight_changed(1)

    assert dock.confidence_spin.value() == pytest.approx(0.3)
    assert dock.airdata_path_edit.text() == "/data/lake/airdata.csv"


def test_settings_kept_in_the_project_belong_to_the_flight_that_left_them(
        dock, quiet, asked, monkeypatch, folders):
    """A flight waiting for its folder keeps its settings in the QGIS
    project. A flight whose folder was missing was shown - and then given -
    whatever lay there."""
    import shutil

    a, _b, c = folders
    _enter(dock, a)
    dock.confidence_spin.setValue(0.9)
    # a second flight with a folder, then a third still waiting for its own
    from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: c)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Lake", True))
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: True)
    dock.browse_and_add_flight()
    dock.confidence_spin.setValue(0.3)
    dock.flight_name_edit.setText("Ridge")
    dock.add_flight()
    dock.confidence_spin.setValue(0.77)            # Ridge's own
    dock.flight_combo.setCurrentIndex(0)           # Ridge's go to the project

    shutil.rmtree(c)                               # Lake's folder is gone
    dock.flight_combo.setCurrentIndex(1)
    assert dock._active_flight()["name"] == "Lake"
    assert dock.confidence_spin.value() == pytest.approx(
        dock._pristine_config["Detection/Confidence"])

    dock.flight_combo.setCurrentIndex(2)
    assert dock._active_flight()["name"] == "Ridge"
    assert dock.confidence_spin.value() == pytest.approx(0.77)


def test_renaming_the_waiting_flight_keeps_its_settings(
        dock, quiet, asked, monkeypatch, folders):
    a, _b, _c = folders
    _enter(dock, a)
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: True)
    dock.flight_name_edit.setText("Ridge")
    dock.add_flight()
    dock.confidence_spin.setValue(0.77)
    dock.save_config_to_project()
    assert dock._rename_active_flight("High Ridge")

    dock.flight_combo.setCurrentIndex(0)
    dock.flight_combo.setCurrentIndex(1)

    assert dock._active_flight()["name"] == "High Ridge"
    assert dock.confidence_spin.value() == pytest.approx(0.77)


def test_browse_and_add_says_when_a_folder_cannot_be_read(
        dock, quiet, asked, monkeypatch, folders):
    from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog

    a, _b, c = folders
    with open(store.project_path(c), "w") as handle:
        handle.write("not a GeoPackage")
    _enter(dock, a)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: c)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Lake", True))

    dock.browse_and_add_flight()

    assert len(dock._flights()) == 1
    assert any("cannot be read" in str(args) for args in quiet["warn"])


# ---------------------------------------------------------------------------
# Projects saved by earlier versions
# ---------------------------------------------------------------------------

def _plain_entries(dock, **changed):
    """The form as plain entries of the QGIS project - where versions before
    6.0 kept every setting, and versions before 9.1 those of a flight that
    had no folder to keep them in."""
    from qgis.core import QgsProject

    from bambi_wildlife_detection.core.config_schema import save_config_entries
    from bambi_wildlife_detection.gui_utils import PLUGIN_SCOPE

    project = QgsProject.instance()
    values = dict(dock._current_config_values(), **changed)
    save_config_entries(
        values,
        lambda key, value: project.writeEntry(PLUGIN_SCOPE, key, value),
        lambda key, value: project.writeEntryDouble(PLUGIN_SCOPE, key, value),
        lambda key, value: project.writeEntryBool(PLUGIN_SCOPE, key, value))


def _flight_list(flight_list, active=0):
    import json

    from qgis.core import QgsProject

    from bambi_wildlife_detection.gui_utils import PLUGIN_SCOPE

    project = QgsProject.instance()
    project.writeEntry(PLUGIN_SCOPE, "Flights/List", json.dumps(flight_list))
    project.writeEntry(PLUGIN_SCOPE, "Flights/Active", str(active))


def test_a_project_from_before_6_0_brings_its_settings_into_the_folder(
        dock, quiet, asked, folders):
    a, _b, _c = folders
    _plain_entries(dock, **{"Input/TargetFolder": a,
                            "Detection/Confidence": 0.61})

    dock._on_project_read(None)

    assert dock._active_flight()["target_folder"] == a
    assert dock.confidence_spin.value() == pytest.approx(0.61)
    assert flights.load_config(a)["Detection/Confidence"] == "0.61"


def test_a_flight_still_waiting_for_its_folder_takes_the_plain_entries(
        dock, quiet, asked, folders):
    """A 9.0 project: the second flight was added by name, and its settings
    were kept in the project's plain entries."""
    a, _b, _c = folders
    flights.save_config(a, {"Input/TargetFolder": a,
                            "Detection/Confidence": "0.9"})
    _plain_entries(dock, **{"Input/TargetFolder": "",
                            "Detection/Confidence": 0.33})
    _flight_list([{"name": "Meadow", "target_folder": a},
                  {"name": "Ridge", "target_folder": ""}], active=1)

    dock._on_project_read(None)

    assert dock._active_flight()["name"] == "Ridge"
    assert dock.confidence_spin.value() == pytest.approx(0.33)


def test_plain_entries_are_not_given_to_one_flight_among_several(
        dock, quiet, asked, folders, tmp_path):
    """They are the last flight's that had no folder - not those of a flight
    whose folder happens to be missing. It was shown them, and they were
    then "moved" into its folder as its own."""
    a, _b, _c = folders
    flights.save_config(a, {"Input/TargetFolder": a,
                            "Detection/Confidence": "0.9"})
    away = str(tmp_path / "not" / "connected")
    _plain_entries(dock, **{"Input/TargetFolder": "",
                            "Detection/Confidence": 0.33})
    _flight_list([{"name": "Meadow", "target_folder": a},
                  {"name": "Lake", "target_folder": away}], active=1)

    dock._on_project_read(None)

    assert dock._active_flight()["name"] == "Lake"
    assert dock.target_folder_edit.text() == away
    assert dock.confidence_spin.value() == pytest.approx(
        dock._pristine_config["Detection/Confidence"])
    assert not os.path.exists(away)


def test_a_project_without_a_flight_keeps_what_was_entered(
        dock, quiet, asked, monkeypatch, folders):
    """No name yet, so no flight - the form is kept with the project all
    the same, and comes back with it."""
    monkeypatch.setattr(type(dock), "_prompt_flight_name",
                        lambda self, folder: "")
    a, _b, _c = folders
    _enter(dock, a)
    dock.confidence_spin.setValue(0.61)
    _save_the_qgis_project(dock)
    dock.confidence_spin.setValue(0.2)
    dock.target_folder_edit.setText("")

    dock._on_project_read(None)

    assert dock._flights() == []
    assert dock.target_folder_edit.text() == a
    assert dock.confidence_spin.value() == pytest.approx(0.61)
    assert not os.path.isfile(store.project_path(a))      # still not written to
