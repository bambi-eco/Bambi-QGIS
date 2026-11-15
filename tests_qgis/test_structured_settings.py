# -*- coding: utf-8 -*-
"""Frame-range corrections and the thermal curve are stored with the flight.

Both used to be saved by hand: written into the QGIS project, but read back
through the flight's own store, where they never were. Every load of a
flight's configuration - reopening the project, switching flights - therefore
came back without them, and the next save made that permanent. The pipeline
reads the corrections from the form, so the frames were then geo-referenced
without them.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import json
import os

import pytest

from qgis.core import QgsProject
from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from bambi_wildlife_detection.bambi_dock_widget import PLUGIN_SCOPE
from bambi_wildlife_detection.core import flights
from bambi_wildlife_detection.core.thermal_curve import ThermalCurve

FIRST = {"start": 0, "end": 250,
         "translation": {"x": 1.5, "y": -2.0, "z": 0.25},
         "rotation": {"x": 0.01, "y": -0.02, "z": 0.3}}
SECOND = {"start": 251, "end": 900,
          "translation": {"x": 0.0, "y": 0.0, "z": 4.0},
          "rotation": {"x": 0.0, "y": 0.0, "z": -0.1}}
CURVE = {"domain_lo": 2.0, "domain_hi": 31.5,
         "points": [[0.0, 0.0], [0.4, 0.7], [1.0, 1.0]]}


@pytest.fixture
def folders(tmp_path):
    a = str(tmp_path / "meadow")
    b = str(tmp_path / "forest")
    os.makedirs(a)
    os.makedirs(b)
    return a, b


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)


def _first_flight(dock, folder):
    dock.target_folder_edit.setText(folder)
    dock._on_target_folder_changed()


def _add_flight(dock, monkeypatch, folder, name):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: folder)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (name, True))
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: False)          # start from defaults
    dock.browse_and_add_flight()


def _listed(dock):
    return [dock.additional_corrections_list.item(row).text()
            for row in range(dock.additional_corrections_list.count())]


def test_corrections_survive_a_save_and_a_load(dock, folders):
    a, _b = folders
    _first_flight(dock, a)
    dock._add_correction_to_list(dict(FIRST))
    dock._add_correction_to_list(dict(SECOND))
    shown = _listed(dock)
    dock.save_config_to_project()

    dock.reset_config_to_defaults()
    assert dock._additional_corrections == [] and _listed(dock) == []

    dock.load_config_from_project()

    # the first one too: it used to be dropped for sitting in row 0
    assert dock._additional_corrections == [FIRST, SECOND]
    assert _listed(dock) == shown
    assert dock.get_config()["additional_corrections"] == [FIRST, SECOND]


def test_corrections_are_stored_with_the_flight(dock, folders):
    a, _b = folders
    _first_flight(dock, a)
    dock._add_correction_to_list(dict(FIRST))
    dock.save_config_to_project()

    stored = flights.load_config(a)
    assert json.loads(stored["Correction/AdditionalCorrections"]) == [FIRST]
    # ... and no longer beside it in the QGIS project
    _value, found = QgsProject.instance().readEntry(
        PLUGIN_SCOPE, "Correction/AdditionalCorrections", "")
    assert not found


def test_the_thermal_curve_survives_a_save_and_a_load(dock, folders):
    a, _b = folders
    _first_flight(dock, a)
    dock._thermal_vis_curve = ThermalCurve.from_dict(CURVE)
    dock.save_config_to_project()
    assert json.loads(flights.load_config(a)["Input/ThermalVisCurve"]) == CURVE

    dock.reset_config_to_defaults()
    assert dock._thermal_vis_curve is None

    dock.load_config_from_project()

    assert dock._thermal_vis_curve.to_dict() == CURVE
    assert dock.thermal_vis_curve_label.text() == \
        ThermalCurve.from_dict(CURVE).describe()


def test_each_flight_keeps_its_own(dock, quiet, monkeypatch, folders):
    a, b = folders
    _first_flight(dock, a)
    dock._add_correction_to_list(dict(FIRST))
    dock._thermal_vis_curve = ThermalCurve.from_dict(CURVE)

    _add_flight(dock, monkeypatch, b, "Forest")
    assert dock._additional_corrections == []
    assert dock._thermal_vis_curve is None
    dock._add_correction_to_list(dict(SECOND))

    dock.flight_combo.setCurrentIndex(0)
    assert dock._additional_corrections == [FIRST]
    assert dock._thermal_vis_curve.to_dict() == CURVE

    dock.flight_combo.setCurrentIndex(1)
    assert dock._additional_corrections == [SECOND]
    assert dock._thermal_vis_curve is None


@pytest.mark.parametrize("copy_configuration", [True, False])
def test_a_new_flight_starts_without_them(dock, quiet, monkeypatch, folders,
                                          copy_configuration):
    """Frame ranges name frames of one recording, and the curve was fitted to
    one flight's photos: neither is carried into the next flight, whether or
    not the rest of the configuration is copied."""
    a, b = folders
    _first_flight(dock, a)
    dock.confidence_spin.setValue(0.66)
    dock._add_correction_to_list(dict(FIRST))
    dock._thermal_vis_curve = ThermalCurve.from_dict(CURVE)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: b)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Forest", True))
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: copy_configuration)
    dock.browse_and_add_flight()

    assert dock.target_folder_edit.text() == b
    assert dock._additional_corrections == [] and _listed(dock) == []
    assert dock._thermal_vis_curve is None
    if copy_configuration:
        assert dock.confidence_spin.value() == pytest.approx(0.66)
    # ... and the first flight still has its own
    dock.flight_combo.setCurrentIndex(0)
    assert dock._additional_corrections == [FIRST]
    assert dock._thermal_vis_curve.to_dict() == CURVE


def test_no_corrections_and_no_curve_load_as_none(dock, folders):
    a, _b = folders
    _first_flight(dock, a)
    dock.save_config_to_project()
    dock._add_correction_to_list(dict(FIRST))        # not saved
    dock._thermal_vis_curve = ThermalCurve.from_dict(CURVE)

    dock.load_config_from_project()

    assert dock._additional_corrections == []
    assert isinstance(dock._additional_corrections, list)
    assert dock._thermal_vis_curve is None
    assert dock.thermal_vis_curve_label.text() == "(not defined)"


@pytest.mark.parametrize("corrections, curve", [
    ("[1, 2]", "[1, 2]"),               # the row numbers older versions wrote
    ("nonsense", "nonsense"),           # not JSON at all
    ('{"start": 3}', '{"domain_lo": "warm"}'),   # JSON, but not the setting
    ("", ""),
])
def test_what_cannot_be_read_loads_as_none(dock, folders, corrections, curve):
    """Before 9.1 the row numbers of the list were written, not the
    corrections; nothing of that can be brought back, and none of it may
    stop the configuration from loading."""
    a, _b = folders
    _first_flight(dock, a)
    dock.confidence_spin.setValue(0.61)
    dock.save_config_to_project()
    config = flights.load_config(a)
    config["Correction/AdditionalCorrections"] = corrections
    config["Input/ThermalVisCurve"] = curve
    flights.save_config(a, config)
    dock.confidence_spin.setValue(0.2)

    dock.load_config_from_project()

    assert dock._additional_corrections == []
    assert dock._thermal_vis_curve is None
    assert dock.confidence_spin.value() == pytest.approx(0.61)


def test_a_project_without_a_flight_folder_carries_them(dock):
    """With no folder yet there is no store: the QGIS project holds the
    configuration, these two settings included."""
    dock.target_folder_edit.setText("/nonexistent/bambi/flight")
    dock._add_correction_to_list(dict(FIRST))
    dock._thermal_vis_curve = ThermalCurve.from_dict(CURVE)
    dock.save_config_to_project()
    dock.reset_config_to_defaults()

    dock.load_config_from_project()

    assert dock._additional_corrections == [FIRST]
    assert dock._thermal_vis_curve.to_dict() == CURVE
