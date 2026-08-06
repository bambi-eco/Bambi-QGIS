# -*- coding: utf-8 -*-
"""The species filter on the Survey Analytics tab (§8.2).

The filtering itself is covered headlessly in ``tests/test_analytics_source.py``;
what needs a real QGIS is the wiring — that the list comes from the active
flight's own vocabulary, that "All species" means no filter rather than a list
of today's ids, and that the choice reaches every analytic that counts animals.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.core import detection_store, track_store


@pytest.fixture
def project_folder(tmp_path):
    root = str(tmp_path / "flight")
    os.makedirs(root, exist_ok=True)
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "1"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": ids[0], "gx1": 500000.0, "gy1": 5300000.0,
         "gz1": 400.0, "gx2": 500010.0, "gy2": 5300010.0, "gz2": 400.0}])
    return root


@pytest.fixture
def quiet(monkeypatch):
    shown = {"info": [], "warn": []}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    return shown


def _tick(dock, name):
    for index in range(dock.analytics_species_list.count()):
        item = dock.analytics_species_list.item(index)
        if item.text() == name:
            item.setCheckState(Qt.CheckState.Checked)
            return item
    raise AssertionError(f"species '{name}' is not offered")


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

def test_all_species_is_the_default(dock):
    """The behaviour every earlier version had."""
    assert dock.analytics_all_species_check.isChecked() is True
    assert dock.analytics_species_ids() is None


def test_the_list_is_disabled_while_all_species_is_ticked(dock):
    assert dock.analytics_species_list.isEnabled() is False
    dock.analytics_all_species_check.setChecked(False)
    assert dock.analytics_species_list.isEnabled() is True


def test_the_list_comes_from_the_projects_own_vocabulary(dock,
                                                         project_folder):
    dock.target_folder_edit.setText(project_folder)
    assert dock.refresh_analytics_species() > 0

    names = [dock.analytics_species_list.item(i).text()
             for i in range(dock.analytics_species_list.count())]
    assert "roe deer" in names


def test_the_false_positive_class_is_never_offered(dock, project_folder):
    """It is excluded from every analytic, so offering it would be a lie."""
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()

    names = [dock.analytics_species_list.item(i).text()
             for i in range(dock.analytics_species_list.count())]
    assert "not-an-animal" not in names


def test_ticking_a_species_selects_only_it(dock, project_folder):
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()
    dock.analytics_all_species_check.setChecked(False)
    item = _tick(dock, "roe deer")

    assert dock.analytics_species_ids() == [
        item.data(Qt.ItemDataRole.UserRole)]


def test_all_species_means_no_filter_not_a_list_of_ids(dock, project_folder):
    """A species added later stays included without revisiting the tab."""
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()
    dock.analytics_all_species_check.setChecked(False)
    _tick(dock, "roe deer")
    dock.analytics_all_species_check.setChecked(True)

    assert dock.analytics_species_ids() is None


def test_a_tick_survives_a_refresh(dock, project_folder):
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()
    dock.analytics_all_species_check.setChecked(False)
    _tick(dock, "roe deer")

    dock.refresh_analytics_species()
    assert [name for name, _id in dock._analytics_species_selection()] == [
        "roe deer"]


def test_the_status_says_what_is_counted(dock, project_folder):
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()
    dock.analytics_all_species_check.setChecked(False)
    _tick(dock, "roe deer")

    assert "roe deer" in dock.analytics_species_status.text()


def test_a_folder_without_a_project_offers_nothing(dock, tmp_path):
    dock.target_folder_edit.setText(str(tmp_path / "nothing"))
    assert dock.refresh_analytics_species() == 0


# ---------------------------------------------------------------------------
# Reaching the analytics
# ---------------------------------------------------------------------------

def test_the_choice_reaches_the_config(dock, project_folder):
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()
    dock.analytics_all_species_check.setChecked(False)
    item = _tick(dock, "roe deer")

    config = dock.get_config()
    assert config["analytics_species_ids"] == [
        item.data(Qt.ItemDataRole.UserRole)]


def test_all_species_reaches_the_config_as_no_filter(dock, project_folder):
    dock.target_folder_edit.setText(project_folder)
    assert dock.get_config()["analytics_species_ids"] is None


def test_running_with_nothing_ticked_is_refused(dock, quiet, project_folder):
    """Otherwise the analytic runs and reports zero, which reads as "no
    animals here" rather than "you excluded them all"."""
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()
    dock.analytics_all_species_check.setChecked(False)

    dock.run_density_heatmap()
    assert quiet["warn"]


def test_the_coverage_map_is_not_blocked_by_the_filter(dock, quiet,
                                                       project_folder):
    """It counts frames, not animals, so the species filter does not apply."""
    dock.target_folder_edit.setText(project_folder)
    dock.refresh_analytics_species()
    dock.analytics_all_species_check.setChecked(False)

    dock.run_coverage_map()
    warnings = " ".join(str(a) for a in quiet["warn"])
    assert "species" not in warnings.lower()
