# -*- coding: utf-8 -*-
"""Real-QGIS test: choosing the alfspy engine and ray caster.

The choice is made in the Dependency Manager, stored in the QGIS project, and
read back by the dock so every step that renders can export it. That path
crosses three modules and a QgsProject, which is exactly the kind of wiring the
text-level unit tests cannot check - so it is exercised here for real.
"""
import pytest

from qgis.core import QgsProject

from bambi_wildlife_detection.core import alfs_runtime
from bambi_wildlife_detection.gui_utils import (
    read_alfs_selection, write_alfs_selection)


@pytest.fixture
def manager():
    from bambi_wildlife_detection.bambi_dependency_manager import (
        DependencyManagerDialog)
    dialog = DependencyManagerDialog()
    yield dialog
    dialog.close()
    dialog.deleteLater()


@pytest.fixture(autouse=True)
def clean_project():
    """Each test starts from a project that has never chosen an engine."""
    QgsProject.instance().clear()
    yield
    QgsProject.instance().clear()


# ---------------------------------------------------------------------------
# Project storage
# ---------------------------------------------------------------------------

def test_an_untouched_project_reads_as_the_defaults():
    assert read_alfs_selection() == ("moderngl", "embree")


def test_the_selection_round_trips_through_the_project():
    write_alfs_selection("vulkan", "warp")
    assert read_alfs_selection() == ("vulkan", "warp")


def test_a_nonsense_stored_value_loads_as_the_default():
    """A project written by a newer plugin still has to open."""
    from bambi_wildlife_detection.gui_utils import PLUGIN_SCOPE

    QgsProject.instance().writeEntry(
        PLUGIN_SCOPE, "Processing/AlfsEngine", "metal")
    assert read_alfs_selection()[0] == "moderngl"


# ---------------------------------------------------------------------------
# The dropdowns
# ---------------------------------------------------------------------------

def test_the_dropdowns_offer_every_engine_and_raycaster(manager):
    engines = [manager._engine_combo.itemData(i)
               for i in range(manager._engine_combo.count())]
    casters = [manager._raycaster_combo.itemData(i)
               for i in range(manager._raycaster_combo.count())]
    assert engines == list(alfs_runtime.engine_names())
    assert casters == list(alfs_runtime.raycaster_names())


def test_the_dropdowns_start_on_the_projects_choice():
    from bambi_wildlife_detection.bambi_dependency_manager import (
        DependencyManagerDialog)

    write_alfs_selection("torch", "warp")
    dialog = DependencyManagerDialog()
    try:
        assert dialog._engine_combo.currentData() == "torch"
        assert dialog._raycaster_combo.currentData() == "warp"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_changing_a_dropdown_saves_without_any_confirmation(manager):
    """The dialog is a separate window with no OK button, so a selection that
    only counted after some other save would simply be lost."""
    combo = manager._engine_combo
    combo.setCurrentIndex(combo.findData("vulkan"))
    assert read_alfs_selection() == ("vulkan", "embree")


def test_the_install_spec_follows_the_dropdowns(manager):
    manager._engine_combo.setCurrentIndex(
        manager._engine_combo.findData("torch"))
    manager._raycaster_combo.setCurrentIndex(
        manager._raycaster_combo.findData("warp"))

    spec = manager._selected_alfs_spec()
    assert spec["extras"] == ("torch", "warp")
    assert spec["git_url"].startswith("AlfsPy[torch,warp] @ git+")


def test_the_hint_describes_the_current_pair(manager):
    manager._engine_combo.setCurrentIndex(
        manager._engine_combo.findData("vulkan"))
    text = manager._alfs_hint.text()
    assert "Python 3.11" in text        # the Vulkan engine's caveat
    assert "Embree" in text or "default" in text


# ---------------------------------------------------------------------------
# Reaching the run
# ---------------------------------------------------------------------------

def test_the_dock_config_carries_the_selection(dock):
    write_alfs_selection("torch", "warp")
    config = dock.get_config()
    assert config["alfs_engine"] == "torch"
    assert config["alfs_raycaster"] == "warp"


def test_a_render_step_exports_what_the_project_chose(dock, monkeypatch):
    """End of the chain: what the dropdown stored is what alfspy will read."""
    from bambi_wildlife_detection.bambi_processing import BambiProcessor

    write_alfs_selection("vulkan", "warp")
    env = {}
    monkeypatch.setattr(alfs_runtime.os, "environ", env)

    BambiProcessor._apply_render_selection(dock.get_config())
    assert env["ALFS_ENGINE"] == "vulkan"
    assert env["ALFS_RAYCASTER"] == "warp"
