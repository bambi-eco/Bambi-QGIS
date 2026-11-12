# -*- coding: utf-8 -*-
"""Every "Add to QGIS" action puts its layers in a group (§10.2).

Flight products go into the flight's group; survey-wide products (density,
coverage, inventory, survey area) belong to no single flight and go into the
"BAMBI Survey" group.

Layers are grouped per flight so it is obvious which outputs belong together,
and so renaming or removing a flight takes its layers with it. A layer added
with ``QgsProject.addMapLayer(layer)`` lands at the root of the layer tree
instead - outside the flight - which is easy to write and invisible until
someone looks at the tree with two flights loaded.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import ast
import inspect
import os

import pytest

from qgis.core import QgsProject, QgsVectorLayer

import bambi_wildlife_detection.bambi_dock_widget as dock_module


def _add_actions():
    """Every ``add_*_to_qgis`` method on the dock widget."""
    return sorted(
        name for name in dir(dock_module.BambiDockWidget)
        if name.startswith("add_") and name.endswith("_to_qgis"))


def _dock_sources():
    """Source files the dock widget's methods live in: its own module first,
    then the plugin modules of the mixins it inherits from (the environment
    panel) - their ``add_*_to_qgis`` actions are held to the same rule."""
    paths = [inspect.getsourcefile(dock_module)]
    for base in dock_module.BambiDockWidget.__mro__:
        if not base.__module__.startswith("bambi_wildlife_detection"):
            continue
        path = inspect.getsourcefile(base)
        if path and path not in paths:
            paths.append(path)
    return paths


def _bodies():
    parts = []
    for path in _dock_sources():
        with open(path, encoding="utf-8") as fh:
            parts.append(fh.read())
    # One text, the dock module first so its line numbers stay true.
    src = "\n".join(parts)
    tree = ast.parse(src)
    return src, tree, {n.name: n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef)}


def _reaches_group(name, bodies, src, seen=None):
    """True when the method, or something it calls, places layers in a group."""
    seen = seen or set()
    if name in seen or name not in bodies:
        return False
    seen.add(name)
    body = ast.get_source_segment(src, bodies[name]) or ""
    markers = ("_create_layer_group", "_flight_group", "_add_layer_to_flight",
               "_add_layer_to_survey")
    if any(marker in body for marker in markers):
        return True
    for other in bodies:
        if other == name or f"self.{other}(" not in body:
            continue
        if _reaches_group(other, bodies, src, seen):
            return True
    return False


def _roots(name, bodies, src, seen=None):
    """Calls that add a layer straight to the tree root."""
    seen = seen or set()
    found = []
    if name in seen or name not in bodies:
        return found
    seen.add(name)
    for call in ast.walk(bodies[name]):
        if not isinstance(call, ast.Call):
            continue
        if (isinstance(call.func, ast.Attribute)  # noqa: W503
                and call.func.attr == "addMapLayer"  # noqa: W503
                and len(call.args) == 1):  # noqa: W503
            found.append(f"{name}:{call.lineno}")
    body = ast.get_source_segment(src, bodies[name]) or ""
    for other in bodies:
        if other != name and f"self.{other}(" in body:
            found.extend(_roots(other, bodies, src, seen))
    return found


# ---------------------------------------------------------------------------
# The rule, over every action at once
# ---------------------------------------------------------------------------

def test_there_are_add_actions_to_check():
    """A guard on the guard: a renamed convention would empty this suite."""
    assert len(_add_actions()) >= 12


def test_every_add_action_reaches_the_flight_group():
    src, _tree, bodies = _bodies()
    outside = [name for name in _add_actions()
               if not _reaches_group(name, bodies, src)]
    assert outside == [], (
        "these add their layers outside the flight group: " + ", ".join(outside))


def test_no_add_action_puts_a_layer_at_the_tree_root():
    """addMapLayer(layer) with one argument bypasses the group entirely, even
    where the same method groups its other layers."""
    src, _tree, bodies = _bodies()
    offenders = []
    for name in _add_actions():
        offenders.extend(_roots(name, bodies, src))
    assert offenders == [], (
        "add layers to the tree root: " + ", ".join(offenders))


# ---------------------------------------------------------------------------
# The helper the actions use
# ---------------------------------------------------------------------------

@pytest.fixture
def flight(dock, tmp_path):
    folder = str(tmp_path / "meadow" / "qgis")
    os.makedirs(folder, exist_ok=True)
    dock.target_folder_edit.setText(folder)
    dock._on_target_folder_changed()
    return folder


def _layer(name="probe"):
    return QgsVectorLayer("Point?crs=EPSG:32633", name, "memory")


def test_a_single_layer_lands_in_the_flight_group(dock, flight):
    dock._add_layer_to_flight(_layer())

    group = QgsProject.instance().layerTreeRoot().findGroup("meadow")
    assert group is not None
    assert [n.name() for n in group.children()] == ["probe"]


def test_it_is_not_left_at_the_tree_root(dock, flight):
    dock._add_layer_to_flight(_layer())

    root = QgsProject.instance().layerTreeRoot()
    assert [c.name() for c in root.children() if c.name() == "probe"] == []


def test_the_layer_is_still_registered_with_the_project(dock, flight):
    """addMapLayer(layer, False) skips the tree, not the registry."""
    layer = dock._add_layer_to_flight(_layer())
    assert QgsProject.instance().mapLayer(layer.id()) is not None


def test_layers_stack_newest_first(dock, flight):
    dock._add_layer_to_flight(_layer("first"))
    dock._add_layer_to_flight(_layer("second"))

    group = QgsProject.instance().layerTreeRoot().findGroup("meadow")
    assert [n.name() for n in group.children()] == ["second", "first"]


def test_it_can_append_instead(dock, flight):
    dock._add_layer_to_flight(_layer("first"))
    dock._add_layer_to_flight(_layer("second"), at_top=False)

    group = QgsProject.instance().layerTreeRoot().findGroup("meadow")
    assert [n.name() for n in group.children()] == ["first", "second"]


# ---------------------------------------------------------------------------
# Camera subgroups and the fixed order (9.0.0)
# ---------------------------------------------------------------------------

def _names(group):
    return [n.name() for n in group.children()]


def test_camera_products_go_into_the_camera_subgroup(dock, flight):
    dock._add_layer_to_flight(_layer("BAMBI Orthomosaic (Thermal)"))
    dock._create_layer_group("BAMBI Wildlife Tracks (RGB)")

    group = QgsProject.instance().layerTreeRoot().findGroup("meadow")
    assert _names(group) == ["Thermal", "RGB"]
    assert _names(group.findGroup("Thermal")) == ["BAMBI Orthomosaic (Thermal)"]
    assert _names(group.findGroup("RGB")) == ["BAMBI Wildlife Tracks (RGB)"]


def test_thermal_comes_before_rgb_whatever_was_added_first(dock, flight):
    dock._add_layer_to_flight(_layer("BAMBI ALFS (RGB)"))
    dock._add_layer_to_flight(_layer("BAMBI ALFS (Thermal)"))

    group = QgsProject.instance().layerTreeRoot().findGroup("meadow")
    assert _names(group) == ["Thermal", "RGB"]


def test_the_order_does_not_depend_on_when_a_layer_was_added(dock, flight):
    """Rasters first, overlays last - the overlays still end up on top."""
    dock._create_layer_group("BAMBI Frame GeoTIFFs (Thermal)")
    dock._add_layer_to_flight(_layer("BAMBI Orthomosaic (Thermal)"))
    dock._add_layer_to_flight(_layer("BAMBI ALFS (Thermal)"))
    dock._create_layer_group("BAMBI Wildlife Tracks (Thermal)")
    dock._create_layer_group("BAMBI Frame Detections (Thermal)")
    dock._create_layer_group("BAMBI Flight Route (Thermal)")
    dock._create_layer_group("BAMBI FoV Polygons (Thermal)")

    thermal = QgsProject.instance().layerTreeRoot().findGroup("Thermal")
    assert _names(thermal) == [
        "BAMBI Flight Route (Thermal)",
        "BAMBI FoV Polygons (Thermal)",
        "BAMBI Wildlife Tracks (Thermal)",
        "BAMBI Frame Detections (Thermal)",
        "BAMBI Orthomosaic (Thermal)",
        "BAMBI ALFS (Thermal)",
        "BAMBI Frame GeoTIFFs (Thermal)",
    ]


def test_a_group_is_replaced_inside_its_camera_subgroup(dock, flight):
    dock._create_layer_group("BAMBI Wildlife Tracks (Thermal)")
    assert dock._remove_layer_group("BAMBI Wildlife Tracks (Thermal)") is True
    dock._create_layer_group("BAMBI Wildlife Tracks (Thermal)")

    thermal = QgsProject.instance().layerTreeRoot().findGroup("Thermal")
    assert _names(thermal) == ["BAMBI Wildlife Tracks (Thermal)"]


def test_the_users_own_layers_are_not_moved(dock, flight):
    dock._add_layer_to_flight(_layer("BAMBI Orthomosaic (Thermal)"))
    thermal = QgsProject.instance().layerTreeRoot().findGroup("Thermal")
    mine = _layer("my notes")
    QgsProject.instance().addMapLayer(mine, False)
    thermal.insertLayer(0, mine)

    dock._create_layer_group("BAMBI Flight Route (Thermal)")
    assert _names(thermal) == ["my notes", "BAMBI Flight Route (Thermal)",
                               "BAMBI Orthomosaic (Thermal)"]


def test_survey_layers_follow_the_order_too(dock, flight):
    dock._add_layer_to_survey(_layer("BAMBI Track Inventory (Thermal)"))
    dock._add_layer_to_survey(_layer("BAMBI Coverage Map (Thermal)"))
    dock._add_layer_to_survey(_layer("Survey Areas"))

    survey = QgsProject.instance().layerTreeRoot().findGroup(dock.SURVEY_GROUP)
    assert _names(survey) == ["Survey Areas", "BAMBI Coverage Map (Thermal)",
                              "BAMBI Track Inventory (Thermal)"]


def test_without_a_flight_it_falls_back_to_the_root(dock):
    """A project with no flight has no group to put anything in."""
    dock._flight_list = []
    layer = dock._add_layer_to_flight(_layer("loose"))

    root = QgsProject.instance().layerTreeRoot()
    assert root.findLayer(layer.id()) is not None


# ---------------------------------------------------------------------------
# "Added" statuses look at the active flight only
# ---------------------------------------------------------------------------

def test_another_flights_orthomosaic_does_not_count_as_added(dock, flight,
                                                             tmp_path):
    dock._add_layer_to_flight(_layer("BAMBI Orthomosaic (Thermal)"))
    dock._check_existing_qgis_layers()
    assert "Added" in dock.add_orthomosaic_status.text()

    # a second flight, active, with nothing added yet
    other = str(tmp_path / "forest" / "qgis")
    os.makedirs(other, exist_ok=True)
    dock._flight_list = dock._flights() + [
        {"name": "forest", "target_folder": other}]
    dock._active_flight_index = 1

    dock._check_existing_qgis_layers()
    assert "Added" not in dock.add_orthomosaic_status.text()

    dock._active_flight_index = 0
    dock._check_existing_qgis_layers()
    assert "Added" in dock.add_orthomosaic_status.text()


def test_added_groups_are_found_inside_the_camera_subgroup(dock, flight):
    dock._create_layer_group("BAMBI Wildlife Tracks (Thermal)")
    dock._check_existing_qgis_layers()
    assert "Added" in dock.layers_status.text()
