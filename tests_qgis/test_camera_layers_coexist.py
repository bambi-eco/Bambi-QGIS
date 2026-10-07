# -*- coding: utf-8 -*-
"""Thermal and RGB layers of the same kind must live side by side.

Each "Add ... to QGIS" persists its layer to a GeoPackage under the target
folder, and ``_persist_memory_layer`` first removes every layer loaded from a
file it is about to rewrite. When both cameras used the same file name, adding
the RGB merged FoV silently removed the thermal one (and vice versa).
"""
import os

from qgis.core import QgsCoordinateReferenceSystem, QgsProject


def _polygons():
    square = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
    return {0: square, 1: [(x + 5.0, y, z) for x, y, z in square]}


def _fov_layers():
    return sorted(layer.name() for layer in QgsProject.instance().mapLayers().values()
                  if layer.customProperty("bambi_layer_type") == "fov")


def test_combined_fov_layers_of_both_cameras_coexist(dock, tmp_path):
    dock.target_folder_edit.setText(str(tmp_path))
    crs = QgsCoordinateReferenceSystem("EPSG:32633")

    dock._add_fov_combined_layer(_polygons(), crs, "Thermal")
    dock._add_fov_combined_layer(_polygons(), crs, "RGB")

    assert _fov_layers() == ["BAMBI FoV Polygons - Combined (RGB)",
                             "BAMBI FoV Polygons - Combined (Thermal)"]
    files = sorted(f for f in os.listdir(tmp_path / "fov_layers") if f.endswith(".gpkg"))
    assert files == ["FoV_Combined_t.gpkg", "FoV_Combined_w.gpkg"]


def test_rewriting_one_camera_keeps_the_other(dock, tmp_path):
    """Re-adding the thermal layer replaces it, and only it."""
    dock.target_folder_edit.setText(str(tmp_path))
    crs = QgsCoordinateReferenceSystem("EPSG:32633")

    dock._add_fov_combined_layer(_polygons(), crs, "Thermal")
    dock._add_fov_combined_layer(_polygons(), crs, "RGB")
    dock._add_fov_combined_layer(_polygons(), crs, "Thermal")

    assert _fov_layers() == ["BAMBI FoV Polygons - Combined (RGB)",
                             "BAMBI FoV Polygons - Combined (Thermal)"]
