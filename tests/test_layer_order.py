# -*- coding: utf-8 -*-
"""Layer grouping and ordering rules (core/layer_order.py)."""
import pytest

from bambi_wildlife_detection.core import layer_order as lo


@pytest.mark.parametrize("name,expected", [
    ("BAMBI Flight Route (Thermal)", "flight_route"),
    ("BAMBI FoV Polygons (RGB)", "fov"),
    ("BAMBI FoV Polygons - Combined (Thermal)", "fov"),
    ("BAMBI Density Tracks (Thermal)", "density"),
    ("BAMBI Coverage Map (RGB)", "coverage"),
    ("BAMBI Transect Areas (Thermal)", "transects"),
    ("BAMBI Track Inventory (Thermal)", "track_inventory"),
    ("BAMBI Matched Pairs", "matched_pairs"),
    ("BAMBI Wildlife Tracks (RGB)", "tracks"),
    ("BAMBI Frame Detections (Thermal)", "detections"),
    ("BAMBI Detections - All Frames (Thermal)", "detections"),
    ("BAMBI Perpendicular (Thermal)", "distance"),
    ("BAMBI Track Perpendicular (RGB)", "distance"),
    ("SAM3 Segmentation (Thermal)", "segmentation"),
    ("SAM3 Segmentation Orthomosaic (RGB)", "segmentation"),
    ("Reconstruction dsm", "reconstruction"),
    ("BAMBI Orthomosaic (Thermal)", "orthomosaic"),
    ("BAMBI ALFS (RGB)", "alfs"),
    ("BAMBI Frame GeoTIFFs (Thermal)", "geotiffs"),
    ("Survey Plot", "survey_area"),
    ("Exclusion Zones", "survey_area"),
    ("my own layer", None),
])
def test_category(name, expected):
    assert lo.category(name) == expected


def test_modality_from_the_camera_label():
    assert lo.modality("BAMBI ALFS (Thermal)") == "t"
    assert lo.modality("BAMBI ALFS (RGB)") == "w"
    assert lo.modality("BAMBI Matched Pairs") is None


def test_layers_added_in_any_order_end_up_in_the_fixed_order():
    wanted = ["BAMBI Flight Route (Thermal)", "BAMBI FoV Polygons (Thermal)",
              "BAMBI Transect Areas (Thermal)", "BAMBI Wildlife Tracks (Thermal)",
              "BAMBI Frame Detections (Thermal)", "BAMBI Perpendicular (Thermal)",
              "SAM3 Segmentation (Thermal)", "BAMBI Orthomosaic (Thermal)",
              "BAMBI ALFS (Thermal)", "BAMBI Frame GeoTIFFs (Thermal)"]
    assert lo.ordered(list(reversed(wanted))) == wanted
    assert lo.ordered(wanted[::2] + wanted[1::2]) == wanted


def test_survey_group_order():
    wanted = ["Survey Plot", "BAMBI Density Tracks (Thermal)",
              "BAMBI Coverage Map (Thermal)", "BAMBI Track Inventory (Thermal)"]
    assert lo.ordered(wanted[::-1]) == wanted


def test_flight_group_keeps_camera_groups_below_shared_layers():
    assert lo.ordered(["RGB", "Thermal", "BAMBI Matched Pairs"]) == \
        ["BAMBI Matched Pairs", "Thermal", "RGB"]


def test_user_layers_are_left_alone():
    siblings = ["my notes", "BAMBI Flight Route (Thermal)", "basemap",
                "BAMBI ALFS (Thermal)"]
    # tracks go between route and ALFS, after the user's "basemap"
    assert lo.insert_index(siblings, "BAMBI Wildlife Tracks (Thermal)") == 3
    # an unknown name goes to the top, as before
    assert lo.insert_index(siblings, "probe") == 0
    assert lo.insert_index(siblings, "probe", at_top=False) == 4


def test_a_reordered_group_is_respected():
    # The user dragged ALFS above the route: a new layer still goes before
    # the first sibling that ranks after it, nothing is moved.
    siblings = ["BAMBI ALFS (Thermal)", "BAMBI Flight Route (Thermal)"]
    assert lo.insert_index(siblings, "BAMBI FoV Polygons (Thermal)") == 0
    assert lo.insert_index(siblings, "BAMBI Frame GeoTIFFs (Thermal)") == 2


def test_empty_group():
    assert lo.insert_index([], "BAMBI ALFS (Thermal)") == 0
