# -*- coding: utf-8 -*-
"""Field-of-view footprints in the store (§11).

The FoV step wrote only ``fov_polygons.txt`` until now, so ``fov.gpkg`` existed
solely where a 5.x project had been migrated — while the coverage map, the
transect areas a population estimate divides by, and the QGIS layers all read
the store. The step writes it now, and this is that contract.
"""
import os

from bambi_wildlife_detection.core import fov_store, pipeline_outputs, store


SQUARE = [(0.0, 0.0, 1.0), (10.0, 0.0, 1.0), (10.0, 10.0, 1.0),
          (0.0, 10.0, 1.0)]


def test_a_polygon_round_trips(tmp_path):
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {3: SQUARE})
    assert fov_store.load_fov(root, "t") == {3: SQUARE}


def test_the_vertex_order_is_preserved(tmp_path):
    """A footprint is a ring; a set of points has no edges."""
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {0: SQUARE})
    assert fov_store.load_fov(root, "t")[0] == SQUARE


def test_frames_are_independent(tmp_path):
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {0: SQUARE, 1: SQUARE})
    assert sorted(fov_store.load_fov(root, "t")) == [0, 1]


def test_re_recording_a_frame_replaces_it(tmp_path):
    """Re-running the step for one frame must not leave the old ring behind."""
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {0: SQUARE})
    smaller = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
    fov_store.record_fov(root, "t", {0: smaller})

    assert fov_store.load_fov(root, "t") == {0: smaller}


def test_re_recording_leaves_other_frames_alone(tmp_path):
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {0: SQUARE, 1: SQUARE})
    fov_store.record_fov(root, "t", {0: SQUARE[:3]})

    assert len(fov_store.load_fov(root, "t")[1]) == 4


def test_a_frame_with_no_footprint_is_recorded_as_absent(tmp_path):
    """The step writes "frame 5, 0 points" when nothing projected; that is not
    a polygon and must not come back as one."""
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {5: []})
    assert fov_store.load_fov(root, "t") == {}


def test_a_frame_that_loses_its_footprint_is_removed(tmp_path):
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {5: SQUARE})
    fov_store.record_fov(root, "t", {5: []})
    assert fov_store.load_fov(root, "t") == {}


def test_none_vertices_are_dropped(tmp_path):
    """The projection returns None for a corner that missed the DEM."""
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {0: [SQUARE[0], None, SQUARE[2]]})
    assert len(fov_store.load_fov(root, "t")[0]) == 2


def test_the_modalities_do_not_share(tmp_path):
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {0: SQUARE})
    assert fov_store.load_fov(root, "w") == {}


def test_no_store_is_empty_not_an_error(tmp_path):
    assert fov_store.load_fov(str(tmp_path / "nothing"), "t") == {}


def test_frames_lists_what_has_a_footprint(tmp_path):
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {4: SQUARE, 1: SQUARE})
    assert list(fov_store.frames(root, "t")) == [1, 4]


def test_the_layer_reader_sees_what_the_step_wrote(tmp_path):
    """The two halves have to agree, since nothing else connects them."""
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {2: SQUARE})

    polygons = pipeline_outputs.load_fov_polygons_3d(
        os.path.join(root, "fov_t", "fov_polygons.txt"))
    assert polygons == {2: SQUARE}


def test_the_file_lands_where_the_other_stages_live(tmp_path):
    root = str(tmp_path)
    fov_store.record_fov(root, "t", {0: SQUARE})
    assert os.path.isfile(store.stage_path(root, store.FOV, "t"))
