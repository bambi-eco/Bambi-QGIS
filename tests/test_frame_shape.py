# -*- coding: utf-8 -*-
"""A camera has the shape of its frames.

Frame extraction writes square frames unless "Preserve original aspect
ratio" is ticked. The projection into the other camera, the cross hair of a
map click and the Correction Wizard built their camera square whatever the
frames were - and a square camera casts every ray of a 16:9 frame too close
to the optical axis, sideways by the aspect ratio. Geo-referencing itself
always used the real shape, so these tools disagreed with the results they
were showing.
"""
import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_box_projector import (
    _match_boxes_to_georef)
from bambi_wildlife_detection.core import inspection, poses

WIDTH, HEIGHT = 1600, 900


def _frame(folder, modality="w", width=WIDTH, height=HEIGHT,
           name="frame_000000.png"):
    import cv2

    frames = os.path.join(folder, f"frames_{modality}")
    os.makedirs(frames, exist_ok=True)
    cv2.imwrite(os.path.join(frames, name),
                np.zeros((height, width, 3), dtype=np.uint8))
    return os.path.join(frames, name)


# ---------------------------------------------------------------------------
# The size of a camera's frames
# ---------------------------------------------------------------------------

def test_the_size_is_read_off_a_frame(tmp_path):
    root = str(tmp_path)
    path = _frame(root)
    assert poses.image_size(path) == (WIDTH, HEIGHT)
    images = [{"imagefile": "frame_000000.png"}]
    assert poses.frame_size(root, "w", images) == (WIDTH, HEIGHT)
    assert poses.aspect_ratio((WIDTH, HEIGHT)) == pytest.approx(16 / 9)


def test_the_first_frame_that_is_there_decides(tmp_path):
    root = str(tmp_path)
    _frame(root, name="frame_000007.png")
    images = [{"imagefile": "frame_000000.png"}, {"imagefile": ""}, {},
              {"imagefile": "frame_000007.png"}]
    assert poses.frame_size(root, "w", images) == (WIDTH, HEIGHT)


def test_without_frames_the_size_is_unknown(tmp_path):
    assert poses.frame_size(str(tmp_path), "w", [{"imagefile": "x.png"}]) \
        is None
    assert poses.frame_size(str(tmp_path), "w", []) is None
    assert poses.image_size(str(tmp_path / "nope.png")) is None
    assert poses.image_size("") is None
    # ... and a camera of unknown shape is taken to be square, as before
    assert poses.aspect_ratio(None) == 1.0
    assert poses.aspect_ratio((0, 0)) == 1.0


def test_a_file_that_is_no_image_has_no_size(tmp_path):
    path = tmp_path / "frame.png"
    path.write_text("not an image")
    assert poses.image_size(str(path)) is None


# ---------------------------------------------------------------------------
# Out onto the ground and back, on a 16:9 frame
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_flight(tmp_path):
    """A flight with 16:9 RGB frames and a camera 100 m above flat ground,
    tilted and turned so that nothing is symmetric by accident."""
    pytest.importorskip("alfspy")
    pytest.importorskip("trimesh")
    root = str(tmp_path)
    _frame(root)
    meta = {"imagefile": "frame_000000.png",
            "location": [0.0, 0.0, 100.0],
            "rotation": [8.0, -5.0, 30.0], "fovy": [50.0]}
    with open(os.path.join(root, "poses_w.json"), "w") as handle:
        json.dump({"images": [meta]}, handle)
    with open(os.path.join(root, "dem.json"), "w") as handle:
        json.dump({"origin": [0.0, 0.0, 0.0]}, handle)
    return root, meta


def _ground_point(meta, pixel):
    """Where the ray of *pixel* meets the ground - by the camera the
    geo-referencing builds: one with the frame's own shape."""
    import trimesh
    from alfspy.core.convert.convert import pixel_to_world_coord
    from bambi_wildlife_detection.core.camera_pose import build_camera

    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    camera = build_camera(meta, zero, zero, aspect_ratio=WIDTH / HEIGHT)
    ground = trimesh.Trimesh(
        vertices=[[-5000, -5000, 0], [5000, -5000, 0], [5000, 5000, 0],
                  [-5000, 5000, 0]], faces=[[0, 1, 2], [0, 2, 3]])
    hits = pixel_to_world_coord([pixel[0]], [pixel[1]], WIDTH, HEIGHT,
                                ground, camera, include_misses=False)
    hit = np.reshape(np.asarray(hits, dtype=float), (-1, 3))[0]
    return float(hit[0]), float(hit[1])


@pytest.mark.parametrize("pixel", [(1400.0, 300.0), (200.0, 700.0),
                                   (800.0, 450.0), (1550.0, 60.0)])
def test_a_map_click_lands_on_the_pixel_it_came_from(wide_flight, pixel):
    """The cross hair sat too far from the image centre: for a point seen
    near the right edge, outside the frame altogether."""
    root, meta = wide_flight
    x, y = _ground_point(meta, pixel)

    found = inspection.project_map_point(
        (x, y), 0, root, os.path.join(root, "dem.glb"), "", "w",
        mesh_cache={}, sample_elevation=lambda *a: 0.0)

    assert found is not None
    assert found == (pytest.approx(pixel[0], abs=0.5),
                     pytest.approx(pixel[1], abs=0.5))


def test_a_square_camera_would_miss_by_the_aspect_ratio(wide_flight):
    """What the tools did - kept here so the test above is known to bite."""
    from bambi_wildlife_detection.core.camera_pose import (
        build_camera, world_to_pixel)

    _root, meta = wide_flight
    x, y = _ground_point(meta, (1400.0, 300.0))
    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    square = build_camera(meta, zero, zero, aspect_ratio=1.0)
    xs, _ys = world_to_pixel(np.array([[x, y, 0.0]]), WIDTH, HEIGHT, square)
    off_centre = (float(xs[0]) - WIDTH / 2) / (1400.0 - WIDTH / 2)
    assert off_centre == pytest.approx(WIDTH / HEIGHT, rel=1e-3)


# ---------------------------------------------------------------------------
# Two boxes that look alike
# ---------------------------------------------------------------------------

GEOREF = [
    {"detection_id": 41, "class_id": 1, "confidence": 0.8712, "x1": 0.0},
    {"detection_id": 42, "class_id": 1, "confidence": 0.8712, "x1": 50.0},
]


def test_look_alike_boxes_get_an_entry_each():
    """Both were matched to the first entry, and its box drawn twice."""
    boxes = [(0, 0, 9, 9, 0.8712, 1), (60, 0, 69, 9, 0.8712, 1)]
    matched = _match_boxes_to_georef(boxes, GEOREF)
    assert [g["detection_id"] for g in matched] == [41, 42]


def test_the_clicked_box_and_the_others_do_not_share_an_entry():
    taken = set()
    green = _match_boxes_to_georef([(0, 0, 9, 9, 0.8712, 1)], GEOREF,
                                   taken=taken)
    blue = _match_boxes_to_georef([(60, 0, 69, 9, 0.8712, 1)], GEOREF,
                                  taken=taken)
    assert [g["detection_id"] for g in green + blue] == [41, 42]
    assert taken == {41, 42}
