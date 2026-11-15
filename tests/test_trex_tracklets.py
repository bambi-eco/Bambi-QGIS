# -*- coding: utf-8 -*-
"""Reading TRex tracklets and bringing them into the frames' pixel space
(core.trex_tracklets)."""
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import trex_tracklets


def _tracklet(folder, name, frames, xs, ys, **fields):
    """One ``.npz`` as TRex writes it: per frame, a list of pose points."""
    os.makedirs(folder, exist_ok=True)
    data = {"frame": np.asarray(frames),
            "detection_p": np.asarray(fields.pop(
                "confidence", [0.9] * len(frames)), dtype=float),
            "detection_class": np.asarray(fields.pop(
                "classes", [0] * len(frames)), dtype=float)}
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    for point in range(xs.shape[1]):
        data[f"poseX{point}"] = xs[:, point]
        data[f"poseY{point}"] = ys[:, point]
    data.update(fields)
    path = os.path.join(folder, name)
    np.savez(path, **data)
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_a_box_is_the_extent_of_the_pose_points(tmp_path):
    folder = str(tmp_path)
    _tracklet(folder, "a.npz", [4], [[10.0, 30.0, 20.0]], [[50.0, 40.0, 70.0]],
              id=np.array([7]), video_size=np.array([640, 512]))

    boxes, video_size = trex_tracklets.read_tracklets(folder)

    assert video_size == (640, 512)
    assert boxes == [{"frame": 4, "track_id": 7, "x1": 10.0, "y1": 40.0,
                      "x2": 30.0, "y2": 70.0, "confidence": 0.9,
                      "class_id": 0}]


def test_the_boxes_come_sorted_by_frame_then_tracklet(tmp_path):
    folder = str(tmp_path)
    _tracklet(folder, "b.npz", [2, 0], [[1.0, 2.0]] * 2, [[1.0, 2.0]] * 2,
              id=np.array([9]))
    _tracklet(folder, "a.npz", [0, 2], [[5.0, 6.0]] * 2, [[5.0, 6.0]] * 2,
              id=np.array([3]))

    boxes, _size = trex_tracklets.read_tracklets(folder)

    assert [(box["frame"], box["track_id"]) for box in boxes] == \
        [(0, 3), (0, 9), (2, 3), (2, 9)]


def test_the_id_falls_back_to_the_file_name(tmp_path):
    folder = str(tmp_path)
    _tracklet(folder, "video_id12.npz", [0], [[1.0, 2.0]], [[1.0, 2.0]])
    boxes, _size = trex_tracklets.read_tracklets(folder)
    assert boxes[0]["track_id"] == 12


def test_a_file_name_without_a_number_still_gets_an_id(tmp_path):
    folder = str(tmp_path)
    _tracklet(folder, "idle.npz", [0], [[1.0, 2.0]], [[1.0, 2.0]])
    boxes, _size = trex_tracklets.read_tracklets(folder)
    assert boxes[0]["track_id"] == 0


def test_points_trex_did_not_see_are_left_out(tmp_path):
    """A frame with no finite point has no box; one with some has theirs."""
    nan = float("nan")
    folder = str(tmp_path)
    _tracklet(folder, "a.npz", [0, 1, 2],
              [[nan, nan], [10.0, nan], [10.0, 20.0]],
              [[nan, nan], [30.0, nan], [30.0, 50.0]],
              id=np.array([1]), confidence=[nan, 0.5, 0.6],
              classes=[nan, 2, 2])

    boxes, _size = trex_tracklets.read_tracklets(folder)

    assert [box["frame"] for box in boxes] == [1, 2]
    assert (boxes[0]["x1"], boxes[0]["x2"]) == (10.0, 10.0)
    assert boxes[0]["confidence"] == 0.5 and boxes[0]["class_id"] == 2
    assert (boxes[1]["x1"], boxes[1]["y2"]) == (10.0, 50.0)


def test_a_tracklet_without_pose_points_is_skipped_and_said(tmp_path):
    folder = str(tmp_path)
    np.savez(os.path.join(folder, "empty.npz"), id=np.array([1]),
             frame=np.array([0]), detection_p=np.array([1.0]),
             detection_class=np.array([0.0]))
    _tracklet(folder, "ok.npz", [0], [[1.0, 2.0]], [[1.0, 2.0]],
              id=np.array([2]))
    said = []

    boxes, _size = trex_tracklets.read_tracklets(folder, said.append)

    assert [box["track_id"] for box in boxes] == [2]
    assert any("empty.npz has no pose points" in line for line in said)


def test_an_x_series_without_its_y_is_not_a_point(tmp_path):
    folder = str(tmp_path)
    path = _tracklet(folder, "a.npz", [0], [[1.0, 2.0]], [[3.0, 4.0]],
                     id=np.array([1]))
    data = dict(np.load(path))
    data["poseX7"] = np.array([999.0])      # no poseY7
    data["poseXcenter"] = np.array([999.0])
    np.savez(path, **data)

    boxes, _size = trex_tracklets.read_tracklets(folder)
    assert boxes[0]["x2"] == 2.0


def test_a_folder_without_tracklets_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        trex_tracklets.read_tracklets(str(tmp_path))


# ---------------------------------------------------------------------------
# The size of the frames
# ---------------------------------------------------------------------------

def test_the_extracted_frames_decide_the_size():
    assert trex_tracklets.frame_size_for((640, 512), (512, 512), True) == \
        (512, 512)


def test_without_frames_the_extraction_setting_does():
    assert trex_tracklets.frame_size_for((640, 512), None, True) == (640, 512)
    assert trex_tracklets.frame_size_for((640, 512), None, False) == \
        (512, 512)
    assert trex_tracklets.frame_size_for(None, None, False) is None


# ---------------------------------------------------------------------------
# From the video's pixels to the frames'
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")

MATRIX = [[520.0, 0.0, 322.0], [0.0, 518.0, 254.0], [0.0, 0.0, 1.0]]
DISTORTION = [-0.25, 0.08, 0.001, -0.0005, 0.0]


def _raw(points, undistorter):
    """Where points of the frame are on the raw video (the lens forwards)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    new = undistorter._new_matrix
    rays = np.column_stack([(points[:, 0] - new[0, 2]) / new[0, 0],
                            (points[:, 1] - new[1, 2]) / new[1, 1],
                            np.ones(len(points))])
    raw, _jac = cv2.projectPoints(rays, np.zeros(3), np.zeros(3),
                                  np.asarray(MATRIX), np.asarray(DISTORTION))
    return raw.reshape(-1, 2)


@pytest.mark.parametrize("frame_size", [(640, 512), (512, 512)])
def test_a_point_lands_where_the_extraction_puts_its_pixel(frame_size):
    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (640, 512), frame_size)
    in_frame = np.array([[100.0, 90.0], [256.0, 256.0], [430.0, 400.0]])

    moved = undistorter.points(_raw(in_frame, undistorter))

    assert moved == pytest.approx(in_frame, abs=0.05)


@pytest.mark.parametrize("frame_size", [(640, 512), (512, 512)])
def test_the_frames_have_one_focal_length(frame_size):
    """Whatever their shape - the extraction equalises it always; the
    import did so for square frames only."""
    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (640, 512), frame_size)
    assert undistorter._new_matrix[0, 0] == undistorter._new_matrix[1, 1]


@pytest.mark.parametrize("frame_size", [(640, 512), (512, 512)])
def test_it_is_the_detection_frameworks_recipe(frame_size):
    """The extraction and this import must share one matrix; the framework
    states it in ``bambi.geo.calibration``."""
    calibration = pytest.importorskip("bambi.geo.calibration")
    if not hasattr(calibration, "undistort_boxes"):
        pytest.skip("this bambi_detection has no undistort_boxes yet")
    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (640, 512), frame_size)
    boxes = [[100.0, 80.0, 180.0, 150.0], [500.0, 100.0, 540.0, 140.0],
             [20.0, 400.0, 60.0, 480.0]]

    expected = calibration.undistort_boxes(
        boxes, MATRIX, DISTORTION, (640, 512), frame_size)

    for box, wanted in zip(boxes, expected):
        assert undistorter.box(*box) == pytest.approx(tuple(wanted), abs=1e-3)


def test_a_box_stays_a_box_around_its_corners():
    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (640, 512), (512, 512))
    corners = undistorter.points([[100, 80], [180, 80], [180, 150], [100, 150]])

    box = undistorter.box(100, 80, 180, 150)

    assert box == pytest.approx((corners[:, 0].min(), corners[:, 1].min(),
                                 corners[:, 0].max(), corners[:, 1].max()))
    assert box[0] < box[2] and box[1] < box[3]


def test_the_frames_pixels_are_not_the_videos():
    """What storing the box as TRex gave it gets wrong: 640x512 video,
    512x512 frames - a box moves by tens of pixels."""
    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (640, 512), (512, 512))
    x1, _y1, _x2, _y2 = undistorter.box(500, 100, 540, 140)
    assert abs(x1 - 500) > 30


def test_to_frames_copies_and_keeps_the_rest():
    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (640, 512), (512, 512))
    given = [{"frame": 3, "track_id": 5, "x1": 100.0, "y1": 80.0,
              "x2": 180.0, "y2": 150.0, "confidence": 0.7, "class_id": 1}]

    moved = trex_tracklets.to_frames(given, undistorter)

    assert given[0]["x1"] == 100.0                       # untouched
    assert (moved[0]["x1"], moved[0]["y1"], moved[0]["x2"], moved[0]["y2"]) \
        == pytest.approx(undistorter.box(100, 80, 180, 150))
    assert {key: moved[0][key] for key in
            ("frame", "track_id", "confidence", "class_id")} == \
        {"frame": 3, "track_id": 5, "confidence": 0.7, "class_id": 1}


def test_boxes_made_on_the_frames_are_taken_as_they_are():
    given = [{"frame": 0, "track_id": 1, "x1": 1.0, "y1": 2.0, "x2": 3.0,
              "y2": 4.0, "confidence": 1.0, "class_id": 0}]
    moved = trex_tracklets.to_frames(given)
    assert moved == given and moved[0] is not given[0]
