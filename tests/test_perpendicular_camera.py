# -*- coding: utf-8 -*-
"""Perpendicular distances use the poses of the camera the boxes are from.

The distance of an animal to the flight line is measured to the part of the
route its frame saw, which takes the camera position of that frame. Both
perpendicular steps read the first poses file they found - the thermal one -
whichever camera the boxes came from, so the frame number of an RGB box
picked an unrelated thermal frame and the distance was measured from there.
"""
import json
import os

import pytest

from bambi_wildlife_detection import bambi_processing
from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import detection_store, store, track_store


def _pose(x, y):
    return {"imagefile": f"{int(x)}_{int(y)}.jpg", "location": [x, y, 30.0],
            "rotation": [0.0, 0.0, 0.0], "fovy": [50.0]}


@pytest.fixture
def flight(tmp_path):
    """A route of two legs 100 m apart. The thermal camera's frame 0 looks
    at the first leg, the RGB camera's frame 0 at the second - where an
    animal stands 10 m off the line."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    os.makedirs(os.path.join(root, "flight_route_t"))
    with open(os.path.join(root, "flight_route_t", "flight_route.geojson"),
              "w") as fh:
        json.dump({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "LineString", "coordinates": [
                [0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]}}]},
            fh)

    with open(os.path.join(root, "poses_t.json"), "w") as fh:
        json.dump({"images": [_pose(50.0, 0.0)]}, fh)
    with open(os.path.join(root, "poses_w.json"), "w") as fh:
        json.dump({"images": [_pose(50.0, 100.0)]}, fh)

    for modality in ("t", "w"):
        detection_store.record_detections(root, modality, [
            {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
             "confidence": 0.9, "source_class": "0"}])
        (row,) = track_store.load_detections(root, modality)
        # the thermal animal 10 m off the first leg, the RGB one 10 m off
        # the second
        y = 10.0 if modality == "t" else 90.0
        track_store.record_georeference(root, modality, [{
            "detection_id": row["detection_id"],
            "gx1": 49.0, "gy1": y - 1.0, "gz1": 0.0,
            "gx2": 51.0, "gy2": y + 1.0, "gz2": 0.0}])
        track_store.record_tracks(
            root, modality,
            [{"track_id": 1, "detection_id": row["detection_id"]}],
            kind="builtin")
    return root


def _config(root, camera):
    return {"target_folder": root, "target_epsg": 32633,
            "flight_route_camera": "T", "detection_camera": camera,
            "tracking_camera": camera}


@pytest.mark.parametrize("camera, modality", [("T", "t"), ("W", "w")])
def test_a_box_is_measured_from_its_own_cameras_frame(flight, camera,
                                                      modality):
    BambiProcessor().run_perpendicular(_config(flight, camera))

    with open(os.path.join(flight, "flight_route_t",
                           f"perpendicular_{modality}.json")) as fh:
        (result,) = json.load(fh)["perpendiculas"]
    assert result["distance_m"] == pytest.approx(10.0)

    with open(os.path.join(flight, "flight_route_t",
                           f"perpendicular_by_image_{modality}.json")) as fh:
        by_image = json.load(fh)
    assert list(by_image) == ["50_0.jpg" if modality == "t" else "50_100.jpg"]


@pytest.mark.parametrize("camera, modality", [("T", "t"), ("W", "w")])
def test_a_track_is_measured_from_its_own_cameras_frame(flight, camera,
                                                        modality):
    BambiProcessor().run_track_perpendicular(_config(flight, camera))

    with open(os.path.join(flight, "flight_route_t",
                           f"perpendicular_tracks_{modality}.json")) as fh:
        document = json.load(fh)
    results = document.get("perpendiculas") or document.get("tracks") or []
    assert len(results) == 1
    assert results[0]["distance_m"] == pytest.approx(10.0)


class TestCameraFrames:
    def test_the_named_cameras_file_is_read(self, flight):
        images, names = bambi_processing._camera_frames(flight, "w")
        assert images[0]["location"][:2] == [50.0, 100.0]
        assert names == {0: "50_100.jpg"}
        images, names = bambi_processing._camera_frames(flight, "t")
        assert names == {0: "50_0.jpg"}

    def test_the_other_cameras_file_is_no_substitute(self, flight):
        os.remove(os.path.join(flight, "poses_w.json"))
        log = []
        assert bambi_processing._camera_frames(
            flight, "w", log.append) == ([], {})
        assert any("poses_w.json not found" in line for line in log)

    def test_an_unreadable_file(self, flight):
        with open(os.path.join(flight, "poses_w.json"), "w") as fh:
            fh.write("{ not json")
        log = []
        assert bambi_processing._camera_frames(
            flight, "w", log.append) == ([], {})
        assert any("could not read poses_w.json" in line for line in log)
