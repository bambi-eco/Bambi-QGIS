# -*- coding: utf-8 -*-
"""core.agl_profile - flight altitude over the DEM relief along the route.

The ground is injected: what is under test is the bookkeeping - the DEM
origin added back to mesh-local poses, the distance and time axes, the AGL
arithmetic and the gaps where the DEM has nothing.
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import agl_profile


def _poses(root, modality, entries):
    """``entries``: ``[(x, y, z, timestamp), …]`` in mesh-local metres."""
    with open(os.path.join(root, f"poses_{modality}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"images": [
            {"imagefile": f"{modality}{i}.jpg", "location": [x, y, z],
             "timestamp": stamp}
            for i, (x, y, z, stamp) in enumerate(entries)]}, fh)


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path)
    with open(os.path.join(root, "dem_mesh.json"), "w", encoding="utf-8") as fh:
        json.dump({"origin": [467047.5, 5327459.9, 404.75]}, fh)
    _poses(root, "t", [
        (0.0, 0.0, 100.0, "2026-06-26T08:03:32+00:00"),
        (30.0, 40.0, 101.0, "2026-06-26T08:03:34+00:00"),
        (60.0, 80.0, 99.0, "2026-06-26T08:03:37+00:00"),
    ])
    _poses(root, "w", [(0.0, 0.0, 50.0, "")])
    return root


class TestOrigin:

    def test_found_beside_the_mesh(self, flight):
        mesh = os.path.join(flight, "dem_mesh.glb")
        assert agl_profile.dem_origin(dem_path=mesh) == (467047.5, 5327459.9, 404.75)

    def test_found_in_the_target_folder(self, flight):
        assert agl_profile.dem_origin(target_folder=flight) == (467047.5, 5327459.9, 404.75)

    def test_explicit_json_wins(self, flight, tmp_path):
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"origin": [1, 2, 3]}))
        assert agl_profile.dem_origin(dem_json_path=str(other),
                                      target_folder=flight) == (1.0, 2.0, 3.0)

    def test_nothing_found(self, tmp_path):
        assert agl_profile.dem_origin(target_folder=str(tmp_path)) is None


class TestProfile:

    def test_absolute_altitudes_distance_time_and_agl(self, flight):
        profile = agl_profile.build_profile(
            flight, "t", dem_path=os.path.join(flight, "dem_mesh.glb"),
            ground_fn=lambda xy: [10.0, None, 12.0])
        points = profile["points"]
        assert [p["frame"] for p in points] == [0, 1, 2]
        assert profile["origin"] == (467047.5, 5327459.9, 404.75)
        # Altitude and ground carry the DEM origin height; AGL does not.
        assert points[0]["altitude"] == pytest.approx(504.75)
        assert points[0]["ground"] == pytest.approx(414.75)
        assert points[0]["agl"] == pytest.approx(90.0)
        assert points[1]["ground"] is None and points[1]["agl"] is None
        assert points[2]["agl"] == pytest.approx(87.0)
        # Positions in the project CRS.
        assert points[1]["east"] == pytest.approx(467077.5)
        assert points[1]["north"] == pytest.approx(5327499.9)
        # 3-4-5 triangles: 50 m per leg.
        assert [p["distance_m"] for p in points] == pytest.approx([0.0, 50.0, 100.0])
        assert [p["time_s"] for p in points] == pytest.approx([0.0, 2.0, 5.0])
        assert profile["n_ground"] == 2

    def test_without_an_origin_altitudes_stay_local_and_it_says_so(self, tmp_path):
        root = str(tmp_path)
        _poses(root, "t", [(0.0, 0.0, 100.0, "")])
        profile = agl_profile.build_profile(root, "t", ground_fn=lambda xy: [20.0])
        assert profile["origin"] is None
        assert profile["points"][0]["altitude"] == 100.0
        assert profile["points"][0]["agl"] == 80.0
        assert profile["points"][0]["time_s"] is None
        assert any("no DEM origin" in line for line in agl_profile.summarise(profile))

    def test_the_other_camera_has_its_own_poses(self, flight):
        profile = agl_profile.build_profile(flight, "w", ground_fn=lambda xy: [None])
        assert len(profile["points"]) == 1
        assert profile["points"][0]["altitude"] == pytest.approx(454.75)
        assert profile["n_ground"] == 0
        assert any("No DEM ground" in line for line in agl_profile.summarise(profile))

    def test_poses_without_a_position_are_skipped(self, tmp_path):
        root = str(tmp_path)
        with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
            json.dump({"images": [{"imagefile": "a.jpg"},
                                  {"imagefile": "b.jpg", "location": [1, 2, 3]}]}, fh)
        profile = agl_profile.build_profile(root, "t", ground_fn=lambda xy: [0.0])
        assert [p["frame"] for p in profile["points"]] == [1]

    def test_missing_poses_give_an_empty_profile(self, tmp_path):
        profile = agl_profile.build_profile(str(tmp_path), "t", ground_fn=lambda xy: [])
        assert profile["points"] == []
        assert agl_profile.summarise(profile) == ["No poses with a position."]

    def test_summary_reports_the_agl_range(self, flight):
        profile = agl_profile.build_profile(
            flight, "t", ground_fn=lambda xy: [10.0, 11.0, 12.0])
        lines = agl_profile.summarise(profile)
        assert lines[0] == "3 frame(s), 100 m flown in 5 s"
        assert lines[1].startswith("AGL 87.0-90.0 m, mean 89.0 m over 3 frame(s)")

    def test_without_a_mesh_stack_the_ground_is_unknown(self, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "trimesh", None)
        assert agl_profile.ground_heights("x.glb", [(0.0, 0.0)]) == [None]
        assert agl_profile.ground_heights("", [(0.0, 0.0)]) == [None]
