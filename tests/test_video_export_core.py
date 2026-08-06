# -*- coding: utf-8 -*-
"""Unit tests for core.video_export (video creator data loaders, availability
checks and canvas maths).

Everything the overlays draw comes from the store. The pairing strategies that
used to sit below it — a pixel CSV, a geo-coordinate match, a row-index match —
are gone: each could hand a box to the wrong animal when a file was re-sorted
or a detection was dropped (§8.2).
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import detection_store, fov_store
from bambi_wildlife_detection.core import track_store
from bambi_wildlife_detection.core import video_export as ve


def _tracked(root, modality="t"):
    """Two detections in one track, geo-referenced."""
    detection_store.record_detections(root, modality, [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 1, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.8, "source_class": "0"},
    ])
    ids = [d["detection_id"]
           for d in track_store.load_detections(root, modality)]
    track_store.record_georeference(root, modality, [
        {"detection_id": ids[0], "gx1": 100.0, "gy1": 200.0, "gz1": 5.0,
         "gx2": 110.0, "gy2": 210.0, "gz2": 5.0},
        {"detection_id": ids[1], "gx1": 300.0, "gy1": 400.0, "gz1": 5.0,
         "gx2": 310.0, "gy2": 410.0, "gz2": 5.0},
    ])
    track_store.record_tracks(root, modality, [
        {"track_id": 5, "detection_id": ids[0]},
        {"track_id": 5, "detection_id": ids[1]},
    ])
    return ids


class TestColors:
    def test_track_color_bgr_reverses_palette_rgb(self):
        r, g, b = ve.TRACK_COLORS[0]
        assert ve.track_color_bgr(0) == (b, g, r)
        assert ve.track_color_bgr(len(ve.TRACK_COLORS)) == (b, g, r)

    def test_id_to_color_is_deterministic(self):
        assert ve.id_to_color("track-1") == ve.id_to_color("track-1")
        assert ve.id_to_color("track-1") != ve.id_to_color("track-2")
        assert all(0 <= c <= 255 for c in ve.id_to_color(42))


class TestPixelTracksAvailable:
    def test_a_pixel_csv_is_not_enough(self, tmp_path):
        """It has ids but no way to tie them to detections, which is what
        made the old pairing guesswork."""
        (tmp_path / "tracks_t").mkdir()
        (tmp_path / "tracks_t" / "tracks_pixel.csv").write_text("")
        assert ve.pixel_tracks_available(str(tmp_path), "t") is False

    def test_the_legacy_pair_is_not_enough_either(self, tmp_path):
        (tmp_path / "detections_t").mkdir()
        (tmp_path / "detections_t" / "detections.txt").write_text("")
        (tmp_path / "tracks_t").mkdir()
        (tmp_path / "tracks_t" / "tracks.csv").write_text("")
        assert ve.pixel_tracks_available(str(tmp_path), "t") is False

    def test_a_track_store_is(self, tmp_path):
        _tracked(str(tmp_path))
        assert ve.pixel_tracks_available(str(tmp_path), "t") is True

    def test_nothing_present(self, tmp_path):
        assert ve.pixel_tracks_available(str(tmp_path), "w") is False


def _params(target, **overrides):
    params = {
        "target": target,
        "video_suffixes": [],
        "source": "extracted",
        "ortho_kind": "geotiffs",
        "overlay": "none",
        "map": False,
        "map_camera": "t",
        "map_flight": False, "map_fov": False, "map_det": False,
        "map_trk": False, "map_perp": False,
        "info": False,
        "info_camera": "t",
        "info_dets": False, "info_tracks": False, "info_area": False,
    }
    params.update(overrides)
    return params


class TestAvailabilityWarnings:
    def test_extracted_frames_missing(self, tmp_path):
        warns = ve.availability_warnings(
            _params(str(tmp_path), video_suffixes=["t"]))
        assert len(warns) == 1
        assert "extracted frames missing" in warns[0]

    def test_extracted_frames_present(self, tmp_path):
        (tmp_path / "poses_t.json").write_text("{}")
        (tmp_path / "frames_t").mkdir()
        warns = ve.availability_warnings(
            _params(str(tmp_path), video_suffixes=["t"]))
        assert warns == []

    def test_alfs_ortho_missing(self, tmp_path):
        warns = ve.availability_warnings(_params(
            str(tmp_path), video_suffixes=["w"],
            source="ortho", ortho_kind="alfs"))
        assert "ALFS orthophoto missing" in warns[0]

    def test_detections_overlay_missing(self, tmp_path):
        (tmp_path / "poses_t.json").write_text("{}")
        (tmp_path / "frames_t").mkdir()
        warns = ve.availability_warnings(_params(
            str(tmp_path), video_suffixes=["t"], overlay="detections"))
        assert any("detections overlay" in w for w in warns)

    def test_the_warning_names_the_step_not_the_file(self, tmp_path):
        """A 5.x project has detections.txt and still cannot be drawn from
        it, so pointing at the file would send someone the wrong way."""
        (tmp_path / "poses_t.json").write_text("{}")
        (tmp_path / "frames_t").mkdir()
        (tmp_path / "detections_t").mkdir()
        (tmp_path / "detections_t" / "detections.txt").write_text("0 1 2 3 4 5")

        warns = ve.availability_warnings(_params(
            str(tmp_path), video_suffixes=["t"], overlay="detections"))
        assert any("Detect Animals" in w for w in warns)

    def test_a_stored_overlay_raises_no_warning(self, tmp_path):
        (tmp_path / "poses_t.json").write_text("{}")
        (tmp_path / "frames_t").mkdir()
        _tracked(str(tmp_path))

        warns = ve.availability_warnings(_params(
            str(tmp_path), video_suffixes=["t"], overlay="tracks"))
        assert warns == []

    def test_map_layers_missing(self, tmp_path):
        warns = ve.availability_warnings(_params(
            str(tmp_path), map=True,
            map_flight=True, map_fov=True, map_det=True,
            map_trk=True, map_perp=True))
        assert len(warns) == 5

    def test_info_panel_missing(self, tmp_path):
        warns = ve.availability_warnings(_params(
            str(tmp_path), info=True,
            info_dets=True, info_tracks=True, info_area=True))
        assert len(warns) == 3


class TestLoaders:
    def test_load_poses(self, tmp_path):
        (tmp_path / "poses_t.json").write_text(json.dumps(
            {"images": [{"imagefile": "a.jpg"}, {"imagefile": "b.jpg"}]}))
        assert ve.load_poses(str(tmp_path), "t") == ["a.jpg", "b.jpg"]
        assert ve.load_poses(str(tmp_path), "w") == []

    def test_load_poses_broken_json_logs(self, tmp_path):
        (tmp_path / "poses_t.json").write_text("{broken")
        logs = []
        assert ve.load_poses(str(tmp_path), "t", log_fn=logs.append) == []
        assert any("could not read" in m for m in logs)

    def test_load_pixel_detections(self, tmp_path):
        root = str(tmp_path)
        _tracked(root)
        out = ve.load_pixel_detections(root, "t")
        assert out[0][0][:4] == (10.0, 20.0, 30.0, 40.0)
        assert len(out[1]) == 1

    def test_pixel_detections_ignore_the_text_file(self, tmp_path):
        det = tmp_path / "detections_t"
        det.mkdir()
        (det / "detections.txt").write_text("0 1 2 3 4 0.9 2")
        assert ve.load_pixel_detections(str(tmp_path), "t") == {}

    def test_load_fov_polygons(self, tmp_path):
        root = str(tmp_path)
        fov_store.record_fov(root, "t", {
            0: [(10.0, 20.0, 0.0), (30.0, 20.0, 0.0),
                (30.0, 40.0, 0.0), (10.0, 40.0, 0.0)],
            1: [(1.0, 2.0, 0.0), (3.0, 4.0, 0.0)],   # 2 points -> dropped
        })
        out = ve.load_fov_polygons(root, "t")
        assert out[0] == [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
        assert 1 not in out

    def test_fov_polygons_ignore_the_text_file(self, tmp_path):
        fov = tmp_path / "fov_t"
        fov.mkdir()
        (fov / "fov_polygons.txt").write_text(
            "0 4 10 20 0 30 20 0 30 40 0 10 40 0")
        assert ve.load_fov_polygons(str(tmp_path), "t") == {}

    def test_load_perpendicular_with_legacy_fallback(self, tmp_path):
        route = tmp_path / "flight_route_t"
        route.mkdir()
        (route / "perpendicular.json").write_text(json.dumps({
            "perpendiculas": [{
                "frame": 3,
                "detection_center": [1.0, 2.0],
                "foot_point": [3.0, 4.0],
                "distance": 5.5,
            }]}))
        out = ve.load_perpendicular(str(tmp_path), "t")
        assert out[3] == [((1.0, 2.0), (3.0, 4.0), 5.5)]

    def test_load_camera_positions(self, tmp_path):
        route = tmp_path / "flight_route_t"
        route.mkdir()
        (route / "camera_positions.geojson").write_text(json.dumps({
            "features": [
                {"geometry": {"coordinates": [10.0, 20.0]},
                 "properties": {"frame_idx": 4}},
                {"geometry": {"coordinates": [11.0, 21.0]}, "properties": {}},
            ]}))
        out = ve.load_camera_positions(str(tmp_path), "t")
        assert out[4] == (10.0, 20.0)
        assert out[1] == (11.0, 21.0)    # index fallback

    def test_load_geo_tracks(self, tmp_path):
        root = str(tmp_path)
        _tracked(root)
        out = ve.load_geo_tracks(root, "t")
        assert sorted(out) == [0, 1]
        assert out[0][0]["x1"] == 100.0
        # Both frames belong to the one track, whatever id the store gave it.
        assert out[0][0]["tid"] == out[1][0]["tid"]

    def test_geo_tracks_ignore_the_text_file(self, tmp_path):
        trk = tmp_path / "tracks_t"
        trk.mkdir()
        (trk / "tracks.csv").write_text("0,99,1,2,3,4,5,6,0.9,0,0")
        assert ve.load_geo_tracks(str(tmp_path), "t") == {}

    def test_load_geo_detections(self, tmp_path):
        root = str(tmp_path)
        _tracked(root)
        out = ve.load_geo_detections(root, "t")
        assert sorted(out) == [0, 1]
        assert out[0][0]["x1"] == 100.0
        # A running index stands in for a track id, to colour them apart.
        assert out[0][0]["tid"] == 0

    def test_geo_detections_omit_what_could_not_be_placed(self, tmp_path):
        """A failure has no geo row at all, rather than a sentinel box."""
        root = str(tmp_path)
        detection_store.record_detections(root, "t", [
            {"frame": 0, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
             "confidence": 0.9, "source_class": "0"},
            {"frame": 1, "x1": 5.0, "y1": 6.0, "x2": 7.0, "y2": 8.0,
             "confidence": 0.9, "source_class": "0"},
        ])
        ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
        track_store.record_georeference(
            root, "t",
            [{"detection_id": ids[0], "gx1": 1.0, "gy1": 2.0, "gz1": 0.0,
              "gx2": 3.0, "gy2": 4.0, "gz2": 0.0}],
            [{"detection_id": ids[1], "reason": "beyond_mesh"}])

        assert sorted(ve.load_geo_detections(root, "t")) == [0]


class TestLoadPixelTracks:
    def test_the_store_is_the_only_source(self, tmp_path):
        root = str(tmp_path)
        _tracked(root)
        folder = os.path.join(root, "tracks_t")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tracks_pixel.csv"), "w") as fh:
            fh.write("0,99,1.0,1.0,2.0,2.0,0.9,0,0")

        overlay = ve.load_pixel_tracks(root, "t")
        ids = {row[0] for rows in overlay.values() for row in rows}
        assert len(ids) == 1 and 99 not in ids

    def test_without_a_store_there_is_nothing_to_draw(self, tmp_path):
        """A CSV of ids that cannot be tied to detections is not an overlay."""
        root = str(tmp_path)
        folder = os.path.join(root, "tracks_t")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tracks_pixel.csv"), "w") as fh:
            fh.write("0,99,1.0,1.0,2.0,2.0,0.9,0,0")

        assert ve.load_pixel_tracks(root, "t") == {}

    def test_and_it_says_so(self, tmp_path):
        logs = []
        ve.load_pixel_tracks(str(tmp_path), "t", log_fn=logs.append)
        assert any("Migrate" in m for m in logs)


class TestCanvasMath:
    def test_pad_extent_widens_narrow_data(self):
        # 100x100 data on a 2:1 canvas -> width doubled around the centre
        extent = ve.pad_extent_to_aspect((0, 100, 0, 100), 220, 120, 10)
        min_x, max_x, min_y, max_y = extent
        assert (min_y, max_y) == (0, 100)
        assert max_x - min_x == pytest.approx(200)
        assert (min_x + max_x) / 2 == pytest.approx(50)

    def test_pad_extent_heightens_wide_data(self):
        extent = ve.pad_extent_to_aspect((0, 200, 0, 10), 120, 120, 10)
        min_x, max_x, min_y, max_y = extent
        assert (min_x, max_x) == (0, 200)
        assert max_y - min_y == pytest.approx(200)

    def test_world_to_canvas_corners(self):
        cfg = ve.make_canvas_cfg((0, 100, 0, 100), 220, 220, 10)
        assert ve.world_to_canvas(0, 0, cfg) == (10, 210)      # bottom-left
        assert ve.world_to_canvas(100, 100, cfg) == (210, 10)  # top-right

    def test_fmt_area_units(self):
        assert ve.fmt_area(2_500_000) == "2.50 km2"
        assert ve.fmt_area(25_000) == "2.50 ha"
        assert ve.fmt_area(500) == "500 m2"

    def test_fmt_area_ratio_uses_total_unit(self):
        s = ve.fmt_area_ratio(500_000, 2_000_000)
        assert s == "0.50 / 2.00 km2 (25%)"
        assert ve.fmt_area_ratio(0, 0).endswith("(0%)")


class TestPixelTracksFromStore:
    """The video creator reads the store rather than reconstructing (§8.2).

    Steps 1-3 of load_pixel_tracks all recover something the pipeline knew and
    threw away; the store answers directly, so they only run for projects that
    have none.
    """

    @staticmethod
    def _project(root):
        from bambi_wildlife_detection.core import detection_store, track_store

        detection_store.record_detections(root, "t", [
            {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
             "confidence": 0.9, "source_class": "0"},
            {"frame": 1, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
             "confidence": 0.8, "source_class": "0"},
        ])
        ids = [d["detection_id"]
               for d in track_store.load_detections(root, "t")]
        track_store.record_tracks(root, "t", [
            {"track_id": 5, "detection_id": ids[0]},
            {"track_id": 5, "detection_id": ids[1]},
        ])
        return ids

    def test_reads_membership_from_the_store(self, tmp_path):
        from bambi_wildlife_detection.core import video_export

        root = str(tmp_path)
        self._project(root)
        overlay = video_export.pixel_tracks_from_store(root, "t")
        assert sorted(overlay) == [0, 1]
        assert overlay[0][0][1:] == (10.0, 20.0, 30.0, 40.0)

    def test_load_pixel_tracks_prefers_the_store(self, tmp_path):
        """Even with a legacy CSV present, the exact answer wins."""
        import os

        from bambi_wildlife_detection.core import video_export

        root = str(tmp_path)
        self._project(root)
        folder = os.path.join(root, "tracks_t")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tracks_pixel.csv"), "w") as fh:
            fh.write("0,99,1.0,1.0,2.0,2.0,0.9,0,0\n")

        overlay = video_export.load_pixel_tracks(root, "t")
        assert {row[0] for rows in overlay.values() for row in rows} != {99}

    def test_without_a_store_there_is_nothing(self, tmp_path):
        root = str(tmp_path)
        folder = os.path.join(root, "tracks_t")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tracks_pixel.csv"), "w") as fh:
            fh.write("0,99,1.0,1.0,2.0,2.0,0.9,0,0")

        assert ve.load_pixel_tracks(root, "t") == {}

    def test_unknown_modality_is_ignored(self, tmp_path):
        from bambi_wildlife_detection.core import video_export

        assert video_export.pixel_tracks_from_store(str(tmp_path), "x") == {}

    def test_a_dropped_detection_does_not_cost_its_frame_mates(self, tmp_path):
        """What the positional pairing got wrong (§8.2)."""
        from bambi_wildlife_detection.core import (
            detection_store, track_store, video_export)

        root = str(tmp_path)
        detection_store.record_detections(root, "t", [
            {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
             "confidence": 0.9, "source_class": "0"},
            {"frame": 0, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
             "confidence": 0.8, "source_class": "0"},
        ])
        ids = [d["detection_id"]
               for d in track_store.load_detections(root, "t")]
        # Only one of the two frame-0 detections made it into a track.
        track_store.record_tracks(
            root, "t", [{"track_id": 5, "detection_id": ids[0]}])

        overlay = video_export.pixel_tracks_from_store(root, "t")
        # The tracked detection keeps its overlay; the untracked one simply has
        # no id to draw. The old positional pairing lost both.
        assert len(overlay[0]) == 1
        assert overlay[0][0][1:] == (10.0, 20.0, 30.0, 40.0)
