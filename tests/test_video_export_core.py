# -*- coding: utf-8 -*-
"""Unit tests for core.video_export (video creator data loaders, track
pairing strategies, availability checks and canvas maths)."""
import json

import pytest

from bambi_wildlife_detection.core import video_export as ve


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
    def test_direct_pixel_csv(self, tmp_path):
        (tmp_path / "tracks_t").mkdir()
        (tmp_path / "tracks_t" / "tracks_pixel.csv").write_text("")
        assert ve.pixel_tracks_available(str(tmp_path), "t") is True

    def test_fallback_needs_detections_and_tracks(self, tmp_path):
        (tmp_path / "detections_t").mkdir()
        (tmp_path / "detections_t" / "detections.txt").write_text("")
        assert ve.pixel_tracks_available(str(tmp_path), "t") is False
        (tmp_path / "tracks_t").mkdir()
        (tmp_path / "tracks_t" / "tracks.csv").write_text("")
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
        det = tmp_path / "detections_t"
        det.mkdir()
        (det / "detections.txt").write_text(
            "# h\n0 1 2 3 4 0.9 2\n0 5 6 7 8 0.9\n1 1 1 2 2 0.8 1\n")
        out = ve.load_pixel_detections(str(tmp_path), "t")
        assert out[0][0] == (1.0, 2.0, 3.0, 4.0, "2")
        assert out[0][1][4] == "0"          # missing class defaults to "0"
        assert len(out[1]) == 1

    def test_load_detection_rows_in_file_order(self, tmp_path):
        det = tmp_path / "detections_t"
        det.mkdir()
        (det / "detections.txt").write_text(
            "5 1 1 2 2 0.9 1\n0 3 3 4 4 0.9 1\nnot a row\n")
        rows = ve.load_detection_rows(str(tmp_path), "t")
        assert [r[0] for r in rows] == [5, 0]

    def test_load_track_id_rows_prefers_pixel_and_skips_interpolated(self, tmp_path):
        trk = tmp_path / "tracks_t"
        trk.mkdir()
        (trk / "tracks.csv").write_text("0,99,0,0,0,1,1,1,0.9,1,0\n")
        (trk / "tracks_pixel.csv").write_text(
            "0,7,1,1,2,2,0.9,1,0\n"
            "1,7,1,1,2,2,0.9,1,1\n"     # interpolated -> skipped
            "2,8,1,1,2,2,0.9,1\n")
        rows = ve.load_track_id_rows(str(tmp_path), "t")
        assert rows == [(0, 7), (2, 8)]

    def test_load_georef_rows(self, tmp_path):
        geo = tmp_path / "georeferenced_t"
        geo.mkdir()
        (geo / "georeferenced.txt").write_text(
            "# h\n0 3 100.0 200.0 5.0 101.0 201.0 5.0 0.9 1\n")
        rows = ve.load_georef_rows(str(tmp_path), "t")
        assert rows == [(3, 100.0, 200.0, 101.0, 201.0)]

    def test_parse_pixel_tracks_csv(self, tmp_path):
        f = tmp_path / "tracks_pixel.csv"
        f.write_text(
            "0,7,1,2,3,4,0.9,1,0\n"
            "0,8,5,6,7,8,0.9,1,1\n")     # interpolated -> skipped
        out = ve.parse_pixel_tracks_csv(str(f), 8)
        assert out == {0: [(7, 1.0, 2.0, 3.0, 4.0)]}

    def test_coord_key_rounds(self):
        assert ve.coord_key((1.00049, 2.0)) == (1.0, 2.0)
        assert ve.coord_key((1.0006, 2.0)) == (1.001, 2.0)

    def test_load_fov_polygons(self, tmp_path):
        fov = tmp_path / "fov_t"
        fov.mkdir()
        (fov / "fov_polygons.txt").write_text(
            "# h\n"
            "0 4 10 20 0 30 20 0 30 40 0 10 40 0\n"
            "1 2 1 2 0 3 4 0\n")        # only 2 points -> dropped
        out = ve.load_fov_polygons(str(tmp_path), "t")
        assert out[0] == [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
        assert 1 not in out

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
        trk = tmp_path / "tracks_t"
        trk.mkdir()
        (trk / "tracks.csv").write_text("0,7,100,200,5,101,201,5,0.9,1\n")
        out = ve.load_geo_tracks(str(tmp_path), "t")
        assert out[0][0] == {"tid": 7, "x1": 100.0, "y1": 200.0,
                             "x2": 101.0, "y2": 201.0}

    def test_load_geo_detections_filters_negative(self, tmp_path):
        geo = tmp_path / "georeferenced_t"
        geo.mkdir()
        (geo / "georeferenced.txt").write_text(
            "0 0 100 200 5 101 201 5 0.9 1\n"
            "1 0 -1 200 5 101 201 5 0.9 1\n"     # negative -> dropped
            "2 0 110 210 5 111 211 5 0.9 1\n")
        out = ve.load_geo_detections(str(tmp_path), "t")
        assert [d["tid"] for d in out[0]] == [0, 1]   # running pseudo ids
        assert out[0][1]["x1"] == 110.0


class TestLoadPixelTracksStrategies:
    def _write(self, tmp_path, rel, content):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_direct_pixel_csv_wins(self, tmp_path):
        self._write(tmp_path, "tracks_t/tracks_pixel.csv",
                    "0,7,1,2,3,4,0.9,1,0\n")
        out = ve.load_pixel_tracks(str(tmp_path), "t")
        assert out == {0: [(7, 1.0, 2.0, 3.0, 4.0)]}

    def test_geo_join_assigns_track_ids(self, tmp_path):
        self._write(tmp_path, "detections_t/detections.txt",
                    "0 10 10 20 20 0.9000 1\n"
                    "0 30 30 40 40 0.8000 1\n"
                    "1 11 11 21 21 0.9000 1\n")
        self._write(tmp_path, "georeferenced_t/georeferenced.txt",
                    "0 0 100.0 200.0 5 101.0 201.0 5 0.9 1\n"
                    "1 0 110.0 210.0 5 111.0 211.0 5 0.8 1\n"
                    "2 1 102.0 202.0 5 103.0 203.0 5 0.9 1\n")
        self._write(tmp_path, "tracks_t/tracks.csv",
                    "1,9,102.0,202.0,5,103.0,203.0,5,0.9,1,0\n"
                    "0,7,100.0,200.0,5,101.0,201.0,5,0.9,1,0\n"
                    "0,8,110.0,210.0,5,111.0,211.0,5,0.8,1,0\n")
        out = ve.load_pixel_tracks(str(tmp_path), "t")
        assert out[0] == [(7, 10.0, 10.0, 20.0, 20.0),
                          (8, 30.0, 30.0, 40.0, 40.0)]
        assert out[1] == [(9, 11.0, 11.0, 21.0, 21.0)]

    def test_geo_join_skips_misaligned_frames(self, tmp_path):
        # Frame 0 has 2 detections but only 1 geo row -> cannot align safely
        self._write(tmp_path, "detections_t/detections.txt",
                    "0 10 10 20 20 0.9 1\n0 30 30 40 40 0.8 1\n")
        self._write(tmp_path, "georeferenced_t/georeferenced.txt",
                    "0 0 100.0 200.0 5 101.0 201.0 5 0.9 1\n")
        self._write(tmp_path, "tracks_t/tracks.csv",
                    "0,7,100.0,200.0,5,101.0,201.0,5,0.9,1,0\n")
        assert ve.pair_tracks_via_geo(str(tmp_path), "t") == {}

    def test_line_index_fallback_logs_skew(self, tmp_path):
        self._write(tmp_path, "detections_t/detections.txt",
                    "0 10 10 20 20 0.9 1\n1 30 30 40 40 0.8 1\n")
        # No georeferenced.txt -> geo join impossible; tracks.csv rows align
        # by line index except the second (frame mismatch).
        self._write(tmp_path, "tracks_t/tracks.csv",
                    "0,7,0,0,0,1,1,1,0.9,1,0\n"
                    "5,8,0,0,0,1,1,1,0.8,1,0\n")
        logs = []
        out = ve.load_pixel_tracks(str(tmp_path), "t", log_fn=logs.append)
        assert out == {0: [(7, 10.0, 10.0, 20.0, 20.0)]}
        assert any("out of sync" in m for m in logs)


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

    def test_falls_back_without_a_store(self, tmp_path):
        import os

        from bambi_wildlife_detection.core import video_export

        root = str(tmp_path)
        folder = os.path.join(root, "tracks_t")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tracks_pixel.csv"), "w") as fh:
            fh.write("0,99,1.0,1.0,2.0,2.0,0.9,0,0\n")

        overlay = video_export.load_pixel_tracks(root, "t")
        assert overlay[0][0][0] == 99

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
