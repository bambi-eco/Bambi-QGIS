# -*- coding: utf-8 -*-
"""Unit tests for core.track_export (pixel MOT file from geo-referenced tracks)."""
from bambi_wildlife_detection.core.track_export import write_pixel_tracks_from_geo


def _write(tmp_path, rel, content):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _full_flight(tmp_path):
    """A minimal but coherent detections/georeferenced/tracks triplet."""
    _write(tmp_path, "detections_t/detections.txt",
           "# frame x1 y1 x2 y2 conf cls\n"
           "0 10 10 20 20 0.90 1\n"
           "0 30 30 40 40 0.80 1\n"
           "1 11 11 21 21 0.70 2\n")
    _write(tmp_path, "georeferenced_t/georeferenced.txt",
           "# idx frame x1 y1 z1 x2 y2 z2 conf cls\n"
           "0 0 100.0 200.0 5 101.0 201.0 5 0.90 1\n"
           "1 0 110.0 210.0 5 111.0 211.0 5 0.80 1\n"
           "2 1 102.0 202.0 5 103.0 203.0 5 0.70 2\n")
    # tracks.csv is re-sorted by (frame, track_id) and includes an
    # interpolated row that must be ignored.
    _write(tmp_path, "tracks_t/tracks.csv",
           "0,7,100.0,200.0,5,101.0,201.0,5,0.90,1,0\n"
           "0,8,110.0,210.0,5,111.0,211.0,5,0.80,1,0\n"
           "1,7,102.0,202.0,5,103.0,203.0,5,0.70,2,0\n"
           "2,7,999.0,999.0,5,999.0,999.0,5,0.50,2,1\n")


class TestWritePixelTracksFromGeo:
    def test_reconstructs_pixel_tracks(self, tmp_path):
        _full_flight(tmp_path)
        logs = []
        out = write_pixel_tracks_from_geo(str(tmp_path), "t", log_fn=logs.append)
        assert out.endswith("tracks_t/tracks_pixel.csv") or out.endswith(
            "tracks_t\\tracks_pixel.csv")

        lines = [ln for ln in (tmp_path / "tracks_t" / "tracks_pixel.csv")
                 .read_text().splitlines() if not ln.startswith("#")]
        # frame 0: two boxes matched to tracks 7 and 8; frame 1: track 7
        assert lines == [
            "0,7,10.00,10.00,20.00,20.00,0.9000,1,0",
            "0,8,30.00,30.00,40.00,40.00,0.8000,1,0",
            "1,7,11.00,11.00,21.00,21.00,0.7000,2,0",
        ]
        assert any("3 rows" in m for m in logs)

    def test_noop_when_pixel_file_exists(self, tmp_path):
        _full_flight(tmp_path)
        _write(tmp_path, "tracks_t/tracks_pixel.csv", "# pre-existing\n")
        assert write_pixel_tracks_from_geo(str(tmp_path), "t") == ""
        assert (tmp_path / "tracks_t" / "tracks_pixel.csv").read_text() == \
            "# pre-existing\n"

    def test_missing_inputs_returns_empty(self, tmp_path):
        assert write_pixel_tracks_from_geo(str(tmp_path), "t") == ""

    def test_frame_count_mismatch_skipped(self, tmp_path):
        # frame 0 has 2 detections but only 1 geo row -> cannot align safely
        _write(tmp_path, "detections_t/detections.txt",
               "0 10 10 20 20 0.9 1\n0 30 30 40 40 0.8 1\n")
        _write(tmp_path, "georeferenced_t/georeferenced.txt",
               "0 0 100.0 200.0 5 101.0 201.0 5 0.9 1\n")
        _write(tmp_path, "tracks_t/tracks.csv",
               "0,7,100.0,200.0,5,101.0,201.0,5,0.9,1,0\n")
        logs = []
        out = write_pixel_tracks_from_geo(str(tmp_path), "t", log_fn=logs.append)
        assert out == ""      # no rows written
        assert not (tmp_path / "tracks_t" / "tracks_pixel.csv").exists()

    def test_camera_suffix_respected(self, tmp_path):
        _full_flight(tmp_path)   # builds _t folders only
        assert write_pixel_tracks_from_geo(str(tmp_path), "w") == ""
