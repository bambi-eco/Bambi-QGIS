# -*- coding: utf-8 -*-
"""Unit tests for track gap interpolation (pixel-space and geo-space)."""
from dataclasses import dataclass


@dataclass
class PixelTrack:
    frame: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    interpolated: int = 0


@dataclass
class GeoDetection:
    source_id: int
    frame: int
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    conf: float
    cls: int
    interpolated: int = 0


def _pixel_track(frame, track_id=7, x=0.0, conf=0.9):
    return PixelTrack(frame=frame, track_id=track_id,
                      x1=x, y1=x + 1, x2=x + 2, y2=x + 3, conf=conf, cls=1)


class TestInterpolatePixelTracks:
    def test_fills_gap_linearly(self, processor):
        tracks = [_pixel_track(0, x=0.0, conf=0.8), _pixel_track(3, x=3.0, conf=0.6)]
        out = processor._interpolate_pixel_tracks(tracks, PixelTrack)

        assert [t.frame for t in out] == [0, 1, 2, 3]
        interp1 = out[1]
        assert interp1.x1 == 1.0
        assert interp1.y1 == 2.0
        assert interp1.x2 == 3.0
        assert interp1.y2 == 4.0
        # Interpolated frames get the mean confidence and are flagged.
        assert interp1.conf == 0.7
        assert interp1.interpolated == 1
        assert out[0].interpolated == 0
        assert out[3].interpolated == 0

    def test_consecutive_frames_unchanged(self, processor):
        tracks = [_pixel_track(0), _pixel_track(1)]
        out = processor._interpolate_pixel_tracks(tracks, PixelTrack)
        assert [t.frame for t in out] == [0, 1]

    def test_single_detection_track_passthrough(self, processor):
        tracks = [_pixel_track(5)]
        out = processor._interpolate_pixel_tracks(tracks, PixelTrack)
        assert out == tracks

    def test_unsorted_input_is_sorted_per_track(self, processor):
        tracks = [_pixel_track(3, x=3.0), _pixel_track(0, x=0.0)]
        out = processor._interpolate_pixel_tracks(tracks, PixelTrack)
        assert [t.frame for t in out] == [0, 1, 2, 3]

    def test_independent_tracks_do_not_interpolate_across_ids(self, processor):
        tracks = [_pixel_track(0, track_id=1), _pixel_track(10, track_id=2)]
        out = processor._interpolate_pixel_tracks(tracks, PixelTrack)
        assert len(out) == 2  # no frames were invented between the two tracks


class TestInterpolateTracks:
    @staticmethod
    def _det(frame, x=0.0, z=100.0, conf=0.8, cls=2):
        return GeoDetection(source_id=1, frame=frame,
                            x1=x, y1=x + 10, z1=z, x2=x + 1, y2=x + 11, z2=z + 1,
                            conf=conf, cls=cls)

    def test_fills_gap_including_z(self, processor):
        results = [(0, 5, self._det(0, x=0.0, z=100.0)),
                   (4, 5, self._det(4, x=4.0, z=104.0))]
        out = processor._interpolate_tracks(results, GeoDetection)

        assert [frame for frame, _, _ in out] == [0, 1, 2, 3, 4]
        frame2 = out[2][2]
        assert frame2.x1 == 2.0
        assert frame2.z1 == 102.0
        assert frame2.z2 == 103.0
        assert frame2.interpolated == 1
        assert frame2.source_id == -1  # synthetic detections are marked

    def test_output_sorted_by_frame_then_track(self, processor):
        results = [
            (5, 2, self._det(5)),
            (0, 1, self._det(0)),
            (5, 1, self._det(5)),
            (0, 2, self._det(0)),
        ]
        out = processor._interpolate_tracks(results, GeoDetection)
        keys = [(frame, tid) for frame, tid, _ in out]
        assert keys == sorted(keys)

    def test_single_detection_track_passthrough(self, processor):
        det = self._det(3)
        out = processor._interpolate_tracks([(3, 9, det)], GeoDetection)
        assert out == [(3, 9, det)]
