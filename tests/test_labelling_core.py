# -*- coding: utf-8 -*-
"""Unit tests for core.labelling (labelling tool data model & geometry).

The GUI dialog re-exports everything tested here; these tests import from
the new core location. ``_GeoPropagator`` is exercised with fake
alfspy/trimesh/bambi modules and the identity camera from tests.fakes.
"""
import json
import sys

import pytest

from bambi_wildlife_detection.core.labelling import (
    LabelStore,
    LabelTrack,
    SPECIES_CLASSES,
    TRACK_COLORS_RGB,
    _FrameMatcher,
    _GeoPropagator,
    _load_detections_by_frame,
    _load_pixel_tracks,
    _pose_epochs,
    propagation_frames,
    track_color_rgb,
)
from tests.fakes import install_fake_render_stack, make_module


class TestTrackColor:
    def test_cycles_through_palette(self):
        assert track_color_rgb(0) == TRACK_COLORS_RGB[0]
        assert track_color_rgb(len(TRACK_COLORS_RGB)) == TRACK_COLORS_RGB[0]
        assert track_color_rgb(3) == TRACK_COLORS_RGB[3]


class TestLabelTrack:
    def test_interpolation_between_keyframes(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(10, (100, 50, 110, 60))
        box, is_kf, occ = track.box_at(5)
        assert box == (50.0, 25.0, 60.0, 35.0)
        assert is_kf is False
        assert occ == "none"

    def test_keyframe_returned_exactly(self):
        track = LabelTrack(1)
        track.set_keyframe(3, (1, 2, 3, 4), occlusion="partially")
        box, is_kf, occ = track.box_at(3)
        assert box == (1.0, 2.0, 3.0, 4.0)
        assert is_kf is True
        assert occ == "partially"

    def test_outside_range_returns_none(self):
        track = LabelTrack(1)
        track.set_keyframe(5, (0, 0, 1, 1))
        track.set_keyframe(8, (0, 0, 1, 1))
        assert track.box_at(4) is None
        assert track.box_at(9) is None
        assert LabelTrack(2).box_at(0) is None

    def test_stop_frame_creates_gap(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), stop=True)
        track.set_keyframe(10, (5, 5, 6, 6))
        assert track.box_at(0) is not None      # the stop frame itself has a box
        assert track.box_at(5) is None          # gap after the stop frame
        assert track.box_at(10) is not None

    def test_visible_segments_split_at_stops(self):
        track = LabelTrack(1)
        for f in (0, 5, 10, 20):
            track.set_keyframe(f, (0, 0, 1, 1))
        assert track.visible_segments() == [(0, 20)]
        track.set_keyframe(5, (0, 0, 1, 1), stop=True)
        assert track.visible_segments() == [(0, 5), (10, 20)]

    def test_occlusion_inherited_from_interpolation_state(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), occlusion="partially")
        track.set_keyframe(10, (0, 0, 1, 1), occlusion="fully")
        # a new keyframe in between inherits the interpolated occlusion
        track.set_keyframe(5, (2, 2, 3, 3))
        assert track.keyframes[5]["occlusion"] == "partially"

    def test_box_update_preserves_stop_and_occlusion(self):
        track = LabelTrack(1)
        track.set_keyframe(4, (0, 0, 1, 1), occlusion="fully", stop=True)
        track.set_keyframe(4, (9, 9, 10, 10))  # box-only update
        assert track.is_stop(4) is True
        assert track.keyframes[4]["occlusion"] == "fully"

    def test_remove_keyframe(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1))
        assert track.remove_keyframe(0) is True
        assert track.remove_keyframe(0) is False
        assert track.frame_range() is None

    def test_dict_round_trip(self):
        track = LabelTrack(7, species="red deer", sex="female", age="adult")
        track.set_keyframe(2, (1, 2, 3, 4), occlusion="partially", stop=True)
        track.set_keyframe(9, (5, 6, 7, 8))
        clone = LabelTrack.from_dict(json.loads(json.dumps(track.to_dict())))
        assert clone.track_id == 7
        assert clone.species == "red deer"
        assert clone.keyframes == track.keyframes


class TestLabelStore:
    @pytest.fixture
    def store(self, tmp_path):
        return LabelStore(str(tmp_path), "t")

    def _add_track(self, store, track_id=1, species="red deer"):
        track = LabelTrack(track_id, species=species)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(2, (20, 20, 30, 30))
        store.tracks[track_id] = track
        return track

    def test_save_load_round_trip(self, store, tmp_path):
        self._add_track(store)
        store.save()
        reloaded = LabelStore(str(tmp_path), "t")
        reloaded.load()
        assert reloaded.tracks[1].keyframes == store.tracks[1].keyframes
        assert reloaded.tracks[1].species == "red deer"

    def test_next_track_id(self, store):
        assert store.next_track_id() == 1
        self._add_track(store, 5)
        assert store.next_track_id() == 6

    def test_species_class_ids_appends_custom_species(self, store):
        self._add_track(store, 1, species="red deer")
        self._add_track(store, 2, species="Moose")   # custom, mixed case
        self._add_track(store, 3, species="ibex")    # custom
        mapping = store.species_class_ids()
        assert mapping["red deer"] == SPECIES_CLASSES.index("red deer")
        # customs appended alphabetically after the default taxonomy
        assert mapping["ibex"] == len(SPECIES_CLASSES)
        assert mapping["moose"] == len(SPECIES_CLASSES) + 1

    def test_csv_export_contract(self, store):
        self._add_track(store)
        store.save()
        with open(store.csv_path, encoding="utf-8") as fh:
            lines = fh.readlines()
        assert lines[0].startswith("# frame,track_id,x1")
        # 3 interpolated frames (0, 1, 2)
        assert len(lines) == 4
        frame1 = lines[2].strip().split(",")
        assert frame1[0] == "1"
        assert frame1[2] == "10.00"     # interpolated x1
        assert frame1[-1] == "0"        # not a keyframe

    def test_export_to_detections_and_round_trip(self, store):
        self._add_track(store)
        det_file, count = store.export_to_detections()
        assert count == 3
        by_frame = _load_detections_by_frame(det_file)
        assert sorted(by_frame.keys()) == [0, 1, 2]
        x1, y1, x2, y2, conf, cls = by_frame[1][0]
        assert (x1, y1) == (10.0, 10.0)
        assert conf == 1.0
        assert cls == SPECIES_CLASSES.index("red deer")

    def test_export_preserves_detector_output_and_replaces_block(self, store, tmp_path):
        det_dir = tmp_path / "detections_t"
        det_dir.mkdir()
        det_file = det_dir / "detections.txt"
        det_file.write_text(
            "# frame x1 y1 x2 y2 confidence class_id\n"
            "0 1.00 1.00 2.00 2.00 0.8000 1\n")
        self._add_track(store)
        store.export_to_detections()
        store.export_to_detections()  # re-export must replace, not append
        content = det_file.read_text()
        assert content.count(LabelStore.DETECTIONS_MARKER) == 1
        assert "0.8000 1" in content            # detector line preserved
        assert "# class_id mapping:" in content


class TestPoseEpochs:
    def test_parses_iso_timestamps(self):
        epochs = _pose_epochs([
            {"timestamp": "2023-09-20T10:00:00+02:00"},
            {"timestamp": ""},
            {"timestamp": "not-a-date"},
            {},
        ])
        assert epochs[0] is not None
        assert epochs[1:] == [None, None, None]


class TestFrameMatcher:
    def _images(self, *offsets):
        return [{"timestamp": f"2023-09-20T10:00:{o:02d}+02:00"} for o in offsets]

    def test_matches_nearest_by_time(self):
        src = self._images(0, 10, 20)
        dst = self._images(1, 9, 21)
        matcher = _FrameMatcher(src, dst)
        assert matcher.usable
        assert matcher.match(0) == (0, pytest.approx(1.0))
        assert matcher.match(1) == (1, pytest.approx(1.0))
        assert matcher.match(2) == (2, pytest.approx(1.0))

    def test_out_of_range_and_missing_timestamps(self):
        matcher = _FrameMatcher(self._images(0), self._images(5))
        assert matcher.match(-1) is None
        assert matcher.match(1) is None
        no_ts = _FrameMatcher([{}], [{}])
        assert not no_ts.usable
        assert no_ts.match(0) is None


class TestLoaders:
    def test_load_detections_by_frame(self, tmp_path):
        f = tmp_path / "detections.txt"
        f.write_text(
            "# header\n"
            "0 1 2 3 4 0.9 2\n"
            "0 5 6 7 8 0.8\n"      # legacy line without class id
            "garbage line\n"
            "1 1 1 2 2 0.7 0\n")
        result = _load_detections_by_frame(str(f))
        assert len(result[0]) == 2
        assert result[0][0] == (1.0, 2.0, 3.0, 4.0, 0.9, 2)
        assert result[0][1][5] == 0     # missing class id defaults to 0
        assert len(result[1]) == 1

    def test_load_pixel_tracks_sorted_by_frame(self, tmp_path):
        f = tmp_path / "tracks_pixel.csv"
        f.write_text(
            "# header\n"
            "5,1,0,0,1,1,0.9,2\n"
            "3,1,0,0,1,1,0.9,2,1\n"
            "4,2,0,0,1,1,0.5,0\n")
        result = _load_pixel_tracks(str(f))
        assert [e["frame"] for e in result[1]] == [3, 5]
        assert result[2][0]["conf"] == 0.5

    def test_missing_files_return_empty(self, tmp_path):
        assert _load_detections_by_frame(str(tmp_path / "x.txt")) == {}
        assert _load_pixel_tracks(str(tmp_path / "x.csv")) == {}


# ---------------------------------------------------------------------------
# _GeoPropagator (fake alfspy / trimesh / bambi stack)
# ---------------------------------------------------------------------------

WORLD_QUAD = [(-0.5, -0.25, 0.0), (0.5, -0.25, 0.0),
              (0.5, 0.25, 0.0), (-0.5, 0.25, 0.0)]


@pytest.fixture
def fake_geo_stack(monkeypatch):
    """alfspy/trimesh/bambi fakes: ray-casting always returns WORLD_QUAD."""
    alfspy = install_fake_render_stack(monkeypatch)
    rendering = sys.modules["alfspy.core.rendering"]
    rendering.Resolution = lambda w, h: (w, h)

    render_mod = make_module("alfspy.render")
    render_mod.__path__ = []
    render_render = make_module("alfspy.render.render")

    class _MeshData:
        vertices = [(0, 0, 0)]
        indices = [(0, 0, 0)]

    render_render.read_gltf = lambda path: (_MeshData(), None)
    alfspy.render = render_mod
    render_mod.render = render_render
    monkeypatch.setitem(sys.modules, "alfspy.render", render_mod)
    monkeypatch.setitem(sys.modules, "alfspy.render.render", render_render)

    class Trimesh:
        def __init__(self, vertices=None, faces=None):
            self.vertices = vertices
            self.faces = faces

    monkeypatch.setitem(sys.modules, "trimesh", make_module("trimesh", Trimesh=Trimesh))

    projection_util = make_module("bambi.util.projection_util")
    projection_util.label_to_world_coordinates = (
        lambda label_coords, resolution, mesh, cam: list(WORLD_QUAD))
    util_pkg = make_module("bambi.util")
    util_pkg.__path__ = []
    util_pkg.projection_util = projection_util
    bambi_pkg = make_module("bambi")
    bambi_pkg.__path__ = []
    bambi_pkg.util = util_pkg
    monkeypatch.setitem(sys.modules, "bambi", bambi_pkg)
    monkeypatch.setitem(sys.modules, "bambi.util", util_pkg)
    monkeypatch.setitem(sys.modules, "bambi.util.projection_util", projection_util)


def _images(n=3):
    return [{"location": [0.0, 0.0, 10.0], "rotation": [0.0, 0.0, 0.0],
             "fovy": [50.0]} for _ in range(n)]


class TestPropagationFrames:
    def test_no_sampling_returns_target_only(self):
        assert propagation_frames(0, 25, 0) == [25]

    def test_samples_forward_and_ends_on_target(self):
        assert propagation_frames(0, 25, 10) == [10, 20, 25]

    def test_samples_backward(self):
        assert propagation_frames(25, 0, 10) == [15, 5, 0]

    def test_target_on_the_grid_is_not_duplicated(self):
        assert propagation_frames(0, 20, 10) == [10, 20]

    def test_step_larger_than_offset_yields_target_only(self):
        assert propagation_frames(0, 5, 10) == [5]

    def test_same_frame_yields_nothing(self):
        assert propagation_frames(7, 7, 10) == []


class TestGeoPropagator:
    def _propagator(self, tmp_path, with_dem=True):
        dem = tmp_path / "dem.glb"
        if with_dem:
            dem.write_bytes(b"glb")
        return _GeoPropagator(str(tmp_path), str(dem), "", "t")

    def test_missing_dem_raises(self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path, with_dem=False)
        with pytest.raises(RuntimeError, match="DEM mesh file not found"):
            prop.load()

    def test_frame_range_validation(self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        with pytest.raises(RuntimeError, match="poses file range"):
            prop.propagate((0, 0, 1, 1), 0, 9, _images(3), 640, 512)

    def test_propagate_projects_through_dem(self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        result = prop.propagate((10, 10, 20, 20), 0, 1, _images(3), 640, 512)
        # WORLD_QUAD through the identity camera on a 640x512 frame
        assert result == pytest.approx((160.0, 192.0, 480.0, 320.0))
        assert prop.is_loaded

    def test_propagate_between_modalities_uses_target_resolution(
            self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        result = prop.propagate_between(
            (10, 10, 20, 20), 0, _images(2), 1280, 1024,
            1, _images(4), 640, 512)
        assert result == pytest.approx((160.0, 192.0, 480.0, 320.0))

    def test_propagate_series_samples_intermediate_frames(
            self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        boxes, failures = prop.propagate_series(
            (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=10)
        assert [f for f, _ in boxes] == [10, 20, 25]
        assert failures == []
        for _, box in boxes:
            assert box == pytest.approx((160.0, 192.0, 480.0, 320.0))

    def test_propagate_series_without_sampling_hits_target_only(
            self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        boxes, _ = prop.propagate_series(
            (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=0)
        assert [f for f, _ in boxes] == [25]

    def test_propagate_series_reports_unprojectable_frames(
            self, fake_geo_stack, tmp_path, monkeypatch):
        prop = self._propagator(tmp_path)
        real = prop._world_to_box

        def fail_on_frame_10(world, dst_frame, *args, **kwargs):
            if dst_frame == 10:
                raise RuntimeError("outside frame 10")
            return real(world, dst_frame, *args, **kwargs)

        monkeypatch.setattr(prop, "_world_to_box", fail_on_frame_10)
        boxes, failures = prop.propagate_series(
            (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=10)
        assert [f for f, _ in boxes] == [20, 25]
        assert [f for f, _ in failures] == [10]

    def test_propagate_series_raises_when_source_misses_dem(
            self, fake_geo_stack, tmp_path, monkeypatch):
        import sys as _sys
        monkeypatch.setattr(
            _sys.modules["bambi.util.projection_util"],
            "label_to_world_coordinates",
            lambda *a, **k: [])
        prop = self._propagator(tmp_path)
        with pytest.raises(RuntimeError, match="no mesh intersection"):
            prop.propagate_series(
                (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=10)

    def test_correction_json_is_loaded(self, fake_geo_stack, tmp_path):
        (tmp_path / "correction.json").write_text(json.dumps({
            "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        }))
        prop = self._propagator(tmp_path)
        prop.load()
        assert prop._correction["translation"]["z"] == 3.0
