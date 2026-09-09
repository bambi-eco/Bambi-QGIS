# -*- coding: utf-8 -*-
"""The Segmentation tool's model: requests, the run loop, merging, GeoJSON.

The backends themselves need torch / transformers / Meta's package; here a
fake backend stands in, so what is tested is the contract every backend
must satisfy and everything built on top of it.
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import segmentation as seg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _project(tmp_path, modality="t", frames=4):
    root = str(tmp_path / "out")
    os.makedirs(os.path.join(root, f"frames_{modality}"))
    images = []
    for index in range(frames):
        name = f"frame_{index:06d}.jpg"
        with open(os.path.join(root, f"frames_{modality}", name), "wb") as fh:
            fh.write(b"\xff\xd8\xff\xd9")
        images.append({"imagefile": name, "timestamp": ""})
    with open(os.path.join(root, f"poses_{modality}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"images": images}, fh)
    return root


def _square(x=1.0, y=1.0, size=4.0):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


class FakeBackend(seg.SegmentationBackend):
    """Answers every text prompt with one square, every object with one."""

    key = "fake"
    capabilities = frozenset({seg.CAP_TEXT, seg.CAP_POINTS, seg.CAP_SEQUENCE})

    def __init__(self):
        super().__init__()
        self.loaded = 0
        self.closed = 0
        self.prepared = None
        self.image_calls = []
        self.sequence_calls = []

    def load(self):
        self.loaded += 1

    def prepare(self, texts, points, mode):
        self.prepared = (list(texts), len(points), mode)

    def close(self):
        self.closed += 1

    def segment_image(self, image_path, texts, points, names, confidence):
        self.image_calls.append((os.path.basename(image_path), list(texts),
                                 len(points), confidence))
        entries = [seg.text_entry(t, [{"confidence": 0.9,
                                       "polygons": [_square()]}])
                   for t in texts]
        for object_id in sorted({p.object_id for p in points}):
            entries.append(seg.point_entry(
                object_id, names, [{"confidence": 1.0,
                                    "polygons": [_square(10, 10)]}]))
        return entries

    def segment_sequence(self, image_paths, texts, points, names, confidence,
                         progress_fn=None, cancel_check=None):
        self.sequence_calls.append(([os.path.basename(p) for p in image_paths],
                                    list(texts), list(points)))
        results = {}
        for local_idx in range(len(image_paths)):
            if progress_fn:
                progress_fn(local_idx + 1, len(image_paths))
            if texts:
                results[local_idx] = [seg.text_entry(
                    t, [{"confidence": 0.8, "polygons": [_square()],
                         "object_id": 7}]) for t in texts]
            else:
                results[local_idx] = [
                    seg.point_entry(p.object_id, names,
                                    [{"confidence": 1.0,
                                      "polygons": [_square()]}])
                    for p in points]
        return results


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class TestRequest:
    def test_text_and_points_cannot_mix(self, tmp_path):
        request = seg.SegmentationRequest(
            _project(tmp_path), "t", frames=[0], texts=["deer"],
            points=[seg.PointPrompt(1, 0, 5, 5)])
        with pytest.raises(seg.SegmentationError, match="separately"):
            request.validate()

    def test_roboflow_takes_text_on_images_only(self, tmp_path):
        root = _project(tmp_path)
        with pytest.raises(seg.SegmentationError, match="text prompts only"):
            seg.SegmentationRequest(
                root, "t", backend=seg.BACKEND_ROBOFLOW, frames=[0],
                points=[seg.PointPrompt(1, 0, 5, 5)], api_key="k").validate()
        with pytest.raises(seg.SegmentationError, match="single images only"):
            seg.SegmentationRequest(
                root, "t", backend=seg.BACKEND_ROBOFLOW, mode=seg.MODE_SEQUENCE,
                frames=[0, 1], texts=["deer"], api_key="k").validate()
        with pytest.raises(seg.SegmentationError, match="API key"):
            seg.SegmentationRequest(
                root, "t", backend=seg.BACKEND_ROBOFLOW, frames=[0],
                texts=["deer"]).validate()

    def test_points_must_lie_on_selected_frames(self, tmp_path):
        request = seg.SegmentationRequest(
            _project(tmp_path), "t", frames=[0, 1],
            points=[seg.PointPrompt(1, 3, 5, 5)])
        with pytest.raises(seg.SegmentationError, match=r"\[3\]"):
            request.validate()

    def test_prompt_names_cover_texts_and_objects(self):
        request = seg.SegmentationRequest(
            "x", "t", texts=["deer", ""], object_names={2: "big one"},
            points=[seg.PointPrompt(2, 0, 1, 1), seg.PointPrompt(1, 0, 2, 2)])
        assert request.prompt_names() == ["deer", "object 1", "big one"]

    def test_backend_catalogue(self):
        assert seg.CAP_POINTS not in seg.backend_capabilities(seg.BACKEND_ROBOFLOW)
        assert seg.CAP_SEQUENCE in seg.backend_capabilities(seg.BACKEND_META)
        with pytest.raises(seg.SegmentationError):
            seg.backend_capabilities("nope")
        assert "SAM 3.1" in seg.backend_label(seg.BACKEND_META)


class TestFrames:
    def test_select_frames_follows_the_pipeline_rule(self):
        assert seg.select_frames(5) == [0, 1, 2, 3, 4]
        assert seg.select_frames(5, False, 1, 3) == [1, 2, 3]
        assert seg.select_frames(5, False, 2, 99) == [2, 3, 4]
        assert seg.select_frames(10, True, step=4) == [0, 4, 8]
        assert seg.select_frames(0) == []
        with pytest.raises(seg.SegmentationError):
            seg.select_frames(5, step=0)

    def test_missing_poses_are_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="poses_w.json"):
            seg.pose_images(str(tmp_path), "w")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class TestRun:
    def test_image_mode_asks_per_frame_and_keeps_frame_order(self, tmp_path):
        root = _project(tmp_path)
        backend = FakeBackend()
        progress = []
        results = seg.run_segmentation(
            seg.SegmentationRequest(root, "t", frames=[2, 0], texts=["deer"]),
            progress_fn=progress.append, backend=backend)
        assert [c[0] for c in backend.image_calls] == [
            "frame_000002.jpg", "frame_000000.jpg"]
        assert [r["frame_idx"] for r in results] == [2, 0]
        assert results[0]["prompts"][0]["prompt_type"] == seg.PROMPT_TEXT
        assert progress[0] == 5 and progress[-1] == 100
        assert backend.prepared == (["deer"], 0, seg.MODE_IMAGE)
        # An injected backend is the caller's to close.
        assert backend.closed == 0

    def test_point_prompts_run_only_their_frames(self, tmp_path):
        root = _project(tmp_path)
        backend = FakeBackend()
        results = seg.run_segmentation(
            seg.SegmentationRequest(
                root, "t", frames=[0, 1, 2, 3],
                points=[seg.PointPrompt(1, 2, 5, 5), seg.PointPrompt(1, 2, 8, 8, False),
                        seg.PointPrompt(2, 2, 30, 30)],
                object_names={2: "calf"}),
            backend=backend)
        assert [c[0] for c in backend.image_calls] == ["frame_000002.jpg"]
        assert [p["prompt"] for p in results[0]["prompts"]] == ["object 1", "calf"]
        assert results[0]["prompts"][1]["object_id"] == 2
        assert results[0]["prompts"][1]["prompt_type"] == seg.PROMPT_POINT

    def test_sequence_mode_hands_the_clip_over_with_local_indices(self, tmp_path):
        root = _project(tmp_path)
        backend = FakeBackend()
        results = seg.run_segmentation(
            seg.SegmentationRequest(
                root, "t", mode=seg.MODE_SEQUENCE, frames=[1, 3],
                points=[seg.PointPrompt(1, 3, 5, 5)]),
            backend=backend)
        paths, texts, points = backend.sequence_calls[0]
        assert paths == ["frame_000001.jpg", "frame_000003.jpg"]
        assert points[0].frame_idx == 1  # local index of frame 3
        assert [r["frame_idx"] for r in results] == [1, 3]
        assert backend.prepared == ([], 1, seg.MODE_SEQUENCE)

    def test_a_single_frame_sequence_is_an_image(self, tmp_path):
        backend = FakeBackend()
        seg.run_segmentation(
            seg.SegmentationRequest(_project(tmp_path), "t",
                                    mode=seg.MODE_SEQUENCE, frames=[2],
                                    texts=["deer"]),
            backend=backend)
        assert backend.sequence_calls == []
        assert len(backend.image_calls) == 1

    def test_missing_frames_are_skipped_and_none_left_is_an_error(self, tmp_path):
        root = _project(tmp_path, frames=2)
        os.remove(os.path.join(root, "frames_t", "frame_000001.jpg"))
        logs = []
        backend = FakeBackend()
        results = seg.run_segmentation(
            seg.SegmentationRequest(root, "t", frames=[0, 1, 9], texts=["deer"]),
            log_fn=logs.append, backend=backend)
        assert [r["frame_idx"] for r in results] == [0]
        assert any("Frame not found" in line for line in logs)
        with pytest.raises(seg.SegmentationError, match="extract"):
            seg.run_segmentation(
                seg.SegmentationRequest(root, "t", frames=[1], texts=["deer"]),
                backend=backend)

    def test_one_bad_frame_is_logged_not_fatal(self, tmp_path):
        class Flaky(FakeBackend):
            def segment_image(self, image_path, *args):
                if image_path.endswith("000001.jpg"):
                    raise RuntimeError("boom")
                return super().segment_image(image_path, *args)

        logs = []
        results = seg.run_segmentation(
            seg.SegmentationRequest(_project(tmp_path), "t", frames=[0, 1, 2],
                                    texts=["deer"]),
            log_fn=logs.append, backend=Flaky())
        assert [r["frame_idx"] for r in results] == [0, 2]
        assert any("boom" in line for line in logs)

    def test_cancellation_stops_the_loop(self, tmp_path):
        calls = []

        class Counting(FakeBackend):
            def segment_image(self, *args):
                calls.append(1)
                return super().segment_image(*args)

        with pytest.raises(seg.CancelledError):
            seg.run_segmentation(
                seg.SegmentationRequest(_project(tmp_path), "t",
                                        frames=[0, 1, 2], texts=["deer"]),
                cancel_check=lambda: len(calls) >= 1, backend=Counting())
        assert len(calls) == 1

    def test_a_built_backend_is_closed_even_on_failure(self, tmp_path, monkeypatch):
        backend = FakeBackend()

        def failing_load():
            raise RuntimeError("no model")
        backend.load = failing_load
        monkeypatch.setattr(seg, "make_backend", lambda request, log_fn=None: backend)
        with pytest.raises(RuntimeError, match="no model"):
            seg.run_segmentation(seg.SegmentationRequest(
                _project(tmp_path), "t", frames=[0], texts=["deer"]))
        assert backend.closed == 1


# ---------------------------------------------------------------------------
# Store and merge
# ---------------------------------------------------------------------------

class TestStore:
    def test_round_trip_and_staleness(self, tmp_path):
        store = seg.SegmentationStore(str(tmp_path), "w")
        assert store.load_pixel() == [] and not store.has_pixel()
        path = store.save_pixel([seg.frame_result(3, "c.jpg", []),
                                 seg.frame_result(1, "a.jpg", [])])
        assert path.endswith(os.path.join("segmentation_w", "segmentation_pixel.json"))
        assert [r["frame_idx"] for r in store.load_pixel()] == [1, 3]
        assert not store.georef_is_stale()
        with open(store.georef_path, "w", encoding="utf-8") as fh:
            fh.write("[]")
        os.utime(store.georef_path, (1, 1))
        assert store.has_georef() and store.georef_is_stale()

    def test_merge_replaces_only_the_rerun_prompts(self):
        existing = [
            seg.frame_result(0, "a", [seg.text_entry("deer", [{"c": 1}]),
                                      seg.text_entry("boar", [{"c": 2}])]),
            seg.frame_result(2, "c", [seg.text_entry("deer", [{"c": 3}])]),
        ]
        new = [seg.frame_result(0, "a", [seg.text_entry("deer", [{"c": 9}])]),
               seg.frame_result(1, "b", [seg.text_entry("deer", [])])]
        merged = seg.merge_results(existing, new, ["deer"])
        assert [r["frame_idx"] for r in merged] == [0, 1, 2]
        frame0 = {p["prompt"]: p["predictions"] for p in merged[0]["prompts"]}
        assert frame0 == {"boar": [{"c": 2}], "deer": [{"c": 9}]}
        assert merged[1]["prompts"][0]["predictions"] == []
        # Frame 2 was not processed, so its deer stays.
        assert merged[2]["prompts"][0]["predictions"] == [{"c": 3}]

    def test_summary_counts_masks_and_frames(self):
        results = [
            seg.frame_result(0, "a", [seg.text_entry("deer", [{}, {}]),
                                      seg.text_entry("boar", [])]),
            seg.frame_result(1, "b", [seg.text_entry("deer", [{}])]),
        ]
        assert seg.prompt_summary(results) == {"deer": (2, 3)}
        assert seg.count_predictions(results) == 3


# ---------------------------------------------------------------------------
# Masks and normalisation helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_prediction_from_mask_scales_back_to_the_original_frame(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        prediction = seg.prediction_from_mask(mask, 0.7, scale=2.0,
                                              object_id=4, box=[2, 2, 7, 7])
        xs = [x for x, _y in prediction["polygons"][0]]
        assert min(xs) == 4.0 and max(xs) == 14.0
        assert prediction["object_id"] == 4 and prediction["box"] == [4, 4, 14, 14]
        assert seg.prediction_from_mask(np.zeros((5, 5)), 0.5) is None

    def test_roboflow_response_is_normalised(self):
        response = {"prompt_results": [
            {"echo": {"text": "deer"},
             "predictions": [{"confidence": 0.8, "masks": [_square()]}]},
            {"prompt": "boar", "predictions": []},
        ]}
        entries = seg.RoboflowBackend.normalise(response)
        assert [e["prompt"] for e in entries] == ["deer", "boar"]
        assert entries[0]["predictions"][0]["polygons"] == [_square()]
        assert entries[0]["prompt_type"] == seg.PROMPT_TEXT

    def test_video_entries_group_objects_by_prompt(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        masks = np.zeros((3, 10, 10), dtype=np.uint8)
        masks[0, 1:4, 1:4] = 1
        masks[1, 5:9, 5:9] = 1
        masks[2, 0:2, 0:2] = 1
        processed = {
            "object_ids": np.array([11, 12, 13]),
            "scores": np.array([0.9, 0.3, 0.95]),
            "masks": masks,
            "boxes": np.array([[1, 1, 3, 3], [5, 5, 8, 8], [0, 0, 1, 1]]),
            "prompt_to_obj_ids": {"deer": [11, 12], "boar": [13]},
        }
        entries = seg.TransformersBackend._video_entries(
            processed, ["deer", "boar"], 1.0, 0.5)
        by_prompt = {e["prompt"]: e["predictions"] for e in entries}
        assert [p["object_id"] for p in by_prompt["deer"]] == [11]  # 12 < 0.5
        assert [p["object_id"] for p in by_prompt["boar"]] == [13]

    def test_meta_outputs_become_entries(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        masks = np.zeros((2, 10, 10), dtype=bool)
        masks[0, 1:5, 1:5] = True
        masks[1, 6:9, 6:9] = True
        outputs = {"out_obj_ids": np.array([1, 2]),
                   "out_probs": np.array([0.6, 0.7]),
                   "out_binary_masks": masks}
        predictions = seg.MetaSam3Backend._predictions(outputs, 1.0, with_ids=True)
        assert [p["object_id"] for p in predictions] == [1, 2]
        entries = seg.MetaSam3Backend._point_entries(outputs, 2.0, {2: "calf"})
        assert [e["prompt"] for e in entries] == ["object 1", "calf"]
        xs = [x for x, _y in entries[1]["predictions"][0]["polygons"][0]]
        assert min(xs) == 12.0


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def _georef(frame=0, rings=1, object_id=None, prompt="deer",
            prompt_type=seg.PROMPT_TEXT):
    prediction = {"confidence": 0.87,
                  "world_polygons": [[[500000 + i, 5200000, 300],
                                      [500010 + i, 5200000, 300],
                                      [500010 + i, 5200010, 300]]
                                     for i in range(rings)]}
    if object_id is not None:
        prediction["object_id"] = object_id
    entry = {"prompt": prompt, "prompt_type": prompt_type,
             "predictions": [prediction]}
    return {"frame_idx": frame, "imagefile": f"f{frame}.jpg", "prompts": [entry]}


class TestGeoJson:
    def test_features_carry_prompt_object_and_frame(self, tmp_path):
        pytest.importorskip("pyproj")
        out = str(tmp_path / "seg.geojson")
        counts = seg.export_geojson(
            [_georef(0, object_id=3), _georef(1, rings=2, prompt="object 1",
                                              prompt_type=seg.PROMPT_POINT)],
            out, 32633, "w", log_fn=lambda _m: None)
        assert counts == {"features": 2, "skipped": 0}
        with open(out, encoding="utf-8") as fh:
            document = json.load(fh)
        first, second = document["features"]
        assert first["geometry"]["type"] == "Polygon"
        assert second["geometry"]["type"] == "MultiPolygon"
        assert first["properties"]["object_id"] == 3
        assert first["properties"]["camera"] == "rgb"
        assert second["properties"]["prompt_type"] == seg.PROMPT_POINT
        ring = first["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]
        lon, lat = ring[0]
        assert 14 < lon < 16 and 46 < lat < 48  # UTM 33N sample point

    def test_frames_filter_and_empty_masks(self, tmp_path):
        pytest.importorskip("pyproj")
        out = str(tmp_path / "seg.geojson")
        empty = _georef(2)
        empty["prompts"][0]["predictions"][0]["world_polygons"] = [[[1, 2, 3]]]
        counts = seg.export_geojson([_georef(0), _georef(1), empty], out,
                                    32633, "t", frames=[1, 2])
        assert counts == {"features": 1, "skipped": 1}

    def test_no_crs_means_no_file(self, tmp_path):
        from bambi_wildlife_detection.core.exporters.common import ExportError
        out = str(tmp_path / "seg.geojson")
        with pytest.raises(ExportError):
            seg.export_geojson([_georef(0)], out, None, "t")
        assert not os.path.exists(out)
