# -*- coding: utf-8 -*-
"""Unit tests for ProcessingWorker dispatch and the frame-extraction helpers.

The ``bambi`` framework modules used by ``count_srt_frames`` and
``patch_frame_extraction_progress`` are replaced with minimal fakes so the
progress/cancellation plumbing can be tested without the real dependencies.
"""
import sys
import types

import pytest

from bambi_wildlife_detection.bambi_processing import (
    CancelledException,
    ProcessingWorker,
    count_srt_frames,
    patch_frame_extraction_progress,
)


class SignalRecorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)

    def connect(self, *args, **kwargs):
        pass


# Mapping of worker step names to the BambiProcessor method each dispatches to.
STEP_TO_METHOD = {
    "extract_thermal_frames": "extract_thermal_frames",
    "extract_rgb_frames": "extract_rgb_frames",
    "flight_route": "run_flight_route",
    "detection": "run_detection",
    "georeference": "run_georeference",
    "calculate_fov": "run_calculate_fov",
    "tracking": "run_tracking",
    "alfs": "run_alfs",
    "export_geotiffs": "run_export_geotiffs",
    "orthomosaic": "run_orthomosaic",
    "sam3_segmentation": "run_sam3_segmentation",
    "sam3_georeference": "run_sam3_georeference",
    "perpendicular": "run_perpendicular",
    "track_perpendicular": "run_track_perpendicular",
    "trex_import": "run_trex_import",
    "density_heatmap": "run_density_heatmap",
    "coverage_map": "run_coverage_map",
    "distance_sampling": "run_distance_sampling",
}


class FakeProcessor:
    """Records which pipeline method the worker dispatched to."""

    def __init__(self, raise_exc=None):
        self.calls = []
        self._raise_exc = raise_exc

    def __getattr__(self, name):
        def method(config, progress_fn, log_fn, cancel_check):
            self.calls.append((name, config))
            if self._raise_exc is not None:
                raise self._raise_exc
        return method


def _make_worker(step, processor=None, config=None):
    worker = ProcessingWorker(processor or FakeProcessor(), step, config or {"k": "v"})
    worker.finished = SignalRecorder()
    worker.error = SignalRecorder()
    worker.progress = SignalRecorder()
    worker.log = SignalRecorder()
    return worker


class TestProcessingWorkerDispatch:
    @pytest.mark.parametrize("step,method", sorted(STEP_TO_METHOD.items()))
    def test_each_step_dispatches_to_its_processor_method(self, step, method):
        processor = FakeProcessor()
        worker = _make_worker(step, processor)
        worker.run()
        assert processor.calls == [(method, {"k": "v"})]
        assert worker.finished.calls == [(step, True)]
        assert worker.error.calls == []

    def test_unknown_step_reports_error(self):
        worker = _make_worker("bogus_step")
        worker.run()
        assert worker.finished.calls == [("bogus_step", False)]
        assert len(worker.error.calls) == 1
        assert "Unknown step" in worker.error.calls[0][1]

    def test_cancelled_exception_finishes_without_error(self):
        processor = FakeProcessor(raise_exc=CancelledException("stop"))
        worker = _make_worker("detection", processor)
        worker.run()
        assert worker.finished.calls == [("detection", False)]
        assert worker.error.calls == []
        assert any("cancelled" in args[0] for args in worker.log.calls)

    def test_generic_exception_reports_error_with_traceback(self):
        processor = FakeProcessor(raise_exc=ValueError("boom"))
        worker = _make_worker("detection", processor)
        worker.run()
        assert worker.finished.calls == [("detection", False)]
        step, message = worker.error.calls[0]
        assert step == "detection"
        assert "boom" in message
        assert "Traceback" in message

    def test_cancellation_flag_marks_step_unsuccessful(self):
        worker = _make_worker("detection")
        worker.cancel()
        assert worker.is_cancelled() is True
        worker.run()
        assert worker.finished.calls == [("detection", False)]


# ---------------------------------------------------------------------------
# count_srt_frames / patch_frame_extraction_progress (fake bambi modules)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_srt_parser(monkeypatch):
    """SrtParser whose parse() returns one entry per character of the path stem."""
    bambi = types.ModuleType("bambi")
    bambi.__path__ = []
    srt = types.ModuleType("bambi.srt")
    srt.__path__ = []
    srt_parser = types.ModuleType("bambi.srt.srt_parser")

    class SrtParser:
        def parse(self, path):
            return [None] * int(path.rsplit("=", 1)[1])

    srt_parser.SrtParser = SrtParser
    bambi.srt = srt
    srt.srt_parser = srt_parser
    monkeypatch.setitem(sys.modules, "bambi", bambi)
    monkeypatch.setitem(sys.modules, "bambi.srt", srt)
    monkeypatch.setitem(sys.modules, "bambi.srt.srt_parser", srt_parser)


@pytest.fixture
def fake_extractor_callback(monkeypatch):
    """Install a fake TimedFrameExtractorCallback and return the class."""
    class TimedFrameExtractorCallback:
        def __call__(self, idx, img):
            return True

    bambi = types.ModuleType("bambi")
    bambi.__path__ = []
    webgl = types.ModuleType("bambi.webgl")
    webgl.__path__ = []
    tpe = types.ModuleType("bambi.webgl.timed_pose_extractor")
    tpe.TimedFrameExtractorCallback = TimedFrameExtractorCallback
    bambi.webgl = webgl
    webgl.timed_pose_extractor = tpe
    monkeypatch.setitem(sys.modules, "bambi", bambi)
    monkeypatch.setitem(sys.modules, "bambi.webgl", webgl)
    monkeypatch.setitem(sys.modules, "bambi.webgl.timed_pose_extractor", tpe)
    return TimedFrameExtractorCallback


class TestCountSrtFrames:
    def test_sums_frames_over_all_files(self, fake_srt_parser):
        assert count_srt_frames(["video=10", "video=25"]) == 35

    def test_empty_list(self, fake_srt_parser):
        assert count_srt_frames([]) == 0


class TestPatchFrameExtractionProgress:
    def test_reports_percent_per_extracted_frame(self, fake_extractor_callback):
        percents = []
        logs = []
        with patch_frame_extraction_progress(percents.append, total_frames=4,
                                             log_fn=logs.append,
                                             log_interval_percent=50):
            callback = fake_extractor_callback()
            for i in range(4):
                assert callback(i, "img") is True

        assert percents == [25, 50, 75, 100]
        # Logged at >= 50% steps only
        assert len(logs) == 2
        assert "2 / 4" in logs[0]

    def test_original_callback_restored_after_exit(self, fake_extractor_callback):
        original = fake_extractor_callback.__call__
        with patch_frame_extraction_progress(None, total_frames=1):
            assert fake_extractor_callback.__call__ is not original
        assert fake_extractor_callback.__call__ is original

    def test_restored_even_when_body_raises(self, fake_extractor_callback):
        original = fake_extractor_callback.__call__
        with pytest.raises(RuntimeError):
            with patch_frame_extraction_progress(None, total_frames=1):
                raise RuntimeError("extraction failed")
        assert fake_extractor_callback.__call__ is original

    def test_cancel_check_raises_cancelled_exception(self, fake_extractor_callback):
        with patch_frame_extraction_progress(None, total_frames=2,
                                             cancel_check=lambda: True):
            callback = fake_extractor_callback()
            with pytest.raises(CancelledException):
                callback(0, "img")

    def test_zero_total_frames_reports_no_progress(self, fake_extractor_callback):
        percents = []
        with patch_frame_extraction_progress(percents.append, total_frames=0):
            callback = fake_extractor_callback()
            callback(0, "img")
        assert percents == []
