# -*- coding: utf-8 -*-
"""Unit tests for TrackerManager backend discovery, ReID handling and
tracker creation.

Neither boxmot nor georef_tracker are installed in the test environment;
fake modules are injected into sys.modules to exercise the discovery code.
"""
import sys
import types

import pytest

pytest.importorskip("requests")  # imported at module level by tracker_manager

import bambi_wildlife_detection.tracker_manager as tm_module  # noqa: E402
from bambi_wildlife_detection.tracker_manager import (  # noqa: E402
    BuiltinTracker,
    ReIDModel,
    TrackerBackend,
    TrackerManager,
    get_tracker_manager,
)
from tests.fakes import FakeResponse, make_module  # noqa: E402


@pytest.fixture
def manager(tmp_path):
    return TrackerManager(models_folder=str(tmp_path / "models"))


class FakeDeepOcSort:
    def __init__(self, reid_weights=None, device="cpu", half: bool = True,
                 det_thresh: float = 0.3, per_class: bool = False):
        self.reid_weights = reid_weights
        self.device = device
        self.half = half
        self.det_thresh = det_thresh
        self.per_class = per_class


class FakeByteTrack:
    def __init__(self, track_thresh: float = 0.45):
        self.track_thresh = track_thresh


@pytest.fixture
def fake_boxmot(monkeypatch):
    """A boxmot module exposing two of the seven expected tracker classes."""
    module = make_module(
        "boxmot", DeepOcSort=FakeDeepOcSort, ByteTrack=FakeByteTrack)
    monkeypatch.setitem(sys.modules, "boxmot", module)
    return module


@pytest.fixture
def fake_georef(monkeypatch):
    class GeoNativeDeepOcSort:
        pass

    class GeoHybridDeepOcSort:
        pass

    module = make_module(
        "georef_tracker",
        GeoNativeDeepOcSort=GeoNativeDeepOcSort,
        GeoHybridDeepOcSort=GeoHybridDeepOcSort,
    )
    monkeypatch.setitem(sys.modules, "georef_tracker", module)
    return module


class TestGetAvailableTrackers:
    def test_builtin_always_available(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "is_boxmot_available", lambda: False)
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        trackers = manager.get_available_trackers()
        assert list(trackers.keys()) == ["builtin"]
        assert trackers["builtin"].backend == TrackerBackend.BUILTIN
        assert trackers["builtin"].supports_geo is True

    def test_result_is_cached_until_refresh(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "is_boxmot_available", lambda: False)
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        first = manager.get_available_trackers()
        assert manager.get_available_trackers() is first
        manager.refresh_available_trackers()
        assert manager.get_available_trackers() is not first

    def test_boxmot_trackers_discovered_from_module(self, manager, fake_boxmot, monkeypatch):
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        trackers = manager.get_available_trackers()
        assert "boxmot_deepocsort" in trackers
        assert "boxmot_bytetrack" in trackers
        assert trackers["boxmot_deepocsort"].tracker_class is FakeDeepOcSort
        assert trackers["boxmot_deepocsort"].requires_reid is True
        assert trackers["boxmot_bytetrack"].requires_reid is False
        # Classes the fake module does not provide are reported as errors
        errors = manager.get_import_errors()
        assert "boxmot:boxmot_strongsort" in errors
        assert "boxmot_strongsort" not in trackers

    def test_georef_trackers_discovered_from_module(self, manager, fake_georef, monkeypatch):
        monkeypatch.setattr(manager, "is_boxmot_available", lambda: False)
        trackers = manager.get_available_trackers()
        assert trackers["georef_native"].backend == TrackerBackend.GEOREF_NATIVE
        assert trackers["georef_hybrid"].backend == TrackerBackend.GEOREF_HYBRID
        assert trackers["georef_native"].supports_geo is True


class TestGetImportErrors:
    def test_empty_when_nothing_failed(self, manager):
        assert manager.get_import_errors() == {}

    def test_collects_backend_import_errors(self, manager):
        manager._boxmot_import_error = "no stderr"
        manager._georef_import_error = "missing dep"
        errors = manager.get_import_errors()
        assert errors["boxmot"] == "no stderr"
        assert errors["georef_tracker"] == "missing dep"


class TestGetTrackerConstructorParams:
    def test_builtin_parameter_spec(self, manager):
        params = manager.get_tracker_constructor_params("builtin")
        assert params["iou_threshold"][:2] == (float, 0.3)
        assert params["max_age"][:2] == (int, -1)
        assert set(params) == {
            "iou_threshold", "max_age", "max_center_distance",
            "class_aware", "interpolate"}

    def test_unknown_tracker_returns_empty(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "is_boxmot_available", lambda: False)
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        assert manager.get_tracker_constructor_params("nope") == {}

    def test_params_inspected_from_tracker_class(self, manager, fake_boxmot, monkeypatch):
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        params = manager.get_tracker_constructor_params("boxmot_deepocsort")
        assert params["det_thresh"][:2] == (float, 0.3)
        assert params["per_class"][:2] == (bool, False)
        assert "reid_weights" in params
        assert "self" not in params

    def test_validate_filters_against_inspected_params(self, manager, fake_boxmot, monkeypatch):
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        filtered = manager.validate_and_filter_params(
            "boxmot_deepocsort",
            {"det_thresh": "0.6", "per_class": "true", "unknown_param": 5},
        )
        assert filtered == {"det_thresh": 0.6, "per_class": True}


class TestGetReidWeightsPath:
    def test_custom_with_existing_file(self, manager, tmp_path):
        weights = tmp_path / "weights.pt"
        weights.write_bytes(b"w")
        result = manager.get_reid_weights_path(ReIDModel.CUSTOM, str(weights))
        assert result == str(weights)

    def test_custom_with_missing_file(self, manager, tmp_path):
        logs = []
        result = manager.get_reid_weights_path(
            ReIDModel.CUSTOM, str(tmp_path / "nope.pt"), log_fn=logs.append)
        assert result is None
        assert any("not found" in m for m in logs)

    def test_osnet_delegates_to_downloader(self, manager, monkeypatch):
        monkeypatch.setattr(
            manager, "_download_osnet_reid", lambda log_fn=None: "/models/osnet.pt")
        assert manager.get_reid_weights_path(ReIDModel.OSNET) == "/models/osnet.pt"

    def test_bambi_delegates_to_downloader(self, manager, monkeypatch):
        monkeypatch.setattr(
            manager, "_download_bambi_reid", lambda log_fn=None: "/models/bambi.pt")
        assert manager.get_reid_weights_path(ReIDModel.BAMBI) == "/models/bambi.pt"


class TestDownloadFile:
    def _patch_requests(self, monkeypatch, get_fn):
        fake_requests = types.SimpleNamespace(get=get_fn)
        monkeypatch.setattr(tm_module, "requests", fake_requests)

    def test_successful_download(self, manager, tmp_path, monkeypatch):
        self._patch_requests(
            monkeypatch, lambda url, **kw: FakeResponse(chunks=[b"x" * 4096]))
        target = tmp_path / "model.pt"
        ok = manager._download_file(
            "https://example.com/model.pt", str(target), "model", min_size=1000)
        assert ok is True
        assert target.stat().st_size == 4096

    def test_too_small_download_removed(self, manager, tmp_path, monkeypatch):
        self._patch_requests(monkeypatch, lambda url, **kw: FakeResponse(chunks=[b"tiny"]))
        target = tmp_path / "model.pt"
        logs = []
        ok = manager._download_file(
            "https://example.com/model.pt", str(target), "model",
            min_size=1000, log_fn=logs.append)
        assert ok is False
        assert not target.exists()
        assert any("Failed to download" in m for m in logs)

    def test_rejects_non_http_url(self, manager, tmp_path, monkeypatch):
        self._patch_requests(monkeypatch, lambda url, **kw: FakeResponse(chunks=[b"x" * 4096]))
        ok = manager._download_file(
            "ftp://example.com/model.pt", str(tmp_path / "m.pt"), "model")
        assert ok is False

    def test_falls_back_to_second_attempt(self, manager, tmp_path, monkeypatch):
        attempts = []

        def flaky_get(url, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise RuntimeError("SSL error")
            return FakeResponse(chunks=[b"y" * 4096])

        self._patch_requests(monkeypatch, flaky_get)
        target = tmp_path / "model.pt"
        logs = []
        ok = manager._download_file(
            "https://example.com/model.pt", str(target), "model",
            min_size=1000, log_fn=logs.append)
        assert ok is True
        assert len(attempts) == 2
        assert any("Trying standard download" in m for m in logs)


class TestReidDownloaders:
    def test_osnet_reuses_existing_file(self, manager):
        import os
        path = os.path.join(manager.models_folder, manager.OSNET_REID_FILENAME)
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 200000)
        assert manager._download_osnet_reid() == path

    def test_osnet_falls_back_to_boxmot_name_on_failure(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_download_file", lambda *a, **kw: False)
        assert manager._download_osnet_reid() == manager.OSNET_REID_FILENAME

    def test_bambi_reuses_existing_file(self, manager):
        import os
        path = os.path.join(manager.models_folder, manager.BAMBI_REID_LOCAL_NAME)
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 200000)
        assert manager._download_bambi_reid() == path

    def test_bambi_returns_none_on_failure(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_download_file", lambda *a, **kw: False)
        assert manager._download_bambi_reid() is None


class TestCreateTracker:
    def test_builtin_tracker_with_params(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "is_boxmot_available", lambda: False)
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        tracker, backend = manager.create_tracker(
            "builtin", json_params='{"iou_threshold": 0.5, "max_age": 10}')
        assert backend == TrackerBackend.BUILTIN
        assert isinstance(tracker, BuiltinTracker)
        assert tracker.iou_threshold == 0.5
        assert tracker.max_age == 10

    def test_unknown_tracker_raises(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "is_boxmot_available", lambda: False)
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        with pytest.raises(ValueError, match="Unknown tracker"):
            manager.create_tracker("boxmot_deepocsort")

    def test_boxmot_tracker_receives_reid_weights_and_device(
            self, manager, fake_boxmot, monkeypatch, tmp_path):
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        weights = tmp_path / "osnet.pt"
        weights.write_bytes(b"w")
        monkeypatch.setattr(
            manager, "get_reid_weights_path",
            lambda model, custom, log_fn=None: str(weights))

        tracker, backend = manager.create_tracker(
            "boxmot_deepocsort", json_params='{"det_thresh": 0.6}')
        assert backend == TrackerBackend.BOXMOT
        assert isinstance(tracker, FakeDeepOcSort)
        assert tracker.reid_weights == str(weights)
        assert tracker.det_thresh == 0.6
        assert tracker.half is False  # forced off for CPU inference

    def test_missing_reid_weights_raises(self, manager, fake_boxmot, monkeypatch):
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)
        monkeypatch.setattr(
            manager, "get_reid_weights_path",
            lambda model, custom, log_fn=None: None)
        with pytest.raises(RuntimeError, match="ReID weights are required"):
            manager.create_tracker("boxmot_deepocsort")

    def test_tracker_without_reid_needs_no_weights(self, manager, fake_boxmot, monkeypatch):
        monkeypatch.setattr(manager, "is_georef_available", lambda: False)

        def fail(*args, **kwargs):
            raise AssertionError("ReID weights must not be requested")

        monkeypatch.setattr(manager, "get_reid_weights_path", fail)
        tracker, backend = manager.create_tracker("boxmot_bytetrack")
        assert isinstance(tracker, FakeByteTrack)
        assert backend == TrackerBackend.BOXMOT


class TestGetTrackerManager:
    def test_returns_manager_instance(self, tmp_path):
        manager = get_tracker_manager(str(tmp_path / "m"))
        assert isinstance(manager, TrackerManager)
        assert manager.models_folder == str(tmp_path / "m")
