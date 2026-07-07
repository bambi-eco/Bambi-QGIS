# -*- coding: utf-8 -*-
"""Unit tests for TrackerManager configuration parsing and validation."""
import pytest

pytest.importorskip("requests")  # imported at module level by tracker_manager

from bambi_wildlife_detection.tracker_manager import (  # noqa: E402
    BuiltinTracker,
    TrackerBackend,
    TrackerInfo,
    TrackerManager,
)


@pytest.fixture
def manager(tmp_path):
    return TrackerManager(models_folder=str(tmp_path / "models"))


class TestInit:
    def test_creates_models_folder(self, tmp_path):
        folder = tmp_path / "models"
        TrackerManager(models_folder=str(folder))
        assert folder.is_dir()


class TestParseJsonParams:
    def test_empty_and_whitespace_strings(self, manager):
        assert manager.parse_json_params("") == {}
        assert manager.parse_json_params("   ") == {}
        assert manager.parse_json_params(None) == {}

    def test_valid_object(self, manager):
        assert manager.parse_json_params('{"det_thresh": 0.4}') == {"det_thresh": 0.4}

    def test_invalid_json_logs_warning_and_returns_empty(self, manager):
        messages = []
        assert manager.parse_json_params("{broken", log_fn=messages.append) == {}
        assert any("Invalid JSON" in m for m in messages)

    def test_non_object_json_rejected(self, manager):
        messages = []
        assert manager.parse_json_params("[1, 2, 3]", log_fn=messages.append) == {}
        assert any("must be an object" in m for m in messages)


class TestValidateAndFilterParams:
    PARAM_SPEC = {
        "det_thresh": (float, 0.3, "detection threshold"),
        "per_class": (bool, False, "class-aware matching"),
        "max_age": (int, 30, "max frames without detection"),
    }

    @pytest.fixture
    def spec_manager(self, manager, monkeypatch):
        monkeypatch.setattr(
            manager, "get_tracker_constructor_params", lambda tracker_id: self.PARAM_SPEC)
        return manager

    def test_string_values_converted_to_expected_types(self, spec_manager):
        filtered = spec_manager.validate_and_filter_params(
            "any", {"det_thresh": "0.55", "max_age": "40"})
        assert filtered == {"det_thresh": 0.55, "max_age": 40}
        assert isinstance(filtered["det_thresh"], float)
        assert isinstance(filtered["max_age"], int)

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("True", True), ("1", True), ("yes", True),
        ("false", False), ("no", False), ("0", False),
        (1, True), (0, False),
    ])
    def test_boolean_coercion(self, spec_manager, raw, expected):
        filtered = spec_manager.validate_and_filter_params("any", {"per_class": raw})
        assert filtered == {"per_class": expected}

    def test_unconvertible_value_dropped_with_warning(self, spec_manager):
        messages = []
        filtered = spec_manager.validate_and_filter_params(
            "any", {"max_age": "not-a-number"}, log_fn=messages.append)
        assert filtered == {}
        assert any("Invalid value" in m for m in messages)

    def test_unknown_parameter_dropped_with_warning(self, spec_manager):
        messages = []
        filtered = spec_manager.validate_and_filter_params(
            "any", {"nonexistent": 1}, log_fn=messages.append)
        assert filtered == {}
        assert any("not valid for tracker" in m for m in messages)


class TestAvailabilityChecks:
    def test_is_package_installed(self, manager):
        assert manager._is_package_installed("json")
        assert not manager._is_package_installed("definitely_not_a_real_package_xyz")

    def test_boxmot_unavailable_when_not_installed(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_is_package_installed", lambda name: False)
        assert manager.is_boxmot_available() is False
        assert manager.is_georef_available() is False

    def test_availability_result_is_cached(self, manager, monkeypatch):
        calls = []

        def probe(name):
            calls.append(name)
            return False

        monkeypatch.setattr(manager, "_is_package_installed", probe)
        manager.is_boxmot_available()
        manager.is_boxmot_available()
        assert calls == ["boxmot"]

    def test_refresh_clears_cache(self, manager, monkeypatch):
        monkeypatch.setattr(manager, "_is_package_installed", lambda name: False)
        manager.is_boxmot_available()
        assert manager._boxmot_available is False
        manager.refresh_available_trackers()
        assert manager._boxmot_available is None


class TestBuiltinTracker:
    def test_defaults_roundtrip_through_get_config(self):
        tracker = BuiltinTracker()
        assert tracker.get_config() == {
            "iou_threshold": 0.3,
            "max_age": -1,
            "max_center_distance": 0.2,
            "class_aware": True,
            "interpolate": True,
        }

    def test_custom_params(self):
        tracker = BuiltinTracker(iou_threshold=0.5, max_age=10, class_aware=False)
        config = tracker.get_config()
        assert config["iou_threshold"] == 0.5
        assert config["max_age"] == 10
        assert config["class_aware"] is False


class TestTrackerInfo:
    def test_tracker_class_defaults_to_none(self):
        info = TrackerInfo(
            name="Test", backend=TrackerBackend.BUILTIN,
            description="d", requires_reid=False, supports_geo=True)
        assert info.tracker_class is None
