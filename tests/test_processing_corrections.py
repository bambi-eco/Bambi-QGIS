# -*- coding: utf-8 -*-
"""Unit tests for per-frame correction selection (BambiProcessor)."""
from bambi_wildlife_detection.bambi_processing import BambiProcessor


def _corr(x):
    return {"x": x, "y": 0, "z": 0}


class TestGetCorrectionForFrame:
    def test_empty_config_returns_zero_correction(self):
        result = BambiProcessor.get_correction_for_frame(0, {})
        assert result == {
            "translation": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
        }

    def test_default_correction_when_no_ranges(self):
        config = {"translation": _corr(1.5), "rotation": _corr(2.5)}
        result = BambiProcessor.get_correction_for_frame(100, config)
        assert result["translation"] == _corr(1.5)
        assert result["rotation"] == _corr(2.5)

    def test_frame_inside_additional_range(self):
        config = {
            "translation": _corr(1.0),
            "rotation": _corr(0.0),
            "additional_corrections": [
                {"start": 10, "end": 20, "translation": _corr(9.0), "rotation": _corr(3.0)},
            ],
        }
        result = BambiProcessor.get_correction_for_frame(15, config)
        assert result["translation"] == _corr(9.0)
        assert result["rotation"] == _corr(3.0)

    def test_range_bounds_are_inclusive(self):
        config = {
            "translation": _corr(1.0),
            "additional_corrections": [{"start": 10, "end": 20, "translation": _corr(9.0)}],
        }
        assert BambiProcessor.get_correction_for_frame(10, config)["translation"] == _corr(9.0)
        assert BambiProcessor.get_correction_for_frame(20, config)["translation"] == _corr(9.0)
        assert BambiProcessor.get_correction_for_frame(9, config)["translation"] == _corr(1.0)
        assert BambiProcessor.get_correction_for_frame(21, config)["translation"] == _corr(1.0)

    def test_open_ended_range(self):
        # An entry without "end" applies to all frames from "start" onwards.
        config = {"additional_corrections": [{"start": 100, "translation": _corr(7.0)}]}
        assert BambiProcessor.get_correction_for_frame(10 ** 9, config)["translation"] == _corr(7.0)
        before = BambiProcessor.get_correction_for_frame(99, config)["translation"]
        assert before == {"x": 0, "y": 0, "z": 0}

    def test_first_matching_range_wins(self):
        config = {
            "additional_corrections": [
                {"start": 0, "end": 50, "translation": _corr(1.0)},
                {"start": 40, "end": 60, "translation": _corr(2.0)},
            ],
        }
        assert BambiProcessor.get_correction_for_frame(45, config)["translation"] == _corr(1.0)

    def test_range_entry_missing_rotation_defaults_to_zero(self):
        config = {
            "rotation": _corr(5.0),
            "additional_corrections": [{"start": 0, "end": 10, "translation": _corr(1.0)}],
        }
        # Inside a matching range the entry's own (missing) rotation applies,
        # not the global default.
        result = BambiProcessor.get_correction_for_frame(5, config)
        assert result["rotation"] == {"x": 0, "y": 0, "z": 0}
