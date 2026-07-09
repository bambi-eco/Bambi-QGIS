# -*- coding: utf-8 -*-
"""Unit tests for core.config_schema.

The centrepiece is a save→load round-trip through a fake QgsProject store
(QGIS keeps everything as text in XML, so the fake mirrors that), proving
that every storage kind survives serialisation and that the schema and the
widget bindings can never drift apart.
"""
import pytest

from bambi_wildlife_detection.core import config_schema as cs


class FakeProjectStore:
    """Mimics QgsProject entry storage: everything persisted as strings."""

    def __init__(self):
        self.data = {}

    # writers -------------------------------------------------------------
    def write_str(self, key, value):
        self.data[key] = str(value)

    def write_double(self, key, value):
        self.data[key] = repr(float(value))

    def write_bool(self, key, value):
        self.data[key] = "true" if value else "false"

    # readers (return default when the key is absent) --------------------
    def read_str(self, key, default=""):
        return self.data.get(key, default)

    def read_double(self, key, default=0.0):
        raw = self.data.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def read_bool(self, key, default=False):
        raw = self.data.get(key)
        if raw is None:
            return default
        return raw == "true"


class TestCheckSchema:
    def test_schema_is_consistent(self):
        assert cs.check_schema() is None

    def test_no_duplicate_keys(self):
        assert len(cs.CONFIG_KEYS) == len(set(cs.CONFIG_KEYS))

    def test_bindings_cover_every_key_exactly(self):
        assert set(cs.WIDGET_BINDINGS) == set(cs.CONFIG_KEYS)

    def test_binding_roles_are_known(self):
        for attr, role in cs.WIDGET_BINDINGS.values():
            assert role in cs.ROLES
            assert attr

    def test_detects_binding_mismatch(self, monkeypatch):
        broken = dict(cs.WIDGET_BINDINGS)
        broken.pop("Input/TargetCrs")
        monkeypatch.setattr(cs, "WIDGET_BINDINGS", broken)
        assert "mismatch" in cs.check_schema()


def _sample_values():
    """A representative value for every schema key, exercising each kind."""
    values = {}
    for entry in cs.CONFIG_ENTRIES:
        if entry.kind == "str":
            values[entry.key] = f"val::{entry.key}"
        elif entry.kind == "bool01":
            values[entry.key] = True
        elif entry.kind == "int_str":
            values[entry.key] = 7
        elif entry.kind == "int_double":
            values[entry.key] = 42
        elif entry.kind == "double":
            values[entry.key] = 1.25
        elif entry.kind == "bool":
            values[entry.key] = True
    return values


class TestRoundTrip:
    def test_all_kinds_survive_save_then_load(self):
        store = FakeProjectStore()
        values = _sample_values()
        cs.save_config_entries(
            values, store.write_str, store.write_double, store.write_bool)

        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))

        for entry in cs.CONFIG_ENTRIES:
            assert loaded[entry.key] == values[entry.key], entry.key
            # types must match too (an int must not come back as float)
            assert type(loaded[entry.key]) is type(values[entry.key]), entry.key

    def test_bool01_false_survives(self):
        store = FakeProjectStore()
        values = _sample_values()
        for entry in cs.CONFIG_ENTRIES:
            if entry.kind in ("bool01", "bool"):
                values[entry.key] = False
        cs.save_config_entries(
            values, store.write_str, store.write_double, store.write_bool)
        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))
        for entry in cs.CONFIG_ENTRIES:
            if entry.kind in ("bool01", "bool"):
                assert loaded[entry.key] is False, entry.key

    def test_missing_entries_yield_defaults(self):
        store = FakeProjectStore()   # nothing saved
        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))
        for entry in cs.CONFIG_ENTRIES:
            if entry.kind == "bool01" and entry.default is None:
                # "skip when missing" — omitted entirely
                assert entry.key not in loaded, entry.key
            elif entry.kind == "str":
                assert loaded[entry.key] == ""
            else:
                assert loaded[entry.key] == entry.default, entry.key

    def test_save_requires_every_key(self):
        store = FakeProjectStore()
        incomplete = _sample_values()
        del incomplete["Input/TargetFolder"]
        with pytest.raises(KeyError):
            cs.save_config_entries(
                incomplete, store.write_str, store.write_double, store.write_bool)


class TestBool01Semantics:
    def test_video_mode_skipped_when_unset(self):
        # bool01 with default None must not appear when missing (historic
        # "leave the widget untouched" behaviour)
        store = FakeProjectStore()
        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))
        assert "Input/VideoMode" not in loaded

    def test_ne0_truthiness(self):
        # TimezoneAuto uses "ne0": stored "0" is False, anything else True
        store = FakeProjectStore()
        store.data["Config/TimezoneAuto"] = "0"
        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))
        assert loaded["Config/TimezoneAuto"] is False
        store.data["Config/TimezoneAuto"] = "1"
        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))
        assert loaded["Config/TimezoneAuto"] is True

    def test_ne0_default_true_when_missing(self):
        store = FakeProjectStore()
        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))
        # TimezoneAuto default True is applied when absent
        assert loaded["Config/TimezoneAuto"] is True


class TestIntKinds:
    def test_int_str_written_as_plain_string(self):
        store = FakeProjectStore()
        values = _sample_values()
        cs.save_config_entries(
            values, store.write_str, store.write_double, store.write_bool)
        # int_str keys are stored via write_str (no decimal point)
        assert store.data["Extraction/Skip"] == "7"

    def test_int_double_truncates_on_load(self):
        store = FakeProjectStore()
        store.data["Detection/StartFrame"] = "12.9"
        loaded = dict(cs.load_config_entries(
            store.read_str, store.read_double, store.read_bool))
        assert loaded["Detection/StartFrame"] == 12
        assert type(loaded["Detection/StartFrame"]) is int


class TestExpectedContent:
    def test_known_sections_present(self):
        prefixes = {k.split("/")[0] for k in cs.CONFIG_KEYS}
        assert {"Input", "Detection", "Tracking", "FoV", "ALFS",
                "Ortho", "SAM3", "FlightRoute", "Correction",
                "Extraction", "TRex", "Config"} <= prefixes

    def test_target_crs_binds_to_crs_role(self):
        # the CRS default-fallback behaviour lives in the dock's crs role
        assert cs.WIDGET_BINDINGS["Input/TargetCrs"] == ("target_crs_edit", "crs")

    def test_api_key_not_in_schema(self):
        # SAM3 API key is intentionally never persisted
        assert not any("apikey" in k.lower() or "api_key" in k.lower()
                       for k in cs.CONFIG_KEYS)
