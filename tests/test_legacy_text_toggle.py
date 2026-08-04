# -*- coding: utf-8 -*-
"""The "Also write legacy text outputs" toggle (EXCHANGE_FORMAT_PLAN.md §9).

6.0 stores detections, geo-referencing and tracks in GeoPackages and keeps the
5.x text files alongside them for one release. The toggle turns the text files
off — but only for outputs the store fully replaces. Switching it off must
never remove the *only* copy of something, which is why FoV polygons and
segmentation JSON are deliberately not routed through it.
"""
import os

import pytest

from bambi_wildlife_detection.bambi_processing import (
    _DiscardedWriter, _legacy_text_writer)
from bambi_wildlife_detection.core import config_schema


def test_writer_writes_when_enabled(tmp_path):
    from contextlib import ExitStack

    path = str(tmp_path / "detections.txt")
    with ExitStack() as stack:
        writer = _legacy_text_writer(
            {"write_legacy_text_outputs": True}, path, stack)
        writer.write("1 10.0 20.0 30.0 40.0 0.9 0\n")
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as fh:
        assert fh.read().startswith("1 10.0")


def test_writer_discards_when_disabled(tmp_path):
    from contextlib import ExitStack

    path = str(tmp_path / "detections.txt")
    with ExitStack() as stack:
        writer = _legacy_text_writer(
            {"write_legacy_text_outputs": False}, path, stack)
        writer.write("1 10.0 20.0 30.0 40.0 0.9 0\n")
    assert not os.path.exists(path)


def test_writer_defaults_to_writing(tmp_path):
    """Absent key means on: an old saved config must not lose its outputs."""
    from contextlib import ExitStack

    path = str(tmp_path / "detections.txt")
    with ExitStack() as stack:
        _legacy_text_writer({}, path, stack).write("x\n")
    assert os.path.isfile(path)


def test_discarded_writer_accepts_writes():
    writer = _DiscardedWriter()
    assert writer.write("anything") is None


def test_toggle_is_in_the_config_schema():
    keys = {entry.key for entry in config_schema.CONFIG_ENTRIES}
    assert "Input/WriteLegacyTextOutputs" in keys


def test_toggle_defaults_to_true_in_the_schema():
    entry = [e for e in config_schema.CONFIG_ENTRIES
             if e.key == "Input/WriteLegacyTextOutputs"][0]
    assert entry.kind == "bool" and entry.default is True


def test_toggle_is_bound_to_a_widget():
    binding = config_schema.WIDGET_BINDINGS["Input/WriteLegacyTextOutputs"]
    assert binding == ("legacy_text_outputs_check", "check")


@pytest.mark.parametrize("filename", [
    "fov_polygons.txt", "segmentation_pixel.json", "labels.json", "labels.csv",
])
def test_outputs_without_a_store_equivalent_are_not_gated(filename):
    """Guards the rule rather than the wiring.

    These have no complete representation in the store yet, so routing them
    through the toggle would let a user delete their only copy. If one of them
    gains a store writer, gating it becomes a deliberate change here.
    """
    import inspect

    from bambi_wildlife_detection import bambi_processing

    source = inspect.getsource(bambi_processing)
    for line in source.splitlines():
        if "_legacy_text_writer" in line and filename.split(".")[0] in line:
            pytest.fail(f"{filename} must not be routed through the toggle")
