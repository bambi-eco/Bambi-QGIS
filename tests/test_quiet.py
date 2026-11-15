# -*- coding: utf-8 -*-
"""An error that is survived is still said.

Sixty-two places caught any exception and did nothing with it. A correction
file that did not parse meant an uncorrected flight; a raster's CRS that
could not be set, a raster without one; DEM metadata that could not be read,
a flight route at the wrong place - and nothing, anywhere, said so.
"""
import ast
import glob
import json
import logging
import os

import pytest

from bambi_wildlife_detection.core import quiet

PACKAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "bambi_wildlife_detection")


@pytest.fixture(autouse=True)
def fresh():
    quiet.forget()
    yield
    quiet.forget()


@pytest.fixture
def notes(caplog):
    caplog.set_level(logging.DEBUG, logger=quiet.LOGGER_NAME)
    return lambda: [(record.levelname, record.getMessage())
                    for record in caplog.records
                    if record.name == quiet.LOGGER_NAME]


class TestNotes:
    def test_a_problem_is_a_warning_with_its_reason(self, notes):
        try:
            json.loads("{not json")
        except Exception:
            quiet.problem("The correction file c.json could not be read; "
                          "no correction is applied")

        (level, message), = notes()
        assert level == "WARNING"
        assert message.startswith(
            "The correction file c.json could not be read; no correction is "
            "applied (JSONDecodeError: ")
        assert message.endswith(").")

    def test_a_detail_is_kept_for_whoever_looks(self, notes):
        try:
            raise OSError("device busy")
        except Exception:
            quiet.detail("The temporary file x.tmp could not be removed.")

        assert notes() == [("DEBUG", "The temporary file x.tmp could not be "
                                     "removed (OSError: device busy).")]

    def test_an_error_that_says_nothing_still_has_a_name(self, notes):
        try:
            raise KeyError()
        except Exception:
            quiet.problem("No origin in the metadata")
        assert notes() == [("WARNING", "No origin in the metadata (KeyError).")]

    def test_outside_a_handler_it_is_just_the_note(self, notes):
        quiet.problem("Nothing to release")
        assert notes() == [("WARNING", "Nothing to release.")]

    def test_a_credential_in_the_reason_is_masked(self, notes):
        try:
            raise RuntimeError(
                "401 Unauthorized for url: https://example.org/tiles/3/2/1"
                "?api_key=rf_AbC123456789")
        except Exception:
            quiet.detail("Map tile 3/2/1 could not be fetched")

        (_level, message), = notes()
        assert "rf_AbC123456789" not in message and "api_key=***" in message

    def test_the_same_note_is_written_a_few_times_at_most(self, notes):
        """A handler in a loop over a thousand frames."""
        for _frame in range(1000):
            try:
                raise RuntimeError("context lost")
            except Exception:
                quiet.detail("A frame's image could not be released")

        messages = [message for _level, message in notes()]
        assert len(messages) == quiet.REPEATS == 3
        assert messages[-1].endswith("Further occurrences are not listed.")

    def test_another_note_is_another_note(self, notes):
        for name in ("a", "b"):
            try:
                raise OSError("gone")
            except Exception:
                quiet.problem(f"The poses file {name} could not be read")
        assert len(notes()) == 2


# ---------------------------------------------------------------------------
# The plugin's code
# ---------------------------------------------------------------------------

def _handlers():
    """Every ``except`` clause of the plugin: (file, line, what it catches,
    whether it does nothing)."""
    found = []
    for path in sorted(glob.glob(os.path.join(PACKAGE, "**", "*.py"),
                                 recursive=True)):
        if os.sep + "ext" + os.sep in path:
            continue
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                caught = ast.unparse(node.type) if node.type else ""
                silent = len(node.body) == 1 and isinstance(node.body[0],
                                                            ast.Pass)
                found.append((os.path.relpath(path, PACKAGE), node.lineno,
                              caught, silent))
    return found


def test_nothing_catches_everything_and_says_nothing():
    """``except Exception: pass`` - or a bare ``except: pass`` - hides
    whatever goes wrong, including what nobody thought of. A handler that
    names the one failure it expects (a file that is not there, a signal
    that is not connected) may stay silent."""
    silent = [f"{name}:{line}" for name, line, caught, does_nothing
              in _handlers()
              if does_nothing and caught in ("", "Exception", "BaseException")]
    assert silent == []


def test_the_plugin_says_what_it_survives():
    used = 0
    for path in glob.glob(os.path.join(PACKAGE, "**", "*.py"), recursive=True):
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        used += source.count("quiet.problem(") + source.count("quiet.detail(")
    assert used >= 62


def test_a_correction_file_that_does_not_parse_is_said_to_be_ignored(
        tmp_path, notes):
    """It meant "no correction" - an uncorrected flight - without a word."""
    from bambi_wildlife_detection.core import corrections

    broken = tmp_path / "correction.json"
    broken.write_text('{"translation": {"x": 1.0,')

    assert corrections.read_correction(str(tmp_path), str(broken)) == {}

    (level, message), = notes()
    assert level == "WARNING"
    assert str(broken) in message and "no correction is applied" in message
