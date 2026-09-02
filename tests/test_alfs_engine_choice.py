# -*- coding: utf-8 -*-
"""Choosing the alfspy render engine and ray caster.

alfspy 3.0 merged the separate PyTorch fork back in: one release, three
engines and two ray casters, each an optional pip extra and each selected at
run time by an environment variable. That makes one rule load-bearing - the
name in the dropdown, the name of the extra that gets installed, and the value
exported to ``$ALFS_ENGINE`` / ``$ALFS_RAYCASTER`` must all be the same string.
These tests pin that, and pin that the superseded ``AlfsTorch`` distribution is
still removed on install: it owns the same ``alfspy`` import path and would
shadow the selection entirely.
"""
import pathlib

import pytest

from bambi_wildlife_detection.core import alfs_runtime
from bambi_wildlife_detection.core.dependency_ops import (
    ALFS_DIST, ALFS_LEGACY_DIST, ALFS_PY_TAG, ALFS_REPO, _VERSION_RANGES,
    alfs_install_spec)

PLUGIN_DIR = (pathlib.Path(__file__).resolve().parent.parent
              / "bambi_wildlife_detection")


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

def test_the_three_engines_and_two_raycasters_are_offered():
    assert alfs_runtime.engine_names() == ("moderngl", "torch", "vulkan")
    assert alfs_runtime.raycaster_names() == ("embree", "warp")


def test_the_defaults_match_alfspys_own():
    """An unconfigured project and an unset environment must render alike."""
    assert alfs_runtime.DEFAULT_ENGINE == "moderngl"
    assert alfs_runtime.DEFAULT_RAYCASTER == "embree"


def test_every_option_is_labelled_and_explained():
    for option in alfs_runtime.ENGINES + alfs_runtime.RAYCASTERS:
        assert option.label and option.hint
        assert option.name == option.name.lower()


def test_an_unknown_selection_falls_back_rather_than_raising():
    """A project written by a newer plugin still has to open."""
    assert alfs_runtime.resolve_engine("metal") == "moderngl"
    assert alfs_runtime.resolve_engine("") == "moderngl"
    assert alfs_runtime.resolve_engine(None) == "moderngl"
    assert alfs_runtime.resolve_raycaster("optix") == "embree"


def test_selection_is_case_and_space_insensitive():
    assert alfs_runtime.resolve_engine(" Torch ") == "torch"
    assert alfs_runtime.resolve_raycaster("WARP") == "warp"


# ---------------------------------------------------------------------------
# Exporting the choice
# ---------------------------------------------------------------------------

def test_apply_exports_both_variables():
    env = {}
    engine, raycaster = alfs_runtime.apply_selection(
        {"alfs_engine": "torch", "alfs_raycaster": "warp"}, env=env)
    assert (engine, raycaster) == ("torch", "warp")
    assert env["ALFS_ENGINE"] == "torch"
    assert env["ALFS_RAYCASTER"] == "warp"


def test_apply_exports_the_defaults_for_an_unconfigured_project():
    env = {}
    alfs_runtime.apply_selection({}, env=env)
    assert env["ALFS_ENGINE"] == "moderngl"
    assert env["ALFS_RAYCASTER"] == "embree"


def test_apply_overwrites_a_previous_run_rather_than_inheriting_it():
    """A step run on its own must not render with the last step's engine."""
    env = {"ALFS_ENGINE": "vulkan", "ALFS_RAYCASTER": "warp"}
    alfs_runtime.apply_selection({"alfs_engine": "torch"}, env=env)
    assert env["ALFS_ENGINE"] == "torch"
    assert env["ALFS_RAYCASTER"] == "embree"


def test_the_device_is_exported_only_when_asked_for():
    env = {"ALFS_DEVICE": "cuda"}
    alfs_runtime.apply_selection({}, device="", env=env)
    assert "ALFS_DEVICE" not in env

    alfs_runtime.apply_selection({}, device="cuda:1", env=env)
    assert env["ALFS_DEVICE"] == "cuda:1"

    # "auto" is the UI's word for "do not decide", not a device name.
    alfs_runtime.apply_selection({}, device="auto", env=env)
    assert "ALFS_DEVICE" not in env


def test_apply_reports_the_pair_in_words():
    logged = []
    alfs_runtime.apply_selection({"alfs_engine": "torch"},
                                 env={}, log_fn=logged.append)
    assert logged and "PyTorch" in logged[0] and "Embree" in logged[0]


def test_the_variable_names_are_the_ones_alfspy_reads():
    assert alfs_runtime.ENGINE_ENV_VAR == "ALFS_ENGINE"
    assert alfs_runtime.RAYCASTER_ENV_VAR == "ALFS_RAYCASTER"
    assert alfs_runtime.DEVICE_ENV_VAR == "ALFS_DEVICE"


# ---------------------------------------------------------------------------
# The install spec
# ---------------------------------------------------------------------------

def test_the_extras_are_named_after_the_selection():
    """alfspy 3.0 renamed its extras to match the selector values exactly, so
    a choice cannot be spelled one way in the UI and another in pip."""
    for engine in alfs_runtime.engine_names():
        for raycaster in alfs_runtime.raycaster_names():
            spec = alfs_install_spec(engine, raycaster)
            assert spec["extras"] == (engine, raycaster)
            assert spec["extras_suffix"] == f"[{engine},{raycaster}]"


def test_the_default_spec_is_moderngl_and_embree():
    spec = alfs_install_spec(None, None)
    assert spec["engine"] == "moderngl"
    assert spec["raycaster"] == "embree"


def test_the_git_url_carries_the_extras():
    """``pip install "Dist[extra] @ git+..."`` is the only spelling that gets
    extras through a VCS install; ``#egg=`` is deprecated."""
    spec = alfs_install_spec("torch", "warp")
    assert spec["git_url"] == (
        "AlfsPy[torch,warp] @ git+https://github.com/bambi-eco/"
        f"alfs_py.git@{ALFS_PY_TAG}")
    assert "#egg" not in spec["git_url"]


def test_the_zip_url_points_at_the_pinned_tag():
    spec = alfs_install_spec("moderngl", "embree")
    assert spec["zip_url"] == (
        "https://github.com/bambi-eco/"
        f"{ALFS_REPO}/archive/refs/tags/{ALFS_PY_TAG}.zip")
    assert "refs/heads/main" not in spec["zip_url"]


def test_the_spec_names_the_superseded_distribution():
    """AlfsTorch installs a package called ``alfspy`` too, so it has to go."""
    spec = alfs_install_spec("torch", "embree")
    assert spec["dist"] == ALFS_DIST == "AlfsPy"
    assert spec["legacy_dist"] == ALFS_LEGACY_DIST == "AlfsTorch"


def test_the_label_reads_as_a_pair():
    assert alfs_install_spec("torch", "warp")["label"] == \
        "PyTorch + Warp (GPU)"


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------

def test_the_pinned_tag_is_the_accepted_floor():
    min_ver, max_ver = _VERSION_RANGES[ALFS_DIST]
    assert min_ver == ALFS_PY_TAG.lstrip("v")
    assert max_ver is None


def test_the_floor_covers_the_merged_release():
    """3.0.0 is where the fork came back and ``make_context`` replaced the
    per-backend factories; 2.x has neither."""
    packaging = pytest.importorskip("packaging.version")
    min_ver, _max = _VERSION_RANGES[ALFS_DIST]
    assert packaging.Version(min_ver) >= packaging.Version("3.0.0")


def test_the_superseded_fork_has_no_version_range_left():
    """Nothing installs it any more; it is only ever uninstalled."""
    assert ALFS_LEGACY_DIST not in _VERSION_RANGES


# ---------------------------------------------------------------------------
# Dialog wiring - read as text, the dialog module needs QtWidgets
# ---------------------------------------------------------------------------

def _manager_source():
    return (PLUGIN_DIR / "bambi_dependency_manager.py").read_text(
        encoding="utf-8")


def test_the_dialog_offers_two_dropdowns():
    source = _manager_source()
    assert "self._engine_combo = self._make_alfs_combo(" in source
    assert "self._raycaster_combo = self._make_alfs_combo(" in source
    assert "alfs_runtime.ENGINES" in source
    assert "alfs_runtime.RAYCASTERS" in source


def test_the_dropdowns_start_from_the_project_not_from_the_install():
    """There is no installed distribution to read the engine off any more -
    one package provides all three - so the project is the only source."""
    source = _manager_source()
    assert "engine, raycaster = read_alfs_selection()" in source
    assert "AlfsTorch" not in source.split("def _refresh_alfs_hint")[0]


def test_changing_a_dropdown_saves_to_the_project_immediately():
    source = _manager_source()
    start = source.index("def _on_alfs_selection_changed")
    body = source[start:start + 700]
    assert "write_alfs_selection(" in body


def test_install_removes_the_superseded_fork_first():
    source = _manager_source()
    start = source.index("def _install_alfs_py")
    body = source[start:start + 1600]
    assert "uninstall" in body and "legacy_dist" in body
    assert body.index("uninstall") < body.index("--force-reinstall")


def test_install_uses_the_selected_spec_and_carries_the_extras():
    source = _manager_source()
    start = source.index("def _install_alfs_py")
    body = source[start:start + 1600]
    assert "_selected_alfs_spec()" in body
    assert "git_url" in body and "zip_url" in body
    # The ZIP fallback installs from a local path, which needs the extras
    # appended explicitly - the URL cannot carry them.
    assert "extras=spec['extras_suffix']" in body


def test_no_reference_to_the_removed_fork_remains():
    """alfs_pytorch was a separate repository; 3.0 is one package."""
    source = _manager_source()
    assert "alfs_pytorch" not in source
    assert "_torch_backend_check" not in source


def test_the_dropdowns_use_the_qt6_safe_signal():
    """``currentIndexChanged`` carries an int on both Qt5 and Qt6."""
    source = _manager_source()
    assert "currentIndexChanged.connect" in source
