# -*- coding: utf-8 -*-
"""Selecting between the two interchangeable alfspy backends.

The Dependency Manager's "Use the PyTorch backend" checkbox resolves to an
install spec here. Both backends provide a package called ``alfspy``, so the
spec must also name the distribution to remove — otherwise two dists end up
owning the same import path and pip will not clean it up on its own.
"""
import pathlib
import re

import pytest

from bambi_wildlife_detection.core.dependency_ops import (
    ALFS_BACKENDS, ALFS_PY_TAG, ALFS_TORCH_TAG, _VERSION_RANGES,
    alfs_backend_spec)

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent / "bambi_wildlife_detection"


def test_moderngl_is_the_default_selection():
    """Unchecked box (the default) must keep the current ModernGL backend."""
    spec = alfs_backend_spec(False)
    assert spec["dist"] == "AlfsPy"
    assert spec["repo"] == "alfs_py"
    assert spec["tag"] == ALFS_PY_TAG


def test_checked_box_selects_the_torch_backend():
    spec = alfs_backend_spec(True)
    assert spec["dist"] == "AlfsTorch"
    assert spec["repo"] == "alfs_pytorch"
    assert spec["tag"] == ALFS_TORCH_TAG


@pytest.mark.parametrize("use_torch", [False, True])
def test_spec_names_the_distribution_to_remove(use_torch):
    """Both ship ``alfspy``; the other one has to go or imports get ambiguous."""
    spec = alfs_backend_spec(use_torch)
    assert spec["other_dist"] != spec["dist"]
    assert {spec["dist"], spec["other_dist"]} == {"AlfsPy", "AlfsTorch"}


@pytest.mark.parametrize("use_torch", [False, True])
def test_urls_point_at_the_pinned_tag(use_torch):
    spec = alfs_backend_spec(use_torch)
    repo, tag = spec["repo"], spec["tag"]
    assert spec["zip_url"] == (
        f"https://github.com/bambi-eco/{repo}/archive/refs/tags/{tag}.zip")
    assert spec["git_url"] == f"git+https://github.com/bambi-eco/{repo}.git@{tag}"
    assert "refs/heads/main" not in spec["zip_url"]


@pytest.mark.parametrize("key", sorted(ALFS_BACKENDS))
def test_every_backend_has_a_version_floor_matching_its_tag(key):
    entry = ALFS_BACKENDS[key]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", entry["tag"]), entry
    min_ver, max_ver = _VERSION_RANGES[entry["dist"]]
    assert min_ver == entry["tag"].lstrip("v")
    assert max_ver is None


def test_backends_are_labelled_for_the_ui():
    assert ALFS_BACKENDS["moderngl"]["label"] == "ModernGL"
    assert ALFS_BACKENDS["torch"]["label"] == "PyTorch"


# ---------------------------------------------------------------------------
# Dialog wiring — read as text, the dialog module needs QtWidgets
# ---------------------------------------------------------------------------

def _manager_source():
    return (PLUGIN_DIR / "bambi_dependency_manager.py").read_text(encoding="utf-8")


def test_checkbox_defaults_to_the_installed_backend_not_a_stored_flag():
    """State comes from what is installed, so it cannot drift from reality."""
    source = _manager_source()
    assert "_torch_backend_check.setChecked(" in source
    assert "_get_version_status('AlfsTorch'" in source
    assert "QSettings" not in source


def test_install_removes_the_other_distribution_first():
    source = _manager_source()
    start = source.index("def _install_alfs_py")
    body = source[start:start + 1200]
    assert "uninstall" in body and "other_dist" in body
    assert body.index("uninstall") < body.index("--force-reinstall")


def test_install_uses_the_selected_spec():
    source = _manager_source()
    start = source.index("def _install_alfs_py")
    body = source[start:start + 1200]
    assert "_selected_alfs_spec()" in body
    assert "git_url" in body and "zip_url" in body


def test_toggle_repoints_the_status_row():
    source = _manager_source()
    start = source.index("def _on_backend_toggled")
    body = source[start:start + 600]
    assert "_dist_names['alfs_py']" in body
    assert "_refresh_single_status" in body


def test_checkbox_uses_the_qt6_safe_signal():
    """``stateChanged`` is deprecated in Qt6; ``toggled`` works on both."""
    source = _manager_source()
    assert "_torch_backend_check.toggled.connect" in source
    assert "_torch_backend_check.stateChanged" not in source
