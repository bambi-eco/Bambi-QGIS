# -*- coding: utf-8 -*-
"""The pinned GitHub releases.

``ALFS_PY_TAG`` / ``BAMBI_DETECTION_TAG`` decide what the dependency manager
downloads; the matching ``_VERSION_RANGES`` lower bounds decide what it calls
out of date. If the two drift apart, the plugin installs a release it then
flags as untested, or worse accepts one that predates the pose→camera fix (see
``tests_qgis/test_camera_pose_conventions.py``).
"""
import pathlib
import re

import pytest

from bambi_wildlife_detection.core.dependency_ops import (
    ALFS_PY_TAG, BAMBI_DETECTION_TAG, _VERSION_RANGES)

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent / "bambi_wildlife_detection"

# (tag, distribution name, installer function, repo slug)
PINS = [
    (ALFS_PY_TAG, "AlfsPy", "_install_alfs_py", "alfs_py"),
    (BAMBI_DETECTION_TAG, "bambi-detection", "_install_bambi_detection",
     "bambi_detection"),
]


@pytest.mark.parametrize("tag,dist,_fn,_repo", PINS)
def test_tag_is_a_version_tag(tag, dist, _fn, _repo):
    assert re.fullmatch(r"v\d+\.\d+\.\d+", tag), f"{dist}: {tag}"


@pytest.mark.parametrize("tag,dist,_fn,_repo", PINS)
def test_pin_matches_the_accepted_lower_bound(tag, dist, _fn, _repo):
    """Installing the pinned tag must produce an install the plugin calls ok."""
    min_ver, _max_ver = _VERSION_RANGES[dist]
    assert min_ver == tag.lstrip("v")


@pytest.mark.parametrize("tag,dist,_fn,_repo", PINS)
def test_no_upper_bound(tag, dist, _fn, _repo):
    """A newer release must not be reported as untested by default."""
    assert _VERSION_RANGES[dist][1] is None


def test_alfspy_bound_covers_the_oblique_heading_fix():
    """2.1.0 is the first release that steers the tilt by the gimbal heading."""
    packaging = pytest.importorskip("packaging.version")
    min_ver, _max_ver = _VERSION_RANGES["AlfsPy"]
    assert packaging.Version(min_ver) >= packaging.Version("2.1.0")


def test_bambi_detection_bound_covers_the_engine_release():
    """1.0.0 is the engine-first release the plugin's steps are built on."""
    packaging = pytest.importorskip("packaging.version")
    min_ver, _max_ver = _VERSION_RANGES["bambi-detection"]
    assert packaging.Version(min_ver) >= packaging.Version("1.0.0")


def test_bambi_detection_urls_use_the_pinned_tag():
    """Read as text: the dialog module needs QtWidgets, which the stub omits."""
    source = (PLUGIN_DIR / "bambi_dependency_manager.py").read_text(encoding="utf-8")
    start = source.index("def _install_bambi_detection")
    body = source[start:start + 600]
    assert "refs/tags/" in body
    assert "bambi_detection.git@" in body
    assert "bambi_detection/archive/refs/heads/main" not in source


def test_alfspy_urls_come_from_the_backend_spec():
    """alfspy has two selectable repos, so its URLs are built in core.

    ``tests/test_alfs_backend_choice.py`` pins the URLs themselves; here we only
    check the dialog defers to the spec instead of hard-coding one backend.
    """
    source = (PLUGIN_DIR / "bambi_dependency_manager.py").read_text(encoding="utf-8")
    start = source.index("def _install_alfs_py")
    body = source[start:start + 1200]
    assert "_selected_alfs_spec()" in body
    assert "alfs_py/archive/refs/heads/main" not in source
    assert "alfs_pytorch/archive/refs/heads/main" not in source
