# -*- coding: utf-8 -*-
"""The alfs_py release pin.

``ALFS_PY_TAG`` decides what the dependency manager downloads;
``_VERSION_RANGES['AlfsPy']`` decides what it calls out of date. If the two
drift apart, the plugin installs a release it then flags as untested, or worse
accepts one that predates the pose→camera fix (see
``tests_qgis/test_camera_pose_conventions.py``).
"""
import pathlib
import re

import pytest

from bambi_wildlife_detection.core.dependency_ops import (
    ALFS_PY_TAG, _VERSION_RANGES)


def test_tag_is_a_version_tag():
    assert re.fullmatch(r"v\d+\.\d+\.\d+", ALFS_PY_TAG), ALFS_PY_TAG


def test_pin_matches_the_accepted_lower_bound():
    """Installing the pinned tag must produce an install the plugin calls ok."""
    min_ver, _max_ver = _VERSION_RANGES["AlfsPy"]
    assert min_ver == ALFS_PY_TAG.lstrip("v")


def test_lower_bound_covers_the_oblique_heading_fix():
    """2.1.0 is the first release that steers the tilt by the gimbal heading."""
    packaging = pytest.importorskip("packaging.version")
    min_ver, _max_ver = _VERSION_RANGES["AlfsPy"]
    assert packaging.Version(min_ver) >= packaging.Version("2.1.0")


def test_no_upper_bound_on_alfspy():
    """A newer alfs_py must not be reported as untested by default."""
    assert _VERSION_RANGES["AlfsPy"][1] is None


def test_install_urls_use_the_pinned_tag():
    """Read as text: the dialog module needs QtWidgets, which the stub omits."""
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "bambi_wildlife_detection" / "bambi_dependency_manager.py"
              ).read_text(encoding="utf-8")
    start = source.index("def _install_alfs_py")
    body = source[start:start + 500]
    assert "refs/tags/{ALFS_PY_TAG}" in body
    assert "alfs_py.git@{ALFS_PY_TAG}" in body
    assert "alfs_py/archive/refs/heads/main" not in source
