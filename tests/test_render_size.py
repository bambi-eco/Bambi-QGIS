# -*- coding: utf-8 -*-
"""Unit tests for core.render_size (ALFS/GeoTIFF output sizing)."""
import pytest

from bambi_wildlife_detection.core import render_size


# ---------------------------------------------------------------------------
# Ground-resolution mode (the historic behaviour)
# ---------------------------------------------------------------------------

def test_ground_resolution_divides_the_extent():
    assert render_size.resolve_render_size(100.0, 50.0, 0.05, {}) == (2000, 1000)


def test_ground_resolution_rounds_up():
    # A partial pixel still needs a whole pixel to land in.
    assert render_size.resolve_render_size(100.01, 50.0, 0.05, {}) == (2001, 1000)


def test_missing_flag_means_ground_resolution():
    assert not render_size.uses_fixed_size({})
    assert not render_size.uses_fixed_size({"alfs_fixed_render_size": False})


def test_zero_extent_still_yields_a_pixel():
    assert render_size.resolve_render_size(0.0, 0.0, 0.05, {}) == (1, 1)


@pytest.mark.parametrize("bad", [0, 0.0, -0.5, None])
def test_non_positive_ground_resolution_is_rejected(bad):
    with pytest.raises(ValueError):
        render_size.resolve_render_size(100.0, 50.0, bad, {})


# ---------------------------------------------------------------------------
# Fixed-render-size mode
# ---------------------------------------------------------------------------

FIXED = {
    "alfs_fixed_render_size": True,
    "alfs_render_width": 2048,
    "alfs_render_height": 2048,
}


def test_fixed_size_ignores_the_extent():
    assert render_size.resolve_render_size(100.0, 50.0, 0.05, FIXED) == (2048, 2048)
    assert render_size.resolve_render_size(9999.0, 1.0, 0.05, FIXED) == (2048, 2048)


def test_fixed_size_ignores_the_ground_resolution_entirely():
    # Not even a nonsensical m/px matters once the size is given in pixels.
    assert render_size.resolve_render_size(100.0, 50.0, 0.0, FIXED) == (2048, 2048)


def test_fixed_size_axes_are_independent():
    config = dict(FIXED, alfs_render_width=1024, alfs_render_height=4096)
    assert render_size.resolve_render_size(100.0, 100.0, 0.05, config) == (1024, 4096)


def test_fixed_size_defaults_to_2048_square():
    assert render_size.resolve_render_size(
        100.0, 50.0, 0.05, {"alfs_fixed_render_size": True}) == (
            render_size.DEFAULT_RENDER_WIDTH, render_size.DEFAULT_RENDER_HEIGHT)


# ---------------------------------------------------------------------------
# The memory cap
# ---------------------------------------------------------------------------

def test_max_dim_scales_the_longer_side_down():
    width, height = render_size.resolve_render_size(
        1000.0, 500.0, 0.01, {}, max_dim=8000)
    assert width == 8000
    assert height == 4000


def test_max_dim_leaves_a_small_render_alone():
    assert render_size.resolve_render_size(
        100.0, 50.0, 0.05, {}, max_dim=8000) == (2000, 1000)


def test_max_dim_also_caps_a_fixed_size():
    config = dict(FIXED, alfs_render_width=16000, alfs_render_height=8000)
    assert render_size.resolve_render_size(
        100.0, 50.0, 0.05, config, max_dim=8000) == (8000, 4000)


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

def test_describe_names_the_active_mode():
    assert "0.05 m/px" in render_size.describe({}, 0.05)
    assert "2048 x 2048" in render_size.describe(FIXED, 0.05)
