# -*- coding: utf-8 -*-
"""Unit tests for core.render_context - backend-neutral alfspy contexts.

The plugin must not care which rasteriser alfspy uses. These tests pin that:
the torch factory is preferred when present (it also selects CUDA on its own),
the ModernGL factory is the fallback, and nothing in the plugin reaches for
``moderngl`` directly.
"""
import sys

import pytest

from bambi_wildlife_detection.core import render_context
from tests.fakes import make_module


def install_alfspy(monkeypatch, *, torch=False, present=True):
    """Fake ``alfspy.render.render`` exposing one or both context factories."""
    if not present:
        for name in ("alfspy", "alfspy.render", "alfspy.render.render"):
            monkeypatch.setitem(sys.modules, name, None)
        return None

    calls = {"mgl": [], "torch": []}
    render = make_module("alfspy.render.render")

    def make_mgl_context(standalone=True):
        calls["mgl"].append({"standalone": standalone})
        return "MGL_CONTEXT"

    render.make_mgl_context = make_mgl_context
    if torch:
        def make_torch_context(device=None, **kwargs):
            calls["torch"].append({"device": device, **kwargs})
            return "TORCH_CONTEXT"
        render.make_torch_context = make_torch_context

    alfspy = make_module("alfspy")
    alfspy.__path__ = []
    pkg = make_module("alfspy.render")
    pkg.__path__ = []
    pkg.render = render
    alfspy.render = pkg
    monkeypatch.setitem(sys.modules, "alfspy", alfspy)
    monkeypatch.setitem(sys.modules, "alfspy.render", pkg)
    monkeypatch.setitem(sys.modules, "alfspy.render.render", render)
    return calls


# ---------------------------------------------------------------------------
# render_backend
# ---------------------------------------------------------------------------

def test_backend_is_torch_when_the_torch_factory_exists(monkeypatch):
    install_alfspy(monkeypatch, torch=True)
    assert render_context.render_backend() == "torch"


def test_backend_is_moderngl_without_the_torch_factory(monkeypatch):
    install_alfspy(monkeypatch, torch=False)
    assert render_context.render_backend() == "moderngl"


def test_backend_is_unavailable_without_alfspy(monkeypatch):
    install_alfspy(monkeypatch, present=False)
    assert render_context.render_backend() == "unavailable"


# ---------------------------------------------------------------------------
# make_render_context
# ---------------------------------------------------------------------------

def test_torch_backend_is_preferred(monkeypatch):
    calls = install_alfspy(monkeypatch, torch=True)
    assert render_context.make_render_context() == "TORCH_CONTEXT"
    assert calls["torch"] == [{"device": None}] or calls["torch"] == [{}]
    assert calls["mgl"] == []


def test_moderngl_backend_is_the_fallback(monkeypatch):
    calls = install_alfspy(monkeypatch, torch=False)
    assert render_context.make_render_context() == "MGL_CONTEXT"
    assert calls["mgl"] == [{"standalone": True}]


def test_device_is_forwarded_to_the_torch_backend(monkeypatch):
    calls = install_alfspy(monkeypatch, torch=True)
    render_context.make_render_context(device="cuda")
    assert calls["torch"] == [{"device": "cuda"}]


def test_device_is_ignored_by_the_moderngl_backend(monkeypatch):
    """A device argument must not break the older backend."""
    calls = install_alfspy(monkeypatch, torch=False)
    assert render_context.make_render_context(device="cuda") == "MGL_CONTEXT"
    assert calls["mgl"] == [{"standalone": True}]


def test_missing_alfspy_raises_runtime_error(monkeypatch):
    install_alfspy(monkeypatch, present=False)
    with pytest.raises(RuntimeError, match="alfspy is not available"):
        render_context.make_render_context()


# ---------------------------------------------------------------------------
# require_render_stack
# ---------------------------------------------------------------------------

def test_require_passes_with_a_complete_stack(monkeypatch):
    install_alfspy(monkeypatch, torch=True)
    monkeypatch.setitem(sys.modules, "pyrr", make_module("pyrr"))
    monkeypatch.setitem(sys.modules, "trimesh", make_module("trimesh"))
    render_context.require_render_stack()


def test_require_names_the_missing_array_libraries(monkeypatch):
    install_alfspy(monkeypatch, torch=True)
    monkeypatch.setitem(sys.modules, "pyrr", make_module("pyrr"))
    monkeypatch.setitem(sys.modules, "trimesh", None)
    with pytest.raises(ImportError) as excinfo:
        render_context.require_render_stack()
    assert "trimesh" in str(excinfo.value)
    assert "pyrr" not in str(excinfo.value).split("pip install")[1]


def test_require_reports_missing_alfspy(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrr", make_module("pyrr"))
    monkeypatch.setitem(sys.modules, "trimesh", make_module("trimesh"))
    install_alfspy(monkeypatch, present=False)
    with pytest.raises(ImportError, match="alfspy is not installed"):
        render_context.require_render_stack()


def test_require_never_mentions_moderngl(monkeypatch):
    """The rasteriser is alfspy's dependency, not the plugin's."""
    install_alfspy(monkeypatch, torch=True)
    monkeypatch.setitem(sys.modules, "pyrr", None)
    monkeypatch.setitem(sys.modules, "trimesh", None)
    with pytest.raises(ImportError) as excinfo:
        render_context.require_render_stack()
    assert "moderngl" not in str(excinfo.value).lower()
