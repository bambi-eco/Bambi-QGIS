# -*- coding: utf-8 -*-
"""Unit tests for core.render_context - engine-neutral alfspy contexts.

The plugin must not name a render engine at any call site. These tests pin
that: the context comes from alfspy's single ``make_context``, the engine is
whatever ``$ALFS_ENGINE`` resolves to, and a selected-but-missing engine fails
with a message that names the ones that do work rather than an ImportError from
three frames down.
"""
import builtins
import sys

import pytest

from bambi_wildlife_detection.core import render_context
from tests.fakes import make_module


def install_alfspy(monkeypatch, *, present=True, engines=("moderngl",),
                   resolved="moderngl", fails=None, render=True):
    """Fake ``alfspy.core.backends`` - the 3.0 context registry.

    *engines* are the ones that can create a context here, *resolved* is what
    ``$ALFS_ENGINE`` resolves to, and *fails* makes ``make_context`` raise.
    *render* installs ``alfspy.render.render``, which the real package cannot
    import without torch - pass ``False`` to reproduce that.
    """
    if not present:
        for name in ("alfspy", "alfspy.core", "alfspy.core.backends",
                     "alfspy.render", "alfspy.render.render"):
            monkeypatch.setitem(sys.modules, name, None)
        return None

    calls = []
    backends = make_module("alfspy.core.backends")

    def make_context(engine=None, device=None, **options):
        calls.append({"engine": engine, "device": device, **options})
        if fails:
            raise RuntimeError(fails)
        return f"CONTEXT:{engine or resolved}"

    backends.make_context = make_context
    backends.available_engines = lambda: list(engines)
    backends.resolve_engine = lambda engine=None: engine or resolved

    alfspy = make_module("alfspy")
    alfspy.__path__ = []
    core = make_module("alfspy.core")
    core.__path__ = []
    core.backends = backends
    alfspy.core = core
    monkeypatch.setitem(sys.modules, "alfspy", alfspy)
    monkeypatch.setitem(sys.modules, "alfspy.core", core)
    monkeypatch.setitem(sys.modules, "alfspy.core.backends", backends)

    if render:
        render_pkg = make_module("alfspy.render")
        render_pkg.__path__ = []
        render_mod = make_module("alfspy.render.render")
        render_pkg.render = render_mod
        alfspy.render = render_pkg
        monkeypatch.setitem(sys.modules, "alfspy.render", render_pkg)
        monkeypatch.setitem(sys.modules, "alfspy.render.render", render_mod)
    return calls


# ---------------------------------------------------------------------------
# render_backend / available_engines
# ---------------------------------------------------------------------------

def test_backend_is_whatever_alfspy_resolves(monkeypatch):
    """The engine is $ALFS_ENGINE's business, not a property of the install."""
    install_alfspy(monkeypatch, resolved="torch")
    assert render_context.render_backend() == "torch"


def test_backend_is_unavailable_without_alfspy(monkeypatch):
    install_alfspy(monkeypatch, present=False)
    assert render_context.render_backend() == "unavailable"


def test_available_engines_are_reported(monkeypatch):
    install_alfspy(monkeypatch, engines=("moderngl", "vulkan"))
    assert render_context.available_engines() == ["moderngl", "vulkan"]


def test_available_engines_is_empty_without_alfspy(monkeypatch):
    install_alfspy(monkeypatch, present=False)
    assert render_context.available_engines() == []


def test_a_probe_that_raises_is_not_an_error(monkeypatch):
    """Probing creates contexts, which is exactly what can blow up."""
    install_alfspy(monkeypatch)
    backends = sys.modules["alfspy.core.backends"]

    def explode():
        raise RuntimeError("no GL driver")
    backends.available_engines = explode
    assert render_context.available_engines() == []


# ---------------------------------------------------------------------------
# make_render_context
# ---------------------------------------------------------------------------

def test_the_context_comes_from_the_single_factory(monkeypatch):
    calls = install_alfspy(monkeypatch, resolved="torch")
    assert render_context.make_render_context() == "CONTEXT:torch"
    # No engine named by the caller: the project's choice reaches alfspy
    # through the environment, not through an argument.
    assert calls == [{"engine": None, "device": None}]


def test_the_device_is_forwarded(monkeypatch):
    calls = install_alfspy(monkeypatch)
    render_context.make_render_context(device="cuda")
    assert calls == [{"engine": None, "device": "cuda"}]


def test_an_engine_can_still_be_forced(monkeypatch):
    calls = install_alfspy(monkeypatch)
    assert render_context.make_render_context(engine="vulkan") == \
        "CONTEXT:vulkan"
    assert calls == [{"engine": "vulkan", "device": None}]


def test_missing_alfspy_raises_runtime_error(monkeypatch):
    install_alfspy(monkeypatch, present=False)
    with pytest.raises(RuntimeError, match="alfspy is not available"):
        render_context.make_render_context()


def test_a_missing_engine_names_the_ones_that_work(monkeypatch):
    """Selecting an engine whose extra was never installed is the common
    failure, so the message has to say what to pick instead."""
    install_alfspy(monkeypatch, engines=("moderngl",), resolved="vulkan",
                   fails="No module named 'wgpu'")
    with pytest.raises(RuntimeError) as excinfo:
        render_context.make_render_context()
    message = str(excinfo.value)
    assert "vulkan" in message
    assert "moderngl" in message
    assert "Dependency Manager" in message


def test_a_machine_with_no_working_engine_says_so(monkeypatch):
    install_alfspy(monkeypatch, engines=(), resolved="moderngl",
                   fails="no GL driver")
    with pytest.raises(RuntimeError, match="No render engine works"):
        render_context.make_render_context()


# ---------------------------------------------------------------------------
# require_render_stack
# ---------------------------------------------------------------------------

def test_require_passes_with_a_complete_stack(monkeypatch):
    install_alfspy(monkeypatch)
    monkeypatch.setitem(sys.modules, "pyrr", make_module("pyrr"))
    monkeypatch.setitem(sys.modules, "trimesh", make_module("trimesh"))
    render_context.require_render_stack()


def test_require_names_the_missing_array_libraries(monkeypatch):
    install_alfspy(monkeypatch)
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


def test_require_explains_alfspys_eager_torch_import(monkeypatch):
    """The first alfspy 3.0.0 build loaded its PyTorch rasteriser with
    ``alfspy.render``, so a ModernGL-only install failed there with a bare "No
    module named 'torch'" that explains nothing. The fix shipped under the same
    version, so an install made before it cannot be told apart by version."""
    install_alfspy(monkeypatch, render=False)
    monkeypatch.setitem(sys.modules, "pyrr", make_module("pyrr"))
    monkeypatch.setitem(sys.modules, "trimesh", make_module("trimesh"))

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "alfspy.render.render":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        render_context.require_render_stack()
    message = str(excinfo.value)
    assert "PyTorch" in message
    assert "Dependency Manager" in message


def test_require_never_mentions_moderngl(monkeypatch):
    """The rasteriser is alfspy's dependency, not the plugin's."""
    install_alfspy(monkeypatch)
    monkeypatch.setitem(sys.modules, "pyrr", None)
    monkeypatch.setitem(sys.modules, "trimesh", None)
    with pytest.raises(ImportError) as excinfo:
        render_context.require_render_stack()
    assert "moderngl" not in str(excinfo.value).lower()
