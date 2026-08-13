# -*- coding: utf-8 -*-
"""Architecture guard for the headless ``core`` package.

Enforces the boundary documented in ``bambi_wildlife_detection/core/__init__``
and DECOUPLING_PLAN.md: core modules must be importable without QGIS and
without the heavy runtime dependencies. GUI toolkits are forbidden outright;
heavy dependencies are allowed only inside function bodies (lazy imports).
"""
import ast
import importlib
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parent.parent / "bambi_wildlife_detection" / "core"

# Never allowed in core, not even lazily.
FORBIDDEN_ANYWHERE_PREFIXES = (
    "qgis.PyQt.QtWidgets",
    "qgis.PyQt.QtGui",
    "qgis.core",
    "qgis.gui",
)

# Allowed only as lazy (function-local) imports, never at module level.
HEAVY_TOP_LEVEL_PACKAGES = {
    "alfspy", "bambi", "boxmot", "cv2", "geopandas", "fiona", "huggingface_hub",
    "pycolmap", "rasterio", "shapely", "simplekml", "torch", "transformers",
    "trimesh", "pyrr",
}

CORE_MODULES = sorted(p.stem for p in CORE_DIR.glob("*.py") if p.stem != "__init__")


def _imported_names(nodes):
    """Yield (dotted_name, node) for Import/ImportFrom in *nodes* (non-recursive)."""
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module, node


def _is_forbidden_anywhere(name):
    return any(name == p or name.startswith(p + ".")
               for p in FORBIDDEN_ANYWHERE_PREFIXES)


def _is_heavy(name):
    return name.split(".")[0] in HEAVY_TOP_LEVEL_PACKAGES


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_no_gui_or_heavy_imports_at_module_level(module_name):
    source = (CORE_DIR / f"{module_name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Module-level imports (including inside top-level if/try blocks).
    top_level_nodes = list(tree.body)
    for node in tree.body:
        if isinstance(node, (ast.If, ast.Try)):
            top_level_nodes.extend(ast.walk(node))

    violations = [
        name for name, _ in _imported_names(top_level_nodes)
        if _is_forbidden_anywhere(name) or _is_heavy(name)
    ]
    assert not violations, (
        f"core/{module_name}.py imports {violations} at module level — "
        "GUI toolkits are forbidden in core; heavy dependencies must be "
        "imported lazily inside functions (see core/__init__.py)."
    )


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_no_gui_imports_even_lazily(module_name):
    source = (CORE_DIR / f"{module_name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations = [
        name for name, _ in _imported_names(ast.walk(tree))
        if _is_forbidden_anywhere(name)
    ]
    assert not violations, (
        f"core/{module_name}.py imports GUI/QGIS modules {violations} — "
        "these are forbidden in core even inside functions."
    )


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_importable_under_stub(module_name):
    # conftest installs the qgis stub; anything beyond it must not be needed
    # at import time.
    module = importlib.import_module(f"bambi_wildlife_detection.core.{module_name}")
    assert module is not None


PLUGIN_DIR = CORE_DIR.parent


def test_plugin_never_imports_moderngl():
    """The rasteriser is alfspy's dependency, not the plugin's.

    Naming ModernGL here would tie the plugin to one alfspy backend and break
    the moment it is swapped for the PyTorch build. Contexts come from
    ``core.render_context`` instead — see ``tests/test_render_context.py``.
    """
    offenders = []
    for path in PLUGIN_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, _node in _imported_names(ast.walk(tree)):
            if name == "moderngl" or name.startswith("moderngl."):
                offenders.append(str(path.relative_to(PLUGIN_DIR)))
    assert not offenders, (
        f"{sorted(set(offenders))} import moderngl directly; use "
        "core.render_context.make_render_context() instead."
    )


def test_plugin_does_not_call_the_backend_specific_factory():
    """``make_mgl_context`` exists only on the ModernGL build of alfspy."""
    offenders = [
        str(path.relative_to(PLUGIN_DIR))
        for path in PLUGIN_DIR.rglob("*.py")
        if path.name != "render_context.py"
        and "make_mgl_context" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{sorted(offenders)} call make_mgl_context(); use "
        "core.render_context.make_render_context() instead."
    )
