# -*- coding: utf-8 -*-
"""Every relative import of the plugin leads somewhere.

Most imports of the plugin are made inside a function, where they are needed
- so that the panel opens without the heavy packages, and so that a step
that is not used costs nothing. Such an import is not tried until its
function runs: one dot too few in a sub-package (``from .core import x`` in
``dock/``, where it has to be ``from ..core import x``) is an error that
only the user finds, on the day they press that button.

This reads every module without running it and follows each relative import
to the file it names - and, where a name is taken from a module, checks that
the module has it.
"""
import ast
import os

import pytest

PACKAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bambi_wildlife_detection")


def _modules():
    found = []
    for folder, folders, files in os.walk(PACKAGE):
        folders[:] = sorted(f for f in folders if f != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                found.append(os.path.join(folder, name))
    return found


def _module_file(base: str, dotted: str):
    """The file of module *dotted* below folder *base*: a ``.py`` file or a
    package's ``__init__.py``; ``None`` when there is neither."""
    path = os.path.join(base, *dotted.split(".")) if dotted else base
    if os.path.isfile(path + ".py"):
        return path + ".py"
    if os.path.isfile(os.path.join(path, "__init__.py")):
        return os.path.join(path, "__init__.py")
    return None


def _bound_at_top_level(tree) -> set:
    """Every name a module binds when it is imported - also inside ``if``,
    ``try`` and ``with`` at its top level. ``None`` in the set means the
    module takes everything from another one (``import *``): any name may
    be there."""
    names = set()

    def targets(node):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                targets(element)
        elif isinstance(node, ast.Starred):
            targets(node.value)

    def visit(statements):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    targets(target)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets(node.target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(None if alias.name == "*"
                              else (alias.asname or alias.name))
            elif isinstance(node, (ast.If, ast.For, ast.While)):
                if isinstance(node, ast.For):
                    targets(node.target)
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars is not None:
                        targets(item.optional_vars)
                visit(node.body)

    visit(tree.body)
    return names


_trees = {}


def _tree(path):
    if path not in _trees:
        with open(path, encoding="utf-8") as handle:
            _trees[path] = ast.parse(handle.read())
    return _trees[path]


def _problems(path) -> list:
    found = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.ImportFrom) or node.level < 1:
            continue
        base = os.path.dirname(path)
        for _step in range(node.level - 1):
            base = os.path.dirname(base)
        said = "." * node.level + (node.module or "")
        where = f"{os.path.relpath(path, PACKAGE)}:{node.lineno}"
        if not os.path.abspath(base).startswith(PACKAGE):
            found.append(f"{where}: 'from {said}' leaves the plugin")
            continue
        target = _module_file(base, node.module or "")
        if target is None:
            found.append(f"{where}: 'from {said}' names no module")
            continue
        if os.path.basename(target) != "__init__.py":
            folder = None                       # a module has no sub-modules
        else:
            folder = os.path.dirname(target)
        bound = _bound_at_top_level(_tree(target))
        for alias in node.names:
            if alias.name == "*" or None in bound or alias.name in bound:
                continue
            if folder and _module_file(folder, alias.name):
                continue
            found.append(f"{where}: 'from {said} import {alias.name}' - "
                         f"{os.path.relpath(target, PACKAGE)} has no such name")
    return found


def test_the_plugin_has_modules():
    names = {os.path.relpath(path, PACKAGE) for path in _modules()}
    assert "bambi_dock_widget.py" in names
    assert os.path.join("core", "store.py") in names
    assert os.path.join("dock", "input_tab.py") in names
    assert len(names) > 100


@pytest.mark.parametrize(
    "path", _modules(), ids=lambda p: os.path.relpath(p, PACKAGE))
def test_relative_imports_lead_somewhere(path):
    assert _problems(path) == []


# --- the check itself --------------------------------------------------------

def _write(folder, name, text):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    root = tmp_path / "plugin"
    _write(root, "__init__.py", "")
    _write(root, "tools.py", "LIMIT = 3\n\ndef helper():\n    pass\n")
    _write(root, "core/__init__.py", "")
    _write(root, "core/store.py", "try:\n    import numpy\nexcept ImportError:\n"
                                  "    numpy = None\nKIND = 'x'\n")
    _write(root, "dock/__init__.py", "")
    monkeypatch.setitem(globals(), "PACKAGE", str(root))
    _trees.clear()
    yield root
    _trees.clear()


def test_one_dot_too_few_in_a_sub_package_is_found(plugin):
    path = _write(plugin, "dock/part.py",
                  "def run():\n    from .core import store\n    return store\n")
    assert _problems(path) == ["dock/part.py:2: 'from .core' names no module"]


def test_the_right_number_of_dots_passes(plugin):
    path = _write(plugin, "dock/part.py",
                  "def run():\n"
                  "    from ..core import store\n"
                  "    from ..core.store import KIND, numpy\n"
                  "    from .. import tools\n"
                  "    from ..tools import helper, LIMIT\n"
                  "    return store, KIND, numpy, tools, helper, LIMIT\n")
    assert _problems(path) == []


def test_a_name_the_module_does_not_have_is_found(plugin):
    path = _write(plugin, "dock/part.py",
                  "def run():\n    from ..tools import missing\n")
    assert _problems(path) == [
        "dock/part.py:2: 'from ..tools import missing' - tools.py has no such name"]


def test_a_module_that_is_not_there_is_found(plugin):
    path = _write(plugin, "part.py", "from .nowhere import thing\n")
    assert _problems(path) == ["part.py:1: 'from .nowhere' names no module"]


def test_leaving_the_plugin_is_found(plugin):
    path = _write(plugin, "part.py", "from ..elsewhere import thing\n")
    assert _problems(path) == ["part.py:1: 'from ..elsewhere' leaves the plugin"]
