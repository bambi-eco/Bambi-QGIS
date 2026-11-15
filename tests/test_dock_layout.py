# -*- coding: utf-8 -*-
"""The main panel stays split by tab.

``BambiDockWidget`` was one class of 13,800 lines in one file. It still is
one class when the plugin runs - every control is an attribute of the dock
widget - but its code is kept in the ``dock`` package, one module per tab
or concern, each a class that is mixed into the dock widget.

These tests keep it that way. They read the modules and need no QGIS.
"""
import ast
import os

import pytest

PACKAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bambi_wildlife_detection")
DOCK = os.path.join(PACKAGE, "dock")
MAIN = os.path.join(PACKAGE, "bambi_dock_widget.py")

#: More lines than this in one part, and it should become two.
PART_LIMIT = 2000
#: The dock module itself holds the frame around the tabs and nothing else.
MAIN_LIMIT = 400


def _tree(path):
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read())


def _lines(path) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def _part_modules():
    return sorted(name[:-3] for name in os.listdir(DOCK)
                  if name.endswith(".py") and name != "__init__.py")


def _classes(path):
    return [node for node in _tree(path).body if isinstance(node, ast.ClassDef)]


def _dock_class():
    return next(node for node in _classes(MAIN)
                if node.name == "BambiDockWidget")


def _mixins():
    """{class name: (module, class node)} of every class that is mixed in."""
    found = {}
    for module in _part_modules():
        for node in _classes(os.path.join(DOCK, f"{module}.py")):
            if node.name.endswith("Mixin"):
                found[node.name] = (f"dock/{module}.py", node)
    for node in _classes(os.path.join(PACKAGE, "bambi_environment_panel.py")):
        if node.name.endswith("Mixin"):
            found[node.name] = ("bambi_environment_panel.py", node)
    return found


def _members(node) -> list:
    """Names a class body defines: methods and class attributes."""
    names = []
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(statement.name)
        elif isinstance(statement, ast.Assign):
            names.extend(target.id for target in statement.targets
                         if isinstance(target, ast.Name))
    return names


def test_the_dock_is_made_of_its_parts():
    bases = [ast.unparse(base) for base in _dock_class().bases]
    mixins = _mixins()

    assert bases[-1] == "QDockWidget"
    assert sorted(bases[:-1]) == sorted(mixins), \
        "every part of the dock is mixed into BambiDockWidget, and nothing else"
    # the tabs, by name
    for name in ("InputTabMixin", "ConfigurationTabMixin", "StepTabsMixin",
                 "AnalyticsTabMixin"):
        assert name in bases


def test_no_name_is_defined_by_two_parts():
    """Which of the two would run depends on the order of the base classes -
    the kind of thing nobody remembers a year later."""
    owner = {}
    twice = []
    classes = dict(_mixins())
    classes["BambiDockWidget"] = ("bambi_dock_widget.py", _dock_class())
    for name, (module, node) in sorted(classes.items()):
        for member in set(_members(node)):
            if member in owner:
                twice.append(f"{member}: {owner[member]} and {module}")
            owner[member] = module
    assert twice == []
    assert len(owner) > 350


def test_a_part_has_no_state_of_its_own():
    """Everything the dock remembers is set up in one place: its own
    ``__init__``."""
    for name, (module, node) in _mixins().items():
        assert "__init__" not in _members(node), f"{name} in {module}"
    assert "__init__" in _members(_dock_class())


def test_a_part_does_not_import_the_dock():
    """The dock imports its parts; the other way round is a circle."""
    for module in _part_modules():
        for node in ast.walk(_tree(os.path.join(DOCK, f"{module}.py"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any("bambi_dock_widget" in name for name in names), \
                f"dock/{module}.py imports the dock module"


@pytest.mark.parametrize("module", _part_modules())
def test_a_part_stays_readable(module):
    count = _lines(os.path.join(DOCK, f"{module}.py"))
    assert count <= PART_LIMIT, (
        f"dock/{module}.py has {count} lines. Give a part of it a module of "
        "its own (see dock/__init__.py for how the others are cut).")


def test_the_dock_module_is_the_frame_only():
    count = _lines(MAIN)
    assert count <= MAIN_LIMIT, (
        f"bambi_dock_widget.py has {count} lines. What belongs to a tab goes "
        "into that tab's module in dock/.")
    methods = [name for name in _members(_dock_class())
               if not name.startswith("__")]
    assert len(methods) <= 12, methods


def test_the_step_runner_is_an_object_that_knows_no_form():
    """The one part that is not mixed in: it is handed the step and the
    settings, and whoever shows a running step is told through callbacks."""
    tree = _tree(os.path.join(DOCK, "step_runner.py"))
    runner = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "StepRunner")
    assert runner.bases == []
    assert {"start", "ended", "cancel", "shut_down",
            "is_running"} <= set(_members(runner))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "qgis.PyQt.QtWidgets" not in imported
    # the dock makes one, and its parts reach it as self.steps
    main = open(MAIN, encoding="utf-8").read()
    assert "self.steps = StepRunner()" in main
    panel = open(os.path.join(DOCK, "run_panel.py"), encoding="utf-8").read()
    assert "self.steps.start(" in panel and "QThread" not in panel


def test_the_package_says_what_is_where():
    described = ast.get_docstring(_tree(os.path.join(DOCK, "__init__.py")))
    for module in _part_modules():
        assert f"``{module}``" in described, \
            f"dock/__init__.py does not say what dock/{module}.py is"
    assert "bambi_environment_panel" in described


def test_every_part_says_what_it_is():
    for module in _part_modules():
        tree = _tree(os.path.join(DOCK, f"{module}.py"))
        assert len(ast.get_docstring(tree) or "") > 40, module
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                assert ast.get_docstring(node), f"{module}.{node.name}"
