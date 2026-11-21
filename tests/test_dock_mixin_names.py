# -*- coding: utf-8 -*-
"""The dock is put together from mixins that share one namespace: a method
defined by two of them is silently the first one's. (The dataset export's
``_dataset_flights`` once shadowed the dataset download's, and each got
the other's answer.)"""
import ast
import glob
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "bambi_wildlife_detection")


def defined(paths):
    """``{method name: {where}}`` of every method a mixin defines."""
    found = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith("Mixin"):
                continue
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    found.setdefault(member.name, set()).add(
                        f"{os.path.basename(path)}:{node.name}")
    return found


def test_no_two_parts_of_the_dock_define_the_same_method():
    paths = sorted(glob.glob(os.path.join(ROOT, "dock", "*.py")))
    paths.append(os.path.join(ROOT, "bambi_environment_panel.py"))

    twice = {name: sorted(where) for name, where in defined(paths).items()
             if len(where) > 1}

    assert twice == {}


def test_no_two_parts_of_the_processor_define_the_same_method():
    paths = sorted(glob.glob(os.path.join(ROOT, "bambi_survey_*.py")))
    paths.append(os.path.join(ROOT, "bambi_environment.py"))

    twice = {name: sorted(where) for name, where in defined(paths).items()
             if len(where) > 1}

    assert twice == {}
