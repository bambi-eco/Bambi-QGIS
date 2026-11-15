# -*- coding: utf-8 -*-
"""Every GUI module of the plugin imports under Qt 6."""
import importlib
import os

import pytest

PACKAGE = "bambi_wildlife_detection"
ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), PACKAGE)


def _gui_modules():
    names = []
    for name in sorted(os.listdir(ROOT)):
        if name.endswith(".py") and name != "__init__.py":
            names.append(f"{PACKAGE}.{name[:-3]}")
    for package in ("dock", "viewer3d"):
        for name in sorted(os.listdir(os.path.join(ROOT, package))):
            if name.endswith(".py") and name != "__init__.py":
                names.append(f"{PACKAGE}.{package}.{name[:-3]}")
    return names


@pytest.mark.parametrize("module", _gui_modules())
def test_it_imports(module, app):
    importlib.import_module(module)
