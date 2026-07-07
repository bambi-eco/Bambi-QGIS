# -*- coding: utf-8 -*-
"""Shared pytest configuration for the BAMBI plugin unit tests.

The unit tests exercise pure Python/NumPy logic and must run without a QGIS
installation, so a minimal stub of the ``qgis`` package is installed into
``sys.modules`` before any plugin module is imported (see tests/qgis_stub.py).
"""
import sys
from pathlib import Path

import pytest

# Make the repository root importable so tests can use
# ``from bambi_wildlife_detection... import ...`` regardless of the pytest
# invocation directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.qgis_stub import install_qgis_stubs  # noqa: E402

install_qgis_stubs()


@pytest.fixture
def processor():
    """A fresh BambiProcessor instance (its __init__ is side-effect free)."""
    from bambi_wildlife_detection.bambi_processing import BambiProcessor
    return BambiProcessor()
