# -*- coding: utf-8 -*-
"""Fixtures for the real-QGIS smoke tests.

Unlike the unit suite (which stubs the ``qgis`` package), these tests run
inside a headless QGIS: a real ``QgsApplication`` is started and the plugin's
Qt widgets are constructed for real. Run via the QGIS image only:

    docker compose run --rm qgis-tests
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Start a headless QGIS application once, before anything touches qgis.core
# providers. start_app() is safe to call repeatedly and returns the running
# QgsApplication.
from qgis.testing import start_app  # noqa: E402

QGIS_APP = start_app()

from qgis.core import QgsProject  # noqa: E402
from qgis.testing.mocked import get_iface  # noqa: E402


@pytest.fixture
def iface():
    """A mocked QGIS interface (real QgsMapCanvas underneath)."""
    return get_iface()


@pytest.fixture
def clean_project():
    """A pristine QgsProject for a test, cleared again afterwards."""
    QgsProject.instance().clear()
    yield QgsProject.instance()
    QgsProject.instance().clear()


@pytest.fixture
def dock(iface, clean_project):
    """A freshly constructed BambiDockWidget on an empty project."""
    from bambi_wildlife_detection.bambi_dock_widget import BambiDockWidget
    widget = BambiDockWidget(iface)
    yield widget
    widget.deleteLater()


@pytest.fixture(autouse=True)
def flight_name_prompt(monkeypatch):
    """Since 9.0.0 a flight needs a name from the user: setting a target
    folder asks for it. The tests answer that prompt with the folder-derived
    name, so a test that only sets a target folder still gets its flight.
    A test about the prompt itself patches ``_prompt_flight_name`` again."""
    from bambi_wildlife_detection.bambi_dock_widget import BambiDockWidget
    from bambi_wildlife_detection.core import flights
    monkeypatch.setattr(BambiDockWidget, "_prompt_flight_name",
                        lambda self, folder: flights.default_name(folder))
