# -*- coding: utf-8 -*-
"""OpenGL side of the 3D Flight Viewer.

Kept apart from ``core`` (which must not touch Qt) and imported lazily by
:mod:`bambi_flight_viewer`, so the plugin loads without PyOpenGL / pyqtgraph.
"""

from typing import List


def missing_dependencies() -> List[str]:
    """Packages the interactive 3D view needs that are not installed."""
    missing = []
    for label, module in (("PyOpenGL", "OpenGL.GL"),
                          ("pyqtgraph", "pyqtgraph")):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - any import failure means unusable
            missing.append(label)
    return missing
