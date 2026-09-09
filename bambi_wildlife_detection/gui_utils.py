# -*- coding: utf-8 -*-
"""Small Qt helpers shared by the plugin's dialogs."""

from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QApplication


def fit_to_screen(widget, width: int, height: int):
    """Resize *widget* to its preferred size, capped to the screen it opens on.

    Dialogs whose preferred size was chosen on a large monitor otherwise open
    taller than a laptop screen, with their buttons below the desktop edge.
    Widgets that must survive the cap need their content in a QScrollArea -
    resize() cannot shrink a window below its layout's minimum height.
    """
    screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
    if screen is None:
        widget.resize(width, height)
        return

    avail = screen.availableGeometry()
    # Leave room for the window frame, which resize() does not include.
    max_w = max(640, avail.width() - 40)
    max_h = max(480, avail.height() - 80)
    widget.resize(min(width, max_w), min(height, max_h))

    frame = widget.frameGeometry()
    frame.moveCenter(avail.center())
    widget.move(max(avail.left(), frame.left()), max(avail.top(), frame.top()))


#: Scope the plugin's settings live under in a QGIS project file.
PLUGIN_SCOPE = "BambiWildlifeDetection"

#: Project entries holding the alfspy engine and ray-caster choice.  They sit
#: in the QGIS project rather than in a flight's ``project.gpkg``: the choice
#: describes how *this machine* renders, not what a flight contains, and a
#: flight folder is shared between people whose installs differ.
_ALFS_ENGINE_ENTRY = "Processing/AlfsEngine"
_ALFS_RAYCASTER_ENTRY = "Processing/AlfsRaycaster"

#: The Hugging Face token, in the user's QGIS settings - never in a project
#: file, which gets shared. One key serves the Classification tab and the
#: Segmentation tool: both download gated Meta models with it.
HF_TOKEN_SETTING = f"{PLUGIN_SCOPE}/classification/hfToken"


def read_hf_token() -> str:
    """The stored token, or ``""`` (the environment / ``hf auth login`` are
    consulted at run time by ``core.hf_access.resolve_token``)."""
    from qgis.core import QgsSettings
    return QgsSettings().value(HF_TOKEN_SETTING, "", type=str) or ""


def write_hf_token(token: str) -> None:
    from qgis.core import QgsSettings
    QgsSettings().setValue(HF_TOKEN_SETTING, (token or "").strip())


def read_alfs_selection():
    """The project's ``(engine, raycaster)``, falling back to the defaults.

    Reads through :mod:`core.alfs_runtime`, so a value written by a newer
    plugin - or edited by hand into something unknown - loads as the default
    rather than reaching alfspy and failing there.
    """
    from .core import alfs_runtime

    try:
        from qgis.core import QgsProject
        project = QgsProject.instance()
        engine, _ok = project.readEntry(PLUGIN_SCOPE, _ALFS_ENGINE_ENTRY, "")
        raycaster, _ok = project.readEntry(
            PLUGIN_SCOPE, _ALFS_RAYCASTER_ENTRY, "")
    except Exception:
        engine = raycaster = ""
    return (alfs_runtime.resolve_engine(engine),
            alfs_runtime.resolve_raycaster(raycaster))


def write_alfs_selection(engine, raycaster):
    """Store ``(engine, raycaster)`` in the QGIS project.

    Written as soon as the dropdown changes rather than on the next save: the
    dependency manager is a separate window, and a selection that had to be
    confirmed somewhere else before it counted would be a trap.
    """
    from .core import alfs_runtime
    from qgis.core import QgsProject

    project = QgsProject.instance()
    project.writeEntry(PLUGIN_SCOPE, _ALFS_ENGINE_ENTRY,
                       alfs_runtime.resolve_engine(engine))
    project.writeEntry(PLUGIN_SCOPE, _ALFS_RAYCASTER_ENTRY,
                       alfs_runtime.resolve_raycaster(raycaster))
    project.setDirty(True)
