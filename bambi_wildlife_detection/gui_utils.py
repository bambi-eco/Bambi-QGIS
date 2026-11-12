# -*- coding: utf-8 -*-
"""Small Qt helpers shared by the plugin's dialogs."""

from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtCore import QEvent, QSize
from qgis.PyQt.QtWidgets import QApplication, QStackedWidget, QTabWidget


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


class FitTabWidget(QTabWidget):
    """A tab widget as tall as the page it is showing.

    A plain ``QTabWidget`` asks for the height of its *tallest* page whatever
    page is in front, so inside a scroll area every short tab is followed by
    the empty space of the longest one - on the dock, a long scroll past
    nothing to reach the log. This one asks for the height of the current
    page only and says so again whenever that changes: when another tab is
    chosen, and when the page itself grows or shrinks (a nested tab widget
    switching its own page, a section being shown or hidden).

    The width stays that of the widest page, so the dock does not change
    its width while clicking through the tabs.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(self._refit)

    def _refit(self, _index: int = 0) -> None:
        self.updateGeometry()

    def _fitted(self, full: QSize, page_hint, stack_hint) -> QSize:
        """*full* (the hint for the tallest page) cut down to the current
        page: everything around the pages - tab bar, frame - is kept."""
        page = self.currentWidget()
        stack = self.findChild(QStackedWidget)
        if page is None or stack is None:
            return full
        around = full.height() - stack_hint(stack).height()
        # A page without a layout has no hint of its own, only a minimum.
        height = max(page_hint(page).height(), page.minimumHeight())
        return QSize(full.width(), height + max(0, around))

    def sizeHint(self) -> QSize:
        return self._fitted(super().sizeHint(),
                            lambda w: w.sizeHint(), lambda w: w.sizeHint())

    def minimumSizeHint(self) -> QSize:
        return self._fitted(super().minimumSizeHint(),
                            lambda w: w.minimumSizeHint(),
                            lambda w: w.minimumSizeHint())

    def hasHeightForWidth(self) -> bool:
        page = self.currentWidget()
        return page is not None and page.hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:
        """Height for *width* of the current page alone. The inherited
        answer is again the tallest page's, and it is the one a scroll area
        goes by when the pages hold word-wrapped text."""
        page = self.currentWidget()
        stack = self.findChild(QStackedWidget)
        if page is None or stack is None or not page.hasHeightForWidth():
            return self.sizeHint().height()
        full = super().sizeHint()
        stack_hint = stack.sizeHint()
        around = max(0, full.height() - stack_hint.height())
        beside = max(0, full.width() - stack_hint.width())
        return page.heightForWidth(max(0, width - beside)) + around

    def event(self, event) -> bool:
        handled = super().event(event)
        # A page whose own size changed tells its tab widget with a layout
        # request, and a QTabWidget keeps that to itself; pass it on, or
        # the layouts above would go on using the old height.
        if event.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return handled
