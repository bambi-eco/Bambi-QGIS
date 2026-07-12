# -*- coding: utf-8 -*-
"""Small Qt helpers shared by the plugin's dialogs."""

from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QApplication


def fit_to_screen(widget, width: int, height: int):
    """Resize *widget* to its preferred size, capped to the screen it opens on.

    Dialogs whose preferred size was chosen on a large monitor otherwise open
    taller than a laptop screen, with their buttons below the desktop edge.
    Widgets that must survive the cap need their content in a QScrollArea —
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
