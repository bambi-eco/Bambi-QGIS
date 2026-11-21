# -*- coding: utf-8 -*-
"""A scale over the 3D view: how long a stretch of the terrain is.

A bar at the bottom of the viewport with a round length written over it
("50 m"). The terrain model is drawn in its own coordinates, which are
metres, so a length on the screen is a length on the ground - at one
distance from the eye. A view in perspective has no one scale: what is
nearer is drawn larger. The bar holds where the middle of the view meets
the terrain, and is drawn anew as the view zooms, with the next round
length when it would grow too wide or too narrow.

It lies over the OpenGL widget as a widget of its own, as the compass
does, and lets every mouse event through to the view below.
"""

from ..core.view_camera import scale_length, scale_text
from .qt import QtCore, QtGui, QtWidgets

_Qt = QtCore.Qt
#: The widest the bar gets, and the room around it.
WIDEST = 160
PADDING = 10
HEIGHT = 34


class ScaleBar(QtWidgets.QWidget):
    """The bar. :meth:`set_scale` says how much a pixel covers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._per_pixel = 0.0
        self.setFixedSize(WIDEST + 2 * PADDING, HEIGHT)
        self.setAttribute(_Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(_Qt.WidgetAttribute.WA_NoSystemBackground)

    def set_scale(self, per_pixel: float) -> None:
        """*per_pixel*: metres a pixel of the view covers; 0 for none."""
        per_pixel = max(0.0, float(per_pixel))
        if abs(per_pixel - self._per_pixel) > 1e-4 * max(per_pixel, 1e-9):
            self._per_pixel = per_pixel
            self.update()

    def length(self) -> float:
        """The metres the bar stands for; 0 where there is no scale."""
        return scale_length(self._per_pixel, WIDEST)

    def pixels(self) -> float:
        """How wide the bar is drawn."""
        return self.length() / self._per_pixel if self._per_pixel else 0.0

    def text(self) -> str:
        """What is written over the bar: ``50 m``."""
        return scale_text(self.length()) if self.length() else ""

    def paintEvent(self, _event):
        wide = self.pixels()
        if wide <= 0.0:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        left = (self.width() - wide) / 2.0
        painter.setPen(_Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(20, 24, 30, 150))
        painter.drawRoundedRect(QtCore.QRectF(
            left - PADDING + 1.0, 0.0, wide + 2.0 * PADDING - 2.0, HEIGHT),
            4.0, 4.0)

        # the bar: two halves, light and dark, between end marks
        base = HEIGHT - 9.0
        half = wide / 2.0
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1.0))
        painter.setBrush(QtGui.QColor(255, 255, 255, 230))
        painter.drawRect(QtCore.QRectF(left, base - 4.0, half, 4.0))
        painter.setBrush(QtGui.QColor(40, 44, 50, 230))
        painter.drawRect(QtCore.QRectF(left + half, base - 4.0, half, 4.0))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1.5))
        for x in (left, left + wide):
            painter.drawLine(QtCore.QPointF(x, base - 8.0),
                             QtCore.QPointF(x, base + 2.0))

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(QtCore.QRectF(0.0, 1.0, self.width(), 14.0),
                         _Qt.AlignmentFlag.AlignCenter, self.text())
