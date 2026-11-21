# -*- coding: utf-8 -*-
"""A compass over the 3D view: which way the view looks.

A small rose in a corner of the viewport. The top of the rose is the
direction the view looks in; the rose turns under it, so that N is where
north is on the screen, and the bearing is written below ("NE 47°").
North is the map's north - the +y of the project's CRS, in which the
terrain and the flight are drawn.

It lies over the OpenGL widget as a widget of its own, painted with
QPainter, and lets every mouse event through to the view below.
"""

import math

from ..core.view_camera import compass_point
from .qt import QtCore, QtGui, QtWidgets

_Qt = QtCore.Qt
SIZE = 92
MARGIN = 10


class Compass(QtWidgets.QWidget):
    """The rose. :meth:`set_bearing` turns it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bearing = 0.0
        self.setFixedSize(SIZE, SIZE + 16)
        self.setAttribute(_Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(_Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setToolTip("")

    def bearing(self) -> float:
        return self._bearing

    def text(self) -> str:
        """What is written under the rose: ``NE 47°``."""
        return f"{compass_point(self._bearing)} {self._bearing:.0f}°"

    def set_bearing(self, bearing: float) -> None:
        bearing = float(bearing) % 360.0
        if abs(bearing - self._bearing) > 0.05:
            self._bearing = bearing
            self.update()

    def north_on_screen(self):
        """``(x, y)`` of the unit vector that points north on the screen
        (y down): straight up when the view looks north."""
        angle = math.radians(-self._bearing)
        return math.sin(angle), -math.cos(angle)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        centre = QtCore.QPointF(SIZE / 2.0, SIZE / 2.0)
        radius = SIZE / 2.0 - 4.0

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1.5))
        painter.setBrush(QtGui.QColor(20, 24, 30, 150))
        painter.drawEllipse(centre, radius, radius)

        # the direction looked in: a mark at the top that does not turn
        painter.setPen(_Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 230))
        painter.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(centre.x(), 1.0),
            QtCore.QPointF(centre.x() - 5.0, 9.0),
            QtCore.QPointF(centre.x() + 5.0, 9.0)]))

        painter.save()
        painter.translate(centre)
        painter.rotate(-self._bearing)
        # the needle: red to the north, pale to the south
        for colour, sign in ((QtGui.QColor(225, 60, 50), -1.0),
                             (QtGui.QColor(225, 225, 225), 1.0)):
            painter.setBrush(colour)
            painter.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(0.0, sign * (radius - 20.0)),
                QtCore.QPointF(-5.0, 0.0), QtCore.QPointF(5.0, 0.0)]))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 170), 1.0))
        for tick in range(8):
            painter.save()
            painter.rotate(45.0 * tick)
            length = 5.0 if tick % 2 == 0 else 3.0
            painter.drawLine(QtCore.QPointF(0.0, -radius + 1.0),
                             QtCore.QPointF(0.0, -radius + 1.0 + length))
            painter.restore()
        painter.restore()

        # the letters stay upright, each where its direction is
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        for letter, degrees in (("N", 0.0), ("E", 90.0), ("S", 180.0),
                                ("W", 270.0)):
            angle = math.radians(degrees - self._bearing)
            x = centre.x() + math.sin(angle) * (radius - 10.0)
            y = centre.y() - math.cos(angle) * (radius - 10.0)
            painter.setPen(QtGui.QColor(255, 120, 110) if letter == "N"
                           else QtGui.QColor(255, 255, 255))
            painter.drawText(QtCore.QRectF(x - 8.0, y - 8.0, 16.0, 16.0),
                             _Qt.AlignmentFlag.AlignCenter, letter)

        font.setBold(False)
        painter.setFont(font)
        label = QtCore.QRectF(0.0, SIZE - 2.0, SIZE, 16.0)
        painter.setPen(_Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(20, 24, 30, 150))
        painter.drawRoundedRect(label.adjusted(14.0, 0.0, -14.0, 0.0),
                                4.0, 4.0)
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(label, _Qt.AlignmentFlag.AlignCenter, self.text())
