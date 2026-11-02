# -*- coding: utf-8 -*-
"""Camera-calibration annotation label: magnifier and click must agree.

The loupe used to map the cursor over the whole widget (letterbox bars
included) while clicks mapped over the image only, so on any image that did
not fill the widget exactly the point landed somewhere other than where the
loupe pointed. A typo in the same method (``_magnifier_enabledm``) also made
every repaint raise.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import types

import numpy as np
import pytest

from qgis.PyQt.QtCore import QPoint, Qt
from qgis.PyQt.QtTest import QTest

from bambi_wildlife_detection import bambi_camera_calibration as calib
from bambi_wildlife_detection.bambi_camera_calibration import _AnnotImageLabel

W = H = 200          # image size; widget is wider -> horizontal letterbox
OX = (600 - W) // 2  # left bar width at scale 1.0


@pytest.fixture(autouse=True)
def _cv2_stub(monkeypatch):
    """The QGIS image has no OpenCV; the label only needs BGR->RGB."""
    if getattr(calib, "cv2", None) is None:
        stub = types.SimpleNamespace(
            COLOR_BGR2RGB=4,
            cvtColor=lambda img, _code: np.ascontiguousarray(img[..., ::-1]))
        monkeypatch.setattr(calib, "cv2", stub, raising=False)


def _gradient_label():
    """Label showing an image whose pixel (x, y) has red=x, green=y."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 2] = np.arange(W, dtype=np.uint8)[None, :]   # R (BGR order)
    img[:, :, 1] = np.arange(H, dtype=np.uint8)[:, None]   # G
    label = _AnnotImageLabel()
    label.resize(600, 200)
    label.load_image(img)
    return label


def test_loupe_centre_matches_click():
    label = _gradient_label()
    label.set_magnifier_enabled(True)
    label.set_accepting(True)

    # Off-centre on purpose: at the widget centre the old bug was invisible.
    img_x, img_y = 50, 120
    cursor = QPoint(OX + img_x, img_y)
    label._mag_pos = cursor

    shot = label.grab().toImage()          # runs paintEvent -> _draw_magnifier
    # Sample just outside the crosshair: +15 screen px = +5 image px at 3x.
    c = shot.pixelColor(cursor.x() + 15, cursor.y() + 15)
    assert abs(c.red() - (img_x + 5)) <= 2
    assert abs(c.green() - (img_y + 5)) <= 2

    placed = []
    label.pointPlaced.connect(lambda x, y: placed.append((x, y)))
    QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=cursor)
    assert placed == [(img_x, img_y)]
    label.deleteLater()


def test_click_in_letterbox_is_ignored():
    label = _gradient_label()
    label.set_accepting(True)
    placed = []
    label.pointPlaced.connect(lambda x, y: placed.append((x, y)))
    QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=QPoint(OX // 2, 100))
    assert placed == []
    label.deleteLater()
