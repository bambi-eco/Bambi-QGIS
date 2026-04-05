# -*- coding: utf-8 -*-
"""
BAMBI Feature Viewer
====================

Non-modal dialog that displays a drone frame image with detection bounding boxes.
Green border = clicked/highlighted detection; blue border = all other detections.
For tracks, forward/backward navigation through every frame of the track is provided.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QWidget
)
from qgis.PyQt.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from qgis.PyQt.QtCore import Qt


class FeatureViewerDialog(QDialog):
    """
    Singleton non-modal dialog showing a frame image with bounding boxes.

    Usage
    -----
    Detection (single frame)::

        viewer = FeatureViewerDialog.get_instance(parent)
        viewer.show_detection(title, image_path, green_boxes, blue_boxes)

    Track (multiple navigable frames)::

        viewer = FeatureViewerDialog.get_instance(parent)
        viewer.show_track(title, frames_list, start_idx)

    Box format: (x1, y1, x2, y2) or (x1, y1, x2, y2, confidence, class_id)
    in pixel coordinates of the source frame image.
    """

    _instance = None

    @classmethod
    def get_instance(cls, parent=None):
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls(parent)
        return cls._instance

    # ------------------------------------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BAMBI Feature Viewer")
        self.setWindowFlags(
            Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
        )
        # Keep the Python object alive even when the user closes the window
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(800, 620)

        self._frames = []       # list of frame-data dicts
        self._current_idx = 0   # index into self._frames

        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title / description
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        bold = QFont()
        bold.setBold(True)
        self.title_label.setFont(bold)
        layout.addWidget(self.title_label)

        # Image area
        self.image_label = QLabel("No image loaded.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #aaa;")
        layout.addWidget(self.image_label)

        # Navigation row (hidden for single-frame detections)
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev Frame")
        self.next_btn = QPushButton("Next Frame >")
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.prev_btn.clicked.connect(self._go_prev)
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.frame_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)
        self.nav_widget = QWidget()
        self.nav_widget.setLayout(nav_layout)
        layout.addWidget(self.nav_widget)

        # Info row (confidence, class, …)
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_detection(self, title, image_path, green_boxes, blue_boxes):
        """Show a single detection frame.

        :param title: String shown in the title label.
        :param image_path: Absolute path to the frame image file.
        :param green_boxes: List of (x1,y1,x2,y2[,conf,cls]) tuples — highlighted.
        :param blue_boxes:  List of (x1,y1,x2,y2[,conf,cls]) tuples — background.
        """
        self._frames = [{
            "frame_idx":  None,
            "image_path": image_path,
            "boxes_green": list(green_boxes),
            "boxes_blue":  list(blue_boxes),
        }]
        self._current_idx = 0
        self.title_label.setText(title)
        self.nav_widget.setVisible(False)
        self._render_current_frame()
        self._show_and_raise()

    def show_track(self, title, frames, start_idx=0):
        """Show a track with navigable frames.

        :param title: String shown in the title label.
        :param frames: List of dicts, each with keys:
                       ``frame_idx``   (int)
                       ``image_path``  (str)
                       ``boxes_green`` (list of box tuples)
                       ``boxes_blue``  (list of box tuples)
        :param start_idx: Index into *frames* to display first.
        """
        self._frames = list(frames)
        self._current_idx = max(0, min(start_idx, len(frames) - 1))
        self.title_label.setText(title)
        self.nav_widget.setVisible(len(frames) > 1)
        self._render_current_frame()
        self._show_and_raise()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_prev(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._render_current_frame()

    def _go_next(self):
        if self._current_idx < len(self._frames) - 1:
            self._current_idx += 1
            self._render_current_frame()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_frame(self):
        if not self._frames:
            return

        data = self._frames[self._current_idx]
        image_path  = data.get("image_path", "")
        boxes_green = data.get("boxes_green", [])
        boxes_blue  = data.get("boxes_blue", [])
        frame_idx   = data.get("frame_idx")
        total       = len(self._frames)

        # Navigation label + button states
        if total > 1:
            self.frame_label.setText(
                f"Frame {frame_idx}   ({self._current_idx + 1} / {total})"
            )
        else:
            self.frame_label.setText(
                f"Frame {frame_idx}" if frame_idx is not None else ""
            )
        self.prev_btn.setEnabled(self._current_idx > 0)
        self.next_btn.setEnabled(self._current_idx < total - 1)

        # Load image
        if not image_path:
            self.image_label.setText("Image path not available.")
            self.image_label.setPixmap(QPixmap())
            return

        img = QImage(image_path)
        if img.isNull():
            self.image_label.setText(f"Could not load image:\n{image_path}")
            self.image_label.setPixmap(QPixmap())
            return

        # Draw bounding boxes and scale to widget
        annotated = self._draw_boxes(img, boxes_green, boxes_blue)
        scaled = annotated.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

        # Info text from first green box
        info_parts = []
        if frame_idx is not None:
            info_parts.append(f"Frame: {frame_idx}")
        if boxes_green:
            b = boxes_green[0]
            if len(b) >= 6:
                info_parts.append(f"Conf: {float(b[4]):.3f}")
                info_parts.append(f"Class: {int(b[5])}")
        self.info_label.setText("   |   ".join(info_parts))

    def _draw_boxes(self, img, green_boxes, blue_boxes):
        """Paint bounding boxes onto a copy of *img* and return the result."""
        result = img.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, False)

        lw_blue  = max(2, img.width() // 400)
        lw_green = max(3, img.width() // 280)

        pen_blue  = QPen(QColor(80, 140, 255), lw_blue)
        pen_green = QPen(QColor(0, 220, 0),    lw_green)

        font = QFont("Arial", max(8, img.width() // 80))
        painter.setFont(font)

        # Draw blue first so green is always on top
        for boxes, pen in [(blue_boxes, pen_blue), (green_boxes, pen_green)]:
            painter.setPen(pen)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                if len(box) >= 6:
                    label = f"cls:{int(box[5])} {float(box[4]):.2f}"
                    painter.drawText(x1 + 2, max(y1 - 4, 12), label)

        painter.end()
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _show_and_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def resizeEvent(self, event):
        """Re-render on dialog resize so the image fills the new size."""
        super().resizeEvent(event)
        self._render_current_frame()
