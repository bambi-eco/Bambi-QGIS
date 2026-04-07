# -*- coding: utf-8 -*-
"""
BAMBI Feature Viewer
====================

Non-modal dialog that displays a drone frame image with detection bounding boxes.
Green border = clicked/highlighted detection; blue border = all other detections.
For tracks, forward/backward navigation through every frame of the track is provided.

When both thermal and RGB frames have been extracted a toggle button lets the
user switch between the two views without losing the current frame position.

When viewing the modality that was *not* used for detection the user can press
"Project bounding boxes" to re-project the geo-referenced world-space boxes
back into that modality's pixel space via camera projection math.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QWidget, QProgressBar
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
        viewer.show_detection(title, green_boxes, blue_boxes,
                              image_path_t=..., image_path_w=...,
                              boxes_modality="t",
                              target_folder=..., dem_path=...,
                              correction_path=...)

    Track (multiple navigable frames)::

        viewer = FeatureViewerDialog.get_instance(parent)
        viewer.show_track(title, frames_list, start_idx,
                          target_folder=..., dem_path=...,
                          correction_path=...)

    Box format: (x1, y1, x2, y2) or (x1, y1, x2, y2, confidence, class_id)
    in pixel coordinates of the source frame image.

    Frame dict keys
    ---------------
    ``frame_idx``       : int or None
    ``image_path_t``    : str — path to thermal frame (empty if not extracted)
    ``image_path_w``    : str — path to RGB frame (empty if not extracted)
    ``boxes_modality``  : str — "t" or "w", pixel space of the boxes
    ``boxes_green``     : list of box tuples (highlighted detection)
    ``boxes_blue``      : list of box tuples (other detections)
    ``boxes_green_proj``: list of box tuples projected to the other modality (optional)
    ``boxes_blue_proj`` : list of box tuples projected to the other modality (optional)
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
        self.resize(800, 660)

        self._frames = []       # list of frame-data dicts
        self._current_idx = 0   # index into self._frames
        self._view_mode = "t"   # "t" = thermal, "w" = RGB

        # Projection context
        self._target_folder   = ""
        self._dem_path        = ""
        self._correction_path = ""
        self._projection_worker = None

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

        # View toggle row (hidden when only one modality is available)
        toggle_layout = QHBoxLayout()
        self.view_toggle_btn = QPushButton("Switch to RGB")
        self.view_toggle_btn.clicked.connect(self._toggle_view_mode)
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.view_toggle_btn)
        toggle_layout.addStretch()
        self.toggle_widget = QWidget()
        self.toggle_widget.setLayout(toggle_layout)
        self.toggle_widget.setVisible(False)
        layout.addWidget(self.toggle_widget)

        # Box projection row (hidden until user is on the non-source modality)
        proj_layout = QHBoxLayout()
        self.proj_btn = QPushButton("Project bounding boxes")
        self.proj_btn.setToolTip(
            "Re-project geo-referenced bounding boxes into this modality's "
            "pixel space via camera projection."
        )
        self.proj_btn.clicked.connect(self._start_box_projection)
        self.proj_progress = QProgressBar()
        self.proj_progress.setRange(0, 100)
        self.proj_progress.setFixedHeight(16)
        self.proj_progress.setVisible(False)
        proj_layout.addStretch()
        proj_layout.addWidget(self.proj_btn)
        proj_layout.addWidget(self.proj_progress)
        proj_layout.addStretch()
        self.proj_widget = QWidget()
        self.proj_widget.setLayout(proj_layout)
        self.proj_widget.setVisible(False)
        layout.addWidget(self.proj_widget)

        # Projection quality notice (shown alongside the projection button)
        self.proj_info_label = QLabel(
            "Projection quality depends on camera calibration accuracy and correction "
            "factors. If either is imprecise, projected boxes will be misaligned."
        )
        self.proj_info_label.setAlignment(Qt.AlignCenter)
        self.proj_info_label.setWordWrap(True)
        self.proj_info_label.setStyleSheet(
            "color: #888; font-style: italic; background: #2a2a2a; "
            "border-radius: 4px; padding: 4px 8px;"
        )
        self.proj_info_label.setVisible(False)
        layout.addWidget(self.proj_info_label)

        # Info row (confidence, class, …)
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_detection(self, title, green_boxes, blue_boxes,
                       image_path_t="", image_path_w="", boxes_modality="t",
                       target_folder="", dem_path="", correction_path="",
                       frame_idx=None):
        """Show a single detection frame.

        :param title: String shown in the title label.
        :param green_boxes: List of (x1,y1,x2,y2[,conf,cls]) tuples — highlighted.
        :param blue_boxes:  List of (x1,y1,x2,y2[,conf,cls]) tuples — background.
        :param image_path_t: Absolute path to the thermal frame image (may be empty).
        :param image_path_w: Absolute path to the RGB frame image (may be empty).
        :param boxes_modality: ``"t"`` if boxes are in thermal pixel space,
                               ``"w"`` if in RGB pixel space.
        :param target_folder: Root output folder for geo-referenced data.
        :param dem_path: Path to the DEM GLTF/GLB (needed for box projection).
        :param correction_path: Explicit correction.json path (may be empty).
        :param frame_idx: Integer index of the frame in the poses file (needed for
                          box projection); None when unknown.
        """
        self._stop_projection_worker()
        self._target_folder   = target_folder
        self._dem_path        = dem_path
        self._correction_path = correction_path

        self._frames = [{
            "frame_idx":      frame_idx,
            "image_path_t":   image_path_t,
            "image_path_w":   image_path_w,
            "boxes_modality": boxes_modality,
            "boxes_green":    list(green_boxes),
            "boxes_blue":     list(blue_boxes),
        }]
        self._current_idx = 0
        self.title_label.setText(title)
        self.nav_widget.setVisible(False)
        self._reset_view_mode()
        self._update_toggle_btn()
        self._update_proj_btn()
        self._render_current_frame()
        self._show_and_raise()

    def show_track(self, title, frames, start_idx=0,
                   target_folder="", dem_path="", correction_path=""):
        """Show a track with navigable frames.

        :param title: String shown in the title label.
        :param frames: List of dicts, each with keys:
                       ``frame_idx``    (int)
                       ``image_path_t`` (str) — thermal frame path
                       ``image_path_w`` (str) — RGB frame path
                       ``boxes_modality`` (str) — "t" or "w"
                       ``boxes_green``  (list of box tuples)
                       ``boxes_blue``   (list of box tuples)
        :param start_idx: Index into *frames* to display first.
        :param target_folder: Root output folder for geo-referenced data.
        :param dem_path: Path to the DEM GLTF/GLB (needed for box projection).
        :param correction_path: Explicit correction.json path (may be empty).
        """
        self._stop_projection_worker()
        self._target_folder   = target_folder
        self._dem_path        = dem_path
        self._correction_path = correction_path

        self._frames = list(frames)
        self._current_idx = max(0, min(start_idx, len(frames) - 1))
        self.title_label.setText(title)
        self.nav_widget.setVisible(len(frames) > 1)
        self._reset_view_mode()
        self._update_toggle_btn()
        self._update_proj_btn()
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
    # View mode toggle
    # ------------------------------------------------------------------

    def _reset_view_mode(self):
        """Set initial view mode: thermal if available, otherwise RGB."""
        if not self._frames:
            return
        has_any_thermal = any(f.get("image_path_t") for f in self._frames)
        self._view_mode = "t" if has_any_thermal else "w"
        self._update_toggle_btn_text()

    def _toggle_view_mode(self):
        self._view_mode = "w" if self._view_mode == "t" else "t"
        self._update_toggle_btn_text()
        self._update_proj_btn()
        self._render_current_frame()

    def _update_toggle_btn_text(self):
        if self._view_mode == "t":
            self.view_toggle_btn.setText("Switch to RGB")
        else:
            self.view_toggle_btn.setText("Switch to Thermal")

    def _update_toggle_btn(self):
        """Show the toggle button only when both modalities are available."""
        has_thermal = any(f.get("image_path_t") for f in self._frames)
        has_rgb     = any(f.get("image_path_w") for f in self._frames)
        self.toggle_widget.setVisible(has_thermal and has_rgb)

    # ------------------------------------------------------------------
    # Box projection
    # ------------------------------------------------------------------

    def _is_on_non_source_modality(self) -> bool:
        """True when the displayed modality differs from the detection source."""
        if not self._frames:
            return False
        boxes_modality = self._frames[0].get("boxes_modality", "t")
        return self._view_mode != boxes_modality

    def _projection_done(self) -> bool:
        """True when at least one frame already has projected boxes."""
        return any(
            f.get("boxes_green_proj") is not None
            for f in self._frames
        )

    def _update_proj_btn(self):
        """Show the projection button when on the non-source modality and
        projection has not yet been computed (or is in progress)."""
        if not self.toggle_widget.isVisible():
            self.proj_widget.setVisible(False)
            return

        # Projection requires a single shared target_folder across all frames.
        # When frames come from different layers (e.g. overlapping FoVs from
        # multiple extractions), the projection worker cannot handle them all
        # with one target_folder, so we hide the button.
        frame_folders = {
            f.get("target_folder") for f in self._frames if "target_folder" in f
        }
        if len(frame_folders) > 1:
            self.proj_widget.setVisible(False)
            self.proj_info_label.setVisible(False)
            self.proj_progress.setVisible(False)
            return

        visible = self._is_on_non_source_modality() and not self._projection_done()
        self.proj_widget.setVisible(visible)
        self.proj_info_label.setVisible(visible)
        if not visible:
            self.proj_progress.setVisible(False)

    def _start_box_projection(self):
        """Launch the background projection worker."""
        from .bambi_box_projector import BoxProjectionWorker

        if not self._target_folder:
            return

        # Determine the source modality from the first frame
        src_modality = self._frames[0].get("boxes_modality", "t") if self._frames else "t"

        self._stop_projection_worker()

        self.proj_btn.setEnabled(False)
        self.proj_progress.setValue(0)
        self.proj_progress.setVisible(True)

        self._projection_worker = BoxProjectionWorker(
            target_folder   = self._target_folder,
            dem_path        = self._dem_path,
            correction_path = self._correction_path,
            src_modality    = src_modality,
            frames          = list(self._frames),
        )
        self._projection_worker.progress.connect(self._on_proj_progress)
        self._projection_worker.finished.connect(self._on_proj_finished)
        self._projection_worker.error.connect(self._on_proj_error)
        self._projection_worker.start()

    def _stop_projection_worker(self):
        if self._projection_worker is not None and self._projection_worker.isRunning():
            self._projection_worker.quit()
            self._projection_worker.wait(3000)
        self._projection_worker = None

    def _on_proj_progress(self, value: int):
        self.proj_progress.setValue(value)

    def _on_proj_finished(self, results: dict):
        """Store projected boxes in frames and re-render."""
        for i, frame in enumerate(self._frames):
            if i in results:
                frame["boxes_green_proj"] = results[i].get("green", [])
                frame["boxes_blue_proj"]  = results[i].get("blue",  [])

        self.proj_btn.setEnabled(True)
        self.proj_progress.setVisible(False)
        self._update_proj_btn()
        self._render_current_frame()

    def _on_proj_error(self, msg: str):
        from qgis.PyQt.QtWidgets import QMessageBox
        self.proj_btn.setEnabled(True)
        self.proj_progress.setVisible(False)
        QMessageBox.warning(self, "Box Projection Failed", msg)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_frame(self):
        if not self._frames:
            return

        data = self._frames[self._current_idx]

        # Per-frame projection context: FoV navigation stores target_folder etc.
        # in each frame dict so that different overlapping FoVs (potentially from
        # different layers) carry their own metadata.
        if "target_folder" in data:
            self._target_folder   = data["target_folder"]
            self._dem_path        = data.get("dem_path", "")
            self._correction_path = data.get("correction_path", "")

        frame_idx   = data.get("frame_idx")
        total       = len(self._frames)

        # Pick image path according to current view mode with fallback
        if self._view_mode == "t":
            image_path = data.get("image_path_t") or data.get("image_path_w", "")
        else:
            image_path = data.get("image_path_w") or data.get("image_path_t", "")

        # Select boxes: native boxes on matching modality, projected otherwise
        boxes_modality = data.get("boxes_modality", "t")
        if self._view_mode == boxes_modality:
            boxes_green = data.get("boxes_green", [])
            boxes_blue  = data.get("boxes_blue",  [])
            projected   = False
        else:
            # Use projected boxes if available, otherwise show none
            boxes_green = data.get("boxes_green_proj", [])
            boxes_blue  = data.get("boxes_blue_proj",  [])
            projected   = True

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
        annotated = self._draw_boxes(img, boxes_green, boxes_blue, projected=projected)
        scaled = annotated.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

        # Info text from first green box.
        # FoV frames (identified by per-frame target_folder) only show the
        # frame index and detection count — confidence/class are not meaningful.
        is_fov_frame = "target_folder" in data
        info_parts = []
        if frame_idx is not None:
            info_parts.append(f"Frame: {frame_idx}")
        if is_fov_frame:
            info_parts.append(f"{len(boxes_green)} detection(s)")
        elif boxes_green:
            b = boxes_green[0]
            if len(b) >= 6:
                info_parts.append(f"Conf: {float(b[4]):.3f}")
                info_parts.append(f"Class: {int(b[5])}")
            if len(b) >= 7 and b[6]:
                info_parts.append("interpolated")
        self.info_label.setText("   |   ".join(info_parts))

    def _draw_boxes(self, img, green_boxes, blue_boxes, projected=False):
        """Paint bounding boxes onto a copy of *img* and return the result.

        Box tuple format: (x1, y1, x2, y2[, conf, cls[, is_interpolated]])
        Interpolated boxes (is_interpolated=1) are drawn with a dashed line.
        Native boxes use green/blue; projected boxes use red/orange.
        """
        result = img.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, False)

        lw_secondary = max(2, img.width() // 400)
        lw_primary   = max(3, img.width() // 280)

        def make_pen(color, lw, dashed):
            pen = QPen(color, lw, Qt.DashLine if dashed else Qt.SolidLine)
            return pen

        font = QFont("Arial", max(8, img.width() // 80))
        painter.setFont(font)

        if projected:
            color_primary   = QColor(220, 40,  40)   # red   — highlighted
            color_secondary = QColor(255, 160, 0)    # orange — others
        else:
            color_primary   = QColor(0,   220, 0)    # green — highlighted
            color_secondary = QColor(80,  140, 255)  # blue  — others

        # Draw secondary first so primary is always on top
        for boxes, color, lw in [
            (blue_boxes,  color_secondary, lw_secondary),
            (green_boxes, color_primary,   lw_primary),
        ]:
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                is_interp = len(box) >= 7 and bool(box[6])

                painter.setPen(make_pen(color, lw, is_interp))
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                if len(box) >= 6:
                    label = f"cls:{int(box[5])} {float(box[4]):.2f}"
                    if is_interp:
                        label += " (interp)"
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

    def closeEvent(self, event):
        self._stop_projection_worker()
        super().closeEvent(event)
