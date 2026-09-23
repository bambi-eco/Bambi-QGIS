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

The viewer is also where a result is reviewed. Below the image a details
panel lists what the project knows about the track being shown - the same
facts as a row of the track inventory: species, sex and age with their
votes, box counts, times, positions, the match on the other camera and the
annotator's verdict - with the inventory's "Approved" checkmark beside it.
And a wrong result can be removed from the project here: the box on the
current frame, or the whole track, which takes it out of every store and
off the map (see ``core.review`` and ``bambi_layer_sync``).
"""

import html
from typing import List, Optional, Tuple

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QSizePolicy, QWidget, QProgressBar, QMessageBox, QApplication,
)
from qgis.PyQt.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from qgis.PyQt.QtCore import Qt, pyqtSignal


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
                              correction_path=...,
                              detection_ids=[...], track_id=...)

    Track (multiple navigable frames)::

        viewer = FeatureViewerDialog.get_instance(parent)
        viewer.show_track(title, frames_list, start_idx,
                          target_folder=..., dem_path=...,
                          correction_path=..., track_id=..., modality="t")

    Box format: (x1, y1, x2, y2) or (x1, y1, x2, y2, confidence, class_id)
    in pixel coordinates of the source frame image.

    Frame dict keys
    ---------------
    ``frame_idx``          : int or None
    ``image_path_t``       : str - path to thermal frame (empty if not extracted)
    ``image_path_w``       : str - path to RGB frame (empty if not extracted)
    ``boxes_modality``     : str - "t" or "w", pixel space of the boxes
    ``boxes_green``        : list of box tuples (highlighted detection)
    ``boxes_blue``         : list of box tuples (other detections)
    ``boxes_green_proj``   : list of box tuples projected to the other modality (optional)
    ``boxes_blue_proj``    : list of box tuples projected to the other modality (optional)
    ``detection_ids_green``: store ids behind the green boxes (optional; what
                             "Delete detection" removes)

    Signals
    -------
    ``trackDeleted(target_folder, modality, track_id)``,
    ``detectionDeleted(target_folder, modality, detection_id, track_id)``
    (``track_id`` is -1 for a detection outside any track) and
    ``trackApproved(target_folder, modality, track_id, approved)`` tell the
    dock widget what the reviewer did, so an open inventory report follows.
    """

    _instance = None

    trackDeleted = pyqtSignal(str, str, int)
    detectionDeleted = pyqtSignal(str, str, int, int)
    trackApproved = pyqtSignal(str, str, int, bool)

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
            Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowMinimizeButtonHint
        )
        # Keep the Python object alive even when the user closes the window
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(800, 760)

        self._frames = []       # list of frame-data dicts
        self._current_idx = 0   # index into self._frames
        self._view_mode = "t"   # "t" = thermal, "w" = RGB

        # Projection context
        self._target_folder = ""
        self._dem_path = ""
        self._correction_path = ""
        self._projection_worker = None

        # Review context: which store the boxes come from, and which track
        # (if any) the view is about.
        self._modality: Optional[str] = None
        self._track_id: Optional[int] = None
        self._epsg: Optional[int] = None
        self._details_row: Optional[dict] = None
        self._details_open = True

        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title / description
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bold = QFont()
        bold.setBold(True)
        self.title_label.setFont(bold)
        layout.addWidget(self.title_label)

        # Image area
        self.image_label = QLabel("No image loaded.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #aaa;")
        layout.addWidget(self.image_label, 1)

        # Navigation row (hidden for single-frame detections)
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev Frame")
        self.next_btn = QPushButton("Next Frame >")
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self.proj_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.proj_info_label.setWordWrap(True)
        self.proj_info_label.setStyleSheet(
            "color: #888; font-style: italic; background: #2a2a2a; "
            "border-radius: 4px; padding: 4px 8px;"
        )
        self.proj_info_label.setVisible(False)
        layout.addWidget(self.proj_info_label)

        # Info row (confidence, class, …)
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        # Track details: what the inventory knows about this animal.
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.details_toggle_btn = QPushButton("Track details")
        self.details_toggle_btn.setFlat(True)
        self.details_toggle_btn.setToolTip(
            "Show or hide what the project knows about this track - the "
            "same facts as its row in the track inventory.")
        self.details_toggle_btn.clicked.connect(self._toggle_details)
        header.addWidget(self.details_toggle_btn)
        header.addStretch()
        self.approved_check = QCheckBox("Approved")
        self.approved_check.setToolTip(
            "Your own checkmark: this track has been checked and is correct. "
            "Written to the track, so it shows up in the inventory and the "
            "exports.")
        self.approved_check.toggled.connect(self._on_approved_toggled)
        header.addWidget(self.approved_check)
        details_layout.addLayout(header)
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.TextFormat.RichText)
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details_label.setStyleSheet(
            "background: #2a2a2a; color: #ddd; border-radius: 4px; "
            "padding: 4px 8px;")
        details_layout.addWidget(self.details_label)
        self.details_widget = QWidget()
        self.details_widget.setLayout(details_layout)
        self.details_widget.setVisible(False)
        layout.addWidget(self.details_widget)

        # Review row: remove a wrong result from the project.
        review_layout = QHBoxLayout()
        self.delete_detection_btn = QPushButton("Delete detection")
        self.delete_detection_btn.setToolTip(
            "Remove the highlighted box on this frame from the project: the "
            "detection, its ground position, its place in the track, its "
            "classifier results and its match. The map layers follow. "
            "This cannot be undone.")
        self.delete_detection_btn.clicked.connect(self._delete_current_detection)
        self.delete_track_btn = QPushButton("Delete track")
        self.delete_track_btn.setToolTip(
            "Remove this track with every one of its detections from the "
            "project and from the map. This cannot be undone.")
        self.delete_track_btn.clicked.connect(self._delete_current_track)
        review_layout.addStretch()
        review_layout.addWidget(self.delete_detection_btn)
        review_layout.addWidget(self.delete_track_btn)
        review_layout.addStretch()
        self.review_widget = QWidget()
        self.review_widget.setLayout(review_layout)
        self.review_widget.setVisible(False)
        layout.addWidget(self.review_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_detection(self, title, green_boxes, blue_boxes,
                       image_path_t="", image_path_w="", boxes_modality="t",
                       target_folder="", dem_path="", correction_path="",
                       frame_idx=None, frame_idx_t=None, frame_idx_w=None,
                       other_green=(), other_blue=(), detection_ids=(),
                       track_id=None, epsg=None):
        """Show a single detection frame.

        :param title: String shown in the title label.
        :param green_boxes: List of (x1,y1,x2,y2[,conf,cls]) tuples - highlighted.
        :param blue_boxes:  List of (x1,y1,x2,y2[,conf,cls]) tuples - background.
        :param image_path_t: Absolute path to the thermal frame image (may be empty).
        :param image_path_w: Absolute path to the RGB frame image (may be empty).
        :param boxes_modality: ``"t"`` if boxes are in thermal pixel space,
                               ``"w"`` if in RGB pixel space.
        :param target_folder: Root output folder for geo-referenced data.
        :param dem_path: Path to the DEM GLTF/GLB (needed for box projection).
        :param correction_path: Explicit correction.json path (may be empty).
        :param frame_idx: Integer index of the frame in the poses file (needed for
                          box projection); None when unknown.
        :param frame_idx_t: Thermal frame index of this moment; ``frame_idx``
                            itself when the boxes are thermal, otherwise the
                            time-matched partner (None when unknown).
        :param frame_idx_w: The same for the RGB camera.
        :param other_green: Boxes the *other* camera detected on its frame of
                            this moment that are matched to the highlighted
                            detection, in that camera's pixel space.
        :param other_blue: The other camera's remaining boxes on that frame.
        :param detection_ids: Store ids of the green boxes, for deletion.
        :param track_id: The track the highlighted detection belongs to, if
                         any - its details are shown and it can be deleted.
        :param epsg: The project CRS, for the WGS84 positions in the details.
        """
        self._stop_projection_worker()
        self._target_folder = target_folder
        self._dem_path = dem_path
        self._correction_path = correction_path
        self._modality = boxes_modality if target_folder else None
        self._track_id = None if track_id is None else int(track_id)
        self._epsg = epsg

        if boxes_modality == "t" and frame_idx_t is None:
            frame_idx_t = frame_idx
        if boxes_modality == "w" and frame_idx_w is None:
            frame_idx_w = frame_idx
        self._frames = [{
            "frame_idx": frame_idx,
            "frame_idx_t": frame_idx_t,
            "frame_idx_w": frame_idx_w,
            "image_path_t": image_path_t,
            "image_path_w": image_path_w,
            "boxes_modality": boxes_modality,
            "boxes_green": list(green_boxes),
            "boxes_blue": list(blue_boxes),
            "boxes_green_other": list(other_green),
            "boxes_blue_other": list(other_blue),
            "detection_ids_green": [int(i) for i in detection_ids],
        }]
        self._current_idx = 0
        self.title_label.setText(title)
        self.nav_widget.setVisible(False)
        self._reset_view_mode()
        self._update_toggle_btn()
        self._update_proj_btn()
        self._render_current_frame()
        self._load_details()
        self._show_and_raise()

    def show_track(self, title, frames, start_idx=0,
                   target_folder="", dem_path="", correction_path="",
                   track_id=None, modality=None, epsg=None):
        """Show a track with navigable frames.

        :param title: String shown in the title label.
        :param frames: List of dicts, each with keys:
                       ``frame_idx``    (int)
                       ``image_path_t`` (str) - thermal frame path
                       ``image_path_w`` (str) - RGB frame path
                       ``boxes_modality`` (str) - "t" or "w"
                       ``boxes_green``  (list of box tuples)
                       ``boxes_blue``   (list of box tuples)
        :param start_idx: Index into *frames* to display first.
        :param target_folder: Root output folder for geo-referenced data.
        :param dem_path: Path to the DEM GLTF/GLB (needed for box projection).
        :param correction_path: Explicit correction.json path (may be empty).
        :param track_id: The store id of the track shown; ``None`` for a
                         frame sequence that is not one track (FoV views).
        :param modality: The store the boxes come from (``"t"`` / ``"w"``);
                         defaults to the frames' ``boxes_modality``.
        :param epsg: The project CRS, for the WGS84 positions in the details.
        """
        self._stop_projection_worker()
        self._target_folder = target_folder
        self._dem_path = dem_path
        self._correction_path = correction_path
        self._track_id = None if track_id is None else int(track_id)
        if modality is None and frames:
            modality = frames[0].get("boxes_modality")
        self._modality = modality if target_folder else None
        self._epsg = epsg

        self._frames = list(frames)
        self._current_idx = max(0, min(start_idx, len(frames) - 1))
        self.title_label.setText(title)
        self.nav_widget.setVisible(len(frames) > 1)
        self._reset_view_mode()
        self._update_toggle_btn()
        self._update_proj_btn()
        self._render_current_frame()
        self._load_details()
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
        has_rgb = any(f.get("image_path_w") for f in self._frames)
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
            target_folder=self._target_folder,
            dem_path=self._dem_path,
            correction_path=self._correction_path,
            src_modality=src_modality,
            frames=list(self._frames),
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
                frame["boxes_blue_proj"] = results[i].get("blue", [])

        self.proj_btn.setEnabled(True)
        self.proj_progress.setVisible(False)
        self._update_proj_btn()
        self._render_current_frame()

    def _on_proj_error(self, msg: str):
        self.proj_btn.setEnabled(True)
        self.proj_progress.setVisible(False)
        QMessageBox.warning(self, "Box Projection Failed", msg)

    # ------------------------------------------------------------------
    # Track details and approval
    # ------------------------------------------------------------------

    def _is_fov_view(self) -> bool:
        """FoV navigation stores its context per frame; there is no single
        track or store behind such a view."""
        return any("target_folder" in f for f in self._frames)

    def _load_details(self):
        """Fetch the inventory row of the shown track and render it."""
        self._details_row = None
        reviewable = bool(self._target_folder and self._modality) \
            and not self._is_fov_view()
        has_track = reviewable and self._track_id is not None
        self.details_widget.setVisible(has_track)
        self._update_review_buttons()
        if not has_track:
            return

        from .core import track_inventory

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rows = track_inventory.build_inventory(
                self._target_folder, self._modality, epsg=self._epsg,
                track_ids=[self._track_id])
            self._details_row = rows[0] if rows else None
        except Exception as exc:  # noqa: BLE001 - shown in the panel
            self.details_label.setText(
                f"<i>Track details unavailable: {html.escape(str(exc))}</i>")
            self._set_approved_silently(False, enabled=False)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._render_details()

    def _render_details(self):
        row = self._details_row
        if row is None:
            self.details_label.setText(
                f"<i>Track {self._track_id} is not in the store any more.</i>")
            self._set_approved_silently(False, enabled=False)
            self.details_label.setVisible(self._details_open)
            return

        from .core import track_inventory

        facts = [(label, text) for label, text in
                 track_inventory.describe_track(row) if label != "Approved"]
        self.details_label.setText(self._facts_html(facts))
        self._set_approved_silently(bool(row.get("approved")), enabled=True)
        self.details_label.setVisible(self._details_open)
        self.details_toggle_btn.setText(
            "Track details ▾" if self._details_open else "Track details ▸")

    @staticmethod
    def _facts_html(facts: List[Tuple[str, str]]) -> str:
        """Two facts per row, label in grey, value beside it."""
        cells = []
        for label, text in facts:
            cells.append(
                f"<td style='color:#999; padding-right:4px; white-space:nowrap'>"
                f"{html.escape(label)}</td>"
                f"<td style='padding-right:14px'>{html.escape(text)}</td>")
        rows = []
        for i in range(0, len(cells), 2):
            rows.append("<tr>" + "".join(cells[i:i + 2]) + "</tr>")
        return "<table cellspacing='0' cellpadding='1'>" + "".join(rows) + "</table>"

    def _toggle_details(self):
        self._details_open = not self._details_open
        self.details_label.setVisible(self._details_open)
        self.details_toggle_btn.setText(
            "Track details ▾" if self._details_open else "Track details ▸")

    def _set_approved_silently(self, checked: bool, enabled: bool):
        self.approved_check.blockSignals(True)
        try:
            self.approved_check.setChecked(checked)
            self.approved_check.setEnabled(enabled)
        finally:
            self.approved_check.blockSignals(False)

    def _on_approved_toggled(self, checked: bool):
        """Record the verdict on the track, as the inventory report does."""
        if self._track_id is None or not self._target_folder or not self._modality:
            return
        from .core import track_store

        try:
            track_store.set_track_attribute(
                self._target_folder, self._modality, self._track_id,
                "approved", True if checked else None)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "BAMBI Feature Viewer",
                                f"Could not record the verdict: {exc}")
            self._set_approved_silently(not checked, enabled=True)
            return
        if self._details_row is not None:
            self._details_row["approved"] = checked
        self.trackApproved.emit(self._target_folder, self._modality,
                                self._track_id, checked)

    # ------------------------------------------------------------------
    # Deleting
    # ------------------------------------------------------------------

    def _current_detection_ids(self) -> List[int]:
        if not self._frames:
            return []
        return list(self._frames[self._current_idx].get("detection_ids_green") or [])

    def _update_review_buttons(self):
        reviewable = bool(self._target_folder and self._modality) \
            and not self._is_fov_view() and bool(self._frames)
        self.review_widget.setVisible(reviewable)
        if not reviewable:
            return
        ids = self._current_detection_ids()
        camera = "thermal" if self._modality == "t" else "RGB"
        self.delete_detection_btn.setEnabled(bool(ids))
        self.delete_detection_btn.setText(
            f"Delete {camera} detection" if len(ids) < 2
            else f"Delete {len(ids)} {camera} detections")
        self.delete_track_btn.setVisible(self._track_id is not None)
        if self._track_id is not None:
            self.delete_track_btn.setText(f"Delete track {self._track_id}")

    def _delete_current_detection(self):
        """Remove the highlighted box(es) on the current frame from the project."""
        from .core import review

        ids = self._current_detection_ids()
        if not ids or not self._target_folder or not self._modality:
            return
        frame = self._frames[self._current_idx]
        frame_idx = frame.get("frame_idx")
        camera = "thermal" if self._modality == "t" else "RGB"
        what = (f"{camera} detection {ids[0]}" if len(ids) == 1
                else f"{len(ids)} {camera} detections")
        lines = [f"Delete {what} on frame {frame_idx} from the project?", ""]
        if self._track_id is not None:
            members = review.track_members(
                self._target_folder, self._modality, self._track_id)
            remaining = len([m for m in members if m not in ids])
            if remaining == 0:
                lines.append(f"It is the last box of track {self._track_id}, "
                             "so the track is removed with it.")
            else:
                lines.append(f"Track {self._track_id} keeps its other "
                             f"{remaining} box(es).")
        lines += ["The box, its ground position, its classifier results and "
                  "its match on the other camera are removed from the store "
                  "and from the map layers.", "", "This cannot be undone."]
        reply = QMessageBox.question(
            self, "Delete detection", "\n".join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        boxes = list(frame.get("boxes_green") or [])
        try:
            result = review.delete_detections(
                self._target_folder, self._modality, ids)
        except review.ReviewError as exc:
            QMessageBox.information(self, "Delete detection", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Delete detection",
                                 f"Could not delete: {exc}")
            return

        # The map: the polygon(s) go; a track that lost a member is redrawn.
        try:
            from . import bambi_layer_sync
            for detection_id, box in zip(ids, boxes):
                if frame_idx is not None and len(box) >= 6:
                    bambi_layer_sync.remove_detection(
                        self._target_folder, self._modality, detection_id,
                        int(frame_idx), float(box[4]), int(box[5]))
            for track_id in result.get("tracks_removed", []):
                bambi_layer_sync.remove_track(
                    self._target_folder, self._modality, track_id)
            for track_id in result.get("tracks_kept", []):
                bambi_layer_sync.redraw_track(
                    self._target_folder, self._modality, track_id)
        except Exception as exc:  # noqa: BLE001 - the store is done, say so
            QMessageBox.warning(
                self, "Delete detection",
                "Deleted from the project, but the map layers could not be "
                f"updated: {exc}\nRe-add the detection and track layers.")

        track_for_signal = self._track_id if self._track_id is not None else -1
        for detection_id in ids:
            self.detectionDeleted.emit(self._target_folder, self._modality,
                                       int(detection_id), int(track_for_signal))
        for track_id in result.get("tracks_removed", []):
            self.trackDeleted.emit(self._target_folder, self._modality,
                                   int(track_id))

        self._after_detection_deleted(result)

    def _after_detection_deleted(self, result: dict):
        """Show what is left: the track's remaining frames, or the frame
        without its box."""
        if self._track_id is not None:
            if self._track_id in result.get("tracks_removed", []):
                QMessageBox.information(
                    self, "Delete detection",
                    f"Track {self._track_id} had no other box and was removed.")
                self._track_id = None
                self._frames = []
                self.close()
                return
            from .core import inspection
            try:
                frames = inspection.track_frames(
                    self._target_folder, self._modality, self._track_id)
            except Exception:  # noqa: BLE001 - keep what is shown
                frames = []
            if frames:
                self._frames = frames
                self._current_idx = min(self._current_idx, len(frames) - 1)
                self.title_label.setText(
                    f"Track {self._track_id}   |   {len(frames)} frame(s)")
                self.nav_widget.setVisible(len(frames) > 1)
                self._update_toggle_btn()
                self._update_proj_btn()
                self._render_current_frame()
                self._load_details()
                return
        # A single detection view: keep the frame, drop the box.
        frame = self._frames[self._current_idx]
        frame["boxes_green"] = []
        frame["detection_ids_green"] = []
        frame["boxes_green_other"] = []
        frame["boxes_green_proj"] = None
        self.title_label.setText(self.title_label.text() + "   (deleted)")
        self._render_current_frame()
        self._update_review_buttons()

    def _delete_current_track(self):
        """Remove the shown track with all its detections from the project."""
        from .core import review

        if self._track_id is None or not self._target_folder or not self._modality:
            return
        track_id = self._track_id
        facts = review.describe_track_deletion(
            self._target_folder, self._modality, track_id)
        if not facts["exists"]:
            QMessageBox.information(
                self, "Delete track",
                f"Track {track_id} is not in the store any more.")
            return
        camera = "thermal" if self._modality == "t" else "RGB"
        lines = [f"Delete {camera} track {track_id} with its "
                 f"{facts['detections']} detection(s) from the project?", ""]
        if facts["matched"]:
            lines.append("Its match with the other camera is removed; the "
                         "partner track stays.")
        if facts["label_track_ids"]:
            ids = ", ".join(str(i) for i in facts["label_track_ids"])
            lines.append(f"It was drawn in the labelling tool (label track "
                         f"{ids}); that annotation is deleted with it.")
        lines += ["The track, its boxes, their ground positions and "
                  "classifier results are removed from the store and from "
                  "the map layers.", "", "This cannot be undone."]
        reply = QMessageBox.question(
            self, "Delete track", "\n".join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Every box of the track, read before they go: the per-frame
        # detection layers hold them too, whichever frame is shown here.
        from .core import track_store
        try:
            boxes = [m for m in track_store.load_pixel_tracks(
                self._target_folder, self._modality)
                if int(m["track_id"]) == track_id]
        except Exception:  # noqa: BLE001 - then only the shown frames are known
            boxes = [{
                "detection_id": detection_id, "frame": frame.get("frame_idx"),
                "confidence": box[4], "species_id": box[5]}
                for frame in self._frames
                for detection_id, box in zip(
                    frame.get("detection_ids_green") or [],
                    frame.get("boxes_green") or [])
                if frame.get("frame_idx") is not None and len(box) >= 6]

        try:
            review.delete_tracks(self._target_folder, self._modality, [track_id])
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Delete track",
                                 f"Could not delete: {exc}")
            return

        try:
            from . import bambi_layer_sync
            bambi_layer_sync.remove_track(
                self._target_folder, self._modality, track_id)
            for box in boxes:
                if box.get("frame") is None:
                    continue
                confidence = box.get("confidence")
                bambi_layer_sync.remove_detection(
                    self._target_folder, self._modality,
                    int(box["detection_id"]), int(box["frame"]),
                    1.0 if confidence is None else float(confidence),
                    int(box.get("species_id") or 0))
        except Exception as exc:  # noqa: BLE001 - the store is done, say so
            QMessageBox.warning(
                self, "Delete track",
                "Deleted from the project, but the map layers could not be "
                f"updated: {exc}\nRe-add the detection and track layers.")

        self.trackDeleted.emit(self._target_folder, self._modality, int(track_id))
        self._track_id = None
        self._frames = []
        self.close()

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
            self._target_folder = data["target_folder"]
            self._dem_path = data.get("dem_path", "")
            self._correction_path = data.get("correction_path", "")

        frame_idx = data.get("frame_idx")
        total = len(self._frames)

        # Pick image path according to current view mode with fallback
        if self._view_mode == "t":
            image_path = data.get("image_path_t") or data.get("image_path_w", "")
        else:
            image_path = data.get("image_path_w") or data.get("image_path_t", "")

        # Select boxes. On the source camera: its own boxes. On the other
        # camera: what that camera detected on its frame of this moment
        # (green = matched to the highlighted detection, blue = the rest),
        # with the source boxes projected through the DEM drawn on top in
        # red/orange once they have been computed.
        boxes_modality = data.get("boxes_modality", "t")
        projected_green, projected_blue = [], []
        if self._view_mode == boxes_modality:
            boxes_green = data.get("boxes_green", [])
            boxes_blue = data.get("boxes_blue", [])
            projected = False
        else:
            boxes_green = data.get("boxes_green_other", [])
            boxes_blue = data.get("boxes_blue_other", [])
            projected_green = data.get("boxes_green_proj") or []
            projected_blue = data.get("boxes_blue_proj") or []
            projected = bool(projected_green or projected_blue)

        # Navigation label + button states. On the other camera the frame
        # shown is the time-matched partner, which has its own index.
        label = f"Frame {frame_idx}" if frame_idx is not None else ""
        if self._view_mode != boxes_modality:
            partner = data.get(f"frame_idx_{self._view_mode}")
            camera = "RGB" if self._view_mode == "w" else "thermal"
            if partner is not None and partner != frame_idx:
                label += f"   ({camera} frame {partner})"
        if total > 1:
            self.frame_label.setText(
                f"{label}   ({self._current_idx + 1} / {total})"
            )
        else:
            self.frame_label.setText(label)
        self.prev_btn.setEnabled(self._current_idx > 0)
        self.next_btn.setEnabled(self._current_idx < total - 1)
        self._update_review_buttons()

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

        # Draw bounding boxes and optional click-position crosshair.
        annotated = self._draw_boxes(img, boxes_green, boxes_blue, projected=False)
        if projected:
            annotated = self._draw_boxes(annotated, projected_green,
                                         projected_blue, projected=True)
        click_key = "click_point_t" if self._view_mode == "t" else "click_point_w"
        click_pt = data.get(click_key)
        if click_pt is not None:
            annotated = self._draw_crosshair(annotated, click_pt[0], click_pt[1])
        scaled = annotated.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(QPixmap.fromImage(scaled))

        # Info text from first green box.
        # FoV frames (identified by per-frame target_folder) only show the
        # frame index and detection count - confidence/class are not meaningful.
        is_fov_frame = "target_folder" in data
        info_parts = []
        if frame_idx is not None:
            info_parts.append(f"Frame: {frame_idx}")
        if self._view_mode != boxes_modality:
            camera = "RGB" if self._view_mode == "w" else "thermal"
            info_parts.append(
                f"{len(boxes_green) + len(boxes_blue)} {camera} detection(s), "
                f"{len(boxes_green)} matched")
            if projected:
                info_parts.append("projected boxes in red/orange")
        elif is_fov_frame:
            info_parts.append(f"{len(boxes_green)} detection(s)")
        elif boxes_green:
            b = boxes_green[0]
            if len(b) >= 6:
                info_parts.append(f"Conf: {float(b[4]):.3f}")
                info_parts.append(f"Class: {int(b[5])}")
            if len(b) >= 7 and b[6]:
                info_parts.append("interpolated")
            ids = data.get("detection_ids_green") or []
            if ids:
                info_parts.append(f"Detection: {ids[0]}")
        self.info_label.setText("   |   ".join(info_parts))

    def _draw_boxes(self, img, green_boxes, blue_boxes, projected=False):
        """Paint bounding boxes onto a copy of *img* and return the result.

        Box tuple format: (x1, y1, x2, y2[, conf, cls[, is_interpolated]])
        Interpolated boxes (is_interpolated=1) are drawn with a dashed line.
        Native boxes use green/blue; projected boxes use red/orange.
        """
        result = img.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        lw_secondary = max(2, img.width() // 400)
        lw_primary = max(3, img.width() // 280)

        def make_pen(color, lw, dashed):
            pen = QPen(color, lw, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
            return pen

        font = QFont("Arial", max(8, img.width() // 80))
        painter.setFont(font)

        if projected:
            color_primary = QColor(220, 40, 40)  # red - highlighted
            color_secondary = QColor(255, 160, 0)  # orange - others
        else:
            color_primary = QColor(0, 220, 0)  # green - highlighted
            color_secondary = QColor(80, 140, 255)  # blue - others

        # Draw secondary first so primary is always on top
        for boxes, color, lw in [
            (blue_boxes, color_secondary, lw_secondary),
            (green_boxes, color_primary, lw_primary),
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

    def _draw_crosshair(self, img: "QImage", px: float, py: float) -> "QImage":
        """Draw a yellow crosshair at *(px, py)* in image pixel coordinates.

        The crosshair consists of two diagonal lines (×) centred on the point,
        scaled proportionally to the image size so it remains visible regardless
        of resolution.
        """
        result = img.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        arm = max(8, img.width() // 60)
        lw = max(2, img.width() // 300)

        pen = QPen(QColor(255, 220, 0), lw, Qt.PenStyle.SolidLine)   # yellow
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        cx, cy = int(px), int(py)
        painter.drawLine(cx - arm, cy - arm, cx + arm, cy + arm)
        painter.drawLine(cx + arm, cy - arm, cx - arm, cy + arm)

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
