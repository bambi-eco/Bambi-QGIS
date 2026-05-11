# -*- coding: utf-8 -*-
"""
BAMBI Thermal Image Viewer
==========================

Dialog for loading and visualising DJI radiometric thermal JPG images.
Uses thermal_parser to extract per-pixel temperature values (°C), applies a
configurable colormap and optional lower/upper clipping thresholds (pixels
outside the range are rendered black).  Mouse hover shows the temperature at
the cursor position.

All heavy / native imports (thermal_parser, matplotlib) are deferred to the
functions that actually need them so that a DLL conflict on Windows cannot
cause a hard crash during module import or dialog creation.
"""

import os

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QDoubleSpinBox, QGroupBox, QFormLayout,
    QSizePolicy, QCheckBox, QComboBox, QMessageBox, QWidget,
)
from qgis.PyQt.QtGui import QPixmap, QImage
from qgis.PyQt.QtCore import Qt


_COLORMAPS = ['white-hotspot', 'black-hotspot', 'plasma', 'inferno', 'magma', 'viridis', 'jet']

# Maps display names to matplotlib colormap names.
_CMAP_ALIASES = {
    'white-hotspot': 'gray',    # hot → white, cold → black
    'black-hotspot': 'gray_r',  # hot → black, cold → white
}


# ---------------------------------------------------------------------------
# Lazy helpers – imported only on first use
# ---------------------------------------------------------------------------

def _load_numpy():
    import numpy as np
    return np


def _load_thermal():
    """Return (Thermal class, error_string).  error_string is None on success."""
    try:
        from .bambi_thermal import Thermal
        return Thermal, None
    except Exception as exc:
        return None, str(exc)


def _make_get_cmap():
    """Return a callable get_cmap(name) using whatever matplotlib version is present."""
    try:
        import matplotlib as mpl
        if hasattr(mpl, 'colormaps'):
            def _get(name):
                return mpl.colormaps[_CMAP_ALIASES.get(name, name)]
        else:
            import matplotlib.cm as cm
            def _get(name):
                return cm.get_cmap(_CMAP_ALIASES.get(name, name))
        return _get, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Custom label
# ---------------------------------------------------------------------------

class _ThermalLabel(QLabel):
    """QLabel that maps mouse hover positions to image pixel coordinates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1a1a1a;")
        self._temp_array = None
        self._hover_cb = None

    def set_temperature_array(self, arr):
        self._temp_array = arr

    def set_hover_callback(self, cb):
        self._hover_cb = cb

    def mouseMoveEvent(self, event):
        if self._temp_array is None or self._hover_cb is None:
            return super().mouseMoveEvent(event)
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return super().mouseMoveEvent(event)

        rows, cols = self._temp_array.shape
        pw, ph = pix.width(), pix.height()
        # QLabel with AlignCenter centres the pixmap inside the widget
        ox = (self.width() - pw) // 2
        oy = (self.height() - ph) // 2
        mx, my = event.x(), event.y()

        if mx < ox or my < oy or mx >= ox + pw or my >= oy + ph:
            self._hover_cb(None, None, None)
        else:
            img_x = min(int((mx - ox) * cols / pw), cols - 1)
            img_y = min(int((my - oy) * rows / ph), rows - 1)
            self._hover_cb(img_y, img_x, float(self._temp_array[img_y, img_x]))

        return super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover_cb:
            self._hover_cb(None, None, None)
        super().leaveEvent(event)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ThermalViewerDialog(QDialog):
    """Non-modal dialog for viewing DJI radiometric thermal JPG images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thermal Image Viewer")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowCloseButtonHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(860, 680)

        self._temp_array = None
        self._full_pixmap = None
        self._cmap_name = 'white-hotspot'
        self._file_info = ""
        self._get_cmap = None   # resolved on first render
        self._thermal = None    # Thermal instance, kept alive to hold DLL refs
        self._image_list = []   # paths when a folder is open
        self._image_index = -1  # -1 = single-file mode

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # File selection row
        file_row = QHBoxLayout()
        self._browse_btn = QPushButton("Open Image…")
        self._browse_btn.clicked.connect(self._browse)
        self._browse_folder_btn = QPushButton("Open Folder…")
        self._browse_folder_btn.clicked.connect(self._browse_folder)
        self._path_label = QLabel("No file loaded")
        self._path_label.setWordWrap(True)
        file_row.addWidget(self._browse_btn)
        file_row.addWidget(self._browse_folder_btn)
        file_row.addWidget(self._path_label, 1)
        root.addLayout(file_row)

        # Navigation row — visible only in folder mode
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        self._prev_btn = QPushButton("◄  Previous")
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn = QPushButton("Next  ►")
        self._next_btn.clicked.connect(self._go_next)
        self._nav_label = QLabel("")
        self._nav_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self._prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self._nav_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self._next_btn)
        self._nav_widget = nav_widget
        self._nav_widget.setVisible(False)
        root.addWidget(self._nav_widget)

        # Display options
        ctrl = QGroupBox("Display Options")
        form = QFormLayout(ctrl)

        self._cmap_box = QComboBox()
        self._cmap_box.addItems(_COLORMAPS)
        self._cmap_box.currentTextChanged.connect(self._on_cmap_changed)
        form.addRow("Colormap:", self._cmap_box)

        # Lower threshold
        lo_row = QHBoxLayout()
        self._lo_check = QCheckBox("Enable")
        self._lo_spin = QDoubleSpinBox()
        self._lo_spin.setRange(-200.0, 3000.0)
        self._lo_spin.setDecimals(1)
        self._lo_spin.setSuffix(" °C")
        self._lo_spin.setEnabled(False)
        self._lo_check.toggled.connect(self._lo_spin.setEnabled)
        self._lo_check.toggled.connect(self._refresh_display)
        self._lo_spin.valueChanged.connect(self._refresh_display)
        lo_row.addWidget(self._lo_check)
        lo_row.addWidget(self._lo_spin)
        lo_row.addStretch()
        form.addRow("Lower threshold (→ black):", lo_row)

        # Upper threshold
        hi_row = QHBoxLayout()
        self._hi_check = QCheckBox("Enable")
        self._hi_spin = QDoubleSpinBox()
        self._hi_spin.setRange(-200.0, 3000.0)
        self._hi_spin.setDecimals(1)
        self._hi_spin.setSuffix(" °C")
        self._hi_spin.setEnabled(False)
        self._hi_check.toggled.connect(self._hi_spin.setEnabled)
        self._hi_check.toggled.connect(self._refresh_display)
        self._hi_spin.valueChanged.connect(self._refresh_display)
        hi_row.addWidget(self._hi_check)
        hi_row.addWidget(self._hi_spin)
        hi_row.addStretch()
        form.addRow("Upper threshold (→ black):", hi_row)

        root.addWidget(ctrl)

        # Image display label
        self._img_label = _ThermalLabel()
        self._img_label.set_hover_callback(self._on_hover)
        root.addWidget(self._img_label, 1)

        # Status row: file info + cursor temperature
        status_row = QHBoxLayout()
        self._info_label = QLabel("Load a DJI thermal JPG to begin.")
        self._cursor_label = QLabel("")
        self._cursor_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self._info_label, 1)
        status_row.addWidget(self._cursor_label)
        root.addLayout(status_row)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DJI Thermal Image", "",
            "JPEG Images (*.jpg *.jpeg);;All Files (*.*)",
        )
        if path:
            self._image_list = []
            self._image_index = -1
            self._nav_widget.setVisible(False)
            self._load(path)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open Folder of Thermal Images", ""
        )
        if not folder:
            return
        exts = {'.jpg', '.jpeg'}
        images = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not images:
            QMessageBox.information(
                self, "No Images Found",
                f"No JPEG images found in:\n{folder}"
            )
            return
        self._image_list = images
        self._nav_widget.setVisible(True)
        self._navigate(0)

    def _navigate(self, index):
        self._image_index = index
        total = len(self._image_list)
        self._nav_label.setText(f"{index + 1}  /  {total}")
        self._prev_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < total - 1)
        self._load(self._image_list[index])

    def _go_prev(self):
        if self._image_index > 0:
            self._navigate(self._image_index - 1)

    def _go_next(self):
        if self._image_index < len(self._image_list) - 1:
            self._navigate(self._image_index + 1)

    def _load(self, path):
        # Lazy-create the Thermal instance once and reuse it for all files so
        # the SDK DLLs are loaded only once and the OS reference count stays at 1.
        if self._thermal is None:
            Thermal, err = _load_thermal()
            if Thermal is None:
                QMessageBox.warning(
                    self, "Thermal parser unavailable",
                    f"Could not load the thermal parser:\n{err}\n\n"
                    "Make sure numpy and Pillow are installed in QGIS's Python."
                )
                return
            import numpy as np
            self._thermal = Thermal(dtype=np.float32)

        import numpy as np  # already loaded by QGIS, safe here
        self._path_label.setText(os.path.basename(path))
        self._info_label.setText("Parsing thermal data…")

        try:
            arr = self._thermal.parse(filepath_image=path)
        except Exception as exc:
            self._info_label.setText(f"Error loading file: {exc}")
            return

        self._temp_array = arr
        self._img_label.set_temperature_array(arr)

        t_min, t_max = float(arr.min()), float(arr.max())
        # Only reseed a spin when its threshold is not active so a user-set
        # threshold persists across image navigation.
        for spin in (self._lo_spin, self._hi_spin):
            spin.blockSignals(True)
        if not self._lo_check.isChecked():
            self._lo_spin.setValue(round(t_min, 1))
        if not self._hi_check.isChecked():
            self._hi_spin.setValue(round(t_max, 1))
        for spin in (self._lo_spin, self._hi_spin):
            spin.blockSignals(False)

        self._file_info = (
            f"{os.path.basename(path)}  |  "
            f"{arr.shape[1]} × {arr.shape[0]} px  |  "
            f"range: {t_min:.1f} – {t_max:.1f} °C"
        )
        self._refresh_display()

    # ------------------------------------------------------------------
    # Display pipeline
    # ------------------------------------------------------------------

    def _on_cmap_changed(self, name):
        self._cmap_name = name
        self._refresh_display()

    def _refresh_display(self):
        """Recompute the full-resolution colormapped pixmap and update the label."""
        if self._temp_array is None:
            return

        import numpy as np  # safe: already in QGIS's process

        arr = self._temp_array

        # Build out-of-range mask
        lo_val = self._lo_spin.value() if self._lo_check.isChecked() else None
        hi_val = self._hi_spin.value() if self._hi_check.isChecked() else None

        mask = np.zeros(arr.shape, dtype=bool)
        if lo_val is not None:
            mask |= arr < lo_val
        if hi_val is not None:
            mask |= arr > hi_val

        lo = lo_val if lo_val is not None else float(arr.min())
        hi = hi_val if hi_val is not None else float(arr.max())
        if hi <= lo:
            hi = lo + 1.0

        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        # Lazy-resolve colormap function (imports matplotlib only once)
        if self._get_cmap is None:
            fn, err = _make_get_cmap()
            self._get_cmap = fn  # may be None if matplotlib missing

        if self._get_cmap is not None:
            try:
                cmap = self._get_cmap(self._cmap_name)
                rgba = cmap(norm)           # H×W×4 float64
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            except Exception:
                self._get_cmap = None       # fall back to grayscale
                gray = (norm * 255).astype(np.uint8)
                rgb = np.stack([gray, gray, gray], axis=2)
        else:
            gray = (norm * 255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=2)

        rgb[mask] = 0   # pixels outside thresholds → black

        h, w = arr.shape
        # Keep the bytes object alive until QPixmap.fromImage() copies the data
        raw = rgb.tobytes()
        q_img = QImage(raw, w, h, 3 * w, QImage.Format_RGB888)
        self._full_pixmap = QPixmap.fromImage(q_img)
        del q_img, raw

        self._update_label()
        self._info_label.setText(self._file_info)

    def _update_label(self):
        """Scale the stored full-res pixmap to fit the current label size."""
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        scaled = self._full_pixmap.scaled(
            self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._img_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Hover callback
    # ------------------------------------------------------------------

    def _on_hover(self, row, col, temp):
        if temp is None:
            self._cursor_label.setText("")
        else:
            self._cursor_label.setText(f"({col}, {row})  =  {temp:.2f} °C")

    # ------------------------------------------------------------------
    # Resize handling
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_label()

    def keyPressEvent(self, event):
        if self._image_list:
            if event.key() == Qt.Key_Left:
                self._go_prev()
                return
            if event.key() == Qt.Key_Right:
                self._go_next()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._thermal is not None:
            self._thermal.close()
            self._thermal = None
        super().closeEvent(event)
