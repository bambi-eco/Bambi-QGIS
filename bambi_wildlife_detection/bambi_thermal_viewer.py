# -*- coding: utf-8 -*-
"""
BAMBI Thermal Image Viewer
==========================

Dialog for loading and visualising DJI radiometric thermal JPG images.
Uses thermal_parser to extract per-pixel temperature values (°C), applies a
configurable colormap and either optional lower/upper clipping thresholds
(pixels outside the range are rendered black) or a curve
mapping (see ``bambi_curve_widget``).  Mouse hover shows the temperature at
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

from .bambi_curve_widget import CurveEditorPanel


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
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        flags = Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint
        flags |= Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint
        self.setWindowFlags(flags)
        self.resize(860, 680)

        self._temp_array = None
        self._full_pixmap = None
        self._cmap_name = 'white-hotspot'
        self._file_info = ""
        self._get_cmap = None   # resolved on first render
        self._thermal = None    # Thermal instance, kept alive to hold DLL refs
        self._image_list = []   # paths when a folder is open
        self._image_index = -1  # -1 = single-file mode
        self._thermal_meta = {}  # last-loaded measurement parameters
        self._current_path = None

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
        self._nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        # Tone mapping mode: simple thresholds vs. curve mapping
        self._tone_mode_box = QComboBox()
        self._tone_mode_box.addItems(["Thresholds (lower/upper)",
                                      "Curve (custom mapping)"])
        self._tone_mode_box.setToolTip(
            "Thresholds: linear stretch with optional black clipping.\n"
            "Curve: fine-granular tone mapping over a fixed temperature "
            "range, like the Curves tool in image editors.")
        self._tone_mode_box.currentIndexChanged.connect(self._on_tone_mode_changed)
        form.addRow("Tone mapping:", self._tone_mode_box)

        # Lower threshold
        lo_row = QHBoxLayout()
        self._lo_label = QLabel("Lower threshold (→ black):")
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
        self._lo_row_widget = QWidget()
        self._lo_row_widget.setLayout(lo_row)
        form.addRow(self._lo_label, self._lo_row_widget)

        # Upper threshold
        hi_row = QHBoxLayout()
        self._hi_label = QLabel("Upper threshold (→ black):")
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
        self._hi_row_widget = QWidget()
        self._hi_row_widget.setLayout(hi_row)
        form.addRow(self._hi_label, self._hi_row_widget)

        # Curve mapping panel (hidden while tone mapping = thresholds)
        self._curve_panel = CurveEditorPanel(
            image_paths_provider=self._curve_image_paths,
            parse_factory=self._curve_parse_factory,
        )
        self._curve_panel.curveChanged.connect(self._refresh_display)
        self._curve_panel.setVisible(False)
        form.addRow(self._curve_panel)

        root.addWidget(ctrl)

        # Image display label
        self._img_label = _ThermalLabel()
        self._img_label.set_hover_callback(self._on_hover)
        root.addWidget(self._img_label, 1)

        # Status row: file info + cursor temperature
        status_row = QHBoxLayout()
        self._info_label = QLabel("Load a DJI thermal JPG to begin.")
        self._cursor_label = QLabel("")
        self._cursor_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_row.addWidget(self._info_label, 1)
        status_row.addWidget(self._cursor_label)
        root.addLayout(status_row)

        # Metadata row: measurement parameters from EXIF/XMP
        self._meta_label = QLabel("")
        root.addWidget(self._meta_label)

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

    def _ensure_thermal(self):
        """Lazy-create the shared Thermal instance; returns it or None.

        Created once and reused for all files so the SDK DLLs are loaded
        only once and the OS reference count stays at 1.
        """
        if self._thermal is None:
            Thermal, err = _load_thermal()
            if Thermal is None:
                QMessageBox.warning(
                    self, "Thermal parser unavailable",
                    f"Could not load the thermal parser:\n{err}\n\n"
                    "Make sure numpy and Pillow are installed in QGIS's Python."
                )
                return None
            import numpy as np
            self._thermal = Thermal(dtype=np.float32)
        return self._thermal

    # Providers for the curve panel's "Auto Detect" scan ----------------

    def _curve_image_paths(self):
        if self._image_list:
            return list(self._image_list)
        return [self._current_path] if self._current_path else []

    def _curve_parse_factory(self):
        thermal = self._ensure_thermal()
        if thermal is None:
            raise RuntimeError("thermal parser could not be loaded")
        # The viewer owns the instance — nothing to close after the scan.
        return thermal.parse, (lambda: None)

    def _load(self, path):
        if self._ensure_thermal() is None:
            return

        self._current_path = path
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

        # Feed the curve panel: seed the domain once, refresh the histogram
        self._curve_panel.seed_domain_if_default(t_min, t_max)
        self._curve_panel.set_temperature_array(arr)

        self._file_info = (
            f"{os.path.basename(path)}  |  "
            f"{arr.shape[1]} × {arr.shape[0]} px  |  "
            f"range: {t_min:.1f} – {t_max:.1f} °C"
        )

        try:
            from .bambi_thermal import read_thermal_meta
            self._thermal_meta = read_thermal_meta(path)
        except Exception:
            self._thermal_meta = {}
        self._meta_label.setText(self._format_meta(self._thermal_meta))

        self._refresh_display()

    # ------------------------------------------------------------------
    # Display pipeline
    # ------------------------------------------------------------------

    def _on_cmap_changed(self, name):
        self._cmap_name = name
        self._refresh_display()

    def _on_tone_mode_changed(self, index):
        use_curve = index == 1
        for w in (self._lo_label, self._lo_row_widget,
                  self._hi_label, self._hi_row_widget):
            w.setVisible(not use_curve)
        self._curve_panel.setVisible(use_curve)
        self._refresh_display()

    def _refresh_display(self):
        """Recompute the full-resolution colormapped pixmap and update the label."""
        if self._temp_array is None:
            return

        import numpy as np  # safe: already in QGIS's process

        arr = self._temp_array
        mask = None

        if self._tone_mode_box.currentIndex() == 1:
            # Curve mapping: out-of-domain temperatures clamp to the curve's
            # endpoint values instead of being blacked out.
            norm = self._curve_panel.curve().apply(arr)
        else:
            # Threshold mode: linear stretch + optional black clipping
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

        if mask is not None:
            rgb[mask] = 0   # pixels outside thresholds → black

        h, w = arr.shape
        # Keep the bytes object alive until QPixmap.fromImage() copies the data
        raw = rgb.tobytes()
        q_img = QImage(raw, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._full_pixmap = QPixmap.fromImage(q_img)
        del q_img, raw

        self._update_label()
        self._info_label.setText(self._file_info)

    def _update_label(self):
        """Scale the stored full-res pixmap to fit the current label size."""
        if self._full_pixmap is None or self._full_pixmap.isNull():
            return
        scaled = self._full_pixmap.scaled(
            self._img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self._img_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Metadata formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_meta(meta: dict) -> str:
        if not meta:
            return ""
        parts = []
        if 'reflected_apparent_temperature' in meta:
            parts.append(f"Reflected: {meta['reflected_apparent_temperature']:.1f} °C")
        if 'object_distance' in meta:
            parts.append(f"Distance: {meta['object_distance']:.1f} m")
        if 'emissivity' in meta:
            parts.append(f"Emissivity: {meta['emissivity']:.2f}")
        if 'relative_humidity' in meta:
            parts.append(f"Humidity: {meta['relative_humidity']:.1f} %")
        return "  |  ".join(parts)

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
            if event.key() == Qt.Key.Key_Left:
                self._go_prev()
                return
            if event.key() == Qt.Key.Key_Right:
                self._go_next()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._thermal is not None:
            self._thermal.close()
            self._thermal = None
        super().closeEvent(event)
