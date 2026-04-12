# -*- coding: utf-8 -*-
"""
BAMBI Camera Calibration Wizard
================================
Two-mode wizard dialog for camera calibration:

Single Camera Mode
    Uses Structure from Motion (SfM) via pycolmap to compute intrinsic
    camera parameters from multiple images or a video.

Stereo Mode (RGB + Thermal)
    Uses manually placed corresponding point pairs across image pairs and
    Nelder-Mead optimisation to align the two cameras.  The RGB camera is
    treated as the reference; only the thermal intrinsics are optimised.

Outputs are saved as JSON files compatible with the BAMBI calibration
preset system.
"""

import json
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from scipy.optimize import minimize as scipy_minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import pycolmap
    _HAS_PYCOLMAP = True
except ImportError:
    _HAS_PYCOLMAP = False

from qgis.PyQt.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer, QRect
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDialog, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QProgressBar, QProgressDialog,
    QPushButton, QRadioButton, QScrollArea, QShortcut, QSizePolicy, QSplitter,
    QSpinBox, QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)
from qgis.PyQt.QtGui import (
    QColor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg'}
_MIN_SFM_IMAGES = 3
_MIN_STEREO_POINTS = 4
_DEFAULT_SFM_FRAMES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_qpixmap(img: "np.ndarray") -> "QPixmap":
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _load_image_or_video_central(path: str) -> Optional["np.ndarray"]:
    """Return central frame from video, or full image from image file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _VIDEO_EXTS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    else:
        return cv2.imread(path)


def _extract_n_frames_to_dir(video_path: str, out_dir: str, n: int,
                              progress_cb=None) -> List[str]:
    """Extract *n* evenly-spaced frames from *video_path* into *out_dir*.

    *progress_cb*, if given, is called as ``progress_cb(frames_done, frames_total)``
    after each frame is written so callers can update a progress indicator.
    Returns list of written file paths."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n = min(n, total)
    indices = sorted({int(i * (total - 1) / max(1, n - 1)) for i in range(n)})
    paths = []
    stem = os.path.splitext(os.path.basename(video_path))[0]
    for done, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            out = os.path.join(out_dir, f"{stem}_frame{idx:06d}.jpg")
            cv2.imwrite(out, frame)
            paths.append(out)
        if progress_cb is not None:
            progress_cb(done + 1, len(indices))
    cap.release()
    return paths


def _estimate_intrinsics_from_image(img: "np.ndarray") -> dict:
    """Rough estimate of camera matrix from image dimensions."""
    h, w = img.shape[:2]
    f = max(w, h) * 1.2
    return {
        "ret": None,
        "mtx": [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]],
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "name": "Camera",
    }


# ---------------------------------------------------------------------------
# Clickable annotation image label
# ---------------------------------------------------------------------------

class _AnnotImageLabel(QLabel):
    """QLabel that renders an image and records click positions as numbered
    correspondence points.  Supports an optional magnifier loupe that zooms
    into the full-resolution source image under the cursor.

    Signals
    -------
    pointPlaced(x, y)
        Pixel coordinates in the *original* image space.
    """

    pointPlaced = pyqtSignal(int, int)

    _COLORS = [
        QColor(255, 80, 80), QColor(80, 200, 80), QColor(80, 120, 255),
        QColor(255, 200, 0), QColor(200, 0, 255), QColor(0, 200, 255),
        QColor(255, 140, 0), QColor(180, 255, 100),
    ]

    _MAG_RADIUS = 90    # loupe radius in screen pixels
    _MAG_ZOOM   = 3.0   # zoom factor inside the loupe

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig_pixmap: Optional[QPixmap] = None
        self._orig_w: int = 1
        self._orig_h: int = 1
        self._points: List[Tuple[int, int]] = []
        self._accepting: bool = False
        self._magnifier_enabled: bool = False
        self._mag_pos = None            # QPoint under cursor, or None
        self.setMouseTracking(True)
        self.setMinimumSize(280, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #444;")

    # ---- Public API -------------------------------------------------------

    def load_image(self, img: "np.ndarray") -> None:
        self._orig_pixmap = _bgr_to_qpixmap(img)
        self._orig_h, self._orig_w = img.shape[:2]
        self._points.clear()
        self._redraw()

    def set_accepting(self, v: bool) -> None:
        self._accepting = v
        self.setCursor(Qt.CrossCursor if v else Qt.ArrowCursor)
        border = "2px solid #00aaff" if v else "1px solid #444"
        self.setStyleSheet(f"background: #1a1a1a; border: {border};")

    def set_magnifier_enabled(self, enabled: bool) -> None:
        self._magnifier_enabled = enabled
        if not enabled:
            self._mag_pos = None
        self.update()

    def set_points(self, pts: List[Tuple[int, int]]) -> None:
        self._points = list(pts)
        self._redraw()

    def add_point(self, x: int, y: int) -> None:
        self._points.append((x, y))
        self._redraw()

    def pop_point(self) -> Optional[Tuple[int, int]]:
        if self._points:
            pt = self._points.pop()
            self._redraw()
            return pt
        return None

    def clear_points(self) -> None:
        self._points.clear()
        self._redraw()

    @property
    def points(self) -> List[Tuple[int, int]]:
        return list(self._points)

    # ---- Qt overrides -----------------------------------------------------

    def mouseMoveEvent(self, event):
        self._mag_pos = event.pos()
        if self._magnifier_enabled:
            self.update()

    def leaveEvent(self, event):
        self._mag_pos = None
        if self._magnifier_enabled:
            self.update()

    def mousePressEvent(self, event):
        if not self._accepting or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        if self._orig_pixmap is None:
            return
        lw, lh = max(1, self.width()), max(1, self.height())
        scale = min(lw / self._orig_w, lh / self._orig_h)
        dw = self._orig_w * scale
        dh = self._orig_h * scale
        ox = (lw - dw) / 2.0
        oy = (lh - dh) / 2.0
        px = int((event.x() - ox) / scale)
        py = int((event.y() - oy) / scale)
        if 0 <= px < self._orig_w and 0 <= py < self._orig_h:
            self._points.append((px, py))
            self._redraw()
            self.pointPlaced.emit(px, py)

    def resizeEvent(self, event):
        self._redraw()
        super().resizeEvent(event)

    def paintEvent(self, event):
        """Draw the base pixmap (via parent), then overlay the magnifier loupe."""
        super().paintEvent(event)
        self._draw_magnifier()

    # ---- Internal ---------------------------------------------------------

    def _redraw(self) -> None:
        """Rebuild the composite pixmap (scaled image + point markers) and set it."""
        lw, lh = max(1, self.width()), max(1, self.height())
        result = QPixmap(lw, lh)
        result.fill(QColor(26, 26, 26))
        painter = QPainter(result)

        if self._orig_pixmap is not None:
            scale = min(lw / self._orig_w, lh / self._orig_h)
            dw = int(self._orig_w * scale)
            dh = int(self._orig_h * scale)
            scaled = self._orig_pixmap.scaled(
                dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            ox = (lw - dw) // 2
            oy = (lh - dh) // 2
            painter.drawPixmap(ox, oy, scaled)

            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            painter.setFont(font)
            for i, (x, y) in enumerate(self._points):
                sx = int(ox + x * scale)
                sy = int(oy + y * scale)
                color = self._COLORS[i % len(self._COLORS)]
                pen = QPen(color, 2)
                painter.setPen(pen)
                r = 6
                painter.drawEllipse(sx - r, sy - r, 2 * r, 2 * r)
                painter.drawLine(sx - r - 3, sy, sx + r + 3, sy)
                painter.drawLine(sx, sy - r - 3, sx, sy + r + 3)
                # Shadow + label
                painter.setPen(QPen(Qt.black, 1))
                painter.drawText(sx + r + 2, sy - r + 2, str(i + 1))
                painter.setPen(pen)
                painter.drawText(sx + r + 1, sy - r + 1, str(i + 1))

        painter.end()
        self.setPixmap(result)

    def _img_rect(self) -> Optional[QRect]:
        """Return the QRect occupied by the current pixmap inside the widget."""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        pw, ph = pm.width(), pm.height()
        x = (self.width()  - pw) // 2
        y = (self.height() - ph) // 2
        return QRect(x, y, pw, ph)

    def _draw_magnifier(self) -> None:
        """Paint a circular magnifier loupe on top of the widget contents."""
        if (not self._magnifier_enabled
                or self._mag_pos is None
                or self._orig_pixmap is None):
            return
        ir = self._img_rect()
        if ir is None:
            return

        mx, my = self._mag_pos.x(), self._mag_pos.y()
        if not ir.contains(mx, my):
            return

        # Map widget coords → original-image coords
        rel_x = (mx - ir.x()) / ir.width()
        rel_y = (my - ir.y()) / ir.height()
        fp_w  = self._orig_pixmap.width()
        fp_h  = self._orig_pixmap.height()
        fp_cx = rel_x * fp_w
        fp_cy = rel_y * fp_h

        src_half = self._MAG_RADIUS / self._MAG_ZOOM
        src_rect = QRect(
            max(0, int(fp_cx - src_half)),
            max(0, int(fp_cy - src_half)),
            int(src_half * 2),
            int(src_half * 2),
        ).intersected(QRect(0, 0, fp_w, fp_h))
        if src_rect.isEmpty():
            return

        r = self._MAG_RADIUS
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        clip = QPainterPath()
        clip.addEllipse(mx - r, my - r, 2 * r, 2 * r)
        painter.setClipPath(clip)
        painter.drawPixmap(
            QRect(mx - r, my - r, 2 * r, 2 * r),
            self._orig_pixmap,
            src_rect,
        )

        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255, 160), 1))
        half_hair = 10
        painter.drawLine(mx - half_hair, my, mx + half_hair, my)
        painter.drawLine(mx, my - half_hair, mx, my + half_hair)
        painter.setPen(QPen(QColor(220, 220, 220, 220), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(mx - r, my - r, 2 * r, 2 * r)
        painter.end()


# ---------------------------------------------------------------------------
# Background calibration worker
# ---------------------------------------------------------------------------

class _CalibWorker(QObject):
    """Runs calibration in a background QThread.

    Signals
    -------
    finished(result)   dict with calibration result
    error(message)     str
    progress(percent)  int 0–100
    log(text)          str
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, mode: str, params: dict):
        super().__init__()
        self._mode = mode
        self._params = params
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            result = self._run_single() if self._mode == "single" else self._run_stereo()
            if not self._cancelled:
                self.finished.emit(result)
        except Exception as exc:
            if not self._cancelled:
                self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Single-camera SfM via pycolmap
    # ------------------------------------------------------------------

    def _run_single(self) -> dict:
        if not _HAS_CV2:
            raise RuntimeError("opencv-python (cv2) is required but not installed.")
        if not _HAS_PYCOLMAP:
            raise RuntimeError(
                "pycolmap is required for single-camera SfM calibration.\n\n"
                "Install it with:\n  pip install pycolmap"
            )

        image_paths: List[str] = self._params["image_paths"]
        camera_name: str = self._params.get("camera_name", "Camera")

        if len(image_paths) < _MIN_SFM_IMAGES:
            raise RuntimeError(
                f"SfM calibration requires at least {_MIN_SFM_IMAGES} images, "
                f"got {len(image_paths)}.\nAdd more images or extract more frames from the video."
            )

        self.log.emit(f"Preparing {len(image_paths)} images…")
        self.progress.emit(5)

        tmpdir = tempfile.mkdtemp(prefix="bambi_calib_")
        try:
            img_dir = os.path.join(tmpdir, "images")
            os.makedirs(img_dir)
            for i, src in enumerate(image_paths):
                ext = os.path.splitext(src)[1].lower() or ".jpg"
                dst = os.path.join(img_dir, f"img_{i:04d}{ext}")
                shutil.copy2(src, dst)
                if self._cancelled:
                    return {}

            db_path = os.path.join(tmpdir, "database.db")
            out_dir = os.path.join(tmpdir, "sparse")
            os.makedirs(out_dir)

            self.log.emit("Extracting image features (SIFT)…")
            self.progress.emit(15)

            reader_opts = pycolmap.ImageReaderOptions()
            reader_opts.camera_model = "OPENCV"
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                reader_options=reader_opts,
            )
            if self._cancelled:
                return {}

            self.log.emit("Matching features between image pairs…")
            self.progress.emit(35)
            pycolmap.match_exhaustive(database_path=db_path)
            if self._cancelled:
                return {}

            self.log.emit("Running incremental SfM reconstruction…")
            self.progress.emit(50)
            maps = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_dir,
            )
            if self._cancelled:
                return {}

            if not maps:
                raise RuntimeError(
                    "SfM reconstruction failed — no valid reconstruction found.\n\n"
                    "Tips:\n"
                    "• Images need significant overlap (≥60%)\n"
                    "• Images must have rich, distinct features\n"
                    "• Try adding more images or adjusting the viewpoint"
                )

            best_rec = max(maps.values(), key=lambda r: len(r.images))
            self.log.emit(
                f"Reconstruction: {len(best_rec.images)} registered images, "
                f"{len(best_rec.points3D)} 3D points"
            )
            self.progress.emit(85)

            cam = next(iter(best_rec.cameras.values()))
            params = cam.params
            model = cam.model_name.upper()

            if "OPENCV" in model:
                fx, fy = float(params[0]), float(params[1])
                cx, cy = float(params[2]), float(params[3])
                k1 = float(params[4]) if len(params) > 4 else 0.0
                k2 = float(params[5]) if len(params) > 5 else 0.0
                p1 = float(params[6]) if len(params) > 6 else 0.0
                p2 = float(params[7]) if len(params) > 7 else 0.0
                k3 = float(params[8]) if len(params) > 8 else 0.0
            elif "RADIAL" in model or "FISHEYE" in model:
                fx = fy = float(params[0])
                cx, cy = float(params[1]), float(params[2])
                k1 = float(params[3]) if len(params) > 3 else 0.0
                k2 = float(params[4]) if len(params) > 4 else 0.0
                p1 = p2 = k3 = 0.0
            else:
                if len(params) >= 4:
                    fx, fy, cx, cy = float(params[0]), float(params[1]), float(params[2]), float(params[3])
                elif len(params) >= 3:
                    fx = fy = float(params[0])
                    cx, cy = float(params[1]), float(params[2])
                else:
                    raise RuntimeError(f"Unsupported COLMAP camera model: {model}")
                k1 = k2 = p1 = p2 = k3 = 0.0

            errors = [pt.error for pt in best_rec.points3D.values()]
            ret = float(np.mean(errors)) if errors else None

            self.progress.emit(100)
            self.log.emit("SfM calibration complete.")

            return {
                "mode": "single",
                "camera_name": camera_name,
                "ret": ret,
                "mtx": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "dist": [k1, k2, p1, p2, k3],
                "n_images": len(best_rec.images),
                "n_points3d": len(best_rec.points3D),
            }

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Stereo calibration: homography + Nelder-Mead
    # ------------------------------------------------------------------

    def _run_stereo(self) -> dict:
        if not _HAS_CV2:
            raise RuntimeError("opencv-python (cv2) is required but not installed.")
        if not _HAS_SCIPY:
            raise RuntimeError(
                "scipy is required for stereo calibration.\n\n"
                "Install it with:\n  pip install scipy"
            )

        T_pts_list: List[List[float]] = self._params["T_points"]
        W_pts_list: List[List[float]] = self._params["W_points"]
        initial_calib: dict = self._params["initial_calibration"]

        n = len(T_pts_list)
        if n < _MIN_STEREO_POINTS:
            raise RuntimeError(
                f"Need at least {_MIN_STEREO_POINTS} point correspondences, got {n}."
            )

        self.log.emit(f"Computing homography from {n} point correspondences…")
        self.progress.emit(10)

        T_pts = np.array(T_pts_list, dtype=np.float64).reshape(-1, 1, 2)
        W_pts = np.array(W_pts_list, dtype=np.float64).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(W_pts, T_pts, cv2.RANSAC, 15)
        if M is None:
            raise RuntimeError(
                "Homography computation failed.\n"
                "Add more point correspondences spread across the image."
            )

        W_warped = cv2.perspectiveTransform(W_pts, M)
        hom_mse = float(np.sum((W_warped - T_pts) ** 2) / n)
        n_inliers = int(mask.sum())
        self.log.emit(f"Homography MSE: {hom_mse:.4f}  ({n_inliers}/{n} inliers)")
        self.progress.emit(25)

        if self._cancelled:
            return {}

        # Parse initial calibration
        rgb_data = (
            initial_calib.get("Wide") or initial_calib.get("RGB")
            or initial_calib.get("wide") or initial_calib.get("rgb")
        )
        th_data = initial_calib.get("Thermal") or initial_calib.get("thermal")
        if rgb_data is None or th_data is None:
            raise RuntimeError(
                "Initial calibration must contain both 'Wide'/'RGB' and 'Thermal' sections."
            )

        rgb_mtx = np.array(rgb_data.get("mtx") or rgb_data.get("K"), dtype=np.float64)
        rgb_dist = np.array(rgb_data["dist"], dtype=np.float64).flatten()
        th_mtx = np.array(th_data.get("mtx") or th_data.get("K"), dtype=np.float64)
        th_dist = np.array(th_data["dist"], dtype=np.float64).flatten()

        # Pad distortion to 5 elements
        def _pad5(d):
            d = np.array(d, dtype=np.float64).flatten()
            if len(d) < 5:
                d = np.concatenate([d, np.zeros(5 - len(d))])
            return d[:5]

        th_dist5 = _pad5(th_dist)
        rgb_dist5 = _pad5(rgb_dist)
        new_cm = rgb_mtx.copy()

        def _warp(pts, cm, dc):
            p = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            return cv2.undistortPoints(p, cm, dc.reshape(1, -1), P=new_cm).reshape(-1, 1, 2)

        def mse_func(x):
            th_fx, th_fy, th_cx, th_cy = x[0], x[1], x[2], x[3]
            th_d = np.array(x[4:9])
            th_cm = np.array([[th_fx, 0.0, th_cx], [0.0, th_fy, th_cy], [0.0, 0.0, 1.0]])
            w_w = _warp(W_pts, rgb_mtx, rgb_dist5)
            t_w = _warp(T_pts, th_cm, th_d)
            stacked = np.concatenate([w_w, t_w], axis=1)
            return float(np.sum(np.diff(stacked, axis=1) ** 2, axis=2).mean())

        x0 = [th_mtx[0, 0], th_mtx[1, 1], th_mtx[0, 2], th_mtx[1, 2]] + th_dist5.tolist()

        self.log.emit("Optimising thermal intrinsics (Nelder-Mead, 10 iterations)…")
        n_iters = 10
        res = None
        for i in range(n_iters):
            if self._cancelled:
                return {}
            self.progress.emit(30 + int(60 * i / n_iters))
            res = scipy_minimize(
                mse_func, x0, method="Nelder-Mead",
                options={"maxiter": 50000, "disp": False}, tol=1e-4,
            )
            x0 = res.x.tolist()

        opt_mse = float(res.fun)
        self.log.emit(f"Optimisation complete.  MSE: {opt_mse:.6f}")
        self.progress.emit(100)

        opt_th_mtx = [
            [res.x[0], 0.0, res.x[2]],
            [0.0, res.x[1], res.x[3]],
            [0.0, 0.0, 1.0],
        ]
        opt_th_dist = res.x[4:9].tolist()

        return {
            "mode": "stereo",
            "homography_mse": hom_mse,
            "optimized_mse": opt_mse,
            "Thermal": {
                "ret": opt_mse,
                "mtx": opt_th_mtx,
                "dist": opt_th_dist,
                "name": "Thermal",
            },
            "Wide": {
                "ret": rgb_data.get("ret"),
                "mtx": rgb_mtx.tolist(),
                "dist": rgb_dist5.tolist(),
                "name": "Wide",
            },
        }


# ---------------------------------------------------------------------------
# Main wizard dialog
# ---------------------------------------------------------------------------

class CameraCalibrationWizard(QDialog):
    """Camera calibration wizard (single-camera SfM or stereo RGB+thermal).

    Signals
    -------
    calibrationSaved(path)
        Emitted when calibration JSON is successfully saved.
    """

    calibrationSaved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Calibration Wizard")
        self.setMinimumSize(920, 640)
        self.resize(1060, 720)

        # ---- Internal state -----------------------------------------------
        self._mode: str = "single"          # "single" | "stereo"
        self._input_type: str = "photo"     # "photo" | "video"

        # Single camera
        self._single_paths: List[str] = []  # resolved image paths (after frame extraction)
        self._single_video_paths: List[str] = []  # original video file paths (video mode)

        # Stereo
        self._rgb_paths: List[str] = []     # resolved image paths
        self._th_paths: List[str] = []      # resolved image paths
        self._rgb_video_path: str = ""      # single RGB video (video mode)
        self._th_video_path: str = ""       # single thermal video (video mode)
        self._initial_calib_data: Optional[dict] = None
        # Per-pair annotation: list of {"rgb_pts": [(x,y)…], "th_pts": [(x,y)…]}
        self._pairs_annot: List[dict] = []
        self._pair_idx: int = 0
        # Annotation turn derived from per-pair counts; no explicit state needed

        # Results
        self._result: Optional[dict] = None

        # Worker
        self._worker: Optional[_CalibWorker] = None
        self._worker_thread: Optional[QThread] = None

        self._setup_ui()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Step header
        self._step_lbl = QLabel()
        self._step_lbl.setAlignment(Qt.AlignCenter)
        hdr_font = QFont()
        hdr_font.setBold(True)
        hdr_font.setPointSize(hdr_font.pointSize() + 1)
        self._step_lbl.setFont(hdr_font)
        root.addWidget(self._step_lbl)

        # Stacked pages
        self._pages = QStackedWidget()
        root.addWidget(self._pages)

        self._page_setup = self._build_setup_page()
        self._page_calib = QWidget()       # built dynamically in _enter_calib_page
        self._page_results = self._build_results_page()

        self._pages.addWidget(self._page_setup)    # 0
        self._pages.addWidget(self._page_calib)    # 1
        self._pages.addWidget(self._page_results)  # 2

        # Navigation row
        nav = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel")
        self._back_btn = QPushButton("< Back")
        self._next_btn = QPushButton("Next >")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._back_btn.clicked.connect(self._on_back)
        self._next_btn.clicked.connect(self._on_next)
        nav.addWidget(self._cancel_btn)
        nav.addStretch()
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

        # Dialog-level keyboard shortcut: M toggles magnifier on the stereo page
        sc_m = QShortcut(QKeySequence("M"), self)
        sc_m.activated.connect(self._on_shortcut_m)

        self._pages.setCurrentIndex(0)
        self._update_step_label()
        self._update_nav()

    # ------------------------------------------------------------------
    # Page 0 – Setup
    # ------------------------------------------------------------------

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        # ---- Calibration Mode ----
        mode_grp = QGroupBox("Calibration Mode")
        mode_lay = QVBoxLayout(mode_grp)
        self._rb_single = QRadioButton(
            "Single Camera — SfM intrinsic calibration"
        )
        self._rb_stereo = QRadioButton(
            "Stereo (RGB + Thermal) — manual point correspondences + optimisation"
        )
        self._rb_single.setChecked(True)
        self._rb_single.toggled.connect(self._on_mode_toggled)

        self._calib_mode_info_btn = QPushButton("?")
        self._calib_mode_info_btn.setFixedSize(20, 20)
        self._calib_mode_info_btn.setStyleSheet(
            "QPushButton {"
            "  border-radius: 10px;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "  font-weight: bold;"
            "  font-size: 11px;"
            "}"
            "QPushButton:hover { background: palette(light); }"
            "QPushButton:pressed { background: palette(mid); }"
        )
        self._calib_mode_info_btn.clicked.connect(self._show_calib_mode_info)

        radio_row = QHBoxLayout()
        radio_row.setContentsMargins(0, 0, 0, 0)
        radio_col = QVBoxLayout()
        radio_col.setSpacing(4)
        radio_col.addWidget(self._rb_single)
        radio_col.addWidget(self._rb_stereo)
        radio_row.addLayout(radio_col)
        radio_row.addStretch()
        radio_row.addWidget(self._calib_mode_info_btn, alignment=Qt.AlignTop)
        mode_lay.addLayout(radio_row)
        layout.addWidget(mode_grp)

        # ---- Input Type ----
        input_grp = QGroupBox("Input Type")
        input_lay = QVBoxLayout(input_grp)
        radio_row = QHBoxLayout()
        self._rb_photo = QRadioButton("Photo (images)")
        self._rb_video = QRadioButton("Video (extract frames)")
        self._rb_photo.setChecked(True)
        self._rb_photo.toggled.connect(self._on_input_type_toggled)
        radio_row.addWidget(self._rb_photo)
        radio_row.addWidget(self._rb_video)
        radio_row.addStretch()
        input_lay.addLayout(radio_row)
        input_type_note = QLabel(
            "Note: Most drones do not support open gate recording, so video and photo "
            "modes use different sensor crops and field-of-view settings. Calibrations "
            "from video footage are therefore not interchangeable with photo calibrations."
        )
        input_type_note.setWordWrap(True)
        note_font = input_type_note.font()
        note_font.setItalic(True)
        input_type_note.setFont(note_font)
        input_lay.addWidget(input_type_note)
        layout.addWidget(input_grp)

        # ---- Mode-specific input panels ----
        self._single_widget = self._build_single_inputs()
        self._stereo_widget = self._build_stereo_inputs()
        self._stereo_widget.setVisible(False)
        layout.addWidget(self._single_widget)
        layout.addWidget(self._stereo_widget)

        layout.addStretch()
        return page

    def _build_single_inputs(self) -> QWidget:
        outer = QGroupBox("Input Files")
        outer_lay = QVBoxLayout(outer)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.addStretch()
        _single_info_btn = QPushButton("?")
        _single_info_btn.setFixedSize(20, 20)
        _single_info_btn.setStyleSheet(
            "QPushButton {"
            "  border-radius: 10px;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "  font-weight: bold;"
            "  font-size: 11px;"
            "}"
            "QPushButton:hover { background: palette(light); }"
            "QPushButton:pressed { background: palette(mid); }"
        )
        _single_info_btn.clicked.connect(self._show_input_files_info)
        info_row.addWidget(_single_info_btn)
        outer_lay.addLayout(info_row)

        # --- Photo sub-panel ---
        self._single_photo_widget = QWidget()
        photo_lay = QVBoxLayout(self._single_photo_widget)
        photo_lay.setContentsMargins(0, 0, 0, 0)
        photo_lay.addWidget(QLabel("Image files (multiple views of the same scene):"))
        self._single_photo_list = QListWidget()
        self._single_photo_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._single_photo_list.setMaximumHeight(130)
        photo_lay.addWidget(self._single_photo_list)
        ph_btn_row = QHBoxLayout()
        add_ph_btn = QPushButton("Add Images…")
        rem_ph_btn = QPushButton("Remove Selected")
        add_ph_btn.clicked.connect(self._single_add_photos)
        rem_ph_btn.clicked.connect(self._single_remove_photos)
        ph_btn_row.addWidget(add_ph_btn)
        ph_btn_row.addWidget(rem_ph_btn)
        ph_btn_row.addStretch()
        photo_lay.addLayout(ph_btn_row)
        outer_lay.addWidget(self._single_photo_widget)

        # --- Video sub-panel ---
        self._single_video_widget = QWidget()
        video_lay = QVBoxLayout(self._single_video_widget)
        video_lay.setContentsMargins(0, 0, 0, 0)
        video_lay.addWidget(QLabel("Video file(s) — frames are extracted evenly:"))
        self._single_video_list = QListWidget()
        self._single_video_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._single_video_list.setMaximumHeight(110)
        video_lay.addWidget(self._single_video_list)
        vid_btn_row = QHBoxLayout()
        add_vid_btn = QPushButton("Add Videos…")
        rem_vid_btn = QPushButton("Remove Selected")
        add_vid_btn.clicked.connect(self._single_add_videos)
        rem_vid_btn.clicked.connect(self._single_remove_videos)
        vid_btn_row.addWidget(add_vid_btn)
        vid_btn_row.addWidget(rem_vid_btn)
        vid_btn_row.addStretch()
        video_lay.addLayout(vid_btn_row)
        self._sfm_central_only_chk = QCheckBox("Central frame only (one frame per video)")
        self._sfm_central_only_chk.setChecked(True)
        self._sfm_central_only_chk.toggled.connect(self._on_sfm_central_only_toggled)
        video_lay.addWidget(self._sfm_central_only_chk)
        frames_row = QHBoxLayout()
        self._sfm_frames_lbl = QLabel("Frames to extract per video:")
        frames_row.addWidget(self._sfm_frames_lbl)
        self._sfm_frames_spin = QSpinBox()
        self._sfm_frames_spin.setRange(5, 200)
        self._sfm_frames_spin.setValue(_DEFAULT_SFM_FRAMES)
        self._sfm_frames_spin.setToolTip(
            "How many evenly-spaced frames to extract from each video for SfM."
        )
        frames_row.addWidget(self._sfm_frames_spin)
        frames_row.addStretch()
        video_lay.addLayout(frames_row)
        self._single_video_widget.setVisible(False)
        outer_lay.addWidget(self._single_video_widget)

        # --- Camera name (common) ---
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Camera name:"))
        self._single_cam_name = QLineEdit("Camera")
        name_row.addWidget(self._single_cam_name)
        outer_lay.addLayout(name_row)

        return outer

    def _build_stereo_inputs(self) -> QWidget:
        outer = QGroupBox("Input Files")
        outer_lay = QVBoxLayout(outer)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.addStretch()
        _stereo_info_btn = QPushButton("?")
        _stereo_info_btn.setFixedSize(20, 20)
        _stereo_info_btn.setStyleSheet(
            "QPushButton {"
            "  border-radius: 10px;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "  font-weight: bold;"
            "  font-size: 11px;"
            "}"
            "QPushButton:hover { background: palette(light); }"
            "QPushButton:pressed { background: palette(mid); }"
        )
        _stereo_info_btn.clicked.connect(self._show_input_files_info)
        info_row.addWidget(_stereo_info_btn)
        outer_lay.addLayout(info_row)

        # --- Photo sub-panel ---
        self._stereo_photo_widget = QWidget()
        sp_lay = QVBoxLayout(self._stereo_photo_widget)
        sp_lay.setContentsMargins(0, 0, 0, 0)

        splitter_ph = QSplitter(Qt.Horizontal)

        rgb_ph_grp = QGroupBox("RGB / Wide Camera")
        rgb_ph_lay = QVBoxLayout(rgb_ph_grp)
        self._rgb_photo_list = QListWidget()
        self._rgb_photo_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._rgb_photo_list.setMaximumHeight(110)
        rgb_ph_lay.addWidget(self._rgb_photo_list)
        rp_row = QHBoxLayout()
        add_rp = QPushButton("Add Images…")
        rem_rp = QPushButton("Remove")
        add_rp.clicked.connect(lambda: self._stereo_add_photos("rgb"))
        rem_rp.clicked.connect(lambda: self._stereo_remove_photos("rgb"))
        rp_row.addWidget(add_rp)
        rp_row.addWidget(rem_rp)
        rp_row.addStretch()
        rgb_ph_lay.addLayout(rp_row)
        splitter_ph.addWidget(rgb_ph_grp)

        th_ph_grp = QGroupBox("Thermal Camera")
        th_ph_lay = QVBoxLayout(th_ph_grp)
        self._th_photo_list = QListWidget()
        self._th_photo_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._th_photo_list.setMaximumHeight(110)
        th_ph_lay.addWidget(self._th_photo_list)
        tp_row = QHBoxLayout()
        add_tp = QPushButton("Add Images…")
        rem_tp = QPushButton("Remove")
        add_tp.clicked.connect(lambda: self._stereo_add_photos("th"))
        rem_tp.clicked.connect(lambda: self._stereo_remove_photos("th"))
        tp_row.addWidget(add_tp)
        tp_row.addWidget(rem_tp)
        tp_row.addStretch()
        th_ph_lay.addLayout(tp_row)
        splitter_ph.addWidget(th_ph_grp)

        sp_lay.addWidget(splitter_ph)
        outer_lay.addWidget(self._stereo_photo_widget)

        # --- Video sub-panel ---
        self._stereo_video_widget = QWidget()
        sv_lay = QVBoxLayout(self._stereo_video_widget)
        sv_lay.setContentsMargins(0, 0, 0, 0)
        sv_lay.addWidget(QLabel(
            "Select one video per camera — the central frame will be extracted from each:"
        ))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        rgb_vid_row = QHBoxLayout()
        self._rgb_video_edit = QLineEdit()
        self._rgb_video_edit.setReadOnly(True)
        self._rgb_video_edit.setPlaceholderText("No video selected…")
        rgb_vid_browse = QPushButton("Browse…")
        rgb_vid_browse.clicked.connect(lambda: self._stereo_browse_video("rgb"))
        rgb_vid_row.addWidget(self._rgb_video_edit, 1)
        rgb_vid_row.addWidget(rgb_vid_browse)
        form.addRow("RGB video:", rgb_vid_row)

        th_vid_row = QHBoxLayout()
        self._th_video_edit = QLineEdit()
        self._th_video_edit.setReadOnly(True)
        self._th_video_edit.setPlaceholderText("No video selected…")
        th_vid_browse = QPushButton("Browse…")
        th_vid_browse.clicked.connect(lambda: self._stereo_browse_video("th"))
        th_vid_row.addWidget(self._th_video_edit, 1)
        th_vid_row.addWidget(th_vid_browse)
        form.addRow("Thermal video:", th_vid_row)

        sv_lay.addLayout(form)
        self._stereo_video_widget.setVisible(False)
        outer_lay.addWidget(self._stereo_video_widget)

        # --- Initial calibration (common to both input types) ---
        sep = QLabel()
        sep.setFrameShape(QLabel.HLine)
        sep.setFrameShadow(QLabel.Sunken)
        outer_lay.addWidget(sep)

        calib_row = QHBoxLayout()
        self._init_calib_lbl = QLabel("(none — will be estimated from image size)")
        self._init_calib_lbl.setStyleSheet("color: #888;")
        load_calib_btn = QPushButton("Load JSON…")
        clear_calib_btn = QPushButton("Clear")
        load_calib_btn.clicked.connect(self._load_initial_calib)
        clear_calib_btn.clicked.connect(self._clear_initial_calib)
        calib_row.addWidget(QLabel("Initial calibration:"))
        calib_row.addWidget(self._init_calib_lbl, 1)
        calib_row.addWidget(load_calib_btn)
        calib_row.addWidget(clear_calib_btn)
        outer_lay.addLayout(calib_row)

        return outer

    # ------------------------------------------------------------------
    # Page 1 – Calibration (built dynamically)
    # ------------------------------------------------------------------

    def _build_calib_page_single(self) -> QWidget:
        """Page 1 for single-camera mode: image preview + progress."""
        page = QWidget()
        lay = QVBoxLayout(page)

        info = QLabel(
            f"Ready to run SfM calibration on {len(self._single_paths)} image(s).\n"
            "Click 'Run Calibration' to start."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        # Image list
        self._calib_single_list = QListWidget()
        self._calib_single_list.setMaximumHeight(140)
        for p in self._single_paths:
            self._calib_single_list.addItem(os.path.basename(p))
        lay.addWidget(self._calib_single_list)

        # Progress + log
        self._calib_progress = QProgressBar()
        self._calib_progress.setRange(0, 100)
        self._calib_progress.setValue(0)
        lay.addWidget(self._calib_progress)

        self._calib_log = QTextEdit()
        self._calib_log.setReadOnly(True)
        self._calib_log.setMaximumHeight(120)
        lay.addWidget(self._calib_log)

        # Run button
        run_row = QHBoxLayout()
        self._run_calib_btn = QPushButton("Run Calibration")
        self._run_calib_btn.setMinimumHeight(32)
        self._run_calib_btn.clicked.connect(self._start_calibration)
        run_row.addStretch()
        run_row.addWidget(self._run_calib_btn)
        lay.addLayout(run_row)

        lay.addStretch()
        return page

    def _build_calib_page_stereo(self) -> QWidget:
        """Page 1 for stereo mode: side-by-side annotation + progress."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(6)

        # Instruction label
        self._annot_instr = QLabel()
        self._annot_instr.setAlignment(Qt.AlignCenter)
        self._annot_instr.setStyleSheet(
            "background: #2a3a4a; color: #aaddff; padding: 4px; border-radius: 3px;"
        )
        lay.addWidget(self._annot_instr)

        # Image panels
        splitter = QSplitter(Qt.Horizontal)

        rgb_widget = QWidget()
        rgb_lay = QVBoxLayout(rgb_widget)
        rgb_lay.setContentsMargins(0, 0, 0, 0)
        rgb_lay.addWidget(QLabel("RGB / Wide"))
        self._rgb_img_lbl = _AnnotImageLabel()
        self._rgb_img_lbl.pointPlaced.connect(self._on_rgb_point_placed)
        rgb_lay.addWidget(self._rgb_img_lbl)
        splitter.addWidget(rgb_widget)

        th_widget = QWidget()
        th_lay = QVBoxLayout(th_widget)
        th_lay.setContentsMargins(0, 0, 0, 0)
        th_lay.addWidget(QLabel("Thermal"))
        self._th_img_lbl = _AnnotImageLabel()
        self._th_img_lbl.pointPlaced.connect(self._on_th_point_placed)
        th_lay.addWidget(self._th_img_lbl)
        splitter.addWidget(th_widget)

        lay.addWidget(splitter, 1)

        # Pair navigation + point controls
        ctrl_row = QHBoxLayout()
        self._pair_prev_btn = QPushButton("< Prev Pair")
        self._pair_lbl = QLabel()
        self._pair_next_btn = QPushButton("Next Pair >")
        self._undo_pt_btn = QPushButton("Undo Last Point")
        self._clear_pts_btn = QPushButton("Clear Pair Points")
        self._pt_count_lbl = QLabel()

        self._pair_prev_btn.clicked.connect(lambda: self._navigate_pair(-1))
        self._pair_next_btn.clicked.connect(lambda: self._navigate_pair(+1))
        self._undo_pt_btn.clicked.connect(self._undo_last_point)
        self._clear_pts_btn.clicked.connect(self._clear_pair_points)

        ctrl_row.addWidget(self._pair_prev_btn)
        ctrl_row.addWidget(self._pair_lbl)
        ctrl_row.addWidget(self._pair_next_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self._pt_count_lbl)
        ctrl_row.addWidget(self._undo_pt_btn)
        ctrl_row.addWidget(self._clear_pts_btn)
        lay.addLayout(ctrl_row)

        # Magnifier checkbox + point save/load
        tool_row = QHBoxLayout()
        self._magnifier_chk = QCheckBox("Magnifier on hover  [M]")
        self._magnifier_chk.setChecked(False)
        self._magnifier_chk.toggled.connect(self._on_magnifier_toggled)
        tool_row.addWidget(self._magnifier_chk)
        tool_row.addStretch()
        save_pts_btn = QPushButton("Save Points…")
        load_pts_btn = QPushButton("Load Points…")
        save_pts_btn.clicked.connect(self._save_annot_points)
        load_pts_btn.clicked.connect(self._load_annot_points)
        tool_row.addWidget(save_pts_btn)
        tool_row.addWidget(load_pts_btn)
        lay.addLayout(tool_row)

        # Progress + log
        self._calib_progress = QProgressBar()
        self._calib_progress.setRange(0, 100)
        self._calib_progress.setValue(0)
        lay.addWidget(self._calib_progress)

        self._calib_log = QTextEdit()
        self._calib_log.setReadOnly(True)
        self._calib_log.setMaximumHeight(80)
        lay.addWidget(self._calib_log)

        # Run button
        run_row = QHBoxLayout()
        self._run_calib_btn = QPushButton("Run Calibration")
        self._run_calib_btn.setMinimumHeight(32)
        self._run_calib_btn.clicked.connect(self._start_calibration)
        run_row.addStretch()
        run_row.addWidget(self._run_calib_btn)
        lay.addLayout(run_row)

        return page

    # ------------------------------------------------------------------
    # Page 2 – Results
    # ------------------------------------------------------------------

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        page_lay.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setSpacing(8)
        lay.setContentsMargins(4, 4, 4, 4)

        # Quality metrics
        self._result_metrics_lbl = QLabel()
        self._result_metrics_lbl.setStyleSheet(
            "background: #1e3a1e; color: #88ff88; padding: 6px; border-radius: 3px; "
            "font-family: monospace;"
        )
        self._result_metrics_lbl.setWordWrap(True)
        lay.addWidget(self._result_metrics_lbl)

        # Visual preview — single: distorted vs undistorted; stereo: edge overlay
        self._result_preview_grp = QGroupBox("Visual Preview")
        preview_lay = QVBoxLayout(self._result_preview_grp)
        preview_lay.setSpacing(4)

        # Single-camera: side-by-side distorted / undistorted
        self._preview_single_widget = QWidget()
        single_preview_lay = QHBoxLayout(self._preview_single_widget)
        single_preview_lay.setContentsMargins(0, 0, 0, 0)
        single_preview_lay.setSpacing(6)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        lbl_orig_title = QLabel("Original (distorted)")
        lbl_orig_title.setAlignment(Qt.AlignCenter)
        lbl_undist_title = QLabel("Undistorted")
        lbl_undist_title.setAlignment(Qt.AlignCenter)
        self._preview_orig_lbl = QLabel()
        self._preview_orig_lbl.setAlignment(Qt.AlignCenter)
        self._preview_orig_lbl.setMinimumHeight(180)
        self._preview_orig_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview_undist_lbl = QLabel()
        self._preview_undist_lbl.setAlignment(Qt.AlignCenter)
        self._preview_undist_lbl.setMinimumHeight(180)
        self._preview_undist_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_col.addWidget(lbl_orig_title)
        left_col.addWidget(self._preview_orig_lbl, 1)
        right_col.addWidget(lbl_undist_title)
        right_col.addWidget(self._preview_undist_lbl, 1)
        single_preview_lay.addLayout(left_col, 1)
        single_preview_lay.addLayout(right_col, 1)
        preview_lay.addWidget(self._preview_single_widget)

        # Stereo: single edge overlay label
        self._preview_stereo_widget = QWidget()
        stereo_preview_lay = QVBoxLayout(self._preview_stereo_widget)
        stereo_preview_lay.setContentsMargins(0, 0, 0, 0)
        lbl_overlay_title = QLabel(
            "Edge overlay — RGB Canny edges (green) on undistorted thermal"
        )
        lbl_overlay_title.setAlignment(Qt.AlignCenter)
        self._preview_overlay_lbl = QLabel()
        self._preview_overlay_lbl.setAlignment(Qt.AlignCenter)
        self._preview_overlay_lbl.setMinimumHeight(200)
        self._preview_overlay_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stereo_preview_lay.addWidget(lbl_overlay_title)
        stereo_preview_lay.addWidget(self._preview_overlay_lbl, 1)
        preview_lay.addWidget(self._preview_stereo_widget)
        self._preview_stereo_widget.setVisible(False)

        lay.addWidget(self._result_preview_grp)

        # Feature coverage heatmap
        self._heatmap_grp = QGroupBox("Feature Coverage Heatmap")
        heatmap_grp_lay = QVBoxLayout(self._heatmap_grp)
        heatmap_grp_lay.setSpacing(4)
        self._heatmap_lbl = QLabel()
        self._heatmap_lbl.setAlignment(Qt.AlignCenter)
        self._heatmap_lbl.setMinimumHeight(120)
        self._heatmap_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._heatmap_subtitle = QLabel()
        self._heatmap_subtitle.setAlignment(Qt.AlignCenter)
        self._heatmap_subtitle.setStyleSheet("color: palette(mid); font-size: 11px;")
        heatmap_grp_lay.addWidget(self._heatmap_lbl)
        heatmap_grp_lay.addWidget(self._heatmap_subtitle)
        lay.addWidget(self._heatmap_grp)

        # Parameter display
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setFontFamily("Courier New")
        lay.addWidget(self._result_text, 1)

        # Save row(s) — single mode uses one path, stereo uses two
        save_grp = QGroupBox("Save Calibration")
        save_grp_lay = QVBoxLayout(save_grp)

        # --- single-camera row ---
        self._save_single_widget = QWidget()
        single_row = QHBoxLayout(self._save_single_widget)
        single_row.setContentsMargins(0, 0, 0, 0)
        self._save_path_edit = QLineEdit()
        self._save_path_edit.setPlaceholderText("Select output JSON path…")
        browse_single_btn = QPushButton("Browse…")
        browse_single_btn.clicked.connect(lambda: self._browse_save_path("single"))
        single_row.addWidget(self._save_path_edit, 1)
        single_row.addWidget(browse_single_btn)

        # --- stereo rows ---
        self._save_stereo_widget = QWidget()
        stereo_rows = QVBoxLayout(self._save_stereo_widget)
        stereo_rows.setContentsMargins(0, 0, 0, 0)
        stereo_rows.setSpacing(4)

        th_row = QHBoxLayout()
        th_lbl = QLabel("Thermal:")
        th_lbl.setFixedWidth(60)
        self._save_th_path_edit = QLineEdit()
        self._save_th_path_edit.setPlaceholderText("calibration_thermal.json")
        browse_th_btn = QPushButton("Browse…")
        browse_th_btn.clicked.connect(lambda: self._browse_save_path("thermal"))
        th_row.addWidget(th_lbl)
        th_row.addWidget(self._save_th_path_edit, 1)
        th_row.addWidget(browse_th_btn)
        stereo_rows.addLayout(th_row)

        rgb_row = QHBoxLayout()
        rgb_lbl = QLabel("RGB:")
        rgb_lbl.setFixedWidth(60)
        self._save_rgb_path_edit = QLineEdit()
        self._save_rgb_path_edit.setPlaceholderText("calibration_rgb.json")
        browse_rgb_btn = QPushButton("Browse…")
        browse_rgb_btn.clicked.connect(lambda: self._browse_save_path("rgb"))
        rgb_row.addWidget(rgb_lbl)
        rgb_row.addWidget(self._save_rgb_path_edit, 1)
        rgb_row.addWidget(browse_rgb_btn)
        stereo_rows.addLayout(rgb_row)

        save_grp_lay.addWidget(self._save_single_widget)
        save_grp_lay.addWidget(self._save_stereo_widget)
        self._save_stereo_widget.setVisible(False)

        self._save_btn = QPushButton("Save")
        self._save_btn.setMinimumWidth(80)
        self._save_btn.clicked.connect(self._save_calibration)
        save_btn_row = QHBoxLayout()
        save_btn_row.addStretch()
        save_btn_row.addWidget(self._save_btn)
        save_grp_lay.addLayout(save_btn_row)

        lay.addWidget(save_grp)

        return page

    # =========================================================================
    # Navigation logic
    # =========================================================================

    def _update_nav(self) -> None:
        idx = self._pages.currentIndex()
        self._back_btn.setEnabled(idx > 0)
        if idx == 0:
            self._next_btn.setText("Next >")
            self._next_btn.setEnabled(True)
        elif idx == 1:
            self._next_btn.setText("Next >")
            # Enabled only after successful calibration
            self._next_btn.setEnabled(self._result is not None)
        else:
            self._next_btn.setText("Close")
            self._next_btn.setEnabled(True)

    def _on_next(self) -> None:
        idx = self._pages.currentIndex()
        if idx == 0:
            self._enter_calib_page()
        elif idx == 1:
            self._pages.setCurrentIndex(2)
            self._update_step_label()
            self._update_nav()
        else:
            self.accept()

    def _on_back(self) -> None:
        idx = self._pages.currentIndex()
        if idx > 0:
            self._pages.setCurrentIndex(idx - 1)
            self._update_step_label()
            self._update_nav()

    def _on_cancel(self) -> None:
        self._stop_worker()
        self.reject()

    def _update_step_label(self) -> None:
        idx = self._pages.currentIndex()
        if self._mode == "single":
            labels = [
                "Step 1 / 2  —  Setup",
                "Step 2 / 2  —  SfM Calibration",
                "Step 2 / 2  —  Results",
            ]
        else:
            labels = [
                "Step 1 / 3  —  Setup",
                "Step 2 / 3  —  Point Annotation & Calibration",
                "Step 3 / 3  —  Results",
            ]
        self._step_lbl.setText(labels[min(idx, len(labels) - 1)])

    # =========================================================================
    # Keyboard shortcuts
    # =========================================================================

    def _on_shortcut_m(self) -> None:
        """Toggle magnifier when on the stereo annotation page."""
        if self._pages.currentIndex() == 1 and self._mode == "stereo":
            if hasattr(self, "_magnifier_chk"):
                self._magnifier_chk.toggle()

    # =========================================================================
    # Setup page callbacks
    # =========================================================================

    def _show_input_files_info(self) -> None:
        """Show an info popup explaining calibration data collection."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Input Files — Calibration Guide")
        dlg.setMinimumWidth(520)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        lbl = QLabel(
            "Cameras show distortions due to the lenses. Because of this, image "
            "positions can't be mapped accurately. To address this, we need to "
            "calibrate the cameras and calculate distortion coefficients.<br><br>"
            "For accurate calibration of the RGB and thermal cameras, a dedicated "
            "drone flight should be performed under conditions similar to your mission "
            "setup (e.g. when you typically fly with 100 m AGL, the calibration setup "
            "should also have about 100 m distance to the object used for calibration). "
            "For stereo mode, the recorded data should contain clearly distinguishable structures that "
            "are visible in both RGB and thermal imagery and appear across at least once "
            "everywhere in the image space (upper left corner, upper right corner, "
            "center of the image, lower left, lower right, etc.), to ensure robust "
            "calibration over the full field of view. For single-camera mode, the same requirement applies but only for one modality.<br><br>"
            "Buildings have proven to be particularly suitable targets, especially roofs "
            "with sharp edges or solar panels, as they provide both geometric detail and "
            "strong thermal contrast. Facades with windows can also be used, although "
            "maintaining a consistent distance is more challenging in side views. It is "
            "important that the selected features are clearly recognizable in both "
            "modalities to achieve reliable results.<br><br>"
            "A key limitation of most cameras mounted to DJI drones is that they do not operate in open gate "
            "mode, which leads to differences between video and photo data. As a result, "
            "separate calibrations are required for each, as parameters cannot be "
            "transferred between these acquisition types.<br><br>"
            "Depending on what you use for your surveys, create images and/or short "
            "video sequences (~1 sec) with the drone hovering stably and showing the "
            "object of interest used for calibration.<br><br>"
        )
        lbl.setWordWrap(True)
        lbl.setOpenExternalLinks(True)
        lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
        lay.addWidget(lbl)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        dlg.exec_()

    def _show_calib_mode_info(self) -> None:
        """Show an info popup describing the currently selected calibration mode."""
        if self._rb_single.isChecked():
            title = "Single Camera Calibration"
            text = (
                "<b>Single Camera — Structure from Motion (SfM)</b><br><br>"
                "Estimates intrinsic camera parameters (focal length, principal point, "
                "and distortion coefficients) from a set of overlapping images or video "
                "frames of a static scene using pycolmap's incremental SfM pipeline.<br><br>"
                "<b>When to use:</b> Your drone has only one camera (e.g. RGB only), or "
                "you need to calibrate each camera independently before a stereo "
                "session.<br><br>"
                "<b>Input:</b> At least 10–20 images or video frames showing the scene "
                "from different angles with sufficient overlap. More images improve "
                "accuracy.<br><br>"
                "<b>Output:</b> A single calibration JSON with focal length, principal "
                "point, and distortion coefficients."
            )
        else:
            title = "Stereo Calibration (RGB + Thermal)"
            text = (
                "<b>Stereo — Manual Point Correspondences + Nelder-Mead Optimisation"
                "</b><br><br>"
                "Calibrates a paired RGB and thermal camera system. You manually place "
                "corresponding point pairs on the same ground features in both images. "
                "The thermal intrinsics are then optimised so that thermal points "
                "reproject correctly into the RGB camera space. The RGB camera is used "
                "as the fixed reference and is not modified.<br><br>"
                "<b>When to use:</b> Your drone carries both an RGB and a thermal camera "
                "and you need them spatially aligned for overlaid analysis.<br><br>"
                "<b>Input:</b> One or more image pairs (RGB + thermal) showing the same "
                "scene. At least 6–10 well-spread point correspondences per pair are "
                "recommended.<br><br>"
                "<b>Algorithm:</b> Initial homography (RANSAC) followed by Nelder-Mead "
                "optimisation over thermal fx/fy/cx/cy and 5 distortion coefficients, "
                "minimising reprojection error into RGB space.<br><br>"
                "<b>Output:</b> Two separate calibration JSON files — one for the "
                "thermal camera and one for the RGB camera."
            )
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def _on_mode_toggled(self) -> None:
        self._mode = "single" if self._rb_single.isChecked() else "stereo"
        self._single_widget.setVisible(self._mode == "single")
        self._stereo_widget.setVisible(self._mode == "stereo")

    def _on_input_type_toggled(self) -> None:
        new_type = "photo" if self._rb_photo.isChecked() else "video"
        if new_type == self._input_type:
            return

        # Warn if files are already selected
        has_files = bool(
            self._single_paths or self._single_video_paths
            or self._rgb_paths or self._th_paths
            or self._rgb_video_path or self._th_video_path
        )
        if has_files:
            ans = QMessageBox.question(
                self, "Clear current selection",
                "Switching the input type will clear all currently selected files.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                # Revert radio buttons without triggering signal again
                self._rb_photo.blockSignals(True)
                self._rb_video.blockSignals(True)
                self._rb_photo.setChecked(self._input_type == "photo")
                self._rb_video.setChecked(self._input_type == "video")
                self._rb_photo.blockSignals(False)
                self._rb_video.blockSignals(False)
                return

        self._input_type = new_type
        self._clear_all_inputs()

        # Show/hide sub-panels
        self._single_photo_widget.setVisible(new_type == "photo")
        self._single_video_widget.setVisible(new_type == "video")
        self._stereo_photo_widget.setVisible(new_type == "photo")
        self._stereo_video_widget.setVisible(new_type == "video")

    def _clear_all_inputs(self) -> None:
        self._single_paths.clear()
        self._single_video_paths.clear()
        self._single_photo_list.clear()
        self._single_video_list.clear()
        self._rgb_paths.clear()
        self._th_paths.clear()
        self._rgb_photo_list.clear()
        self._th_photo_list.clear()
        self._rgb_video_path = ""
        self._th_video_path = ""
        self._rgb_video_edit.clear()
        self._th_video_edit.clear()

    # ---- Single camera — photo mode ----------------------------------------

    def _single_add_photos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        for p in paths:
            if p not in self._single_paths:
                self._single_paths.append(p)
                self._single_photo_list.addItem(os.path.basename(p))

    def _single_remove_photos(self) -> None:
        for item in self._single_photo_list.selectedItems():
            row = self._single_photo_list.row(item)
            self._single_photo_list.takeItem(row)
            if row < len(self._single_paths):
                self._single_paths.pop(row)

    # ---- Single camera — video mode ----------------------------------------

    def _on_sfm_central_only_toggled(self, checked: bool) -> None:
        self._sfm_frames_lbl.setEnabled(not checked)
        self._sfm_frames_spin.setEnabled(not checked)

    def _single_add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video(s)", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg)"
        )
        for p in paths:
            if p not in self._single_video_paths:
                self._single_video_paths.append(p)
                self._single_video_list.addItem(os.path.basename(p))

    def _single_remove_videos(self) -> None:
        for item in self._single_video_list.selectedItems():
            row = self._single_video_list.row(item)
            self._single_video_list.takeItem(row)
            if row < len(self._single_video_paths):
                self._single_video_paths.pop(row)

    # ---- Stereo — photo mode -----------------------------------------------

    def _stereo_add_photos(self, side: str) -> None:
        lst = self._rgb_photo_list if side == "rgb" else self._th_photo_list
        paths_ref = self._rgb_paths if side == "rgb" else self._th_paths
        label = "RGB" if side == "rgb" else "Thermal"
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Select {label} Images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        for p in paths:
            if p not in paths_ref:
                paths_ref.append(p)
                lst.addItem(os.path.basename(p))

    def _stereo_remove_photos(self, side: str) -> None:
        lst = self._rgb_photo_list if side == "rgb" else self._th_photo_list
        paths_ref = self._rgb_paths if side == "rgb" else self._th_paths
        for item in lst.selectedItems():
            row = lst.row(item)
            lst.takeItem(row)
            if row < len(paths_ref):
                paths_ref.pop(row)

    # ---- Stereo — video mode -----------------------------------------------

    def _stereo_browse_video(self, side: str) -> None:
        label = "RGB" if side == "rgb" else "Thermal"
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {label} Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg)"
        )
        if not path:
            return
        if side == "rgb":
            self._rgb_video_path = path
            self._rgb_video_edit.setText(os.path.basename(path))
        else:
            self._th_video_path = path
            self._th_video_edit.setText(os.path.basename(path))

    def _load_initial_calib(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Initial Calibration JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Validate minimal structure
            has_rgb = bool(
                data.get("Wide") or data.get("RGB")
                or data.get("wide") or data.get("rgb")
            )
            has_th = bool(data.get("Thermal") or data.get("thermal"))
            if not (has_rgb and has_th):
                raise ValueError(
                    "File must contain 'Wide'/'RGB' and 'Thermal' sections."
                )
            self._initial_calib_data = data
            self._init_calib_lbl.setText(os.path.basename(path))
            self._init_calib_lbl.setStyleSheet("color: #88ff88;")
        except Exception as exc:
            QMessageBox.warning(self, "Invalid calibration file", str(exc))

    def _clear_initial_calib(self) -> None:
        self._initial_calib_data = None
        self._init_calib_lbl.setText("(none — will be estimated)")
        self._init_calib_lbl.setStyleSheet("color: #888;")

    # =========================================================================
    # Entering the calibration page
    # =========================================================================

    def _resolve_inputs(self) -> bool:
        """Resolve selected files into concrete image paths.

        For video mode this extracts frames and writes them to a temp directory.
        Returns True if everything is valid and ready, False if the user should
        fix their input selection.
        """
        if self._mode == "single":
            if self._input_type == "photo":
                if not self._single_paths:
                    QMessageBox.warning(self, "No input",
                                        "Please add at least one image.")
                    return False
                n = len(self._single_paths)
            else:  # video
                if not self._single_video_paths:
                    QMessageBox.warning(self, "No input",
                                        "Please add at least one video.")
                    return False
                if not _HAS_CV2:
                    QMessageBox.critical(self, "Missing dependency",
                                         "opencv-python is required to extract video frames.")
                    return False
                # Extract frames from every listed video
                central_only = self._sfm_central_only_chk.isChecked()
                n_videos = len(self._single_video_paths)
                n_per_video = 1 if central_only else self._sfm_frames_spin.value()
                total_frames = n_videos * n_per_video

                prog = QProgressDialog("Extracting video frames…", None, 0, total_frames, self)
                prog.setWindowTitle("Frame Extraction")
                prog.setWindowModality(Qt.WindowModal)
                prog.setMinimumDuration(0)
                prog.setValue(0)

                self._single_paths.clear()
                tmpdir = tempfile.mkdtemp(prefix="bambi_sfm_")
                extracted_so_far = 0
                for vi, vpath in enumerate(self._single_video_paths):
                    vname = os.path.basename(vpath)
                    prog.setLabelText(f"Extracting frames from {vname}…")
                    if central_only:
                        frame = _load_image_or_video_central(vpath)
                        if frame is not None:
                            out = os.path.join(
                                tmpdir,
                                os.path.splitext(vname)[0] + "_central.jpg"
                            )
                            cv2.imwrite(out, frame)
                            self._single_paths.append(out)
                        extracted_so_far += 1
                        prog.setValue(extracted_so_far)
                        QApplication.processEvents()
                    else:
                        def _cb(done, total, _base=extracted_so_far):
                            prog.setValue(_base + done)
                            QApplication.processEvents()
                        frames = _extract_n_frames_to_dir(
                            vpath, tmpdir, n_per_video, progress_cb=_cb
                        )
                        self._single_paths.extend(frames)
                        extracted_so_far += n_per_video
                        prog.setValue(extracted_so_far)
                        QApplication.processEvents()

                prog.close()

                if not self._single_paths:
                    QMessageBox.critical(self, "Frame extraction failed",
                                         "Could not extract any frames from the selected videos.")
                    return False
                n = len(self._single_paths)

            if n < _MIN_SFM_IMAGES:
                ans = QMessageBox.question(
                    self, "Too few images",
                    f"SfM calibration typically needs ≥{_MIN_SFM_IMAGES} images "
                    f"(you have {n}).\n\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ans != QMessageBox.Yes:
                    return False

        else:  # stereo
            if self._input_type == "photo":
                if not self._rgb_paths or not self._th_paths:
                    QMessageBox.warning(self, "No input",
                                        "Please add RGB and thermal images.")
                    return False
            else:  # video
                if not self._rgb_video_path or not self._th_video_path:
                    QMessageBox.warning(self, "No input",
                                        "Please select one RGB video and one thermal video.")
                    return False
                if not _HAS_CV2:
                    QMessageBox.critical(self, "Missing dependency",
                                         "opencv-python is required to extract video frames.")
                    return False
                # Extract central frame from each video into a temp dir
                tmpdir = tempfile.mkdtemp(prefix="bambi_stereo_")
                rgb_frame = _load_image_or_video_central(self._rgb_video_path)
                th_frame = _load_image_or_video_central(self._th_video_path)
                if rgb_frame is None or th_frame is None:
                    QMessageBox.critical(self, "Frame extraction failed",
                                         "Could not extract the central frame from one or "
                                         "both of the selected videos.")
                    return False
                rgb_out = os.path.join(
                    tmpdir,
                    os.path.splitext(os.path.basename(self._rgb_video_path))[0] + "_central.jpg"
                )
                th_out = os.path.join(
                    tmpdir,
                    os.path.splitext(os.path.basename(self._th_video_path))[0] + "_central.jpg"
                )
                cv2.imwrite(rgb_out, rgb_frame)
                cv2.imwrite(th_out, th_frame)
                self._rgb_paths = [rgb_out]
                self._th_paths = [th_out]

            n_pairs = min(len(self._rgb_paths), len(self._th_paths))
            if n_pairs == 0:
                QMessageBox.warning(self, "No pairs", "No image pairs available.")
                return False

        return True

    def _enter_calib_page(self) -> None:
        # Resolve inputs (extract video frames if needed) and validate
        if not self._resolve_inputs():
            return

        # Build the calibration page
        self._result = None
        new_page = (
            self._build_calib_page_single()
            if self._mode == "single"
            else self._build_calib_page_stereo()
        )
        # Replace the existing page 1
        old = self._pages.widget(1)
        self._pages.removeWidget(old)
        if old is not self._page_setup and old is not self._page_results:
            old.deleteLater()
        self._page_calib = new_page
        self._pages.insertWidget(1, new_page)

        if self._mode == "stereo":
            self._init_stereo_annotation()

        self._pages.setCurrentIndex(1)
        self._update_step_label()
        self._update_nav()

    # =========================================================================
    # Stereo annotation
    # =========================================================================

    def _on_magnifier_toggled(self, enabled: bool) -> None:
        """Enable/disable the magnifier loupe on both image panels."""
        self._rgb_img_lbl.set_magnifier_enabled(enabled)
        self._th_img_lbl.set_magnifier_enabled(enabled)

    def _init_stereo_annotation(self) -> None:
        n = min(len(self._rgb_paths), len(self._th_paths))
        self._pairs_annot = [{"rgb_pts": [], "th_pts": []} for _ in range(n)]
        self._pair_idx = 0
        self._load_pair(0)

    # ---- Point save / load ------------------------------------------------

    def _save_annot_points(self) -> None:
        """Save all annotated point pairs to a JSON file."""
        total = sum(
            min(len(p["rgb_pts"]), len(p["th_pts"])) for p in self._pairs_annot
        )
        if total == 0:
            QMessageBox.information(self, "No points", "No points have been placed yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation Points", "annot_points.json", "JSON (*.json)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"

        data = {
            "version": 1,
            "pairs": [
                {
                    "rgb_file": os.path.basename(self._rgb_paths[i]),
                    "th_file":  os.path.basename(self._th_paths[i]),
                    "rgb_points": [list(pt) for pt in pa["rgb_pts"]],
                    "th_points":  [list(pt) for pt in pa["th_pts"]],
                }
                for i, pa in enumerate(self._pairs_annot)
            ],
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            QMessageBox.information(self, "Saved", f"Points saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load_annot_points(self) -> None:
        """Load annotated point pairs from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotation Points", "", "JSON (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not read file:\n{exc}")
            return

        if data.get("version") != 1 or "pairs" not in data:
            QMessageBox.warning(
                self, "Invalid file",
                "This file does not appear to be a valid annotation points file."
            )
            return

        file_pairs = data["pairs"]
        n_file   = len(file_pairs)
        n_annot  = len(self._pairs_annot)

        if n_file != n_annot:
            ans = QMessageBox.question(
                self, "Pair count mismatch",
                f"The file contains {n_file} pair(s) but the current session has "
                f"{n_annot} pair(s).\n\n"
                f"Points will be loaded for the first "
                f"{min(n_file, n_annot)} pair(s).  Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        # Warn if filenames differ (informational only — load by position)
        mismatches = []
        for i in range(min(n_file, n_annot)):
            fp = file_pairs[i]
            if (fp.get("rgb_file") != os.path.basename(self._rgb_paths[i])
                    or fp.get("th_file") != os.path.basename(self._th_paths[i])):
                mismatches.append(i + 1)
        if mismatches:
            ans = QMessageBox.question(
                self, "Filename mismatch",
                f"Pair(s) {mismatches} have different filenames than expected.\n\n"
                "Points will still be loaded by position.  Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        # Restore points
        for i in range(min(n_file, n_annot)):
            fp = file_pairs[i]
            self._pairs_annot[i]["rgb_pts"] = [tuple(pt) for pt in fp.get("rgb_points", [])]
            self._pairs_annot[i]["th_pts"]  = [tuple(pt) for pt in fp.get("th_points",  [])]

        # Refresh the currently displayed pair
        self._load_pair(self._pair_idx)
        QMessageBox.information(self, "Loaded", f"Points loaded from:\n{path}")

    def _load_pair(self, idx: int) -> None:
        if not _HAS_CV2:
            return
        n = len(self._pairs_annot)
        rgb_img = _load_image_or_video_central(self._rgb_paths[idx])
        th_img = _load_image_or_video_central(self._th_paths[idx])
        if rgb_img is not None:
            self._rgb_img_lbl.load_image(rgb_img)
        if th_img is not None:
            self._th_img_lbl.load_image(th_img)

        # Restore saved points for this pair
        pa = self._pairs_annot[idx]
        self._rgb_img_lbl.set_points(pa["rgb_pts"])
        self._th_img_lbl.set_points(pa["th_pts"])

        self._pair_idx = idx
        self._pair_lbl.setText(f"Pair {idx + 1} / {n}")
        self._pair_prev_btn.setEnabled(idx > 0)
        self._pair_next_btn.setEnabled(idx < n - 1)
        self._refresh_annot_state()

    def _refresh_annot_state(self) -> None:
        pa = self._pairs_annot[self._pair_idx]
        n_rgb = len(pa["rgb_pts"])
        n_th  = len(pa["th_pts"])
        n_complete = min(n_rgb, n_th)

        # When counts are equal both panels are open — the user may start the
        # next pair from either side.  Once one side is ahead the lagging side
        # is the only one that accepts clicks until the pair is complete.
        self._rgb_img_lbl.set_accepting(n_rgb <= n_th)
        self._th_img_lbl.set_accepting(n_th  <= n_rgb)

        next_idx = n_complete + 1
        if n_rgb == n_th:
            msg = f"Click on either image to place point #{next_idx}"
        elif n_rgb > n_th:
            msg = f"Now click on Thermal to complete point pair #{next_idx}"
        else:
            msg = f"Now click on RGB to complete point pair #{next_idx}"
        self._annot_instr.setText(msg)

        total = sum(
            min(len(p["rgb_pts"]), len(p["th_pts"]))
            for p in self._pairs_annot
        )
        self._pt_count_lbl.setText(
            f"Points this pair: {n_complete}  |  Total: {total}"
        )

    def _on_rgb_point_placed(self, x: int, y: int) -> None:
        # Guard is redundant with set_accepting but kept as safety net
        pa = self._pairs_annot[self._pair_idx]
        if len(pa["rgb_pts"]) > len(pa["th_pts"]):
            return
        pa["rgb_pts"].append((x, y))
        self._refresh_annot_state()

    def _on_th_point_placed(self, x: int, y: int) -> None:
        pa = self._pairs_annot[self._pair_idx]
        if len(pa["th_pts"]) > len(pa["rgb_pts"]):
            return
        pa["th_pts"].append((x, y))
        self._refresh_annot_state()

    def _navigate_pair(self, delta: int) -> None:
        new_idx = self._pair_idx + delta
        if 0 <= new_idx < len(self._pairs_annot):
            self._load_pair(new_idx)

    def _undo_last_point(self) -> None:
        pa = self._pairs_annot[self._pair_idx]
        # Remove the last-placed point from whichever side has more
        if len(pa["rgb_pts"]) > len(pa["th_pts"]):
            pa["rgb_pts"].pop()
            self._rgb_img_lbl.set_points(pa["rgb_pts"])
        elif pa["th_pts"]:
            pa["th_pts"].pop()
            self._th_img_lbl.set_points(pa["th_pts"])
        elif pa["rgb_pts"]:
            pa["rgb_pts"].pop()
            self._rgb_img_lbl.set_points(pa["rgb_pts"])
        self._refresh_annot_state()

    def _clear_pair_points(self) -> None:
        pa = self._pairs_annot[self._pair_idx]
        pa["rgb_pts"].clear()
        pa["th_pts"].clear()
        self._rgb_img_lbl.clear_points()
        self._th_img_lbl.clear_points()
        self._refresh_annot_state()

    # =========================================================================
    # Running calibration
    # =========================================================================

    def _start_calibration(self) -> None:
        if self._worker_thread and self._worker_thread.isRunning():
            return

        self._run_calib_btn.setEnabled(False)
        self._calib_progress.setValue(0)
        self._calib_log.clear()
        self._result = None
        self._update_nav()

        if self._mode == "single":
            params = {
                "image_paths": self._single_paths,
                "camera_name": self._single_cam_name.text().strip() or "Camera",
            }
        else:
            # Collect all corresponding points
            T_pts, W_pts = [], []
            for pa in self._pairs_annot:
                n = min(len(pa["rgb_pts"]), len(pa["th_pts"]))
                for i in range(n):
                    W_pts.append(list(pa["rgb_pts"][i]))
                    T_pts.append(list(pa["th_pts"][i]))

            if len(T_pts) < _MIN_STEREO_POINTS:
                QMessageBox.warning(
                    self, "Insufficient points",
                    f"Please place at least {_MIN_STEREO_POINTS} corresponding "
                    f"point pairs across all image pairs.\n"
                    f"Currently: {len(T_pts)} pair(s)."
                )
                self._run_calib_btn.setEnabled(True)
                return

            # Build or estimate initial calibration
            if self._initial_calib_data:
                init_calib = self._initial_calib_data
            else:
                # Estimate from first image pair
                rgb_img = _load_image_or_video_central(self._rgb_paths[0])
                th_img = _load_image_or_video_central(self._th_paths[0])
                rgb_intr = _estimate_intrinsics_from_image(rgb_img) if rgb_img is not None else {
                    "ret": None, "mtx": [[1000, 0, 640], [0, 1000, 512], [0, 0, 1]],
                    "dist": [0] * 5, "name": "Wide"
                }
                th_intr = _estimate_intrinsics_from_image(th_img) if th_img is not None else {
                    "ret": None, "mtx": [[800, 0, 320], [0, 800, 256], [0, 0, 1]],
                    "dist": [0] * 5, "name": "Thermal"
                }
                rgb_intr["name"] = "Wide"
                th_intr["name"] = "Thermal"
                init_calib = {"Wide": rgb_intr, "Thermal": th_intr}

            params = {
                "T_points": T_pts,
                "W_points": W_pts,
                "initial_calibration": init_calib,
            }

        self._worker = _CalibWorker(self._mode, params)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_calib_finished)
        self._worker.error.connect(self._on_calib_error)
        self._worker.progress.connect(self._calib_progress.setValue)
        self._worker.log.connect(self._append_log)
        self._worker_thread.start()

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.cancel()
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        self._worker = None
        self._worker_thread = None

    def _append_log(self, msg: str) -> None:
        self._calib_log.append(msg)

    def _on_calib_finished(self, result: dict) -> None:
        self._worker_thread.quit()
        self._worker_thread.wait()
        self._worker = None
        self._worker_thread = None
        self._result = result
        self._run_calib_btn.setEnabled(True)
        self._populate_results_page(result)
        self._update_nav()
        # Auto-advance to results
        QTimer.singleShot(300, lambda: (
            self._pages.setCurrentIndex(2),
            self._update_step_label(),
            self._update_nav(),
        ))

    def _on_calib_error(self, msg: str) -> None:
        self._worker_thread.quit()
        self._worker_thread.wait()
        self._worker = None
        self._worker_thread = None
        self._run_calib_btn.setEnabled(True)
        self._calib_log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Calibration failed", msg)
        self._update_nav()

    # =========================================================================
    # Results page
    # =========================================================================

    def _populate_results_page(self, result: dict) -> None:
        mode = result.get("mode", self._mode)

        if mode == "single":
            ret = result.get("ret")
            ret_str = f"{ret:.4f} px" if ret is not None else "n/a"
            n_img = result.get("n_images", "?")
            n_pts = result.get("n_points3d", "?")
            metrics = (
                f"SfM calibration complete\n"
                f"  Registered images : {n_img}\n"
                f"  3D points          : {n_pts}\n"
                f"  Mean reprojection  : {ret_str}"
            )
            self._result_metrics_lbl.setText(metrics)

            mtx = result["mtx"]
            dist = result["dist"]
            txt = (
                f"Camera: {result.get('camera_name', 'Camera')}\n\n"
                f"Intrinsic matrix (K):\n"
                f"  fx = {mtx[0][0]:.4f}   fy = {mtx[1][1]:.4f}\n"
                f"  cx = {mtx[0][2]:.4f}   cy = {mtx[1][2]:.4f}\n\n"
                f"Distortion coefficients:\n"
                f"  k1={dist[0]:.6f}  k2={dist[1]:.6f}\n"
                f"  p1={dist[2]:.6f}  p2={dist[3]:.6f}  k3={dist[4]:.6f}\n\n"
                f"JSON preview:\n"
                f"{json.dumps({'ret': result.get('ret'), 'mtx': mtx, 'dist': dist, 'name': result.get('camera_name', 'Camera')}, indent=2)}"
            )
            self._result_text.setText(txt)

            # Suggest save path
            cam = result.get("camera_name", "camera").replace(" ", "_").lower()
            self._save_path_edit.setPlaceholderText(f"calibration_{cam}.json")
            self._save_single_widget.setVisible(True)
            self._save_stereo_widget.setVisible(False)

        else:  # stereo
            hom_mse = result.get("homography_mse", 0.0)
            opt_mse = result.get("optimized_mse", 0.0)
            metrics = (
                f"Stereo calibration complete\n"
                f"  Homography MSE (px²) : {hom_mse:.4f}\n"
                f"  Optimised MSE (px²)  : {opt_mse:.6f}"
            )
            self._result_metrics_lbl.setText(metrics)

            th = result["Thermal"]
            wi = result["Wide"]
            th_mtx = th["mtx"]
            wi_mtx = wi["mtx"]

            def _fmt_cam(name, d):
                m = d["mtx"]
                dist = d["dist"]
                return (
                    f"{name}:\n"
                    f"  fx={m[0][0]:.4f}  fy={m[1][1]:.4f}  cx={m[0][2]:.4f}  cy={m[1][2]:.4f}\n"
                    f"  k1={dist[0]:.6f}  k2={dist[1]:.6f}  "
                    f"p1={dist[2]:.6f}  p2={dist[3]:.6f}  k3={dist[4]:.6f}\n"
                )

            txt = (
                _fmt_cam("Thermal (optimised)", th)
                + "\n"
                + _fmt_cam("Wide / RGB (reference — unchanged)", wi)
                + "\nJSON preview:\n"
                + json.dumps({"Thermal": th, "Wide": wi}, indent=2)
            )
            self._result_text.setText(txt)
            self._save_single_widget.setVisible(False)
            self._save_stereo_widget.setVisible(True)
            self._save_th_path_edit.setPlaceholderText("calibration_thermal.json")
            self._save_rgb_path_edit.setPlaceholderText("calibration_rgb.json")

        QTimer.singleShot(50, lambda: self._populate_result_preview(result))

    # ------------------------------------------------------------------

    def _populate_result_preview(self, result: dict) -> None:
        """Render the visual preview for the results page."""
        if not _HAS_CV2:
            self._result_preview_grp.setVisible(False)
            return

        mode = result.get("mode", self._mode)

        if mode == "single":
            self._preview_single_widget.setVisible(True)
            self._preview_stereo_widget.setVisible(False)

            # Pick first available input image
            img_path = self._single_paths[0] if self._single_paths else None
            if img_path is None or not os.path.isfile(img_path):
                self._preview_orig_lbl.setText("(no image available)")
                self._preview_undist_lbl.setText("(no image available)")
                return

            img = cv2.imread(img_path)
            if img is None:
                self._preview_orig_lbl.setText("(could not load image)")
                self._preview_undist_lbl.setText("(could not load image)")
                return

            mtx = np.array(result["mtx"], dtype=np.float64)
            dist = np.array(result["dist"], dtype=np.float64)
            h, w = img.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0)
            undist = cv2.undistort(img, mtx, dist, None, new_mtx)
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                undist = undist[y:y + rh, x:x + rw]

            self._set_preview_pixmap(self._preview_orig_lbl, img)
            self._set_preview_pixmap(self._preview_undist_lbl, undist)

        else:  # stereo
            self._preview_single_widget.setVisible(False)
            self._preview_stereo_widget.setVisible(True)

            # Need at least one RGB+thermal pair
            if not self._rgb_paths or not self._th_paths:
                self._preview_overlay_lbl.setText("(no images available)")
                return

            rgb_img = cv2.imread(self._rgb_paths[0])
            th_img = cv2.imread(self._th_paths[0])
            if rgb_img is None or th_img is None:
                self._preview_overlay_lbl.setText("(could not load images)")
                return

            th = result["Thermal"]
            wi = result["Wide"]
            th_mtx = np.array(th["mtx"], dtype=np.float64)
            th_dist = np.array(th["dist"], dtype=np.float64)
            wi_mtx = np.array(wi["mtx"], dtype=np.float64)
            wi_dist = np.array(wi["dist"], dtype=np.float64)

            # Use RGB camera matrix as shared output space
            dst_w = int(wi_mtx[0][2] * 2)
            dst_h = int(wi_mtx[1][2] * 2)
            dst_size = (dst_w, dst_h)

            def _remap(img, cam_mtx, cam_dist):
                mapx, mapy = cv2.initUndistortRectifyMap(
                    cam_mtx, cam_dist, None, wi_mtx, dst_size, cv2.CV_32FC1
                )
                return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

            rgb_undist = _remap(rgb_img, wi_mtx, wi_dist)
            th_undist = _remap(th_img, th_mtx, th_dist)

            # Resize thermal to match RGB output size if needed
            if th_undist.shape[:2] != (dst_h, dst_w):
                th_undist = cv2.resize(th_undist, dst_size)

            # Canny edges from RGB → inject into green channel of thermal
            rgb_gray = cv2.cvtColor(rgb_undist, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(rgb_gray, 100, 200)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            overlay = th_undist.copy()
            overlay[:, :, 1] = edges_bgr[:, :, 1]  # green channel = RGB edges

            self._set_preview_pixmap(self._preview_overlay_lbl, overlay)

        # --- Feature coverage heatmap ---
        self._populate_coverage_heatmap(result)

    def _populate_coverage_heatmap(self, result: dict) -> None:
        """Compute and render the feature coverage heatmap."""
        _COLS, _ROWS = 8, 5

        mode = result.get("mode", self._mode)

        if mode == "single":
            # Detect SIFT keypoints across all calibration images
            try:
                sift = cv2.SIFT_create()
            except AttributeError:
                self._heatmap_grp.setVisible(False)
                return

            counts = np.zeros((_ROWS, _COLS), dtype=int)
            img_w = img_h = 0
            total_kp = 0

            for path in self._single_paths:
                gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                if img_w == 0:
                    img_h, img_w = gray.shape[:2]
                kps = sift.detect(gray, None)
                total_kp += len(kps)
                for kp in kps:
                    c = min(int(kp.pt[0] / img_w * _COLS), _COLS - 1)
                    r = min(int(kp.pt[1] / img_h * _ROWS), _ROWS - 1)
                    counts[r, c] += 1

            subtitle = (
                f"SIFT keypoints aggregated across {len(self._single_paths)} image(s) "
                f"— {total_kp:,} total detections"
            )

        else:  # stereo — use annotation correspondence points
            # Determine image dimensions from the first RGB image
            img_w = img_h = 0
            if self._rgb_paths:
                probe = cv2.imread(self._rgb_paths[0])
                if probe is not None:
                    img_h, img_w = probe.shape[:2]
            if img_w == 0:
                # Fall back to cx/cy from the Wide camera matrix
                wi_mtx = result.get("Wide", {}).get("mtx", [[0, 0, 0]] * 3)
                img_w = int(wi_mtx[0][2] * 2)
                img_h = int(wi_mtx[1][2] * 2)
            if img_w == 0:
                self._heatmap_grp.setVisible(False)
                return

            counts = np.zeros((_ROWS, _COLS), dtype=int)
            total_pts = 0
            for pa in self._pairs_annot:
                for pt in pa["rgb_pts"]:
                    c = min(int(pt[0] / img_w * _COLS), _COLS - 1)
                    r = min(int(pt[1] / img_h * _ROWS), _ROWS - 1)
                    counts[r, c] += 1
                    total_pts += 1

            n_pairs = len(self._pairs_annot)
            subtitle = (
                f"RGB correspondence points across {n_pairs} image pair(s) "
                f"— {total_pts} total points"
            )

        self._heatmap_grp.setVisible(True)
        pm = self._render_coverage_heatmap(counts)
        w = self._heatmap_lbl.width() if self._heatmap_lbl.width() > 10 else 600
        self._heatmap_lbl.setPixmap(
            pm.scaled(w, 9999, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._heatmap_subtitle.setText(subtitle)

    @staticmethod
    def _render_coverage_heatmap(counts: "np.ndarray") -> QPixmap:
        """Render *counts* (2-D int array, shape ROWS×COLS) as a colored grid QPixmap."""
        rows, cols = counts.shape
        cell_w, cell_h = 72, 52
        margin = 2
        pm_w = cols * cell_w + 2 * margin
        pm_h = rows * cell_h + 2 * margin

        pm = QPixmap(pm_w, pm_h)
        pm.fill(QColor(30, 30, 30))

        max_count = max(1, int(counts.max()))

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, False)

        cell_font = painter.font()
        cell_font.setPointSize(9)
        cell_font.setBold(True)
        painter.setFont(cell_font)

        for r in range(rows):
            for c in range(cols):
                count = int(counts[r, c])
                t = count / max_count          # 0..1
                hue = int(t * 120)             # 0=red, 60=yellow, 120=green
                sat = 210 if count > 0 else 0
                val = 160 if count > 0 else 55
                cell_color = QColor.fromHsv(hue, sat, val)

                x = margin + c * cell_w
                y = margin + r * cell_h
                painter.fillRect(x, y, cell_w - 1, cell_h - 1, cell_color)

                # Count label
                text_color = QColor(255, 255, 255) if count > 0 else QColor(100, 100, 100)
                painter.setPen(text_color)
                painter.drawText(x, y, cell_w - 1, cell_h - 1, Qt.AlignCenter, str(count))

        # Grid lines
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for c in range(cols + 1):
            x = margin + c * cell_w
            painter.drawLine(x, margin, x, pm_h - margin)
        for r in range(rows + 1):
            y = margin + r * cell_h
            painter.drawLine(margin, y, pm_w - margin, y)

        painter.end()
        return pm

    @staticmethod
    def _set_preview_pixmap(label: QLabel, img: "np.ndarray") -> None:
        """Scale *img* to fit *label* and display it."""
        pm = _bgr_to_qpixmap(img)
        w = label.width() if label.width() > 10 else 480
        h = label.height() if label.height() > 10 else 300
        label.setPixmap(pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _browse_save_path(self, target: str = "single") -> None:
        if target == "thermal":
            default = "calibration_thermal.json"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Thermal Calibration JSON", default, "JSON (*.json)"
            )
            if path:
                if not path.endswith(".json"):
                    path += ".json"
                self._save_th_path_edit.setText(path)
        elif target == "rgb":
            default = "calibration_rgb.json"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save RGB Calibration JSON", default, "JSON (*.json)"
            )
            if path:
                if not path.endswith(".json"):
                    path += ".json"
                self._save_rgb_path_edit.setText(path)
        else:
            default = "calibration.json"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Calibration JSON", default, "JSON (*.json)"
            )
            if path:
                if not path.endswith(".json"):
                    path += ".json"
                self._save_path_edit.setText(path)

    def _save_calibration(self) -> None:
        if not self._result:
            QMessageBox.warning(self, "No result", "Run calibration first.")
            return

        mode = self._result.get("mode", self._mode)

        if mode == "single":
            path = self._save_path_edit.text().strip()
            if not path:
                self._browse_save_path("single")
                path = self._save_path_edit.text().strip()
            if not path:
                return
            if not path.endswith(".json"):
                path += ".json"
            data = {
                "ret": self._result.get("ret"),
                "mtx": self._result["mtx"],
                "dist": self._result["dist"],
                "name": self._result.get("camera_name", "Camera"),
            }
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2)
                QMessageBox.information(self, "Saved", f"Calibration saved to:\n{path}")
                self.calibrationSaved.emit(path)
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))

        else:  # stereo — two separate files
            th_path = self._save_th_path_edit.text().strip()
            rgb_path = self._save_rgb_path_edit.text().strip()
            if not th_path:
                self._browse_save_path("thermal")
                th_path = self._save_th_path_edit.text().strip()
            if not th_path:
                return
            if not rgb_path:
                self._browse_save_path("rgb")
                rgb_path = self._save_rgb_path_edit.text().strip()
            if not rgb_path:
                return
            if not th_path.endswith(".json"):
                th_path += ".json"
            if not rgb_path.endswith(".json"):
                rgb_path += ".json"

            th_data = {
                "ret": self._result["Thermal"].get("ret"),
                "mtx": self._result["Thermal"]["mtx"],
                "dist": self._result["Thermal"]["dist"],
                "name": "Thermal",
            }
            rgb_data = {
                "ret": self._result["Wide"].get("ret"),
                "mtx": self._result["Wide"]["mtx"],
                "dist": self._result["Wide"]["dist"],
                "name": "Wide",
            }
            try:
                for path, data in ((th_path, th_data), (rgb_path, rgb_data)):
                    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2)
                    self.calibrationSaved.emit(path)
                QMessageBox.information(
                    self, "Saved",
                    f"Thermal calibration saved to:\n{th_path}\n\nRGB calibration saved to:\n{rgb_path}"
                )
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))

    # =========================================================================
    # Cleanup
    # =========================================================================

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)

    def reject(self) -> None:
        self._stop_worker()
        super().reject()
