# -*- coding: utf-8 -*-
"""
BAMBI Correction Wizard
=======================
Three-step dialog for computing and storing camera correction factors.

Step 1 — Point Selection
    Side-by-side frame views; the user places one corresponding ground point
    in each view.

Step 2 — Geo-referencing & Calibration
    * The selected pixels are projected onto the DEM to obtain world
      coordinates.  The camera XY position is the circle centre; the XY
      distance from camera to world point is the radius.
    * An automatic z-offset probe finds the z-translation where the two
      circles change from non-intersecting to intersecting.
    * A coarse sweep then finds the yaw correction (rotation Z) that
      minimises the distance between the two geo-referenced points.
    * All six correction values can be fine-tuned manually with live plot
      feedback.

Step 3 — Light-field Preview & Save
    Renders an ALFS integral image from the selected frames (+ optional
    neighbour range) with the found correction.  The correction can be
    written to ``correction.json`` as a global default or as a local
    frame-range entry.

The DEM (``read_gltf``) is loaded once when the wizard opens and released
when it closes.
"""

import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from qgis.PyQt.QtCore import Qt, pyqtSignal, QThread, QPointF, QRect, QTimer
from qgis.PyQt.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)
from qgis.PyQt.QtGui import (
    QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_qpixmap(img: "np.ndarray") -> QPixmap:
    """Convert a BGR (or grayscale) numpy array to QPixmap."""
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _wrap_deg(v: float) -> float:
    """Wrap *v* (degrees) to (−180, +180]."""
    return ((v + 180.0) % 360.0) - 180.0


def _wrap_rad(v: float) -> float:
    """Wrap *v* (radians) to (−π, +π]."""
    return ((v + math.pi) % (2.0 * math.pi)) - math.pi


def _circles_intersect(c1, r1: float, c2, r2: float) -> bool:
    """Return True when the two circles share at least one point."""
    d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    return abs(r1 - r2) <= d <= r1 + r2


# ---------------------------------------------------------------------------
# ClickableImageLabel
# ---------------------------------------------------------------------------

class ClickableImageLabel(QLabel):
    """QLabel subclass that lets the user click to place a single cross-hair point.

    Emits ``pointSet(norm_x, norm_y)`` with coordinates in [0, 1] relative to
    the image (not the widget).
    """

    pointSet = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 200)
        self.setStyleSheet("border: 1px solid #555; background: #12122a;")

        self._src_pixmap: Optional[QPixmap] = None
        self._point: Optional[Tuple[float, float]] = None
        self._img_rect: Optional[QRect] = None

    # ------------------------------------------------------------------
    def set_image(self, img_bgr: "np.ndarray") -> None:
        self._src_pixmap = _bgr_to_qpixmap(img_bgr)
        self._point = None
        self._recompute_rect()
        self.update()

    def clear_image(self) -> None:
        self._src_pixmap = None
        self._point = None
        self._img_rect = None
        self.setText("Load a frame to begin.")
        self.update()

    def get_point(self) -> Optional[Tuple[float, float]]:
        """Return (norm_x, norm_y) in [0, 1] or None."""
        return self._point

    # ------------------------------------------------------------------
    def _recompute_rect(self) -> None:
        if self._src_pixmap is None:
            self._img_rect = None
            return
        scaled = self._src_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._img_rect = QRect(x, y, scaled.width(), scaled.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recompute_rect()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._img_rect is None or self._src_pixmap is None:
            return
        ix = event.x() - self._img_rect.left()
        iy = event.y() - self._img_rect.top()
        if not (0 <= ix < self._img_rect.width() and 0 <= iy < self._img_rect.height()):
            return
        norm_x = ix / self._img_rect.width()
        norm_y = iy / self._img_rect.height()
        self._point = (norm_x, norm_y)
        self.update()
        self.pointSet.emit(norm_x, norm_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 42))
        if self._src_pixmap and self._img_rect:
            scaled = self._src_pixmap.scaled(
                self._img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            painter.drawPixmap(self._img_rect.topLeft(), scaled)
        if self._point and self._img_rect:
            px = self._img_rect.left() + self._point[0] * self._img_rect.width()
            py = self._img_rect.top() + self._point[1] * self._img_rect.height()
            pen = QPen(QColor(255, 60, 60), 2)
            painter.setPen(pen)
            arm = 14
            painter.drawLine(int(px - arm), int(py), int(px + arm), int(py))
            painter.drawLine(int(px), int(py - arm), int(px), int(py + arm))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(px, py), 6.0, 6.0)
        painter.end()


# ---------------------------------------------------------------------------
# CirclePlotWidget
# ---------------------------------------------------------------------------

class CirclePlotWidget(QWidget):
    """Draws two circles (camera-to-point projections) and the world points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #12122a; border: 1px solid #555;")
        self._d: Optional[dict] = None

    def set_data(
        self,
        c1, r1: float,
        c2, r2: float,
        p1=None, p2=None,
        intersects: bool = False,
    ) -> None:
        self._d = dict(c1=c1, r1=r1, c2=c2, r2=r2,
                       p1=p1, p2=p2, intersects=intersects)
        self.update()

    def clear(self) -> None:
        self._d = None
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(18, 18, 42))

        if self._d is None:
            painter.setPen(QColor(140, 140, 160))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Run the probe to see the circles.")
            painter.end()
            return

        d = self._d
        margin = 24
        pw = self.width() - 2 * margin
        ph = self.height() - 2 * margin

        # World bounding box
        all_x = [d['c1'][0], d['c2'][0]]
        all_y = [d['c1'][1], d['c2'][1]]
        for p in (d['p1'], d['p2']):
            if p is not None:
                all_x.append(p[0])
                all_y.append(p[1])
        for c, r in ((d['c1'], d['r1']), (d['c2'], d['r2'])):
            all_x += [c[0] - r, c[0] + r]
            all_y += [c[1] - r, c[1] + r]

        wx_min, wx_max = min(all_x), max(all_x)
        wy_min, wy_max = min(all_y), max(all_y)
        wx_rng = max(wx_max - wx_min, 1e-6)
        wy_rng = max(wy_max - wy_min, 1e-6)

        def sx(wx):
            return margin + (wx - wx_min) / wx_rng * pw

        def sy(wy):
            return self.height() - margin - (wy - wy_min) / wy_rng * ph

        def sr(r):
            return max(1.0, r / wx_rng * pw)

        circle_colors = [QColor(80, 140, 255), QColor(255, 140, 80)]
        point_colors = [QColor(80, 220, 80), QColor(220, 220, 60)]
        intersects = d['intersects']
        line_style = Qt.SolidLine if intersects else Qt.DashLine

        for (c, r, col) in [
            (d['c1'], d['r1'], circle_colors[0]),
            (d['c2'], d['r2'], circle_colors[1]),
        ]:
            cxs, cys = sx(c[0]), sy(c[1])
            rs = sr(r)
            pen = QPen(col, 2, line_style)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cxs, cys), rs, rs)
            painter.setBrush(QBrush(col))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cxs, cys), 4.0, 4.0)

        for p, col in zip((d['p1'], d['p2']), point_colors):
            if p is None:
                continue
            pxs, pys = sx(p[0]), sy(p[1])
            arm = 7
            painter.setPen(QPen(col, 2))
            painter.setBrush(Qt.NoBrush)
            # Draw an X
            painter.drawLine(int(pxs - arm), int(pys - arm),
                             int(pxs + arm), int(pys + arm))
            painter.drawLine(int(pxs + arm), int(pys - arm),
                             int(pxs - arm), int(pys + arm))

        # Status text
        status_color = QColor(80, 220, 80) if intersects else QColor(220, 80, 80)
        painter.setPen(status_color)
        painter.setFont(QFont("monospace", 8))
        status_text = "● Circles intersect" if intersects else "○ Circles do NOT intersect"
        painter.drawText(margin, margin + 12, status_text)
        painter.end()


# ---------------------------------------------------------------------------
# Worker: DEM / Trimesh loading
# ---------------------------------------------------------------------------

class _DemLoadWorker(QThread):
    """Loads the DEM (.glb / .gltf) in a background thread.

    Emits ``finished(tri_mesh, raw_mesh_data)`` on success or
    ``error(message)`` on failure.
    """

    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, dem_path: str):
        super().__init__()
        self._dem_path = dem_path

    def run(self):
        try:
            from alfspy.render.render import read_gltf
            from trimesh import Trimesh

            mesh_data, _ = read_gltf(self._dem_path)
            tri_mesh = Trimesh(
                vertices=mesh_data.vertices, faces=mesh_data.indices
            )
            # Pre-build the ray-casting BVH so the first pixel_to_world_coord
            # call on the main thread returns immediately instead of blocking.
            try:
                _ = tri_mesh.triangles_tree
            except Exception:
                pass
            self.finished.emit(tri_mesh, mesh_data)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Worker: z-offset probe + rotation alignment
# ---------------------------------------------------------------------------

class _ProbeWorker(QThread):
    """Runs z-offset probe and coarse yaw alignment in a background thread.

    All Qt-widget data is collected by the caller *before* starting the
    thread and passed as plain Python objects so the thread never touches
    any QWidget.

    Constructor parameters
    ----------------------
    tri_mesh    : Trimesh   – pre-built mesh (BVH must already be constructed)
    side_params : list[dict]
        Two dicts, one per side, each containing:
          'point'      : (norm_x, norm_y) in [0, 1]
          'frame_type' : 't' or 'w'
          'frame_idx'  : int
          'img_size'   : (width, height) in pixels
          'poses'      : poses dict  (may be None)
    correction  : dict   – {'translation': {x,y,z}, 'rotation': {x,y,z}}
    max_steps   : int
    """

    status   = pyqtSignal(str)   # live progress text for the UI label
    finished = pyqtSignal(dict)  # resulting correction dict
    error    = pyqtSignal(str)

    def __init__(self, tri_mesh, side_params, correction, max_steps, parent=None):
        super().__init__(parent)
        self._tri_mesh   = tri_mesh
        self._sides      = side_params
        self._correction = correction
        self._max_steps  = max_steps

    # ------------------------------------------------------------------
    def run(self):
        try:
            corr = {
                'translation': dict(self._correction['translation']),
                'rotation':    dict(self._correction['rotation']),
            }

            self.status.emit("Step 1 / 2  —  Probing z-offset…")
            new_tz = self._probe_z(corr)
            corr['translation']['z'] = new_tz

            self.status.emit(
                f"Step 2 / 2  —  Scanning yaw  (tz = {new_tz:.2f})…"
            )
            new_rz = self._find_best_rz(corr)
            corr['rotation']['z'] = new_rz

            self.finished.emit(corr)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Geo-referencing (thread-safe: reads pre-built BVH, no Qt objects)
    # ------------------------------------------------------------------

    def _geo_ref(self, side_idx: int, correction: dict):
        try:
            import math as _m
            from pyrr import Quaternion, Vector3
            from alfspy.core.rendering import Camera
            from alfspy.core.convert.convert import pixel_to_world_coord
        except ImportError:
            return None

        s = self._sides[side_idx]
        pt     = s['point']
        img_w, img_h = s['img_size']
        poses  = s['poses']
        fi     = s['frame_idx']

        if poses is None:
            return None
        images = poses.get('images', [])
        if fi >= len(images):
            return None
        pose = images[fi]

        loc     = pose.get('location', [0.0, 0.0, 0.0])
        rot     = pose.get('rotation', [0.0, 0.0, 0.0])
        fov_raw = pose.get('fovy', 50.0)
        if isinstance(fov_raw, (list, tuple)):
            fov_raw = fov_raw[0] if fov_raw else 50.0
        fovy = float(fov_raw)

        t = correction['translation']
        r = correction['rotation']
        cor_t = Vector3([t['x'], t['y'], t['z']], dtype='f4')
        cor_r = Vector3([r['x'], r['y'], r['z']], dtype='f4')
        position  = Vector3(loc, dtype='f4') + cor_t
        cam_euler = Vector3([_m.radians(_wrap_deg(v)) for v in rot[:3]], dtype='f4')
        rot_euler = (cam_euler - cor_r) * -1.0
        cam_quat  = Quaternion.from_eulers(rot_euler)
        camera    = Camera(fovy=fovy, aspect_ratio=1.0,
                           position=position, rotation=cam_quat)

        px, py = pt[0] * img_w, pt[1] * img_h
        results = pixel_to_world_coord(
            [px], [py], img_w, img_h,
            self._tri_mesh, camera, include_misses=False,
        )
        if results is None or len(results) == 0 or results[0] is None:
            return None
        wp = results[0]
        return (float(wp[0]), float(wp[1]), float(wp[2]))

    def _camera_xy(self, side_idx: int, correction: dict):
        s = self._sides[side_idx]
        poses = s['poses']
        fi    = s['frame_idx']
        if poses is None:
            return None
        images = poses.get('images', [])
        if fi >= len(images):
            return None
        loc = images[fi].get('location', [0.0, 0.0, 0.0])
        t   = correction['translation']
        return (loc[0] + t['x'], loc[1] + t['y'])

    def _compute_circles(self, correction: dict):
        c1 = self._camera_xy(0, correction)
        c2 = self._camera_xy(1, correction)
        p1 = self._geo_ref(0, correction)
        p2 = self._geo_ref(1, correction)
        if any(v is None for v in (c1, c2, p1, p2)):
            return None
        r1 = math.sqrt((p1[0] - c1[0]) ** 2 + (p1[1] - c1[1]) ** 2)
        r2 = math.sqrt((p2[0] - c2[0]) ** 2 + (p2[1] - c2[1]) ** 2)
        return (c1, r1, (p1[0], p1[1])), (c2, r2, (p2[0], p2[1]))

    # ------------------------------------------------------------------
    def _probe_z(self, correction: dict) -> float:
        tz = correction['translation']['z']

        def _ok(z: float) -> bool:
            c = {
                'translation': {**correction['translation'], 'z': z},
                'rotation':    dict(correction['rotation']),
            }
            circles = self._compute_circles(c)
            if circles is None:
                return False
            (c1, r1, _), (c2, r2, _) = circles
            return _circles_intersect(c1, r1, c2, r2)

        if not _ok(tz):
            for step in range(self._max_steps):
                tz += 1.0
                self.status.emit(
                    f"Step 1 / 2  —  Probing z-offset…  "
                    f"(step {step + 1} / {self._max_steps},  tz = {tz:.1f})"
                )
                if _ok(tz):
                    break
        else:
            for step in range(self._max_steps):
                tz -= 1.0
                self.status.emit(
                    f"Step 1 / 2  —  Finding z boundary…  "
                    f"(step {step + 1} / {self._max_steps},  tz = {tz:.1f})"
                )
                if not _ok(tz):
                    tz += 1.0
                    break
        return tz

    def _find_best_rz(self, correction: dict) -> float:
        best_rz   = correction['rotation']['z']
        best_dist = float('inf')
        angles    = np.linspace(0.0, 2 * math.pi, 360)

        for i, rz in enumerate(angles):
            if i % 36 == 0:
                self.status.emit(
                    f"Step 2 / 2  —  Scanning yaw…  ({i} / 360)"
                )
            c = {
                'translation': dict(correction['translation']),
                'rotation':    {**correction['rotation'], 'z': float(rz)},
            }
            p1 = self._geo_ref(0, c)
            p2 = self._geo_ref(1, c)
            if p1 is None or p2 is None:
                continue
            d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_rz   = float(rz)

        return _wrap_rad(best_rz)


# ---------------------------------------------------------------------------
# Worker: light-field rendering
# ---------------------------------------------------------------------------

class _LightFieldRenderWorker(QThread):
    """Renders an ALFS integral image (orthographic overview) in a background thread.

    ``params`` dict keys:
        dem_path    : str   – path to DEM file
        frames      : list  – [{'image_path', 'location', 'rotation', 'fovy'}, ...]
        correction  : dict  – {'translation': {x,y,z}, 'rotation': {x,y,z}}
        render_size : int   – output image side length in pixels (default 1024)
    """

    finished = pyqtSignal(object)   # numpy RGBA uint8 array
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, params: dict):
        super().__init__()
        self._p = params

    def run(self):
        try:
            result = self._render()
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _render(self):
        import math as _m
        import numpy as _np
        from pyrr import Quaternion, Vector3
        from alfspy.render.render import (
            read_gltf, process_render_data, make_mgl_context,
            make_camera, make_shot_loader, release_all,
        )
        from alfspy.render.data import BaseSettings, CameraPositioningMode
        from alfspy.core.rendering.renderer import Renderer
        from alfspy.core.rendering import CtxShot, Resolution, TextureData
        from alfspy.core.geo.transform import Transform
        from alfspy.core.util.pyrrs import quaternion_from_eulers
        from alfspy.core.util.geo import get_aabb

        p = self._p
        corr = p['correction']
        tx = corr['translation']['x']
        ty = corr['translation']['y']
        tz = corr['translation']['z']
        rx = corr['rotation']['x']
        ry = corr['rotation']['y']
        rz = corr['rotation']['z']

        cor_translation = Vector3([tx, ty, tz], dtype='f4')
        cor_rot_eulers = Vector3([rx, ry, rz], dtype='f4')
        cor_quat = Quaternion.from_eulers(cor_rot_eulers)
        cor_transform = Transform(cor_translation, cor_quat)

        self.progress.emit(5)

        # Load and process DEM
        mesh_data, texture_data = read_gltf(p['dem_path'])
        mesh_data, texture_data = process_render_data(mesh_data, texture_data)
        mesh_aabb = get_aabb(mesh_data.vertices)

        self.progress.emit(25)

        ctx = make_mgl_context()

        # Load undistortion mask if available
        mask = None
        mask_path = p.get('mask_path')
        if mask_path and os.path.isfile(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                mask = TextureData(CtxShot._cvt_img(mask_img))

        shots = []
        default_fov = 50.0

        for fi in p['frames']:
            img_path = fi['image_path']
            if not os.path.isfile(img_path):
                continue
            loc = fi['location']
            rot = fi['rotation']
            fovy = fi.get('fovy', default_fov)
            if isinstance(fovy, (list, tuple)):
                fovy = fovy[0] if fovy else default_fov

            cam_pos = Vector3(loc, dtype='f4')
            # Rotation values are in degrees in the poses file (matches alfs.py)
            eulers = [_np.deg2rad(_wrap_deg(v)) for v in rot[:3]]
            cam_rot = quaternion_from_eulers(eulers, 'zyx')

            try:
                shot = CtxShot(
                    ctx, img_path, cam_pos, cam_rot,
                    fovy, 1, cor_transform, lazy=False,
                )
                shots.append(shot)
            except Exception:
                pass

        if not shots:
            raise RuntimeError(
                "No valid shots could be loaded. "
                "Check that the frame images exist at the expected paths."
            )

        self.progress.emit(55)

        render_sz = p.get('render_size', 1024)

        # Build camera the same way alfs.py does: fixed world-unit ortho_size,
        # camera positioned above the centre of all shots, looking straight down.
        settings = BaseSettings(
            count=len(shots),
            initial_skip=0,
            add_background=False,
            camera_dist=10.0,
            camera_position_mode=CameraPositioningMode.ShotCentered,
            fovy=default_fov,
            aspect_ratio=1.0,
            orthogonal=True,
            ortho_size=(70, 70),
            correction=cor_transform,
            resolution=Resolution(render_sz, render_sz),
        )

        camera = make_camera(mesh_aabb, shots, settings, rotation=Quaternion())
        cam_cx = float(camera.transform.position.x)
        cam_cy = float(camera.transform.position.y)

        renderer = Renderer(settings.resolution, ctx, camera, mesh_data, texture_data)
        shot_loader = make_shot_loader(shots)
        result = renderer.render_integral(
            shot_loader,
            mask=mask,
            save=False,
            release_shots=False,
            auto_contrast=True,
            alpha_threshold=0.5,
        )

        self.progress.emit(95)
        renderer.release()
        release_all(ctx, shots)
        self.progress.emit(100)
        return {
            'image': result,
            'cam_cx': cam_cx,
            'cam_cy': cam_cy,
            'ortho_size': 70.0,
            'render_sz': render_sz,
        }


# ---------------------------------------------------------------------------
# Main wizard dialog
# ---------------------------------------------------------------------------

class BambiCorrectionWizard(QDialog):
    """Three-step wizard for calibrating camera correction factors.

    Parameters
    ----------
    iface : QgsInterface
    config : dict
        Current BAMBI configuration (as returned by
        ``BambiDockWidget.get_config()``).  Must contain ``dem_path`` and
        ``target_folder``.
    parent : QWidget, optional
    """

    def __init__(self, iface, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.iface = iface
        self._config = config

        # Loaded once; released on close
        self._tri_mesh = None
        self._raw_mesh_data = None
        self._dem_worker: Optional[_DemLoadWorker] = None
        self._render_worker: Optional[_LightFieldRenderWorker] = None
        self._probe_worker: Optional[_ProbeWorker] = None

        # Debounce timer: delays circle-plot refresh while spinboxes are being
        # dragged so the main thread is not flooded with ray-cast requests.
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(250)   # ms
        self._plot_timer.timeout.connect(self._update_circle_plot)

        # Poses data keyed by type ('t' = thermal, 'w' = rgb)
        self._poses: Dict[str, Optional[dict]] = {'t': None, 'w': None}

        # Per-side state (side 0 = left / reference, side 1 = right / target)
        self._frame_idx = [0, 0]
        self._frame_type = ['t', 't']
        self._img_size = [(1, 1), (1, 1)]   # (width, height) in pixels

        # Current correction – kept in sync with page-2 spinboxes; radians
        self._correction: dict = {
            'translation': dict(config.get('translation', {'x': 0.0, 'y': 0.0, 'z': 0.0})),
            'rotation': dict(config.get('rotation', {'x': 0.0, 'y': 0.0, 'z': 0.0})),
        }

        # Render overlay state
        self._render_base_pixmap: Optional[QPixmap] = None
        self._render_cam_info: Optional[dict] = None
        self._render_geo_world_points: Optional[list] = None

        self.setWindowTitle("BAMBI Correction Wizard")
        self.setMinimumSize(920, 660)
        self.resize(1120, 760)

        err = self._validate_config()
        if err:
            QTimer.singleShot(
                0,
                lambda: (
                    QMessageBox.critical(self, "Cannot open wizard", err),
                    self.reject(),
                ),
            )
            return

        self._load_poses()

        if not any(self._poses.values()):
            QTimer.singleShot(
                0,
                lambda: (
                    QMessageBox.warning(
                        self,
                        "No Poses Found",
                        "No poses file was found in the target folder.\n\n"
                        "Please extract thermal or RGB frames first.",
                    ),
                    self.reject(),
                ),
            )
            return

        self._setup_ui()
        self._start_dem_loading()

    # ------------------------------------------------------------------
    # Validation & setup helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> Optional[str]:
        if not _HAS_CV2:
            return "opencv-python (cv2) is required but not installed."
        dem = self._config.get('dem_path', '').strip()
        folder = self._config.get('target_folder', '').strip()
        if not dem:
            return ("No DEM file is configured.\n"
                    "Please set the DEM path in the BAMBI panel first.")
        if not os.path.isfile(dem):
            return f"DEM file not found:\n{dem}"
        if not folder or not os.path.isdir(folder):
            return ("No valid target folder is configured.\n"
                    "Please set it in the BAMBI panel first.")
        return None

    def _load_poses(self) -> None:
        folder = self._config['target_folder']
        for suffix in ('t', 'w'):
            path = os.path.join(folder, f'poses_{suffix}.json')
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        self._poses[suffix] = json.load(fh)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Step header
        self._step_label = QLabel()
        self._step_label.setAlignment(Qt.AlignCenter)
        hdr_font = QFont()
        hdr_font.setBold(True)
        hdr_font.setPointSize(hdr_font.pointSize() + 1)
        self._step_label.setFont(hdr_font)
        root.addWidget(self._step_label)

        # DEM loading indicator (hidden once ready)
        self._dem_loading_widget = QWidget()
        dl_layout = QHBoxLayout(self._dem_loading_widget)
        dl_layout.setContentsMargins(0, 0, 0, 0)
        dl_layout.addWidget(QLabel("Loading DEM mesh…"))
        self._dem_pbar = QProgressBar()
        self._dem_pbar.setRange(0, 0)  # indeterminate
        dl_layout.addWidget(self._dem_pbar)
        root.addWidget(self._dem_loading_widget)

        # Pages
        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        self._page1 = self._build_page1()
        self._page2 = self._build_page2()
        self._page3 = self._build_page3()
        self._stack.addWidget(self._page1)
        self._stack.addWidget(self._page2)
        self._stack.addWidget(self._page3)

        self._goto(0)

    # ---- Page 1 -------------------------------------------------------

    def _build_page1(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._side_widgets: List[dict] = []

        for side, title in enumerate(
            ("Reference Frame (Left)", "Target Frame (Right)")
        ):
            splitter.addWidget(self._build_frame_picker(side, title))
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        info = QLabel(
            "Tip: choose a sharp, unambiguous ground feature "
            "(e.g. road corner, building edge) visible in both frames."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addWidget(info)

        nav = QHBoxLayout()
        nav.addStretch()
        self._p1_next_btn = QPushButton("Next: Calibration  →")
        self._p1_next_btn.setEnabled(False)
        self._p1_next_btn.clicked.connect(self._goto_page2)
        nav.addWidget(self._p1_next_btn)
        layout.addLayout(nav)
        return w

    def _build_frame_picker(self, side: int, title: str) -> QWidget:
        panel = QGroupBox(title)
        layout = QVBoxLayout(panel)

        # --- controls row ---
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Type:"))

        type_combo = QComboBox()
        for lbl, key in [("Thermal (T)", 't'), ("RGB (W)", 'w')]:
            if self._poses.get(key):
                type_combo.addItem(lbl, key)
        if type_combo.count() == 0:
            type_combo.addItem("No poses found", 't')
        ctrl.addWidget(type_combo)

        ctrl.addWidget(QLabel("  Frame:"))
        idx_spin = QSpinBox()
        idx_spin.setRange(0, self._max_frame_idx(type_combo.currentData() or 't'))
        ctrl.addWidget(idx_spin)

        load_btn = QPushButton("Load")
        ctrl.addWidget(load_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # --- image view ---
        img_lbl = ClickableImageLabel()
        layout.addWidget(img_lbl, stretch=1)

        # --- point status ---
        pt_lbl = QLabel("No point selected — click on the image above.")
        pt_lbl.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addWidget(pt_lbl)

        # --- wire signals ---
        def _type_changed():
            key = type_combo.currentData() or 't'
            idx_spin.setRange(0, self._max_frame_idx(key))
            self._frame_type[side] = key

        def _load_clicked():
            key = type_combo.currentData() or 't'
            idx = idx_spin.value()
            self._frame_type[side] = key
            self._frame_idx[side] = idx
            self._load_frame(side, key, idx, img_lbl)

        def _point_set(nx: float, ny: float):
            pt_lbl.setText(f"Point set: ({nx:.4f}, {ny:.4f})")
            pt_lbl.setStyleSheet("color: #6f6;")
            self._check_page1_ready()

        type_combo.currentIndexChanged.connect(_type_changed)
        load_btn.clicked.connect(_load_clicked)
        img_lbl.pointSet.connect(_point_set)

        self._side_widgets.append(
            {'type_combo': type_combo, 'idx_spin': idx_spin,
             'img_lbl': img_lbl, 'pt_lbl': pt_lbl}
        )
        return panel

    # ---- Page 2 -------------------------------------------------------

    def _build_page2(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        # Info box
        info_lbl = QLabel(
            "Try to match the two geo-referenced positions (cross marks). "
            "Typically only translational and rotational offsets along the z-axis are necessary. "
            "Use the automatic mode as a possible starting point."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet(
            "background: #1e3a5f; color: #cce4ff; "
            "border: 1px solid #3a6a9f; border-radius: 4px; padding: 6px;"
        )
        layout.addWidget(info_lbl)

        # Z-offset probe
        probe_grp = QGroupBox("Step 1 — Automatic Z-Offset Probe & Rotation Alignment")
        pg_layout = QHBoxLayout(probe_grp)

        pg_layout.addWidget(QLabel("Max steps:"))
        self._max_steps_spin = QSpinBox()
        self._max_steps_spin.setRange(1, 200)
        self._max_steps_spin.setValue(20)
        self._max_steps_spin.setToolTip(
            "Maximum number of ±1 m z-steps to probe before giving up."
        )
        pg_layout.addWidget(self._max_steps_spin)

        self._probe_btn = QPushButton("Run Z-Probe + Rotation Alignment")
        self._probe_btn.clicked.connect(self._run_probe)
        pg_layout.addWidget(self._probe_btn)

        self._probe_status = QLabel()
        self._probe_status.setWordWrap(True)
        pg_layout.addWidget(self._probe_status, stretch=1)
        layout.addWidget(probe_grp)

        # Plot | manual spinboxes
        mid = QSplitter(Qt.Horizontal)

        plot_grp = QGroupBox("Circle Visualization")
        pp = QVBoxLayout(plot_grp)
        self._circle_plot = CirclePlotWidget()
        pp.addWidget(self._circle_plot)
        self._compute_status = QLabel()
        self._compute_status.setStyleSheet("color: #f90; font-style: italic;")
        self._compute_status.setAlignment(Qt.AlignCenter)
        pp.addWidget(self._compute_status)
        mid.addWidget(plot_grp)

        self._ctrl_grp = QGroupBox("Step 2 — Manual Fine-Tuning")
        cf = QFormLayout(self._ctrl_grp)
        cf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Rotation unit toggle
        self._rot_unit_combo = QComboBox()
        self._rot_unit_combo.addItem("Radians", "rad")
        self._rot_unit_combo.addItem("Degrees", "deg")
        cf.addRow("Rotation unit:", self._rot_unit_combo)

        def _dspin(lo=-500.0, hi=500.0, decimals=4, step=0.001):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            return s

        self._tx = _dspin()
        self._ty = _dspin()
        self._tz = _dspin(lo=-1000.0, hi=1000.0)
        self._rx = _dspin(-7.0, 7.0, decimals=5)
        self._ry = _dspin(-7.0, 7.0, decimals=5)
        self._rz = _dspin(-7.0, 7.0, decimals=5)

        cf.addRow("Translation X:", self._tx)
        cf.addRow("Translation Y:", self._ty)
        cf.addRow("Translation Z:", self._tz)

        # Keep label references so we can update the unit suffix on toggle
        self._rx_label = QLabel("Rotation X  (pitch, rad):")
        self._ry_label = QLabel("Rotation Y  (roll, rad):")
        self._rz_label = QLabel("Rotation Z  (yaw, rad):")
        cf.addRow(self._rx_label, self._rx)
        cf.addRow(self._ry_label, self._ry)
        cf.addRow(self._rz_label, self._rz)

        for sp in (self._tx, self._ty, self._tz, self._rx, self._ry, self._rz):
            sp.valueChanged.connect(self._on_corr_changed)

        self._rot_unit_combo.currentIndexChanged.connect(self._on_rot_unit_changed)

        refresh_btn = QPushButton("↺  Refresh Plot")
        refresh_btn.clicked.connect(self._update_circle_plot)
        cf.addRow("", refresh_btn)

        mid.addWidget(self._ctrl_grp)
        mid.setSizes([450, 380])
        layout.addWidget(mid, stretch=1)

        nav = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(lambda: self._goto(0))
        nav.addWidget(back_btn)
        nav.addStretch()
        self._p2_next_btn = QPushButton("Accept Correction → Light Field Preview")
        self._p2_next_btn.clicked.connect(self._goto_page3)
        nav.addWidget(self._p2_next_btn)
        layout.addLayout(nav)
        return w

    # ---- Page 3 -------------------------------------------------------

    def _build_page3(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)

        frame_grp = QGroupBox("Frames to Render")
        fg = QFormLayout(frame_grp)

        self._render_info_lbl = QLabel()
        fg.addRow("Selected frames:", self._render_info_lbl)

        self._use_range_chk = QCheckBox("Add neighbouring frames")
        self._use_range_chk.setChecked(False)
        fg.addRow("", self._use_range_chk)

        range_w = QWidget()
        rl = QHBoxLayout(range_w)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("Start:"))
        self._rng_start = QSpinBox()
        self._rng_start.setRange(0, 999999)
        rl.addWidget(self._rng_start)
        rl.addWidget(QLabel("End:"))
        self._rng_end = QSpinBox()
        self._rng_end.setRange(0, 999999)
        rl.addWidget(self._rng_end)
        rl.addWidget(QLabel("Step:"))
        self._rng_step = QSpinBox()
        self._rng_step.setRange(1, 100)
        self._rng_step.setValue(1)
        rl.addWidget(self._rng_step)
        rl.addStretch()
        self._use_range_chk.toggled.connect(range_w.setEnabled)
        range_w.setEnabled(False)
        fg.addRow("", range_w)

        size_combo = QComboBox()
        for sz in (512, 1024, 2048):
            size_combo.addItem(f"{sz} × {sz}", sz)
        size_combo.setCurrentIndex(1)
        self._render_size_combo = size_combo
        fg.addRow("Render size:", size_combo)
        layout.addWidget(frame_grp)

        render_btn = QPushButton("▶  Render Light Field")
        render_btn.clicked.connect(self._render_light_field)
        layout.addWidget(render_btn)

        self._render_pbar = QProgressBar()
        self._render_pbar.setRange(0, 100)
        self._render_pbar.setVisible(False)
        layout.addWidget(self._render_pbar)

        self._show_geopoint_chk = QCheckBox("Show geo-referenced points")
        self._show_geopoint_chk.setChecked(True)
        self._show_geopoint_chk.setEnabled(False)   # enabled once a render is available
        self._show_geopoint_chk.toggled.connect(self._update_render_display)
        layout.addWidget(self._show_geopoint_chk)

        self._render_img_lbl = QLabel(
            "Press 'Render Light Field' to generate a preview."
        )
        self._render_img_lbl.setAlignment(Qt.AlignCenter)
        self._render_img_lbl.setMinimumHeight(280)
        self._render_img_lbl.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self._render_img_lbl.setStyleSheet(
            "border: 1px solid #555; background: #12122a;"
        )
        layout.addWidget(self._render_img_lbl, stretch=1)

        # Save controls
        save_grp = QGroupBox("Save Correction")
        sg = QHBoxLayout(save_grp)

        save_global_btn = QPushButton("Save as Global Default")
        save_global_btn.setToolTip(
            "Writes the current translation/rotation as the default "
            "correction in correction.json."
        )
        save_global_btn.clicked.connect(self._save_global)
        sg.addWidget(save_global_btn)

        sg.addWidget(QLabel("  |  Frame range:"))
        self._loc_start = QSpinBox()
        self._loc_start.setRange(0, 999999)
        sg.addWidget(self._loc_start)
        sg.addWidget(QLabel("—"))
        self._loc_end = QSpinBox()
        self._loc_end.setRange(0, 999999)
        self._loc_end.setValue(999999)
        sg.addWidget(self._loc_end)
        save_local_btn = QPushButton("Save as Local Correction")
        save_local_btn.setToolTip(
            "Appends the correction as a frame-range entry in correction.json."
        )
        save_local_btn.clicked.connect(self._save_local)
        sg.addWidget(save_local_btn)
        sg.addStretch()
        layout.addWidget(save_grp)

        nav = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(lambda: self._goto(1))
        nav.addWidget(back_btn)
        nav.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        nav.addWidget(close_btn)
        layout.addLayout(nav)
        return w

    # ------------------------------------------------------------------
    # DEM loading
    # ------------------------------------------------------------------

    def _start_dem_loading(self) -> None:
        self._dem_worker = _DemLoadWorker(self._config['dem_path'])
        self._dem_worker.finished.connect(self._on_dem_loaded)
        self._dem_worker.error.connect(self._on_dem_error)
        self._dem_worker.start()

    def _on_dem_loaded(self, tri_mesh, raw_mesh_data) -> None:
        self._tri_mesh = tri_mesh
        self._raw_mesh_data = raw_mesh_data
        self._dem_loading_widget.hide()
        self._check_page1_ready()

    def _on_dem_error(self, msg: str) -> None:
        self._dem_loading_widget.hide()
        QMessageBox.critical(
            self, "DEM Load Error",
            f"Failed to load the DEM:\n{msg}\n\n"
            "Geo-referencing will not be available.",
        )

    # ------------------------------------------------------------------
    # Frame loading helpers (page 1)
    # ------------------------------------------------------------------

    def _max_frame_idx(self, frame_type: str) -> int:
        poses = self._poses.get(frame_type)
        if poses is None:
            return 0
        return max(0, len(poses.get('images', [])) - 1)

    def _load_frame(
        self, side: int, frame_type: str, frame_idx: int,
        img_lbl: ClickableImageLabel,
    ) -> None:
        folder = self._config['target_folder']
        frames_dir = 'frames_t' if frame_type == 't' else 'frames_w'
        poses = self._poses.get(frame_type)
        if poses is None:
            img_lbl.clear_image()
            img_lbl.setText(f"No '{frame_type}' poses loaded.")
            return
        images = poses.get('images', [])
        if frame_idx >= len(images):
            img_lbl.clear_image()
            img_lbl.setText(f"Frame {frame_idx} is out of range.")
            return
        imagefile = images[frame_idx].get('imagefile', '')
        img_path = os.path.join(folder, frames_dir, imagefile)
        if not os.path.isfile(img_path):
            img_lbl.clear_image()
            img_lbl.setText(f"File not found:\n{os.path.basename(img_path)}")
            return
        img = cv2.imread(img_path)
        if img is None:
            img_lbl.clear_image()
            img_lbl.setText("Could not read image file.")
            return
        self._img_size[side] = (img.shape[1], img.shape[0])
        img_lbl.set_image(img)

    def _check_page1_ready(self) -> None:
        ready = (
            self._tri_mesh is not None
            and all(sw['img_lbl'].get_point() is not None for sw in self._side_widgets)
        )
        self._p1_next_btn.setEnabled(ready)

    # ------------------------------------------------------------------
    # Page transitions
    # ------------------------------------------------------------------

    def _goto(self, index: int) -> None:
        titles = [
            "Step 1 of 3  —  Select Corresponding Ground Points",
            "Step 2 of 3  —  Calibrate Correction Factors",
            "Step 3 of 3  —  Light Field Preview & Save",
        ]
        self._step_label.setText(titles[index])
        self._stack.setCurrentIndex(index)

    def _goto_page2(self) -> None:
        self._load_corr_into_spins()
        self._goto(1)
        if self._tri_mesh is not None:
            # Defer by one event-loop cycle so the page renders before the
            # first (potentially slow) geo-referencing call.
            self._compute_status.setText("Computing initial geo-referencing…")
            QTimer.singleShot(50, self._deferred_initial_plot)

    def _deferred_initial_plot(self) -> None:
        self._update_circle_plot()
        self._compute_status.setText("")

    def _goto_page3(self) -> None:
        self._correction = self._read_corr_from_spins()

        # Snapshot geo-referenced world points for the render overlay
        circles = self._compute_circles(self._correction)
        if circles:
            (_, _, p1), (_, _, p2) = circles
            self._render_geo_world_points = [p1, p2]
        else:
            self._render_geo_world_points = None

        f0, f1 = self._frame_idx[0], self._frame_idx[1]
        lo, hi = min(f0, f1), max(f0, f1)
        self._rng_start.setValue(lo)
        self._rng_end.setValue(hi)
        self._loc_start.setValue(lo)
        self._loc_end.setValue(hi)
        self._render_info_lbl.setText(f"Frame {f0} and frame {f1}")
        self._goto(2)

    # ------------------------------------------------------------------
    # Geo-referencing helpers
    # ------------------------------------------------------------------

    def _pose_images(self, frame_type: str) -> Optional[list]:
        poses = self._poses.get(frame_type)
        return poses.get('images') if poses else None

    def _frame_pose(self, frame_type: str, frame_idx: int) -> Optional[dict]:
        images = self._pose_images(frame_type)
        if images is None or frame_idx >= len(images):
            return None
        return images[frame_idx]

    def _build_camera(self, frame_type: str, frame_idx: int, correction: dict):
        """Build a Camera from pose data and correction.  Returns Camera or raises."""
        try:
            from pyrr import Quaternion, Vector3
            from alfspy.core.rendering import Camera
            from alfspy.core.util.pyrrs import quaternion_from_eulers
        except ImportError as exc:
            raise RuntimeError(f"alfspy / pyrr not available: {exc}") from exc

        pose = self._frame_pose(frame_type, frame_idx)
        if pose is None:
            raise ValueError(
                f"No pose data for frame {frame_idx} (type='{frame_type}')"
            )

        loc = pose.get('location', [0.0, 0.0, 0.0])
        rot = pose.get('rotation', [0.0, 0.0, 0.0])
        fov_raw = pose.get('fovy', 50.0)
        if isinstance(fov_raw, (list, tuple)):
            fov_raw = fov_raw[0] if fov_raw else 50.0
        fovy = float(fov_raw)

        t = correction['translation']
        r = correction['rotation']
        cor_t = Vector3([t['x'], t['y'], t['z']], dtype='f4')
        cor_r = Vector3([r['x'], r['y'], r['z']], dtype='f4')

        # Mirror the logic in alfs.get_camera_for_frame
        position = Vector3(loc, dtype='f4') + cor_t
        cam_euler = Vector3(
            [math.radians(_wrap_deg(v)) for v in rot[:3]], dtype='f4'
        )
        rot_eulers = (cam_euler - cor_r) * -1.0
        cam_quat = Quaternion.from_eulers(rot_eulers)

        return Camera(fovy=fovy, aspect_ratio=1.0,
                      position=position, rotation=cam_quat)

    def _geo_ref(self, side: int, correction: dict) -> Optional[Tuple[float, float, float]]:
        """Project the selected pixel for *side* onto the DEM.

        Returns (x, y, z) in world space or None.
        """
        if self._tri_mesh is None:
            return None
        sw = self._side_widgets[side]
        pt = sw['img_lbl'].get_point()
        if pt is None:
            return None

        try:
            from alfspy.core.convert.convert import pixel_to_world_coord
        except ImportError:
            return None

        ft = self._frame_type[side]
        fi = self._frame_idx[side]
        img_w, img_h = self._img_size[side]

        try:
            cam = self._build_camera(ft, fi, correction)
        except Exception:
            return None

        px, py = pt[0] * img_w, pt[1] * img_h
        results = pixel_to_world_coord(
            [px], [py], img_w, img_h,
            self._tri_mesh, cam, include_misses=False,
        )
        if results is None or len(results) == 0 or results[0] is None:
            return None
        wp = results[0]
        return (float(wp[0]), float(wp[1]), float(wp[2]))

    def _camera_xy(self, side: int, correction: dict) -> Optional[Tuple[float, float]]:
        """Return camera XY position with translation correction applied."""
        pose = self._frame_pose(self._frame_type[side], self._frame_idx[side])
        if pose is None:
            return None
        loc = pose.get('location', [0.0, 0.0, 0.0])
        t = correction['translation']
        return (loc[0] + t['x'], loc[1] + t['y'])

    def _compute_circles(self, correction: dict) -> Optional[tuple]:
        """Compute circle parameters for both sides.

        Returns ``((c1, r1, p1_xy), (c2, r2, p2_xy))`` or ``None``.
        """
        c1 = self._camera_xy(0, correction)
        c2 = self._camera_xy(1, correction)
        p1 = self._geo_ref(0, correction)
        p2 = self._geo_ref(1, correction)
        if any(v is None for v in (c1, c2, p1, p2)):
            return None
        r1 = math.sqrt((p1[0] - c1[0]) ** 2 + (p1[1] - c1[1]) ** 2)
        r2 = math.sqrt((p2[0] - c2[0]) ** 2 + (p2[1] - c2[1]) ** 2)
        return (c1, r1, (p1[0], p1[1])), (c2, r2, (p2[0], p2[1]))

    # ------------------------------------------------------------------
    # Z-offset probe & rotation alignment
    # ------------------------------------------------------------------

    def _run_probe(self) -> None:
        self._probe_btn.setEnabled(False)
        self._p2_next_btn.setEnabled(False)
        self._probe_status.setText("Starting…")

        correction = self._read_corr_from_spins()
        max_steps  = self._max_steps_spin.value()

        # Collect all data the worker needs *before* leaving the main thread
        # so the worker never accesses any QWidget.
        side_params = [
            {
                'point':      self._side_widgets[s]['img_lbl'].get_point(),
                'frame_type': self._frame_type[s],
                'frame_idx':  self._frame_idx[s],
                'img_size':   self._img_size[s],
                'poses':      self._poses.get(self._frame_type[s]),
            }
            for s in range(2)
        ]

        self._probe_worker = _ProbeWorker(
            self._tri_mesh, side_params, correction, max_steps
        )
        self._probe_worker.status.connect(self._probe_status.setText)
        self._probe_worker.finished.connect(self._on_probe_done)
        self._probe_worker.error.connect(self._on_probe_error)
        self._probe_worker.start()

    def _on_probe_done(self, correction: dict) -> None:
        self._correction = correction
        self._load_corr_into_spins()
        self._update_circle_plot()

        circles = self._compute_circles(self._correction)
        if circles:
            (c1, r1, _), (c2, r2, _) = circles
            ok      = _circles_intersect(c1, r1, c2, r2)
            tz      = correction['translation']['z']
            rz      = correction['rotation']['z']
            verdict = "intersect ✓" if ok else "do NOT intersect"
            self._probe_status.setText(
                f"Done.  tz = {tz:.3f},  rz = {rz:.5f} rad  →  circles {verdict}"
            )
        else:
            self._probe_status.setText(
                "Done. (Could not verify circles — check DEM coverage)"
            )

        self._probe_btn.setEnabled(True)
        self._p2_next_btn.setEnabled(True)

    def _on_probe_error(self, msg: str) -> None:
        self._probe_status.setText(f"Error: {msg[:200]}")
        self._probe_btn.setEnabled(True)
        self._p2_next_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Circle plot
    # ------------------------------------------------------------------

    def _update_circle_plot(self) -> None:
        correction = self._read_corr_from_spins()
        circles = self._compute_circles(correction)
        self._compute_status.setText("")
        if circles is None:
            self._circle_plot.clear()
            return
        (c1, r1, p1), (c2, r2, p2) = circles
        self._circle_plot.set_data(
            c1, r1, c2, r2, p1, p2,
            intersects=_circles_intersect(c1, r1, c2, r2),
        )

    def _on_corr_changed(self) -> None:
        if self._tri_mesh is not None:
            # Restart the debounce timer; the plot refreshes 250 ms after the
            # last spinbox change so rapid dragging does not flood the thread.
            self._compute_status.setText("Computing…")
            self._plot_timer.start()

    # ------------------------------------------------------------------
    # Correction spinbox <-> dict synchronisation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Rotation unit helpers
    # ------------------------------------------------------------------

    def _is_degrees(self) -> bool:
        return self._rot_unit_combo.currentData() == "deg"

    def _on_rot_unit_changed(self) -> None:
        """Convert the three rotation spinboxes when the unit toggle changes."""
        deg = self._is_degrees()

        # Read current radian values from the internal correction dict so we
        # convert from a canonical source rather than from a half-converted UI.
        r = self._correction['rotation']
        rad_vals = [r.get('x', 0.0), r.get('y', 0.0), r.get('z', 0.0)]

        for sp in (self._rx, self._ry, self._rz):
            sp.blockSignals(True)

        if deg:
            for sp, rv in zip((self._rx, self._ry, self._rz), rad_vals):
                sp.setRange(-360.0, 360.0)
                sp.setDecimals(3)
                sp.setSingleStep(0.1)
                sp.setValue(math.degrees(rv))
            unit_str = "deg"
        else:
            for sp, rv in zip((self._rx, self._ry, self._rz), rad_vals):
                sp.setRange(-7.0, 7.0)
                sp.setDecimals(5)
                sp.setSingleStep(0.001)
                sp.setValue(rv)
            unit_str = "rad"

        for sp in (self._rx, self._ry, self._rz):
            sp.blockSignals(False)

        self._rx_label.setText(f"Rotation X  (pitch, {unit_str}):")
        self._ry_label.setText(f"Rotation Y  (roll, {unit_str}):")
        self._rz_label.setText(f"Rotation Z  (yaw, {unit_str}):")

    # ------------------------------------------------------------------
    # Correction spinbox <-> dict synchronisation
    # ------------------------------------------------------------------

    def _load_corr_into_spins(self) -> None:
        for sp in (self._tx, self._ty, self._tz, self._rx, self._ry, self._rz):
            sp.blockSignals(True)
        t = self._correction['translation']
        r = self._correction['rotation']
        self._tx.setValue(t.get('x', 0.0))
        self._ty.setValue(t.get('y', 0.0))
        self._tz.setValue(t.get('z', 0.0))
        rx_rad = r.get('x', 0.0)
        ry_rad = r.get('y', 0.0)
        rz_rad = r.get('z', 0.0)
        if self._is_degrees():
            self._rx.setValue(math.degrees(rx_rad))
            self._ry.setValue(math.degrees(ry_rad))
            self._rz.setValue(math.degrees(rz_rad))
        else:
            self._rx.setValue(rx_rad)
            self._ry.setValue(ry_rad)
            self._rz.setValue(rz_rad)
        for sp in (self._tx, self._ty, self._tz, self._rx, self._ry, self._rz):
            sp.blockSignals(False)

    def _read_corr_from_spins(self) -> dict:
        if self._is_degrees():
            rx = math.radians(self._rx.value())
            ry = math.radians(self._ry.value())
            rz = math.radians(self._rz.value())
        else:
            rx = self._rx.value()
            ry = self._ry.value()
            rz = self._rz.value()
        return {
            'translation': {
                'x': self._tx.value(),
                'y': self._ty.value(),
                'z': self._tz.value(),
            },
            'rotation': {'x': rx, 'y': ry, 'z': rz},
        }

    # ------------------------------------------------------------------
    # Light-field rendering (page 3)
    # ------------------------------------------------------------------

    def _collect_render_frames(self) -> List[dict]:
        folder = self._config['target_folder']
        indices = set(self._frame_idx)

        if self._use_range_chk.isChecked():
            start = self._rng_start.value()
            end = self._rng_end.value()
            step = self._rng_step.value()
            indices.update(range(start, end + 1, step))

        frames_info = []
        for idx in sorted(indices):
            for ft in ('t', 'w'):
                pose = self._frame_pose(ft, idx)
                if pose is None:
                    continue
                frames_dir = 'frames_t' if ft == 't' else 'frames_w'
                img_path = os.path.join(
                    folder, frames_dir, pose.get('imagefile', '')
                )
                if os.path.isfile(img_path):
                    frames_info.append({
                        'image_path': img_path,
                        'location': pose.get('location', [0.0, 0.0, 0.0]),
                        'rotation': pose.get('rotation', [0.0, 0.0, 0.0]),
                        'fovy': pose.get('fovy', 50.0),
                    })
                    break   # use first available type for each index
        return frames_info

    def _render_light_field(self) -> None:
        frames_info = self._collect_render_frames()
        if not frames_info:
            QMessageBox.warning(
                self, "No Frames",
                "No valid frame images were found for the selected indices.",
            )
            return

        render_size = self._render_size_combo.currentData() or 1024

        # Resolve mask: prefer thermal, fall back to RGB, then None
        folder = self._config.get('target_folder', '')
        mask_path = None
        for candidate in ('mask_T.png', 'mask_W.png'):
            p = os.path.join(folder, candidate)
            if os.path.isfile(p):
                mask_path = p
                break

        params = {
            'dem_path': self._config['dem_path'],
            'frames': frames_info,
            'correction': self._correction,
            'render_size': render_size,
            'mask_path': mask_path,
        }

        self._render_pbar.setVisible(True)
        self._render_pbar.setValue(0)

        self._render_worker = _LightFieldRenderWorker(params)
        self._render_worker.progress.connect(self._render_pbar.setValue)
        self._render_worker.finished.connect(self._on_render_done)
        self._render_worker.error.connect(self._on_render_error)
        self._render_worker.start()

    def _on_render_done(self, result) -> None:
        self._render_pbar.setVisible(False)
        if result is None or result.get('image') is None:
            self._render_img_lbl.setText("Render returned no image.")
            return
        img_bgr = cv2.cvtColor(result['image'], cv2.COLOR_RGBA2BGR)
        self._render_base_pixmap = _bgr_to_qpixmap(img_bgr)
        self._render_cam_info = {
            'cam_cx':    result['cam_cx'],
            'cam_cy':    result['cam_cy'],
            'ortho_size': result['ortho_size'],
            'render_sz': result['render_sz'],
        }
        self._show_geopoint_chk.setEnabled(True)
        self._update_render_display()

    def _update_render_display(self) -> None:
        """Redraw the render label from the cached base pixmap, optionally
        overlaying the geo-referenced point markers.  No re-rendering needed."""
        if self._render_base_pixmap is None:
            return

        px = QPixmap(self._render_base_pixmap)

        if (self._show_geopoint_chk.isChecked()
                and self._render_geo_world_points
                and self._render_cam_info):
            cam = self._render_cam_info
            cx   = cam['cam_cx']
            cy   = cam['cam_cy']
            half = cam['ortho_size'] / 2.0
            sz   = cam['render_sz']
            arm  = max(8, sz // 64)
            lw   = max(2, sz // 256)

            painter = QPainter(px)
            painter.setRenderHint(QPainter.Antialiasing)
            colors = [QColor(255, 80, 80), QColor(80, 200, 255)]
            for i, (wx, wy) in enumerate(self._render_geo_world_points):
                px_x = int((wx - (cx - half)) / cam['ortho_size'] * sz)
                px_y = int(((cy + half) - wy) / cam['ortho_size'] * sz)
                pen = QPen(colors[i % len(colors)], lw)
                painter.setPen(pen)
                painter.drawLine(px_x - arm, px_y - arm, px_x + arm, px_y + arm)
                painter.drawLine(px_x + arm, px_y - arm, px_x - arm, px_y + arm)
            painter.end()

        scaled = px.scaled(
            self._render_img_lbl.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._render_img_lbl.setPixmap(scaled)

    def _on_render_error(self, msg: str) -> None:
        self._render_pbar.setVisible(False)
        QMessageBox.critical(
            self, "Render Error",
            f"Light field rendering failed:\n\n{msg[:800]}",
        )

    # ------------------------------------------------------------------
    # Saving corrections
    # ------------------------------------------------------------------

    def _correction_json_path(self) -> str:
        explicit = self._config.get('correction_path', '').strip()
        if explicit and os.path.isfile(explicit):
            return explicit
        return os.path.join(self._config.get('target_folder', ''), 'correction.json')

    def _read_correction_json(self) -> dict:
        path = self._correction_json_path()
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _write_correction_json(self, data: dict) -> None:
        path = self._correction_json_path()
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=4)

    def _save_global(self) -> None:
        data = self._read_correction_json()
        data['translation'] = dict(self._correction['translation'])
        data['rotation'] = dict(self._correction['rotation'])
        try:
            self._write_correction_json(data)
            QMessageBox.information(
                self, "Saved",
                f"Global correction saved to:\n{self._correction_json_path()}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _save_local(self) -> None:
        start = self._loc_start.value()
        end = self._loc_end.value()
        if start > end:
            QMessageBox.warning(
                self, "Invalid Range",
                "Start frame must be ≤ end frame.",
            )
            return
        data = self._read_correction_json()
        data.setdefault('additional', []).append({
            'start': start,
            'end': end,
            'translation': dict(self._correction['translation']),
            'rotation': dict(self._correction['rotation']),
        })
        try:
            self._write_correction_json(data)
            QMessageBox.information(
                self, "Saved",
                f"Local correction for frames {start}–{end} appended to:\n"
                f"{self._correction_json_path()}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._plot_timer.stop()
        for worker in (self._dem_worker, self._probe_worker, self._render_worker):
            if worker is not None and worker.isRunning():
                worker.quit()
                worker.wait(3000)
        self._tri_mesh = None
        self._raw_mesh_data = None
        super().closeEvent(event)
