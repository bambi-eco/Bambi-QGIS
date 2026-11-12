# -*- coding: utf-8 -*-
"""ALFS tab of the 3D Flight Viewer: live light-field integrals.

The integral of a window of frames - N/2 before to N/2 after the current
frame, every stride-th one - is rendered by alfspy in a persistent worker
thread that owns its own render context, whichever engine the plugin is set
to. Two modes:

* **Fixed on the DEM** - rendered top-down over the window's ground
  footprints and draped onto the DEM, so it stays put while the view moves.
* **From the current view** - rendered from the on-screen camera and
  re-projected onto the DEM with that camera (sharpest for that view); it is
  re-rendered on demand, or after every move with *Continuous*.

The focal surface is the DEM raised by the *focus* offset. Shots are cached
per frame, so moving the window by one frame creates and releases about two
shots instead of the whole window.
"""

import os
import queue
import time
from typing import Dict, Optional

import numpy as np

from ..core import flight_viewer as fv
from ..core import flight_viewer_geometry as geo
from .qt import QtCore, QtWidgets, Signal

MODE_VIEW = "view"
MODE_ORTHO = "ortho"


class AlfsWorker(QtCore.QThread):
    """Renders ALFS requests; only the newest pending request is rendered."""
    rendered = Signal(object)     # dict(result)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = False
        self._shots: Dict[tuple, object] = {}
        self._renderers: Dict[tuple, object] = {}
        self._mesh = None
        self._mesh_key = None
        self._mask_path = None
        self._mask = None

    def submit(self, request: dict):
        self._queue.put(request)

    def shutdown(self):
        self._stop = True
        self._queue.put(None)
        self.wait(3000)

    # -- worker thread -------------------------------------------------------

    def run(self):
        try:
            from ..core.render_context import make_render_context
            self.ctx = make_render_context()
        except Exception as exc:  # noqa: BLE001 - reported in the panel
            self.failed.emit(str(exc))
            return
        while not self._stop:
            req = self._queue.get()
            # Skip to the newest request.
            while True:
                try:
                    newer = self._queue.get_nowait()
                except queue.Empty:
                    break
                req = newer
            if req is None or self._stop:
                break
            try:
                self.rendered.emit(self._render(req))
            except Exception as exc:  # noqa: BLE001
                import traceback
                self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")
        self._release_all()

    def _release_all(self):
        for shot in self._shots.values():
            try:
                shot.release()
            except Exception:  # nosec B110
                pass
        self._shots.clear()
        self._renderers.clear()

    def _mesh_data(self, dem, focus):
        from alfspy.core.rendering import MeshData
        key = (id(dem), round(float(focus), 3))
        if self._mesh_key != key:
            verts = np.array(dem.vertices, dtype=np.float32, copy=True)
            verts[:, 2] += float(focus)
            self._mesh = MeshData(verts, np.asarray(dem.faces, dtype=np.uint32))
            self._mesh_key = key
            self._renderers.clear()
        return self._mesh

    def _renderer(self, resolution, camera, dem, focus):
        from alfspy.core.rendering import Renderer, Resolution
        mesh = self._mesh_data(dem, focus)
        key = (int(resolution[0]), int(resolution[1]))
        r = self._renderers.get(key)
        if r is None:
            r = Renderer(Resolution(*key), self.ctx, camera, mesh)
            self._renderers[key] = r
        r.camera = camera
        return r

    def _shot(self, entry, version, max_edge):
        import cv2
        from alfspy.core.rendering import CtxShot
        key = (entry["key"], version, max_edge)
        shot = self._shots.get(key)
        if shot is not None:
            return shot
        img = cv2.imread(entry["path"], cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        tw, th = geo.scaled_size(w, h, max_edge)
        if (tw, th) != (w, h):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        # A path would be converted by alfspy itself; an array is taken as is.
        from alfspy.core.util.image import to_rgba_f4
        img = to_rgba_f4(img)
        position, rotation = fv.alfs_pose(entry["camera"])
        shot = CtxShot(self.ctx, img, position, rotation, entry["camera"].fovy,
                       entry["camera"].aspect, None)
        self._shots[key] = shot
        return shot

    def _mask_texture(self, path, erode_px=0, max_edge=1024):
        """The (eroded) undistortion mask as alfspy texture, cached."""
        if not path:
            return None
        key = (path, int(erode_px), int(max_edge))
        if key != self._mask_path:
            import cv2
            from alfspy.core.rendering import TextureData
            from alfspy.core.util.image import to_rgba_f4
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = fv.scaled_erosion_mask(img, erode_px, max_edge)
            self._mask = TextureData(to_rgba_f4(img)) if img is not None else None
            self._mask_path = key
        return self._mask

    def _render(self, req):
        start = time.perf_counter()
        version, max_edge = req["version"], req["max_edge"]
        wanted = {(e["key"], version, max_edge) for e in req["frames"]}
        for key in [k for k in self._shots if k not in wanted]:
            try:
                self._shots.pop(key).release()
            except Exception:  # nosec B110
                pass
        shots = [s for s in (self._shot(e, version, max_edge)
                             for e in req["frames"]) if s is not None]
        if not shots:
            raise RuntimeError("None of the window's frame images could be read.")
        if req["mode"] == MODE_ORTHO:
            lo, hi = req["extent"]
            size = np.asarray(hi) - np.asarray(lo)
            res = geo.scaled_size(int(size[0] * 1000), int(size[1] * 1000),
                                  req["resolution"])
            camera = fv.alfs_ortho_camera(lo, hi, req["top"], req["bottom"])
        else:
            res = req["viewport"]
            camera = fv.alfs_camera(req["camera"])
        renderer = self._renderer(res, camera, req["dem"], req["focus"])
        result = renderer.render_integral_raw(shots, mask=self._mask_texture(
            req.get("mask_path"), req.get("mask_erosion", 0), max_edge))
        rgba = fv.integral_to_rgba(result.accum, result.coverage,
                                   req["min_coverage"])
        return dict(req, image=rgba, seconds=time.perf_counter() - start,
                    used=len(shots))


class AlfsPanel(QtWidgets.QWidget):
    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.worker: Optional[AlfsWorker] = None
        self.version = 0
        self.result: Optional[dict] = None
        self._serial = 0
        self._busy = False
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.render_now)
        self._build()

    # -- UI -------------------------------------------------------------------

    def _build(self):
        Q = QtWidgets
        lay = Q.QVBoxLayout(self)
        intro = Q.QLabel(
            "Airborne light-field sampling: the frames around the current one "
            "are integrated on a focal surface (the DEM raised by the focus "
            "offset). Choose \"ALFS integral\" on the View tab to drape it.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        lay.addWidget(intro)

        box = Q.QGroupBox("Frames")
        f = Q.QFormLayout(box)
        self.n_spin = Q.QSpinBox()
        self.n_spin.setRange(0, 5000)
        self.n_spin.setValue(40)
        self.n_spin.setToolTip(
            "Window size in frames: N/2 before to N/2 after the current frame "
            "are integrated (every stride-th one).")
        self.n_spin.valueChanged.connect(self._changed)
        f.addRow("Frames N (± N/2)", self.n_spin)
        self.stride_spin = Q.QSpinBox()
        self.stride_spin.setRange(1, 100)
        self.stride_spin.setValue(2)
        self.stride_spin.setToolTip("Use every s-th frame of the window.")
        self.stride_spin.valueChanged.connect(self._changed)
        f.addRow("Stride", self.stride_spin)
        lay.addWidget(box)

        box = Q.QGroupBox("Integral")
        f = Q.QFormLayout(box)
        self.mode_combo = Q.QComboBox()
        self.mode_combo.addItem("Fixed on the DEM (top-down)", MODE_ORTHO)
        self.mode_combo.addItem("From the current view", MODE_VIEW)
        self.mode_combo.setToolTip(
            "Fixed: rendered top-down over the frames' footprints and draped "
            "onto the DEM - it stays in place while you navigate, like the "
            "draped frame. From the current view: rendered from the on-screen "
            "camera (sharpest for that view); it is re-rendered with Render "
            "(Enter), or after every move with Continuous.")
        self.mode_combo.currentIndexChanged.connect(self._changed)
        f.addRow("Mode", self.mode_combo)
        row = Q.QHBoxLayout()
        self.focus_slider = Q.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.focus_slider.setRange(-500, 500)
        self.focus_spin = Q.QDoubleSpinBox()
        self.focus_spin.setRange(-50.0, 50.0)
        self.focus_spin.setDecimals(1)
        self.focus_spin.setSingleStep(0.5)
        self.focus_spin.setSuffix(" m")
        self.focus_slider.valueChanged.connect(
            lambda v: self.focus_spin.setValue(v / 10.0))
        self.focus_spin.valueChanged.connect(self._on_focus)
        row.addWidget(self.focus_slider, 1)
        row.addWidget(self.focus_spin)
        f.addRow("Focus offset", row)
        self.surface_combo = Q.QComboBox()
        self.surface_combo.addItem("DEM", "dem")
        self.surface_combo.addItem("Reconstructed surface", "recon")
        self.surface_combo.model().item(1).setEnabled(False)
        self.surface_combo.setToolTip(
            "Surface the frames are integrated on (plus the focus offset). "
            "The reconstructed surface becomes available after a dense "
            "reconstruction on the Reconstruction tab.")
        self.surface_combo.currentIndexChanged.connect(self._changed)
        f.addRow("Focal surface", self.surface_combo)
        self.threshold_spin = Q.QSpinBox()
        self.threshold_spin.setRange(1, 1000)
        self.threshold_spin.setToolTip(
            "Hide pixels seen by fewer frames than this.")
        self.threshold_spin.valueChanged.connect(self._changed)
        f.addRow("Min. frames per pixel", self.threshold_spin)
        self.res_combo = Q.QComboBox()
        for r in (512, 1024, 2048, 4096):
            self.res_combo.addItem(f"{r} px", r)
        self.res_combo.setCurrentIndex(1)
        self.res_combo.setToolTip(
            "Ortho: longest edge of the integral. View: the viewport size, "
            "capped to this. Frames are shrunk to this size as well.")
        self.res_combo.currentIndexChanged.connect(self._changed)
        f.addRow("Resolution", self.res_combo)
        lay.addWidget(box)

        row = Q.QHBoxLayout()
        self.live_check = Q.QCheckBox("Live")
        self.live_check.setChecked(True)
        self.live_check.setToolTip(
            "Re-render when the frame or a parameter changes (and in View "
            "mode when the view stops moving).")
        row.addWidget(self.live_check)
        self.continuous_check = Q.QCheckBox("Continuous")
        self.continuous_check.setToolTip(
            "Keep rendering, one integral after the other.")
        self.continuous_check.toggled.connect(self._changed)
        row.addWidget(self.continuous_check)
        btn = Q.QPushButton("Render")
        btn.setToolTip("Render now (Enter)")
        btn.clicked.connect(self.render_now)
        row.addWidget(btn)
        lay.addLayout(row)
        self.export_btn = Q.QPushButton("Export integral…")
        self.export_btn.setToolTip(
            "Save the last Ortho integral as a georeferenced PNG + world file "
            "(opens in QGIS), or the last View integral as a PNG.")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        lay.addWidget(self.export_btn)
        self.info = Q.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray;")
        lay.addWidget(self.info)
        lay.addStretch(1)

        from ..bambi_flight_viewer import _settings, SETTINGS_GROUP
        s = _settings()
        s.beginGroup(SETTINGS_GROUP + "/alfs")
        for name in ("n_spin", "stride_spin", "threshold_spin"):
            v = s.value(name)
            if v is not None:
                try:
                    getattr(self, name).setValue(int(v))
                except (TypeError, ValueError):
                    pass
        for name in ("mode_combo", "res_combo"):
            v = s.value(name)
            if v is not None:
                i = getattr(self, name).findText(str(v))
                if i >= 0:
                    getattr(self, name).setCurrentIndex(i)
        s.endGroup()

    def save_settings(self):
        from ..bambi_flight_viewer import _settings, SETTINGS_GROUP
        s = _settings()
        s.beginGroup(SETTINGS_GROUP + "/alfs")
        for name in ("n_spin", "stride_spin", "threshold_spin"):
            s.setValue(name, getattr(self, name).value())
        for name in ("mode_combo", "res_combo"):
            s.setValue(name, getattr(self, name).currentText())
        s.endGroup()

    # -- dialog hooks ---------------------------------------------------------------

    def _active(self) -> bool:
        from ..bambi_flight_viewer import DRAPE_ALFS
        d = self.dialog
        return (d.scene is not None and d.viewport is not None and d.drape_combo.currentData() == DRAPE_ALFS)

    def surface_available(self, surface):
        self._surface = surface
        self.surface_combo.model().item(1).setEnabled(surface is not None)
        if surface is None and self.surface_combo.currentIndex() == 1:
            self.surface_combo.setCurrentIndex(0)

    def _focal_surface(self):
        surface = getattr(self, "_surface", None)
        if self.surface_combo.currentData() == "recon" and surface is not None:
            return surface
        return self.dialog.display_dem or self.dialog.dem

    def flight_loaded(self):
        self.surface_available(None)
        self.version += 1
        self.result = None
        self._schedule(0)

    def cameras_changed(self):
        self.version += 1
        self._schedule(30)

    def frame_changed(self):
        self._schedule(0)

    def view_changed(self):
        # A render from the current view stays where it was rendered until
        # the next one; only Continuous follows the navigation.
        if self.mode_combo.currentData() == MODE_VIEW \
                and self.continuous_check.isChecked():
            self._schedule(250)

    def _on_focus(self, value):
        self.focus_slider.blockSignals(True)
        self.focus_slider.setValue(int(round(value * 10)))
        self.focus_slider.blockSignals(False)
        self._changed()

    def _changed(self, *_):
        self._schedule(60)

    def _schedule(self, ms):
        if self.live_check.isChecked() and self._active():
            self._debounce.start(ms)
        else:
            self.dialog._refresh_layers()

    def frames(self):
        d = self.dialog
        n = len(d.scene)
        return fv.alfs_window(d.index, n, self.n_spin.value(),
                              self.stride_spin.value(),
                              exclude=d.scene.duplicated())

    def render_now(self):
        if not self._active():
            return
        if self._busy:                 # newest request wins once idle
            self._debounce.start(40)
            return
        d = self.dialog
        if self.worker is None:
            self.worker = AlfsWorker(self)
            self.worker.rendered.connect(self._on_rendered)
            self.worker.failed.connect(self._on_failed)
            self.worker.start()
        indices = self.frames()
        cams = [d.scene.camera(i) for i in indices]
        max_edge = self.res_combo.currentData()
        req = dict(
            serial=self._next_serial(), version=self.version,
            mode=self.mode_combo.currentData(), dem=self._focal_surface(),
            focus=self.focus_spin.value(),
            min_coverage=float(self.threshold_spin.value()),
            resolution=max_edge, max_edge=max_edge,
            mask_path=d.mask_path() if d.use_mask() else "",
            mask_erosion=d.mask_erosion(),
            frames=[dict(key=(d.modality, i), path=d.scene.frame_path(i),
                         camera=c) for i, c in zip(indices, cams)],
            indices=indices, modality=d.modality, index=d.index)
        if req["mode"] == MODE_ORTHO:
            extent = fv.window_extent(cams, d.caster)
            if extent is None:
                self.info.setText("The window's frames do not see the DEM.")
                return
            req["extent"] = extent
            lo, hi = d.dem.bounds
            req["top"] = float(max(hi[2], max(c.position[2] for c in cams))) \
                + abs(self.focus_spin.value())
            req["bottom"] = float(lo[2]) - abs(self.focus_spin.value())
        else:
            vp = d.viewport
            cam = vp.screen_camera()
            w, h = geo.scaled_size(vp.width(), vp.height(), max_edge)
            cam = fv.FrameCamera(cam.view, cam.proj, cam.fovy, w / float(h))
            req["camera"], req["viewport"] = cam, (w, h)
        self._busy = True
        self.info.setText(f"Rendering {len(indices)} frames…")
        self.worker.submit(req)

    def _next_serial(self):
        self._serial += 1
        return self._serial

    def _on_rendered(self, result):
        self._busy = False
        d = self.dialog
        if d.viewport is None:
            return
        self.result = result
        key = ("alfs", result["serial"])
        d.viewport.set_texture(key, result["image"])
        old = getattr(self, "_texture_key", None)
        self._texture_key = key
        d._refresh_layers()
        if old is not None and old != key:
            d.viewport.drop_texture(old)
        self.export_btn.setEnabled(True)
        d.corrections_panel.alfs_rendered(result)
        idx = result["indices"]
        span = f"frames {idx[0]}–{idx[-1]}" if idx else "no frames"
        self.info.setText(
            f"{result['used']} frames ({span}, stride {self.stride_spin.value()}), "
            f"focus {result['focus']:+.1f} m, "
            f"{result['image'].shape[1]}×{result['image'].shape[0]} px in "
            f"{result['seconds'] * 1000:.0f} ms")
        if self.continuous_check.isChecked():
            self._debounce.start(10)

    def _on_failed(self, message):
        self._busy = False
        self.info.setText("ALFS failed: " + message.splitlines()[0])
        self.dialog.status("ALFS failed: " + message.splitlines()[0])

    def layers(self, alpha):
        from .specs import Layer
        r = self.result
        key = getattr(self, "_texture_key", None)
        if r is None or key is None or r.get("modality") != self.dialog.modality:
            return []
        if r["mode"] == MODE_ORTHO:
            matrix = geo.ortho_matrix_for_bounds(*r["extent"])
            return [Layer(key, matrix, alpha)]
        cam = r["camera"]
        return [Layer(key, cam.proj @ cam.view, alpha, z_offset=r["focus"])]

    def _export(self):
        r = self.result
        if r is None:
            return
        from ..core.flight_viewer_io import write_png, write_world_file
        Q = QtWidgets
        d = self.dialog
        default = os.path.join(d.folder or "",
                               f"alfs_{r['modality']}_{r['index']:05d}.png")
        path, _ = Q.QFileDialog.getSaveFileName(self, "Export integral", default,
                                                "PNG image (*.png)")
        if not path:
            return
        write_png(path, r["image"])
        if r["mode"] == MODE_ORTHO:
            lo, hi = r["extent"]
            write_world_file(path, lo, hi, r["image"].shape[1],
                             r["image"].shape[0], d.origin(),
                             d.config.get("target_epsg"))
        d.status(f"Saved {path}")

    def shutdown(self):
        self._debounce.stop()
        if self.worker is not None:
            self.worker.shutdown()
            self.worker = None
