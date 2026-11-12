# -*- coding: utf-8 -*-
"""Corrections tab of the 3D Flight Viewer.

A second way, next to the Correction Wizard, to find a flight's correction
factors: edit the default correction or a frame-range override and watch
the route, frustum, draped frames and ALFS move live; compare two frames far
apart (misaligned poses show as ghosting); read the ALFS sharpness; or let
*Auto-refine* search for the factors that make the ALFS integral sharpest.
Saving writes ``correction.json`` in the wizard's layout (with a backup) and
updates the BAMBI panel.
"""

import math
import time
from typing import List, Optional

import numpy as np

from ..core import flight_corrections as fc
from ..core import flight_viewer as fv
from ..core import flight_viewer_geometry as geo
from .qt import QtCore, QtWidgets, Signal


class RefineWorker(QtCore.QThread):
    """Searches correction factors that maximise ALFS sharpness."""
    progress = Signal(object)      # dict(evals, score, vector)
    finished_search = Signal(object)
    failed = Signal(str)

    def __init__(self, job: dict, parent=None):
        super().__init__(parent)
        self.job = job
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.finished_search.emit(self._search())
        except Exception as exc:  # noqa: BLE001 - reported in the panel
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=4)}")

    def _search(self):
        import cv2
        from alfspy.core.rendering import CtxShot, MeshData, Renderer, Resolution
        from alfspy.core.util.image import to_rgba_f4
        from ..core.render_context import make_render_context

        job = self.job
        ctx = make_render_context()
        images, aspect = job["images"], job["aspect"]
        base, entry = job["correction"], job["entry"]

        pixels = {}
        for i in sorted({i for w in job["windows"] for i in w}):
            img = cv2.imread(job["paths"][i], cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            tw, th = geo.scaled_size(w, h, job["max_edge"])
            pixels[i] = to_rgba_f4(cv2.resize(img, (tw, th),
                                              interpolation=cv2.INTER_AREA))
        mesh = MeshData(np.asarray(job["dem"].vertices, np.float32),
                        np.asarray(job["dem"].faces, np.uint32))

        renderers = []
        for frames, (lo, hi) in zip(job["windows"], job["extents"]):
            size = np.asarray(hi) - np.asarray(lo)
            res = geo.scaled_size(int(size[0] * 1000), int(size[1] * 1000),
                                  job["resolution"])
            cam = fv.alfs_ortho_camera(lo, hi, job["top"], job["bottom"])
            renderers.append(Renderer(Resolution(*res), ctx, cam, mesh))

        mask = None
        if job.get("mask_path"):
            mimg = cv2.imread(job["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mimg is not None:
                mimg = fv.scaled_erosion_mask(mimg, job.get("mask_erosion", 0),
                                              job["max_edge"])
                from alfspy.core.rendering import TextureData
                mask = TextureData(to_rgba_f4(mimg))

        def objective(x):
            t, r = fc.from_vector(x)
            corr = fc.set_entry(base, entry, t, r)
            scores = []
            for frames, renderer in zip(job["windows"], renderers):
                shots = []
                for i in frames:
                    if i not in pixels:
                        continue
                    ti, ri = fv.correction_for(i, corr)
                    cam = fv.camera_for_pose(images[i], ti, ri, aspect)
                    pos, rot = fv.alfs_pose(cam)
                    shots.append(CtxShot(ctx, pixels[i], pos, rot, cam.fovy,
                                         aspect, None))
                if len(shots) < 2:
                    continue
                res = renderer.render_integral_raw(shots, release_shots=True,
                                                   mask=mask)
                s = fc.sharpness(fv.integral_to_rgba(res.accum, res.coverage, 2))
                if np.isfinite(s):
                    scores.append(s)
            return -float(np.mean(scores)) if scores else 0.0

        t0, r0 = fc.get_entry(base, entry)
        x0 = fc.to_vector(t0, r0)
        start = objective(x0)

        def report(x, value, evals):
            self.progress.emit(dict(vector=x, score=-value, evals=evals))

        x, best, evals = fc.pattern_search(
            objective, x0, job["steps"], active=job["active"],
            max_evals=job["max_evals"], callback=report,
            should_stop=lambda: self._stop)
        return dict(vector=x, score=-best, start_score=-start, evals=evals,
                    entry=entry, stopped=self._stop)


class CorrectionsPanel(QtWidgets.QWidget):
    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.correction = fc.normalized({})
        self.saved = fc.normalized({})
        self.entry: Optional[int] = None
        self.undo: List[dict] = []
        self.worker: Optional[RefineWorker] = None
        self.proposal: Optional[dict] = None
        self.scores: List[float] = []
        self._apply_timer = QtCore.QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply)
        self._loading = False
        self._build()

    # -- UI --------------------------------------------------------------------

    def _build(self):
        Q = QtWidgets
        lay = Q.QVBoxLayout(self)
        intro = Q.QLabel(
            "Adjust the correction factors and watch the route, frames and "
            "ALFS move. Correct values make the ALFS integral sharp and two "
            "distant frames line up.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        lay.addWidget(intro)

        box = Q.QGroupBox("Correction")
        f = Q.QFormLayout(box)
        row = Q.QHBoxLayout()
        self.entry_combo = Q.QComboBox()
        self.entry_combo.currentIndexChanged.connect(self._on_entry)
        row.addWidget(self.entry_combo, 1)
        b = Q.QToolButton()
        b.setText("+")
        b.setToolTip("Add a frame range starting at the current frame")
        b.clicked.connect(self._add_range)
        row.addWidget(b)
        self.remove_btn = Q.QToolButton()
        self.remove_btn.setText("−")
        self.remove_btn.setToolTip("Remove this frame range")
        self.remove_btn.clicked.connect(self._remove_range)
        row.addWidget(self.remove_btn)
        f.addRow("Edit", row)
        self.range_row = Q.QHBoxLayout()
        self.start_spin, self.end_spin = Q.QSpinBox(), Q.QSpinBox()
        for s in (self.start_spin, self.end_spin):
            s.setRange(0, 10 ** 6)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._on_range_edit)
            self.range_row.addWidget(s)
        f.addRow("Frames", self.range_row)
        self.uses_label = Q.QLabel("")
        self.uses_label.setStyleSheet("color: gray;")
        f.addRow(self.uses_label)

        self.spins = {}
        for key, label, unit, lim, step, dec in (
                ("tx", "Translation x", " m", 200, 0.05, 3),
                ("ty", "Translation y", " m", 200, 0.05, 3),
                ("tz", "Translation z", " m", 200, 0.05, 3),
                ("rx", "Rotation x", "°", 45, 0.05, 3),
                ("ry", "Rotation y", "°", 45, 0.05, 3),
                ("rz", "Rotation z", "°", 180, 0.1, 3)):
            s = Q.QDoubleSpinBox()
            s.setRange(-lim, lim)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setSuffix(unit)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(self._on_spin)
            self.spins[key] = s
            f.addRow(label, s)
        row = Q.QHBoxLayout()
        for text, slot, tip in (
                ("Undo", self._undo, "Undo the last change"),
                ("Revert", self._revert, "Back to the values last saved"),
                ("Save…", self._save, "Write correction.json and update the "
                                      "BAMBI panel")):
            b = Q.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        f.addRow(row)
        self.dirty_label = Q.QLabel("")
        f.addRow(self.dirty_label)
        lay.addWidget(box)

        box = Q.QGroupBox("Check")
        f = Q.QFormLayout(box)
        self.compare_check = Q.QCheckBox("Compare two frames")
        self.compare_check.setToolTip(
            "Drape the frames k before and k after the current one at half "
            "opacity. With wrong corrections the terrain shows twice.")
        self.compare_check.toggled.connect(self._on_compare)
        f.addRow(self.compare_check)
        self.compare_spin = Q.QSpinBox()
        self.compare_spin.setRange(1, 2000)
        self.compare_spin.setValue(30)
        self.compare_spin.valueChanged.connect(self._on_compare)
        f.addRow("k (frames)", self.compare_spin)
        self.sharp_label = Q.QLabel("ALFS sharpness: render ALFS to see it")
        self.sharp_label.setWordWrap(True)
        f.addRow(self.sharp_label)
        self.plot = None
        try:
            import pyqtgraph as pg
            self.plot = pg.PlotWidget()
            self.plot.setMinimumHeight(90)
            self.plot.setMaximumHeight(130)
            self.plot.setBackground(None)
            self.plot.showGrid(y=True, alpha=0.2)
            self.plot.setLabel("left", "sharpness")
            self.plot.setLabel("bottom", "ALFS render")
            self.plot.setMouseEnabled(x=False, y=False)
            self._curve = self.plot.plot([], [], pen=pg.mkPen((240, 160, 40), width=2),
                                         symbol="o", symbolSize=4)
            f.addRow(self.plot)
        except Exception:  # noqa: BLE001 - the plot is optional
            self.plot = None
        lay.addWidget(box)

        box = Q.QGroupBox("Auto-refine")
        f = Q.QFormLayout(box)
        grid = Q.QGridLayout()
        self.active = {}
        for n, key in enumerate(fc.PARAMS):
            c = Q.QCheckBox(key)
            c.setChecked(key in ("tx", "ty", "rz"))
            c.setToolTip(fc.PARAM_LABELS[key])
            self.active[key] = c
            grid.addWidget(c, n // 3, n % 3)
        f.addRow("Optimise", grid)
        self.windows_spin = Q.QSpinBox()
        self.windows_spin.setRange(1, 9)
        self.windows_spin.setValue(3)
        self.windows_spin.setToolTip(
            "Frame windows evaluated: the current one plus windows spread "
            "along the flight (or along the range being edited). Window size "
            "and stride come from the ALFS tab.")
        f.addRow("Windows", self.windows_spin)
        self.evals_spin = Q.QSpinBox()
        self.evals_spin.setRange(10, 2000)
        self.evals_spin.setValue(120)
        f.addRow("Max. renders", self.evals_spin)
        row = Q.QHBoxLayout()
        self.refine_btn = Q.QPushButton("Start")
        self.refine_btn.clicked.connect(self._toggle_refine)
        row.addWidget(self.refine_btn)
        self.accept_btn = Q.QPushButton("Apply result")
        self.accept_btn.setEnabled(False)
        self.accept_btn.clicked.connect(self._accept_proposal)
        row.addWidget(self.accept_btn)
        f.addRow(row)
        self.refine_label = Q.QLabel("")
        self.refine_label.setWordWrap(True)
        self.refine_label.setStyleSheet("color: gray;")
        f.addRow(self.refine_label)
        lay.addWidget(box)
        lay.addStretch(1)
        # Comfortable row heights (the tab scrolls when the window is small).
        Q = QtWidgets
        for form in self.findChildren(Q.QFormLayout):
            form.setVerticalSpacing(8)
        for w in (self.findChildren(Q.QAbstractSpinBox)
                  + self.findChildren(Q.QComboBox) + self.findChildren(Q.QPushButton)):  # noqa: W503
            w.setMinimumHeight(28)
        self._fill_entries()

    # -- dialog hooks ----------------------------------------------------------

    def initial_correction(self, config: dict) -> dict:
        """The correction the BAMBI panel currently uses for the flight."""
        self.correction = fc.normalized(fv.correction_from_config(config))
        self.saved = fc.normalized(self.correction)
        self.undo.clear()
        self.entry = None
        self._fill_entries()
        return self.correction

    def flight_loaded(self):
        self.frame_changed()

    def frame_changed(self):
        d = self.dialog
        idx = fc.entry_for_frame(self.correction, d.index)
        self.uses_label.setText(
            f"Frame {d.index} uses: " + ("the default" if idx is None
                                         else self._entry_name(idx)))

    def layers(self):
        from .specs import Layer
        d = self.dialog
        if not self.compare_check.isChecked() or d.scene is None \
                or d.viewport is None:
            return []
        out = []
        for i in self._compare_frames():
            if 0 <= i < len(d.scene):
                key = d._frame_key(i)
                if not d.viewport.has_texture(key):
                    arr = d._request_frame(i)
                    if arr is not None:
                        d._upload_frame(key, arr)
                out.append(Layer(key, d.scene.camera(i).shot_matrix, 0.5))
        return out

    def wanted_textures(self):
        d = self.dialog
        if not self.compare_check.isChecked() or d.scene is None:
            return []
        return [d._frame_key(i) for i in self._compare_frames()
                if 0 <= i < len(d.scene)]

    def _compare_frames(self):
        """The two frames compared: k before and k after the current one,
        moved to the nearest frame that is not a frozen thermal image."""
        d = self.dialog
        k = self.compare_spin.value()
        return [d.scene.usable_frame(i) if 0 <= i < len(d.scene) else i
                for i in (d.index - k, d.index + k)]

    def alfs_rendered(self, result):
        s = fc.sharpness(result["image"])
        if not math.isfinite(s):
            self.sharp_label.setText("ALFS sharpness: too little covered")
            return
        self.scores.append(s)
        del self.scores[:-200]
        if self.plot is not None:
            self._curve.setData(np.arange(len(self.scores)), np.array(self.scores))
        best = max(self.scores)
        self.sharp_label.setText(
            f"ALFS sharpness: {s:,.1f} (best this session {best:,.1f}). "
            "Higher is sharper; compare at the same frame, focus and "
            "resolution.")

    # -- entries ---------------------------------------------------------------

    def _entry_name(self, i):
        a = self.correction["additional"][i]
        return f"Frames {a['start']}–{a['end']}"

    def _fill_entries(self):
        self._loading = True
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.addItem("Default (all other frames)", None)
        for i in range(len(self.correction["additional"])):
            self.entry_combo.addItem(self._entry_name(i), i)
        pos = 0 if self.entry is None else self.entry + 1
        self.entry_combo.setCurrentIndex(min(pos, self.entry_combo.count() - 1))
        self.entry_combo.blockSignals(False)
        self._loading = False
        self._load_spins()

    def _load_spins(self):
        self._loading = True
        t, r = fc.get_entry(self.correction, self.entry)
        for key, val in (("tx", t["x"]), ("ty", t["y"]), ("tz", t["z"]),
                         ("rx", math.degrees(r["x"])),
                         ("ry", math.degrees(r["y"])),
                         ("rz", math.degrees(r["z"]))):
            self.spins[key].setValue(val)
        is_range = self.entry is not None
        self.remove_btn.setEnabled(is_range)
        for s in (self.start_spin, self.end_spin):
            s.setEnabled(is_range)
        if is_range:
            a = self.correction["additional"][self.entry]
            self.start_spin.setValue(a["start"])
            self.end_spin.setValue(a["end"])
        self._loading = False
        self._update_dirty()

    def _on_entry(self):
        self.entry = self.entry_combo.currentData()
        self._load_spins()

    def _push_undo(self):
        self.undo.append(fc.normalized(self.correction))
        del self.undo[:-100]

    def _set(self, correction, push=True):
        if push:
            self._push_undo()
        self.correction = fc.normalized(correction)
        self._apply_timer.start(25)
        self._update_dirty()
        self.frame_changed()

    def _apply(self):
        self.dialog.set_correction(self.correction)

    def _on_spin(self):
        if self._loading:
            return
        v = [self.spins[k].value() for k in fc.PARAMS]
        t = {"x": v[0], "y": v[1], "z": v[2]}
        r = {"x": math.radians(v[3]), "y": math.radians(v[4]),
             "z": math.radians(v[5])}
        self._set(fc.set_entry(self.correction, self.entry, t, r))

    def _on_range_edit(self):
        if self._loading or self.entry is None:
            return
        c = fc.normalized(self.correction)
        a = c["additional"][self.entry]
        a["start"], a["end"] = sorted((self.start_spin.value(),
                                       self.end_spin.value()))
        self._set(c)
        self.entry_combo.setItemText(self.entry + 1, self._entry_name(self.entry))

    def _add_range(self):
        d = self.dialog
        n = len(d.scene) if d.scene is not None else 1
        start = d.index
        end = min(n - 1, start + 100)
        t, r = fc.get_entry(self.correction, fc.entry_for_frame(self.correction,
                                                                start))
        c, idx = fc.add_range(self.correction, start, end, t, r)
        self.entry = idx
        self._set(c)
        self._fill_entries()

    def _remove_range(self):
        if self.entry is None:
            return
        c = fc.remove_range(self.correction, self.entry)
        self.entry = None
        self._set(c)
        self._fill_entries()

    def _undo(self):
        if not self.undo:
            return
        self._set(self.undo.pop(), push=False)
        if self.entry is not None and self.entry >= len(self.correction["additional"]):
            self.entry = None
        self._fill_entries()

    def _revert(self):
        self._set(self.saved)
        self.entry = None
        self._fill_entries()

    def _update_dirty(self):
        changed = fc.normalized(self.correction) != fc.normalized(self.saved)
        self.dirty_label.setText("Unsaved changes" if changed else "")
        self.dirty_label.setStyleSheet("color: #b06000;" if changed else "")

    def _on_compare(self):
        d = self.dialog
        for key in self.wanted_textures():
            arr = d._request_frame(key[2])
            if arr is not None and d.viewport is not None:
                d._upload_frame(key, arr)
        d._refresh_layers()

    # -- saving ----------------------------------------------------------------

    def _save(self):
        Q = QtWidgets
        d = self.dialog
        path = fc.correction_path(d.config)
        box = Q.QMessageBox(self)
        box.setWindowTitle("Save correction factors")
        box.setText(f"Write the correction factors to\n{path}\n\n"
                    "The previous file is kept as a timestamped backup next "
                    "to it, and the BAMBI panel is updated.")
        stale = Q.QCheckBox("Mark results computed from the poses as out of "
                            "date (geo-referencing, FoV, ALFS, GeoTIFFs)")
        stale.setChecked(True)
        box.setCheckBox(stale)
        box.setStandardButtons(Q.QMessageBox.StandardButton.Save | Q.QMessageBox.StandardButton.Cancel)
        if box.exec() != Q.QMessageBox.StandardButton.Save:
            return
        try:
            backup = fc.write_correction(path, self.correction)
        except OSError as exc:
            Q.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.saved = fc.normalized(self.correction)
        self._update_dirty()
        flagged = []
        if stale.isChecked() and d.folder:
            try:
                flagged = fc.mark_pose_results_stale(d.folder)
            except Exception as exc:  # noqa: BLE001
                d.status(f"Could not mark results as out of date: {exc}")
        dock = d._dock_widget
        if dock is not None and hasattr(dock, "load_correction_values"):
            try:
                dock.load_correction_values(path)
            except Exception:  # nosec B110
                pass
        msg = f"Saved {path}"
        if backup:
            msg += f" (backup: {backup})"
        if flagged:
            msg += ". Out of date now: " + ", ".join(flagged)
        d.status(msg)

    # -- auto-refine -----------------------------------------------------------

    def _toggle_refine(self):
        if self.worker is not None:
            self.worker.stop()
            self.refine_label.setText("Stopping after the current render…")
            return
        d = self.dialog
        if d.scene is None or d.dem is None:
            return
        active = [self.active[k].isChecked() for k in fc.PARAMS]
        if not any(active):
            self.refine_label.setText("Tick at least one parameter.")
            return
        alfs = d.alfs_panel
        window, stride = alfs.n_spin.value(), alfs.stride_spin.value()
        n = len(d.scene)
        if self.entry is not None:
            a = self.correction["additional"][self.entry]
            lo, hi = a["start"], min(a["end"], n - 1)
            centres = sorted({min(max(int(c), lo), hi) for c in
                              np.linspace(lo, hi, self.windows_spin.value())})
        else:
            centres = fc.refine_windows(n, d.index, self.windows_spin.value(),
                                        window // 2)
        windows = [fv.alfs_window(c, n, window, stride,
                                  exclude=d.scene.duplicated())
                   for c in centres]
        windows = [w for w in windows if len(w) >= 2]
        if not windows:
            self.refine_label.setText(
                "The ALFS window needs at least two frames (N ≥ 2 × stride).")
            return
        extents, keep = [], []
        for w in windows:
            e = fv.window_extent([d.scene.camera(i) for i in w], d.caster)
            if e is not None:
                extents.append(e)
                keep.append(w)
        if not keep:
            self.refine_label.setText("The windows' frames do not see the DEM.")
            return
        lo, hi = d.dem.bounds
        top = float(max(hi[2], np.nanmax(d.scene.positions()[:, 2]))) + 5
        job = dict(images=d.scene.images, aspect=d.scene.aspect,
                   paths=[d.scene.frame_path(i) for i in range(n)],
                   correction=fc.normalized(self.correction), entry=self.entry,
                   windows=keep, extents=extents, top=top,
                   bottom=float(lo[2]) - 5, dem=d.display_dem or d.dem, resolution=512,
                   max_edge=768, steps=fc.default_steps(), active=active,
                   max_evals=self.evals_spin.value(),
                   mask_path=d.mask_path() if d.use_mask() else "",
                   mask_erosion=d.mask_erosion())
        self.worker = RefineWorker(job, self)
        self.worker.progress.connect(self._on_refine_progress)
        self.worker.finished_search.connect(self._on_refine_done)
        self.worker.failed.connect(self._on_refine_failed)
        self.refine_btn.setText("Stop")
        self.accept_btn.setEnabled(False)
        self._refine_started = time.monotonic()
        self.refine_label.setText(
            f"Searching on {len(keep)} window(s) of {len(keep[0])} frames…")
        self.worker.start()

    def _describe(self, v):
        return (f"t = ({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f}) m, "
                f"r = ({math.degrees(v[3]):+.2f}, {math.degrees(v[4]):+.2f}, "
                f"{math.degrees(v[5]):+.2f})°")

    def _on_refine_progress(self, p):
        self.scores.append(p["score"])
        del self.scores[:-200]
        if self.plot is not None:
            self._curve.setData(np.arange(len(self.scores)), np.array(self.scores))
        self.refine_label.setText(
            f"{p['evals']} renders, sharpness {p['score']:,.1f}: " + self._describe(p["vector"]))

    def _finish_worker(self):
        if self.worker is not None:
            self.worker.wait(2000)
        self.worker = None
        self.refine_btn.setText("Start")

    def _on_refine_done(self, result):
        self._finish_worker()
        self.proposal = result
        gain = (result["score"] / result["start_score"] - 1.0) * 100 \
            if result["start_score"] > 0 else 0.0
        state = "Stopped" if result["stopped"] else "Done"
        self.refine_label.setText(
            f"{state} after {result['evals']} renders "
            f"({time.monotonic() - self._refine_started:.0f} s). Sharpness "
            f"{result['start_score']:,.1f} → {result['score']:,.1f} "
            f"({gain:+.1f} %). Proposed: " + self._describe(result["vector"]))
        self.accept_btn.setEnabled(result["score"] > result["start_score"])

    def _on_refine_failed(self, message):
        self._finish_worker()
        self.refine_label.setText("Auto-refine failed: " + message.splitlines()[0])

    def _accept_proposal(self):
        p = self.proposal
        if p is None:
            return
        t, r = fc.from_vector(p["vector"])
        self.entry = p["entry"]
        self._set(fc.set_entry(self.correction, self.entry, t, r))
        self._fill_entries()
        self.accept_btn.setEnabled(False)

    def save_settings(self):
        pass

    def shutdown(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None
