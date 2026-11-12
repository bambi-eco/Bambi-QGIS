# -*- coding: utf-8 -*-
"""Reconstruction tab of the 3D Flight Viewer.

Builds a 3D reconstruction of a stretch of the flight from its RGB frames
with the DEM as prior (:mod:`core.flight_reconstruction`): sparse points,
a dense 2.5D surface from pairwise stereo, and textures from
the RGB or - linked by capture time - the thermal frames. Results are shown
in the 3D view, written to ``<target>/reconstruction_w/`` and can be added
to QGIS (DSM, height above DEM, textures) or used as ALFS focal surface.
"""

import json
import os
import time
from typing import Optional

import numpy as np

from ..core import flight_reconstruction as fr
from ..core import flight_viewer as fv
from ..core import flight_viewer_geometry as geo
from .qt import QtCore, QtWidgets, Signal

COLOR_RGB = "rgb"
COLOR_THERMAL = "thermal"
COLOR_HEIGHT = "height"


def _load_mask(path: str, width: int, height: int,
               erode_px: int = 0) -> Optional[np.ndarray]:
    """An undistortion mask as a boolean array of the image size, eroded
    by *erode_px* pixels at that size."""
    import cv2
    if not path:
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return fv.mask_for(m, width, height, erode_px)


def _load_rgb(path: str, max_edge: int) -> Optional[np.ndarray]:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    tw, th = geo.scaled_size(w, h, max_edge)
    if (tw, th) != (w, h):
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


class ReconWorker(QtCore.QThread):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, job: dict, parent=None):
        super().__init__(parent)
        self.job = job
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.done.emit(self._run())
        except fr.Cancelled:
            self.failed.emit("Cancelled.")
        except Exception as exc:  # noqa: BLE001 - reported in the panel
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=4)}")

    def _step(self, lo, hi):
        """Progress callback mapping a sub-step's 0..1 into lo..hi."""
        return lambda text, f: self.progress.emit(text, lo + (hi - lo) * f)

    def _run(self):
        job = self.job
        cancel = lambda: self._cancel  # noqa: E731
        rgb = job["scenes"]["w"]
        caster = job["caster"]
        keys = job["keyframes"]
        out = job["out_dir"]
        os.makedirs(out, exist_ok=True)
        result = dict(job=job, out_dir=out, files=[])
        started = time.monotonic()

        views = []
        for n, i in enumerate(keys):
            fr._check(cancel)
            self.progress.emit(f"Loading frame {i}", 0.05 * n / len(keys))
            img = _load_rgb(rgb.frame_path(i), job["max_edge"])
            if img is not None:
                views.append(fr.View(i, rgb.camera(i), img, mask=_load_mask(
                    job["mask_w"], img.shape[1], img.shape[0],
                    job["mask_erosion"])))
        if len(views) < 3:
            raise RuntimeError("At least three readable keyframes are needed.")
        result["views"] = [v.index for v in views]

        # Optional: poses and field of view from the images.
        if job["refine"]:
            self.progress.emit("Refining poses from the images", 0.05)
            extent = fv.window_extent([v.camera for v in views], caster,
                                      pad_fraction=0.2)
            ground = None
            if extent is not None:
                coarse = fr.GroundGrid.fitting(extent[0], extent[1], 2.0, 250_000)
                ground = fr.ground_sampler(coarse, fr.dem_heights(coarse, caster))
            views, report = fr.refine_poses(
                views, refine_fov=job["refine_fov"] and ground is not None,
                ground=ground, progress=self._step(0.05, 0.2), cancel=cancel)
            result["refine"] = report

        p0 = 0.2 if job["refine"] else 0.05
        if job["sparse"]:
            pts, cols, err = fr.sparse_points(
                views, caster, band=(job["band_low"], job["band_high"]),
                progress=self._step(p0, p0 + 0.15), cancel=cancel)
            result["sparse"] = (pts, cols, err)
            from ..core.flight_viewer_io import write_ply
            path = os.path.join(out, "sparse_points.ply")
            write_ply(path, pts, cols)
            result["files"].append(path)
            names = [os.path.basename(rgb.frame_path(v.index)) for v in views]
            fr.write_colmap_model(os.path.join(out, "colmap"), views, names)
            result["files"].append(os.path.join(out, "colmap"))

        if job["dense"]:
            self.progress.emit("Preparing the ground grid", 0.35)
            extent = fv.window_extent([v.camera for v in views], caster,
                                      pad_fraction=0.0)
            if extent is None:
                raise RuntimeError("The keyframes do not see the DEM.")
            grid = fr.GroundGrid.fitting(extent[0], extent[1], job["gsd"],
                                         job["max_cells"])
            dem_z = fr.dem_heights(grid, caster)
            heights = np.arange(job["band_low"], job["band_high"] + 1e-9,
                                job["band_step"])
            # With refined poses the pairs use them; otherwise each pair's
            # relative orientation comes from its own images.
            sweep = fr.stereo_surface(grid, dem_z, views, heights,
                                      from_images=not job["refine"],
                                      progress=self._step(0.35, 0.7),
                                      cancel=cancel)
            result["stereo_pairs"] = sweep.extras["pairs"].tolist()
            # Only where the frames cover the ground (enclosed gaps are
            # closed); beyond that there is nothing to reconstruct.
            covered = fr.covered_area(sweep.views > 0)
            valid = np.isfinite(dem_z) & covered
            surface_z = np.where(valid, dem_z + fr.fill_offsets(sweep.offset,
                                                                valid=valid),
                                 np.nan)
            verts, faces, normals = fr.heightfield_mesh(grid, surface_z)
            result.update(grid=grid, dem_z=dem_z, sweep=sweep,
                          surface_z=surface_z,
                          mesh=(verts, faces, normals))
            surface_caster = None
            if job["occlusion"]:
                self.progress.emit("Building the surface ray caster", 0.72)
                surface_caster = fv.make_caster(fv.DemData(verts, faces, normals))

            def ortho(vs, lo, hi):
                vis = fr.visibility_masks(grid, surface_z, vs, surface_caster,
                                          cancel) if surface_caster else None
                return fr.texture_ortho(grid, surface_z, vs, job["tex_gsd"],
                                        job["tex_edge"], vis,
                                        progress=self._step(lo, hi),
                                        cancel=cancel)
            if job["texture_rgb"]:
                result["tex_grid"], result["rgb"] = ortho(views, 0.74, 0.85)
            thermal = job["scenes"].get("t")
            if job["texture_thermal"] and thermal is not None and len(thermal):
                idx = sorted({fv.matching_frame(rgb.images, thermal.images, i)
                              for i in range(job["start"], job["end"] + 1,
                                             max(1, job["thermal_step"]))})
                tviews = []
                for i in idx:
                    fr._check(cancel)
                    img = _load_rgb(thermal.frame_path(i), job["max_edge"])
                    if img is not None:
                        tviews.append(fr.View(i, thermal.camera(i), img,
                                              mask=_load_mask(job["mask_t"],
                                                              img.shape[1],
                                                              img.shape[0],
                                                              job["mask_erosion"])))
                if tviews:
                    result["thermal_grid"], result["thermal"] = \
                        ortho(tviews, 0.85, 0.95)
                    result["thermal_frames"] = idx
            self.progress.emit("Writing results", 0.96)
            result["files"] += self._write_dense(result, job)
            result["files"] += self._write_nadir(result, job, caster)

        meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    seconds=round(time.monotonic() - started, 1),
                    modality="w",
                    correction=rgb.correction, keyframes=result["views"],
                    frame_range=[job["start"], job["end"]],
                    origin=list(job["origin"]) if job["origin"] is not None else None,
                    epsg=job["epsg"],
                    parameters={k: job[k] for k in (
                        "spacing", "max_edge", "gsd", "band_low", "band_high",
                        "band_step", "occlusion", "tex_gsd", "refine",
                        "refine_fov")},
                    method="pairwise stereo",
                    pose_refinement=(vars(result["refine"])
                                     if "refine" in result else None),
                    note="Coordinates in the PLY / COLMAP model are DEM-local "
                         "(add 'origin' for the project CRS).")
        with open(os.path.join(out, "recon.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=float)
        result["files"].append(os.path.join(out, "recon.json"))
        result["seconds"] = meta["seconds"]
        self.progress.emit("Done", 1.0)
        return result

    def _write_nadir(self, result, job, caster):
        """Nadir distance of every camera position (both cameras) to the
        reconstructed surface and to the DEM, as CSV (the AGL tool shows
        the surface from surface.npz itself)."""
        files = []
        surface = fr.ground_sampler(result["grid"], result["surface_z"])

        def dem_height(x, y):
            hits = fv.cast_rays(caster, np.column_stack(
                [x, y, np.full(len(x), 1e4)]), np.tile([0.0, 0.0, -1.0], (len(x), 1)))
            return hits[:, 2]
        for modality, scene in job["scenes"].items():
            if scene is None or not len(scene):
                continue
            try:
                rows = fr.nadir_distances(scene.positions(), surface, dem_height)
                path = os.path.join(job["out_dir"], f"nadir_distances_{modality}.csv")
                fr.write_nadir_csv(path, rows,
                                   [m.get("imagefile", "") for m in scene.images],
                                   job["origin"])
                files.append(path)
            except Exception:  # nosec B112 - an extra; the surface is saved
                continue
        return files

    def _write_dense(self, result, job):
        out, files = job["out_dir"], []
        grid, origin, epsg = result["grid"], job["origin"], job["epsg"]
        z0 = float(origin[2]) if origin is not None else 0.0
        valid = np.isfinite(result["dem_z"])
        rasters = [("dsm.tif", result["surface_z"] + z0, grid),
                   ("ndsm.tif", np.where(valid, result["surface_z"] - result["dem_z"], np.nan), grid),
                   ("confidence.tif", result["sweep"].confidence, grid)]
        if "rgb" in result:
            rasters.append(("rgb.tif", result["rgb"], result["tex_grid"]))
        if "thermal" in result:
            rasters.append(("thermal.tif", result["thermal"],
                            result["thermal_grid"]))
        written = []
        for name, arr, g in rasters:
            path = os.path.join(out, name)
            if fr.write_geotiff(path, arr, g, origin, epsg):
                written.append(path)
        if not written:   # no GDAL: keep the arrays at least
            for name, arr, _ in rasters:
                path = os.path.join(out, name.replace(".tif", ".npy"))
                np.save(path, arr)
                written.append(path)
        files += written
        result["rasters"] = {os.path.basename(p): p for p in written}
        # Everything the viewer needs to show the surface again next time.
        textures = {}
        if "rgb" in result:
            textures["rgb"] = (result["tex_grid"], result["rgb"])
        if "thermal" in result:
            textures["thermal"] = (result["thermal_grid"], result["thermal"])
        try:
            path = os.path.join(out, fr.SURFACE_FILE)
            fr.save_surface(path, grid, result["surface_z"], result["dem_z"],
                            result["sweep"].confidence, result["sweep"].views,
                            textures, origin)
            files.append(path)
        except OSError:
            pass
        try:
            import trimesh
            verts, faces, _ = result["mesh"]
            mesh = trimesh.Trimesh(verts, faces, process=False)
            path = os.path.join(out, "surface.glb")
            mesh.export(path)
            with open(os.path.join(out, "surface.json"), "w", encoding="utf-8") as fh:
                json.dump({"origin": list(origin) if origin is not None else [0, 0, 0],
                           "crs": f"EPSG:{epsg}" if epsg else "",
                           "generator": "BAMBI 3D Flight Viewer reconstruction"},
                          fh, indent=2, default=float)
            files += [path, os.path.join(out, "surface.json")]
        except Exception:  # nosec B110 - the mesh is an extra
            pass
        return files


class ReconstructionPanel(QtWidgets.QWidget):
    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.worker: Optional[ReconWorker] = None
        self.result: Optional[dict] = None
        self._build()

    # -- UI -----------------------------------------------------------------

    def _build(self):
        Q = QtWidgets
        lay = Q.QVBoxLayout(self)
        intro = Q.QLabel(
            "3D reconstruction from the RGB frames with the DEM as prior: "
            "sparse points triangulated with the corrected poses, and a dense "
            "surface from stereo between neighbouring keyframes. The DEM is bare "
            "earth, so the result shows what stands on it (vegetation, "
            "buildings). Thermal frames can texture the surface.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        lay.addWidget(intro)

        box = Q.QGroupBox("Frames (RGB)")
        f = Q.QFormLayout(box)
        row = Q.QHBoxLayout()
        self.start_spin, self.end_spin = Q.QSpinBox(), Q.QSpinBox()
        for s in (self.start_spin, self.end_spin):
            s.setRange(0, 10 ** 6)
            row.addWidget(s)
        b = Q.QToolButton()
        b.setText("Here")
        b.setToolTip("Range around the current frame (±100)")
        b.clicked.connect(self._range_here)
        row.addWidget(b)
        b = Q.QToolButton()
        b.setText("All")
        b.setToolTip("The whole flight (first to last frame)")
        b.clicked.connect(self._range_all)
        row.addWidget(b)
        f.addRow("Range", row)
        self.spacing_spin = Q.QDoubleSpinBox()
        self.spacing_spin.setRange(0.5, 100)
        self.spacing_spin.setValue(4.0)
        self.spacing_spin.setSuffix(" m")
        self.spacing_spin.setToolTip("Distance between keyframes along the route.")
        f.addRow("Keyframe spacing", self.spacing_spin)
        self.edge_spin = Q.QSpinBox()
        self.edge_spin.setRange(320, 8192)
        self.edge_spin.setSingleStep(160)
        self.edge_spin.setValue(1280)
        self.edge_spin.setSuffix(" px")
        f.addRow("Image size", self.edge_spin)
        lay.addWidget(box)

        box = Q.QGroupBox("Surface")
        f = Q.QFormLayout(box)
        self.sparse_check = Q.QCheckBox("Sparse points")
        self.sparse_check.setChecked(True)
        f.addRow(self.sparse_check)
        self.dense_check = Q.QCheckBox("Dense surface (pairwise stereo)")
        self.dense_check.setToolTip(
            "Every keyframe is matched against its next two keyframes as a "
            "rectified stereo pair; each pair's orientation comes from its "
            "images (or from the refined poses) and its scale from the ground "
            "on the DEM. The pairs' depth maps vote for a height per ground "
            "cell and the median wins.")
        self.dense_check.setChecked(True)
        f.addRow(self.dense_check)
        self.gsd_spin = Q.QDoubleSpinBox()
        self.gsd_spin.setRange(0.02, 10)
        self.gsd_spin.setDecimals(2)
        self.gsd_spin.setValue(0.25)
        self.gsd_spin.setSuffix(" m")
        self.gsd_spin.setToolTip("Ground cell size; coarsened automatically "
                                 "above 2.5 million cells.")
        f.addRow("Cell size", self.gsd_spin)
        row = Q.QHBoxLayout()
        self.low_spin, self.high_spin, self.step_spin = (
            Q.QDoubleSpinBox(), Q.QDoubleSpinBox(), Q.QDoubleSpinBox())
        for s, v in ((self.low_spin, -2.0), (self.high_spin, 30.0),
                     (self.step_spin, 0.5)):
            s.setRange(-50, 200)
            s.setDecimals(2)
            s.setValue(v)
            row.addWidget(s)
        self.step_spin.setRange(0.05, 10)
        f.addRow("Heights from / to / step (m)", row)
        lay.addWidget(box)

        box = Q.QGroupBox("Geometry from the images (optional)")
        f = Q.QFormLayout(box)
        self.refine_check = Q.QCheckBox("Refine poses from the images")
        self.refine_check.setToolTip(
            "Before reconstructing, match features between the keyframes and "
            "adjust their positions and rotations so they agree with the "
            "images (the telemetry stays a soft prior; the result is put back "
            "on the DEM). Heights from nearby frames are very sensitive to "
            "small pose errors. Takes longer; needs scipy.")
        f.addRow(self.refine_check)
        self.refine_fov_check = Q.QCheckBox("Also estimate the field of view")
        self.refine_fov_check.setChecked(True)
        self.refine_fov_check.setToolTip(
            "Estimate the camera's effective field of view from the images "
            "and the DEM instead of trusting the poses' fovy. Best effort: "
            "from a single straight pass at one altitude the field of view "
            "is only weakly determined; the pose refinement helps most.")
        self.refine_fov_check.setEnabled(False)
        self.refine_check.toggled.connect(self.refine_fov_check.setEnabled)
        f.addRow(self.refine_fov_check)
        lay.addWidget(box)

        box = Q.QGroupBox("Texture")
        f = Q.QFormLayout(box)
        self.tex_rgb_check = Q.QCheckBox("RGB from the frames (best view)")
        self.tex_rgb_check.setChecked(True)
        f.addRow(self.tex_rgb_check)
        self.tex_thermal_check = Q.QCheckBox("Thermal frames (linked by time)")
        self.tex_thermal_check.setChecked(True)
        f.addRow(self.tex_thermal_check)
        self.tex_gsd_spin = Q.QDoubleSpinBox()
        self.tex_gsd_spin.setRange(0.01, 2.0)
        self.tex_gsd_spin.setDecimals(2)
        self.tex_gsd_spin.setSingleStep(0.01)
        self.tex_gsd_spin.setValue(0.05)
        self.tex_gsd_spin.setSuffix(" m")
        self.tex_gsd_spin.setToolTip(
            "Pixel size of the orthophoto draped on the surface - much finer "
            "than the surface cells, so the model looks sharp on a smooth "
            "shape (at most 4096 pixels along the longer side).")
        f.addRow("Texture pixel size", self.tex_gsd_spin)
        self.occl_check = Q.QCheckBox("Skip cells hidden from a camera")
        self.occl_check.setChecked(True)
        self.occl_check.setToolTip(
            "Ray-cast against the reconstructed surface so e.g. ground behind "
            "a tree is not painted with the tree.")
        f.addRow(self.occl_check)
        lay.addWidget(box)

        row = Q.QHBoxLayout()
        self.run_btn = Q.QPushButton("Reconstruct")
        self.run_btn.clicked.connect(self._toggle)
        row.addWidget(self.run_btn)
        self.progress = Q.QProgressBar()
        self.progress.setRange(0, 1000)
        row.addWidget(self.progress, 1)
        lay.addLayout(row)
        self.info = Q.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray;")
        lay.addWidget(self.info)

        box = Q.QGroupBox("Display")
        f = Q.QFormLayout(box)
        self.show_points = Q.QCheckBox("Sparse points")
        self.show_points.setChecked(True)
        self.show_points.toggled.connect(self.update_display)
        f.addRow(self.show_points)
        self.show_surface = Q.QCheckBox("Surface")
        self.show_surface.setChecked(True)
        self.show_surface.toggled.connect(self.update_display)
        f.addRow(self.show_surface)
        self.color_combo = Q.QComboBox()
        self.color_combo.addItem("RGB", COLOR_RGB)
        self.color_combo.addItem("Thermal", COLOR_THERMAL)
        self.color_combo.addItem("Height above DEM", COLOR_HEIGHT)
        self.color_combo.currentIndexChanged.connect(self.update_display)
        f.addRow("Colour", self.color_combo)
        self.hide_dem_check = Q.QCheckBox("Hide the DEM")
        self.hide_dem_check.toggled.connect(self._on_hide_dem)
        f.addRow(self.hide_dem_check)
        row = Q.QHBoxLayout()
        self.qgis_btn = Q.QPushButton("Add rasters to QGIS")
        self.qgis_btn.clicked.connect(self._add_to_qgis)
        self.qgis_btn.setEnabled(False)
        row.addWidget(self.qgis_btn)
        self.open_btn = Q.QPushButton("Open folder")
        self.open_btn.clicked.connect(self._open_folder)
        self.open_btn.setEnabled(False)
        row.addWidget(self.open_btn)
        f.addRow(row)
        lay.addWidget(box)
        lay.addStretch(1)

    # -- dialog hooks ---------------------------------------------------------

    def flight_loaded(self):
        self.result = None                 # nothing of another flight
        self.update_display()
        self._range_all()
        self.tex_thermal_check.setEnabled("t" in self.dialog.scenes)
        self._load_previous()

    def _range_all(self):
        """Default: the whole flight, first to last RGB frame."""
        d = self.dialog
        scene = d.scenes.get("w") or d.scene
        if scene is None:
            return
        self.start_spin.setValue(0)
        self.end_spin.setValue(max(0, len(scene) - 1))

    def _range_here(self):
        d = self.dialog
        if d.scene is None:
            return
        n = len(d.scenes.get("w", d.scene))
        idx = d.index if d.modality == "w" else fv.matching_frame(
            d.scene.images, d.scenes["w"].images, d.index) if "w" in d.scenes else 0
        self.start_spin.setValue(max(0, idx - 100))
        self.end_spin.setValue(min(n - 1, idx + 100))

    def _out_dir(self):
        return os.path.join(self.dialog.folder, "reconstruction_w")

    def _load_previous(self):
        """Show the result of an earlier run: sparse points and the dense
        surface with its textures, when they are there."""
        out = self._out_dir()
        result = {"out_dir": out, "files": []}
        parts = []
        ply = os.path.join(out, "sparse_points.ply")
        if os.path.isfile(ply):
            try:
                from ..core.flight_viewer_io import read_ply
                pts, cols = read_ply(ply)
                result["sparse"] = (pts, cols, None)
                result["files"].append(ply)
                parts.append(f"{len(pts):,} sparse points")
            except Exception:  # nosec B110 - an unreadable file is skipped
                pass
        npz = os.path.join(out, fr.SURFACE_FILE)
        if os.path.isfile(npz):
            try:
                data = fr.load_surface(npz)
                origin = self.dialog.dem.origin if self.dialog.dem else None
                stored = data["origin"]
                same_place = origin is None or not np.all(np.isfinite(stored)) \
                    or np.allclose(stored, origin, atol=0.01)
                if same_place:
                    grid, surface_z = data["grid"], data["surface_z"]
                    result.update(
                        grid=grid, surface_z=surface_z, dem_z=data["dem_z"],
                        mesh=fr.heightfield_mesh(grid, surface_z),
                        sweep=fr.DenseResult(
                            surface_z - data["dem_z"], data["confidence"],
                            data["views"], np.zeros(surface_z.shape + (3,),
                                                    np.float32), np.zeros(0)))
                    for key in ("rgb", "tex_grid", "thermal", "thermal_grid"):
                        if key in data:
                            result[key] = data[key]
                    result["rasters"] = {
                        name: os.path.join(out, name)
                        for name in ("dsm.tif", "ndsm.tif", "confidence.tif",
                                     "rgb.tif", "thermal.tif")
                        if os.path.isfile(os.path.join(out, name))}
                    result["files"].append(npz)
                    parts.append(f"surface {grid.width}×{grid.height} cells")
            except Exception:  # nosec B110 - an unreadable file is skipped
                pass
        if not parts:
            return
        self.result = result
        self.info.setText("Earlier result: " + ", ".join(parts) + f" in {out}.")
        self.open_btn.setEnabled(True)
        self.qgis_btn.setEnabled(bool(result.get("rasters")))
        self.update_display()
        self.dialog.alfs_panel.surface_available(self.surface())

    # -- running -------------------------------------------------------------

    def _toggle(self):
        if self.worker is not None:
            self.worker.cancel()
            self.info.setText("Cancelling…")
            return
        d = self.dialog
        rgb = d.scenes.get("w")
        if rgb is None:
            self.info.setText("This flight has no RGB frames; the "
                              "reconstruction needs them.")
            return
        start, end = sorted((self.start_spin.value(),
                             min(self.end_spin.value(), len(rgb) - 1)))
        keys = fr.select_keyframes(rgb.positions(), range(start, end + 1),
                                   self.spacing_spin.value())
        if len(keys) < 3:
            self.info.setText("Fewer than three keyframes in the range - "
                              "widen it or reduce the keyframe spacing.")
            return
        if rgb.image_size is None:
            self.info.setText("The RGB frames could not be read.")
            return
        job = dict(scenes=d.scenes, caster=d.caster, keyframes=keys,
                   start=start, end=end, out_dir=self._out_dir(),
                   spacing=self.spacing_spin.value(),
                   max_edge=self.edge_spin.value(), gsd=self.gsd_spin.value(),
                   max_cells=self._max_cells(), band_low=self.low_spin.value(),
                   band_high=max(self.high_spin.value(), self.low_spin.value()),
                   band_step=self.step_spin.value(),
                   sparse=self.sparse_check.isChecked(),
                   dense=self.dense_check.isChecked(),
                   texture_rgb=self.tex_rgb_check.isChecked(),
                   texture_thermal=self.tex_thermal_check.isChecked() and "t" in d.scenes,
                   thermal_step=max(1, (end - start) // 60),
                   occlusion=self.occl_check.isChecked(),
                   tex_gsd=self.tex_gsd_spin.value(), tex_edge=4096,
                   refine=self.refine_check.isChecked(),
                   refine_fov=self.refine_fov_check.isChecked(),
                   mask_w=d.mask_path("w") if d.use_mask("w") else "",
                   mask_t=d.mask_path("t") if d.use_mask("t") else "",
                   mask_erosion=d.mask_erosion(),
                   origin=None if d.dem.origin is None else tuple(d.dem.origin),
                   epsg=d.config.get("target_epsg"))
        self.worker = ReconWorker(job, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.run_btn.setText("Cancel")
        self.info.setText(f"{len(keys)} keyframes from frames {start}–{end}…")
        self.worker.start()

    def _max_cells(self) -> int:
        """Ground cells the surface may have: the pairs' votes (cells x
        height bins) are kept in memory."""
        n = int((max(self.high_spin.value(), self.low_spin.value())
                 - self.low_spin.value()) / max(self.step_spin.value(), 1e-3)) + 1  # noqa: W503
        return int(max(50_000, min(2_500_000, 60_000_000 / max(n, 1))))

    def _on_progress(self, text, fraction):
        self.progress.setValue(int(fraction * 1000))
        self.info.setText(text)

    def _finish(self):
        if self.worker is not None:
            self.worker.wait(2000)
        self.worker = None
        self.run_btn.setText("Reconstruct")

    def _on_failed(self, message):
        self._finish()
        self.progress.setValue(0)
        self.info.setText("Reconstruction stopped: " + message.splitlines()[0])

    def _on_done(self, result):
        self._finish()
        self.result = result
        parts = [f"{len(result['views'])} keyframes"]
        if "sparse" in result:
            parts.append(f"{len(result['sparse'][0]):,} sparse points")
        if "refine" in result:
            rep = result["refine"]
            fov = (f", field of view {rep.fovy_before:.1f}° → {rep.fovy_after:.1f}°"
                   if self.refine_fov_check.isChecked() else "")
            parts.append(
                f"poses refined: reprojection {rep.rms_before:.1f} → "
                f"{rep.rms_after:.1f} px, moved {rep.median_shift:.2f} m / "
                f"{rep.median_turn:.2f}° (median){fov}")
        if "stereo_pairs" in result:
            total, used, from_img = result["stereo_pairs"]
            oriented = f" ({from_img} oriented from the images)" if from_img else ""
            parts.append(f"pairwise stereo: {used} of {total} pairs{oriented}")
        if "sweep" in result:
            g = result["grid"]
            above = result["surface_z"] - result["dem_z"]
            sure = result["sweep"].confidence >= 0.15
            parts.append(f"surface {g.width}×{g.height} cells at {g.gsd:.2f} m "
                         f"({sure.mean() * 100:.0f} % matched with confidence)")
            if np.isfinite(above).any():
                parts.append(f"height above DEM up to "
                             f"{np.nanpercentile(above, 99):.1f} m (99th pct.)")
        if "tex_grid" in result:
            parts.append(f"texture {result['tex_grid'].width}×"
                         f"{result['tex_grid'].height} px at "
                         f"{result['tex_grid'].gsd:.2f} m")
        if "thermal" in result:
            parts.append(f"thermal texture from {len(result['thermal_frames'])} frames")
        self.info.setText(", ".join(parts) + f" in {result['seconds']:.0f} s. "
                          f"Written to {result['out_dir']}.")
        self.qgis_btn.setEnabled("rasters" in result)
        self.open_btn.setEnabled(True)
        self.update_display()
        self.dialog.alfs_panel.surface_available(self.surface())

    # -- display ---------------------------------------------------------------

    def pick_caster(self):
        """A ray caster over the reconstructed surface while it is shown
        (for mouse picking), else None. Built once per result."""
        r = self.result or {}
        if "mesh" not in r or not self.show_surface.isChecked():
            return None
        if r.get("_pick_caster") is None:
            v, f, n = r["mesh"]
            try:
                r["_pick_caster"] = fv.make_caster(fv.DemData(v, f, n))
            except Exception:  # noqa: BLE001 - picking falls back to the DEM
                return None
        return r["_pick_caster"]

    def surface(self) -> Optional[fv.DemData]:
        """The reconstructed surface as a mesh (for ALFS), or None."""
        r = self.result
        if not r or "mesh" not in r:
            return None
        v, f, n = r["mesh"]
        return fv.DemData(v, f, n, origin=self.dialog.dem.origin)

    def update_display(self):
        from .specs import PointSpec, SurfaceSpec
        d = self.dialog
        vp = d.viewport
        if vp is None:
            return
        r = self.result or {}
        mode = self.color_combo.currentData()
        if "sparse" in r and self.show_points.isChecked() and len(r["sparse"][0]):
            pts, cols, _ = r["sparse"]
            if mode == COLOR_HEIGHT and d.caster is not None:
                rgb = fr.height_colors(fr.height_above(pts, d.caster))
            else:
                rgb = np.asarray(cols, float) / 255.0
            colors = np.column_stack([rgb, np.ones(len(pts))])
            vp.set_points("recon_points", PointSpec(pts, colors, size=3.0))
        else:
            vp.set_points("recon_points", None)
        if "mesh" in r and self.show_surface.isChecked():
            v, f, n = r["mesh"]
            texture = matrix = None
            src = {COLOR_RGB: ("rgb", "tex_grid"),
                   COLOR_THERMAL: ("thermal", "thermal_grid")}.get(mode)
            if src and src[0] in r:
                texture = r[src[0]]
                g = r[src[1]]
                matrix = geo.ortho_matrix_for_bounds(g.lower, g.upper)
                colors = np.ones((len(v), 4), np.float32)
            else:
                rgb = fr.height_colors((r["surface_z"] - r["dem_z"]).ravel())
                colors = np.column_stack([rgb, np.ones(len(rgb))]).astype(np.float32)
            lifted = v.copy()
            lifted[:, 2] += 0.03          # above the DEM where they coincide
            vp.set_surface("recon_surface",
                           SurfaceSpec(lifted, f, n, colors, shade=0.3,
                                       texture=texture, texture_matrix=matrix))
        else:
            vp.set_surface("recon_surface", None)

    def _on_hide_dem(self, hidden):
        if self.dialog.viewport is not None:
            self.dialog.viewport.set_visible("dem", not hidden)

    def _add_to_qgis(self):
        r = self.result or {}
        rasters = r.get("rasters", {})
        try:
            from qgis.core import QgsProject, QgsRasterLayer
        except ImportError:
            return
        added = 0
        for name, path in rasters.items():
            if not path.endswith(".tif"):
                continue
            # From the RGB frames: named for that camera, so it lands in the
            # flight's RGB group at its place in the layer order.
            layer = QgsRasterLayer(path, f"BAMBI Reconstruction {name[:-4]} (RGB)")
            if layer.isValid():
                dock = self.dialog._dock_widget
                if dock is not None and hasattr(dock, "_add_layer_to_flight"):
                    with dock._layers_for_flight(self.dialog.folder):
                        dock._add_layer_to_flight(layer)
                else:
                    QgsProject.instance().addMapLayer(layer)
                added += 1
        self.dialog.status(f"Added {added} reconstruction raster(s) to QGIS.")

    def _open_folder(self):
        from .qt import QtGui
        path = (self.result or {}).get("out_dir") or self._out_dir()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def save_settings(self):
        pass

    def shutdown(self):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(5000)
            self.worker = None
