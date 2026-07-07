# -*- coding: utf-8 -*-
"""Headless camera-calibration computation of the Calibration Wizard.

Moved from ``bambi_camera_calibration.py`` (whose helpers re-export and
whose ``_CalibWorker`` delegates here):

* image/video frame helpers and intrinsics estimation,
* :func:`run_single_calibration` — single-camera SfM via pycolmap,
* :func:`run_stereo_calibration` — RGB↔thermal alignment via homography
  check + Nelder-Mead reprojection minimisation (mirrors the standalone
  camera-calib toolkit).

cv2 / scipy / pycolmap are imported lazily so the module stays importable
in the QGIS-free unit-test environment.
"""

import math
import os
import shutil
import tempfile
from typing import List, Optional

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg'}
_MIN_SFM_IMAGES = 3
_MIN_STEREO_POINTS = 4
_DEFAULT_SFM_FRAMES = 20


def _noop(*args, **kwargs):
    pass


def _never_cancelled():
    return False


# ---------------------------------------------------------------------------
# Image / intrinsics helpers
# ---------------------------------------------------------------------------

def load_image_or_video_central(path: str) -> Optional["object"]:
    """Return central frame from video, or full image from image file."""
    import cv2

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


def extract_n_frames_to_dir(video_path: str, out_dir: str, n: int,
                            progress_cb=None) -> List[str]:
    """Extract *n* evenly-spaced frames from *video_path* into *out_dir*.

    *progress_cb*, if given, is called as ``progress_cb(frames_done, frames_total)``
    after each frame is written so callers can update a progress indicator.
    Returns list of written file paths."""
    import cv2

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


def fov_diag_to_mtx(fov_diag_deg: float, w: int, h: int) -> list:
    """Compute a 3×3 camera matrix from a diagonal FoV (degrees) and image size.

    DJI specs report the diagonal FoV.  The focal length is derived as:
        f = (diagonal_px / 2) / tan(fov_diag / 2)
    with the principal point placed at the image centre.
    """
    fov_rad = math.radians(fov_diag_deg)
    d = math.sqrt(w * w + h * h)
    f = (d / 2.0) / math.tan(fov_rad / 2.0)
    return [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]


def undistort_img(img, calib: dict):
    """Return undistorted copy of *img* using calibration *calib* {mtx/K, dist}.
    Falls back to the original image if anything goes wrong."""
    try:
        import cv2
        import numpy as np

        mtx = np.array(calib.get("mtx") or calib.get("K"), dtype=np.float64)
        dist = np.array(calib["dist"], dtype=np.float64).flatten()
        return cv2.undistort(img, mtx, dist)
    except Exception:
        return img


def estimate_intrinsics_from_image(img) -> dict:
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
# Single-camera SfM via pycolmap
# ---------------------------------------------------------------------------

def run_single_calibration(params: dict, log_fn=None, progress_fn=None,
                           cancel_check=None) -> dict:
    log = log_fn or _noop
    progress = progress_fn or _noop
    cancelled = cancel_check or _never_cancelled

    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("opencv-python (cv2) is required but not installed.")
    try:
        import pycolmap
    except ImportError:
        raise RuntimeError(
            "pycolmap is required for single-camera SfM calibration.\n\n"
            "Install it with:\n  pip install pycolmap"
        )

    image_paths: List[str] = params["image_paths"]
    camera_name: str = params.get("camera_name", "Camera")

    if len(image_paths) < _MIN_SFM_IMAGES:
        raise RuntimeError(
            f"SfM calibration requires at least {_MIN_SFM_IMAGES} images, "
            f"got {len(image_paths)}.\nAdd more images or extract more frames from the video."
        )

    log(f"Preparing {len(image_paths)} images…")
    progress(5)

    tmpdir = tempfile.mkdtemp(prefix="bambi_calib_")
    try:
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir)
        for i, src in enumerate(image_paths):
            ext = os.path.splitext(src)[1].lower() or ".jpg"
            dst = os.path.join(img_dir, f"img_{i:04d}{ext}")
            shutil.copy2(src, dst)
            if cancelled():
                return {}

        db_path = os.path.join(tmpdir, "database.db")
        out_dir = os.path.join(tmpdir, "sparse")
        os.makedirs(out_dir)

        log("Extracting image features (SIFT)…")
        progress(15)

        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = "OPENCV"

        fov_diag = params.get("fov_diag", 0.0)
        if fov_diag > 0:
            first_img = cv2.imread(image_paths[0])
            if first_img is not None:
                h, w = first_img.shape[:2]
                f = (math.sqrt(w * w + h * h) / 2.0) / math.tan(math.radians(fov_diag) / 2.0)
                reader_opts.camera_params = f"{f},{f},{w / 2.0},{h / 2.0},0,0,0,0"
                log(
                    f"FoV prior: {fov_diag}° diagonal → f ≈ {f:.1f} px "
                    f"(vert. FoV ≈ {math.degrees(2 * math.atan(h / (2 * f))):.1f}°)"
                )

        pycolmap.extract_features(
            database_path=db_path,
            image_path=img_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_opts,
        )
        if cancelled():
            return {}

        log("Matching features between image pairs…")
        progress(35)
        pycolmap.match_exhaustive(database_path=db_path)
        if cancelled():
            return {}

        log("Running incremental SfM reconstruction…")
        progress(50)
        maps = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=img_dir,
            output_path=out_dir,
        )
        if cancelled():
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
        log(
            f"Reconstruction: {len(best_rec.images)} registered images, "
            f"{len(best_rec.points3D)} 3D points"
        )
        progress(85)

        cam = next(iter(best_rec.cameras.values()))
        cam_params = cam.params
        model = cam.model_name.upper()

        if "OPENCV" in model:
            fx, fy = float(cam_params[0]), float(cam_params[1])
            cx, cy = float(cam_params[2]), float(cam_params[3])
            k1 = float(cam_params[4]) if len(cam_params) > 4 else 0.0
            k2 = float(cam_params[5]) if len(cam_params) > 5 else 0.0
            p1 = float(cam_params[6]) if len(cam_params) > 6 else 0.0
            p2 = float(cam_params[7]) if len(cam_params) > 7 else 0.0
            k3 = float(cam_params[8]) if len(cam_params) > 8 else 0.0
        elif "RADIAL" in model or "FISHEYE" in model:
            fx = fy = float(cam_params[0])
            cx, cy = float(cam_params[1]), float(cam_params[2])
            k1 = float(cam_params[3]) if len(cam_params) > 3 else 0.0
            k2 = float(cam_params[4]) if len(cam_params) > 4 else 0.0
            p1 = p2 = k3 = 0.0
        else:
            if len(cam_params) >= 4:
                fx, fy = float(cam_params[0]), float(cam_params[1])
                cx, cy = float(cam_params[2]), float(cam_params[3])
            elif len(cam_params) >= 3:
                fx = fy = float(cam_params[0])
                cx, cy = float(cam_params[1]), float(cam_params[2])
            else:
                raise RuntimeError(f"Unsupported COLMAP camera model: {model}")
            k1 = k2 = p1 = p2 = k3 = 0.0

        errors = [pt.error for pt in best_rec.points3D.values()]
        ret = float(np.mean(errors)) if errors else None

        progress(100)
        log("SfM calibration complete.")

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


# ---------------------------------------------------------------------------
# Stereo calibration: homography + Nelder-Mead
# ---------------------------------------------------------------------------

def run_stereo_calibration(params: dict, log_fn=None, progress_fn=None,
                           cancel_check=None) -> dict:
    log = log_fn or _noop
    progress = progress_fn or _noop
    cancelled = cancel_check or _never_cancelled

    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("opencv-python (cv2) is required but not installed.")
    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError:
        raise RuntimeError(
            "scipy is required for stereo calibration.\n\n"
            "Install it with:\n  pip install scipy"
        )

    T_pts_list: List[List[float]] = params["T_points"]
    W_pts_list: List[List[float]] = params["W_points"]
    initial_calib: dict = params["initial_calibration"]

    n = len(T_pts_list)
    if n < _MIN_STEREO_POINTS:
        raise RuntimeError(
            f"Need at least {_MIN_STEREO_POINTS} point correspondences, got {n}."
        )

    log(f"Computing homography from {n} point correspondences…")
    progress(10)

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
    log(f"Homography MSE: {hom_mse:.4f}  ({n_inliers}/{n} inliers)")
    progress(25)

    if cancelled():
        return {}

    # Parse initial calibration
    rgb_data = (
        initial_calib.get("Wide") or initial_calib.get("RGB") or  # noqa: W503, W504
        initial_calib.get("wide") or initial_calib.get("rgb")
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

    # When annotation images were pre-undistorted the placed points are
    # already in undistorted pixel space, so skip undistortPoints for that
    # side.  The saved output still uses the original (non-zeroed) distortion.
    rgb_pre_undist = params.get("rgb_pre_undistorted", False)
    th_pre_undist = params.get("th_pre_undistorted", False)
    rgb_dist_warp = np.zeros(5) if rgb_pre_undist else rgb_dist5
    th_dist_x0 = np.zeros(5) if th_pre_undist else th_dist5

    def _warp(pts, cm, dc):
        p = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, cm, dc.reshape(1, -1), P=new_cm).reshape(-1, 1, 2)

    def mse_func(x):
        th_fx, th_fy, th_cx, th_cy = x[0], x[1], x[2], x[3]
        th_d = np.array(x[4:9])
        th_cm = np.array([[th_fx, 0.0, th_cx], [0.0, th_fy, th_cy], [0.0, 0.0, 1.0]])
        w_w = _warp(W_pts, rgb_mtx, rgb_dist_warp)
        t_w = _warp(T_pts, th_cm, th_d)
        stacked = np.concatenate([w_w, t_w], axis=1)
        return float(np.sum(np.diff(stacked, axis=1) ** 2, axis=2).mean())

    x0 = [th_mtx[0, 0], th_mtx[1, 1], th_mtx[0, 2], th_mtx[1, 2]] + th_dist_x0.tolist()

    log("Optimising thermal intrinsics (Nelder-Mead, 10 iterations)…")
    n_iters = 10
    res = None
    for i in range(n_iters):
        if cancelled():
            return {}
        progress(30 + int(60 * i / n_iters))
        res = scipy_minimize(
            mse_func, x0, method="Nelder-Mead",
            options={"maxiter": 50000, "disp": False}, tol=1e-4,
        )
        x0 = res.x.tolist()

    opt_mse = float(res.fun)
    log(f"Optimisation complete.  MSE: {opt_mse:.6f}")
    progress(100)

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
