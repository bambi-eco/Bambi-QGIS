# -*- coding: utf-8 -*-
"""Headless camera-calibration computation of the Calibration Wizard.

Moved from ``bambi_camera_calibration.py`` (whose helpers re-export and
whose ``_CalibWorker`` delegates here):

* image/video frame helpers and intrinsics estimation,
* :func:`run_single_calibration` - single-camera SfM via pycolmap,
* :func:`run_stereo_calibration` - RGB↔thermal alignment via homography
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
#: The stereo fit finds nine numbers for the thermal camera (fx, fy, cx, cy
#: and five distortion terms), and a point pair gives two equations. Four
#: pairs - the old minimum, what a homography needs - are eight equations
#: for nine unknowns: the fit goes through the points whatever the camera
#: is. Six leave something to judge it by.
_MIN_STEREO_POINTS = 6
#: Below this the fit is reported as weak: a pixel of imprecision in
#: placing a point moves the result noticeably.
_RECOMMENDED_STEREO_POINTS = 12
#: How much of the frame's width and height the points should span.
_MIN_STEREO_SPREAD = 0.5
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
                "SfM reconstruction failed - no valid reconstruction found.\n\n"
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

def _raw_points(points, camera_matrix, distortion):
    """Where points of an image undistorted with a calibration lie in the
    camera's raw image - the calibration's distortion applied to them.

    ``cv2.undistort(image, K, d)`` gives an image of a camera with matrix
    ``K`` and no distortion; going back is projecting the points' viewing
    rays through ``K`` *with* ``d``.

    :param points: ``(N, 1, 2)`` or ``(N, 2)`` pixel coordinates
    :return: ``(N, 1, 2)`` float64
    """
    import cv2
    import numpy as np

    pixels = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    rays = np.column_stack([
        (pixels[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
        (pixels[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
        np.ones(len(pixels))])
    raw, _jacobian = cv2.projectPoints(
        rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix,
        np.asarray(distortion, dtype=np.float64).reshape(1, -1))
    return raw.reshape(-1, 1, 2)


def stereo_folds(n: int) -> List[List[int]]:
    """How *n* point pairs are split to check a stereo fit on points it was
    not fitted to: each list is left out once, and the rest - never fewer
    than the fit needs - is fitted. ``[]`` when there are too few pairs to
    leave any out."""
    rounds = 4 if n >= 16 else 3
    if n - math.ceil(n / rounds) < _MIN_STEREO_POINTS:
        return []
    return [list(range(start, n, rounds)) for start in range(rounds)]


def point_spread(points, frame_size) -> tuple:
    """``(across, down)``: the share of the frame's width and height that
    *points* span."""
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    width, height = float(frame_size[0]), float(frame_size[1])
    if not xs or width <= 0 or height <= 0:
        return 0.0, 0.0
    return (min(1.0, (max(xs) - min(xs)) / width),
            min(1.0, (max(ys) - min(ys)) / height))


def stereo_fit_warnings(n: int, spread: tuple, fit_rms: float,
                        held_out_rms: Optional[float], fitted, frame_size,
                        initial_focal=None) -> List[str]:
    """What makes a stereo calibration one not to rely on, in words.

    A fit of nine numbers to a handful of points can go through every one of
    them and still be wrong across the frame; the error of the fit does not
    show it. These are the signs that do: too few points, points in one part
    of the frame, points left out of the fit landing far off, and numbers no
    lens has.

    :param spread: :func:`point_spread` of the thermal points
    :param fitted: fx, fy, cx, cy and the five distortion terms
    :param frame_size: ``(width, height)`` of the thermal frame
    """
    warnings = []
    if n < _RECOMMENDED_STEREO_POINTS:
        text = (f"Only {n} point pairs. Nine numbers are fitted from them, "
                "so a pixel of imprecision in placing one point changes the "
                f"result noticeably. Place {_RECOMMENDED_STEREO_POINTS} or "
                "more.")
        if held_out_rms is None:
            text += (" With so few, the fit cannot even be checked on points "
                     "left out of it.")
        warnings.append(text)
    across, down = spread
    if min(across, down) < _MIN_STEREO_SPREAD:
        warnings.append(
            f"The points span {across * 100:.0f} % of the frame's width and "
            f"{down * 100:.0f} % of its height. Lens distortion shows towards "
            "the edges; where there are no points the calibration is a "
            "guess. Spread the points over the whole frame.")
    if held_out_rms is not None and held_out_rms > max(3.0, 3.0 * fit_rms):
        warnings.append(
            f"The calibration fits its own points to {fit_rms:.1f} px, but "
            f"points left out of the fit are off by {held_out_rms:.1f} px: "
            "it does not hold across the frame. Place more points, spread "
            "over the whole frame.")

    fx, fy, cx, cy = (float(value) for value in fitted[:4])
    k1 = float(fitted[4])
    width, height = float(frame_size[0]), float(frame_size[1])
    odd = []
    if abs(k1) > 1.0:
        odd.append(f"a distortion (k1 = {k1:.2f}) stronger than lenses have")
    if not (0.0 <= cx <= width and 0.0 <= cy <= height):
        odd.append("an optical centre outside the frame")
    if initial_focal:
        change = max(abs(fx / float(initial_focal[0]) - 1.0),
                     abs(fy / float(initial_focal[1]) - 1.0))
        if change > 0.3:
            odd.append(f"a focal length {change * 100:.0f} % away from the "
                       "one it started from")
    if odd:
        warnings.append(
            "The fit arrived at " + ", ".join(odd) + " - a sign that the "
            "points do not pin the camera down.")
    return warnings


def stereo_points_hint(total: int) -> str:
    """What to say next to the number of point pairs placed so far."""
    if total < _MIN_STEREO_POINTS:
        return (f"at least {_MIN_STEREO_POINTS} needed, "
                f"{_RECOMMENDED_STEREO_POINTS} or more recommended")
    if total < _RECOMMENDED_STEREO_POINTS:
        return f"{_RECOMMENDED_STEREO_POINTS} or more recommended"
    return ""


def stereo_summary(result: dict) -> str:
    """The result of a stereo calibration in a few lines: how many point
    pairs, how well it fits them, how well it fits points left out of the
    fit - and, when there is one, why it should not be relied on as it is.
    """
    lines = ["Stereo calibration complete"]
    if result.get("n_points") is not None:
        lines.append(f"  Point pairs                : {result['n_points']}")
    lines.append("  Homography MSE (px²)       : "
                 f"{result.get('homography_mse', 0.0):.4f}")
    lines.append("  Optimised MSE (px²)        : "
                 f"{result.get('optimized_mse', 0.0):.6f}")
    if result.get("fit_rms_px") is not None:
        lines.append("  Fit, its own points (RMS)  : "
                     f"{result['fit_rms_px']:.2f} px")
        held_out = result.get("held_out_rms_px")
        lines.append("  Fit, points left out (RMS) : " + (
            f"{held_out:.2f} px" if held_out is not None
            else "not checked - too few point pairs"))
    warnings = result.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("Not to be relied on as it is:")
        lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines)


def run_stereo_calibration(params: dict, log_fn=None, progress_fn=None,
                           cancel_check=None) -> dict:
    """Fit the thermal camera to the RGB one from pairs of points.

    *params*: ``T_points`` / ``W_points`` (the pairs, thermal and RGB),
    ``initial_calibration`` (``{"Wide": ..., "Thermal": ...}``),
    ``rgb_pre_undistorted`` / ``th_pre_undistorted`` (the points were placed
    on frames undistorted with the calibration given), and for judging the
    result ``th_image_size`` (``(width, height)`` of the thermal frame;
    taken from the optical centre when missing) and ``initial_focal_known``
    (both starting focal lengths are a calibration's or a data sheet's
    rather than a guess from the frame size - only then does a fit that ends
    far from them say something).

    The result carries, next to the two calibrations, ``n_points``,
    ``fit_rms_px``, ``held_out_rms_px`` (``None`` when there were too few
    pairs to leave any out) and ``warnings`` (:func:`stereo_fit_warnings`).
    """
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

    # The thermal side is what is fitted, and what is saved is used on the
    # camera's *raw* frames. With a thermal calibration loaded the points
    # were placed on images undistorted with it, and the fit then started
    # from "no distortion": it found - and saved - only what was left over
    # after the loaded calibration, as if that were the camera's whole
    # distortion. So the points are taken back to where they lie in the raw
    # image, and the fit starts from the loaded calibration in full.
    if th_pre_undist and np.any(th_dist5):
        T_pts = _raw_points(T_pts, th_mtx, th_dist5)
    th_dist_x0 = th_dist5

    def _warp(pts, cm, dc):
        p = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, cm, dc.reshape(1, -1), P=new_cm).reshape(-1, 1, 2)

    def squared_misfit(x, thermal, wide):
        """Per pair: the squared distance, in RGB pixels, between a point
        of the RGB camera and its partner of the thermal one as *x* (fx, fy,
        cx, cy, five distortion terms) maps it."""
        th_cm = np.array([[x[0], 0.0, x[2]], [0.0, x[1], x[3]], [0.0, 0.0, 1.0]])
        w_w = _warp(wide, rgb_mtx, rgb_dist_warp)
        t_w = _warp(thermal, th_cm, np.array(x[4:9]))
        stacked = np.concatenate([w_w, t_w], axis=1)
        return np.sum(np.diff(stacked, axis=1) ** 2, axis=2).ravel()

    def fit(thermal, wide, restarts, report=None):
        """Nelder-Mead from the initial calibration, restarted *restarts*
        times from where it stopped. ``None`` when cancelled."""
        x = [th_mtx[0, 0], th_mtx[1, 1], th_mtx[0, 2], th_mtx[1, 2]] \
            + th_dist_x0.tolist()
        found = None
        for i in range(restarts):
            if cancelled():
                return None
            if report:
                report(i)
            found = scipy_minimize(
                lambda v: float(squared_misfit(v, thermal, wide).mean()),
                x, method="Nelder-Mead",
                options={"maxiter": 50000, "disp": False}, tol=1e-4,
            )
            x = found.x.tolist()
        return found

    log("Optimising thermal intrinsics (Nelder-Mead, 10 iterations)…")
    n_iters = 10
    res = fit(T_pts, W_pts, n_iters,
              lambda i: progress(30 + int(50 * i / n_iters)))
    if res is None:
        return {}

    opt_mse = float(res.fun)
    fit_rms = math.sqrt(max(opt_mse, 0.0))
    log(f"Optimisation complete.  MSE: {opt_mse:.6f}")

    # Does the calibration hold for points it was not fitted to? With few
    # points, or points in one part of the frame, the nine numbers bend to
    # go through them - the fit looks perfect and is wrong everywhere else.
    held_out_rms = None
    folds = stereo_folds(n)
    if folds:
        log(f"Checking the fit on points left out of it ({len(folds)} "
            "rounds)…")
        squared = []
        for round_index, held in enumerate(folds):
            progress(80 + int(18 * round_index / len(folds)))
            used = [i for i in range(n) if i not in held]
            part = fit(T_pts[used], W_pts[used], 3)
            if part is None:
                return {}
            squared.extend(squared_misfit(
                part.x, T_pts[held], W_pts[held]).tolist())
        held_out_rms = math.sqrt(sum(squared) / len(squared))
        log(f"Fitted points: {fit_rms:.2f} px RMS; points left out: "
            f"{held_out_rms:.2f} px RMS")

    frame_size = params.get("th_image_size") or (
        2.0 * th_mtx[0, 2], 2.0 * th_mtx[1, 2])
    warnings = stereo_fit_warnings(
        n, point_spread(T_pts.reshape(-1, 2), frame_size), fit_rms,
        held_out_rms, res.x, frame_size,
        initial_focal=((th_mtx[0, 0], th_mtx[1, 1])
                       if params.get("initial_focal_known") else None))
    for warning in warnings:
        log(f"WARNING: {warning}")
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
        "n_points": n,
        "fit_rms_px": fit_rms,
        # points left out of the fit: None when there were too few to try
        "held_out_rms_px": held_out_rms,
        "warnings": warnings,
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
