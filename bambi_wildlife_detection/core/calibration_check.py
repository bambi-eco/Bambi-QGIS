# -*- coding: utf-8 -*-
"""Guard against a camera calibration that does not belong to the media.

A calibration is silently accepted by OpenCV no matter what it was made for:
``getOptimalNewCameraMatrix`` and ``initUndistortRectifyMap`` happily apply a
640x512 calibration to a 1280x1024 stream.  The result still *looks* like a
frame, so nothing downstream complains -- but the focal length is wrong by the
size ratio and, worse, the output is re-centred on the calibration's principal
point instead of the image centre.  A frame extracted that way is pointing tens
of degrees away from where the pipeline thinks it is.

The check exploits the one property every sane calibration has: the principal
point sits near the centre of the image it was calibrated on.  Read backwards,
``(2*cx, 2*cy)`` recovers that image's size, which can then be compared with the
media actually being extracted.

This module is free of QGIS/Qt imports so it can be unit-tested headless.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Relative size difference above which a mismatch is merely reported.
WARN_TOLERANCE = 0.10
#: Relative size difference above which extraction is refused.
FATAL_TOLERANCE = 0.25


class CalibrationMismatchError(RuntimeError):
    """Raised when a calibration clearly belongs to a different camera."""


def _camera_matrix(calibration: Dict[str, Any]) -> Optional[List[List[float]]]:
    """Return the 3x3 intrinsic matrix from a calibration dict, or ``None``."""
    mtx = calibration.get("mtx") if isinstance(calibration, dict) else None
    if mtx is None:
        return None
    try:
        rows = [list(r) for r in mtx]
    except TypeError:
        return None
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        return None
    return [[float(v) for v in r] for r in rows]


def infer_calibration_resolution(
        calibration: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Infer the image size a calibration was made for, from its principal point.

    :param calibration: calibration dict with an ``mtx`` entry
    :return: ``(width, height)`` in pixels, or ``None`` when ``mtx`` is unusable
    """
    mtx = _camera_matrix(calibration)
    if mtx is None:
        return None
    cx, cy = mtx[0][2], mtx[1][2]
    if cx <= 0 or cy <= 0:
        return None
    return 2.0 * cx, 2.0 * cy


def check_calibration_resolution(
        calibration: Dict[str, Any],
        width: int,
        height: int,
        label: str = "calibration",
        warn_tolerance: float = WARN_TOLERANCE,
        fatal_tolerance: float = FATAL_TOLERANCE,
) -> Tuple[str, str]:
    """Compare a calibration against the media it is about to be applied to.

    :param calibration: calibration dict with an ``mtx`` entry
    :param width: width in pixels of the video/photo being extracted
    :param height: height in pixels of the video/photo being extracted
    :param label: what to call the calibration in the message
    :param warn_tolerance: relative size difference that triggers ``"warn"``
    :param fatal_tolerance: relative size difference that triggers ``"error"``
    :return: ``(severity, message)`` with severity ``"ok"``, ``"warn"``,
        ``"error"`` or ``"unknown"``; the message is empty when ``"ok"``
    """
    inferred = infer_calibration_resolution(calibration)
    if inferred is None:
        return "unknown", (
            f"Could not read a camera matrix from the {label}; "
            "skipping the resolution cross-check."
        )
    if not width or not height:
        return "unknown", (
            f"Could not determine the media resolution; skipping the {label} "
            "cross-check."
        )

    cal_w, cal_h = inferred
    dev_w = abs(cal_w - width) / float(width)
    dev_h = abs(cal_h - height) / float(height)
    deviation = max(dev_w, dev_h)

    if deviation <= warn_tolerance:
        return "ok", ""

    mtx = _camera_matrix(calibration) or [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    detail = (
        f"The {label} looks like it was made for a "
        f"{cal_w:.0f}x{cal_h:.0f} image, but the media is {width}x{height} "
        f"({deviation * 100:.0f}% off).\n"
        f"Its principal point is ({mtx[0][2]:.1f}, {mtx[1][2]:.1f}); for this "
        f"media it should be near ({width / 2:.1f}, {height / 2:.1f})."
    )

    if deviation >= fatal_tolerance:
        return "error", (
            detail + "\n\n"
            "Applying it would re-centre every extracted frame on the wrong "
            "pixel and scale the field of view by roughly "
            f"{cal_w / float(width):.2f}x, so the frames -- and every detection "
            "made on them -- would point in the wrong direction.\n\n"
            "Pick the calibration preset that matches this camera, or supply a "
            "custom calibration file made from this camera's own footage."
        )
    return "warn", detail


def media_resolution(paths: Sequence[str]) -> Optional[Tuple[int, int]]:
    """Read ``(width, height)`` from the first readable video or image path.

    :param paths: candidate video or image paths, tried in order
    :return: ``(width, height)``, or ``None`` when nothing could be read
    """
    import os

    import cv2

    for path in paths or ():
        if not path or not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v"):
            cap = cv2.VideoCapture(path)
            try:
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if w > 0 and h > 0:
                        return w, h
            finally:
                cap.release()
        else:
            img = cv2.imread(path)
            if img is not None:
                return img.shape[1], img.shape[0]
    return None


def first_photo_in(directory: str, extensions: Sequence[str]) -> List[str]:
    """Return up to one photo path per glob pattern, for :func:`media_resolution`.

    :param directory: folder holding the photos
    :param extensions: glob patterns such as ``("*.JPG", "*.jpg")``
    """
    import glob
    import os

    found: List[str] = []
    for ext in extensions or ():
        matches = sorted(glob.glob(os.path.join(directory, ext)))
        if matches:
            found.append(matches[0])
    return found


def enforce_calibration_resolution(
        calibration: Dict[str, Any],
        paths: Sequence[str],
        label: str,
        log_fn=None,
        allow_mismatch: bool = False,
) -> None:
    """Run the cross-check and either log it or refuse to extract.

    :param calibration: calibration dict about to be handed to the extractor
    :param paths: video/photo paths the calibration will be applied to
    :param label: what to call the calibration in messages
    :param log_fn: optional logging callback
    :param allow_mismatch: downgrade a fatal mismatch to a warning
    :raises CalibrationMismatchError: on a gross mismatch, unless *allow_mismatch*
    """
    resolution = media_resolution(paths)
    if resolution is None:
        return
    severity, message = check_calibration_resolution(
        calibration, resolution[0], resolution[1], label=label)

    if severity == "ok":
        return
    if severity == "error" and not allow_mismatch:
        raise CalibrationMismatchError(message)
    if log_fn and message:
        prefix = "Warning: " if severity != "unknown" else ""
        log_fn(f"{prefix}{message}")
