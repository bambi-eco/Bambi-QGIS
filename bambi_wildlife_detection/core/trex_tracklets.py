# -*- coding: utf-8 -*-
"""TRex tracklets, as the rest of the plugin stores animals.

`TRex <https://trex.run>`_ writes one ``.npz`` file per tracked individual:
the frames it was seen on and its pose points there. An import turns each
frame of a tracklet into a box and each tracklet into a track.

**A stored box is in the pixel space of the extracted frames.** That is what
the detector writes and what every reader assumes - the geo-referencing, the
classifier's crops, the overlays, the exports. TRex is usually run on the
original video, whose pixels are another space: the extraction undistorts the
frames and, unless "Preserve original aspect ratio" is ticked, crops them
square. :class:`Undistorter` takes a box from the one to the other, the way
the extraction takes the image.

This module only reads and converts; what is recorded where is the import
step's (``BambiProcessor.run_trex_import``). ``numpy`` and ``cv2`` are
imported where they are needed, so the module itself imports without them.
"""

import glob
import os
from typing import Callable, List, Optional, Sequence, Tuple

LogFn = Optional[Callable[[str], None]]

#: The corners of a box, in the order the pose points are given in.
_BOX_KEYS = ("x1", "y1", "x2", "y2")


def tracklet_id(data, path: str, fallback: int) -> int:
    """The id of the individual in one tracklet file: its ``id`` field, else
    the number after "id" in the file name (``video_id7.npz``), else
    *fallback*."""
    import numpy as np

    if "id" in data and len(np.asarray(data["id"]).ravel()):
        return int(np.asarray(data["id"]).ravel()[0])
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.split("id")[-1] if "id" in stem else ""
    return int(tail) if tail.isdigit() else fallback


def _pose_points(data) -> List[int]:
    """The numbers of the pose points a tracklet file has both an x and a y
    series of (``poseX0``/``poseY0``, ``poseX1``/``poseY1``, ...)."""
    numbers = []
    for key in data.keys():
        number = key[len("poseX"):] if key.startswith("poseX") else ""
        if number.isdigit() and f"poseY{number}" in data:
            numbers.append(int(number))
    return sorted(numbers)


def read_tracklets(npz_dir: str, log_fn: LogFn = None
                   ) -> Tuple[List[dict], Optional[Tuple[int, int]]]:
    """Every box of every tracklet in *npz_dir*.

    :return: ``(boxes, video_size)``. Each box is ``{"frame", "track_id",
        "x1", "y1", "x2", "y2", "confidence", "class_id"}`` - the bounding
        box of the tracklet's pose points on that frame, in the pixel space
        TRex worked in - sorted by frame and tracklet. *video_size* is the
        ``(width, height)`` the files record, ``None`` when none does.
    :raises FileNotFoundError: when the folder holds no ``.npz`` file
    """
    import numpy as np

    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")

    boxes: List[dict] = []
    video_size = None
    for path in files:
        name = os.path.basename(path)
        with np.load(path, allow_pickle=True) as data:
            if "video_size" in data and video_size is None:
                size = np.asarray(data["video_size"]).ravel()
                video_size = (int(size[0]), int(size[1]))

            track_id = tracklet_id(data, path, fallback=len(boxes))
            frames = np.asarray(data["frame"]).astype(int)
            confidence = np.nan_to_num(
                np.asarray(data["detection_p"], dtype=float), nan=0.0)
            classes = np.nan_to_num(np.asarray(
                data["detection_class"], dtype=float), nan=0.0).astype(int)

            points = _pose_points(data)
            if not points:
                if log_fn:
                    log_fn(f"  WARNING: {name} has no pose points, skipped")
                continue
            pose_x = np.stack([np.asarray(data[f"poseX{i}"], dtype=float)
                               for i in points], axis=1)
            pose_y = np.stack([np.asarray(data[f"poseY{i}"], dtype=float)
                               for i in points], axis=1)

        kept = 0
        for i in range(len(frames)):
            seen = np.isfinite(pose_x[i]) & np.isfinite(pose_y[i])
            if not seen.any():
                continue
            xs, ys = pose_x[i][seen], pose_y[i][seen]
            boxes.append({
                "frame": int(frames[i]),
                "track_id": track_id,
                "x1": float(xs.min()), "y1": float(ys.min()),
                "x2": float(xs.max()), "y2": float(ys.max()),
                "confidence": float(confidence[i])
                if i < len(confidence) else 1.0,
                "class_id": int(classes[i]) if i < len(classes) else 0,
            })
            kept += 1
        if log_fn:
            log_fn(f"  {name}: track {track_id}, {kept} detections")

    boxes.sort(key=lambda box: (box["frame"], box["track_id"]))
    return boxes, video_size


def frame_size_for(video_size: Optional[Tuple[int, int]],
                   extracted: Optional[Tuple[int, int]],
                   preserve_aspect_ratio: bool
                   ) -> Optional[Tuple[int, int]]:
    """The size of the frames the poses belong to: that of the extracted
    frames where known, else what the extraction makes of the video - the
    whole frame with "Preserve original aspect ratio", a square otherwise."""
    if extracted is not None:
        return extracted
    if video_size is None:
        return None
    width, height = video_size
    if preserve_aspect_ratio:
        return width, height
    side = min(width, height)
    return side, side


class Undistorter:
    """Pixels of the original video -> pixels of the extracted frames.

    The recipe of the frame extraction (the detection framework's
    ``CalibratedVideoFrameAccessor.prepare_undistort``, and its
    ``bambi.geo.calibration.undistort_points``): the optimal new camera
    matrix at alpha 0.5 with the principal point centred, and one focal
    length - the larger - for both axes. The extraction does the last for
    frames of any shape; the import used to do it for square frames only,
    which put boxes on frames with their aspect ratio preserved up to a
    pixel off.
    """

    def __init__(self, camera_matrix, distortion,
                 video_size: Tuple[int, int], frame_size: Tuple[int, int]):
        import cv2
        import numpy as np

        self._matrix = np.asarray(camera_matrix, dtype=float)
        self._distortion = np.asarray(distortion, dtype=float)
        self.video_size = tuple(video_size)
        self.frame_size = tuple(frame_size)
        new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
            self._matrix, self._distortion, self.video_size, 0.5,
            self.frame_size, centerPrincipalPoint=True)
        focal = max(new_matrix[0, 0], new_matrix[1, 1])
        new_matrix[0, 0] = new_matrix[1, 1] = focal
        self._new_matrix = new_matrix

    def points(self, points_xy):
        """*points_xy* (n x 2) in the video, as (n x 2) in the frames."""
        import cv2
        import numpy as np

        points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        moved = cv2.undistortPoints(points, self._matrix, self._distortion,
                                    P=self._new_matrix)
        return moved.reshape(-1, 2)

    def box(self, x1: float, y1: float, x2: float, y2: float
            ) -> Tuple[float, float, float, float]:
        """The box around the four corners, each taken to the frames."""
        moved = self.points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return (float(moved[:, 0].min()), float(moved[:, 1].min()),
                float(moved[:, 0].max()), float(moved[:, 1].max()))


def to_frames(boxes: Sequence[dict],
              undistorter: Optional[Undistorter] = None) -> List[dict]:
    """*boxes* in the pixel space of the extracted frames: taken there by
    *undistorter*, or as they are when TRex worked on those frames itself.
    Copies - the boxes handed in keep the coordinates TRex gave them."""
    if undistorter is None:
        return [dict(box) for box in boxes]
    moved = []
    for box in boxes:
        corners = undistorter.box(*(box[key] for key in _BOX_KEYS))
        moved.append(dict(box, **dict(zip(_BOX_KEYS, corners))))
    return moved
