# -*- coding: utf-8 -*-
"""The built-in tracker: which box continues which track.

Frame by frame, the boxes of a frame are matched to the live tracks; a box
that matches none opens a new track, and a track that goes unmatched for
more than ``max_age`` frames is closed. How boxes and tracks are matched is
the **mode**:

==================  ======================================================
``GREEDY``          by overlap (IoU), the best pair first
``HUNGARIAN``       by overlap, the assignment with the most overlap in all
``CENTER``          by the distance between the box centres, nearest first,
                    up to ``max_center_distance``
``HUNGARIAN_CENTER``  ``HUNGARIAN``, then ``CENTER`` for the boxes and
                    tracks that overlap could not pair
==================  ======================================================

Until 9.1 the tracker ran ``HUNGARIAN`` whatever was selected and never read
the centre distance: two of the four modes and one setting did nothing. The
centre modes are what pairs an animal whose consecutive boxes do not overlap
- a small animal, a low frame rate, a jump in the geo-referencing.

The matching is the one of ``bambi.tracking.iou.track_boxes`` in the BAMBI
Detection Framework, kept here because the plugin runs without that package
for everything but frame extraction. Ties are broken the same way (float32
overlap, stable sorts), so both give the same tracks.

``max_age`` counts the frames *that have boxes* in which a track went
unmatched - a stretch of frames without any box (the thermal camera's
shutter) does not age a track.
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

GREEDY = "GREEDY"
HUNGARIAN = "HUNGARIAN"
CENTER = "CENTER"
HUNGARIAN_CENTER = "HUNGARIAN_CENTER"

#: The modes in the order the configuration offers them.
MODES = (GREEDY, HUNGARIAN, CENTER, HUNGARIAN_CENTER)
DEFAULT_MODE = HUNGARIAN

#: Modes that match by box overlap first, and those that use the centre
#: distance - what the two thresholds apply to.
OVERLAP_MODES = (GREEDY, HUNGARIAN, HUNGARIAN_CENTER)
CENTER_MODES = (CENTER, HUNGARIAN_CENTER)

_LARGE = 1e6


class Cancelled(Exception):
    """The run was aborted between two frames."""


def normalise_mode(mode) -> str:
    """*mode* as one of :data:`MODES`; anything else is refused rather than
    quietly run as the default."""
    text = str(mode or DEFAULT_MODE).strip().upper().replace(" ", "_")
    if text not in MODES:
        raise ValueError(
            f"Unknown tracker mode {mode!r}; expected one of "
            f"{', '.join(MODES)}.")
    return text


def iou_matrix(a, b):
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` xyxy boxes, float32."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(
        0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(
        0.0, b[:, 3] - b[:, 1])
    denom = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((inter > 0.0) & (denom > 0.0), inter / denom, 0.0)
    return out.astype(np.float32)


def center_distance_sq(a, b):
    """Pairwise squared distance between the centres of xyxy boxes."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ca = 0.5 * (a[:, :2] + a[:, 2:4])
    cb = 0.5 * (b[:, :2] + b[:, 2:4])
    d = ca[:, None, :] - cb[None, :, :]
    return d[..., 0] ** 2 + d[..., 1] ** 2


def _greedy(score, valid, descending: bool
            ) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    """One-to-one over the *valid* cells, best score first; ties in
    (box, track) order."""
    import numpy as np

    di, ti = np.nonzero(valid)
    if len(di) == 0:
        return [], set(), set()
    s = score[di, ti]
    order = np.argsort(-s if descending else s, kind="stable")
    det_used: Set[int] = set()
    trk_used: Set[int] = set()
    matches = []
    for k in order:
        d, t = int(di[k]), int(ti[k])
        if d in det_used or t in trk_used:
            continue
        det_used.add(d)
        trk_used.add(t)
        matches.append((d, t))
    return matches, det_used, trk_used


def _hungarian(iou, iou_threshold: float
               ) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    from scipy.optimize import linear_sum_assignment

    cost = 1.0 - iou
    cost[iou < iou_threshold] = _LARGE
    rows, cols = linear_sum_assignment(cost)
    matches = []
    det_used: Set[int] = set()
    trk_used: Set[int] = set()
    for d, t in zip(rows, cols):
        if iou[d, t] >= iou_threshold and cost[d, t] < _LARGE:
            matches.append((int(d), int(t)))
            det_used.add(int(d))
            trk_used.add(int(t))
    return matches, det_used, trk_used


def match(boxes, track_boxes, allowed, mode: str = DEFAULT_MODE,
          iou_threshold: float = 0.3, max_center_distance: float = 2.0
          ) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    """Pair the boxes of one frame with the live tracks.

    :param boxes: ``(N, 4)`` xyxy boxes of the frame
    :param track_boxes: ``(M, 4)`` last box of each live track
    :param allowed: ``(N, M)`` bool - pairs that may be joined at all (the
        same class, when the tracker is class-aware)
    :return: ``(pairs, boxes used, tracks used)``, pairs as
        ``(box index, track index)``
    """
    import numpy as np

    mode = normalise_mode(mode)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    track_boxes = np.asarray(track_boxes, dtype=np.float64).reshape(-1, 4)
    nd, nt = len(boxes), len(track_boxes)
    if not nd or not nt:
        return [], set(), set()
    allowed = np.asarray(allowed, dtype=bool).reshape(nd, nt)
    max_d2 = float(max_center_distance) ** 2

    if mode == CENTER:
        d2 = center_distance_sq(boxes, track_boxes)
        return _greedy(d2, allowed & (d2 <= max_d2), descending=False)

    iou = iou_matrix(boxes, track_boxes)
    iou[~allowed] = 0.0
    if mode == GREEDY:
        return _greedy(iou, allowed & (iou >= iou_threshold), descending=True)

    matches, det_used, trk_used = _hungarian(iou, iou_threshold)
    if mode == HUNGARIAN_CENTER:
        rest_d = [i for i in range(nd) if i not in det_used]
        rest_t = [i for i in range(nt) if i not in trk_used]
        if rest_d and rest_t:
            d2 = center_distance_sq(boxes[rest_d], track_boxes[rest_t])
            ok = allowed[np.ix_(rest_d, rest_t)] & (d2 <= max_d2)
            for a, b in _greedy(d2, ok, descending=False)[0]:
                matches.append((rest_d[a], rest_t[b]))
                det_used.add(rest_d[a])
                trk_used.add(rest_t[b])
    return matches, det_used, trk_used


def track(frames: Dict[int, Sequence[Tuple[float, float, float, float]]],
          classes: Optional[Dict[int, Sequence[int]]] = None,
          mode: str = DEFAULT_MODE, iou_threshold: float = 0.3,
          max_center_distance: float = 2.0, max_age: int = -1,
          class_aware: bool = True,
          cancel_check: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[float], None]] = None
          ) -> Dict[int, List[int]]:
    """Give every box a track id.

    :param frames: ``{frame: [(x1, y1, x2, y2), ...]}`` - the horizontal
        extent of each box, in pixels or in metres
    :param classes: ``{frame: [class id, ...]}`` aligned with *frames*;
        ``None`` for a single class
    :param max_age: frames with boxes a track may go unmatched before it is
        closed (``-1``: never)
    :param cancel_check: asked before every frame; raises :class:`Cancelled`
    :param progress: called with the share of the frames done, 0..1
    :return: ``{frame: [track id, ...]}`` aligned with *frames*; ids start
        at 1, in the order the tracks were opened
    """
    import numpy as np

    mode = normalise_mode(mode)
    if classes is None:
        class_aware = False

    ids: Dict[int, List[int]] = {}
    trk_box: List = []
    trk_cls: List[int] = []
    trk_age: List[int] = []
    trk_id: List[int] = []
    next_id = 1

    ordered = sorted(frames)
    for done, frame in enumerate(ordered):
        if cancel_check is not None and cancel_check():
            raise Cancelled("Tracking cancelled")
        d_box = np.asarray(frames[frame], dtype=np.float64).reshape(-1, 4)
        nd, nt = len(d_box), len(trk_box)
        d_cls = (np.zeros(nd, dtype=np.int64) if classes is None
                 else np.asarray(classes[frame]).astype(np.int64).ravel())
        if len(d_cls) != nd:
            raise ValueError(
                f"frame {frame}: {nd} boxes but {len(d_cls)} classes")

        matches: List[Tuple[int, int]] = []
        det_used: Set[int] = set()
        trk_used: Set[int] = set()
        if nd and nt:
            allowed = np.ones((nd, nt), dtype=bool)
            if class_aware:
                allowed = d_cls[:, None] == np.asarray(trk_cls)[None, :]
            matches, det_used, trk_used = match(
                d_box, np.vstack(trk_box), allowed, mode, iou_threshold,
                max_center_distance)

        frame_ids = [0] * nd
        for d, t in matches:
            trk_box[t] = d_box[d]
            trk_age[t] = 0
            frame_ids[d] = trk_id[t]
        for d in range(nd):
            if d in det_used:
                continue
            trk_box.append(d_box[d])
            trk_cls.append(int(d_cls[d]))
            trk_age.append(0)
            trk_id.append(next_id)
            frame_ids[d] = next_id
            next_id += 1
        ids[frame] = frame_ids

        keep = []
        for t in range(nt):
            if t not in trk_used:
                trk_age[t] += 1
            if max_age < 0 or trk_age[t] <= max_age:
                keep.append(t)
        keep += list(range(nt, len(trk_box)))      # opened in this frame
        trk_box = [trk_box[t] for t in keep]
        trk_cls = [trk_cls[t] for t in keep]
        trk_age = [trk_age[t] for t in keep]
        trk_id = [trk_id[t] for t in keep]

        if progress is not None:
            progress((done + 1) / len(ordered))
    return ids
