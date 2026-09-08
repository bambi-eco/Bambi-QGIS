# -*- coding: utf-8 -*-
"""Cross-modal track matching: which thermal track and which RGB track are the
same animal.

Implements §3.2 of *When One Modality Is Not Enough*. Three steps:

1. **Frame correspondence** by capture time (:mod:`core.frame_matching`) - the
   two cameras run at different rates, so "the same frame" only means anything
   on the shared clock.
2. **A 2D affine** ``T: RGB -> thermal`` fitted from corresponding detection
   centres, which absorbs the different fields of view, resolutions and the
   small parallax between the two lenses.
3. **Track-level assignment**: every RGB track is compared to every thermal
   track by the *median* inter-centre distance over the frames they share, and
   the pairing is one-to-one.

Two departures from the paper, both deliberate:

* **The affine is bootstrapped, not given.** The paper fits ``T`` from
  corresponding centres, but on real data the correspondence is exactly what is
  being solved for. We seed from frames where each modality holds exactly one
  detection - where correspondence is not in doubt - and then alternate
  assignment and refitting.
* **Assignment is Hungarian, not greedy.** The paper takes the smallest median
  distance first; a global optimum costs nothing extra here (the matrices are
  tiny) and cannot be led astray by an unlucky first pick.

What confirmation *means* is the point of all this: a pair seen in both
modalities is a real animal, and the paper's own numbers show why it matters -
of 34 unmatched tracks only one was real, while applying a length-and-
confidence fallback uniformly would have admitted six phantoms.

Pure NumPy/SciPy, no QGIS and no torch, so the whole thing is unit-testable.
"""

import math
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

#: Cost written into gated-out cells. ``linear_sum_assignment`` needs a finite
#: matrix, so an impossible pair gets a number no real distance can reach and
#: is dropped again after the assignment.
_BLOCKED = 1e9


class MatchConfig(NamedTuple):
    """Gates a candidate pair has to clear to be called the same animal."""

    #: Frames the two tracks must have in common. Below this the median is not
    #: a median of anything.
    min_shared: int = 8
    #: Maximum median inter-centre distance, in thermal pixels. 28 px is the
    #: paper's empirical gate at their 1024x1024 inference size; it is exposed
    #: because it does not transfer to a different resolution unchanged.
    gate_px: float = 28.0
    #: Mean detection confidence both tracks must reach. Stops a match being
    #: built on a track that fires on shadow.
    min_confidence: float = 0.20
    #: How far apart two frames may be on the shared clock and still be called
    #: the same moment, in seconds.
    max_time_offset: float = 0.10
    #: Where the distances are measured: ``"pixel"`` is the paper's recipe
    #: (RGB centres mapped onto the thermal image through the affine, gated
    #: by ``gate_px``); ``"world"`` compares the geo-referenced centres both
    #: modalities already carry, in metres, gated by ``gate_m`` - no affine
    #: to estimate, and the frames' own poses absorb the aircraft's motion
    #: between the two capture instants.
    space: str = "pixel"
    #: Maximum median inter-centre distance in world space, in metres. Two
    #: animals in a herd stand more than a body length apart; a box placed on
    #: the same animal by the two cameras lands well inside one.
    gate_m: float = 1.5

    @property
    def gate(self) -> float:
        """The distance gate in the units of :attr:`space`."""
        return self.gate_m if self.space == "world" else self.gate_px

    @property
    def unit(self) -> str:
        return "m" if self.space == "world" else "px"


class Detection(NamedTuple):
    """One box, as this module needs it."""

    detection_id: int
    track_id: int
    frame: int
    cx: float
    cy: float
    confidence: float


def to_detections(rows: Iterable[dict]) -> List[Detection]:
    """Adapt store rows (``x1``..``y2``) into :class:`Detection` centres."""
    result = []
    for row in rows:
        result.append(Detection(
            detection_id=int(row["detection_id"]),
            track_id=int(row["track_id"]),
            frame=int(row["frame"]),
            cx=(float(row["x1"]) + float(row["x2"])) / 2.0,
            cy=(float(row["y1"]) + float(row["y2"])) / 2.0,
            confidence=float(row.get("confidence") or 0.0),
        ))
    return result


# ---------------------------------------------------------------------------
# The affine
# ---------------------------------------------------------------------------

class Affine(NamedTuple):
    """``q = A p + t``, mapping an RGB image point onto the thermal image."""

    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        return (self.a * x + self.b * y + self.tx,
                self.c * x + self.d * y + self.ty)

    def as_json(self) -> Tuple[List[List[float]], List[float]]:
        """The form :mod:`core.match_store` persists."""
        return ([[self.a, self.b], [self.c, self.d]], [self.tx, self.ty])

    def inverse_scale(self) -> Tuple[float, float]:
        """How many RGB pixels one thermal pixel spans, per axis.

        Used to size a matched RGB crop from its thermal partner's box. Taken
        from the column norms of the linear part rather than from ``a`` and
        ``d`` alone, so a transform with any rotation in it still gives the
        true scale instead of its cosine.
        """
        sx = math.hypot(self.a, self.c)
        sy = math.hypot(self.b, self.d)
        return (1.0 / sx if sx else 1.0, 1.0 / sy if sy else 1.0)

    @classmethod
    def identity(cls) -> "Affine":
        return cls(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    @classmethod
    def from_json(cls, payload) -> "Affine":
        """Rebuild from what :mod:`core.match_store` persisted."""
        if not payload:
            return cls.identity()
        (a, b), (c, d) = payload[0]
        tx, ty = payload[1]
        return cls(float(a), float(b), float(c), float(d),
                   float(tx), float(ty))

    @classmethod
    def scaling(cls, sx: float, sy: float) -> "Affine":
        return cls(sx, 0.0, 0.0, sy, 0.0, 0.0)


def fit_affine(pairs: Sequence[Tuple[Tuple[float, float],
                                     Tuple[float, float]]]
               ) -> Optional[Affine]:
    """Least-squares fit of ``T`` from ``((rgb_x, rgb_y), (th_x, th_y))`` pairs.

    Returns ``None`` when there is too little to fit, or when the points are
    degenerate (all on one line, or all at one place) - an affine through
    collinear points is unconstrained perpendicular to that line, and would
    send every off-line detection somewhere arbitrary.
    """
    import numpy as np

    if len(pairs) < 3:
        return None

    source = np.array([[p[0], p[1], 1.0] for p, _ in pairs], dtype=float)
    target = np.array([[q[0], q[1]] for _, q in pairs], dtype=float)

    # Rank 3 means the source points span the plane; anything less is
    # collinear or coincident.
    if np.linalg.matrix_rank(source, tol=1e-6) < 3:
        return None

    solution, *_ = np.linalg.lstsq(source, target, rcond=None)
    (a, c), (b, d), (tx, ty) = solution
    values = (a, b, c, d, tx, ty)
    if not all(math.isfinite(v) for v in values):
        return None
    return Affine(float(a), float(b), float(c), float(d),
                  float(tx), float(ty))


def affine_rmse(affine: Affine,
                pairs: Sequence[Tuple[Tuple[float, float],
                                      Tuple[float, float]]]) -> float:
    """Root-mean-square residual of *affine* over *pairs*, in pixels."""
    if not pairs:
        return float("inf")
    total = 0.0
    for (px, py), (qx, qy) in pairs:
        mx, my = affine.apply(px, py)
        total += (mx - qx) ** 2 + (my - qy) ** 2
    return math.sqrt(total / len(pairs))


# ---------------------------------------------------------------------------
# Frame correspondence
# ---------------------------------------------------------------------------

def by_frame(detections: Iterable[Detection]) -> Dict[int, List[Detection]]:
    """Group detections by the frame they were found in."""
    grouped: Dict[int, List[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.frame, []).append(detection)
    return grouped


def frame_pairs(frames_t: Iterable[int],
                frame_map: Dict[int, int]) -> List[Tuple[int, int]]:
    """``(thermal_frame, rgb_frame)`` for every thermal frame with a partner."""
    pairs = []
    for frame in sorted(frames_t):
        partner = frame_map.get(frame)
        if partner is not None:
            pairs.append((frame, partner))
    return pairs


# ---------------------------------------------------------------------------
# Bootstrapping the affine
# ---------------------------------------------------------------------------

def seed_pairs(detections_t: Iterable[Detection],
               detections_w: Iterable[Detection],
               frame_map: Dict[int, int]
               ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Correspondences from frames where neither modality is ambiguous.

    A frame holding exactly one detection in each modality pairs them without
    any assumption about the transform - which is what makes it possible to
    estimate the transform at all.
    """
    grouped_t = by_frame(detections_t)
    grouped_w = by_frame(detections_w)

    pairs = []
    for frame_t, frame_w in frame_pairs(grouped_t, frame_map):
        here_t = grouped_t.get(frame_t, [])
        here_w = grouped_w.get(frame_w, [])
        if len(here_t) == 1 and len(here_w) == 1:
            pairs.append(((here_w[0].cx, here_w[0].cy),
                          (here_t[0].cx, here_t[0].cy)))
    return pairs


def _assign_within_frame(affine: Affine, here_t: List[Detection],
                         here_w: List[Detection]) -> List[Tuple[int, int]]:
    """Pair this frame's detections by nearest mapped centre, one-to-one."""
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    if not here_t or not here_w:
        return []

    cost = np.zeros((len(here_w), len(here_t)), dtype=float)
    for i, det_w in enumerate(here_w):
        mx, my = affine.apply(det_w.cx, det_w.cy)
        for j, det_t in enumerate(here_t):
            cost[i, j] = math.hypot(mx - det_t.cx, my - det_t.cy)

    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows, cols))


def estimate_affine(detections_t: Iterable[Detection],
                    detections_w: Iterable[Detection],
                    frame_map: Dict[int, int],
                    frame_size_t: Optional[Tuple[float, float]] = None,
                    frame_size_w: Optional[Tuple[float, float]] = None,
                    max_iterations: int = 5,
                    log_fn=None) -> Tuple[Affine, float, int]:
    """Estimate ``T: RGB -> thermal``. Returns ``(affine, rmse, n_pairs)``.

    Starts from the best initial guess available and then alternates one-to-one
    assignment with refitting until the residual stops improving:

    * **Unambiguous frames** - one detection in each modality - pair without
      assuming anything about the transform, so they are the preferred start.
    * **Failing that, the pure scale implied by the two frame sizes.** A herd
      can put several animals in *every* frame, leaving no unambiguous frame at
      all, and that is exactly the case the paper's flights present. The scale
      guess is good enough to start from because both cameras look straight
      down from one airframe, so the true transform really is close to a scale.

    The returned RMSE is what says whether the result can be trusted; a large
    one means the iteration converged somewhere wrong, and the caller should
    say so rather than report zero matches as though there were no animals.
    """
    detections_t = list(detections_t)
    detections_w = list(detections_w)

    pairs = seed_pairs(detections_t, detections_w, frame_map)
    affine = fit_affine(pairs)
    seeded = affine is not None

    if seeded:
        best_rmse = affine_rmse(affine, pairs)
    else:
        # Nothing unambiguous to fit from: start from the frame-size scale and
        # let the refinement below do the work. best_rmse is infinite so the
        # first refinement is always accepted.
        affine = _size_fallback(frame_size_t, frame_size_w)
        best_rmse = float("inf")

    grouped_t = by_frame(detections_t)
    grouped_w = by_frame(detections_w)
    correspondences = frame_pairs(grouped_t, frame_map)

    for _ in range(max_iterations):
        refined_pairs = []
        for frame_t, frame_w in correspondences:
            here_t = grouped_t.get(frame_t, [])
            here_w = grouped_w.get(frame_w, [])
            for i, j in _assign_within_frame(affine, here_t, here_w):
                refined_pairs.append((
                    (here_w[i].cx, here_w[i].cy),
                    (here_t[j].cx, here_t[j].cy)))

        candidate = fit_affine(refined_pairs)
        if candidate is None:
            break
        rmse = affine_rmse(candidate, refined_pairs)
        if not (rmse < best_rmse - 1e-6):
            break
        affine, best_rmse, pairs = candidate, rmse, refined_pairs

    if log_fn:
        if not math.isfinite(best_rmse):
            log_fn(
                "Cross-modal registration: nothing could be fitted, so the "
                "scale implied by the two frame sizes is used unchanged. "
                "Check the matches before trusting them.")
        else:
            start = "unambiguous frames" if seeded else "the frame-size scale"
            log_fn(f"Cross-modal registration: started from {start}, fitted "
                   f"{len(pairs)} correspondence(s), RMSE {best_rmse:.2f} px")
    return affine, best_rmse, len(pairs)


def _size_fallback(frame_size_t: Optional[Tuple[float, float]],
                   frame_size_w: Optional[Tuple[float, float]]) -> Affine:
    """Scale RGB onto thermal by frame size alone - the last resort."""
    if not frame_size_t or not frame_size_w:
        return Affine.identity()
    width_t, height_t = frame_size_t
    width_w, height_w = frame_size_w
    if not width_w or not height_w:
        return Affine.identity()
    return Affine.scaling(width_t / width_w, height_t / height_w)


# ---------------------------------------------------------------------------
# Track-level cost
# ---------------------------------------------------------------------------

class Candidate(NamedTuple):
    """One (thermal track, RGB track) pair and the evidence for it."""

    track_id_t: int
    track_id_w: int
    shared: int
    median_dist: float
    conf_t: float
    conf_w: float
    pairs: List[dict]


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mean_confidence(detections: Iterable[Detection]) -> Dict[int, float]:
    totals: Dict[int, List[float]] = {}
    for detection in detections:
        totals.setdefault(detection.track_id, []).append(detection.confidence)
    return {track: sum(values) / len(values)
            for track, values in totals.items() if values}


def candidates(detections_t: Iterable[Detection],
               detections_w: Iterable[Detection],
               frame_map: Dict[int, int],
               affine: Affine) -> List[Candidate]:
    """Every track pair sharing at least one frame, with its median distance.

    The *median* rather than the mean is what makes this robust: a track that
    is briefly confused with a neighbour, or a frame where one box is badly
    placed, moves a mean enough to break a real pair.
    """
    detections_t = list(detections_t)
    detections_w = list(detections_w)
    grouped_t = by_frame(detections_t)
    grouped_w = by_frame(detections_w)

    distances: Dict[Tuple[int, int], List[float]] = {}
    frame_pair_rows: Dict[Tuple[int, int], List[dict]] = {}

    for frame_t, frame_w in frame_pairs(grouped_t, frame_map):
        for det_t in grouped_t.get(frame_t, []):
            for det_w in grouped_w.get(frame_w, []):
                mx, my = affine.apply(det_w.cx, det_w.cy)
                distance = math.hypot(mx - det_t.cx, my - det_t.cy)
                key = (det_t.track_id, det_w.track_id)
                distances.setdefault(key, []).append(distance)
                frame_pair_rows.setdefault(key, []).append({
                    "frame_t": det_t.frame,
                    "frame_w": det_w.frame,
                    "detection_id_t": det_t.detection_id,
                    "detection_id_w": det_w.detection_id,
                    "dist": distance,
                })

    confidence_t = _mean_confidence(detections_t)
    confidence_w = _mean_confidence(detections_w)

    result = []
    for (track_t, track_w), values in distances.items():
        result.append(Candidate(
            track_id_t=track_t,
            track_id_w=track_w,
            shared=len(values),
            median_dist=_median(values),
            conf_t=confidence_t.get(track_t, 0.0),
            conf_w=confidence_w.get(track_w, 0.0),
            pairs=frame_pair_rows[(track_t, track_w)],
        ))
    return sorted(result, key=lambda c: (c.median_dist, c.track_id_t))


def passes_gates(candidate: Candidate, config: MatchConfig) -> bool:
    """Whether a candidate is admissible at all, before any assignment."""
    if candidate.shared < config.min_shared:
        return False
    if candidate.median_dist >= config.gate:
        return False
    if candidate.conf_t < config.min_confidence:
        return False
    if candidate.conf_w < config.min_confidence:
        return False
    return True


def assign(candidates_in: Sequence[Candidate],
           config: MatchConfig = MatchConfig()) -> List[Candidate]:
    """One-to-one assignment over the gated candidates.

    Hungarian rather than the paper's greedy pass: the matrices are at most a
    few dozen square, so the global optimum is free, and greedy can spend a
    track on a near miss that a later, better pair then needs.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    admissible = [c for c in candidates_in if passes_gates(c, config)]
    if not admissible:
        return []

    tracks_t = sorted({c.track_id_t for c in admissible})
    tracks_w = sorted({c.track_id_w for c in admissible})
    index_t = {track: i for i, track in enumerate(tracks_t)}
    index_w = {track: i for i, track in enumerate(tracks_w)}

    cost = np.full((len(tracks_t), len(tracks_w)), _BLOCKED, dtype=float)
    lookup: Dict[Tuple[int, int], Candidate] = {}
    for candidate in admissible:
        i, j = index_t[candidate.track_id_t], index_w[candidate.track_id_w]
        cost[i, j] = candidate.median_dist
        lookup[(i, j)] = candidate

    rows, cols = linear_sum_assignment(cost)
    # A rectangular problem forces every row (or column) to take something,
    # including a blocked cell, so the gate is re-applied after the fact.
    matched = [lookup[(i, j)] for i, j in zip(rows, cols)
               if cost[i, j] < _BLOCKED]
    return sorted(matched, key=lambda c: c.median_dist)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def in_world(detections: Iterable[Detection],
             world: Dict[int, Tuple[float, float]]) -> List[Detection]:
    """The same detections with their centres replaced by world coordinates.

    *world* maps ``detection_id`` to a ground-plane ``(x, y)`` - the centre of
    the geo-referenced box. Detections without one are left out: they cannot
    be compared in metres, and geo-referencing has already recorded why.
    """
    result = []
    for detection in detections:
        centre = world.get(detection.detection_id)
        if centre is None:
            continue
        result.append(detection._replace(cx=float(centre[0]),
                                         cy=float(centre[1])))
    return result


def _affine_from_matches(boxes_t: Sequence[Detection],
                         boxes_w: Sequence[Detection],
                         matches: Sequence[Candidate],
                         frame_size_t, frame_size_w,
                         log_fn=None) -> Tuple[Affine, float]:
    """The pixel affine, fitted from pairs that world-space matching confirmed.

    World-space matching needs no affine, but the *matched* classifiers still
    size an RGB crop from its thermal partner's box through one - and
    confirmed pairs are the best correspondences there are to fit it from.
    """
    centre_t = {d.detection_id: (d.cx, d.cy) for d in boxes_t}
    centre_w = {d.detection_id: (d.cx, d.cy) for d in boxes_w}
    pairs = []
    for match in matches:
        for pair in match.pairs:
            p = centre_w.get(pair["detection_id_w"])
            q = centre_t.get(pair["detection_id_t"])
            if p is not None and q is not None:
                pairs.append((p, q))
    affine = fit_affine(pairs)
    if affine is None:
        affine = _size_fallback(frame_size_t, frame_size_w)
        rmse = float("inf")
        if log_fn:
            log_fn("Cross-modal registration: too few confirmed pairs to fit "
                   "the pixel transform, so the scale implied by the two "
                   "frame sizes is used for sizing matched crops.")
    else:
        rmse = affine_rmse(affine, pairs)
        if log_fn:
            log_fn(f"Cross-modal registration: pixel transform fitted from "
                   f"{len(pairs)} confirmed correspondence(s), RMSE "
                   f"{rmse:.2f} px")
    return affine, rmse


def match_tracks(detections_t: Iterable[dict], detections_w: Iterable[dict],
                 frame_map: Dict[int, int],
                 config: MatchConfig = MatchConfig(),
                 frame_size_t: Optional[Tuple[float, float]] = None,
                 frame_size_w: Optional[Tuple[float, float]] = None,
                 world_t: Optional[Dict[int, Tuple[float, float]]] = None,
                 world_w: Optional[Dict[int, Tuple[float, float]]] = None,
                 log_fn=None) -> dict:
    """Match thermal tracks to RGB tracks. The whole of §3.2 in one call.

    *detections_t* / *detections_w* are store rows carrying ``detection_id``,
    ``track_id``, ``frame``, the box, and ``confidence``; *frame_map* maps a
    thermal frame index onto the RGB frame taken at the same moment.

    With ``config.space == "world"``, *world_t* / *world_w* map each
    ``detection_id`` to the ground-plane centre of its geo-referenced box and
    the distances are metres on the ground. That sidesteps the affine, which
    on a real flight has to be bootstrapped from the very correspondences it
    is meant to find - and a wrong one pairs neighbours instead of partners
    while still clearing the pixel gate. The affine is then fitted afterwards
    from the confirmed pairs, for the crop sizing that still needs it.

    Returns ``{"matches", "affine", "affine_rmse", "candidates", "rejected",
    "space"}``, where ``matches`` is ready for
    :func:`core.match_store.record_matches`.
    """
    boxes_t = to_detections(detections_t)
    boxes_w = to_detections(detections_w)

    if config.space == "world":
        if world_t is None or world_w is None:
            raise ValueError("world-space matching needs the geo-referenced "
                             "centres of both modalities")
        compared_t = in_world(boxes_t, world_t)
        compared_w = in_world(boxes_w, world_w)
        if log_fn:
            log_fn(f"Cross-modal matching in world coordinates: "
                   f"{len(compared_t)} of {len(boxes_t)} thermal and "
                   f"{len(compared_w)} of {len(boxes_w)} RGB detection(s) "
                   f"carry ground positions")
        everything = candidates(compared_t, compared_w, frame_map,
                                Affine.identity())
        accepted = assign(everything, config)
        affine, rmse = _affine_from_matches(
            boxes_t, boxes_w, accepted, frame_size_t, frame_size_w,
            log_fn=log_fn)
    else:
        affine, rmse, _ = estimate_affine(
            boxes_t, boxes_w, frame_map, frame_size_t, frame_size_w,
            log_fn=log_fn)
        everything = candidates(boxes_t, boxes_w, frame_map, affine)
        accepted = assign(everything, config)

    matches = [{
        "track_id_t": candidate.track_id_t,
        "track_id_w": candidate.track_id_w,
        "shared": candidate.shared,
        "median_dist": candidate.median_dist,
        "conf_t": candidate.conf_t,
        "conf_w": candidate.conf_w,
        "pairs": candidate.pairs,
    } for candidate in accepted]

    if log_fn and everything:
        if accepted:
            best = min(c.median_dist for c in accepted)
            worst = max(c.median_dist for c in accepted)
            log_fn(f"Cross-modal matching: {len(accepted)} pair(s) confirmed "
                   f"from {len(everything)} candidate(s); median inter-centre "
                   f"distance {best:.1f}–{worst:.1f} {config.unit} inside a "
                   f"{config.gate:g} {config.unit} gate")
        else:
            # "No matches" and "the gate is wrong" look identical from the
            # outside, and the defaults are calibrated to a resolution the
            # user may not share - so say which gate did the rejecting.
            reasons = rejection_reasons(everything, config)
            closest = min(c.median_dist for c in everything)
            log_fn(f"Cross-modal matching: no pair confirmed out of "
                   f"{len(everything)} candidate(s) - "
                   f"{reasons['shared_frames']} shared too few frames, "
                   f"{reasons['distance']} were too far apart, "
                   f"{reasons['confidence']} were too low-confidence. The "
                   f"closest candidate sat at {closest:.1f} {config.unit} "
                   f"against a {config.gate:g} {config.unit} gate.")

    return {
        "matches": matches,
        "affine": affine,
        "affine_rmse": rmse,
        "candidates": len(everything),
        "rejected": len(everything) - len(accepted),
        "space": config.space,
    }


def rejection_reasons(candidates_in: Sequence[Candidate],
                      config: MatchConfig) -> Dict[str, int]:
    """Why candidates were turned away, for the run log.

    "No matches" is otherwise indistinguishable from "the gate is wrong", and
    the gate defaults are calibrated to a resolution the user may not share.
    """
    counts = {"shared_frames": 0, "distance": 0, "confidence": 0}
    for candidate in candidates_in:
        if candidate.shared < config.min_shared:
            counts["shared_frames"] += 1
        elif candidate.median_dist >= config.gate:
            counts["distance"] += 1
        elif min(candidate.conf_t, candidate.conf_w) < config.min_confidence:
            counts["confidence"] += 1
    return counts
