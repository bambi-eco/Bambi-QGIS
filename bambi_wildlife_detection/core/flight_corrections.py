# -*- coding: utf-8 -*-
"""Correction factors as edited in the 3D Flight Viewer.

The viewer is the second way - next to the Correction Wizard - to find a
flight's correction factors. It edits the same ``correction.json`` (global
``translation`` / ``rotation`` plus frame-range overrides in ``additional``),
writes it with the same layout, and adds what a live 3D view makes possible:

* a sharpness score of the ALFS integral - frames only integrate to a sharp
  image when their poses agree with each other and with the DEM, so a
  better correction gives a sharper integral;
* an automatic search for the factors that maximise that score
  (:func:`pattern_search`), run on a few frame windows of the flight.

Translations are metres, rotations radians, as everywhere in the plugin.
"""

import copy
import json
import math
import os
import shutil
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Parameter order of the optimiser vector.
PARAMS = ("tx", "ty", "tz", "rx", "ry", "rz")
PARAM_LABELS = {"tx": "Translation x", "ty": "Translation y",
                "tz": "Translation z", "rx": "Rotation x", "ry": "Rotation y",
                "rz": "Rotation z"}

#: Stages whose results are computed from the corrected poses.
POSE_STAGES = ("georeference", "calculate_fov", "alfs", "export_geotiffs",
               "sam3_georeference")


def _xyz(d) -> Dict[str, float]:
    d = d or {}
    return {k: float(d.get(k, 0.0) or 0.0) for k in ("x", "y", "z")}


def normalized(correction: Optional[dict]) -> dict:
    """A complete correction dict (missing parts filled with zeros)."""
    c = copy.deepcopy(correction or {})
    out = dict(c)
    out["translation"] = _xyz(c.get("translation"))
    out["rotation"] = _xyz(c.get("rotation"))
    out["additional"] = [
        dict(a, start=int(a.get("start", 0)), end=int(a.get("end", 0)),
             translation=_xyz(a.get("translation")),
             rotation=_xyz(a.get("rotation")))
        for a in c.get("additional", []) or []]
    return out


def entry_for_frame(correction: dict, frame: int) -> Optional[int]:
    """Index of the ``additional`` range covering *frame* (first wins, as in
    :func:`core.corrections.correction_for_frame`), or None for the default."""
    for i, a in enumerate(correction.get("additional", []) or []):
        if a.get("start", 0) <= frame <= a.get("end", float("inf")):
            return i
    return None


def get_entry(correction: dict, entry: Optional[int]) -> Tuple[dict, dict]:
    src = correction if entry is None else correction["additional"][entry]
    return _xyz(src.get("translation")), _xyz(src.get("rotation"))


def set_entry(correction: dict, entry: Optional[int], translation: dict,
              rotation: dict) -> dict:
    """A copy of *correction* with one entry replaced."""
    out = normalized(correction)
    dst = out if entry is None else out["additional"][entry]
    dst["translation"], dst["rotation"] = _xyz(translation), _xyz(rotation)
    return out


def add_range(correction: dict, start: int, end: int,
              translation: Optional[dict] = None,
              rotation: Optional[dict] = None) -> Tuple[dict, int]:
    """Add a frame-range override (starting from the default values).

    :return: ``(new correction, index of the new range)``
    """
    out = normalized(correction)
    if translation is None or rotation is None:
        translation, rotation = out["translation"], out["rotation"]
    start, end = sorted((int(start), int(end)))
    out["additional"].append({"start": start, "end": end,
                              "translation": _xyz(translation),
                              "rotation": _xyz(rotation)})
    return out, len(out["additional"]) - 1


def remove_range(correction: dict, entry: int) -> dict:
    out = normalized(correction)
    del out["additional"][entry]
    return out


def to_vector(translation: dict, rotation: dict) -> np.ndarray:
    t, r = _xyz(translation), _xyz(rotation)
    return np.array([t["x"], t["y"], t["z"], r["x"], r["y"], r["z"]])


def from_vector(v: Sequence[float]) -> Tuple[dict, dict]:
    v = [float(x) for x in v]
    return ({"x": v[0], "y": v[1], "z": v[2]},
            {"x": v[3], "y": v[4], "z": v[5]})


# ---------------------------------------------------------------------------
# Reading and writing correction.json
# ---------------------------------------------------------------------------

def correction_path(config: dict) -> str:
    """The file the wizard writes: the configured one, else the target's."""
    explicit = (config.get("correction_path") or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    return os.path.join(config.get("target_folder", ""), "correction.json")


def write_correction(path: str, correction: dict, backup: bool = True) -> str:
    """Write *correction* (wizard layout, indented) and back up the old file.

    Keys this module does not know are kept from the existing file.

    :return: the backup's path, or ``""`` when there was nothing to back up
    """
    existing = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except (OSError, ValueError):
            existing = {}
    data = dict(existing)
    clean = normalized(correction)
    data["translation"] = clean["translation"]
    data["rotation"] = clean["rotation"]
    if clean["additional"] or "additional" in data:
        data["additional"] = clean["additional"]
    saved = ""
    if backup and os.path.isfile(path):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        saved = f"{os.path.splitext(path)[0]}.backup-{stamp}.json"
        shutil.copy2(path, saved)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)
    return saved


def mark_pose_results_stale(target_folder: str,
                            modalities: Sequence[str] = ("t", "w")) -> List[str]:
    """Flag completed results computed from the poses (and their dependents)
    as stale. Nothing is deleted.

    :return: ``"stage (camera)"`` labels of what was flagged
    """
    from . import stages
    flagged = []
    for modality in modalities:
        try:
            recorded = stages.states(target_folder, modality)
        except Exception:  # nosec B112 - no project store yet
            continue
        for stage in POSE_STAGES:
            row = recorded.get(stage)
            if not row or row.get("state") != stages.COMPLETE:
                continue
            stages.set_state(target_folder, stage, modality, stages.STALE,
                             row_count=row.get("row_count"),
                             fingerprint=row.get("fingerprint") or "")
            flagged.append(f"{stage} ({modality})")
            flagged += [f"{d} ({modality})" for d in
                        stages.mark_dependents_stale(target_folder, stage,
                                                     modality)]
    return flagged


# ---------------------------------------------------------------------------
# Sharpness of an integral
# ---------------------------------------------------------------------------

def sharpness(rgba, min_pixels: int = 200) -> float:
    """Focus measure of an ALFS integral: the variance of the Laplacian of
    its grey values over fully covered pixels (higher = sharper).

    Pixels next to uncovered ones are ignored, so the edge of the footprint
    does not count as detail. ``nan`` when too little is covered.
    """
    img = np.asarray(rgba, dtype=np.float64)
    if img.ndim != 3 or img.shape[0] < 3 or img.shape[1] < 3:
        return float("nan")
    grey = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    covered = img[..., 3] > 0 if img.shape[2] > 3 else np.ones(grey.shape, bool)
    lap = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4.0 * grey[1:-1, 1:-1])
    ok = (covered[1:-1, 1:-1] & covered[:-2, 1:-1] & covered[2:, 1:-1] & covered[1:-1, :-2] & covered[1:-1, 2:])
    if ok.sum() < min_pixels:
        return float("nan")
    return float(lap[ok].var())


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def default_steps() -> np.ndarray:
    """Initial search steps: 1 m and about 0.5 degrees."""
    return np.array([1.0, 1.0, 1.0] + [math.radians(0.5)] * 3)


def pattern_search(objective: Callable[[np.ndarray], float], x0,
                   steps, active: Optional[Sequence[bool]] = None,
                   shrink: float = 0.5, min_scale: float = 1.0 / 16,
                   max_evals: int = 200,
                   callback: Optional[Callable] = None,
                   should_stop: Optional[Callable[[], bool]] = None):
    """Minimise *objective* by compass search over the active parameters.

    Each parameter is tried at ``± step``; an improvement is kept, otherwise
    the steps shrink until they reach ``min_scale`` of the start. No
    derivatives, robust to the noisy objective a rendered image gives.

    :return: ``(best x, best value, evaluations)``
    """
    x = np.array(x0, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64).copy()
    active = np.ones(len(x), bool) if active is None else np.asarray(active, bool)
    best = objective(x)
    evals = 1
    scale = 1.0
    if callback:
        callback(x.copy(), best, evals)
    while scale >= min_scale and evals < max_evals:
        improved = False
        for i in np.flatnonzero(active):
            for sign in (1.0, -1.0):
                if evals >= max_evals or (should_stop and should_stop()):
                    return x, best, evals
                trial = x.copy()
                trial[i] += sign * steps[i] * scale
                value = objective(trial)
                evals += 1
                if value < best:
                    x, best, improved = trial, value, True
                    if callback:
                        callback(x.copy(), best, evals)
                    break
        if not improved:
            scale *= shrink
    return x, best, evals


def refine_windows(count: int, current: int, n_windows: int,
                   half_width: int) -> List[int]:
    """Centre frames of the windows auto-refine evaluates: the current frame
    plus ``n_windows - 1`` windows spread evenly along the flight."""
    if count <= 0:
        return []
    current = min(max(int(current), 0), count - 1)
    centres = {current}
    if n_windows > 1:
        lo, hi = half_width, max(half_width, count - 1 - half_width)
        for c in np.linspace(lo, hi, n_windows - 1):
            centres.add(min(max(int(round(c)), 0), count - 1))
    return sorted(centres)
