# -*- coding: utf-8 -*-
"""Point selection for the survey analytics (EXCHANGE_FORMAT_PLAN.md §8.2).

Density, coverage, distance sampling and population estimation all start from
the same question — *which animals count?* — and 5.x answered it implicitly by
reading whatever files happened to be there. That is unauditable: a result gave
no indication of what went into it.

This module makes the answer explicit and records it. Three rules apply to
every analytic:

* **``not-an-animal`` never counts.** A labelled false positive in a density
  estimate biases it upward. ``unknown`` and ``animal`` *do* count — the
  distinction is determinacy, not presence.
* **One tracker run, plus the manual run.** Builtin, boxmot and TRex are
  alternative descriptions of the same animals, so pooling two double-counts.
  The labelling tool's run is additive — its tracks are usually animals the
  detector missed — except where a label was imported from a tracker track, in
  which case the original is superseded.
* **The filter travels with the result.** Every analytic records the runs,
  species and exclusions it used, so a number can be traced back to the rows
  that produced it.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from . import store, track_store
from .exporters import common

#: Species that never contribute to an analytic.
EXCLUDED_SPECIES = (store.NOT_AN_ANIMAL_SPECIES_ID,)

DETECTIONS = "detections"
TRACKS = "tracks"


class AnalyticsError(RuntimeError):
    """There is nothing to analyse, or the request cannot be honoured."""


def species_options(target_folder: str) -> List[dict]:
    """Species available as a filter, excluding those that never count.

    Ordered as the schema editor orders them, so a UI list matches the
    Project Schema dialog.
    """
    from . import label_store

    vocabulary = label_store.vocabulary(target_folder)
    if not vocabulary:
        return []
    return [dict(entry) for entry in vocabulary["species"]
            if entry["species_id"] not in EXCLUDED_SPECIES]


def load_rows(target_folder: str, modality: str,
              species_ids: Optional[Sequence[int]] = None,
              include_manual: bool = True,
              run_id: Optional[int] = None) -> Tuple[List[dict], dict]:
    """Geo-referenced detections that count, plus the filter that produced them.

    Returns ``(rows, provenance)``. *provenance* is a plain dict meant to be
    written into the result document — it is what makes a number auditable.
    """
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(det_path):
        raise AnalyticsError(
            f"No detections stored for modality '{modality}'. Run the "
            "pipeline, or migrate a 5.x project first.")

    rows = [row for row in common.load_detections(
        target_folder, modality, include_not_an_animal=False)
        if row.get("gx1") is not None]

    excluded_species = len(common.load_detections(
        target_folder, modality, include_not_an_animal=True)) - \
        len(common.load_detections(target_folder, modality))

    if run_id is not None:
        runs = [run_id]
    else:
        runs = track_store.analysis_runs(
            target_folder, modality, include_manual=include_manual)

    selected_species = None
    if species_ids is not None:
        selected_species = sorted(set(species_ids))
        rows = [row for row in rows if row["species_id"] in selected_species]

    active = track_store.active_run(target_folder, modality)
    manual = track_store.manual_run(target_folder, modality)
    manual_included = bool(include_manual) and manual is not None
    if manual_included:
        manual_included = manual["run_id"] in runs

    provenance = {
        "modality": modality,
        "tracker_run": None if active is None else {
            "run_id": active["run_id"], "kind": active["kind"],
            "tracker": active["tracker"]},
        "manual_run_included": manual_included,
        "superseded_tracks": sorted(
            track_store.superseded_track_ids(target_folder, modality)),
        "species_filter": selected_species,
        "excluded_not_an_animal": excluded_species,
        "n_detections": len(rows),
    }
    return rows, provenance


def _centre(row: dict) -> Tuple[float, float]:
    return ((row["gx1"] + row["gx2"]) / 2.0, (row["gy1"] + row["gy2"]) / 2.0)


def load_points(target_folder: str, modality: str, source: str = DETECTIONS,
                species_ids: Optional[Sequence[int]] = None,
                include_manual: bool = True,
                run_id: Optional[int] = None
                ) -> Tuple[List[Tuple[float, float]], dict]:
    """World-coordinate points for an analytic, plus its provenance.

    ``source='detections'`` gives one point per detection; ``source='tracks'``
    gives one per track — the centroid of its detections — so an animal
    followed across two hundred frames counts once.
    """
    rows, provenance = load_rows(
        target_folder, modality, species_ids, include_manual, run_id)

    if source == DETECTIONS:
        points = [_centre(row) for row in rows]
        provenance["source"] = DETECTIONS
        provenance["n_points"] = len(points)
        return points, provenance

    if source != TRACKS:
        raise AnalyticsError(f"Unknown analytics source '{source}'.")

    grouped: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    untracked = 0
    for row in rows:
        track_id = row.get("track_id")
        if track_id is None:
            untracked += 1
            continue
        grouped[track_id].append(_centre(row))

    points = [(sum(x for x, _ in centres) / len(centres),
               sum(y for _, y in centres) / len(centres))
              for centres in grouped.values()]

    provenance["source"] = TRACKS
    provenance["n_tracks"] = len(points)
    provenance["n_points"] = len(points)
    provenance["untracked_detections"] = untracked
    return points, provenance


def load_points_by_species(target_folder: str, modality: str,
                           source: str = DETECTIONS,
                           include_manual: bool = True
                           ) -> Tuple[Dict[str, List[Tuple[float, float]]], dict]:
    """Points grouped by species name, for stratified analytics.

    Possible now that ``species_id`` has a stable meaning; in 5.x this was a
    comparison against an integer whose meaning could change between exports.
    """
    from . import label_store

    vocabulary = label_store.vocabulary(target_folder)
    names = {entry["species_id"]: entry["name"]
             for entry in vocabulary.get("species", [])}

    strata: Dict[str, List[Tuple[float, float]]] = {}
    provenance = {}
    for entry in species_options(target_folder):
        species_id = entry["species_id"]
        points, provenance = load_points(
            target_folder, modality, source, [species_id], include_manual)
        if points:
            strata[names.get(species_id, str(species_id))] = points

    provenance = dict(provenance)
    provenance["species_filter"] = None
    provenance["strata"] = {name: len(points) for name, points in strata.items()}
    return strata, provenance


def describe_filter(provenance: dict) -> str:
    """One-line summary of what an analytic counted, for the log."""
    parts = []
    tracker = provenance.get("tracker_run")
    if tracker:
        parts.append(f"tracker run {tracker['run_id']} ({tracker['kind']})")
    else:
        parts.append("no tracker run")
    if provenance.get("manual_run_included"):
        parts.append("manual tracks included")
    if provenance.get("superseded_tracks"):
        parts.append(
            f"{len(provenance['superseded_tracks'])} superseded track(s) "
            "excluded")
    if provenance.get("excluded_not_an_animal"):
        parts.append(
            f"{provenance['excluded_not_an_animal']} not-an-animal excluded")
    if provenance.get("species_filter"):
        parts.append(f"species {provenance['species_filter']}")
    return "; ".join(parts)
