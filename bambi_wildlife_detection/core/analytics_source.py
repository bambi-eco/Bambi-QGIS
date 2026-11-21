# -*- coding: utf-8 -*-
"""Point selection for the survey analytics.

Density, coverage, distance sampling and population estimation all start from
the same question - *which animals count?* - and 5.x answered it implicitly by
reading whatever files happened to be there. That is unauditable: a result gave
no indication of what went into it.

This module makes the answer explicit and records it. Three rules apply to
every analytic:

* **``not-an-animal`` never counts.** A labelled false positive in a density
  estimate biases it upward. ``unknown`` and ``animal`` *do* count - the
  distinction is determinacy, not presence.
* **One tracker run, plus the manual run.** Builtin, boxmot and TRex are
  alternative descriptions of the same animals, so pooling two double-counts.
  The labelling tool's run is additive - its tracks are usually animals the
  detector missed - except where a label was imported from a tracker track, in
  which case the original is superseded.
* **The filter travels with the result.** Every analytic records the runs,
  species and exclusions it used, so a number can be traced back to the rows
  that produced it.
* **An animal is one thing in every tool.** What a survey counts is either
  animals (tracks) or boxes (detections), and :func:`load_animals` says for
  both what species, sex and age each one is and where it is. A tracked
  animal has one species (:func:`core.track_store.animal_species`) and one
  position, the centroid of its boxes - the tools used to ask the track, each
  box or the last box, and so disagreed about the same animal.
"""

import json
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


def has_no_animals(target_folder: str, modality: str) -> bool:
    """True when a flight was searched and nothing was found: detection ran,
    and no box that counts is left - none at all, or only ``not-an-animal``.

    Such a flight has no tracks and can have none, so the steps that need
    tracks cannot run on it. It still was surveyed: its ground counts, with
    zero animals on it, and leaving it out of a density inflates the density.
    A flight on which detection never ran is not one of these.
    """
    path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(path):
        return False
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        return not any(
            int(row["species_id"]) not in EXCLUDED_SPECIES
            for row in conn.execute(
                "SELECT DISTINCT species_id FROM detections"))
    finally:
        conn.close()


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


def confirmed_tracks(target_folder: str, modality: str) -> set:
    """The tracks of *modality* that the other camera saw too: those the
    cross-modal matching paired with a track of the other camera
    (:func:`core.match_store.partner_tracks`, the active run). Empty where
    the flight was never matched."""
    from . import match_store

    return {int(track) for track in match_store.partner_tracks(
        target_folder, modality)}


def confirmed_ids(target_folder: str, modality: str,
                  source: str = TRACKS) -> set:
    """What counts when only animals both cameras saw do: the ids of the
    confirmed tracks (``tracks``), or of their boxes (``detections``)."""
    tracks = confirmed_tracks(target_folder, modality)
    if source == TRACKS or not tracks:
        return tracks
    return {int(row["detection_id"]) for row in common.load_detections(
        target_folder, modality, include_not_an_animal=True)
        if int(row.get("track_id") or -1) in tracks}


def load_rows(target_folder: str, modality: str,
              species_ids: Optional[Sequence[int]] = None,
              include_manual: bool = True,
              run_id: Optional[int] = None,
              confirmed: bool = False) -> Tuple[List[dict], dict]:
    """Geo-referenced detections that count, plus the filter that produced them.

    Returns ``(rows, provenance)``. *provenance* is a plain dict meant to be
    written into the result document - it is what makes a number auditable.

    :param confirmed: only the boxes of tracks the other camera saw too
        (:func:`confirmed_tracks`)
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

    unconfirmed = 0
    if confirmed:
        both = confirmed_tracks(target_folder, modality)
        kept = [row for row in rows
                if int(row.get("track_id") or -1) in both]
        unconfirmed = len(rows) - len(kept)
        rows = kept

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
    if confirmed:
        provenance["confirmed_by_other_camera"] = True
        provenance["unconfirmed"] = unconfirmed
    return rows, provenance


def _centre(row: dict) -> Tuple[float, float]:
    return ((row["gx1"] + row["gx2"]) / 2.0, (row["gy1"] + row["gy2"]) / 2.0)


def counts(species_id) -> bool:
    """Whether something of this species is counted at all."""
    return species_id not in EXCLUDED_SPECIES


def load_animals(target_folder: str, modality: str, source: str = TRACKS,
                 species_ids: Optional[Sequence[int]] = None,
                 include_manual: bool = True,
                 confirmed: bool = False) -> Tuple[List[dict], dict]:
    """What a survey counts on one flight, and as what.

    One record per tracked animal (``source='tracks'``) or per box
    (``'detections'``)::

        {"track_id", "detection_id", "species_id", "species", "sex", "age",
         "approved", "x", "y", "first_frame", "last_frame", "n_boxes"}

    * **tracks** - the species is the animal's
      (:func:`core.track_store.animal_species`): the track's own where it
      was identified, else what most of its boxes were called. Sex, age and
      the approval are the track's. The position is the centroid of *all*
      its geo-referenced boxes, so an animal is in one place and counts for
      one species.
    * **detections** - the species is the box's own. Sex, age and approval
      are the box's, and where it has none those of the animal it belongs
      to: a box of a known female is a box of a female.

    ``x`` / ``y`` are ``None`` for what was never geo-referenced - it still
    counts, it just cannot be placed. ``not-an-animal`` is left out, and
    *species_ids* (``None`` = every species) selects.

    :param confirmed: only animals the other camera saw too - tracks the
        cross-modal matching paired (:func:`confirmed_tracks`), and under
        ``detections`` their boxes. A lone warm stone has no partner; an
        animal only one camera could see has none either.
    :return: ``(records, facts)`` - *facts* is ``{"excluded_not_an_animal",
        "untracked_detections", "unplaced", "unconfirmed"}``: how many
        records the false positive rule removed, how many counting boxes
        belong to no track (they are no animal under ``tracks``), how many
        records have no position, how many the other camera did not
        confirm.
    """
    if source not in (TRACKS, DETECTIONS):
        raise AnalyticsError(f"Unknown analytics source '{source}'.")
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(det_path):
        raise AnalyticsError(
            f"No detections stored for modality '{modality}'. Run the "
            "pipeline, or migrate a 5.x project first.")

    vocabulary = common.load_vocabulary(target_folder)
    names = vocabulary["species"]
    enum_labels = vocabulary["enum_labels"]
    wanted = None if species_ids is None else {int(s) for s in species_ids}

    rows = common.load_detections(target_folder, modality,
                                  include_not_an_animal=True)
    species = track_store.animal_species(target_folder, modality,
                                         include_manual=include_manual)
    known = track_store.track_labels(target_folder, modality)
    resolved: Dict[int, dict] = {}

    def of_track(track_id) -> dict:
        """Sex, age, approval... of a track, with enum values as labels."""
        if track_id not in resolved:
            stored = (known.get(track_id) or {}).get("attributes")
            resolved[track_id] = common.resolve_attributes(
                json.dumps(stored or {}), enum_labels)
        return resolved[track_id]

    facts = {"excluded_not_an_animal": 0, "untracked_detections": 0,
             "unplaced": 0, "unconfirmed": 0}
    both = confirmed_tracks(target_folder, modality) if confirmed else None
    records: List[dict] = []

    if source == DETECTIONS:
        for row in rows:
            species_id = int(row["species_id"])
            if not counts(species_id):
                facts["excluded_not_an_animal"] += 1
                continue
            track_id = row.get("track_id")
            if track_id not in species:
                track_id = None
                facts["untracked_detections"] += 1
            if wanted is not None and species_id not in wanted:
                continue
            if both is not None and int(track_id or -1) not in both:
                facts["unconfirmed"] += 1
                continue
            own = common.resolve_attributes(row.get("attributes"),
                                            enum_labels)
            animal = {} if track_id is None else of_track(track_id)
            placed = row.get("gx1") is not None
            x, y = _centre(row) if placed else (None, None)
            if not placed:
                facts["unplaced"] += 1
            records.append({
                "track_id": track_id,
                "detection_id": int(row["detection_id"]),
                "species_id": species_id,
                "species": names.get(species_id, ""),
                "sex": own.get("sex") or animal.get("sex"),
                "age": own.get("age") or animal.get("age"),
                "approved": bool(
                    own.get("approved") or animal.get("approved")),
                "x": x, "y": y,
                "first_frame": row["frame"], "last_frame": row["frame"],
                "n_boxes": 1,
            })
        return records, facts

    members: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        track_id = row.get("track_id")
        if track_id is None or track_id not in species:
            if counts(int(row["species_id"])):
                facts["untracked_detections"] += 1
            continue
        members[int(track_id)].append(row)

    for track_id in sorted(members):
        species_id = species[track_id]
        if not counts(species_id):
            facts["excluded_not_an_animal"] += 1
            continue
        if wanted is not None and species_id not in wanted:
            continue
        if both is not None and track_id not in both:
            facts["unconfirmed"] += 1
            continue
        boxes = members[track_id]
        centres = [_centre(row) for row in boxes
                   if row.get("gx1") is not None]
        if centres:
            x = sum(c[0] for c in centres) / len(centres)
            y = sum(c[1] for c in centres) / len(centres)
        else:
            x = y = None
            facts["unplaced"] += 1
        animal = of_track(track_id)
        frames = [row["frame"] for row in boxes]
        records.append({
            "track_id": track_id,
            "detection_id": None,
            "species_id": species_id,
            "species": names.get(species_id, ""),
            "sex": animal.get("sex"), "age": animal.get("age"),
            "approved": bool(animal.get("approved")),
            "x": x, "y": y,
            "first_frame": min(frames), "last_frame": max(frames),
            "n_boxes": len(boxes),
        })
    return records, facts


def current_species(target_folder: str, modality: str, source: str = TRACKS,
                    include_manual: bool = True) -> Dict[int, int]:
    """``{id: species_id}`` as the store has it now - by track id for
    ``tracks`` (:func:`core.track_store.animal_species`), by detection id
    for ``detections``. Empty where the flight has no store."""
    if source == TRACKS:
        return track_store.animal_species(target_folder, modality,
                                          include_manual=include_manual)
    path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(path):
        return {}
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        return {int(row["detection_id"]): int(row["species_id"])
                for row in conn.execute(
                    "SELECT detection_id, species_id FROM detections")}
    finally:
        conn.close()


def select_entries(entries: Sequence[dict], id_key: str,
                   species_now: Dict[int, int],
                   species_ids: Optional[Sequence[int]] = None,
                   only: Optional[set] = None
                   ) -> Tuple[List[dict], int]:
    """The entries of a perpendicular-distance file that count.

    Such a file is written once and read by distance sampling and the
    population estimate later; the ``class_id`` it carries is what the last
    box was called when it was written. The species an entry counts as is
    the one its animal (or box) has **now** - looked up in *species_now* by
    the entry's *id_key* - so a classification made after the distances
    were calculated applies, and both tools agree with the census. An entry
    whose id the store does not know keeps its ``class_id``.

    :param only: the ids that count at all (:func:`confirmed_ids`, when
        the survey counts what both cameras saw); ``None`` = every id
    :return: ``(kept, false positives dropped)``; each kept entry is a copy
        with ``class_id`` set to the species it counts as.
    """
    wanted = None if species_ids is None else {int(s) for s in species_ids}
    kept, dropped = [], 0
    for entry in entries:
        if only is not None:
            try:
                if int(entry.get(id_key)) not in only:
                    continue
            except (TypeError, ValueError):
                continue
        species_id = entry.get("class_id")
        try:
            species_id = species_now.get(int(entry.get(id_key)), species_id)
        except (TypeError, ValueError):
            pass
        if species_id is not None and not counts(species_id):
            dropped += 1
            continue
        if wanted is not None and species_id not in wanted:
            continue
        kept.append(dict(entry, class_id=species_id))
    return kept, dropped


def _apply_area(points, provenance, area):
    """Keep the points inside *area* and record what the area removed."""
    if area is None:
        return points
    points, dropped = area.filter_points(points)
    provenance["survey_area"] = area.describe()
    provenance["outside_survey_area"] = dropped
    provenance["n_points"] = len(points)
    if "n_tracks" in provenance:
        provenance["n_tracks"] = len(points)
    return points


def load_points(target_folder: str, modality: str, source: str = DETECTIONS,
                species_ids: Optional[Sequence[int]] = None,
                include_manual: bool = True,
                run_id: Optional[int] = None,
                area=None, confirmed: bool = False,
                ) -> Tuple[List[Tuple[float, float]], dict]:
    """World-coordinate points for an analytic, plus its provenance.

    ``source='detections'`` gives one point per detection; ``source='tracks'``
    gives one per track - the centroid of its detections - so an animal
    followed across two hundred frames counts once, and for the one species
    it is (:func:`load_animals`).

    :param area: a :class:`core.survey_area.SurveyArea`; points outside it
        (outside the plot, or inside an exclusion zone) are dropped. For
        tracks the centroid decides, so a track is never half counted.
    """
    rows, provenance = load_rows(
        target_folder, modality, species_ids, include_manual, run_id,
        confirmed)

    if source == DETECTIONS:
        points = [_centre(row) for row in rows]
        provenance["source"] = DETECTIONS
        provenance["n_points"] = len(points)
        return _apply_area(points, provenance, area), provenance

    if source != TRACKS:
        raise AnalyticsError(f"Unknown analytics source '{source}'.")

    # The species filter chose boxes above; an animal is chosen by its own
    # species and placed by all its boxes (load_animals).
    animals, facts = load_animals(target_folder, modality, TRACKS,
                                  species_ids, include_manual, confirmed)
    points = [(animal["x"], animal["y"]) for animal in animals
              if animal["x"] is not None]
    if confirmed:
        provenance["unconfirmed"] = facts["unconfirmed"]

    provenance["source"] = TRACKS
    provenance["n_tracks"] = len(points)
    provenance["n_points"] = len(points)
    provenance["untracked_detections"] = facts["untracked_detections"]
    provenance["unplaced_tracks"] = facts["unplaced"]
    provenance["excluded_not_an_animal_tracks"] = (
        facts["excluded_not_an_animal"])
    return _apply_area(points, provenance, area), provenance


def track_centroids(rows: Sequence[dict]
                    ) -> Tuple[Dict[int, Tuple[float, float]], int]:
    """``({track_id: centroid}, untracked)`` of geo-referenced rows.

    The centroid of a track's box centres is where every analytic places the
    animal, so a track is counted inside or outside an area as a whole.
    """
    grouped: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    untracked = 0
    for row in rows:
        track_id = row.get("track_id")
        if track_id is None:
            untracked += 1
            continue
        if row.get("gx1") is None:
            continue
        grouped[track_id].append(_centre(row))
    centroids = {
        track_id: (sum(x for x, _ in centres) / len(centres),
                   sum(y for _, y in centres) / len(centres))
        for track_id, centres in grouped.items()}
    return centroids, untracked


def load_points_by_species(target_folder: str, modality: str,
                           source: str = DETECTIONS,
                           include_manual: bool = True,
                           area=None,
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
            target_folder, modality, source, [species_id], include_manual,
            area=area)
        if points:
            strata[names.get(species_id, str(species_id))] = points

    provenance = dict(provenance)
    provenance["species_filter"] = None
    provenance["strata"] = {name: len(points) for name, points in strata.items()}
    return strata, provenance


def resolve_species_names(target_folder: str,
                          names: Sequence[str]) -> Tuple[List[int], List[str]]:
    """Translate species *names* into this project's ids.

    Returns ``(ids, missing)``. Ids are assigned per project, so
    project A's species 12 is unrelated to project B's - anything pooling
    several projects must match on the name and translate, never carry an id
    across. ``missing`` names are reported rather than silently dropped.
    """
    from . import label_store

    vocabulary = label_store.vocabulary(target_folder)
    by_name = {entry["name"]: entry["species_id"]
               for entry in vocabulary.get("species", [])}
    ids, missing = [], []
    for name in names:
        if name in by_name:
            ids.append(by_name[name])
        else:
            missing.append(name)
    return sorted(set(ids)), missing


def compare_vocabularies(target_folders: Sequence[str]) -> dict:
    """Report how the species vocabularies of several projects differ.

    Pooling projects whose taxonomies disagree is legitimate - one survey may
    simply not have seen wild boar - but it has to be visible, because a
    species missing from one project looks identical to a species that was
    present and never detected.
    """
    from . import label_store

    per_project = {}
    for folder in target_folders:
        vocabulary = label_store.vocabulary(folder)
        per_project[folder] = {
            entry["name"] for entry in vocabulary.get("species", [])
            if entry["species_id"] not in EXCLUDED_SPECIES}

    known = [names for names in per_project.values() if names]
    shared = set.intersection(*known) if known else set()
    everything = set.union(*known) if known else set()
    return {
        "shared": sorted(shared),
        "only_in": {folder: sorted(names - shared)
                    for folder, names in per_project.items()
                    if names - shared},
        "all": sorted(everything),
    }


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
    if provenance.get("confirmed_by_other_camera"):
        parts.append(
            "only what both cameras saw "
            f"({provenance.get('unconfirmed', 0)} unconfirmed left out)")
    if provenance.get("survey_area"):
        parts.append(
            f"survey area applied ({provenance.get('outside_survey_area', 0)} "
            "outside)")
    return "; ".join(parts)
