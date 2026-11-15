# -*- coding: utf-8 -*-
"""Write classification results where the rest of the plugin reads them.

The classifiers record their answers in ``classification.gpkg`` with full
provenance - the model, the per-frame probability, the vote margin. That file
is the authority, but nothing else in the plugin reads it. This module projects
those answers onto the fields the 6.0 store already has, and from there the
exporters, the QGIS layers, the survey analytics and the labelling tool all see
them with no changes to any of them:

==================  ===============================================
occlusion (frame)   ``detections.attributes["occlusion"]``
species (track)     ``tracks.species_id`` **and** its detections'
sex (track)         ``tracks.attributes["sex"]``
life stage (track)  ``tracks.attributes["age"]``
==================  ===============================================

**Species is written at both levels on purpose.** Exports and analytics filter
on the *detection's* species (``core/exporters/common.py`` selects
``d.species_id``), so writing only the track would leave every consumer blind
to the result. The track carries the individual-level answer; the detections
carry it too so the existing queries find it.

Four rules keep the write safe:

* **A detection that someone else identified keeps its species.** Rows still
  on the fallback ``animal`` are filled in, and so are rows this projection
  filled in before; a species from a detector class mapping someone
  configured is never silently overwritten - unless they ask for that.
* **Manual tracks are never touched.** A hand annotation outranks a model.
  So does a reviewer's word on a tracker's track: a field marked as set by
  hand (:mod:`core.track_review`) is left as it is.
* **The projection is idempotent and re-runnable.** ``track_predictions``
  remains the source of truth, so if something else rewrites the detection
  species later - re-applying a detector class mapping does exactly that -
  the fix is to run this again, not to re-run the classifiers.
* **The fields follow the results, also back.** What was written is on
  record (``classification_store.applied_values``), so a call that a later
  run makes differently is replaced where it was written - on the track and
  on its detections - and a call a later run no longer makes (the vote fell
  below the quorum) is taken back. Without the record the earlier species
  stayed on the detections, which is where the exports and the analytics
  read it, and an earlier sex or age stayed on the track.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

from . import store, track_review

#: Task name → the field it lands in, and at which scope.
TRACK_FIELDS = {"sex": "sex", "life_stage": "age"}

#: The species id a detection has when nothing has identified it.
FALLBACK = store.FALLBACK_SPECIES_ID


def normalise_label(label: str) -> str:
    """The form two labels are compared in when matching by name.

    A head's classes are identifiers - ``red_deer`` - and the project's are
    names - ``red deer``. Those are the same word to anyone reading them, so
    they match here: case, underscores, hyphens and runs of whitespace are all
    folded. Only used to *find* a match; the labels themselves are never
    rewritten.
    """
    text = str(label or "").replace("_", " ").replace("-", " ")
    return " ".join(text.split()).lower()


#: Separators a head uses to join two ideas into one class name.
_COMPOUND_SEPARATORS = ("/", "_", "-", "+", "&", ",")


def label_candidates(label: str) -> List[str]:
    """Names to try for a head label, best first.

    The label itself, normalised; then, for a compound label, its leading
    part. The published red deer sex head answers ``female_juvenile`` -
    females and juveniles look alike from above - and a project whose sex
    vocabulary is ``female`` / ``male`` should count that as female without
    anyone opening the mapping dialog. Only the leading part is tried:
    ``female_juvenile`` says female first, and ``juvenile`` is a life stage,
    not a sex.
    """
    whole = normalise_label(label)
    candidates = [whole] if whole else []
    text = str(label or "")
    for separator in _COMPOUND_SEPARATORS:
        text = text.replace(separator, " ")
    parts = text.split()
    if len(parts) > 1:
        leading = normalise_label(parts[0])
        if leading and leading not in candidates:
            candidates.append(leading)
    return candidates


def match_by_name(label: str, known: Dict[str, object]):
    """The value of *label* in a ``{normalised name: value}`` table, or
    ``None`` - trying the whole label, then its leading part."""
    for candidate in label_candidates(label):
        if candidate in known:
            return known[candidate]
    return None


def label_values(spec: dict, vocabulary: dict, task: str) -> Dict[str, int]:
    """``{model label: project value id}`` for one task.

    Prefers the mapping the user configured - keyed on class index, which is
    the only thing a model guarantees - and falls back to matching names, so a
    model that already speaks the project's language needs no configuration.
    """
    labels = list(spec.get("class_labels") or [])
    configured = spec.get("labels") or {}

    resolved: Dict[str, int] = {}
    for index, label in enumerate(labels):
        value = configured.get(str(index))
        if value is not None:
            resolved[label] = int(value)

    known = _vocabulary_values(vocabulary, task)
    for label in labels:
        if label not in resolved:
            value = match_by_name(label, known)
            if value is not None:
                resolved[label] = value
    # A head whose classes were never listed can still be resolved by name.
    for label, value in known.items():
        resolved.setdefault(label, value)
    return resolved


def label_names(spec: dict, vocabulary: dict, task: str) -> Dict[str, str]:
    """``{model label: project value name}`` for one task.

    The heads speak class labels and the store speaks ids, but the per-species
    model choice is keyed on the project's species *names* - so a species vote
    of ``red_deer`` has to become ``red deer`` before it can pick the red deer
    sex model. Same resolution as :func:`label_values`, one step further.
    """
    names = _vocabulary_names(vocabulary, task)
    return {label: names[value]
            for label, value in label_values(spec, vocabulary, task).items()
            if value in names}


def resolve(values: Dict[str, object], label: str):
    """Look *label* up in a :func:`label_values` / :func:`label_names` table.

    An exact hit wins - that is the configured, index-keyed mapping. Failing
    that the normalised name is tried, which is how a stored ``red_deer`` finds
    the project's ``red deer`` when nobody ever read the classes off the model.
    """
    value = values.get(label)
    if value is None:
        value = match_by_name(label, values)
    return value


def _vocabulary_rows(vocabulary: dict, task: str):
    """``(name, id)`` pairs of the project values a task can map onto."""
    if not vocabulary:
        return []
    if task == "species":
        return [(row["name"], row["species_id"])
                for row in vocabulary.get("species", [])]
    enum_name = {"life_stage": "age"}.get(task, task)
    return [(row["label"], row["value_id"])
            for row in vocabulary.get("enums", {}).get(enum_name, [])]


def _vocabulary_values(vocabulary: dict, task: str) -> Dict[str, int]:
    """Project values a task can map onto, keyed by normalised label."""
    return {normalise_label(name): value
            for name, value in _vocabulary_rows(vocabulary, task)}


def _vocabulary_names(vocabulary: dict, task: str) -> Dict[int, str]:
    """Project value names, keyed by id."""
    return {value: name for name, value in _vocabulary_rows(vocabulary, task)}


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def _load_json(raw) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def apply_occlusion(target_folder: str, modality: str,
                    predictions: Iterable[dict], values: Dict[str, int],
                    log_fn=None) -> int:
    """Write per-frame occlusion onto the detections that carry it."""
    path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(path):
        return 0

    wanted = {}
    for row in predictions:
        value = resolve(values, row["label"])
        if value is not None:
            wanted[int(row["detection_id"])] = int(value)
    if not wanted:
        return 0

    written = 0
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        with store.transaction(conn):
            existing = {
                int(r["detection_id"]): _load_json(r["attributes"])
                for r in conn.execute(
                    "SELECT detection_id, attributes FROM detections")}
            for detection_id, value in wanted.items():
                attributes = existing.get(detection_id)
                if attributes is None:
                    continue
                if attributes.get("occlusion") == value:
                    continue
                attributes["occlusion"] = value
                conn.execute(
                    "UPDATE detections SET attributes = ? "
                    "WHERE detection_id = ?",
                    (json.dumps(attributes), detection_id))
                written += 1
    finally:
        conn.close()

    if log_fn and written:
        log_fn(f"Applied occlusion to {written} detection(s)")
    return written


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def _manual_run_id(target_folder: str, modality: str) -> Optional[int]:
    from . import track_store

    manual = track_store.manual_run(target_folder, modality)
    return None if manual is None else int(manual["run_id"])


def apply_track_species(target_folder: str, modality: str,
                        predictions: Iterable[dict], values: Dict[str, int],
                        overwrite_detections: bool = False,
                        log_fn=None) -> Dict[str, int]:
    """Write the species vote onto tracks and their detections.

    Both levels, because the exporters and the analytics filter on the
    detection's species while the track is where the individual's answer
    belongs. A track that had a vote applied and has none now gets its
    species taken back, on both levels.
    """
    from . import classification_store

    tracks_path = store.stage_path(target_folder, store.TRACKS, modality)
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not (os.path.isfile(tracks_path) and os.path.isfile(det_path)):
        return {"tracks": 0, "detections": 0}

    wanted = {}
    for row in predictions:
        value = resolve(values, row["label"])
        if value is not None:
            wanted[int(row["track_id"])] = int(value)
    previous = classification_store.applied_values(
        target_folder, modality, "species")
    if not wanted and not previous:
        return {"tracks": 0, "detections": 0}

    manual_run = _manual_run_id(target_folder, modality)
    tracks_written = 0
    applied: Dict[int, int] = {}
    members: Dict[int, List[int]] = {}

    conn = store.open_store(tracks_path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            for row in conn.execute(
                    "SELECT track_id, run_id, species_id, attributes "
                    "FROM tracks"):
                track_id = int(row["track_id"])
                if manual_run is not None and row["run_id"] == manual_run:
                    continue      # a hand annotation outranks a model
                if "species" in track_review.by_hand(row["attributes"]):
                    continue      # ... and so does a reviewer's word
                if track_id in wanted:
                    target = wanted[track_id]
                    applied[track_id] = target
                elif track_id not in previous:
                    continue
                elif row["species_id"] == previous[track_id]:
                    # The vote is gone, and the species is still the one it
                    # put there: it goes too.
                    target = FALLBACK
                else:
                    # ... changed by someone since: theirs, on the track. Its
                    # detections may still carry what was written.
                    members[track_id] = []
                    continue
                if row["species_id"] != target:
                    conn.execute(
                        "UPDATE tracks SET species_id = ? WHERE track_id = ?",
                        (target, track_id))
                    tracks_written += 1
                members[track_id] = []

            for row in conn.execute(
                    "SELECT track_id, detection_id FROM track_members"):
                track_id = int(row["track_id"])
                if track_id in members:
                    members[track_id].append(int(row["detection_id"]))
    finally:
        conn.close()

    detections_written = _write_detection_species(
        target_folder, det_path, modality, members, applied, previous,
        overwrite_detections)
    classification_store.record_applied_values(
        target_folder, modality, "species", applied)

    if log_fn:
        log_fn(f"Applied species to {tracks_written} track(s) and "
               f"{detections_written} detection(s)")
    return {"tracks": tracks_written, "detections": detections_written}


def _write_detection_species(target_folder: str, path: str, modality: str,
                             members: Dict[int, List[int]],
                             wanted: Dict[int, int],
                             previous: Dict[int, int],
                             overwrite: bool) -> int:
    """Bring the species of the member detections in line with their track.

    *wanted* is the species each track has now, *previous* the one this
    projection gave it last time; a track in *members* but not in *wanted*
    has had its vote withdrawn.

    A detection is written when it is still unidentified, or when its
    species is the one this projection wrote before - and not also what its
    detector's class mapping says, which is then the detector's answer and
    stays unless *overwrite* is set. A hand-labelled box is never written.
    """
    if not any(members.values()):
        return 0

    project = None
    own: Dict[tuple, int] = {}

    def detector_species(row) -> int:
        """What the detection's producer made of it, by the class mapping."""
        nonlocal project
        key = (row["source_id"], row["source_class"])
        if key not in own:
            if project is None:
                project = store.open_store(
                    store.project_path(target_folder), store.PROJECT)
            own[key] = store.resolve_species(project, *key)
        return own[key]

    written = 0
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        with store.transaction(conn):
            current = {int(r["detection_id"]): r for r in conn.execute(
                "SELECT detection_id, species_id, source_id, source_class, "
                "label_track_id FROM detections")}
            for track_id, detection_ids in members.items():
                target = wanted.get(track_id)
                before = previous.get(track_id)
                for detection_id in detection_ids:
                    row = current.get(detection_id)
                    if row is None or row["label_track_id"] is not None:
                        continue
                    have = int(row["species_id"])
                    ours = False
                    if before is not None and have == before:
                        ours = detector_species(row) != have
                    if target is None:
                        if not ours:
                            continue        # never ours, or changed since
                        species_id = detector_species(row)
                    elif have == target:
                        continue
                    elif have != FALLBACK and not ours and not overwrite:
                        # A detector that identified the animal itself keeps
                        # its answer, so a configured class mapping is not
                        # undone.
                        continue
                    else:
                        species_id = target
                    conn.execute(
                        "UPDATE detections SET species_id = ? "
                        "WHERE detection_id = ?", (species_id, detection_id))
                    written += 1
    finally:
        conn.close()
        if project is not None:
            project.close()
    return written


def apply_track_attribute(target_folder: str, modality: str, field: str,
                          predictions: Iterable[dict],
                          values: Dict[str, int], log_fn=None) -> int:
    """Write a per-track enum attribute (``sex``, ``age``), and take it back
    from the tracks that had one applied and have no call now."""
    from . import classification_store

    path = store.stage_path(target_folder, store.TRACKS, modality)
    if not os.path.isfile(path):
        return 0

    wanted = {}
    for row in predictions:
        value = resolve(values, row["label"])
        if value is not None:
            wanted[int(row["track_id"])] = int(value)
    previous = classification_store.applied_values(
        target_folder, modality, field)
    if not wanted and not previous:
        return 0

    manual_run = _manual_run_id(target_folder, modality)
    written = withdrawn = 0
    applied: Dict[int, int] = {}
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            for row in conn.execute(
                    "SELECT track_id, run_id, attributes FROM tracks"):
                track_id = int(row["track_id"])
                if manual_run is not None and row["run_id"] == manual_run:
                    continue
                attributes = _load_json(row["attributes"])
                if field in track_review.by_hand(attributes):
                    continue      # a reviewer's word stands
                if track_id in wanted:
                    applied[track_id] = wanted[track_id]
                    if attributes.get(field) == wanted[track_id]:
                        continue
                    attributes[field] = wanted[track_id]
                    written += 1
                elif track_id not in previous:
                    continue
                elif attributes.get(field) == previous[track_id]:
                    # No call any more, and the value is still the one the
                    # earlier call put there.
                    del attributes[field]
                    withdrawn += 1
                else:
                    continue
                conn.execute(
                    "UPDATE tracks SET attributes = ? WHERE track_id = ?",
                    (json.dumps(attributes), track_id))
    finally:
        conn.close()

    classification_store.record_applied_values(
        target_folder, modality, field, applied)

    if log_fn and written:
        log_fn(f"Applied {field} to {written} track(s)")
    if log_fn and withdrawn:
        log_fn(f"Took {field} back from {withdrawn} track(s) that have no "
               "call any more")
    return written + withdrawn


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------

def apply_all(target_folder: str, modality: str, models: dict,
              tasks: Sequence[str] = (), overwrite_detections: bool = False,
              log_fn=None) -> Dict[str, int]:
    """Project every stored classification result onto the canonical fields.

    Idempotent: running it twice changes nothing the second time, and running
    it after something else rewrote the detection species puts the answer back.
    """
    from . import classification_store, label_store

    vocabulary = label_store.vocabulary(target_folder)
    if not vocabulary:
        if log_fn:
            log_fn("This project has no vocabulary to write results into.")
        return {}

    wanted = set(tasks) if tasks else {"occlusion", "species", "sex",
                                       classification_store.LIFE_STAGE}
    counts: Dict[str, int] = {}

    if "occlusion" in wanted:
        values = label_values(models.get("occlusion") or {}, vocabulary,
                              "occlusion")
        counts["occlusion"] = apply_occlusion(
            target_folder, modality,
            classification_store.frame_predictions(
                target_folder, modality, "occlusion"),
            values, log_fn=log_fn)

    if "species" in wanted:
        values = label_values(models.get("species") or {}, vocabulary,
                              "species")
        applied = apply_track_species(
            target_folder, modality,
            classification_store.track_predictions(
                target_folder, modality, "species"),
            values, overwrite_detections=overwrite_detections, log_fn=log_fn)
        counts["species_tracks"] = applied["tracks"]
        counts["species_detections"] = applied["detections"]

    for task, field in TRACK_FIELDS.items():
        if task not in wanted:
            continue
        values = label_values(models.get(task) or {}, vocabulary, task)
        counts[field] = apply_track_attribute(
            target_folder, modality, field,
            classification_store.track_predictions(
                target_folder, modality, task),
            values, log_fn=log_fn)

    return counts
