# -*- coding: utf-8 -*-
"""How right the pipeline was: its output against hand labels.

A flight that somebody went through in the labelling tool carries its own
answer key: every animal that person drew is known, frame by frame, with its
species, sex and age. Comparing the pipeline's output with it says how far
the numbers of the flights nobody went through can be trusted - and where
the pipeline fails: which species it misses, at which confidence a box is
worth keeping, whether it splits one animal into several.

Three questions, in this order, because each needs the one before:

**Were the animals found?** Frame by frame, the detector's boxes are paired
with the labelled ones: the best overlapping pairs first, a box used once, a
pair needing an overlap (intersection over union) of at least a threshold
(:data:`DEFAULT_IOU`). A labelled animal with a box on it is *found*, one
without is *missed*, a box on nothing is a *false alarm*. From these:
**precision** (the share of boxes that are animals), **recall** (the share
of animals that got a box) - overall, per species, and for every confidence
a box may be asked to have.

**Was each animal one track?** A census counts tracks. So per labelled
animal: whether any track follows it, for how much of the time, and in how
many pieces; per track: whether it follows a labelled animal at all, or
more than one. The number of tracks against the number of labelled animals
is what the census would have got wrong.

**Was it named right?** For the animals that were found, the species the
survey tools count the track as against the labelled one - a confusion
table - and the same for sex and age; where a classifier made the call,
also per classifier model.

**What counts as labelled.** A box drawn on a frame says there is an animal.
No box says nothing by itself: the frame may be empty, or nobody looked. The
comparison therefore needs to know which frames were gone through - by
default every frame from the first to the last one that has a label
(:func:`labelled_frames`), or the range the caller names. A label of the
species ``not-an-animal`` is the reviewer saying "this is none": a box on it
is a confirmed false alarm, and it is not an animal that could be missed.

Labels made by importing a pipeline track and correcting it are compared
like any other. They flatter the pipeline a little - a box nobody moved
matches itself exactly - which the report says when there are such labels.

Free of QGIS and Qt.
"""

import json
import os
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import store

#: Overlap two boxes need to be the same animal. Hand labels are drawn on
#: key frames and interpolated between them, and a thermal animal is a few
#: pixels across: the 0.5 of benchmark data sets would call many a hit a
#: miss.
DEFAULT_IOU = 0.3
MIN_IOU, MAX_IOU = 0.05, 0.95

#: Confidences the precision and recall are worked out at.
THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

NOT_AN_ANIMAL = "not-an-animal"
UNDETERMINED = ("animal", "unknown", "")
NO_TRACK = "(no track)"

Box = Tuple[float, float, float, float]


class AccuracyError(RuntimeError):
    """There is nothing to compare."""


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes ``(x1, y1, x2, y2)``."""
    ax1, ax2 = sorted((a[0], a[2]))
    ay1, ay2 = sorted((a[1], a[3]))
    bx1, bx2 = sorted((b[0], b[2]))
    by1, by2 = sorted((b[1], b[3]))
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - shared
    return shared / union if union > 0 else 0.0


def pair_boxes(labels: Sequence[Box], boxes: Sequence[Box],
               threshold: float) -> List[Tuple[int, int, float]]:
    """Pair the labelled boxes of a frame with the detector's.

    The best overlapping pair first, then the best of what is left: a box is
    in one pair at most, and a pair needs an overlap of at least
    *threshold*.

    :return: ``(label index, box index, overlap)`` per pair
    """
    candidates = []
    for i, label in enumerate(labels):
        for j, box in enumerate(boxes):
            overlap = iou(label, box)
            if overlap >= threshold:
                candidates.append((overlap, i, j))
    # the highest overlap first; of equals, the earlier boxes
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_labels, used_boxes, pairs = set(), set(), []
    for overlap, i, j in candidates:
        if i in used_labels or j in used_boxes:
            continue
        used_labels.add(i)
        used_boxes.add(j)
        pairs.append((i, j, overlap))
    return pairs


def _rate(part: float, whole: float) -> Optional[float]:
    return part / whole if whole else None


def scores(found: int, missed: int, false_alarms: int) -> dict:
    """Precision, recall and their harmonic mean (F1) from the counts."""
    precision = _rate(found, found + false_alarms)
    recall = _rate(found, found + missed)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {"found": found, "missed": missed, "false_alarms": false_alarms,
            "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

def labelled_frames(labels: Sequence[dict]) -> Optional[Tuple[int, int]]:
    """The stretch of frames that was gone through, as far as the labels
    say: from the first to the last frame that has one. ``None`` without
    labels."""
    frames = [frame for label in labels for frame in label["boxes"]]
    return (min(frames), max(frames)) if frames else None


def compare(labels: Sequence[dict], boxes: Sequence[dict],
            tracks: Optional[Dict[int, dict]] = None,
            threshold: float = DEFAULT_IOU,
            frames: Optional[Tuple[int, int]] = None,
            classifiers: Optional[Dict[str, Dict[int, dict]]] = None) -> dict:
    """Compare what the pipeline found with the hand labels.

    :param labels: the labelled animals - ``{"id", "species", "sex", "age",
        "boxes": {frame: (x1, y1, x2, y2)}, "imported": bool}``; species
        ``not-an-animal`` marks something a reviewer ruled out
    :param boxes: the pipeline's boxes - ``{"id", "frame", "box",
        "confidence", "species", "track"}``; ``track`` is the track the box
        is part of, or ``None``
    :param tracks: ``track id -> {"species", "sex", "age"}`` as the survey
        tools count it
    :param threshold: the overlap a pair of boxes needs
    :param frames: ``(first, last)`` frame that was gone through; ``None``
        = from the first to the last labelled frame
    :param classifiers: ``task -> {track id: {"label", "model"}}`` - what a
        classifier called a track, for the per-model tables
    :return: the result document (see the keys below and :func:`describe`)
    :raises AccuracyError: without any label in the frames
    """
    threshold = min(MAX_IOU, max(MIN_IOU, float(threshold)))
    tracks = tracks or {}
    span = frames or labelled_frames(labels)
    if span is None:
        raise AccuracyError(
            "There are no hand labels to compare with. Label some animals "
            "in the Labelling Tool first.")
    first, last = int(min(span)), int(max(span))

    by_frame_labels: Dict[int, List[Tuple[dict, Box]]] = defaultdict(list)
    for label in labels:
        for frame, box in label["boxes"].items():
            if first <= frame <= last:
                by_frame_labels[frame].append((label, box))
    by_frame_boxes: Dict[int, List[dict]] = defaultdict(list)
    for box in boxes:
        if first <= box["frame"] <= last:
            by_frame_boxes[box["frame"]].append(box)
    if not by_frame_labels:
        raise AccuracyError(
            f"No hand label lies in the frames {first} to {last}.")

    # ---- frame by frame: which box is on which label --------------------- #
    rows: List[dict] = []                 # one per label box and per box
    for frame in sorted(set(by_frame_labels) | set(by_frame_boxes)):
        of_labels = by_frame_labels.get(frame, [])
        of_boxes = by_frame_boxes.get(frame, [])
        pairs = pair_boxes([box for _label, box in of_labels],
                           [box["box"] for box in of_boxes], threshold)
        paired_labels = {i: (j, overlap) for i, j, overlap in pairs}
        paired_boxes = {j for _i, j, _overlap in pairs}
        for i, (label, _box) in enumerate(of_labels):
            negative = label["species"] == NOT_AN_ANIMAL
            row = {"frame": frame, "label": label["id"],
                   "label_species": label["species"], "negative": negative,
                   "box": None, "track": None, "confidence": None,
                   "species": None, "iou": None}
            if i in paired_labels:
                j, overlap = paired_labels[i]
                found = of_boxes[j]
                row.update({"box": found["id"], "track": found.get("track"),
                            "confidence": found.get("confidence"),
                            "species": found.get("species"), "iou": overlap})
            row["kind"] = ("ruled out" if negative and row["box"] is not None
                           else "ruled out, no box" if negative
                           else "found" if row["box"] is not None
                           else "missed")
            rows.append(row)
        for j, found in enumerate(of_boxes):
            if j in paired_boxes:
                continue
            rows.append({"frame": frame, "label": None, "label_species": None,
                         "negative": False, "box": found["id"],
                         "track": found.get("track"),
                         "confidence": found.get("confidence"),
                         "species": found.get("species"), "iou": None,
                         "kind": "false alarm"})

    animals = [label for label in labels
               if label["species"] != NOT_AN_ANIMAL and any(
                   first <= frame <= last for frame in label["boxes"])]
    ruled_out = [label for label in labels
                 if label["species"] == NOT_AN_ANIMAL and any(
                     first <= frame <= last for frame in label["boxes"])]

    document = OrderedDict()
    document["frames"] = {
        "first": first, "last": last, "given": frames is not None,
        "with_labels": len({frame for frame, pairs in by_frame_labels.items()
                            if any(label["species"] != NOT_AN_ANIMAL
                                   for label, _box in pairs)}),
        "with_boxes": len(by_frame_boxes)}
    document["iou_threshold"] = threshold
    document["labels"] = {
        "animals": len(animals), "ruled_out": len(ruled_out),
        "imported": sum(1 for label in animals if label.get("imported")),
        "boxes": sum(1 for row in rows
                     if row["label"] is not None and not row["negative"])}
    document["detection"] = _detection(rows)
    document["tracking"] = _tracking(animals, rows, tracks, by_frame_boxes)
    document["naming"] = _naming(animals, rows, tracks, classifiers or {})
    document["rows"] = rows
    return document


def _detection(rows: Sequence[dict]) -> dict:
    """Were the animals found: counts and rates over the boxes."""
    def count(kind):
        return sum(1 for row in rows if row["kind"] == kind)

    found, missed = count("found"), count("missed")
    # a box on something a reviewer ruled out is a false alarm like any
    # other - only one that somebody confirmed
    confirmed = count("ruled out")
    false_alarms = count("false alarm") + confirmed
    overlaps = [row["iou"] for row in rows if row["kind"] == "found"]
    result = scores(found, missed, false_alarms)
    result["confirmed_false_alarms"] = confirmed
    result["mean_iou"] = sum(overlaps) / len(overlaps) if overlaps else None

    # per labelled species: how many of its boxes were found
    by_species: Dict[str, dict] = OrderedDict()
    for row in rows:
        if row["label"] is None or row["negative"]:
            continue
        entry = by_species.setdefault(
            row["label_species"] or "animal", {"found": 0, "missed": 0})
        entry["found" if row["kind"] == "found" else "missed"] += 1
    for entry in by_species.values():
        entry["recall"] = _rate(entry["found"],
                                entry["found"] + entry["missed"])
    result["by_species"] = by_species

    # per species the detector called a box: how many of them are animals
    called: Dict[str, dict] = OrderedDict()
    for row in rows:
        if row["box"] is None:
            continue
        entry = called.setdefault(row["species"] or "animal",
                                  {"animals": 0, "false_alarms": 0})
        entry["animals" if row["kind"] == "found" else "false_alarms"] += 1
    for entry in called.values():
        entry["precision"] = _rate(
            entry["animals"], entry["animals"] + entry["false_alarms"])
    result["by_called_species"] = called

    result["by_confidence"] = _by_confidence(rows)
    best = [entry for entry in result["by_confidence"]
            if entry["f1"] is not None]
    result["best_confidence"] = max(
        best, key=lambda entry: (entry["f1"], -entry["confidence"])
    )["confidence"] if best else None
    return result


def _by_confidence(rows: Sequence[dict]) -> List[dict]:
    """Precision and recall if only boxes of at least a confidence were
    kept. An animal whose box falls below it becomes a missed one."""
    labelled = sum(1 for row in rows
                   if row["label"] is not None and not row["negative"])
    table = []
    for threshold in THRESHOLDS:
        found = false_alarms = 0
        for row in rows:
            if row["box"] is None:
                continue
            confidence = row["confidence"]
            if confidence is not None and confidence < threshold:
                continue
            if row["kind"] == "found":
                found += 1
            else:
                false_alarms += 1
        entry = scores(found, labelled - found, false_alarms)
        entry["confidence"] = threshold
        table.append(entry)
    return table


def _tracking(animals: Sequence[dict], rows: Sequence[dict],
              tracks: Dict[int, dict],
              by_frame_boxes: Dict[int, List[dict]]) -> dict:
    """Was each animal one track."""
    # which tracks have a box on which labelled animal, and how often
    on_animal: Dict[object, Counter] = defaultdict(Counter)
    on_track: Dict[object, Counter] = defaultdict(Counter)
    negative_tracks: Counter = Counter()
    for row in rows:
        if row["track"] is None or row["label"] is None:
            continue
        if row["negative"]:
            negative_tracks[row["track"]] += 1
            continue
        on_animal[row["label"]][row["track"]] += 1
        on_track[row["track"]][row["label"]] += 1

    in_frames = {box["track"] for boxes in by_frame_boxes.values()
                 for box in boxes if box.get("track") is not None}
    per_animal = []
    for label in animals:
        followed = on_animal.get(label["id"], Counter())
        frames = sum(1 for row in rows
                     if row["label"] == label["id"])
        boxed = sum(1 for row in rows
                    if row["label"] == label["id"] and row["box"] is not None)
        tracked = sum(followed.values())
        best = followed.most_common(1)[0][0] if followed else None
        per_animal.append({
            "label": label["id"], "species": label["species"],
            "frames": frames, "frames_with_box": boxed,
            "frames_tracked": tracked,
            "share_tracked": _rate(tracked, frames),
            "tracks": sorted(followed), "pieces": len(followed),
            "best_track": best})
    per_track = []
    for track in sorted(in_frames):
        animals_of = on_track.get(track, Counter())
        per_track.append({
            "track": track,
            "species": (tracks.get(track) or {}).get("species"),
            "animals": sorted(animals_of),
            "frames_on_animals": sum(animals_of.values()),
            "frames_on_ruled_out": negative_tracks.get(track, 0)})

    found = [a for a in per_animal if a["pieces"]]
    counted = [t for t in per_track
               if (t["species"] or "") != NOT_AN_ANIMAL]
    true_tracks = [t for t in counted if t["animals"]]
    shares = [a["share_tracked"] for a in found
              if a["share_tracked"] is not None]
    return {
        "animals": len(per_animal),
        "animals_tracked": len(found),
        "animals_not_tracked": len(per_animal) - len(found),
        "animals_in_pieces": sum(1 for a in found if a["pieces"] > 1),
        "share_of_animals_tracked": _rate(len(found), len(per_animal)),
        "mean_share_of_time_tracked": (sum(shares) / len(shares)
                                       if shares else None),
        "tracks": len(counted),
        "tracks_on_animals": len(true_tracks),
        "tracks_on_nothing": len(counted) - len(true_tracks),
        "tracks_on_several_animals": sum(
            1 for t in counted if len(t["animals"]) > 1),
        # what a census would have counted, against what was there
        "count_error": len(counted) - len(per_animal),
        "per_animal": per_animal,
        "per_track": per_track,
    }


def confusion(pairs: Iterable[Tuple[str, str]]) -> dict:
    """A confusion table from ``(labelled, called)`` pairs::

        {"labels": [...], "called": [...], "counts": {labelled: {called: n}},
         "n", "right", "accuracy",
         "per_label": {label: {"n", "right", "recall", "precision"}}}
    """
    counts: Dict[str, Counter] = OrderedDict()
    for labelled, called in pairs:
        counts.setdefault(labelled, Counter())[called] += 1
    labels = sorted(counts)
    called_names = sorted({name for row in counts.values() for name in row})
    n = sum(sum(row.values()) for row in counts.values())
    right = sum(counts[name].get(name, 0) for name in labels)
    per_label = OrderedDict()
    for name in labels:
        total = sum(counts[name].values())
        called_as = sum(row.get(name, 0) for row in counts.values())
        per_label[name] = {
            "n": total, "right": counts[name].get(name, 0),
            "recall": _rate(counts[name].get(name, 0), total),
            "precision": _rate(counts[name].get(name, 0), called_as)}
    return {"labels": labels, "called": called_names,
            "counts": {name: dict(row) for name, row in counts.items()},
            "n": n, "right": right, "accuracy": _rate(right, n),
            "per_label": per_label}


def _known(value) -> Optional[str]:
    text = "" if value is None else str(value).strip().lower()
    return None if text in UNDETERMINED or text in ("none", "n/a") else text


def _naming(animals: Sequence[dict], rows: Sequence[dict],
            tracks: Dict[int, dict],
            classifiers: Dict[str, Dict[int, dict]]) -> dict:
    """Was it named right: for the animals that were found, what the
    pipeline called them against the label."""
    followed: Dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        if row["label"] is not None and not row["negative"] and \
                row["track"] is not None:
            followed[row["label"]][row["track"]] += 1

    result = OrderedDict()
    for field in ("species", "sex", "age"):
        pairs = []
        for label in animals:
            truth = _known(label.get(field))
            if truth is None:
                continue                  # the label does not say
            of_label = followed.get(label["id"])
            if not of_label:
                called = NO_TRACK
            else:
                track = of_label.most_common(1)[0][0]
                called = _known((tracks.get(track) or {}).get(field)) or \
                    "undetermined"
            pairs.append((truth, called))
        result[field] = confusion(pairs)

    # what each classifier model said, where a classifier made the call
    by_model: Dict[str, dict] = OrderedDict()
    for task, field in (("species", "species"), ("sex", "sex"),
                        ("life_stage", "age")):
        calls = classifiers.get(task) or {}
        pairs_of: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for label in animals:
            truth = _known(label.get(field))
            of_label = followed.get(label["id"])
            if truth is None or not of_label:
                continue
            call = calls.get(of_label.most_common(1)[0][0])
            if not call or not call.get("label"):
                continue
            model = os.path.basename(str(call.get("model") or "")) or \
                "(unnamed model)"
            pairs_of[model].append(
                (truth, _known(call["label"]) or "undetermined"))
        for model, pairs in sorted(pairs_of.items()):
            by_model.setdefault(field, OrderedDict())[model] = confusion(pairs)
    result["by_model"] = by_model
    return result


# ---------------------------------------------------------------------------
# A flight
# ---------------------------------------------------------------------------

def load_labels(target_folder: str, modality: str) -> List[dict]:
    """The hand labels of a flight, a box per frame each."""
    from . import label_store
    from .exporters import common

    vocabulary = common.load_vocabulary(target_folder)
    names = vocabulary["species"]
    enum_labels = vocabulary["enum_labels"]
    labels = []
    for track in label_store.load_tracks(target_folder, modality):
        boxes = {int(row["frame"]): (row["x1"], row["y1"], row["x2"],
                                     row["y2"])
                 for row in label_store.interpolate(track["keyframes"])}
        if not boxes:
            continue
        attributes = common.resolve_attributes(
            json.dumps(track.get("attributes") or {}), enum_labels)
        labels.append({
            "id": int(track["label_track_id"]),
            "species": names.get(int(track["species_id"]), "animal"),
            "sex": attributes.get("sex"), "age": attributes.get("age"),
            "imported": track.get("origin_track_id") is not None,
            "boxes": boxes})
    return labels


def load_pipeline(target_folder: str, modality: str) -> dict:
    """What the pipeline itself produced for a flight: the boxes that are
    not hand-drawn, the tracks of the tracker's run - every one of them,
    also where a label has taken its place since - and what the classifiers
    called them.

    :return: ``{"boxes", "tracks", "classifiers", "sources", "tracker"}``
    """
    from . import classification_store, gpkg, track_store
    from .exporters import common

    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(det_path):
        raise AccuracyError(
            "This camera has no detections yet: run 'Detect Animals' first.")
    vocabulary = common.load_vocabulary(target_folder)
    names = vocabulary["species"]
    enum_labels = vocabulary["enum_labels"]

    sources = {}
    project = store.open_store(store.project_path(target_folder),
                               store.PROJECT)
    try:
        for row in project.execute(
                "SELECT source_id, kind, model FROM detection_sources"):
            sources[int(row["source_id"])] = {
                "kind": row["kind"], "model": row["model"] or ""}
    finally:
        project.close()

    run = track_store.active_run(target_folder, modality)
    trk_path = store.stage_path(target_folder, store.TRACKS, modality)
    conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        boxes = {}
        for row in conn.execute(
                "SELECT detection_id, frame, x1, y1, x2, y2, confidence, "
                "species_id, source_id FROM detections "
                "WHERE label_track_id IS NULL ORDER BY detection_id"):
            boxes[int(row["detection_id"])] = {
                "id": int(row["detection_id"]), "frame": int(row["frame"]),
                "box": (row["x1"], row["y1"], row["x2"], row["y2"]),
                "confidence": row["confidence"],
                "species": names.get(int(row["species_id"]), "animal"),
                "species_id": int(row["species_id"]),
                "source": int(row["source_id"]), "track": None}
        members: Dict[int, List[int]] = defaultdict(list)
        own_species: Dict[int, int] = {}
        attributes: Dict[int, dict] = {}
        if run is not None and os.path.isfile(trk_path):
            gpkg.attach(conn, trk_path, "trk")
            for row in conn.execute(
                    "SELECT t.track_id, t.species_id, t.attributes, "
                    "m.detection_id FROM trk.tracks t "
                    "JOIN trk.track_members m ON m.track_id = t.track_id "
                    "WHERE t.run_id = ? ORDER BY t.track_id",
                    (run["run_id"],)):
                track = int(row["track_id"])
                own_species[track] = int(row["species_id"])
                attributes[track] = common.resolve_attributes(
                    row["attributes"], enum_labels)
                if int(row["detection_id"]) in boxes:
                    members[track].append(int(row["detection_id"]))
            gpkg.detach(conn, "trk")
    finally:
        conn.close()

    tracks = {}
    for track, detections in members.items():
        for detection in detections:
            boxes[detection]["track"] = track
        # the species the survey tools count the animal as: the track's own
        # where it was identified, else what most of its boxes were called
        species_id = own_species.get(track, store.FALLBACK_SPECIES_ID)
        if species_id == store.FALLBACK_SPECIES_ID:
            species_id = track_store.majority_species(
                (boxes[d]["species_id"], boxes[d]["confidence"])
                for d in detections)
        tracks[track] = {
            "species": names.get(species_id, "animal"),
            "sex": attributes.get(track, {}).get("sex"),
            "age": attributes.get(track, {}).get("age")}

    classifiers = {}
    for task in ("species", "sex", classification_store.LIFE_STAGE):
        calls = {int(p["track_id"]): {"label": p["label"],
                                      "model": p.get("model") or ""}
                 for p in classification_store.track_predictions(
                     target_folder, modality, task)}
        if calls:
            classifiers[task] = calls
    used = {box["source"] for box in boxes.values()}
    return {
        "boxes": list(boxes.values()), "tracks": tracks,
        "classifiers": classifiers,
        "sources": [dict(sources.get(source, {"kind": "unknown",
                                              "model": ""}),
                         boxes=sum(1 for b in boxes.values()
                                   if b["source"] == source))
                    for source in sorted(used)],
        "tracker": None if run is None else {
            "kind": run["kind"], "tracker": run.get("tracker") or "",
            "tracks": len(tracks)},
    }


def for_flight(target_folder: str, modality: str,
               threshold: float = DEFAULT_IOU,
               frames: Optional[Tuple[int, int]] = None) -> dict:
    """The accuracy of a flight's pipeline output against its hand labels
    (:func:`compare`), with what was compared."""
    labels = load_labels(target_folder, modality)
    pipeline = load_pipeline(target_folder, modality)
    document = compare(labels, pipeline["boxes"], pipeline["tracks"],
                       threshold, frames, pipeline["classifiers"])
    document["modality"] = modality
    document["sources"] = pipeline["sources"]
    document["tracker"] = pipeline["tracker"]
    document["classifier_models"] = {
        task: sorted({os.path.basename(call["model"]) or "(unnamed model)"
                      for call in calls.values()})
        for task, calls in pipeline["classifiers"].items()}
    return document


# ---------------------------------------------------------------------------
# In words, and on disk
# ---------------------------------------------------------------------------

def percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.0f} %"


def describe(document: dict) -> List[str]:
    """The result in a few lines, for a log."""
    frames = document["frames"]
    labels = document["labels"]
    found = document["detection"]
    tracking = document["tracking"]
    lines = [
        f"Compared with {labels['animals']} labelled animal(s) "
        f"({labels['boxes']} boxes) in the frames {frames['first']} to "
        f"{frames['last']}, at an overlap of at least "
        f"{document['iou_threshold']:g}",
        f"Boxes: {found['found']} on an animal, {found['missed']} animals "
        f"missed, {found['false_alarms']} false alarm(s) - precision "
        f"{percent(found['precision'])}, recall {percent(found['recall'])}",
    ]
    if found.get("best_confidence") is not None:
        lines.append(
            "The balance of precision and recall is best with boxes of a "
            f"confidence of at least {found['best_confidence']:g}")
    if tracking["tracks"] or tracking["animals_tracked"]:
        error = tracking["count_error"]
        counted = ("as many as were labelled" if error == 0 else
                   f"{abs(error)} {'more' if error > 0 else 'fewer'} than "
                   "were labelled")
        lines.append(
            f"Tracks: {tracking['animals_tracked']} of "
            f"{tracking['animals']} animals are followed by a track "
            f"({tracking['animals_in_pieces']} in several pieces); "
            f"{tracking['tracks_on_nothing']} of {tracking['tracks']} "
            f"track(s) follow nothing that was labelled. A census counts "
            f"{tracking['tracks']} animals - {counted}")
    for field, what in (("species", "Species"), ("sex", "Sex"),
                        ("age", "Age")):
        table = document["naming"].get(field) or {}
        if table.get("n"):
            lines.append(
                f"{what}: {table['right']} of {table['n']} named as "
                f"labelled ({percent(table['accuracy'])})")
    if labels.get("imported"):
        lines.append(
            f"{labels['imported']} of the labels were made from a pipeline "
            "track and corrected: where nobody moved a box it matches "
            "itself, which flatters the figures a little")
    return lines


#: Columns of the list of boxes (``accuracy_{m}_boxes.csv``).
ROW_COLUMNS = ["frame", "kind", "label", "label_species", "box", "track",
               "species", "confidence", "iou"]


def write_rows(document: dict, path: str) -> str:
    """Every labelled box and every box of the pipeline in the compared
    frames, one row each, with what became of it: ``found``, ``missed``,
    ``false alarm``, ``ruled out``."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ROW_COLUMNS)
        for row in document["rows"]:
            values = []
            for column in ROW_COLUMNS:
                value = row.get(column)
                if value is None:
                    value = ""
                elif isinstance(value, float):
                    value = round(value, 3)
                values.append(value)
            writer.writerow(values)
    return path


def write_json(document: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    return path


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


# ---------------------------------------------------------------------------
# As a page
# ---------------------------------------------------------------------------

#: Rows of a per-animal or per-track table that are written out; a flight
#: with thousands of animals is summed up above them anyway.
MAX_LISTED = 300

_CSS = """
table { border-collapse: collapse; margin: 4px 0 12px 0; }
th, td { border: 1px solid #c8c8c8; padding: 3px 8px; }
th { background: #efefef; text-align: left; }
td.n { text-align: right; }
td.hit { background: #e3f1e3; font-weight: bold; }
tr.best td { font-weight: bold; }
p.note { color: #666666; }
"""


def _esc(value) -> str:
    import html
    return html.escape("" if value is None else str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence],
           numeric: Sequence[int] = (), best: Optional[int] = None) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    out = [f"<table><tr>{head}</tr>"]
    for index, row in enumerate(rows):
        cells = "".join(
            f"<td class='n'>{_esc(cell)}</td>" if column in numeric
            else f"<td>{_esc(cell)}</td>"
            for column, cell in enumerate(row))
        mark = " class='best'" if index == best else ""
        out.append(f"<tr{mark}>{cells}</tr>")
    out.append("</table>")
    return "".join(out)


def _confusion_table(table: dict) -> str:
    """Rows: what was labelled; columns: what the pipeline called it."""
    called = table["called"]
    head = "".join(f"<th>{_esc(name)}</th>" for name in called)
    out = ["<table><tr><th>labelled \\ called</th>"
           f"{head}<th>n</th><th>right</th></tr>"]
    for label in table["labels"]:
        counts = table["counts"][label]
        cells = "".join(
            f"<td class='n{' hit' if name == label else ''}'>"
            f"{counts.get(name, 0) or ''}</td>" for name in called)
        entry = table["per_label"][label]
        out.append(f"<tr><th>{_esc(label)}</th>{cells}"
                   f"<td class='n'>{entry['n']}</td>"
                   f"<td class='n'>{percent(entry['recall'])}</td></tr>")
    out.append("</table>")
    return "".join(out)


_FIRST_TO_LAST = "first to last labelled frame"


def html_report(document: dict, title: str = "Accuracy against hand labels",
                page: bool = True) -> str:
    """The comparison as an HTML page (or, with ``page=False``, as the part
    of one that goes between ``<body>`` and ``</body>``)."""
    frames, labels = document["frames"], document["labels"]
    found, tracking = document["detection"], document["tracking"]
    naming = document["naming"]
    out = [f"<h2>{_esc(title)}</h2>"]

    facts = [("Labelled animals", labels["animals"]),
             ("Their boxes", labels["boxes"]),
             ("Ruled out by a reviewer (not-an-animal)", labels["ruled_out"]),
             ("Frames compared",
              f"{frames['first']} to {frames['last']} "
              f"({'as given' if frames.get('given') else _FIRST_TO_LAST})"),
             ("Overlap a pair of boxes needs (IoU)",
              f"{document['iou_threshold']:g}")]
    for source in document.get("sources") or []:
        name = source.get("model") or "(no model name)"
        facts.append((f"Boxes of the {source.get('kind', '')}",
                      f"{source.get('boxes', 0)} - {name}"))
    tracker = document.get("tracker")
    if tracker:
        facts.append(("Tracker", f"{tracker.get('tracker') or tracker['kind']}"
                                 f" - {tracker['tracks']} track(s)"))
    for task, models in (document.get("classifier_models") or {}).items():
        facts.append((f"Classifier ({task})", ", ".join(models)))
    out.append(_table(["", ""], facts))

    # ---- found ------------------------------------------------------------ #
    out.append("<h3>Were the animals found?</h3>")
    out.append(_table(
        ["Boxes on an animal", "Animals missed", "False alarms",
         "... confirmed by a reviewer", "Precision", "Recall", "F1",
         "Mean overlap"],
        [[found["found"], found["missed"], found["false_alarms"],
          found["confirmed_false_alarms"], percent(found["precision"]),
          percent(found["recall"]), percent(found["f1"]),
          "-" if found["mean_iou"] is None else f"{found['mean_iou']:.2f}"]],
        numeric=range(8)))
    out.append("<p class='note'>Counted per frame: an animal on six frames "
               "can be found on four and missed on two. Precision is the "
               "share of boxes that are on an animal, recall the share of "
               "labelled boxes that got one.</p>")
    if found["by_species"]:
        out.append("<h4>Per labelled species</h4>")
        out.append(_table(
            ["Species", "Boxes", "Found", "Missed", "Recall"],
            [[name, entry["found"] + entry["missed"], entry["found"],
              entry["missed"], percent(entry["recall"])]
             for name, entry in found["by_species"].items()],
            numeric=(1, 2, 3, 4)))
    if found["by_called_species"]:
        out.append("<h4>What the detector called its boxes</h4>")
        out.append(_table(
            ["Called", "Boxes", "On an animal", "False alarms", "Precision"],
            [[name, entry["animals"] + entry["false_alarms"],
              entry["animals"], entry["false_alarms"],
              percent(entry["precision"])]
             for name, entry in found["by_called_species"].items()],
            numeric=(1, 2, 3, 4)))
    confidences = [row for row in found["by_confidence"]]
    if any(row["found"] or row["false_alarms"] for row in confidences):
        best = next((index for index, row in enumerate(confidences)
                     if row["confidence"] == found.get("best_confidence")),
                    None)
        out.append("<h4>If only boxes of at least this confidence were "
                   "kept</h4>")
        out.append(_table(
            ["Confidence", "Found", "Missed", "False alarms", "Precision",
             "Recall", "F1"],
            [[f"{row['confidence']:g}", row["found"], row["missed"],
              row["false_alarms"], percent(row["precision"]),
              percent(row["recall"]), percent(row["f1"])]
             for row in confidences], numeric=range(7), best=best))
        out.append("<p class='note'>The row in bold has the best balance of "
                   "precision and recall (F1). Boxes below the detector's "
                   "own threshold were never stored, so the table starts "
                   "where that threshold was.</p>")

    # ---- tracks ----------------------------------------------------------- #
    out.append("<h3>Was each animal one track?</h3>")
    error = tracking["count_error"]
    out.append(_table(
        ["Labelled animals", "Followed by a track", "... in several pieces",
         "Not followed", "Mean share of the time followed", "Tracks",
         "Tracks on nothing labelled", "Tracks over several animals",
         "A census counts"],
        [[tracking["animals"], tracking["animals_tracked"],
          tracking["animals_in_pieces"], tracking["animals_not_tracked"],
          percent(tracking["mean_share_of_time_tracked"]),
          tracking["tracks"], tracking["tracks_on_nothing"],
          tracking["tracks_on_several_animals"],
          f"{tracking['tracks']} ({error:+d})"]], numeric=range(9)))
    out.append("<p class='note'>A census counts tracks. An animal in two "
               "pieces is counted twice, a track on nothing adds an animal "
               "that was not there, and an animal no track follows is not "
               "counted at all.</p>")
    pieces = [a for a in tracking["per_animal"] if a["pieces"] != 1]
    if pieces:
        out.append("<h4>Animals that are not exactly one track</h4>")
        out.append(_table(
            ["Label", "Species", "Frames", "With a box", "Followed",
             "Pieces", "Tracks"],
            [[a["label"], a["species"], a["frames"], a["frames_with_box"],
              percent(a["share_tracked"]), a["pieces"],
              ", ".join(str(t) for t in a["tracks"]) or "-"]
             for a in pieces[:MAX_LISTED]], numeric=(0, 2, 3, 4, 5)))
        if len(pieces) > MAX_LISTED:
            out.append(f"<p class='note'>... and {len(pieces) - MAX_LISTED} "
                       "more; all of them are in the JSON file.</p>")
    stray = [t for t in tracking["per_track"]
             if not t["animals"] and (t["species"] or "") != NOT_AN_ANIMAL]
    if stray:
        out.append("<h4>Tracks that follow nothing that was labelled</h4>")
        out.append(_table(
            ["Track", "Counted as", "Frames on something ruled out"],
            [[t["track"], t["species"] or "animal", t["frames_on_ruled_out"]]
             for t in stray[:MAX_LISTED]], numeric=(0, 2)))
        if len(stray) > MAX_LISTED:
            out.append(f"<p class='note'>... and {len(stray) - MAX_LISTED} "
                       "more.</p>")

    # ---- names ------------------------------------------------------------ #
    out.append("<h3>Was it named right?</h3>")
    said = False
    for field, what in (("species", "Species"), ("sex", "Sex"),
                        ("age", "Age")):
        table = naming.get(field) or {}
        if not table.get("n"):
            continue
        said = True
        out.append(f"<h4>{what}: {table['right']} of {table['n']} as "
                   f"labelled ({percent(table['accuracy'])})</h4>")
        out.append(_confusion_table(table))
        for model, of_model in (naming.get("by_model") or {}).get(
                field, {}).items():
            out.append(f"<p>The classifier's own call ({_esc(model)}): "
                       f"{of_model['right']} of {of_model['n']} "
                       f"({percent(of_model['accuracy'])})</p>")
            out.append(_confusion_table(of_model))
    if said:
        out.append(
            "<p class='note'>Per labelled animal that says what it is: what "
            "the survey tools count the track as that follows it for the "
            "longest. \"undetermined\" is a track nobody named; "
            f"\"{NO_TRACK}\" an animal no track follows.</p>")
    else:
        out.append("<p class='note'>No label says a species, sex or age "
                   "that could be compared.</p>")
    if labels.get("imported"):
        out.append(
            f"<p class='note'>{labels['imported']} of the labels were made "
            "from a pipeline track and corrected. Where nobody moved a box "
            "it matches itself, which flatters the figures a little.</p>")
    body = "".join(out)
    if not page:
        return body
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")
