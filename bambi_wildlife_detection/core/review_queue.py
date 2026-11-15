# -*- coding: utf-8 -*-
"""The tracks worth a human look.

Nobody goes through every track of a survey. Most are plain - a confident
detector, a long track, a classifier in no doubt, both cameras agreeing -
and a few are not. This module finds the few, says why each is in doubt, and
puts the most doubtful first, so that an hour of reviewing goes where it
changes the count.

**Why a track is in doubt** (:func:`reasons`):

* the detector was not sure - the mean confidence of its boxes is low;
* it is very short - a few boxes are what a false alarm looks like;
* nobody knows what it is - no species;
* the classifier was not sure - a species or sex call by a narrow vote;
* the detector's own classes say another species than the track carries;
* the other camera says something else - another species or sex on the
  track it was matched with;
* the other camera did not see it at all, where the cameras were matched;
* it was hidden on most frames.

**What is not in the queue**: tracks somebody approved, tracks that were
labelled by hand, and what a person has already said - a species set by
hand is not asked about again, though the track may still be too short or
too faint to count without a look.

The rows come from the track inventory (:mod:`core.track_inventory`), so
the queue knows what the inventory knows, for every flight of a survey.

Free of QGIS and Qt.
"""

from typing import Dict, List, Optional, Sequence, Tuple

#: Below this mean confidence a track's boxes are doubtful.
MIN_CONFIDENCE = 0.4
#: Fewer boxes than this make a track very short.
MIN_BOXES = 4
#: A classifier call by less than this share of the votes is a narrow one.
MIN_VOTE = 0.7
#: Hidden on more than this share of its frames.
MAX_OCCLUDED = 0.5

DEFAULTS = {"min_confidence": MIN_CONFIDENCE, "min_boxes": MIN_BOXES,
            "min_vote": MIN_VOTE}

#: How much each kind of doubt weighs when the queue is put in order: what
#: changes a count comes before what changes a label.
WEIGHTS = {
    "low_confidence": 3.0,
    "short": 3.0,
    "no_partner": 2.0,
    "cameras_species": 2.5,
    "detector_species": 2.0,
    "no_species": 1.5,
    "species_vote": 1.5,
    "occluded": 1.0,
    "cameras_sex": 1.0,
    "sex_vote": 0.5,
}

_UNDETERMINED = ("", "animal", "unknown")
_OTHER_CAMERA = {"thermal": "RGB", "rgb": "thermal"}


def _known(value) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    return None if text.lower() in _UNDETERMINED else text


def _same(a, b) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def in_queue(row: dict) -> bool:
    """Whether a track can be in the queue at all: not one somebody
    approved, not one that was labelled by hand, and not something a
    reviewer already ruled out."""
    if row.get("approved"):
        return False
    if row.get("label_track_ids"):
        return False
    return not _same(row.get("species") or "", "not-an-animal")


def reasons(row: dict, partner: Optional[dict] = None, matched: bool = False,
            settings: Optional[dict] = None) -> List[Tuple[str, str]]:
    """Why a track is worth a look: ``(kind, sentence)`` pairs, the kinds
    being the keys of :data:`WEIGHTS`. Empty for a plain track.

    :param row: the track's inventory row
    :param partner: the inventory row of the track it was matched with on
        the other camera, if any
    :param matched: whether the cameras were matched on this flight at all -
        only then does a track without a partner mean something
    :param settings: ``min_confidence``, ``min_boxes``, ``min_vote``
        (:data:`DEFAULTS`)
    """
    limits = dict(DEFAULTS)
    limits.update({key: value for key, value in (settings or {}).items()
                   if value is not None})
    hand = {field.strip() for field in
            str(row.get("by_hand") or "").split(";") if field.strip()}
    found: List[Tuple[str, str]] = []

    confidence = row.get("mean_confidence")
    if confidence is not None and confidence < limits["min_confidence"]:
        found.append(("low_confidence",
                      f"boxes of low confidence (mean {confidence:.2f})"))
    boxes = row.get("n_boxes") or 0
    if boxes < limits["min_boxes"]:
        found.append(("short", f"only {boxes} box{'es' if boxes != 1 else ''}"))

    species = _known(row.get("species"))
    if "species" not in hand:
        if species is None:
            found.append(("no_species", "species not determined"))
        vote = row.get("species_vote_fraction")
        if species is not None and vote is not None and \
                vote < limits["min_vote"]:
            found.append(("species_vote",
                          f"{species} by {vote * 100:.0f} % of the votes "
                          "only"))
        detector = _known(row.get("detector_species"))
        if species is not None and detector is not None and \
                not _same(species, detector):
            found.append(("detector_species",
                          f"the detector calls it {detector}"))
    if "sex" not in hand:
        vote = row.get("sex_vote_fraction")
        sex = _known(row.get("sex"))
        if sex is not None and vote is not None and vote < limits["min_vote"]:
            found.append(("sex_vote",
                          f"{sex} by {vote * 100:.0f} % of the votes only"))

    other = _OTHER_CAMERA.get(str(row.get("camera") or "").lower(),
                              "other")
    if partner is not None:
        theirs = _known(partner.get("species"))
        if species is not None and theirs is not None and \
                not _same(species, theirs) and "species" not in hand:
            found.append(("cameras_species",
                          f"the {other} camera says {theirs}"))
        mine_sex, their_sex = _known(row.get("sex")), _known(
            partner.get("sex"))
        if mine_sex is not None and their_sex is not None and \
                not _same(mine_sex, their_sex) and "sex" not in hand:
            found.append(("cameras_sex",
                          f"the {other} camera says {their_sex}"))
    elif matched:
        found.append(("no_partner", f"not seen by the {other} camera"))

    occluded = row.get("n_occluded") or 0
    if boxes and occluded / boxes > MAX_OCCLUDED:
        found.append(("occluded",
                      f"hidden on {occluded} of {boxes} frames"))
    return found


def priority(found: Sequence[Tuple[str, str]]) -> float:
    """How much a track is worth a look: the weights of its doubts."""
    return sum(WEIGHTS.get(kind, 1.0) for kind, _text in found)


def build(rows: Sequence[dict],
          partners: Optional[Dict[Tuple[object, int], dict]] = None,
          matched_flights: Optional[Sequence] = None,
          settings: Optional[dict] = None) -> List[dict]:
    """The queue: the tracks in doubt, the most doubtful first.

    :param rows: inventory rows (with a ``flight`` where several flights
        are listed)
    :param partners: ``(flight, track id) -> inventory row of the partner
        on the other camera``
    :param matched_flights: the flights on which the cameras were matched
    :return: copies of the rows of the tracks in doubt, each with
        ``reasons`` (the sentences), ``reason_kinds`` and ``priority``
    """
    partners = partners or {}
    matched = set(matched_flights or ())
    queue = []
    for row in rows:
        if not in_queue(row):
            continue
        flight = row.get("flight")
        found = reasons(row, partners.get((flight, row.get("track_id"))),
                        flight in matched, settings)
        if not found:
            continue
        entry = dict(row)
        entry["reasons"] = [text for _kind, text in found]
        entry["reason_kinds"] = [kind for kind, _text in found]
        entry["priority"] = priority(found)
        queue.append(entry)
    queue.sort(key=lambda entry: (
        -entry["priority"],
        entry.get("mean_confidence") if entry.get("mean_confidence")
        is not None else 1.0,
        str(entry.get("flight") or ""), entry.get("track_id") or 0))
    return queue


def summarise(queue: Sequence[dict], total: int) -> str:
    """The queue in a sentence, for a log or a header."""
    if not queue:
        return (f"None of the {total} track(s) is in doubt - or they are "
                "all approved.")
    kinds: Dict[str, int] = {}
    for entry in queue:
        for kind in entry["reason_kinds"]:
            kinds[kind] = kinds.get(kind, 0) + 1
    names = {
        "low_confidence": "low confidence", "short": "very short",
        "no_partner": "one camera only",
        "cameras_species": "cameras disagree on the species",
        "detector_species": "detector and track disagree",
        "no_species": "no species", "species_vote": "narrow species vote",
        "occluded": "mostly hidden",
        "cameras_sex": "cameras disagree on the sex",
        "sex_vote": "narrow sex vote"}
    parts = ", ".join(
        f"{count} {names.get(kind, kind)}" for kind, count in
        sorted(kinds.items(), key=lambda item: (-item[1], item[0])))
    return f"{len(queue)} of {total} track(s) are worth a look: {parts}"


def for_flights(flights: Sequence[Tuple[str, str, Optional[Sequence[int]]]],
                modality: str, epsg: Optional[int] = None,
                settings: Optional[dict] = None,
                several: bool = False) -> Tuple[List[dict], int]:
    """The queue of a survey.

    :param flights: ``(name, target folder, species ids or None)`` per
        flight - the species filter of the survey, in the flight's own ids
    :param several: whether the rows carry their flight's name (a survey of
        more than one flight)
    :return: ``(queue, number of tracks looked at)``
    """
    from . import match_store, track_inventory

    other = "w" if modality == "t" else "t"
    rows: List[dict] = []
    partners: Dict[Tuple[object, int], dict] = {}
    matched = []
    for name, folder, species_ids in flights:
        if species_ids is not None and not species_ids:
            continue
        key = name if several else None
        of_flight = track_inventory.build_inventory(
            folder, modality, epsg=epsg, species_ids=species_ids)
        for row in of_flight:
            rows.append(dict(row, flight=key))
        try:
            pairs = match_store.partner_tracks(folder, modality)
        except Exception:  # noqa: BLE001 - no matching is no partner
            pairs = {}
        if not pairs:
            continue
        matched.append(key)
        theirs = {int(row["track_id"]): row
                  for row in track_inventory.build_inventory(
                      folder, other, epsg=epsg,
                      track_ids=sorted(set(pairs.values())))}
        for track_id, partner in pairs.items():
            if partner in theirs:
                partners[(key, int(track_id))] = theirs[partner]
    return build(rows, partners, matched, settings), len(rows)
