# -*- coding: utf-8 -*-
"""Groups: animals that were together.

Many of the animals a drone survey counts do not stand alone - a herd of red
deer, a sounder of wild boar, a doe with her fawns. Counting them one by one
is right for a census. It is wrong for everything that takes each sighting as
a separate event: twelve deer in one meadow are one thing the camera came
across, not twelve, and a distance-sampling estimate that treats them as
twelve claims far more precision than it has.

**What a group is.** Two animals of the same species belong together when
they are no further apart than the species' *group distance* - and a chain of
such pairs is one group, so a herd strung out along a forest edge is one herd
although its ends are far apart. The distance is set per species in the
Project Schema, next to its colour (:data:`DEFAULT_DISTANCE_M` until it is);
``0`` means the species is never grouped.

**They must have been seen together.** A flight crosses the same ground more
than once - neighbouring legs overlap - and an animal near the border of two
legs is seen, and counted, on both. Two sightings minutes apart are not a
group of two. So tracked animals are only joined when the drone saw them
within :data:`SAME_PASS_SECONDS` of each other, and boxes (a survey that
counts detections) only within the same frame. Where the capture times of
the frames are not known, distance alone decides.

**A group is the same thing in every tool.** The animals of a flight are
divided into groups once, all of them (:func:`for_flight`): what a tool
leaves out - another species, animals outside the survey area - does not
change which group the others are in. A tool counts the members that count
for it, and a group of which none does is no group there. A group is named
after its member with the lowest id.

An animal that was never geo-referenced has no place and is a group of its
own. Animals of undetermined species ("animal", "unknown") are grouped among
themselves, like any species.

Free of QGIS and Qt.
"""

import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from . import store

META_KEY = "species_group_distances"

#: Group distance of a species until the project sets one, in metres.
DEFAULT_DISTANCE_M = 50.0
#: More than this is not a group distance but a typing error.
MAX_DISTANCE_M = 5000.0
#: Two tracked animals were seen together when the drone saw one within
#: this many seconds of the other. A leg of a survey flight takes longer.
SAME_PASS_SECONDS = 60.0


# ---------------------------------------------------------------------------
# The distance of a species
# ---------------------------------------------------------------------------

def normalise_distance(value) -> float:
    """*value* as a group distance in metres.

    :raises ValueError: when it is not a number from 0 to
        :data:`MAX_DISTANCE_M`
    """
    try:
        metres = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Not a distance: {value!r}.") from None
    if not math.isfinite(metres) or metres < 0 or metres > MAX_DISTANCE_M:
        raise ValueError(
            f"A group distance is between 0 and {MAX_DISTANCE_M:g} m, "
            f"not {value!r}.")
    return metres


def stored_distances(conn) -> Dict[int, float]:
    """``species_id -> metres`` of the distances a project store sets."""
    raw = store.get_meta(conn, META_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    distances: Dict[int, float] = {}
    for key, value in (data.items() if isinstance(data, dict) else ()):
        try:
            distances[int(key)] = normalise_distance(value)
        except (TypeError, ValueError):
            continue
    return distances


def store_distance(conn, species_id: int, metres: Optional[float]) -> None:
    """Set (or, with ``None``, clear) the group distance of a species.

    Written into the caller's connection without committing, so it is part
    of whatever the caller commits or rolls back.
    """
    distances = stored_distances(conn)
    if metres is None:
        distances.pop(int(species_id), None)
    else:
        distances[int(species_id)] = normalise_distance(metres)
    store.set_meta(conn, META_KEY, json.dumps(
        {str(key): value for key, value in sorted(distances.items())}))


def distances_for(species: Iterable[dict],
                  overrides: Dict[int, float]) -> Dict[str, float]:
    """``name -> metres`` for *species* rows (``species_id``, ``name``)."""
    return {row["name"]: overrides.get(int(row["species_id"]),
                                       DEFAULT_DISTANCE_M)
            for row in species}


def species_distances(target_folder: str) -> Dict[str, float]:
    """``species name -> group distance`` of a flight: what its schema sets
    over the default. Empty when the flight has no store yet - every species
    then has the default (:func:`distance_of`)."""
    import os

    path = store.project_path(target_folder) if target_folder else ""
    if not path or not os.path.isfile(path):
        return {}
    conn = store.open_store(path, store.PROJECT)
    try:
        rows = [dict(row) for row in conn.execute(
            "SELECT species_id, name FROM species")]
        return distances_for(rows, stored_distances(conn))
    finally:
        conn.close()


def differ(before: Dict[str, float], after: Dict[str, float]) -> bool:
    """True when a species has another group distance in *after* than in
    *before* (``name -> metres``). A species that only one of them knows
    has changed nothing as long as it is at the default."""
    return any(distance_of(before, name) != distance_of(after, name)
               for name in set(before) | set(after))


def copy_distances(source_folder: str,
                   target_folders: Sequence[str]) -> List[str]:
    """Give other flights the group distances of *source_folder*.

    The schema - and with it the group distance - belongs to a flight, and a
    survey takes each flight's own. A distance that is meant for the whole
    project has to be in every flight: this carries it over **by species
    name**, since ids are assigned per flight. A species the source leaves
    at the default goes back to the default in the target too; a species
    only the target knows keeps what it has.

    :return: the folders that were changed
    :raises sqlite3.OperationalError: when a flight's store is in use
    """
    import os

    wanted = species_distances(source_folder)
    if not wanted:
        return []
    changed = []
    for folder in target_folders:
        path = store.project_path(folder) if folder else ""
        if not path or not os.path.isfile(path):
            continue
        conn = store.open_store(path, store.PROJECT)
        try:
            have = stored_distances(conn)
            want = dict(have)
            for row in conn.execute("SELECT species_id, name FROM species"):
                if row["name"] not in wanted:
                    continue
                species_id = int(row["species_id"])
                if wanted[row["name"]] == DEFAULT_DISTANCE_M:
                    want.pop(species_id, None)
                else:
                    want[species_id] = wanted[row["name"]]
            if want != have:
                with store.transaction(conn):
                    store.set_meta(conn, META_KEY, json.dumps(
                        {str(key): value
                         for key, value in sorted(want.items())}))
                changed.append(folder)
        finally:
            conn.close()
    return changed


def distance_of(distances: Dict[str, float], name: Optional[str]) -> float:
    """The group distance of species *name*, set in *distances* or not."""
    value = distances.get(name or "")
    return DEFAULT_DISTANCE_M if value is None else float(value)


# ---------------------------------------------------------------------------
# Dividing animals into groups
# ---------------------------------------------------------------------------

class _Sets:
    """Disjoint sets of indices (union-find)."""

    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        parent = self._parent
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def join(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self._parent[max(a, b)] = min(a, b)


def _gap(a: dict, b: dict) -> Optional[float]:
    """Seconds between the times two animals were in view - 0 when they
    overlap, ``None`` when a time is not known."""
    times = (a.get("first_time"), a.get("last_time"),
             b.get("first_time"), b.get("last_time"))
    if any(value is None for value in times):
        return None
    a0, a1, b0, b1 = (float(value) for value in times)
    return max(0.0, b0 - a1, a0 - b1)


def partition(animals: Sequence[dict], distances: Dict[str, float],
              same_frame: bool = False,
              max_gap_s: Optional[float] = SAME_PASS_SECONDS) -> List:
    """Which group each animal belongs to.

    *animals* are ``{"id", "species", "x", "y", "first_frame", "last_frame",
    "first_time", "last_time"}`` - positions in metres, times in seconds,
    either ``None`` where not known.

    :param distances: ``species name -> metres`` (:func:`distance_of`)
    :param same_frame: only animals of one frame can be a group - what a
        survey that counts boxes has
    :param max_gap_s: the longest time between two animals being in view for
        them to have been seen together; ``None`` = distance alone decides
    :return: per animal the id of its group: the lowest ``id`` among its
        members. An animal alone is its own group.
    """
    sets = _Sets(len(animals))
    by_species = defaultdict(list)
    for index, animal in enumerate(animals):
        if animal.get("x") is None or animal.get("y") is None:
            continue
        key = animal.get("species") or ""
        if same_frame:
            key = (key, animal.get("first_frame"))
        by_species[key].append(index)

    for key, members in by_species.items():
        name = key[0] if same_frame else key
        reach = distance_of(distances, name)
        if reach <= 0 or len(members) < 2:
            continue
        reach_2 = reach * reach
        # A grid of cells as wide as the distance: whatever is within reach
        # of an animal lies in its own cell or one of the eight around it.
        cells = defaultdict(list)
        for index in members:
            animal = animals[index]
            cells[(math.floor(animal["x"] / reach),
                   math.floor(animal["y"] / reach))].append(index)
        for (cx, cy), inside in cells.items():
            near = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    near.extend(cells.get((cx + dx, cy + dy), ()))
            for index in inside:
                a = animals[index]
                for other in near:
                    if other <= index:
                        continue
                    b = animals[other]
                    if (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 > reach_2:
                        continue
                    if not same_frame and max_gap_s is not None:
                        gap = _gap(a, b)
                        if gap is not None and gap > max_gap_s:
                            continue
                    sets.join(index, other)

    lowest: Dict[int, object] = {}
    for index, animal in enumerate(animals):
        root = sets.find(index)
        if root not in lowest or animal["id"] < lowest[root]:
            lowest[root] = animal["id"]
    return [lowest[sets.find(index)] for index in range(len(animals))]


def describe(animals: Sequence[dict], group_ids: Sequence) -> List[dict]:
    """One row per group, in the order of their ids::

        {"group", "species", "size", "x", "y", "radius_m", "first_frame",
         "last_frame", "first_time", "last_time", "members"}

    ``x`` / ``y`` is the middle of its members (``None`` for an animal that
    has no place), ``radius_m`` how far the furthest of them is from it.
    """
    members = defaultdict(list)
    for animal, group in zip(animals, group_ids):
        members[group].append(animal)

    def lowest(values):
        values = [v for v in values if v is not None]
        return min(values) if values else None

    def highest(values):
        values = [v for v in values if v is not None]
        return max(values) if values else None

    rows = []
    for group in sorted(members):
        of_group = members[group]
        placed = [(a["x"], a["y"]) for a in of_group
                  if a.get("x") is not None and a.get("y") is not None]
        if placed:
            x = sum(p[0] for p in placed) / len(placed)
            y = sum(p[1] for p in placed) / len(placed)
            radius = max(math.hypot(p[0] - x, p[1] - y) for p in placed)
        else:
            x = y = radius = None
        rows.append({
            "group": group,
            "species": of_group[0].get("species") or "",
            "size": len(of_group),
            "x": x, "y": y, "radius_m": radius,
            "first_frame": lowest(a.get("first_frame") for a in of_group),
            "last_frame": highest(a.get("last_frame") for a in of_group),
            "first_time": lowest(a.get("first_time") for a in of_group),
            "last_time": highest(a.get("last_time") for a in of_group),
            "members": sorted(a["id"] for a in of_group),
        })
    return rows


def size_stats(sizes: Iterable[int]) -> dict:
    """What a number of groups amounts to::

        {"groups", "animals", "mean_size", "se_size", "largest", "alone"}

    ``se_size`` is the standard error of the mean size (``None`` for fewer
    than two groups), ``alone`` the number of animals that were by
    themselves. All zero / ``None`` for no group at all.
    """
    sizes = [int(size) for size in sizes if size and size > 0]
    n = len(sizes)
    if not n:
        return {"groups": 0, "animals": 0, "mean_size": None,
                "se_size": None, "largest": 0, "alone": 0}
    mean = sum(sizes) / n
    se = None
    if n > 1:
        variance = sum((size - mean) ** 2 for size in sizes) / (n - 1)
        se = math.sqrt(variance / n)
    return {"groups": n, "animals": sum(sizes), "mean_size": mean,
            "se_size": se, "largest": max(sizes),
            "alone": sum(1 for size in sizes if size == 1)}


# ---------------------------------------------------------------------------
# The groups of a flight
# ---------------------------------------------------------------------------

class FlightGroups:
    """The animals of one flight, divided into groups.

    :ivar rows: one row per group (:func:`describe`)
    :ivar distances: ``species name -> metres`` that was used
    :ivar timed: False when the capture time of a frame is not known: for
        the animals seen in it, distance alone decided
    :ivar source: ``"tracks"`` or ``"detections"`` - what an animal is here
    """

    def __init__(self, animals: Sequence[dict], distances: Dict[str, float],
                 source: str, timed: bool):
        same_frame = source == "detections"
        self.source = source
        self.distances = dict(distances)
        self.timed = bool(timed)
        ids = partition(animals, distances, same_frame=same_frame)
        self._group = {animal["id"]: group
                       for animal, group in zip(animals, ids)}
        self.rows = describe(animals, ids)
        self._size = {row["group"]: row["size"] for row in self.rows}

    def of(self, animal_id):
        """The group of the animal with this id (a track id, or a detection
        id where boxes are counted); ``None`` for one that is not known."""
        return self._group.get(animal_id)

    def size_of(self, group) -> int:
        """How many animals the whole group has."""
        return self._size.get(group, 0)

    def __len__(self) -> int:
        return len(self.rows)

    def describe_rule(self) -> str:
        """The rule in a sentence, for a log or a result's notes."""
        if self.source == "detections":
            together = "in the same frame"
        elif self.timed:
            together = (f"seen within {SAME_PASS_SECONDS:g} s of each "
                        "other")
        else:
            together = (f"seen within {SAME_PASS_SECONDS:g} s of each other "
                        "- or whenever they were seen, where the capture "
                        "time of a frame is not known")
        return ("Animals of one species no further apart than the species' "
                f"group distance are one group, {together}.")


def frame_seconds(target_folder: str, modality: str) -> Dict[int, float]:
    """``frame -> capture time`` in seconds, for the frames that have one."""
    from .exporters import common

    seconds = {}
    for frame in common.load_frames(target_folder, modality):
        moment = common.capture_time(frame.get("epoch"))
        if moment is not None:
            seconds[int(frame["frame"])] = moment.timestamp()
    return seconds


def for_flight(target_folder: str, modality: str, source: str = "tracks",
               include_manual: bool = True) -> FlightGroups:
    """Every counted animal of a flight - all species, wherever it is - in
    its group.

    The animals are the survey's (:func:`core.analytics_source.load_animals`):
    a tracked animal at the middle of its boxes, or every box.

    :raises core.analytics_source.AnalyticsError: when the flight has no
        detections stored
    """
    from . import analytics_source

    animals, _facts = analytics_source.load_animals(
        target_folder, modality, source, None, include_manual)
    key = "track_id" if source == analytics_source.TRACKS else "detection_id"
    seconds = frame_seconds(target_folder, modality)
    records = []
    for animal in animals:
        records.append({
            "id": animal[key],
            "species": animal["species"],
            "x": animal["x"], "y": animal["y"],
            "first_frame": animal["first_frame"],
            "last_frame": animal["last_frame"],
            "first_time": seconds.get(animal["first_frame"]),
            "last_time": seconds.get(animal["last_frame"]),
        })
    timed = bool(records) and all(
        record["first_time"] is not None and record["last_time"] is not None
        for record in records)
    return FlightGroups(records, species_distances(target_folder), source,
                        timed)
