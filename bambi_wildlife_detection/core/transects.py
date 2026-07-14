# -*- coding: utf-8 -*-
"""Headless data model & geometry of the transect splitting tool.

A *transect* is a contiguous frame range ``[start_frame, end_frame]`` of one
modality's extracted frames.  Its length is measured along the flight path:
the polyline through the per-frame camera ground positions (the ``location``
x/y of the poses file, which live in the metric UTM CRS of the flight minus
the DEM origin — so distances are metres without any CRS handling).

Files written (relative to *target_folder*, ``{m}`` = ``t`` / ``w``)
--------------------------------------------------------------------
``transects_{m}/transects.json`` — source of truth
``transects_{m}/transects.csv``  — flat export
    format: ``id,name,start_frame,end_frame,start_time,end_time,length_m``

Like every ``core`` module this file must stay importable without QGIS.
"""

import bisect
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: Format version written to ``transects.json``.
TRANSECTS_VERSION = 1


# ---------------------------------------------------------------------------
# Flight path geometry
# ---------------------------------------------------------------------------

def flight_positions(images: List[dict]) -> List[Optional[Tuple[float, float]]]:
    """Ground position (x, y) of every pose image, ``None`` when missing.

    Positions are the first two components of the poses file's ``location``
    (metric, mesh-local = UTM minus DEM origin); the altitude is ignored so
    transect lengths are horizontal ground distances.
    """
    positions: List[Optional[Tuple[float, float]]] = []
    for img in images:
        loc = img.get("location")
        try:
            positions.append((float(loc[0]), float(loc[1])))
        except (TypeError, ValueError, IndexError):
            positions.append(None)
    return positions


def cumulative_distances(
        positions: List[Optional[Tuple[float, float]]]) -> List[float]:
    """Cumulative flight-path length (metres) at every frame.

    Frames without a position contribute no distance and inherit the previous
    cumulative value, so the list always has one entry per frame and is
    non-decreasing.
    """
    cum: List[float] = []
    total = 0.0
    prev: Optional[Tuple[float, float]] = None
    for pos in positions:
        if pos is not None and prev is not None:
            total += math.hypot(pos[0] - prev[0], pos[1] - prev[1])
        if pos is not None:
            prev = pos
        cum.append(total)
    return cum


def path_length(cum: List[float], frame_a: int, frame_b: int) -> float:
    """Flight-path length (metres) between two frames (order-insensitive)."""
    if not cum:
        return 0.0
    a = max(0, min(frame_a, len(cum) - 1))
    b = max(0, min(frame_b, len(cum) - 1))
    return abs(cum[b] - cum[a])


def frame_after_distance(cum: List[float], start_frame: int,
                         meters: float) -> Optional[int]:
    """First frame at least *meters* along the path after *start_frame*.

    Returns ``None`` when the remaining flight path is shorter than *meters*
    (callers typically clamp to the last frame and tell the user).
    """
    if not cum or meters < 0:
        return None
    start = max(0, min(start_frame, len(cum) - 1))
    target = cum[start] + meters
    idx = bisect.bisect_left(cum, target, lo=start)
    if idx >= len(cum):
        return None
    return idx


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Transect:
    """One transect: a named frame range of the flight."""

    transect_id: int
    name: str = ""
    start_frame: int = 0
    end_frame: int = 0

    @property
    def display_name(self) -> str:
        """The user-given name, or ``Transect {id}`` as the counting default."""
        return self.name.strip() or f"Transect {self.transect_id}"

    @property
    def first_frame(self) -> int:
        return min(self.start_frame, self.end_frame)

    @property
    def last_frame(self) -> int:
        return max(self.start_frame, self.end_frame)

    def clamp(self, total_frames: int) -> None:
        """Clamp both frames into ``[0, total_frames - 1]``."""
        hi = max(0, total_frames - 1)
        self.start_frame = max(0, min(self.start_frame, hi))
        self.end_frame = max(0, min(self.end_frame, hi))

    def to_dict(self) -> dict:
        return {
            "id": self.transect_id,
            "name": self.name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transect":
        return cls(
            transect_id=int(data["id"]),
            name=str(data.get("name", "") or ""),
            start_frame=int(data.get("start_frame", 0)),
            end_frame=int(data.get("end_frame", 0)),
        )


class TransectStore:
    """Loads / saves the transect definitions of one modality."""

    def __init__(self, target_folder: str, modality: str):
        self.target_folder = target_folder
        self.modality = modality
        self.transects: Dict[int, Transect] = {}

    @property
    def folder(self) -> str:
        return os.path.join(self.target_folder, f"transects_{self.modality}")

    @property
    def json_path(self) -> str:
        return os.path.join(self.folder, "transects.json")

    @property
    def csv_path(self) -> str:
        return os.path.join(self.folder, "transects.csv")

    def next_id(self) -> int:
        return max(self.transects.keys(), default=0) + 1

    def add(self, start_frame: int, end_frame: int,
            name: str = "") -> Transect:
        transect = Transect(self.next_id(), name, start_frame, end_frame)
        self.transects[transect.transect_id] = transect
        return transect

    def remove(self, transect_id: int) -> bool:
        return self.transects.pop(transect_id, None) is not None

    def ordered(self) -> List[Transect]:
        """Transects sorted by their first frame (ties by id)."""
        return sorted(self.transects.values(),
                      key=lambda t: (t.first_frame, t.transect_id))

    def load(self) -> bool:
        """Read ``transects.json``. Returns True when the file existed."""
        self.transects = {}
        if not os.path.isfile(self.json_path):
            return False
        with open(self.json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for td in data.get("transects", []):
            transect = Transect.from_dict(td)
            self.transects[transect.transect_id] = transect
        return True

    def save(self, images: Optional[List[dict]] = None) -> None:
        """Write ``transects.json`` and the CSV export.

        When *images* (the poses list) is given, each transect is enriched
        with its capture timestamps and flight-path length — informative
        metadata for downstream consumers; only ``id``/``name``/frames are
        read back by :meth:`load`.
        """
        cum = cumulative_distances(flight_positions(images)) if images else []

        def _timestamp(frame: int) -> str:
            if images and 0 <= frame < len(images):
                return str(images[frame].get("timestamp", "") or "")
            return ""

        rows = []
        for t in self.ordered():
            entry = t.to_dict()
            entry["start_time"] = _timestamp(t.first_frame)
            entry["end_time"] = _timestamp(t.last_frame)
            entry["length_m"] = (
                round(path_length(cum, t.first_frame, t.last_frame), 2)
                if cum else None)
            rows.append(entry)

        os.makedirs(self.folder, exist_ok=True)
        data = {
            "version": TRANSECTS_VERSION,
            "modality": self.modality,
            "transects": rows,
        }
        with open(self.json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

        with open(self.csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "name", "start_frame", "end_frame",
                             "start_time", "end_time", "length_m"])
            for entry in rows:
                writer.writerow([
                    entry["id"], entry["name"] or "",
                    entry["start_frame"], entry["end_frame"],
                    entry["start_time"], entry["end_time"],
                    "" if entry["length_m"] is None else entry["length_m"],
                ])
