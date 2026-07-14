# -*- coding: utf-8 -*-
"""Headless data model and geometry logic of the labelling tool.

Moved verbatim from ``bambi_labelling_tool.py`` (which re-exports every
name here for backwards compatibility); see that module's docstring for
the labelling workflow. Contains:

* the class taxonomies and track colour palette (as RGB tuples — the GUI
  wraps them in ``QColor``),
* :class:`CustomField` — user-defined extra attributes (per track or per
  key frame) that only round-trip through ``labels.json``,
* :class:`LabelTrack` / :class:`LabelStore` — key-frame storage,
  interpolation, stop frames, JSON/CSV persistence and the
  ``detections.txt`` export,
* read-only loaders for pipeline outputs (detections, pixel tracks),
* :class:`_FrameMatcher` — cross-modality frame matching by capture time,
* :class:`_GeoPropagator` — DEM-based box propagation between frames and
  modalities (lazy alfspy/trimesh/bambi imports).
"""

import math
import os
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Class taxonomies
# ---------------------------------------------------------------------------

SPECIES_CLASSES = [
    "unknown", "roe deer", "red deer", "fallow deer", "wild boar",
    "chamois", "fox", "hare", "bird", "other",
]
SEX_CLASSES = ["unknown", "female", "male"]
AGE_CLASSES = ["unknown", "adult", "juvenile"]
OCCLUSION_LEVELS = ["none", "partially", "fully"]

# Distinct colours cycled per label track id (RGB tuples; the GUI layer
# converts them to QColor).
TRACK_COLORS_RGB = [
    (0, 220, 0), (255, 160, 0), (255, 80, 200),
    (0, 200, 255), (255, 230, 0), (170, 110, 255),
    (255, 90, 90), (120, 255, 170),
]


def track_color_rgb(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS_RGB[track_id % len(TRACK_COLORS_RGB)]


# ---------------------------------------------------------------------------
# User-defined attributes
# ---------------------------------------------------------------------------

#: Value types a custom field can hold.  ``datetime`` values are kept as
#: ISO-8601 strings so they survive a JSON round trip unchanged.
FIELD_TYPES = ("int", "float", "string", "bool", "datetime")

#: ``track``  — one value per label track (like species / sex / age)
#: ``keyframe`` — one value per key frame (like occlusion), inherited by the
#: interpolated frames that follow it.
FIELD_SCOPES = ("track", "keyframe")

#: Keys the built-in model already occupies in ``labels.json``; a custom
#: field may not shadow them.
RESERVED_FIELD_NAMES = frozenset({
    "track_id", "species", "sex", "age", "keyframes", "attributes",
    "frame", "x1", "y1", "x2", "y2", "occlusion", "stop",
})

_FIELD_DEFAULTS = {
    "int": 0, "float": 0.0, "string": "", "bool": False, "datetime": "",
}

_TRUE_STRINGS = {"true", "yes", "1", "on"}
_FALSE_STRINGS = {"false", "no", "0", "off", ""}


class CustomField:
    """A user-defined attribute: a *name*, a value *type* and a *scope*.

    Fields are configured in the labelling tool's settings dialog, stored in
    ``labels.json`` alongside the tracks and rendered as extra input widgets.
    They are deliberately **not** written to ``labels.csv`` or
    ``detections.txt``, whose column layouts are consumed by the rest of the
    pipeline and must stay stable.
    """

    def __init__(self, name: str, type: str = "string", scope: str = "track"):
        name = (name or "").strip()
        if not name:
            raise ValueError("A custom field needs a name.")
        if name.lower() in RESERVED_FIELD_NAMES:
            raise ValueError(f"'{name}' is a reserved field name.")
        if type not in FIELD_TYPES:
            raise ValueError(f"Unknown field type '{type}'.")
        if scope not in FIELD_SCOPES:
            raise ValueError(f"Unknown field scope '{scope}'.")
        self.name = name
        self.type = type
        self.scope = scope

    def __eq__(self, other) -> bool:
        if not isinstance(other, CustomField):
            return NotImplemented
        return (self.name, self.type, self.scope) == \
            (other.name, other.type, other.scope)

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"CustomField({self.name!r}, {self.type!r}, {self.scope!r})"

    @property
    def default(self) -> Any:
        return _FIELD_DEFAULTS[self.type]

    def coerce(self, value: Any) -> Any:
        """Return *value* as this field's type.

        Raises ``ValueError`` when the value cannot be converted — callers
        (schema changes, JSON loading) drop such values instead of failing.
        """
        if value is None:
            raise ValueError("None is not a valid field value.")
        if self.type == "int":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, str):
                return int(float(value.strip()))
            return int(value)
        if self.type == "float":
            if isinstance(value, str):
                return float(value.strip())
            return float(value)
        if self.type == "string":
            return value if isinstance(value, str) else str(value)
        if self.type == "bool":
            if isinstance(value, str):
                text = value.strip().lower()
                if text in _TRUE_STRINGS:
                    return True
                if text in _FALSE_STRINGS:
                    return False
                raise ValueError(f"'{value}' is not a boolean.")
            return bool(value)
        # datetime — kept as an ISO-8601 string ('' means unset)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        text = str(value).strip()
        if not text:
            return ""
        from datetime import datetime
        datetime.fromisoformat(text)  # validate, keep the original spelling
        return text

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.type, "scope": self.scope}

    @classmethod
    def from_dict(cls, d: dict) -> "CustomField":
        return cls(d.get("name", ""), d.get("type", "string"),
                   d.get("scope", "track"))


def custom_fields_from_dicts(dicts: Iterable[dict]) -> List["CustomField"]:
    """Build a field list from JSON/QSettings dicts, skipping invalid ones."""
    fields: List[CustomField] = []
    seen = set()
    for d in dicts or []:
        try:
            field = CustomField.from_dict(d)
        except (ValueError, AttributeError):
            continue
        if field.name.lower() in seen:
            continue
        seen.add(field.name.lower())
        fields.append(field)
    return fields


def validate_custom_fields(fields: List["CustomField"]) -> None:
    """Raise ``ValueError`` when two fields share a (case-insensitive) name."""
    seen = set()
    for field in fields:
        key = field.name.lower()
        if key in seen:
            raise ValueError(f"Duplicate field name '{field.name}'.")
        seen.add(key)


# Marker + version of the standalone schema file written by
# :func:`write_custom_fields`, so a wrong file picked in the import dialog is
# rejected with a clear message instead of silently yielding no fields.
FIELD_SCHEMA_FORMAT = "bambi-labelling-fields"
FIELD_SCHEMA_VERSION = 1


def write_custom_fields(path: str, fields: List["CustomField"]) -> None:
    """Write *fields* to a shareable schema file (``*.json``)."""
    data = {
        "format": FIELD_SCHEMA_FORMAT,
        "version": FIELD_SCHEMA_VERSION,
        "custom_fields": [f.to_dict() for f in fields],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def read_custom_fields(path: str) -> List["CustomField"]:
    """Read a schema written by :func:`write_custom_fields`.

    A flight's ``labels.json`` and a bare list of field dicts are accepted
    too, so a colleague's label folder can be used as the source directly.
    Unlike :func:`custom_fields_from_dicts` this is strict: an unusable file
    or a single bad field raises ``ValueError`` with a readable message
    rather than silently importing a partial schema.
    """
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not a valid JSON file: {exc}")

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        fmt = data.get("format")
        if fmt not in (None, FIELD_SCHEMA_FORMAT):
            raise ValueError(f"Unknown file format '{fmt}'.")
        raw_version = data.get("version", FIELD_SCHEMA_VERSION)
        try:
            version = int(raw_version)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid file version '{raw_version}'.")
        if version > FIELD_SCHEMA_VERSION:
            raise ValueError(
                f"The file was written by a newer version of the plugin "
                f"(version {version}, this one reads up to "
                f"{FIELD_SCHEMA_VERSION}).")
        if "custom_fields" not in data:
            raise ValueError(
                "The file contains no 'custom_fields' — expected a schema "
                "export or a flight's labels.json.")
        entries = data["custom_fields"]
    else:
        raise ValueError("The file contains no custom field definitions.")

    if not isinstance(entries, list):
        raise ValueError("'custom_fields' must be a list of field definitions.")

    fields: List[CustomField] = []
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Field {i} is not a field definition.")
        try:
            fields.append(CustomField.from_dict(entry))
        except ValueError as exc:
            raise ValueError(f"Field {i}: {exc}")
    validate_custom_fields(fields)
    return fields


def coerce_attributes(attributes: Dict[str, Any],
                      fields: List["CustomField"]) -> Dict[str, Any]:
    """Keep only values that belong to *fields* and convert them to its type.

    Values of removed fields are dropped and values that no longer fit a
    retyped field are discarded, so a schema change never leaves unreadable
    entries behind in ``labels.json``.
    """
    result: Dict[str, Any] = {}
    for field in fields:
        if field.name not in attributes:
            continue
        try:
            result[field.name] = field.coerce(attributes[field.name])
        except (ValueError, TypeError):
            continue
    return result


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class LabelTrack:
    """A single annotated track made of key frames.

    Key frames map ``frame -> {"x1","y1","x2","y2","occlusion"[,"stop"]
    [,"attributes"]}``; boxes on frames between two key frames are linearly
    interpolated — except after a key frame flagged ``stop`` (the animal
    disappeared): frames between a stop frame and the next key frame have no
    box.

    ``attributes`` holds the values of the user-defined :class:`CustomField`
    of the matching scope: track-scope values live on ``self.attributes``,
    key-frame-scope values inside each key frame entry (and are inherited by
    the interpolated frames that follow it, like ``occlusion``).
    """

    def __init__(self, track_id: int, species: str = "unknown",
                 sex: str = "unknown", age: str = "unknown",
                 attributes: Optional[Dict[str, Any]] = None):
        self.track_id = track_id
        self.species = species
        self.sex = sex
        self.age = age
        self.attributes: Dict[str, Any] = dict(attributes or {})
        self.keyframes: Dict[int, dict] = {}

    # -- key frame access ------------------------------------------------

    def frames(self) -> List[int]:
        return sorted(self.keyframes.keys())

    def frame_range(self) -> Optional[Tuple[int, int]]:
        if not self.keyframes:
            return None
        fs = self.frames()
        return fs[0], fs[-1]

    def set_keyframe(self, frame: int, box: Tuple[float, float, float, float],
                     occlusion: Optional[str] = None,
                     stop: Optional[bool] = None,
                     attributes: Optional[Dict[str, Any]] = None) -> None:
        prev = self.keyframes.get(frame)
        source = prev if prev is not None else self._effective_entry(frame)
        if occlusion is None:
            occlusion = source.get("occlusion", "none") if source else "none"
        if attributes is None:
            # inherit the key-frame attributes in force at this frame
            attributes = dict(source.get("attributes", {})) if source else {}
        if stop is None:
            # preserve an existing stop flag when only the box is updated
            stop = bool(prev.get("stop", False)) if prev else False
        entry = {
            "x1": float(box[0]), "y1": float(box[1]),
            "x2": float(box[2]), "y2": float(box[3]),
            "occlusion": occlusion,
        }
        if stop:
            entry["stop"] = True
        if attributes:
            entry["attributes"] = dict(attributes)
        self.keyframes[frame] = entry

    def is_stop(self, frame: int) -> bool:
        return bool(self.keyframes.get(frame, {}).get("stop", False))

    def visible_segments(self) -> List[Tuple[int, int]]:
        """Return the frame ranges in which the track has boxes.

        Without stop frames this is one ``[(first, last)]`` segment; every
        stop frame ends its segment and the next key frame starts a new one.
        """
        fs = self.frames()
        if not fs:
            return []
        segments: List[Tuple[int, int]] = []
        start = fs[0]
        for i, f in enumerate(fs[:-1]):
            if self.keyframes[f].get("stop"):
                segments.append((start, f))
                start = fs[i + 1]
        segments.append((start, fs[-1]))
        return segments

    def remove_keyframe(self, frame: int) -> bool:
        return self.keyframes.pop(frame, None) is not None

    def _effective_entry(self, frame: int) -> Optional[dict]:
        """The key-frame entry whose per-key-frame state applies at *frame*.

        That is the key frame itself, or the preceding one for interpolated
        frames.  ``None`` when the track has no box at *frame* (outside its
        range or in a gap after a stop frame).
        """
        if not self.keyframes:
            return None
        if frame in self.keyframes:
            return self.keyframes[frame]
        fs = self.frames()
        if frame < fs[0] or frame > fs[-1]:
            return None
        prev = self.keyframes[max(f for f in fs if f < frame)]
        return None if prev.get("stop") else prev

    def attributes_at(self, frame: int) -> Optional[Dict[str, Any]]:
        """Key-frame-scope custom attributes in force at *frame*.

        Interpolated frames inherit the previous key frame's values; returns
        ``None`` where the track has no box.
        """
        entry = self._effective_entry(frame)
        if entry is None:
            return None
        return dict(entry.get("attributes", {}))

    def box_at(self, frame: int) -> Optional[Tuple[tuple, bool, str]]:
        """Return ``((x1,y1,x2,y2), is_keyframe, occlusion)`` or ``None``.

        Frames outside the [first, last] key-frame range return ``None``.
        """
        if not self.keyframes:
            return None
        if frame in self.keyframes:
            kf = self.keyframes[frame]
            return ((kf["x1"], kf["y1"], kf["x2"], kf["y2"]), True,
                    kf.get("occlusion", "none"))

        fs = self.frames()
        if frame < fs[0] or frame > fs[-1]:
            return None

        f_prev = max(f for f in fs if f < frame)
        f_next = min(f for f in fs if f > frame)
        a = self.keyframes[f_prev]
        if a.get("stop"):
            return None  # gap: the animal disappeared after f_prev
        b = self.keyframes[f_next]
        alpha = (frame - f_prev) / (f_next - f_prev)
        box = tuple(
            a[k] + alpha * (b[k] - a[k]) for k in ("x1", "y1", "x2", "y2")
        )
        return box, False, a.get("occlusion", "none")

    def transform_keyframes(
            self, old_box: Tuple[float, float, float, float],
            new_box: Tuple[float, float, float, float],
            bounds: Optional[Tuple[float, float]] = None) -> None:
        """Apply the move/resize taking *old_box* to *new_box* to all key frames.

        The change is decomposed into a translation of the box centre and a
        per-axis scaling of the box size, and every key frame's box is shifted
        and scaled (about its own centre) by those same amounts.  Because
        linear interpolation commutes with this transform, interpolated boxes
        follow along consistently.

        With *bounds* ``(width, height)`` each transformed box is clamped to
        the image area; a box that would leave the image entirely keeps its
        transformed geometry instead (mirroring the single-box clamping).
        """
        old_w = old_box[2] - old_box[0]
        old_h = old_box[3] - old_box[1]
        sx = (new_box[2] - new_box[0]) / old_w if old_w > 1e-9 else 1.0
        sy = (new_box[3] - new_box[1]) / old_h if old_h > 1e-9 else 1.0
        dx = (new_box[0] + new_box[2] - old_box[0] - old_box[2]) / 2.0
        dy = (new_box[1] + new_box[3] - old_box[1] - old_box[3]) / 2.0
        for kf in self.keyframes.values():
            cx = (kf["x1"] + kf["x2"]) / 2.0 + dx
            cy = (kf["y1"] + kf["y2"]) / 2.0 + dy
            hw = (kf["x2"] - kf["x1"]) / 2.0 * sx
            hh = (kf["y2"] - kf["y1"]) / 2.0 * sy
            box = [cx - hw, cy - hh, cx + hw, cy + hh]
            if bounds is not None:
                x1 = max(0.0, box[0])
                y1 = max(0.0, box[1])
                x2 = min(float(bounds[0]), box[2])
                y2 = min(float(bounds[1]), box[3])
                if x2 > x1 and y2 > y1:
                    box = [x1, y1, x2, y2]
            kf["x1"], kf["y1"], kf["x2"], kf["y2"] = box

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "track_id": self.track_id,
            "species": self.species,
            "sex": self.sex,
            "age": self.age,
            "keyframes": {str(f): kf for f, kf in self.keyframes.items()},
        }
        if self.attributes:
            d["attributes"] = dict(self.attributes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LabelTrack":
        track = cls(
            int(d.get("track_id", 0)),
            d.get("species", "unknown"),
            d.get("sex", "unknown"),
            d.get("age", "unknown"),
            attributes=d.get("attributes") or {},
        )
        for f, kf in d.get("keyframes", {}).items():
            entry = {
                "x1": float(kf["x1"]), "y1": float(kf["y1"]),
                "x2": float(kf["x2"]), "y2": float(kf["y2"]),
                "occlusion": kf.get("occlusion", "none"),
            }
            if kf.get("stop"):
                entry["stop"] = True
            if kf.get("attributes"):
                entry["attributes"] = dict(kf["attributes"])
            track.keyframes[int(f)] = entry
        return track


class LabelStore:
    """Loads / saves the label tracks of one modality."""

    def __init__(self, target_folder: str, modality: str):
        self.target_folder = target_folder
        self.modality = modality
        self.tracks: Dict[int, LabelTrack] = {}
        self.custom_fields: List[CustomField] = []
        #: True once a ``custom_fields`` key was seen in ``labels.json`` —
        #: lets the GUI tell "no schema configured yet" (seed it from the
        #: user's defaults) from "schema deliberately empty".
        self.schema_defined = False

    @property
    def folder(self) -> str:
        return os.path.join(self.target_folder, f"labels_{self.modality}")

    @property
    def json_path(self) -> str:
        return os.path.join(self.folder, "labels.json")

    @property
    def csv_path(self) -> str:
        return os.path.join(self.folder, "labels.csv")

    def next_track_id(self) -> int:
        return max(self.tracks.keys(), default=0) + 1

    def load(self) -> None:
        self.tracks = {}
        self.custom_fields = []
        self.schema_defined = False
        if not os.path.isfile(self.json_path):
            return
        with open(self.json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "custom_fields" in data:
            self.schema_defined = True
            self.custom_fields = custom_fields_from_dicts(
                data.get("custom_fields") or [])
        for td in data.get("tracks", []):
            track = LabelTrack.from_dict(td)
            self.tracks[track.track_id] = track
        self._prune_to_schema()

    def save(self) -> None:
        os.makedirs(self.folder, exist_ok=True)
        data = {
            "modality": self.modality,
            "custom_fields": [f.to_dict() for f in self.custom_fields],
            "tracks": [t.to_dict() for t in sorted(
                self.tracks.values(), key=lambda t: t.track_id)],
        }
        with open(self.json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self.schema_defined = True
        self._export_csv()

    # -- custom field schema ---------------------------------------------

    def fields_for(self, scope: str) -> List[CustomField]:
        return [f for f in self.custom_fields if f.scope == scope]

    def set_custom_fields(self, fields: List[CustomField]) -> None:
        """Install a new schema and bring the stored values in line with it.

        Values of dropped fields are removed and values of retyped fields are
        converted (or dropped when they do not fit the new type), so
        ``labels.json`` never carries attributes the schema cannot describe.
        """
        validate_custom_fields(fields)
        self.custom_fields = list(fields)
        self.schema_defined = True
        self._prune_to_schema()

    def count_values(self, field: CustomField) -> int:
        """How many stored values *field* currently has (tracks or key frames).

        The settings dialog uses this to warn before a removal or a retype
        discards data.
        """
        count = 0
        for track in self.tracks.values():
            if field.scope == "track":
                if field.name in track.attributes:
                    count += 1
            else:
                for entry in track.keyframes.values():
                    if field.name in entry.get("attributes", {}):
                        count += 1
        return count

    def _prune_to_schema(self) -> None:
        track_fields = self.fields_for("track")
        kf_fields = self.fields_for("keyframe")
        for track in self.tracks.values():
            track.attributes = coerce_attributes(track.attributes, track_fields)
            for entry in track.keyframes.values():
                attrs = coerce_attributes(
                    entry.get("attributes", {}), kf_fields)
                if attrs:
                    entry["attributes"] = attrs
                else:
                    entry.pop("attributes", None)

    # Marker line that delimits the label-tool block in detections.txt so a
    # re-export replaces the previous block instead of duplicating it.
    DETECTIONS_MARKER = "# --- labelled detections (BAMBI labelling tool) ---"

    def species_class_ids(self) -> Dict[str, int]:
        """Return a ``species -> class_id`` mapping for the pipeline export.

        The pipeline only knows integer class ids, so species from the
        default taxonomy keep their list index and custom species are
        appended after it in alphabetical order.
        """
        mapping = {s: i for i, s in enumerate(SPECIES_CLASSES)}
        custom = sorted({
            (t.species or "unknown").strip().lower()
            for t in self.tracks.values()
        } - set(SPECIES_CLASSES))
        for i, s in enumerate(custom):
            mapping[s] = len(SPECIES_CLASSES) + i
        return mapping

    def export_to_detections(self) -> Tuple[str, int]:
        """Merge the interpolated label boxes into ``detections.txt``.

        Uses the exact line format of the "Detect Animals" stage
        (``frame x1 y1 x2 y2 confidence class_id``, confidence 1.0 for
        manual labels) so downstream steps (geo-referencing, tracking)
        consume the labels like regular detections.  Detector output is
        preserved; a previous label export block is replaced.

        Returns ``(detections_path, number_of_exported_boxes)``.
        """
        det_folder = os.path.join(
            self.target_folder, f"detections_{self.modality}")
        os.makedirs(det_folder, exist_ok=True)
        det_file = os.path.join(det_folder, "detections.txt")

        # Keep everything up to (excluding) a previous export block.
        existing: List[str] = []
        if os.path.isfile(det_file):
            with open(det_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.rstrip("\n") == self.DETECTIONS_MARKER:
                        break
                    existing.append(line)
        if not existing:
            existing = ["# frame x1 y1 x2 y2 confidence class_id\n"]
        if existing and not existing[-1].endswith("\n"):
            existing[-1] += "\n"

        mapping = self.species_class_ids()
        rows: List[tuple] = []  # (frame, track_id, x1, y1, x2, y2, class_id)
        for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
            rng = track.frame_range()
            if rng is None:
                continue
            species = (track.species or "unknown").strip().lower()
            class_id = mapping.get(species, 0)
            for frame in range(rng[0], rng[1] + 1):
                res = track.box_at(frame)
                if res is None:
                    continue
                (x1, y1, x2, y2), _is_kf, _occ = res
                rows.append((frame, track.track_id, x1, y1, x2, y2, class_id))
        rows.sort(key=lambda r: (r[0], r[1]))

        used_ids = sorted({r[6] for r in rows})
        id_to_species = {i: s for s, i in mapping.items()}

        with open(det_file, "w", encoding="utf-8") as fh:
            fh.writelines(existing)
            fh.write(self.DETECTIONS_MARKER + "\n")
            if used_ids:
                pairs = ", ".join(f"{i}={id_to_species[i]}" for i in used_ids)
                fh.write(f"# class_id mapping: {pairs}\n")
            for frame, _tid, x1, y1, x2, y2, class_id in rows:
                fh.write(f"{frame} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                         f"1.0000 {class_id}\n")
        return det_file, len(rows)

    def _export_csv(self) -> None:
        """Write the per-frame interpolated CSV export.

        The column layout is fixed: user-defined :class:`CustomField` values
        stay in ``labels.json`` only, because downstream consumers of this
        file (and of ``detections.txt``) parse it positionally.
        """
        with open(self.csv_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# frame,track_id,x1,y1,x2,y2,species,sex,age,occlusion,keyframe\n"
            )
            for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
                rng = track.frame_range()
                if rng is None:
                    continue
                for frame in range(rng[0], rng[1] + 1):
                    res = track.box_at(frame)
                    if res is None:
                        continue
                    (x1, y1, x2, y2), is_kf, occ = res
                    fh.write(
                        f"{frame},{track.track_id},"
                        f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},"
                        f"{track.species},{track.sex},{track.age},"
                        f"{occ},{1 if is_kf else 0}\n"
                    )


# ---------------------------------------------------------------------------
# Overlay data loaders (pipeline outputs, read-only)
# ---------------------------------------------------------------------------

def _load_detections_by_frame(det_file: str) -> Dict[int, List[tuple]]:
    """Parse ``detections_{m}/detections.txt`` into ``frame -> [boxes]``.

    Format (space-separated, ``#`` comments): ``frame x1 y1 x2 y2 conf cls``
    """
    result: Dict[int, List[tuple]] = {}
    if not os.path.isfile(det_file):
        return result
    try:
        with open(det_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    frame = int(parts[0])
                    result.setdefault(frame, []).append((
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]),
                        float(parts[5]),
                        int(parts[6]) if len(parts) > 6 else 0,
                    ))
    except Exception:  # nosec B110
        pass
    return result


def _load_pixel_tracks(tracks_file: str) -> Dict[int, List[dict]]:
    """Parse ``tracks_{m}/tracks_pixel.csv`` into ``track_id -> [entries]``.

    Format (comma-separated, ``#`` comments):
    ``frame,track_id,x1,y1,x2,y2,conf,cls[,interpolated]``
    """
    result: Dict[int, List[dict]] = {}
    if not os.path.isfile(tracks_file):
        return result
    try:
        with open(tracks_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 8:
                    tid = int(parts[1])
                    result.setdefault(tid, []).append({
                        "frame": int(parts[0]),
                        "x1": float(parts[2]), "y1": float(parts[3]),
                        "x2": float(parts[4]), "y2": float(parts[5]),
                        "conf": float(parts[6]), "cls": int(parts[7]),
                    })
    except Exception:  # nosec B110
        pass
    for entries in result.values():
        entries.sort(key=lambda d: d["frame"])
    return result


# ---------------------------------------------------------------------------
# Cross-modality frame correspondence (by capture timestamp)
# ---------------------------------------------------------------------------

def _pose_epochs(images: List[dict]) -> List[Optional[float]]:
    """Return the capture time (epoch seconds) of every pose image.

    Poses store an ISO-8601 ``timestamp`` (with timezone) taken from the SRT
    capture time, so thermal and RGB frames share the same real-world clock
    and can be matched across modalities.  Entries without a parseable
    timestamp yield ``None``.
    """
    from datetime import datetime

    epochs: List[Optional[float]] = []
    for img in images:
        ts = img.get("timestamp", "")
        if not ts:
            epochs.append(None)
            continue
        try:
            epochs.append(datetime.fromisoformat(ts).timestamp())
        except Exception:
            epochs.append(None)
    return epochs


class _FrameMatcher:
    """Maps a source-modality frame index to the nearest target-modality one.

    Correspondence is by capture time: for a source frame's timestamp the
    target frame with the closest timestamp is returned, together with the
    absolute time difference so callers can reject poor matches.
    """

    def __init__(self, src_images: List[dict], dst_images: List[dict]):
        import bisect

        self._src_epochs = _pose_epochs(src_images)
        dst_epochs = _pose_epochs(dst_images)
        # Sorted (epoch, dst_index) of target frames that have a timestamp.
        self._pairs = sorted(
            (e, i) for i, e in enumerate(dst_epochs) if e is not None)
        self._keys = [e for e, _ in self._pairs]
        self._bisect = bisect.bisect_left

    @property
    def usable(self) -> bool:
        """True when both sides carry enough timestamps to match by time."""
        return bool(self._pairs) and any(e is not None for e in self._src_epochs)

    def match(self, src_frame: int) -> Optional[Tuple[int, float]]:
        """Return ``(dst_frame, dt_seconds)`` nearest in time, or ``None``."""
        if not (0 <= src_frame < len(self._src_epochs)):
            return None
        target = self._src_epochs[src_frame]
        if target is None or not self._keys:
            return None
        pos = self._bisect(self._keys, target)
        best = None
        for j in (pos - 1, pos):
            if 0 <= j < len(self._pairs):
                epoch, idx = self._pairs[j]
                dt = abs(epoch - target)
                if best is None or dt < best[1]:
                    best = (idx, dt)
        return best


# ---------------------------------------------------------------------------
# Geo-referenced box propagation
# ---------------------------------------------------------------------------

#: Pixel-space bounding box ``(x1, y1, x2, y2)``.
Box = Tuple[float, float, float, float]


def propagation_frames(src_frame: int, dst_frame: int,
                       step: int = 0) -> List[int]:
    """Frames that get a key frame when propagating *src_frame* -> *dst_frame*.

    With ``step <= 0`` only the target frame is returned.  With a positive
    *step* additional intermediate frames are sampled every *step* frames
    between source and target, so the linear interpolation between the key
    frames never has to bridge more than *step* frames of drone ego-motion.
    The target frame is always the last entry; the source frame is never
    included.
    """
    if dst_frame == src_frame:
        return []
    sign = 1 if dst_frame > src_frame else -1
    frames: List[int] = []
    if step > 0:
        frame = src_frame + sign * step
        while (dst_frame - frame) * sign > 0:
            frames.append(frame)
            frame += sign * step
    frames.append(dst_frame)
    return frames


def keyframe_window(frames: List[int], current: int,
                    max_shown: int = 12) -> List[Optional[int]]:
    """The key frames to list for a track, elided around *current*.

    Long tracks have more key frames than fit the side panel, so only a window
    of them is listed.  The first and last are always kept — they bound the
    track, which is what the user needs to navigate it — and the remaining
    slots follow *current*.  ``None`` marks a gap where key frames were left
    out, to be rendered as an ellipsis.
    """
    if len(frames) <= max_shown:
        return list(frames)

    inner = max_shown - 2  # first and last take a slot each
    nearest = min(range(len(frames)),
                  key=lambda i: abs(frames[i] - current))
    start = max(1, min(nearest - inner // 2, len(frames) - 1 - inner))

    window: List[Optional[int]] = [frames[0]]
    if start > 1:
        window.append(None)
    window.extend(frames[start:start + inner])
    if start + inner < len(frames) - 1:
        window.append(None)
    window.append(frames[-1])
    return window


def load_valid_mask(target_folder: str, modality: str):
    """Load the undistortion mask for *modality* as a grayscale array.

    Frame extraction moves the undistortion mask into the flight's target
    folder as ``mask_T.png`` (thermal) / ``mask_W.png`` (RGB); its white
    pixels mark the valid image area that survived the undistortion.
    Returns ``None`` when no mask file exists (older flights) or it cannot
    be read — the caller then treats the whole frame as valid.
    """
    letter = "T" if modality.lower().startswith("t") else "W"
    for name in (f"mask_{letter}.png", f"mask_{letter.lower()}.png"):
        path = os.path.join(target_folder, name)
        if os.path.isfile(path):
            try:
                import cv2
            except ImportError:
                return None
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None


def box_in_valid_area(box: Box, mask, img_width: int,
                      img_height: int) -> bool:
    """True when *box* lies entirely in the valid (white) area of *mask*.

    *box* is in pixel coordinates of an ``img_width`` x ``img_height``
    frame; the mask may have a different resolution and is sampled with
    scaled coordinates. Without a mask (``None``) every box is valid.
    """
    if mask is None:
        return True
    import numpy as np

    mh, mw = mask.shape[:2]
    sx, sy = mw / float(img_width), mh / float(img_height)
    x1 = min(mw - 1, max(0, int(math.floor(box[0] * sx))))
    y1 = min(mh - 1, max(0, int(math.floor(box[1] * sy))))
    x2 = min(mw - 1, max(x1, int(math.ceil(box[2] * sx))))
    y2 = min(mh - 1, max(y1, int(math.ceil(box[3] * sy))))
    return bool(np.all(mask[y1:y2 + 1, x1:x2 + 1] > 127))


class _GeoPropagator:
    """Projects a pixel-space box from one frame to another via the DEM.

    Forward step: the four box corners are ray-cast onto the DEM mesh with
    the source frame's camera (``label_to_world_coordinates``).  Reverse
    step: the resulting local world points are projected with the target
    frame's camera (``_world_to_pixel`` from :mod:`bambi_box_projector`).
    Both cameras are built exactly like in ``BambiProcessor.run_georeference``
    (1× rotation correction), so no DEM origin offset is needed — everything
    stays in the mesh-local coordinate space of the poses file.
    """

    def __init__(self, target_folder: str, dem_path: str,
                 correction_path: str, modality: str):
        self.target_folder = target_folder
        self.dem_path = dem_path
        self.correction_path = correction_path
        self.modality = modality
        self._tri_mesh = None
        self._correction = None
        self._valid_mask = None
        self._valid_mask_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._tri_mesh is not None

    def load(self) -> None:
        """Load the DEM mesh (may take a while for large meshes)."""
        from alfspy.render.render import read_gltf
        from trimesh import Trimesh
        from .corrections import read_correction

        if not self.dem_path or not os.path.isfile(self.dem_path):
            raise RuntimeError(
                "DEM mesh file not found. Please set the DEM path (GLB/GLTF) "
                "in the plugin's Input tab or via the field in this dialog."
            )
        mesh_data, _ = read_gltf(self.dem_path)
        tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
        try:
            _ = tri_mesh.triangles_tree  # pre-build BVH
        except Exception:  # nosec B110
            pass
        self._tri_mesh = tri_mesh
        self._correction = read_correction(self.target_folder, self.correction_path)

    def _build_camera(self, images: List[dict], frame_idx: int,
                      img_width: int, img_height: int):
        """Camera construction identical to ``run_georeference``
        (single source: :func:`core.camera_pose.frame_camera`)."""
        from .camera_pose import frame_camera

        return frame_camera(
            images, frame_idx, self._correction,
            aspect_ratio=img_width / img_height)

    def propagate(self, box: Tuple[float, float, float, float],
                  src_frame: int, dst_frame: int, images: List[dict],
                  img_width: int, img_height: int,
                  ) -> Tuple[float, float, float, float]:
        """Return *box* re-projected from *src_frame* to *dst_frame*.

        Both frames belong to the same modality (same ``images`` list and
        resolution).  Raises ``RuntimeError`` with a human-readable message
        on failure.
        """
        if not (0 <= src_frame < len(images)) or not (0 <= dst_frame < len(images)):
            raise RuntimeError("Frame index outside the poses file range.")
        return self.propagate_between(
            box, src_frame, images, img_width, img_height,
            dst_frame, images, img_width, img_height)

    def propagate_series(self, box: Tuple[float, float, float, float],
                         src_frame: int, dst_frame: int, images: List[dict],
                         img_width: int, img_height: int, step: int = 0,
                         ) -> Tuple[List[Tuple[int, Box]],
                                    List[Tuple[int, str]]]:
        """Propagate *box* to the target frame and to intermediate samples.

        The source box is ray-cast onto the DEM once and back-projected into
        every frame from :func:`propagation_frames`, so projection errors do
        not accumulate along the series.  Returns ``(boxes, failures)`` where
        *boxes* is a list of ``(frame, box)`` in propagation order and
        *failures* a list of ``(frame, message)`` for frames the box could not
        be projected into (outside the frame or the valid mask area, frame
        outside the poses range).

        Raises ``RuntimeError`` if the source box itself cannot be ray-cast
        onto the DEM — that failure applies to the whole series.
        """
        frames = propagation_frames(src_frame, dst_frame, step)
        world = self._box_to_world(
            box, src_frame, images, img_width, img_height)

        boxes: List[Tuple[int, Box]] = []
        failures: List[Tuple[int, str]] = []
        for frame in frames:
            try:
                boxes.append((frame, self._world_to_box(
                    world, frame, images, img_width, img_height)))
            except RuntimeError as exc:
                failures.append((frame, str(exc)))
        return boxes, failures

    def propagate_between(self, box: Tuple[float, float, float, float],
                          src_frame: int, src_images: List[dict],
                          src_width: int, src_height: int,
                          dst_frame: int, dst_images: List[dict],
                          dst_width: int, dst_height: int,
                          ) -> Tuple[float, float, float, float]:
        """Re-project *box* from a source frame/camera onto a target one.

        The source and target may come from different modalities (different
        ``images`` lists and resolutions): the box corners are ray-cast onto
        the shared DEM with the source camera and back-projected with the
        target camera.  Both cameras apply the same drone-pose correction
        (``correction.json`` is per-flight, not per-camera), each keyed by its
        own frame index.

        Raises ``RuntimeError`` with a human-readable message on failure.
        """
        world = self._box_to_world(
            box, src_frame, src_images, src_width, src_height)
        return self._world_to_box(
            world, dst_frame, dst_images, dst_width, dst_height)

    def _box_to_world(self, box: Tuple[float, float, float, float],
                      src_frame: int, src_images: List[dict],
                      src_width: int, src_height: int):
        """Ray-cast the four box corners onto the DEM with the source camera."""
        try:
            from alfspy.core.rendering import Resolution
            from bambi.util.projection_util import label_to_world_coordinates
        except ImportError as exc:
            raise RuntimeError(
                "alfspy / bambi packages are not available — cannot "
                f"geo-reference bounding boxes.\n({exc})"
            )

        if not self.is_loaded:
            self.load()
        if not (0 <= src_frame < len(src_images)):
            raise RuntimeError("Source frame index outside the poses range.")

        x1, y1, x2, y2 = box
        label_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
        cam_src = self._build_camera(src_images, src_frame, src_width, src_height)
        world = label_to_world_coordinates(
            label_coords, Resolution(src_width, src_height),
            self._tri_mesh, cam_src)
        if world is None or len(world) == 0:
            raise RuntimeError(
                "The bounding box could not be ray-cast onto the DEM "
                "(no mesh intersection)."
            )
        return world

    def _world_to_box(self, world, dst_frame: int, dst_images: List[dict],
                      dst_width: int, dst_height: int,
                      ) -> Tuple[float, float, float, float]:
        """Project local world points into the target frame's image plane.

        Besides the frame bounds the projected box is validated against the
        undistortion mask (``mask_T.png`` / ``mask_W.png``): the target
        frames are always this propagator's modality, so its mask applies.
        """
        import numpy as np
        from .camera_pose import world_to_pixel

        if not (0 <= dst_frame < len(dst_images)):
            raise RuntimeError("Target frame index outside the poses range.")

        cam_dst = self._build_camera(dst_images, dst_frame, dst_width, dst_height)
        pxs, pys = world_to_pixel(
            np.asarray(world, dtype=np.float64), dst_width, dst_height, cam_dst)

        nx1, ny1 = float(np.min(pxs)), float(np.min(pys))
        nx2, ny2 = float(np.max(pxs)), float(np.max(pys))
        if nx2 < 0 or ny2 < 0 or nx1 > dst_width or ny1 > dst_height:
            raise RuntimeError(
                f"The projected box lies completely outside frame {dst_frame}."
            )
        box = (
            max(0.0, nx1), max(0.0, ny1),
            min(float(dst_width - 1), nx2), min(float(dst_height - 1), ny2),
        )
        if not box_in_valid_area(box, self._mask(), dst_width, dst_height):
            raise RuntimeError(
                f"The projected box lies outside the valid (white) mask "
                f"area in frame {dst_frame}."
            )
        return box

    def _mask(self):
        """The modality's undistortion mask, loaded once (``None`` = no mask)."""
        if not self._valid_mask_loaded:
            self._valid_mask = load_valid_mask(
                self.target_folder, self.modality)
            self._valid_mask_loaded = True
        return self._valid_mask
