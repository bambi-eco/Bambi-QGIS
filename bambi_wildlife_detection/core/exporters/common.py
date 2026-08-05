# -*- coding: utf-8 -*-
"""Shared loading and mapping for the exporters (EXCHANGE_FORMAT_PLAN.md §8.1).

Every exporter reads the same rows and then writes them in its own shape, so
the three decisions that would otherwise be made seven times live here:

* **Class-id remapping.** Internal ``species_id`` values are sparse by
  construction — base classes are ``≤ 0`` and user species accumulate gaps as
  species are deleted — and no external format tolerates that. Exporters emit
  contiguous, non-negative ids and ship the mapping alongside.
* **Enum resolution.** Attributes are stored as ``value_id``; no consumer
  should be handed an integer whose meaning lives in a table it does not have.
* **`not-an-animal`.** Species ``-2`` marks a labelled false positive.
  Training formats drop those rows by default; recording formats keep them.
  Either way it is an explicit option, never a silent filter.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from .. import gpkg, store, track_store

#: Detections whose species is this are labelled false positives (§3.1).
NOT_AN_ANIMAL = store.NOT_AN_ANIMAL_SPECIES_ID


class ExportError(RuntimeError):
    """The project cannot be exported in the requested shape."""


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def load_vocabulary(target_folder: str) -> dict:
    """Species names, enum labels and the custom field schema."""
    path = store.project_path(target_folder)
    if not os.path.isfile(path):
        raise ExportError(
            "This project has no 6.0 store yet — run the pipeline or use "
            "'Migrate 5.x…' before exporting.")

    conn = store.open_store(path, store.PROJECT)
    try:
        species = {}
        taxonomy = {}
        for row in conn.execute(
                "SELECT species_id, name, scientific_name, taxon_rank, "
                "gbif_taxon_key FROM species"):
            species[row["species_id"]] = row["name"]
            taxonomy[row["species_id"]] = {
                "vernacular_name": row["name"],
                "scientific_name": row["scientific_name"],
                "taxon_rank": row["taxon_rank"],
                "gbif_taxon_key": row["gbif_taxon_key"],
            }
        fields = [dict(row) for row in conn.execute(
            "SELECT name, type, scope, enum_id FROM field_schema")]
        enum_labels: Dict[int, Dict[int, str]] = {}
        for row in conn.execute(
                "SELECT enum_id, value_id, label FROM enum_values"):
            enum_labels.setdefault(row["enum_id"], {})[row["value_id"]] = \
                row["label"]
    finally:
        conn.close()

    by_field = {field["name"]: enum_labels.get(field["enum_id"], {})
                for field in fields if field["enum_id"] is not None}
    return {"species": species, "taxonomy": taxonomy, "fields": fields,
            "enum_labels": by_field}


def resolve_attributes(raw, enum_by_field: Dict[str, Dict[int, str]]) -> dict:
    """Turn stored ``attributes`` into consumer-facing values.

    Enum fields become their label; everything else passes through unchanged.
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}

    resolved = {}
    for name, value in data.items():
        labels = enum_by_field.get(name)
        if labels and isinstance(value, int) and value in labels:
            resolved[name] = labels[value]
        else:
            resolved[name] = value
    return resolved


def class_map(species_ids, species_names: Dict[int, str]
              ) -> Tuple[Dict[int, int], List[str]]:
    """Map the species present onto contiguous ids starting at 0.

    Returns ``(species_id -> index, [name per index])``. Sorted by
    ``species_id`` so the mapping is stable between exports of the same project.
    """
    present = sorted(set(species_ids))
    mapping = {species_id: index for index, species_id in enumerate(present)}
    names = [species_names.get(species_id, f"species_{species_id}")
             for species_id in present]
    return mapping, names


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def load_detections(target_folder: str, modality: str,
                    include_not_an_animal: bool = False,
                    include_geo: bool = True) -> List[dict]:
    """Every detection, with its geo box and track id where they exist.

    A detection that failed to geo-reference simply has ``None`` for its world
    coordinates — it is not dropped, because an exporter that only needs pixel
    boxes should still see it.
    """
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(det_path):
        raise ExportError(
            f"No detections stored for modality '{modality}'.")

    geo_path = store.stage_path(target_folder, store.GEOREFERENCED, modality)
    trk_path = store.stage_path(target_folder, store.TRACKS, modality)
    run_ids = track_store.analysis_runs(target_folder, modality)
    superseded = track_store.superseded_track_ids(target_folder, modality)

    columns = ["d.detection_id", "d.frame", "d.x1", "d.y1", "d.x2", "d.y2",
               "d.confidence", "d.species_id", "d.source_class",
               "d.label_track_id", "d.attributes"]
    joins = ""
    if include_geo and os.path.isfile(geo_path):
        columns += ["g.gx1", "g.gy1", "g.gz1", "g.gx2", "g.gy2", "g.gz2"]
        joins += (" LEFT JOIN geo.detections_geo g "
                  "ON g.detection_id = d.detection_id")
    if run_ids and os.path.isfile(trk_path):
        placeholders = ", ".join("?" for _ in run_ids)
        columns.append("m.track_id")
        joins += (" LEFT JOIN trk.track_members m "
                  "ON m.detection_id = d.detection_id"
                  " LEFT JOIN trk.tracks t ON t.track_id = m.track_id"
                  f" AND t.run_id IN ({placeholders})")  # nosec B608 — ints

    conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        if include_geo and os.path.isfile(geo_path):
            gpkg.attach(conn, geo_path, "geo")
        if run_ids and os.path.isfile(trk_path):
            gpkg.attach(conn, trk_path, "trk")

        sql = (f"SELECT {', '.join(columns)} FROM detections d{joins} "
               "ORDER BY d.frame, d.detection_id")  # nosec B608 — fixed names
        rows = [dict(row) for row in conn.execute(sql, run_ids)]

        if include_geo and os.path.isfile(geo_path):
            gpkg.detach(conn, "geo")
        if run_ids and os.path.isfile(trk_path):
            gpkg.detach(conn, "trk")
    finally:
        conn.close()

    for row in rows:
        if row.get("track_id") in superseded:
            row["track_id"] = None
    if not include_not_an_animal:
        rows = [row for row in rows if row["species_id"] != NOT_AN_ANIMAL]
    return rows


def load_frames(target_folder: str, modality: str) -> List[dict]:
    """Extracted frames from ``poses_{m}.json``, in frame order."""
    path = os.path.join(target_folder, f"poses_{modality}.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return []

    frames = []
    for index, image in enumerate(data.get("images", []) or []):
        frames.append({
            "frame": index,
            "imagefile": image.get("imagefile", f"frame_{index:06d}.jpg"),
            "location": image.get("location"),
            "epoch": image.get("epoch") or image.get("timestamp"),
        })
    return frames


def frame_size(target_folder: str, modality: str,
               explicit: Optional[Tuple[int, int]] = None
               ) -> Optional[Tuple[int, int]]:
    """Pixel size of the extracted frames, or ``None`` when unknowable.

    Formats that normalise coordinates need this; it is not in the poses file,
    so an extracted frame is probed. ``cv2`` is imported lazily, so ``core``
    stays importable without it.
    """
    if explicit:
        return explicit

    folder = os.path.join(target_folder, f"frames_{modality}")
    if not os.path.isdir(folder):
        return None
    try:
        import cv2
    except ImportError:  # pragma: no cover — depends on the environment
        return None

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        image = cv2.imread(os.path.join(folder, name))
        if image is not None:
            return image.shape[1], image.shape[0]
    return None


def require_frame_size(target_folder: str, modality: str,
                       explicit: Optional[Tuple[int, int]] = None
                       ) -> Tuple[int, int]:
    size = frame_size(target_folder, modality, explicit)
    if size is None:
        raise ExportError(
            "This format needs the frame size, which could not be determined. "
            "Extract frames first, or pass image_size explicitly.")
    return size


def tracks_of(rows: List[dict]) -> Dict[int, List[dict]]:
    """Group rows by track id, ignoring detections that belong to none."""
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        track_id = row.get("track_id")
        if track_id is None:
            continue
        grouped.setdefault(track_id, []).append(row)
    return grouped


def ensure_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def copy_frames(target_folder: str, modality: str, imagefiles,
                destination: str) -> dict:
    """Copy the named extracted frames into *destination*.

    Formats that reference images are only usable when the images travel with
    them, but the frames are the heaviest thing a project owns — so this is
    always a caller's choice, never implied. Returns what happened, because a
    label with no image is ignored in silence by every consumer.
    """
    import shutil

    source_folder = os.path.join(target_folder, f"frames_{modality}")
    ensure_folder(destination)

    copied, missing = 0, []
    for name in imagefiles:
        source = os.path.join(source_folder, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(destination, name))
            copied += 1
        else:
            missing.append(name)
    return {"copied": copied, "missing": missing, "source": source_folder,
            "destination": destination}


def describe_copy(result: dict) -> List[str]:
    """Log lines for a :func:`copy_frames` result — one, or two if any failed."""
    lines = [f"copied {result['copied']} image(s) into "
             f"{result['destination']}"]
    missing = result["missing"]
    if missing:
        shown = ", ".join(missing[:3])
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        lines.append(f"warning — {len(missing)} frame(s) had no image in "
                     f"{result['source']}: {shown}{more}. Consumers ignore an "
                     "annotation whose image is absent.")
    return lines


def no_tracks_hint(target_folder: str, modality: str) -> str:
    """Why a track-based export found nothing, when the answer is knowable.

    "0 rows" is true but useless: the usual cause is a tracking run that never
    reached the store, and the tracks file on disk makes it look as though
    tracking worked. This turns the symptom into the next action.
    """
    from .. import track_store

    if not track_store.analysis_runs(target_folder, modality):
        legacy = os.path.join(target_folder, f"tracks_{modality}", "tracks.csv")
        if os.path.isfile(legacy):
            return (" — no tracking run is recorded in the store, although "
                    f"{legacy} exists. It was written by a tracker that read "
                    "the legacy text files. Re-run 'Geo-reference detections' "
                    "and then tracking.")
        return (" — no tracking run is recorded for this modality. Run "
                "tracking first.")
    return (" — a tracking run exists but none of these detections belong to "
            "it. Re-run tracking if the detections changed since.")
