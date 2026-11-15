# -*- coding: utf-8 -*-
"""Camtrap DP 1.0 exporter.

`Camtrap DP <https://camtrap-dp.tdwg.org>`_ (a Frictionless Data Package)
describes a camera survey as *deployments*, *media* and *observations*. A
drone flight maps onto it: the flight is the deployment, the extracted frames
are the media, and the observations are

* one **media** observation per box - what was seen on that frame, where on
  it (``bboxX`` ... ``bboxHeight``), and
* one **event** observation per tracked animal - the animal itself, from its
  first frame to its last, with the species, sex and life stage it was given.

The three tables are written with **every field of their schema, in the
schema's order**. Readers such as the R package ``frictionless`` (behind
``camtrapdp``) take the column names from the schema and the values from the
file by position, so a table that leaves columns out - or adds its own - is
read into the wrong columns.

What the format has no column for goes into ``observationTags`` as
``key:value`` pairs, which is what the specification provides them for: the
position of the animal on the ground (the deployment has one location, and a
drone's animals are spread over the whole flight), the track, and custom
fields.

Up to 9.0 this export carried a reference to the Camtrap DP profile and
little of what the profile asks for: no ``contributors``, ``project``,
``spatial``, ``temporal`` or ``taxonomic`` in the package, no table schemas,
no ``filePublic`` / ``fileMediatype`` on the media, no ``eventStart`` /
``eventEnd`` on the observations, and four columns the schema does not know.
"""

import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from . import common
from .biodiversity import (
    _approved, _gbif_taxon_id, _life_stage, _scientific_name, _sex,
)
from .. import store

#: The version of the specification the package declares and follows.
VERSION = "1.0"
_BASE = f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{VERSION}"
PROFILE = f"{_BASE}/camtrap-dp-profile.json"

#: The fields of the three tables, in the order of their schemas.
DEPLOYMENT_FIELDS = (
    "deploymentID", "locationID", "locationName", "latitude", "longitude",
    "coordinateUncertainty", "deploymentStart", "deploymentEnd", "setupBy",
    "cameraID", "cameraModel", "cameraDelay", "cameraHeight", "cameraDepth",
    "cameraTilt", "cameraHeading", "detectionDistance", "timestampIssues",
    "baitUse", "featureType", "habitat", "deploymentGroups",
    "deploymentTags", "deploymentComments",
)
MEDIA_FIELDS = (
    "mediaID", "deploymentID", "captureMethod", "timestamp", "filePath",
    "filePublic", "fileName", "fileMediatype", "exifData", "favorite",
    "mediaComments",
)
OBSERVATION_FIELDS = (
    "observationID", "deploymentID", "mediaID", "eventID", "eventStart",
    "eventEnd", "observationLevel", "observationType", "cameraSetupType",
    "scientificName", "count", "lifeStage", "sex", "behavior",
    "individualID", "individualPositionRadius", "individualPositionAngle",
    "individualSpeed", "bboxX", "bboxY", "bboxWidth", "bboxHeight",
    "classificationMethod", "classifiedBy", "classificationTimestamp",
    "classificationProbability", "observationTags", "observationComments",
)

#: ``project.samplingDesign`` values of the specification.
SAMPLING_DESIGNS = ("simpleRandom", "systematicRandom", "clusteredRandom",
                    "experimental", "targeted", "opportunistic")

#: ``taxonomic[].taxonRank`` values of the specification.
_TAXON_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus",
                "species", "subspecies")

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".tif": "image/tiff",
                ".tiff": "image/tiff"}

#: Who classified what a person has not: this software.
CLASSIFIED_BY = "BAMBI QGIS plugin"


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def package_name(text: str) -> str:
    """*text* as a Data Package ``name``: lower case, and only letters,
    digits, ``.``, ``_`` and ``-``."""
    name = re.sub(r"[^a-z0-9._-]+", "-", (text or "").strip().lower())
    return name.strip("-") or "bambi-survey"


def media_type(name: str) -> str:
    return _MEDIA_TYPES.get(os.path.splitext(name)[1].lower(), "image/jpeg")


def tags(pairs: Sequence[Tuple[str, object]]) -> str:
    """``key:value`` pairs as ``observationTags``: joined with ``|``, which
    therefore cannot be part of a value. What has no value is left out."""
    cells = []
    for key, value in pairs:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        text = str(value).replace("|", "/").strip()
        cells.append(f"{str(key).replace('|', '/').strip()}:{text}")
    return " | ".join(cells)


def relative_box(row: dict, size: Optional[Tuple[int, int]]
                 ) -> Tuple[object, object, object, object]:
    """The box as ``bboxX, bboxY, bboxWidth, bboxHeight`` - fractions of the
    frame, as the specification counts them. Empty when the frame size is
    not known or the box has no area on the frame."""
    if not size or size[0] <= 0 or size[1] <= 0:
        return "", "", "", ""
    width, height = size
    left = min(max(row["x1"] / width, 0.0), 1.0)
    top = min(max(row["y1"] / height, 0.0), 1.0)
    right = min(max(row["x2"] / width, 0.0), 1.0)
    bottom = min(max(row["y2"] / height, 0.0), 1.0)
    if right <= left or bottom <= top:
        return "", "", "", ""
    return (f"{left:.6f}", f"{top:.6f}",
            f"{right - left:.6f}", f"{bottom - top:.6f}")


def _stamp(moment: datetime) -> str:
    """A moment the way the tables want it: ``YYYY-MM-DDThh:mm:ss±hh:mm``."""
    return moment.replace(microsecond=0).isoformat()


def _capture_times(frames: Dict[int, dict]) -> Dict[int, datetime]:
    """The capture time of each frame that has one **with its time zone**.

    A time without a zone cannot be published: the specification asks for
    the offset, and guessing it moves every observation by hours.
    """
    times = {}
    for index, frame in frames.items():
        moment = common.capture_time(frame.get("epoch"))
        if moment is not None and moment.tzinfo is not None:
            times[index] = moment
    return times


def _uncertainty(lonlat: Sequence[Tuple[float, float]],
                 centre: Tuple[float, float]) -> int:
    """How far, in whole metres, the furthest animal is from the location
    given for the deployment - its ``coordinateUncertainty`` (at least 1)."""
    lon0, lat0 = centre
    furthest = 0.0
    for lon, lat in lonlat:
        north = (lat - lat0) * 111_320.0
        east = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        furthest = max(furthest, math.hypot(north, east))
    return max(1, int(math.ceil(furthest)))


def _taxonomic(species_ids, vocabulary: dict) -> List[dict]:
    """The package's ``taxonomic`` list: every species observed that has a
    scientific name, with its GBIF key, rank and working name."""
    seeded = {name for _id, name in store.SEEDED_SPECIES}
    taxa = {}
    for species_id in sorted(set(species_ids)):
        taxonomy = vocabulary["taxonomy"].get(species_id, {})
        scientific = _scientific_name(taxonomy)
        if not scientific or scientific in taxa:
            continue
        taxon = {"scientificName": scientific}
        taxon_id = _gbif_taxon_id(taxonomy)
        if taxon_id:
            taxon["taxonID"] = taxon_id
        rank = (taxonomy.get("taxon_rank") or "").strip().lower()
        if rank in _TAXON_RANKS:
            taxon["taxonRank"] = rank
        name = vocabulary["species"].get(species_id, "")
        if name:
            # The names the plugin ships with are English; one the user
            # entered is in a language nobody recorded ("und").
            taxon["vernacularNames"] = {
                "eng" if name in seeded else "und": name}
        taxa[scientific] = taxon
    return list(taxa.values())


def _row(fields: Sequence[str], values: dict) -> list:
    unknown = set(values) - set(fields)
    if unknown:   # pragma: no cover - a programming error
        raise KeyError(f"not a field of the table: {sorted(unknown)}")
    return [values.get(field, "") for field in fields]


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

def export_camtrap_dp(target_folder: str, modality: str, output_folder: str,
                      epsg: Optional[int] = None,
                      deployment_id: str = "",
                      include_not_an_animal: bool = True,
                      include_images: bool = True,
                      sampling_design: str = "systematicRandom",
                      log_fn=None) -> str:
    """Write a Camtrap DP 1.0 package: deployments, media, observations.

    ``not-an-animal`` is kept by default and published as an observation of
    type ``blank``: a survey record should say what was looked at and rejected,
    not silently omit it.

    *include_images* copies the frames into ``media/`` and points ``filePath``
    there, which is what makes the package self-contained. Without it the paths
    stay relative to the target folder, so the package only resolves next to
    the flight it came from.

    *sampling_design* is the package's ``project.samplingDesign``. A flight
    along planned lines is ``systematicRandom``; the specification's other
    values are in :data:`SAMPLING_DESIGNS`.

    Refuses - rather than writing a package that says something false -
    when the CRS is unknown, when no box has a position (the deployment
    needs one), and when frames have no capture time with a time zone.
    """
    if sampling_design not in SAMPLING_DESIGNS:
        raise common.ExportError(
            f"'{sampling_design}' is not a Camtrap DP sampling design "
            f"({', '.join(SAMPLING_DESIGNS)}).")

    vocabulary = common.load_vocabulary(target_folder)
    taxonomy = vocabulary["taxonomy"]
    rows, animals = common.load_export_rows(
        target_folder, modality, vocabulary, include_not_an_animal)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}
    rows = common.without_duplicated(rows, frames)
    if not rows:
        raise common.ExportError(
            "Nothing to export: this camera has no box on any frame.")

    located = [row for row in rows if row.get("gx1") is not None]
    lonlat = common.to_wgs84(
        [((r["gx1"] + r["gx2"]) / 2.0, (r["gy1"] + r["gy2"]) / 2.0)
         for r in located], epsg)
    if not located:
        raise common.ExportError(
            "Camtrap DP needs to say where the flight was, and none of the "
            f"{len(rows)} box(es) has a position. Run 'Geo-reference "
            "detections' first.")
    positions = {row["detection_id"]: pair
                 for row, pair in zip(located, lonlat)}

    times = _capture_times(frames)
    used_frames = sorted({row["frame"] for row in rows})
    untimed = [frame for frame in used_frames if frame not in times]
    if untimed:
        raise common.ExportError(
            f"Camtrap DP needs the capture time of every frame, with its "
            f"time zone. {len(untimed)} of the {len(used_frames)} frame(s) "
            f"with a box have none (first: frame {untimed[0]}). Extract the "
            "frames again with the flight log and the time zone set.")

    deployment_id = deployment_id or os.path.basename(
        os.path.normpath(target_folder)) or "deployment"
    size = common.frame_size(target_folder, modality)
    common.ensure_folder(output_folder)

    lons = [lon for lon, _lat in lonlat]
    lats = [lat for _lon, lat in lonlat]
    centre = (sum(lons) / len(lons), sum(lats) / len(lats))
    start, end = min(times.values()), max(times.values())
    camera = "thermal" if modality == "t" else "RGB"

    # -- deployments ------------------------------------------------------
    with open(os.path.join(output_folder, "deployments.csv"), "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(DEPLOYMENT_FIELDS)
        writer.writerow(_row(DEPLOYMENT_FIELDS, {
            "deploymentID": deployment_id,
            "locationID": deployment_id,
            "latitude": f"{centre[1]:.7f}",
            "longitude": f"{centre[0]:.7f}",
            # the animals are spread over the flight; this is how far
            "coordinateUncertainty": _uncertainty(lonlat, centre),
            "deploymentStart": _stamp(start),
            "deploymentEnd": _stamp(end),
            "baitUse": "false",
            "deploymentTags": tags([("platform", "drone"),
                                    ("camera", camera)]),
            "deploymentComments": (
                "Aerial survey: one drone flight, exported from the BAMBI "
                "QGIS plugin. The location is the centre of the animals "
                "seen; each observation carries its own position in "
                "observationTags."),
        }))

    # -- media --------------------------------------------------------------
    used_names = [frames.get(frame, {}).get(
        "imagefile", f"frame_{frame:06d}.jpg") for frame in used_frames]
    copied = None
    if include_images:
        copied = common.copy_frames(target_folder, modality, used_names,
                                    os.path.join(output_folder, "media"))

    with open(os.path.join(output_folder, "media.csv"), "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(MEDIA_FIELDS)
        for frame, name in zip(used_frames, used_names):
            # The path has to describe where the file actually is, so it
            # follows whether the media travelled with the package - and it
            # is a path of the package, with "/" on every system.
            folder = "media" if include_images else f"frames_{modality}"
            writer.writerow(_row(MEDIA_FIELDS, {
                "mediaID": f"{deployment_id}-{frame}",
                "deploymentID": deployment_id,
                "captureMethod": "timeLapse",
                "timestamp": _stamp(times[frame]),
                "filePath": f"{folder}/{name}",
                "filePublic": "false",
                "fileName": name,
                "fileMediatype": media_type(name),
            }))

    # -- observations -----------------------------------------------------
    def identity(species_id: int, attributes: dict) -> dict:
        """What an observation says about who was seen."""
        if species_id == common.NOT_AN_ANIMAL:
            return {"observationType": "blank",
                    "observationComments": "rejected: not an animal"}
        scientific = _scientific_name(taxonomy.get(species_id, {}))
        said = {"observationType": "animal", "scientificName": scientific,
                "count": 1, "lifeStage": _life_stage(attributes),
                "sex": _sex(attributes)}
        return said

    def working_name(species_id: int) -> str:
        """The plugin's name for a species the package cannot name
        scientifically - else it would be lost."""
        if species_id == common.NOT_AN_ANIMAL or _scientific_name(
                taxonomy.get(species_id, {})):
            return ""
        return vocabulary["species"].get(species_id, "")

    def further(attributes: dict) -> list:
        """The fields that have no column of their own, as tags."""
        return [(key, value) for key, value in sorted(attributes.items())
                if key not in ("sex", "age", "approved")]

    n_events = 0
    with open(os.path.join(output_folder, "observations.csv"), "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OBSERVATION_FIELDS)

        for row in rows:
            lon, lat = positions.get(row["detection_id"], (None, None))
            attributes = row["properties"]
            by_hand = row.get("label_track_id") is not None or \
                _approved(attributes)
            stamp = _stamp(times[row["frame"]])
            left, top, width, height = relative_box(row, size)
            values = {
                "observationID": f"{deployment_id}-obs-{row['detection_id']}",
                "deploymentID": deployment_id,
                "mediaID": f"{deployment_id}-{row['frame']}",
                "eventID": (f"{deployment_id}-track-{row['track_id']}"
                            if row.get("track_id") is not None else ""),
                "eventStart": stamp,
                "eventEnd": stamp,
                "observationLevel": "media",
                "bboxX": left, "bboxY": top,
                "bboxWidth": width, "bboxHeight": height,
                "classificationMethod": "human" if by_hand else "machine",
                "classifiedBy": "" if by_hand else CLASSIFIED_BY,
                "classificationProbability": "" if by_hand else
                f"{min(max(float(row['confidence']), 0.0), 1.0):.4f}",
                "observationTags": tags([
                    ("latitude", f"{lat:.7f}" if lat is not None else ""),
                    ("longitude", f"{lon:.7f}" if lon is not None else ""),
                    ("species", working_name(row["species_id"])),
                    *further(attributes)]),
            }
            values.update(identity(row["species_id"], attributes))
            writer.writerow(_row(OBSERVATION_FIELDS, values))

        # One observation per animal: the event its boxes belong to.
        placed = {row["detection_id"] for row in located}
        for track_id, boxes in sorted(common.tracks_of(rows).items()):
            animal = animals.get(track_id)
            if animal is None:
                continue
            ordered = sorted(boxes, key=lambda r: r["frame"])
            seen = [positions[b["detection_id"]] for b in ordered
                    if b["detection_id"] in placed]
            lon = sum(p[0] for p in seen) / len(seen) if seen else None
            lat = sum(p[1] for p in seen) / len(seen) if seen else None
            attributes = animal["attributes"]
            by_hand = _approved(attributes) or all(
                b.get("label_track_id") is not None for b in ordered)
            values = {
                "observationID": f"{deployment_id}-animal-{track_id}",
                "deploymentID": deployment_id,
                "eventID": f"{deployment_id}-track-{track_id}",
                "eventStart": _stamp(times[ordered[0]["frame"]]),
                "eventEnd": _stamp(times[ordered[-1]["frame"]]),
                "observationLevel": "event",
                "classificationMethod": "human" if by_hand else "machine",
                "classifiedBy": "" if by_hand else CLASSIFIED_BY,
                "observationTags": tags([
                    ("latitude", f"{lat:.7f}" if lat is not None else ""),
                    ("longitude", f"{lon:.7f}" if lon is not None else ""),
                    ("species", working_name(animal["species_id"])),
                    ("boxes", len(ordered)),
                    *further(attributes)]),
            }
            values.update(identity(animal["species_id"], attributes))
            writer.writerow(_row(OBSERVATION_FIELDS, values))
            n_events += 1

    # -- the package -------------------------------------------------------
    title = f"BAMBI aerial survey - {deployment_id}"
    package = {
        "name": package_name(deployment_id),
        "profile": PROFILE,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "title": title,
        "description": (
            f"One drone flight ({camera} camera): its extracted frames as "
            "media, each box as a media-based observation and each tracked "
            "animal as an event-based observation."),
        # Who is responsible for the survey is not something the plugin
        # knows; the specification wants at least this much.
        "contributors": [{"title": "BAMBI QGIS plugin",
                          "role": "contributor"}],
        "sources": [{"title": "BAMBI QGIS plugin"}],
        "project": {
            "title": title,
            "samplingDesign": sampling_design,
            "captureMethod": ["timeLapse"],
            "individualAnimals": False,
            "observationLevel": ["media", "event"],
        },
        "spatial": {
            "type": "Polygon",
            "coordinates": [[
                [min(lons), min(lats)], [max(lons), min(lats)],
                [max(lons), max(lats)], [min(lons), max(lats)],
                [min(lons), min(lats)]]],
        },
        "temporal": {"start": start.date().isoformat(),
                     "end": end.date().isoformat()},
        "taxonomic": _taxonomic(
            (row["species_id"] for row in rows), vocabulary),
        "resources": [
            {"name": name, "path": f"{name}.csv",
             "profile": "tabular-data-resource",
             "schema": f"{_BASE}/{name}-table-schema.json"}
            for name in ("deployments", "media", "observations")],
    }
    with open(os.path.join(output_folder, "datapackage.json"), "w",
              encoding="utf-8") as fh:
        json.dump(package, fh, indent=2)

    if log_fn:
        log_fn(f"Camtrap DP: {len(rows)} box(es) and {n_events} animal(s) as "
               f"observations over {len(used_frames)} frame(s) → "
               f"{output_folder}")
        log_fn("Camtrap DP: before publishing, add the people responsible "
               "to 'contributors' in datapackage.json and check "
               f"project.samplingDesign ('{sampling_design}').")
        if size is None:
            log_fn("Camtrap DP: the frame size is not known (no extracted "
                   "frame could be read), so the observations have no bbox.")
        for line in common.describe_copy(copied) if copied else []:
            log_fn(f"Camtrap DP: {line}")
    return output_folder
