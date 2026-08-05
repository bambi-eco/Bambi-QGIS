# -*- coding: utf-8 -*-
"""Camtrap DP and Darwin Core Archive exporters (EXCHANGE_FORMAT_PLAN.md §8.1).

These are the two formats that make a BAMBI survey publishable rather than just
portable, and they are the reason the extensible attribute model was worth
building: both carry arbitrary observation fields natively, so a user-defined
field reaches GBIF without anyone writing mapping code for it.

**Camtrap DP** (Frictionless Data) models camera-trap surveys as deployments,
media and observations. A drone flight maps cleanly: the flight is the
deployment, extracted frames are media, detections are observations.

**Darwin Core Archive** is the GBIF publishing format: an ``occurrence.txt``
plus ``meta.xml`` describing it. One occurrence per *track* rather than per
detection — a track is one animal seen once, whereas emitting every detection
would publish the same roe deer two hundred times.

Both need world coordinates in latitude/longitude. The store holds them in the
project CRS, so *epsg* must be given and ``pyproj`` must be available; without
it the export refuses rather than publishing coordinates in the wrong system.
"""

import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import common

#: Darwin Core says machine observations are exactly that.
BASIS_OF_RECORD = "MachineObservation"


def _to_wgs84(points, epsg: int):
    """Project ``[(x, y), …]`` from *epsg* to lon/lat.

    Raises :class:`common.ExportError` rather than guessing — publishing
    coordinates in the wrong reference system is worse than not publishing.
    """
    if not epsg:
        raise common.ExportError(
            "This format publishes latitude/longitude, so the project CRS must "
            "be known. Set the target CRS before exporting.")
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover — environment dependent
        raise common.ExportError(
            "pyproj is required to convert the project CRS to "
            "latitude/longitude. Install it from the Dependencies tab.") from exc

    transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return [transformer.transform(x, y) for x, y in points]


def _scientific_name(taxonomy: dict) -> str:
    """The species' scientific name, or ``""`` when it has none.

    ``species.name`` is the working label the user picks in the combo box —
    "roe deer" — which is a *vernacular* name. Darwin Core and Camtrap DP both
    want the scientific one, and publishing a vernacular as ``scientificName``
    leaves GBIF fuzzy-matching a string.
    """
    return (taxonomy.get("scientific_name") or "").strip()


def _gbif_taxon_id(taxonomy: dict) -> str:
    """``taxonID`` for a species, from its GBIF taxon key. ``""`` when unset."""
    key = taxonomy.get("gbif_taxon_key")
    return GBIF_SPECIES_URL.format(key=key) if key else ""


def _timestamp(frames: Dict[int, dict], frame: int) -> str:
    """ISO-8601 capture time of *frame*, or ``""`` when unknown."""
    epoch = frames.get(frame, {}).get("epoch")
    if not epoch:
        return ""
    try:
        return datetime.fromtimestamp(
            float(epoch), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return ""


def _track_summary(rows: List[dict], vocabulary: dict) -> List[dict]:
    """One row per track: its first frame, centre position and attributes."""
    summaries = []
    for track_id, points in sorted(common.tracks_of(rows).items()):
        located = [p for p in points if p.get("gx1") is not None]
        if not located:
            continue
        ordered = sorted(located, key=lambda r: r["frame"])
        first = ordered[0]
        summaries.append({
            "track_id": track_id,
            "species_id": first["species_id"],
            "species": vocabulary["species"].get(first["species_id"], ""),
            "taxonomy": vocabulary["taxonomy"].get(first["species_id"], {}),
            "frame": first["frame"],
            "x": (first["gx1"] + first["gx2"]) / 2.0,
            "y": (first["gy1"] + first["gy2"]) / 2.0,
            "n_detections": len(ordered),
            "attributes": common.resolve_attributes(
                first["attributes"], vocabulary["enum_labels"]),
        })
    return summaries


# ---------------------------------------------------------------------------
# Camtrap DP
# ---------------------------------------------------------------------------

def export_camtrap_dp(target_folder: str, modality: str, output_folder: str,
                      epsg: Optional[int] = None,
                      deployment_id: str = "",
                      include_not_an_animal: bool = True,
                      include_images: bool = True, log_fn=None) -> str:
    """Write a Camtrap DP package: deployments, media, observations.

    ``not-an-animal`` is kept by default and published as an observation of
    type ``blank``: a survey record should say what was looked at and rejected,
    not silently omit it.

    *include_images* copies the frames into ``media/`` and points ``filePath``
    there, which is what makes the package self-contained. Without it the paths
    stay relative to the target folder, so the package only resolves next to
    the flight it came from.
    """
    vocabulary = common.load_vocabulary(target_folder)
    taxonomy = vocabulary["taxonomy"]
    rows = common.load_detections(target_folder, modality, include_not_an_animal)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}
    common.ensure_folder(output_folder)

    deployment_id = deployment_id or os.path.basename(
        os.path.normpath(target_folder)) or "deployment"

    located = [row for row in rows if row.get("gx1") is not None]
    lonlat = _to_wgs84(
        [((r["gx1"] + r["gx2"]) / 2.0, (r["gy1"] + r["gy2"]) / 2.0)
         for r in located], epsg) if located else []
    positions = {row["detection_id"]: pair
                 for row, pair in zip(located, lonlat)}

    lons = [lon for lon, _ in lonlat] or [0.0]
    lats = [lat for _, lat in lonlat] or [0.0]
    timestamps = [t for t in
                  (_timestamp(frames, f) for f in sorted(frames)) if t]

    deployments = os.path.join(output_folder, "deployments.csv")
    with open(deployments, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "deploymentID", "locationID", "latitude", "longitude",
            "deploymentStart", "deploymentEnd", "cameraModel",
            "deploymentComments"])
        writer.writerow([
            deployment_id, deployment_id,
            f"{sum(lats) / len(lats):.7f}", f"{sum(lons) / len(lons):.7f}",
            timestamps[0] if timestamps else "",
            timestamps[-1] if timestamps else "",
            f"BAMBI drone survey ({'thermal' if modality == 't' else 'RGB'})",
            "Aerial survey exported from the BAMBI QGIS plugin"])

    used_frames = sorted({row["frame"] for row in rows})
    used_names = [frames.get(frame, {}).get(
        "imagefile", f"frame_{frame:06d}.jpg") for frame in used_frames]

    copied = None
    if include_images:
        copied = common.copy_frames(target_folder, modality, used_names,
                                    os.path.join(output_folder, "media"))

    media_path = os.path.join(output_folder, "media.csv")
    with open(media_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mediaID", "deploymentID", "timestamp",
                         "filePath", "fileName", "captureMethod"])
        for frame, name in zip(used_frames, used_names):
            # The path has to describe where the file actually is, so it
            # follows whether the media travelled with the package.
            path = os.path.join("media" if include_images
                                else f"frames_{modality}", name)
            writer.writerow([
                f"{deployment_id}-{frame}", deployment_id,
                _timestamp(frames, frame), path, name, "timeLapse"])

    extra_names = sorted({
        key for row in rows
        for key in common.resolve_attributes(
            row["attributes"], vocabulary["enum_labels"])})

    observations = os.path.join(output_folder, "observations.csv")
    with open(observations, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "observationID", "deploymentID", "mediaID", "eventID",
            "observationLevel", "observationType", "scientificName",
            "vernacularName", "taxonID",
            "count", "latitude", "longitude", "classificationProbability",
        ] + extra_names)
        for row in rows:
            lon, lat = positions.get(row["detection_id"], (None, None))
            is_blank = row["species_id"] == common.NOT_AN_ANIMAL
            attributes = common.resolve_attributes(
                row["attributes"], vocabulary["enum_labels"])
            writer.writerow([
                f"{deployment_id}-obs-{row['detection_id']}",
                deployment_id,
                f"{deployment_id}-{row['frame']}",
                (f"{deployment_id}-track-{row['track_id']}"
                 if row.get("track_id") is not None else ""),
                "media",
                "blank" if is_blank else "animal",
                # scientificName only when there is one; the working label goes
                # to vernacularName, where it belongs.
                "" if is_blank else _scientific_name(
                    taxonomy.get(row["species_id"], {})),
                "" if is_blank else vocabulary["species"].get(
                    row["species_id"], ""),
                "" if is_blank else _gbif_taxon_id(
                    taxonomy.get(row["species_id"], {})),
                1,
                f"{lat:.7f}" if lat is not None else "",
                f"{lon:.7f}" if lon is not None else "",
                row["confidence"],
            ] + [attributes.get(name, "") for name in extra_names])

    package = {
        "name": deployment_id.lower().replace(" ", "-"),
        "profile": "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0/camtrap-dp-profile.json",
        "created": datetime.now(timezone.utc).isoformat(),
        "title": f"BAMBI aerial survey — {deployment_id}",
        "sources": [{"title": "BAMBI QGIS plugin"}],
        "resources": [
            {"name": "deployments", "path": "deployments.csv",
             "profile": "tabular-data-resource"},
            {"name": "media", "path": "media.csv",
             "profile": "tabular-data-resource"},
            {"name": "observations", "path": "observations.csv",
             "profile": "tabular-data-resource"},
        ],
    }
    package_path = os.path.join(output_folder, "datapackage.json")
    with open(package_path, "w", encoding="utf-8") as fh:
        json.dump(package, fh, indent=2)

    if log_fn:
        log_fn(f"Camtrap DP: {len(rows)} observation(s) → {output_folder}")
        for line in common.describe_copy(copied) if copied else []:
            log_fn(f"Camtrap DP: {line}")
    return output_folder


# ---------------------------------------------------------------------------
# Darwin Core Archive
# ---------------------------------------------------------------------------

#: Terms written to ``occurrence.txt``, in column order.
DWC_TERMS = [
    "occurrenceID", "basisOfRecord", "scientificName", "vernacularName",
    "taxonRank", "taxonID", "individualCount", "eventDate",
    "decimalLatitude", "decimalLongitude", "geodeticDatum",
    "coordinateUncertaintyInMeters", "samplingProtocol", "occurrenceRemarks",
    "identificationVerificationStatus",
]

#: A GBIF taxon key resolves against the backbone exactly, instead of leaving
#: GBIF to match a name string.
GBIF_SPECIES_URL = "https://www.gbif.org/species/{key}"

_DWC_URI = "http://rs.tdwg.org/dwc/terms/"


def export_darwin_core(target_folder: str, modality: str, output_folder: str,
                       epsg: Optional[int] = None,
                       dataset_name: str = "",
                       coordinate_uncertainty: float = 10.0,
                       log_fn=None) -> str:
    """Write a Darwin Core Archive for GBIF publishing.

    One occurrence per **track**, not per detection: a track is one animal seen
    once, and publishing every detection would report the same roe deer as
    hundreds of separate occurrences.

    ``not-an-animal`` is always excluded — it is by definition not an
    occurrence of anything — and so are tracks still classed as ``unknown`` or
    the generic ``animal``, because GBIF wants a taxon. How many were dropped
    for that reason is reported, since it is a publishing decision rather than a
    technicality.
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows = common.load_detections(
        target_folder, modality, include_not_an_animal=False)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}
    summaries = _track_summary(rows, vocabulary)

    # A base class is not a taxon, and neither is a species whose scientific
    # name has not been filled in — `name` is the working label ("roe deer"),
    # which is a vernacular name. Publishing it as `scientificName` would leave
    # GBIF fuzzy-matching a string, so those tracks are held back and counted.
    def publishable(summary):
        has_species = summary["species_id"] > 0
        return has_species and bool(_scientific_name(summary["taxonomy"]))

    unnamed = [s for s in summaries if s["species_id"] <= 0]
    untyped = [s for s in summaries
               if s["species_id"] > 0 and not publishable(s)]
    named = [s for s in summaries if publishable(s)]
    if not named:
        raise common.ExportError(
            "Nothing to publish. Darwin Core needs a scientific name, and no "
            "track has one: assign species in the labelling tool, and give "
            "each species its scientific name (and optionally its GBIF taxon "
            "key) in the Project Schema editor.")

    lonlat = _to_wgs84([(s["x"], s["y"]) for s in named], epsg)
    dataset_name = dataset_name or os.path.basename(
        os.path.normpath(target_folder)) or "bambi-survey"
    common.ensure_folder(output_folder)

    occurrence_path = os.path.join(output_folder, "occurrence.txt")
    with open(occurrence_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(DWC_TERMS)
        for summary, (lon, lat) in zip(named, lonlat):
            remarks = ", ".join(
                f"{key}={value}" for key, value in
                sorted(summary["attributes"].items()))
            taxonomy = summary["taxonomy"]
            gbif_key = taxonomy.get("gbif_taxon_key")
            writer.writerow([
                f"{dataset_name}-track-{summary['track_id']}",
                BASIS_OF_RECORD,
                taxonomy.get("scientific_name") or "",
                summary["species"],
                taxonomy.get("taxon_rank") or "",
                GBIF_SPECIES_URL.format(key=gbif_key) if gbif_key else "",
                1,
                _timestamp(frames, summary["frame"]),
                f"{lat:.7f}", f"{lon:.7f}", "WGS84",
                coordinate_uncertainty,
                "aerial drone survey (BAMBI)",
                remarks,
                "unverified",
            ])

    meta_path = os.path.join(output_folder, "meta.xml")
    fields = "\n".join(
        f'    <field index="{index}" term="{_DWC_URI}{term}"/>'
        for index, term in enumerate(DWC_TERMS) if term != "occurrenceID")
    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<archive xmlns="http://rs.tdwg.org/dwc/text/">\n'
            '  <core encoding="UTF-8" fieldsTerminatedBy="\\t" '
            'linesTerminatedBy="\\n" ignoreHeaderLines="1" '
            f'rowType="{_DWC_URI}Occurrence">\n'
            '    <files><location>occurrence.txt</location></files>\n'
            '    <id index="0"/>\n'
            f"{fields}\n"
            '  </core>\n'
            '</archive>\n')

    if log_fn:
        keyed = sum(1 for s in named if s["taxonomy"].get("gbif_taxon_key"))
        message = (f"Darwin Core: {len(named)} occurrence(s) → "
                   f"{occurrence_path}")
        if keyed:
            message += f"; {keyed} carry a GBIF taxon key"
        if unnamed:
            message += (f"; {len(unnamed)} track(s) omitted for having no "
                        "species assigned")
        if untyped:
            names = ", ".join(sorted({s["species"] for s in untyped}))
            message += (f"; {len(untyped)} track(s) omitted because their "
                        f"species has no scientific name ({names})")
        log_fn(message)
    return output_folder
