# -*- coding: utf-8 -*-
"""Darwin Core Archive exporter, and what it shares with Camtrap DP.

**Darwin Core Archive** is the GBIF publishing format: an ``occurrence.txt``
plus ``meta.xml`` describing it. One occurrence per *track* rather than per
detection - a track is one animal seen once, whereas emitting every detection
would publish the same roe deer two hundred times.

It needs world coordinates in latitude/longitude. The store holds them in the
project CRS, so *epsg* must be given and ``pyproj`` must be available; without
it the export refuses rather than publishing coordinates in the wrong system.

The Camtrap DP exporter is in :mod:`.camtrap`; the two share how a species is
named and how sex, age and the approval are worded.
"""

import csv
import os
from typing import Dict, List, Optional

from . import common

#: Darwin Core says machine observations are exactly that.
BASIS_OF_RECORD = "MachineObservation"


#: Shared with the GeoJSON exporters - every publishing format speaks WGS84.
_to_wgs84 = common.to_wgs84


def _scientific_name(taxonomy: dict) -> str:
    """The species' scientific name, or ``""`` when it has none.

    ``species.name`` is the working label the user picks in the combo box -
    "roe deer" - which is a *vernacular* name. Darwin Core and Camtrap DP both
    want the scientific one, and publishing a vernacular as ``scientificName``
    leaves GBIF fuzzy-matching a string.
    """
    return (taxonomy.get("scientific_name") or "").strip()


def _gbif_taxon_id(taxonomy: dict) -> str:
    """``taxonID`` for a species, from its GBIF taxon key. ``""`` when unset."""
    key = taxonomy.get("gbif_taxon_key")
    return GBIF_SPECIES_URL.format(key=key) if key else ""


def _timestamp(frames: Dict[int, dict], frame: int) -> str:
    """ISO-8601 capture time of *frame*, or ``""`` when unknown.

    Written at second precision in the ``YYYY-MM-DDThh:mm:ss±hh:mm`` form
    Camtrap DP and Darwin Core expect - previously the ISO string was fed to
    ``float()`` and every media row silently lost its timestamp.
    """
    moment = common.capture_time(frames.get(frame, {}).get("epoch"))
    if moment is None:
        return ""
    # A naive timestamp is written as such rather than given a zone it
    # never had; the drone's SRT clock normally carries one.
    return moment.replace(microsecond=0).isoformat()


def _track_summary(rows: List[dict], animals: Dict[int, dict],
                   vocabulary: dict) -> List[dict]:
    """One row per tracked animal: where and when it was first seen, and
    what is known about it.

    The species and the attributes are the animal's
    (:func:`common.animals_by_track`). They used to be those of its first
    box - what the detector called it on one frame - so an animal the
    classifier had named, or that had been identified by hand, was
    published as "animal", or held back for having no species.
    """
    summaries = []
    for track_id, points in sorted(common.tracks_of(rows).items()):
        located = [p for p in points if p.get("gx1") is not None]
        if not located or track_id not in animals:
            continue
        ordered = sorted(located, key=lambda r: r["frame"])
        first = ordered[0]
        animal = animals[track_id]
        summaries.append({
            "track_id": track_id,
            "species_id": animal["species_id"],
            "species": animal["species"],
            "taxonomy": vocabulary["taxonomy"].get(animal["species_id"], {}),
            "frame": first["frame"],
            "last_frame": ordered[-1]["frame"],
            "x": (first["gx1"] + first["gx2"]) / 2.0,
            "y": (first["gy1"] + first["gy2"]) / 2.0,
            "n_detections": len(ordered),
            "attributes": dict(animal["attributes"]),
        })
    return summaries


#: Darwin Core's ``sex`` and ``lifeStage`` take the plugin's own words; what
#: is not known is left empty rather than published as "unknown".
_KNOWN_SEX = ("female", "male")
_KNOWN_LIFE_STAGE = ("adult", "subadult", "juvenile")


def _sex(attributes: dict) -> str:
    value = str(attributes.get("sex") or "").strip().lower()
    return value if value in _KNOWN_SEX else ""


def _life_stage(attributes: dict) -> str:
    value = str(attributes.get("age") or "").strip().lower()
    return value if value in _KNOWN_LIFE_STAGE else ""


def _approved(attributes: dict) -> bool:
    """Whether a person has confirmed what is said about the animal."""
    return attributes.get("approved") in (True, 1, "true", "True", "yes")


# ---------------------------------------------------------------------------
# Darwin Core Archive
# ---------------------------------------------------------------------------

#: Terms written to ``occurrence.txt``, in column order.
DWC_TERMS = [
    "occurrenceID", "basisOfRecord", "scientificName", "vernacularName",
    "taxonRank", "taxonID", "individualCount", "eventDate",
    "decimalLatitude", "decimalLongitude", "geodeticDatum",
    "coordinateUncertaintyInMeters", "samplingProtocol", "occurrenceRemarks",
    "identificationVerificationStatus", "sex", "lifeStage",
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

    ``not-an-animal`` is always excluded - it is by definition not an
    occurrence of anything - and so are tracks still classed as ``unknown`` or
    the generic ``animal``, because GBIF wants a taxon. How many were dropped
    for that reason is reported, since it is a publishing decision rather than a
    technicality.
    """
    vocabulary = common.load_vocabulary(target_folder)
    rows, animals = common.load_export_rows(
        target_folder, modality, vocabulary, include_not_an_animal=False)
    frames = {f["frame"]: f for f in common.load_frames(target_folder, modality)}
    summaries = _track_summary(rows, animals, vocabulary)

    # A base class is not a taxon, and neither is a species whose scientific
    # name has not been filled in - `name` is the working label ("roe deer"),
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
            attributes = summary["attributes"]
            # sex, age and the approval have terms of their own
            remarks = ", ".join(
                f"{key}={value}" for key, value in sorted(attributes.items())
                if key not in ("sex", "age", "approved"))
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
                # approved by hand in the inventory or the labelling tool
                "verified" if _approved(attributes) else "unverified",
                _sex(attributes),
                _life_stage(attributes),
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
