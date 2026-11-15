# -*- coding: utf-8 -*-
"""The Camtrap DP export against the Camtrap DP 1.0 specification.

The constraints below are those of the published table schemas and package
profile (https://camtrap-dp.tdwg.org, version 1.0): which fields a table has
and in which order, which of them are required, their value lists, ranges
and formats. A package is checked against all of them - the export used to
name the profile and meet little of it.
"""
import csv
import json
import os
import re
from datetime import datetime

import pytest

pytest.importorskip("pyproj")

from bambi_wildlife_detection.core import (  # noqa: E402
    detection_store, exporters, store, track_store)
from bambi_wildlife_detection.core.exporters import camtrap, common  # noqa: E402

BASE = "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0"
DATETIME = "%Y-%m-%dT%H:%M:%S%z"

# -- the specification ------------------------------------------------------

DEPLOYMENTS = (
    "deploymentID", "locationID", "locationName", "latitude", "longitude",
    "coordinateUncertainty", "deploymentStart", "deploymentEnd", "setupBy",
    "cameraID", "cameraModel", "cameraDelay", "cameraHeight", "cameraDepth",
    "cameraTilt", "cameraHeading", "detectionDistance", "timestampIssues",
    "baitUse", "featureType", "habitat", "deploymentGroups",
    "deploymentTags", "deploymentComments")
MEDIA = (
    "mediaID", "deploymentID", "captureMethod", "timestamp", "filePath",
    "filePublic", "fileName", "fileMediatype", "exifData", "favorite",
    "mediaComments")
OBSERVATIONS = (
    "observationID", "deploymentID", "mediaID", "eventID", "eventStart",
    "eventEnd", "observationLevel", "observationType", "cameraSetupType",
    "scientificName", "count", "lifeStage", "sex", "behavior",
    "individualID", "individualPositionRadius", "individualPositionAngle",
    "individualSpeed", "bboxX", "bboxY", "bboxWidth", "bboxHeight",
    "classificationMethod", "classifiedBy", "classificationTimestamp",
    "classificationProbability", "observationTags", "observationComments")

REQUIRED = {
    "deployments": ("deploymentID", "latitude", "longitude",
                    "deploymentStart", "deploymentEnd"),
    "media": ("mediaID", "deploymentID", "timestamp", "filePath",
              "filePublic", "fileMediatype"),
    "observations": ("observationID", "deploymentID", "eventStart",
                     "eventEnd", "observationLevel", "observationType"),
}
ENUMS = {
    "captureMethod": ("activityDetection", "timeLapse"),
    "observationLevel": ("media", "event"),
    "observationType": ("animal", "human", "vehicle", "blank", "unknown",
                        "unclassified"),
    "cameraSetupType": ("setup", "calibration"),
    "lifeStage": ("adult", "subadult", "juvenile"),
    "sex": ("female", "male"),
    "classificationMethod": ("human", "machine"),
    "featureType": ("roadPaved", "roadDirt", "trailHiking", "trailGame",
                    "roadUnderpass", "roadOverpass", "roadBridge", "culvert",
                    "burrow", "nestSite", "carcass", "waterSource",
                    "fruitingTree"),
}
#: field -> (minimum, maximum, whole numbers only)
NUMBERS = {
    "latitude": (-90, 90, False), "longitude": (-180, 180, False),
    "coordinateUncertainty": (1, None, True),
    "count": (1, None, True),
    "bboxX": (0, 1, False), "bboxY": (0, 1, False),
    "bboxWidth": (1e-15, 1, False), "bboxHeight": (1e-15, 1, False),
    "classificationProbability": (0, 1, False),
}
DATETIMES = ("deploymentStart", "deploymentEnd", "timestamp", "eventStart",
             "eventEnd", "classificationTimestamp")
BOOLEANS = ("timestampIssues", "baitUse", "filePublic", "favorite")
FILE_PATH = re.compile(r"^(?=^[^./~])(^((?!\.{2}).)*$).*$")
MEDIA_TYPE = re.compile(r"^(image|video|audio)/.*$")
UNIQUE = {"deployments": "deploymentID", "media": "mediaID",
          "observations": "observationID"}


def _table(folder, name):
    with open(os.path.join(folder, f"{name}.csv"), encoding="utf-8",
              newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return header, [dict(zip(header, row)) for row in reader]


def problems(folder):
    """Everything about the package in *folder* the specification objects
    to, as a list of sentences."""
    found = []
    tables = {}
    for name, fields in (("deployments", DEPLOYMENTS), ("media", MEDIA),
                         ("observations", OBSERVATIONS)):
        header, rows = _table(folder, name)
        tables[name] = rows
        if tuple(header) != fields:
            found.append(f"{name}: the columns are not the schema's fields "
                         f"in its order: {header}")
        seen = set()
        for number, row in enumerate(rows, start=2):
            where = f"{name} line {number}"
            for field in REQUIRED[name]:
                if row.get(field, "") == "":
                    found.append(f"{where}: {field} is required")
            key = row.get(UNIQUE[name])
            if key in seen:
                found.append(f"{where}: {UNIQUE[name]} {key} twice")
            seen.add(key)
            for field, value in row.items():
                if value == "":
                    continue
                if field in ENUMS and value not in ENUMS[field]:
                    found.append(f"{where}: {field} '{value}' is not one "
                                 f"of {ENUMS[field]}")
                if field in NUMBERS:
                    low, high, whole = NUMBERS[field]
                    try:
                        number_value = int(value) if whole else float(value)
                    except ValueError:
                        found.append(f"{where}: {field} '{value}' is not "
                                     "a number")
                        continue
                    if number_value < low or (
                            high is not None and number_value > high):
                        found.append(f"{where}: {field} {value} is out of "
                                     "range")
                if field in DATETIMES:
                    try:
                        datetime.strptime(value, DATETIME)
                    except ValueError:
                        found.append(f"{where}: {field} '{value}' is not "
                                     f"{DATETIME}")
                if field in BOOLEANS and value not in ("true", "false"):
                    found.append(f"{where}: {field} '{value}' is not a "
                                 "boolean")
                if field == "filePath" and not FILE_PATH.match(value):
                    found.append(f"{where}: filePath '{value}'")
                if field == "fileMediatype" and not MEDIA_TYPE.match(value):
                    found.append(f"{where}: fileMediatype '{value}'")

    deployments = {row["deploymentID"] for row in tables["deployments"]}
    media = {row["mediaID"] for row in tables["media"]}
    for name in ("media", "observations"):
        for row in tables[name]:
            if row["deploymentID"] not in deployments:
                found.append(f"{name}: deploymentID {row['deploymentID']} "
                             "is not a deployment")
    for row in tables["observations"]:
        if row["mediaID"] and row["mediaID"] not in media:
            found.append(f"observations: mediaID {row['mediaID']} is not a "
                         "media file")
        if row["observationLevel"] == "media" and not row["mediaID"]:
            found.append("observations: a media-based observation names "
                         "no media file")

    with open(os.path.join(folder, "datapackage.json"),
              encoding="utf-8") as fh:
        package = json.load(fh)
    for key in ("profile", "created", "contributors", "project", "spatial",
                "temporal", "taxonomic", "resources"):
        if key not in package:
            found.append(f"datapackage.json: {key} is required")
    if found and "resources" not in package:
        return found
    if package.get("profile") != f"{BASE}/camtrap-dp-profile.json":
        found.append("datapackage.json: profile")
    if not re.match(r"^([-a-z0-9._/])+$", package.get("name", "")):
        found.append(f"datapackage.json: name '{package.get('name')}'")
    try:
        datetime.strptime(package.get("created", ""), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        found.append(f"datapackage.json: created '{package.get('created')}'")
    for contributor in package.get("contributors", [{}]) or [{}]:
        if not contributor.get("title"):
            found.append("datapackage.json: a contributor needs a title")
        if contributor.get("role") not in (
                None, "contact", "principalInvestigator", "rightsHolder",
                "publisher", "contributor"):
            found.append("datapackage.json: contributor role")
    project = package.get("project", {})
    for key in ("title", "samplingDesign", "captureMethod",
                "individualAnimals", "observationLevel"):
        if key not in project:
            found.append(f"datapackage.json: project.{key} is required")
    if project.get("samplingDesign") not in (
            "simpleRandom", "systematicRandom", "clusteredRandom",
            "experimental", "targeted", "opportunistic"):
        found.append("datapackage.json: project.samplingDesign")
    if not project.get("captureMethod") or not set(
            project["captureMethod"]) <= set(ENUMS["captureMethod"]):
        found.append("datapackage.json: project.captureMethod")
    if not isinstance(project.get("individualAnimals"), bool):
        found.append("datapackage.json: project.individualAnimals")
    if not set(project.get("observationLevel", ["?"])) <= {"media", "event"}:
        found.append("datapackage.json: project.observationLevel")
    temporal = package.get("temporal", {})
    for key in ("start", "end"):
        try:
            datetime.strptime(temporal.get(key, ""), "%Y-%m-%d")
        except ValueError:
            found.append(f"datapackage.json: temporal.{key}")
    spatial = package.get("spatial", {})
    if spatial.get("type") != "Polygon" or not spatial.get("coordinates"):
        found.append("datapackage.json: spatial is not a GeoJSON polygon")
    else:
        ring = spatial["coordinates"][0]
        if ring[0] != ring[-1] or len(ring) < 4:
            found.append("datapackage.json: spatial ring is not closed")
    for taxon in package.get("taxonomic", []):
        if not taxon.get("scientificName"):
            found.append("datapackage.json: a taxon needs a scientificName")
        if taxon.get("taxonRank") not in (
                None, "kingdom", "phylum", "class", "order", "family",
                "genus", "species", "subspecies"):
            found.append("datapackage.json: taxonRank")
        for language in taxon.get("vernacularNames", {}):
            if not re.match(r"^[a-z]{3}$", language):
                found.append("datapackage.json: vernacularNames language")
    resources = {r.get("name"): r for r in package.get("resources", [])}
    for name in ("deployments", "media", "observations"):
        resource = resources.get(name)
        if resource is None:
            found.append(f"datapackage.json: no resource {name}")
            continue
        if resource.get("path") != f"{name}.csv" or \
                resource.get("profile") != "tabular-data-resource" or \
                resource.get("schema") != f"{BASE}/{name}-table-schema.json":
            found.append(f"datapackage.json: resource {name}")
    return found


# -- a flight -----------------------------------------------------------------

@pytest.fixture
def flight(tmp_path):
    """Two tracked animals on three frames each - the first named a female
    adult roe deer and approved, the second still "animal" -, a box in no
    track, and a box marked not-an-animal."""
    import numpy as np
    cv2 = pytest.importorskip("cv2")

    root = str(tmp_path / "North Meadow (2024)")
    os.makedirs(os.path.join(root, "frames_t"))
    for frame in range(3):
        cv2.imwrite(os.path.join(root, "frames_t", f"frame_{frame:06d}.jpg"),
                    np.zeros((512, 640, 3), dtype=np.uint8))
    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 64.0 + 320 * animal, "y1": 51.2,
         "x2": 128.0 + 320 * animal, "y2": 102.4, "confidence": 0.9,
         "source_class": "0"}
        for animal in range(2) for frame in range(3)]
        + [{"frame": 1, "x1": 400.0, "y1": 300.0, "x2": 420.0, "y2": 320.0,
            "confidence": 0.4, "source_class": "0"},
           {"frame": 2, "x1": 500.0, "y1": 400.0, "x2": 700.0, "y2": 420.0,
            "confidence": 1.7, "source_class": "0"}])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": i, "gx1": 500000.0 + 10 * n, "gy1": 5300000.0 + n,
         "gz1": 400.0, "gx2": 500002.0 + 10 * n, "gy2": 5300002.0 + n,
         "gz2": 400.0} for n, i in enumerate(ids[:7])])
    track_store.record_tracks(root, "t", [
        {"track_id": 1 + n // 3, "detection_id": i}
        for n, i in enumerate(ids[:6])])
    conn = store.open_store(store.stage_path(root, store.DETECTIONS, "t"),
                            store.DETECTIONS, "t")
    conn.execute("UPDATE detections SET species_id = ? WHERE detection_id = ?",
                 (common.NOT_AN_ANIMAL, ids[7]))
    conn.commit()
    conn.close()

    project = store.open_store(store.project_path(root), store.PROJECT)
    roe = project.execute(
        "SELECT species_id FROM species WHERE name = 'roe deer'"
    ).fetchone()["species_id"]
    project.close()
    conn = store.open_store(store.stage_path(root, store.TRACKS, "t"),
                            store.TRACKS, "t")
    first = conn.execute(
        "SELECT MIN(track_id) AS t FROM tracks").fetchone()["t"]
    conn.execute("UPDATE tracks SET species_id = ? WHERE track_id = ?",
                 (roe, first))
    conn.commit()
    conn.close()
    for field, value in (("sex", 1), ("age", 1), ("approved", True),
                         ("collar", "R|114")):
        track_store.set_track_attribute(root, "t", first, field, value)

    with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": [
            {"imagefile": f"frame_{frame:06d}.jpg",
             "timestamp": f"2024-05-29T23:59:5{8 + frame}+02:00"
             if frame < 2 else "2024-05-30T00:00:00+02:00"}
            for frame in range(3)]}, fh)
    return {"root": root, "first": first, "second": first + 1}


def _export(flight, tmp_path, **options):
    folder = str(tmp_path / "package")
    log = []
    exporters.export_camtrap_dp(flight["root"], "t", folder, epsg=32633,
                                log_fn=log.append, **options)
    return folder, log


def _observations(folder):
    return _table(folder, "observations")[1]


# -- the specification, kept ---------------------------------------------------

def test_the_package_meets_the_specification(flight, tmp_path):
    folder, _log = _export(flight, tmp_path)
    assert problems(folder) == []


def test_it_also_does_without_the_media(flight, tmp_path):
    folder, _log = _export(flight, tmp_path, include_images=False)
    assert problems(folder) == []
    assert not os.path.exists(os.path.join(folder, "media"))


def test_it_also_does_without_the_false_positives(flight, tmp_path):
    folder, _log = _export(flight, tmp_path, include_not_an_animal=False)
    assert problems(folder) == []
    assert "blank" not in {r["observationType"] for r in _observations(folder)}


def test_the_checker_itself_objects_to_the_old_export():
    """What the export used to write, so the checker is known to see it."""
    assert not FILE_PATH.match("../frames_t/a.jpg")
    assert not FILE_PATH.match("/abs/a.jpg")
    assert FILE_PATH.match("media/a.jpg")
    with pytest.raises(ValueError):
        datetime.strptime("2024-05-29T16:26:40", DATETIME)     # no zone


def test_the_tables_have_the_schemas_fields(flight, tmp_path):
    """Readers take the names from the schema and the values by position."""
    assert camtrap.DEPLOYMENT_FIELDS == DEPLOYMENTS
    assert camtrap.MEDIA_FIELDS == MEDIA
    assert camtrap.OBSERVATION_FIELDS == OBSERVATIONS
    folder, _log = _export(flight, tmp_path)
    assert tuple(_table(folder, "observations")[0]) == OBSERVATIONS
    assert tuple(_table(folder, "media")[0]) == MEDIA
    assert tuple(_table(folder, "deployments")[0]) == DEPLOYMENTS


# -- what the package says ------------------------------------------------------

def test_the_package_describes_the_survey(flight, tmp_path):
    folder, _log = _export(flight, tmp_path)
    with open(os.path.join(folder, "datapackage.json"),
              encoding="utf-8") as fh:
        package = json.load(fh)

    assert package["name"] == "north-meadow-2024"
    assert package["project"]["observationLevel"] == ["media", "event"]
    assert package["project"]["samplingDesign"] == "systematicRandom"
    assert package["temporal"] == {"start": "2024-05-29", "end": "2024-05-30"}
    assert [t["scientificName"] for t in package["taxonomic"]] == \
        ["Capreolus capreolus"]
    lons = [point[0] for point in package["spatial"]["coordinates"][0]]
    lats = [point[1] for point in package["spatial"]["coordinates"][0]]
    assert 14.9 < min(lons) <= max(lons) < 15.1
    assert 47.8 < min(lats) <= max(lats) < 47.9


def test_the_sampling_design_can_be_said(flight, tmp_path):
    folder, _log = _export(flight, tmp_path, sampling_design="opportunistic")
    with open(os.path.join(folder, "datapackage.json"),
              encoding="utf-8") as fh:
        assert json.load(fh)["project"]["samplingDesign"] == "opportunistic"
    with pytest.raises(common.ExportError, match="sampling design"):
        _export(flight, tmp_path, sampling_design="whenever")


def test_the_deployment_is_the_flight(flight, tmp_path):
    folder, _log = _export(flight, tmp_path)
    (deployment,) = _table(folder, "deployments")[1]

    assert deployment["deploymentID"] == "North Meadow (2024)"
    assert deployment["deploymentStart"] == "2024-05-29T23:59:58+02:00"
    assert deployment["deploymentEnd"] == "2024-05-30T00:00:00+02:00"
    # the animals are up to some 60 m apart: the location says how far
    assert 20 <= int(deployment["coordinateUncertainty"]) <= 60
    assert deployment["cameraModel"] == ""         # not something we know
    assert "camera:thermal" in deployment["deploymentTags"]


def test_media_say_what_and_where_the_files_are(flight, tmp_path):
    folder, _log = _export(flight, tmp_path)
    rows = _table(folder, "media")[1]

    assert [row["filePath"] for row in rows] == [
        f"media/frame_{frame:06d}.jpg" for frame in range(3)]
    assert {row["fileMediatype"] for row in rows} == {"image/jpeg"}
    assert {row["filePublic"] for row in rows} == {"false"}
    assert {row["captureMethod"] for row in rows} == {"timeLapse"}
    for row in rows:
        assert os.path.isfile(os.path.join(folder, *row["filePath"].split("/")))


def test_a_box_is_a_media_observation_with_its_place_on_the_frame(
        flight, tmp_path):
    folder, _log = _export(flight, tmp_path)
    boxes = [row for row in _observations(folder)
             if row["observationLevel"] == "media"]

    assert len(boxes) == 8
    first = boxes[0]
    assert first["eventStart"] == first["eventEnd"] == \
        "2024-05-29T23:59:58+02:00"
    # 64..128 of 640 across, 51.2..102.4 of 512 down
    assert [float(first[key]) for key in
            ("bboxX", "bboxY", "bboxWidth", "bboxHeight")] == \
        pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert "latitude:47." in first["observationTags"]
    assert "longitude:1" in first["observationTags"]


def test_a_box_reaching_out_of_the_frame_is_cut_to_it(flight, tmp_path):
    """500..700 of a 640 px frame: the specification's box ends at 1."""
    folder, _log = _export(flight, tmp_path)
    blank = [row for row in _observations(folder)
             if row["observationType"] == "blank"][0]
    assert float(blank["bboxX"]) + float(blank["bboxWidth"]) == \
        pytest.approx(1.0)


def test_an_animal_is_an_event_observation(flight, tmp_path):
    folder, _log = _export(flight, tmp_path)
    events = {row["eventID"]: row for row in _observations(folder)
              if row["observationLevel"] == "event"}
    first = events[f"North Meadow (2024)-track-{flight['first']}"]
    second = events[f"North Meadow (2024)-track-{flight['second']}"]

    assert len(events) == 2
    assert first["mediaID"] == ""
    assert (first["eventStart"], first["eventEnd"]) == (
        "2024-05-29T23:59:58+02:00", "2024-05-30T00:00:00+02:00")
    assert (first["scientificName"], first["sex"], first["lifeStage"],
            first["count"]) == ("Capreolus capreolus", "female", "adult", "1")
    assert "boxes:3" in first["observationTags"]
    # nobody has named the second animal: no scientific name to give, and
    # the plugin's own word for it is kept as a tag
    assert second["scientificName"] == ""
    assert "species:animal" in second["observationTags"]


def test_who_classified_it_is_said(flight, tmp_path):
    """An approved animal was confirmed by a person; the rest is the
    software's, with the detector's confidence."""
    folder, _log = _export(flight, tmp_path)
    rows = _observations(folder)
    of_first = [r for r in rows if r["eventID"].endswith(
        f"-track-{flight['first']}")]
    of_second = [r for r in rows if r["eventID"].endswith(
        f"-track-{flight['second']}") and r["observationLevel"] == "media"]

    assert {(r["classificationMethod"], r["classifiedBy"],
             r["classificationProbability"]) for r in of_first} == \
        {("human", "", "")}
    assert {(r["classificationMethod"], r["classifiedBy"],
             r["classificationProbability"]) for r in of_second} == \
        {("machine", "BAMBI QGIS plugin", "0.9000")}


def test_a_confidence_above_one_is_not_a_probability_above_one(flight,
                                                               tmp_path):
    folder, _log = _export(flight, tmp_path)
    values = [float(r["classificationProbability"])
              for r in _observations(folder)
              if r["classificationProbability"]]
    assert max(values) == 1.0


def test_custom_fields_are_tags_that_cannot_break_the_list(flight, tmp_path):
    folder, _log = _export(flight, tmp_path)
    first = [r for r in _observations(folder) if r["eventID"].endswith(
        f"-track-{flight['first']}")][0]
    listed = first["observationTags"].split(" | ")
    assert "collar:R/114" in listed
    assert not any(tag.startswith(("sex:", "age:", "approved:"))
                   for tag in listed)


# -- what cannot be written truthfully is not written ---------------------------

def test_frames_without_a_capture_time_are_refused(flight, tmp_path):
    root = flight["root"]
    with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": [{"imagefile": f"frame_{frame:06d}.jpg"}
                              for frame in range(3)]}, fh)
    with pytest.raises(common.ExportError, match="capture time"):
        _export(flight, tmp_path)


def test_a_time_without_a_zone_is_refused(flight, tmp_path):
    """The specification asks for the offset; guessing it moves every
    observation by hours."""
    root = flight["root"]
    with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": [
            {"imagefile": f"frame_{frame:06d}.jpg",
             "timestamp": f"2024-05-29T16:26:4{frame}"}
            for frame in range(3)]}, fh)
    with pytest.raises(common.ExportError, match="time zone"):
        _export(flight, tmp_path)


def test_a_flight_without_any_position_is_refused(flight, tmp_path):
    """The deployment's location was written as latitude 0, longitude 0."""
    os.remove(store.stage_path(flight["root"], store.GEOREFERENCED, "t"))
    with pytest.raises(common.ExportError, match="has a position"):
        _export(flight, tmp_path)


def test_the_log_says_what_is_left_to_the_user(flight, tmp_path):
    _folder, log = _export(flight, tmp_path)
    assert any("contributors" in line for line in log)
    assert any("8 box(es) and 2 animal(s)" in line for line in log)


# -- the small pieces -----------------------------------------------------------

def test_package_names():
    assert camtrap.package_name("North Meadow (2024)") == "north-meadow-2024"
    assert camtrap.package_name("flight_07.b") == "flight_07.b"
    assert camtrap.package_name("  ") == "bambi-survey"
    assert camtrap.package_name("Übung") == "bung"


def test_media_types():
    assert camtrap.media_type("a.JPG") == "image/jpeg"
    assert camtrap.media_type("a.png") == "image/png"
    assert camtrap.media_type("a.tiff") == "image/tiff"


def test_tags():
    assert camtrap.tags([("a", 1), ("b", ""), ("c", None), ("d", True),
                         ("e", "x|y")]) == "a:1 | d:true | e:x/y"
    assert camtrap.tags([]) == ""


def test_relative_boxes():
    row = {"x1": 64.0, "y1": 0.0, "x2": 128.0, "y2": 256.0}
    assert camtrap.relative_box(row, (640, 512)) == \
        ("0.100000", "0.000000", "0.100000", "0.500000")
    assert camtrap.relative_box(row, None) == ("", "", "", "")
    flat = {"x1": 10.0, "y1": 5.0, "x2": 10.0, "y2": 9.0}
    assert camtrap.relative_box(flat, (640, 512)) == ("", "", "", "")
    outside = {"x1": 700.0, "y1": 5.0, "x2": 800.0, "y2": 9.0}
    assert camtrap.relative_box(outside, (640, 512)) == ("", "", "", "")
