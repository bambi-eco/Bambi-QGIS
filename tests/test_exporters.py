# -*- coding: utf-8 -*-
"""Exporter tests (EXCHANGE_FORMAT_PLAN.md §8.1).

Three things are checked for every format: that internal ids never leak (they
are sparse and negative), that enum values arrive as labels rather than
integers, and that ``not-an-animal`` is handled according to what the format is
*for* rather than filtered silently.
"""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.core import (
    detection_store, exporters, label_store, store, track_store)
from bambi_wildlife_detection.core.exporters import common


@pytest.fixture
def survey(tmp_path):
    """A project with three tracked animals, one of them a false positive."""
    root = str(tmp_path)

    with_species = store.open_store(store.project_path(root), store.PROJECT)
    with_species.close()

    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "1"},
        {"frame": 1, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.8, "source_class": "1"},
        {"frame": 0, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
         "confidence": 0.7, "source_class": "5"},
        {"frame": 1, "x1": 90.0, "y1": 90.0, "x2": 95.0, "y2": 95.0,
         "confidence": 0.3, "source_class": "0"},
    ])

    project = store.open_store(store.project_path(root), store.PROJECT)
    source_id = project.execute(
        "SELECT source_id FROM detection_sources").fetchone()["source_id"]
    project.executemany(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, ?, ?)",
        [(source_id, "1", 1), (source_id, "5", 5), (source_id, "0", -2)])
    project.commit()
    project.close()

    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "1",
         "attributes": json.dumps({"occlusion": 1, "collar": "R-114"})},
        {"frame": 1, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.8, "source_class": "1"},
        {"frame": 0, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
         "confidence": 0.7, "source_class": "5"},
        {"frame": 1, "x1": 90.0, "y1": 90.0, "x2": 95.0, "y2": 95.0,
         "confidence": 0.3, "source_class": "0"},
    ])

    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": i, "gx1": 500000.0 + n, "gy1": 5300000.0 + n,
         "gz1": 400.0, "gx2": 500010.0 + n, "gy2": 5300010.0 + n, "gz2": 400.0}
        for n, i in enumerate(ids)])
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 2, "detection_id": ids[2]},
        {"track_id": 3, "detection_id": ids[3]},
    ])

    with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": [
            {"imagefile": "frame_000000.jpg", "epoch": 1717000000},
            {"imagefile": "frame_000001.jpg", "epoch": 1717000001},
        ]}, fh)
    return root


SIZE = (640, 512)


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------

def test_class_map_is_contiguous_and_non_negative(survey):
    vocabulary = common.load_vocabulary(survey)
    rows = common.load_detections(survey, "t", include_not_an_animal=True)
    mapping, names = common.class_map(
        (r["species_id"] for r in rows), vocabulary["species"])
    assert sorted(mapping.values()) == list(range(len(mapping)))
    assert all(index >= 0 for index in mapping.values())
    assert "not-an-animal" in names


def test_not_an_animal_is_excluded_by_default(survey):
    rows = common.load_detections(survey, "t")
    assert all(r["species_id"] != common.NOT_AN_ANIMAL for r in rows)
    assert len(rows) == 3


def test_not_an_animal_can_be_included(survey):
    assert len(common.load_detections(survey, "t", True)) == 4


def test_attributes_resolve_enum_values_to_labels(survey):
    vocabulary = common.load_vocabulary(survey)
    rows = common.load_detections(survey, "t")
    attributes = common.resolve_attributes(
        rows[0]["attributes"], vocabulary["enum_labels"])
    assert attributes["occlusion"] == "occluded"
    assert attributes["collar"] == "R-114"


def test_export_without_a_store_is_refused(tmp_path):
    with pytest.raises(common.ExportError, match="no 6.0 store"):
        common.load_vocabulary(str(tmp_path))


def test_registry_lists_every_format():
    assert set(exporters.EXPORTERS) == {
        "coco", "yolo", "mot", "trex", "geojson", "geojson_segmentation",
        "camtrap", "dwca"}


def test_the_two_geojson_formats_say_what_they_hold():
    """'geo-referenced' described how they were made, not what is in them —
    which is no help when choosing between two of them."""
    labels = {key: exporters.EXPORTERS[key][0]
              for key in ("geojson", "geojson_segmentation")}
    assert labels["geojson"] == "GeoJSON (animals)"
    assert labels["geojson_segmentation"] == "GeoJSON (segmentations)"


def test_both_geojson_formats_get_the_project_crs():
    for key in ("geojson", "geojson_segmentation"):
        assert key in exporters.NEEDS_CRS


def test_each_single_file_format_offers_a_distinct_name():
    names = [name for key, name in exporters.DEFAULT_FILENAME.items()]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------

def test_coco_structure(survey, tmp_path):
    path = str(tmp_path / "out" / "coco.json")
    exporters.export_coco(survey, "t", path, image_size=SIZE)
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)

    assert len(document["annotations"]) == 3
    assert {c["id"] for c in document["categories"]} == {0, 1}
    assert document["images"][0]["width"] == 640
    assert all(a["bbox"][2] > 0 for a in document["annotations"])


def test_coco_carries_custom_fields(survey, tmp_path):
    """The complaint that started the rework: fields could not leave labels.json."""
    path = str(tmp_path / "coco.json")
    exporters.export_coco(survey, "t", path, image_size=SIZE)
    with open(path, encoding="utf-8") as fh:
        annotations = json.load(fh)["annotations"]
    carried = [a for a in annotations if "attributes" in a]
    assert carried and carried[0]["attributes"]["occlusion"] == "occluded"


def test_coco_categories_use_species_names(survey, tmp_path):
    path = str(tmp_path / "coco.json")
    exporters.export_coco(survey, "t", path, image_size=SIZE)
    with open(path, encoding="utf-8") as fh:
        names = {c["name"] for c in json.load(fh)["categories"]}
    assert names == {"roe deer", "chamois"}


def test_coco_includes_track_ids(survey, tmp_path):
    path = str(tmp_path / "coco.json")
    exporters.export_coco(survey, "t", path, image_size=SIZE)
    with open(path, encoding="utf-8") as fh:
        annotations = json.load(fh)["annotations"]
    assert all("track_id" in a for a in annotations)


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

def test_yolo_writes_normalised_boxes(survey, tmp_path):
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE)
    label = os.path.join(folder, "labels", "frame_000000.txt")
    with open(label, encoding="utf-8") as fh:
        parts = fh.readline().split()
    assert len(parts) == 5
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_yolo_data_yaml_lists_the_classes(survey, tmp_path):
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE)
    with open(os.path.join(folder, "data.yaml"), encoding="utf-8") as fh:
        content = fh.read()
    assert "nc: 2" in content
    assert "roe deer" in content


def test_yolo_omits_false_positives(survey, tmp_path):
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE)
    labels = os.listdir(os.path.join(folder, "labels"))
    total = 0
    for name in labels:
        with open(os.path.join(folder, "labels", name), encoding="utf-8") as fh:
            total += len([line for line in fh if line.strip()])
    assert total == 3


def test_yolo_needs_a_frame_size(survey, tmp_path):
    with pytest.raises(common.ExportError, match="frame size"):
        exporters.export_yolo(survey, "t", str(tmp_path / "yolo"))


def _with_frames(root, names=("frame_000000.jpg", "frame_000001.jpg")):
    """Give the survey the extracted frames its detections refer to."""
    folder = os.path.join(root, "frames_t")
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(b"not really a jpeg, but it only has to be copied")
    return folder


def test_yolo_copies_the_images_the_labels_refer_to(survey, tmp_path):
    """Without them the export is unusable: a YOLO dataset is a folder
    layout, and nothing in it names the images."""
    _with_frames(survey)
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE)

    assert sorted(os.listdir(os.path.join(folder, "images"))) == [
        "frame_000000.jpg", "frame_000001.jpg"]


def test_yolo_pairs_every_label_with_an_image(survey, tmp_path):
    """YOLO finds a label by swapping "images" for "labels" in the path, so
    the stems have to match exactly."""
    _with_frames(survey)
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE)

    images = {os.path.splitext(n)[0]
              for n in os.listdir(os.path.join(folder, "images"))}
    labels = {os.path.splitext(n)[0]
              for n in os.listdir(os.path.join(folder, "labels"))}
    assert labels and labels == images


def test_yolo_yaml_points_at_the_folder_it_wrote(survey, tmp_path):
    """It used to name images_t, which was never created."""
    _with_frames(survey)
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE)

    with open(os.path.join(folder, "data.yaml"), encoding="utf-8") as fh:
        content = fh.read()
    lines = content.splitlines()
    assert "train: images" in lines
    assert "images_t" not in content

    for line in lines:
        if line.startswith(("path:", "train:", "val:")):
            value = line.partition(": ")[2].strip()
            if value:
                assert os.path.isdir(os.path.join(folder, value))


def test_yolo_can_skip_copying(survey, tmp_path):
    _with_frames(survey)
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE,
                          include_images=False)

    assert os.listdir(os.path.join(folder, "images")) == []
    assert os.listdir(os.path.join(folder, "labels"))


def test_yolo_reports_frames_it_could_not_find(survey, tmp_path):
    """Labels without an image are silently ignored by YOLO, so the export
    has to say so rather than look complete."""
    _with_frames(survey, names=("frame_000000.jpg",))
    folder = str(tmp_path / "yolo")
    messages = []
    exporters.export_yolo(survey, "t", folder, image_size=SIZE,
                          log_fn=messages.append)

    assert any("frame_000001.jpg" in m for m in messages)
    assert any("copied 1 image" in m for m in messages)


def test_yolo_copies_nothing_it_was_not_asked_for(survey, tmp_path):
    """Only frames carrying a detection — not the whole flight."""
    _with_frames(survey, names=("frame_000000.jpg", "frame_000001.jpg",
                                "frame_000002.jpg", "frame_000003.jpg"))
    folder = str(tmp_path / "yolo")
    exporters.export_yolo(survey, "t", folder, image_size=SIZE)

    assert len(os.listdir(os.path.join(folder, "images"))) == 2


# ---------------------------------------------------------------------------
# MOT
# ---------------------------------------------------------------------------

def test_mot_frames_are_one_based(survey, tmp_path):
    folder = str(tmp_path / "mot")
    exporters.export_mot(survey, "t", folder)
    with open(os.path.join(folder, "gt.txt"), encoding="utf-8") as fh:
        first = fh.readline().split(",")
    assert int(first[0]) == 1          # store frame 0
    assert float(first[4]) > 0         # width, not x2


def test_mot_sidecar_carries_what_the_columns_cannot(survey, tmp_path):
    folder = str(tmp_path / "mot")
    exporters.export_mot(survey, "t", folder)
    with open(os.path.join(folder, "attributes.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["species"] == "roe deer"
    assert rows[0]["occlusion"] == "occluded"


def test_mot_classes_file_is_written(survey, tmp_path):
    folder = str(tmp_path / "mot")
    exporters.export_mot(survey, "t", folder)
    with open(os.path.join(folder, "classes.txt"), encoding="utf-8") as fh:
        assert fh.readline().startswith("0 ")


def test_mot_omits_untracked_detections(survey, tmp_path):
    detection_store.record_detections(survey, "t", [
        {"frame": 5, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "1"}], kind=detection_store.MANUAL)
    folder = str(tmp_path / "mot")
    messages = []
    exporters.export_mot(survey, "t", folder, log_fn=messages.append)
    assert any("untracked" in m for m in messages)


# ---------------------------------------------------------------------------
# GeoJSON — segmentations
# ---------------------------------------------------------------------------

def _add_segments(root, rows):
    conn = store.open_store(
        store.stage_path(root, store.SEGMENTATION, "t"),
        store.SEGMENTATION, "t")
    with store.transaction(conn):
        for row in rows:
            conn.execute(
                "INSERT INTO segments (detection_id, frame, polygon_px, "
                "polygon_geo, attributes) VALUES (?, ?, ?, ?, ?)",
                (row.get("detection_id"), row["frame"],
                 json.dumps(row.get("polygon_px") or []),
                 json.dumps(row["polygon_geo"])
                 if row.get("polygon_geo") is not None else None,
                 json.dumps(row.get("attributes") or {})))
    conn.close()


SQUARE = [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]


def test_segmentation_geojson_writes_polygons(survey, tmp_path):
    _add_segments(survey, [{"detection_id": 1, "frame": 0,
                            "polygon_geo": SQUARE}])
    path = str(tmp_path / "segmentations.geojson")
    exporters.export_segmentation_geojson(survey, "t", path, epsg=32633)

    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    assert len(document["features"]) == 1
    feature = document["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["detection_id"] == 1
    assert "EPSG::32633" in document["crs"]["properties"]["name"]


def test_segmentation_rings_are_closed(survey, tmp_path):
    """GeoJSON polygons must close; the store need not."""
    _add_segments(survey, [{"frame": 0, "polygon_geo": SQUARE}])
    path = str(tmp_path / "s.geojson")
    exporters.export_segmentation_geojson(survey, "t", path)

    ring = json.load(open(path, encoding="utf-8"))[
        "features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert len(ring) == len(SQUARE) + 1


def test_an_already_nested_ring_list_is_accepted(survey, tmp_path):
    _add_segments(survey, [{"frame": 0, "polygon_geo": [SQUARE]}])
    path = str(tmp_path / "s.geojson")
    exporters.export_segmentation_geojson(survey, "t", path)

    rings = json.load(open(path, encoding="utf-8"))[
        "features"][0]["geometry"]["coordinates"]
    assert len(rings) == 1 and len(rings[0]) == len(SQUARE) + 1


def test_masks_without_world_coordinates_are_reported(survey, tmp_path):
    """A pixel-space polygon has no place in a world-coordinate document."""
    _add_segments(survey, [{"frame": 0, "polygon_geo": None,
                            "polygon_px": SQUARE},
                           {"frame": 1, "polygon_geo": SQUARE}])
    logs = []
    path = str(tmp_path / "s.geojson")
    exporters.export_segmentation_geojson(survey, "t", path,
                                          log_fn=logs.append)

    assert len(json.load(open(path, encoding="utf-8"))["features"]) == 1
    assert any("Geo-Reference Segmentation" in line for line in logs)


def test_a_degenerate_polygon_is_skipped(survey, tmp_path):
    _add_segments(survey, [{"frame": 0, "polygon_geo": [[1.0, 2.0]]}])
    path = str(tmp_path / "s.geojson")
    exporters.export_segmentation_geojson(survey, "t", path)
    assert json.load(open(path, encoding="utf-8"))["features"] == []


def test_segment_attributes_reach_the_properties(survey, tmp_path):
    _add_segments(survey, [{"frame": 0, "polygon_geo": SQUARE,
                            "attributes": {"prompt": "deer", "score": 0.8}}])
    path = str(tmp_path / "s.geojson")
    exporters.export_segmentation_geojson(survey, "t", path)

    properties = json.load(open(path, encoding="utf-8"))[
        "features"][0]["properties"]
    assert properties["prompt"] == "deer"
    assert properties["score"] == 0.8


def test_exporting_without_any_segmentation_says_so(survey, tmp_path):
    with pytest.raises(exporters.ExportError, match="No segmentation"):
        exporters.export_segmentation_geojson(
            survey, "t", str(tmp_path / "s.geojson"))


def test_the_two_geojson_exports_are_separate_documents(survey, tmp_path):
    """Different geometries answering different questions — one mixed
    collection could not be styled sensibly in any GIS."""
    _add_segments(survey, [{"frame": 0, "polygon_geo": SQUARE}])
    animals = str(tmp_path / "animals.geojson")
    segments = str(tmp_path / "segmentations.geojson")
    exporters.export_geojson(survey, "t", animals)
    exporters.export_segmentation_geojson(survey, "t", segments)

    kinds = set()
    for path in (animals, segments):
        for feature in json.load(open(path, encoding="utf-8"))["features"]:
            kinds.add(feature["geometry"]["type"])
    assert "Polygon" in kinds and "Point" in kinds


# ---------------------------------------------------------------------------
# GeoJSON — animals
# ---------------------------------------------------------------------------

def test_geojson_points(survey, tmp_path):
    path = str(tmp_path / "detections.geojson")
    exporters.export_geojson(survey, "t", path, epsg=32633)
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    assert document["crs"]["properties"]["name"].endswith("32633")
    assert all(f["geometry"]["type"] == "Point" for f in document["features"])


def test_geojson_keeps_false_positives(survey, tmp_path):
    """A survey record says what was rejected; it does not omit it."""
    path = str(tmp_path / "detections.geojson")
    exporters.export_geojson(survey, "t", path)
    with open(path, encoding="utf-8") as fh:
        species = {f["properties"]["species"] for f in json.load(fh)["features"]}
    assert "not-an-animal" in species


def test_geojson_tracks_are_linestrings(survey, tmp_path):
    path = str(tmp_path / "tracks.geojson")
    exporters.export_geojson(survey, "t", path, tracks_only=True)
    with open(path, encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    assert len(features) == 1          # only track 1 has two points
    assert features[0]["geometry"]["type"] == "LineString"
    assert features[0]["properties"]["n_detections"] == 2


# ---------------------------------------------------------------------------
# TRex
# ---------------------------------------------------------------------------

def test_trex_writes_one_file_per_track(survey, tmp_path):
    folder = str(tmp_path / "trex")
    written = exporters.export_trex_npz(survey, "t", folder)
    assert len(written) == 2          # the false-positive track is excluded


def test_trex_round_trips_through_numpy(survey, tmp_path):
    import numpy as np

    folder = str(tmp_path / "trex")
    written = exporters.export_trex_npz(survey, "t", folder)
    data = np.load(sorted(written)[0])
    assert list(data["frame"]) == [0, 1]
    assert data["X"].shape == (2, 2)


def test_trex_without_tracks_is_refused(tmp_path):
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "1"}])
    with pytest.raises(common.ExportError, match="No tracks"):
        exporters.export_trex_npz(root, "t", str(tmp_path / "trex"))


# ---------------------------------------------------------------------------
# Camtrap DP
# ---------------------------------------------------------------------------

def test_camtrap_writes_the_three_resources(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633)
    for name in ("deployments.csv", "media.csv", "observations.csv",
                 "datapackage.json"):
        assert os.path.isfile(os.path.join(folder, name))


def test_camtrap_observations_carry_species_and_fields(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "observations.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    animals = [r for r in rows if r["observationType"] == "animal"]
    assert animals[0]["scientificName"] == "Capreolus capreolus"
    assert animals[0]["vernacularName"] == "roe deer"
    assert animals[0]["occlusion"] == "occluded"


def test_camtrap_records_false_positives_as_blank(survey, tmp_path):
    """A rejected detection is a survey record, not noise."""
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "observations.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    blanks = [r for r in rows if r["observationType"] == "blank"]
    assert len(blanks) == 1
    assert blanks[0]["scientificName"] == ""


def test_camtrap_coordinates_are_latitude_longitude(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "observations.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    latitudes = [float(r["latitude"]) for r in rows if r["latitude"]]
    assert all(-90 <= lat <= 90 for lat in latitudes)
    assert all(45 < lat < 55 for lat in latitudes)     # UTM 33N, ~5300 km north


def test_camtrap_media_reference_the_frames(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "media.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["fileName"] for r in rows} == \
        {"frame_000000.jpg", "frame_000001.jpg"}


def test_camtrap_without_a_crs_is_refused(survey, tmp_path):
    with pytest.raises(common.ExportError, match="project CRS"):
        exporters.export_camtrap_dp(survey, "t", str(tmp_path / "c"))


# ---------------------------------------------------------------------------
# Darwin Core
# ---------------------------------------------------------------------------

def test_dwca_is_one_occurrence_per_track(survey, tmp_path):
    """Not per detection — that would publish one animal hundreds of times."""
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "dwca")
    exporters.export_darwin_core(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "occurrence.txt"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 2
    assert {r["scientificName"] for r in rows} == \
        {"Capreolus capreolus", "Rupicapra rupicapra"}


def test_dwca_marks_machine_observations(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "dwca")
    exporters.export_darwin_core(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "occurrence.txt"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert all(r["basisOfRecord"] == "MachineObservation" for r in rows)
    assert all(r["geodeticDatum"] == "WGS84" for r in rows)


def test_dwca_meta_xml_describes_the_columns(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "dwca")
    exporters.export_darwin_core(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "meta.xml"), encoding="utf-8") as fh:
        meta = fh.read()
    assert "occurrence.txt" in meta
    assert "decimalLatitude" in meta


def test_dwca_reports_tracks_it_could_not_publish(survey, tmp_path):
    """'animal' and 'unknown' are not taxa, so GBIF cannot take them."""
    pytest.importorskip("pyproj")
    detection_store.record_detections(survey, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "unmapped"}],
        kind=detection_store.MANUAL)
    ids = [d["detection_id"] for d in track_store.load_detections(survey, "t")]
    track_store.record_georeference(survey, "t", [
        {"detection_id": i, "gx1": 500000.0, "gy1": 5300000.0, "gz1": 400.0,
         "gx2": 500010.0, "gy2": 5300010.0, "gz2": 400.0} for i in ids])
    track_store.record_tracks(survey, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 9, "detection_id": ids[-1]},
    ])

    messages = []
    exporters.export_darwin_core(
        survey, "t", str(tmp_path / "dwca"), epsg=32633, log_fn=messages.append)
    assert any("no species assigned" in m for m in messages)


def test_dwca_refuses_when_nothing_is_identified(tmp_path):
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "0"}])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": ids[0], "gx1": 5.0, "gy1": 6.0, "gz1": 0.0,
         "gx2": 7.0, "gy2": 8.0, "gz2": 0.0}])
    track_store.record_tracks(
        root, "t", [{"track_id": 1, "detection_id": ids[0]}])

    with pytest.raises(common.ExportError, match="assign species"):
        exporters.export_darwin_core(
            root, "t", str(tmp_path / "dwca"), epsg=32633)


# ---------------------------------------------------------------------------
# Manual tracks reach the exports (§6.5)
# ---------------------------------------------------------------------------

def test_manual_tracks_are_exported_too(survey, tmp_path):
    label_store.save_tracks(survey, "t", [{
        "label_track_id": 1, "species_id": 1,
        "keyframes": [{"frame": 0, "x1": 5.0, "y1": 5.0, "x2": 9.0, "y2": 9.0}],
    }])
    label_store.materialise(survey, "t")

    path = str(tmp_path / "coco.json")
    exporters.export_coco(survey, "t", path, image_size=SIZE)
    with open(path, encoding="utf-8") as fh:
        annotations = json.load(fh)["annotations"]
    assert len(annotations) == 4      # three detector + one manual


# ---------------------------------------------------------------------------
# Taxonomy: scientific names and GBIF keys (§8.1)
# ---------------------------------------------------------------------------

def _set_taxonomy(root, species_id, scientific_name, rank="species", key=None):
    from bambi_wildlife_detection.core.schema_editor import SchemaEditor

    with SchemaEditor(root) as editor:
        editor.set_taxonomy(species_id, scientific_name, rank, key)
        editor.commit()


def test_seeded_species_carry_scientific_names(survey):
    """The 5.x labels are vernacular; publishing needs the scientific name."""
    vocabulary = common.load_vocabulary(survey)
    assert vocabulary["taxonomy"][1]["scientific_name"] == "Capreolus capreolus"
    assert vocabulary["taxonomy"][5]["scientific_name"] == "Rupicapra rupicapra"


def test_no_gbif_keys_are_invented(survey):
    """An invented identifier would publish confidently wrong data."""
    vocabulary = common.load_vocabulary(survey)
    assert all(entry["gbif_taxon_key"] is None
               for entry in vocabulary["taxonomy"].values())


def test_dwca_publishes_the_scientific_name(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "dwca")
    exporters.export_darwin_core(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "occurrence.txt"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert {r["scientificName"] for r in rows} == \
        {"Capreolus capreolus", "Rupicapra rupicapra"}
    assert {r["vernacularName"] for r in rows} == {"roe deer", "chamois"}


def test_dwca_uses_the_gbif_key_when_present(survey, tmp_path):
    pytest.importorskip("pyproj")
    _set_taxonomy(survey, 1, "Capreolus capreolus", "species", 2440947)

    folder = str(tmp_path / "dwca")
    messages = []
    exporters.export_darwin_core(
        survey, "t", folder, epsg=32633, log_fn=messages.append)
    with open(os.path.join(folder, "occurrence.txt"), encoding="utf-8") as fh:
        rows = {r["scientificName"]: r
                for r in csv.DictReader(fh, delimiter="\t")}

    assert rows["Capreolus capreolus"]["taxonID"] == \
        "https://www.gbif.org/species/2440947"
    assert rows["Rupicapra rupicapra"]["taxonID"] == ""
    assert any("carry a GBIF taxon key" in m for m in messages)


def test_dwca_holds_back_species_without_a_scientific_name(survey, tmp_path):
    pytest.importorskip("pyproj")
    _set_taxonomy(survey, 5, "")

    folder = str(tmp_path / "dwca")
    messages = []
    exporters.export_darwin_core(
        survey, "t", folder, epsg=32633, log_fn=messages.append)
    with open(os.path.join(folder, "occurrence.txt"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert len(rows) == 1
    assert any("no scientific name" in m and "chamois" in m for m in messages)


def test_dwca_reports_the_rank(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "dwca")
    exporters.export_darwin_core(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "occurrence.txt"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert all(r["taxonRank"] == "species" for r in rows)


def test_camtrap_separates_scientific_from_vernacular(survey, tmp_path):
    pytest.importorskip("pyproj")
    _set_taxonomy(survey, 1, "Capreolus capreolus", "species", 2440947)

    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633)
    with open(os.path.join(folder, "observations.csv"), encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["observationType"] == "animal"]

    roe = [r for r in rows if r["vernacularName"] == "roe deer"][0]
    assert roe["scientificName"] == "Capreolus capreolus"
    assert roe["taxonID"] == "https://www.gbif.org/species/2440947"


# ---------------------------------------------------------------------------
# The dispatch used by the export UI
# ---------------------------------------------------------------------------

def test_run_export_rejects_an_unknown_format(survey, tmp_path):
    with pytest.raises(common.ExportError, match="Unknown export format"):
        exporters.run_export("parquet", survey, "t", str(tmp_path))


def test_run_export_passes_the_crs_where_it_is_needed(survey, tmp_path):
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "camtrap")
    exporters.run_export("camtrap", survey, "t", folder, epsg=32633)
    assert os.path.isfile(os.path.join(folder, "observations.csv"))


def test_run_export_does_not_pass_a_crs_to_formats_without_one(survey, tmp_path):
    """COCO has no coordinate system; passing epsg would be a TypeError."""
    path = str(tmp_path / "coco.json")
    exporters.run_export("coco", survey, "t", path, epsg=32633,
                         image_size=SIZE)
    assert os.path.isfile(path)


def test_run_export_says_what_is_missing_without_frames(survey, tmp_path):
    """A clear message beats a traceback when frames were never extracted."""
    with pytest.raises(common.ExportError, match="Extract frames first"):
        exporters.run_export("coco", survey, "t", str(tmp_path / "coco.json"))


@pytest.mark.parametrize("key", ["coco", "yolo", "mot", "trex"])
def test_training_formats_drop_false_positives_by_default(key, survey, tmp_path):
    assert key in exporters.TRAINING_FORMATS


@pytest.mark.parametrize("key", ["geojson", "camtrap"])
def test_survey_formats_keep_false_positives_by_default(key):
    assert key not in exporters.TRAINING_FORMATS


def test_darwin_core_ignores_the_false_positive_option(survey, tmp_path):
    """A rejected detection is not an occurrence, whatever the caller asks."""
    pytest.importorskip("pyproj")
    folder = str(tmp_path / "dwca")
    exporters.run_export("dwca", survey, "t", folder, epsg=32633,
                         include_not_an_animal=True)
    with open(os.path.join(folder, "occurrence.txt"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert all(r["scientificName"] for r in rows)


def test_every_registered_format_declares_its_output_shape():
    for key, (label, function, is_folder) in exporters.EXPORTERS.items():
        assert label and callable(function)
        assert isinstance(is_folder, bool)
        if not is_folder:
            assert key in exporters.DEFAULT_FILENAME


# ---------------------------------------------------------------------------
# Why a track export came back empty
# ---------------------------------------------------------------------------

def _untracked(tmp_path):
    """Detections in the store, but no tracking run — the state a project is
    left in when a tracker read the legacy text files instead."""
    root = str(tmp_path / "untracked")
    os.makedirs(root, exist_ok=True)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.9, "source_class": "0"},
    ])
    return root


def test_mot_explains_an_empty_export(tmp_path):
    """"0 rows" is true but useless — the usual cause is a tracking run that
    never reached the store."""
    root = _untracked(tmp_path)
    messages = []
    exporters.export_mot(root, "t", str(tmp_path / "mot"),
                         log_fn=messages.append)

    assert any("0 row(s)" in m for m in messages)
    assert any("Run tracking first" in m for m in messages)


def test_a_successful_export_is_not_second_guessed(survey, tmp_path):
    messages = []
    exporters.export_mot(survey, "t", str(tmp_path / "mot"),
                         log_fn=messages.append)
    assert not any("Run tracking" in m or "Re-run" in m for m in messages)


def test_the_hint_names_the_legacy_tracks_file(tmp_path):
    """It is what makes tracking look as though it worked."""
    root = _untracked(tmp_path)
    folder = os.path.join(root, "tracks_t")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "tracks.csv"), "w", encoding="utf-8") as fh:
        fh.write("0,1,2,3")

    hint = common.no_tracks_hint(root, "t")
    assert "tracks.csv" in hint
    assert "Geo-reference" in hint


def test_the_hint_without_a_tracks_file_says_run_tracking(tmp_path):
    assert "Run tracking first" in common.no_tracks_hint(
        str(tmp_path / "empty"), "t")


def test_the_hint_notices_a_run_that_does_not_cover_these_rows(survey):
    """A run exists, so the advice is to re-run tracking, not to run it."""
    assert "Re-run tracking" in common.no_tracks_hint(survey, "t")


def test_geojson_reports_detections_that_were_never_geo_referenced(tmp_path):
    """The state the user's project was in: detections but no geo store."""
    root = _untracked(tmp_path)
    messages = []
    exporters.export_geojson(root, "t", str(tmp_path / "t.geojson"),
                             log_fn=messages.append)
    assert any("Geo-reference" in m for m in messages)


def test_geojson_tracks_explain_an_empty_export(survey, tmp_path):
    """Geo-referenced, but the tracking run is gone."""
    trk = store.stage_path(survey, store.TRACKS, "t")
    os.remove(trk)

    messages = []
    exporters.export_geojson(survey, "t", str(tmp_path / "t.geojson"),
                             tracks_only=True, log_fn=messages.append)
    assert any("Run tracking first" in m for m in messages)


def test_trex_explains_an_empty_export(tmp_path):
    root = _untracked(tmp_path)
    with pytest.raises(common.ExportError, match="Run tracking first"):
        exporters.export_trex_npz(root, "t", str(tmp_path / "trex"))


# ---------------------------------------------------------------------------
# "Include images" — the frames are the heaviest thing a project owns
# ---------------------------------------------------------------------------

def test_the_formats_that_can_carry_images_are_the_ones_that_name_them():
    """GeoJSON and TRex reference no image, so the option would be a lie."""
    assert exporters.SUPPORTS_IMAGES == {"coco", "yolo", "mot", "camtrap"}


def test_coco_puts_the_images_beside_the_json(survey, tmp_path):
    _with_frames(survey)
    path = str(tmp_path / "out" / "coco.json")
    exporters.export_coco(survey, "t", path, image_size=SIZE)

    folder = os.path.join(os.path.dirname(path), "images")
    assert sorted(os.listdir(folder)) == ["frame_000000.jpg",
                                          "frame_000001.jpg"]


def test_coco_without_images_writes_only_the_json(survey, tmp_path):
    _with_frames(survey)
    path = str(tmp_path / "out" / "coco.json")
    exporters.export_coco(survey, "t", path, image_size=SIZE,
                          include_images=False)

    assert os.listdir(os.path.dirname(path)) == ["coco.json"]


def test_mot_fills_the_img1_folder(survey, tmp_path):
    """MOTChallenge sequences keep their frames in img1."""
    _with_frames(survey)
    folder = str(tmp_path / "mot")
    exporters.export_mot(survey, "t", folder)

    assert sorted(os.listdir(os.path.join(folder, "img1"))) == [
        "frame_000000.jpg", "frame_000001.jpg"]


def test_mot_without_images_writes_no_img1(survey, tmp_path):
    _with_frames(survey)
    folder = str(tmp_path / "mot")
    exporters.export_mot(survey, "t", folder, include_images=False)

    assert not os.path.exists(os.path.join(folder, "img1"))
    assert os.path.exists(os.path.join(folder, "gt.txt"))


def test_camtrap_media_paths_follow_the_files(survey, tmp_path):
    """filePath has to describe where the file actually is."""
    _with_frames(survey)
    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633)

    with open(os.path.join(folder, "media.csv"), encoding="utf-8") as fh:
        paths = [row["filePath"] for row in csv.DictReader(fh)]
    assert all(p.startswith("media") for p in paths)
    assert os.listdir(os.path.join(folder, "media"))


def test_camtrap_without_media_points_back_at_the_flight(survey, tmp_path):
    _with_frames(survey)
    folder = str(tmp_path / "camtrap")
    exporters.export_camtrap_dp(survey, "t", folder, epsg=32633,
                                include_images=False)

    with open(os.path.join(folder, "media.csv"), encoding="utf-8") as fh:
        paths = [row["filePath"] for row in csv.DictReader(fh)]
    assert all(p.startswith("frames_t") for p in paths)
    assert not os.path.exists(os.path.join(folder, "media"))


def test_run_export_ignores_images_for_formats_without_them(survey, tmp_path):
    """Passing it must not become a TypeError on GeoJSON or TRex."""
    exporters.run_export("geojson", survey, "t",
                         str(tmp_path / "t.geojson"), epsg=32633,
                         include_images=True)
    exporters.run_export("trex", survey, "t", str(tmp_path / "trex"),
                         include_images=True)


def test_run_export_passes_the_choice_through(survey, tmp_path):
    _with_frames(survey)
    folder = str(tmp_path / "yolo")
    exporters.run_export("yolo", survey, "t", folder, image_size=SIZE,
                         include_images=False)
    assert os.listdir(os.path.join(folder, "images")) == []


def test_a_missing_frame_is_reported_not_hidden(survey, tmp_path):
    _with_frames(survey, names=("frame_000000.jpg",))
    messages = []
    exporters.export_coco(survey, "t", str(tmp_path / "c.json"),
                          image_size=SIZE, log_fn=messages.append)
    assert any("frame_000001.jpg" in m for m in messages)
