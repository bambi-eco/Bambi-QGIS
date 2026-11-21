# -*- coding: utf-8 -*-
"""An export format of the user's own (core.custom_models, kind
"exporter"; core.exporters.custom): the plugin reads the store, the
user's class writes the files."""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.core import (
    custom_models as cm, dataset_export as de, exporters, match_store, store)
from bambi_wildlife_detection.core.exporters import custom

from .survey_flights import EPSG, make_flight

EXPORTER = '''
import json
import os

BAMBI_API = 1


class Mine:
    BAMBI_KIND = "exporter"

    def setup(self, context):
        self.options = context.options
        self.camera = context.camera

    def run(self, data, output):
        if self.options.get("fail"):
            raise ValueError("the disk is made of cheese")
        with open(os.path.join(output, "mine.json"), "w") as handle:
            json.dump({"camera": self.camera, "data": data}, handle)
'''


@pytest.fixture
def flight(tmp_path):
    folder = make_flight(str(tmp_path / "Meadow"), [
        {"id": 1, "species": "roe deer", "at": (10, 10), "frames": (0, 2),
         "sex": "female"},
        {"id": 2, "species": "red deer", "at": (30, 10), "frames": (1, 3)},
    ], loose_boxes=[{"species": "roe deer", "at": None, "frame": 2}])
    return folder


def masks(folder, rows):
    """Store segmentation masks: ``(detection_id, frame, pixel ring, world
    ring)``."""
    conn = store.open_store(
        store.stage_path(folder, store.SEGMENTATION, "t"),
        store.SEGMENTATION, "t")
    try:
        with store.transaction(conn):
            for detection_id, frame, pixels, world in rows:
                conn.execute(
                    "INSERT INTO segments (detection_id, frame, polygon_px, "
                    "polygon_geo, attributes) VALUES (?, ?, ?, ?, ?)",
                    (detection_id, frame, json.dumps(pixels),
                     None if world is None else json.dumps(world),
                     json.dumps({"score": 0.9})))
    finally:
        conn.close()


def spec(tmp_path, source=EXPORTER, **options):
    path = tmp_path / "mine.py"
    path.write_text(source, encoding="utf-8")
    return {"path": str(path), "class": "", "options": options}


# ---------------------------------------------------------------------------
# What the exporter is handed
# ---------------------------------------------------------------------------

class TestData:
    def test_the_flight_and_its_frames(self, flight):
        data = custom.build_data(flight, "t", EPSG)

        assert data["flight"] == {"name": "Meadow", "folder": flight,
                                  "camera": "thermal", "crs": "EPSG:32633"}
        assert [frame["index"] for frame in data["frames"]] == [0, 1, 2, 3]
        assert data["frames"][0]["path"] == os.path.join(
            flight, "frames_t", data["frames"][0]["file"])
        assert data["options"] == {"include_images": True,
                                   "include_not_an_animal": True}

    def test_detections_with_names_not_ids(self, flight):
        data = custom.build_data(flight, "t", EPSG)

        assert len(data["detections"]) == 5
        first = data["detections"][0]
        assert set(first) == {"detection_id", "frame", "box", "confidence",
                              "species", "track_id", "attributes", "world",
                              "lonlat"}
        assert first["species"] == "roe deer"
        assert len(first["box"]) == 4
        assert first["attributes"]["sex"] == "female"
        assert data["species"] == ["red deer", "roe deer"]

    def test_where_a_box_is_on_the_ground(self, flight):
        data = custom.build_data(flight, "t", EPSG)

        placed = [d for d in data["detections"] if d["world"] is not None]
        assert len(placed) == 4
        assert len(placed[0]["world"]) == 6
        lon, lat = placed[0]["lonlat"]
        assert 14.9 < lon < 15.1 and 47.8 < lat < 47.9
        # the box nobody placed has neither
        loose = [d for d in data["detections"] if d["track_id"] is None]
        assert [(d["world"], d["lonlat"]) for d in loose] == [(None, None)]

    def test_without_a_crs_there_is_no_longitude(self, flight):
        data = custom.build_data(flight, "t", None)

        assert data["flight"]["crs"] is None
        assert all(d["lonlat"] is None for d in data["detections"])
        assert any(d["world"] is not None for d in data["detections"])

    def test_tracks_are_animals(self, flight):
        data = custom.build_data(flight, "t", EPSG)

        assert [(t["track_id"], t["species"], len(t["detection_ids"]),
                 t["first_frame"], t["last_frame"])
                for t in data["tracks"]] == [
            (1, "roe deer", 2, 0, 2), (2, "red deer", 2, 1, 3)]
        assert data["tracks"][0]["attributes"]["sex"] == "female"
        ids = {d["detection_id"] for d in data["detections"]}
        assert set(data["tracks"][0]["detection_ids"]) <= ids

    def test_segmentation_masks(self, flight):
        ring = [[10, 10], [20, 10], [20, 20]]
        world = [[500010.0, 5300010.0], [500020.0, 5300010.0],
                 [500020.0, 5300020.0]]
        masks(flight, [(1, 0, ring, world), (None, 3, [ring], None)])

        data = custom.build_data(flight, "t", EPSG)

        first, second = data["segmentations"]
        assert first["detection_id"] == 1 and first["frame"] == 0
        assert first["polygon"] == [[[10.0, 10.0], [20.0, 10.0],
                                     [20.0, 20.0]]]
        assert first["world"] == [world]
        assert 14.9 < first["lonlat"][0][0][0] < 15.1
        assert first["attributes"] == {"score": 0.9}
        # a mask of no detection, never geo-referenced
        assert (second["detection_id"], second["world"],
                second["lonlat"]) == (None, None, None)
        assert second["polygon"] is not None

    def test_a_flight_without_masks_has_none(self, flight):
        assert custom.build_data(flight, "t", EPSG)["segmentations"] == []

    def test_only_what_both_cameras_saw(self, flight):
        from bambi_wildlife_detection.core.exporters import common

        make_flight(flight, [{"id": 1, "species": "roe deer",
                              "at": (10, 11), "frames": (0, 2)}],
                    modality="w")
        match_store.record_matches(flight, [
            {"track_id_t": 1, "track_id_w": 1, "shared": 2,
             "median_dist": 0.5, "pairs": []}])
        ring = [[1, 1], [5, 1], [5, 5]]
        ids = {d["track_id"]: d["detection_id"] for d in
               custom.build_data(flight, "t")["detections"]}
        masks(flight, [(ids[1], 0, ring, None), (ids[2], 1, ring, None)])

        with common.only_confirmed():
            data = custom.build_data(flight, "t", EPSG)

        assert {d["track_id"] for d in data["detections"]} == {1}
        assert [t["track_id"] for t in data["tracks"]] == [1]
        assert [m["detection_id"] for m in data["segmentations"]] == [ids[1]]
        assert data["species"] == ["roe deer"]


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

class TestExport:
    def test_it_is_a_format_like_the_others(self):
        label, function, is_folder = exporters.EXPORTERS[exporters.CUSTOM]

        assert label == "Custom exporter…"
        assert function is exporters.export_custom
        assert is_folder is True
        assert exporters.CUSTOM in exporters.SUPPORTS_IMAGES
        assert cm.EXPORTER in cm.KINDS

    def test_the_users_class_writes_the_files(self, flight, tmp_path):
        output = str(tmp_path / "out")
        log = []

        exporters.run_export(exporters.CUSTOM, flight, "t", output,
                             epsg=EPSG, include_images=False,
                             custom=spec(tmp_path), log_fn=log.append)

        with open(os.path.join(output, "mine.json")) as handle:
            written = json.load(handle)
        assert written["camera"] == "thermal"
        assert len(written["data"]["detections"]) == 5
        assert written["data"]["options"]["include_images"] is False
        assert any("5 detection(s), 2 track(s), 0 segmentation(s)" in line
                   for line in log)

    def test_false_positives_only_when_asked_for(self, tmp_path):
        folder = make_flight(str(tmp_path / "Scree"), [
            {"id": 1, "species": "roe deer", "at": (10, 10),
             "frames": (0, 2)}],
            loose_boxes=[{"species": "not-an-animal", "at": (40, 40),
                          "frame": 1}])

        everything = custom.build_data(folder, "t", EPSG)
        without = custom.build_data(folder, "t", EPSG,
                                    include_not_an_animal=False)

        assert len(everything["detections"]) == 3
        assert len(without["detections"]) == 2
        assert "not-an-animal" not in without["species"]
        assert without["options"]["include_not_an_animal"] is False

    def test_without_an_exporter_it_says_where_to_choose_one(
            self, flight, tmp_path):
        with pytest.raises(exporters.ExportError,
                           match="No custom exporter is chosen"):
            exporters.run_export(exporters.CUSTOM, flight, "t",
                                 str(tmp_path / "out"))

    def test_what_the_users_code_says_is_passed_on(self, flight, tmp_path):
        with pytest.raises(exporters.ExportError,
                           match="the disk is made of cheese"):
            exporters.run_export(exporters.CUSTOM, flight, "t",
                                 str(tmp_path / "out"),
                                 custom=spec(tmp_path, fail=True))

    def test_several_flights_with_splits(self, flight, tmp_path):
        other = make_flight(str(tmp_path / "Wood"), [
            {"id": 1, "species": "wild boar", "at": (5, 5),
             "frames": (0, 1)}])
        flights = [
            {"name": "Meadow", "target_folder": flight, "epsg": EPSG,
             "split": "train"},
            {"name": "Wood", "target_folder": other, "epsg": EPSG,
             "split": "test"}]
        output = str(tmp_path / "dataset")

        summary = de.export(exporters.CUSTOM, flights, "t", output,
                            include_images=False, custom=spec(tmp_path))

        assert summary["splits"] == {"train": ["Meadow"], "val": [],
                                     "test": ["Wood"]}
        assert os.path.isfile(os.path.join(
            output, "train", "Meadow", "mine.json"))
        assert os.path.isfile(os.path.join(
            output, "test", "Wood", "mine.json"))


# ---------------------------------------------------------------------------
# The sample
# ---------------------------------------------------------------------------

class TestSample:
    PATH = os.path.join(cm.samples_folder(), "exporter_csv.py")

    def test_it_is_shipped(self):
        assert "exporter_csv.py" in cm.sample_files()
        assert cm.find_classes(self.PATH, cm.EXPORTER) == ["CsvExporter"]

    def test_it_writes_detections_tracks_and_masks(self, flight, tmp_path):
        masks(flight, [(1, 0, [[1, 1], [5, 1], [5, 5]], None)])
        output = str(tmp_path / "out")

        exporters.run_export(
            exporters.CUSTOM, flight, "t", output, epsg=EPSG,
            include_images=False,
            custom={"path": self.PATH, "class": "", "options": ""})

        with open(os.path.join(output, "detections.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        assert rows[0]["species"] == "roe deer"
        assert rows[0]["lon"] and rows[0]["lat"]
        with open(os.path.join(output, "tracks.csv"), newline="") as f:
            tracks = list(csv.DictReader(f))
        assert [(t["track_id"], t["species"], t["sex"], t["boxes"])
                for t in tracks] == [("1", "roe deer", "female", "2"),
                                     ("2", "red deer", "", "2")]
        with open(os.path.join(output, "segmentations.json")) as f:
            assert len(json.load(f)["segmentations"]) == 1
        assert not os.path.isdir(os.path.join(output, "images"))
