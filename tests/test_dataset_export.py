# -*- coding: utf-8 -*-
"""A survey exported as a dataset (core.dataset_export): several flights,
each in one split and never in two, with the classes named alike."""
import json
import os

import pytest

from bambi_wildlife_detection.core import dataset_export as de, exporters

from .survey_flights import make_flight


# ---------------------------------------------------------------------------
# Which flight goes where
# ---------------------------------------------------------------------------

class TestCounts:
    @pytest.mark.parametrize("total, shares, counts", [
        (10, (70, 15, 15), [7, 2, 1]),
        (20, (70, 15, 15), [14, 3, 3]),
        (3, (70, 15, 15), [1, 1, 1]),
        (2, (70, 15, 15), [2, 0, 0]),
        (1, (70, 15, 15), [1, 0, 0]),
        (5, (80, 20, 0), [4, 1, 0]),
        (4, (1, 1, 0), [2, 2, 0]),
        (0, (70, 15, 15), [0, 0, 0]),
        (6, (0, 0, 1), [0, 0, 6]),
    ])
    def test_the_nearest_whole_flights(self, total, shares, counts):
        found = de.counts_for(total, shares)

        assert found == counts
        assert sum(found) == total

    def test_shares_of_nothing_are_refused(self):
        with pytest.raises(de.DatasetExportError, match="needs a share"):
            de.counts_for(4, (0, 0, 0))


class TestAssign:
    FLIGHTS = [f"flight {number}" for number in range(10)]

    def test_every_flight_is_in_exactly_one_split(self):
        dealt = de.assign(self.FLIGHTS)

        assert sorted(dealt) == sorted(self.FLIGHTS)
        lists = de.by_split(dealt)
        assert [len(lists[split]) for split in de.SPLITS] == [7, 2, 1]
        assert sum(len(names) for names in lists.values()) == 10

    def test_the_same_seed_deals_the_same_way(self):
        assert de.assign(self.FLIGHTS, seed=4) == de.assign(
            list(reversed(self.FLIGHTS)), seed=4)

    def test_another_seed_deals_another_way(self):
        deals = {json.dumps(de.assign(self.FLIGHTS, seed=seed),
                            sort_keys=True) for seed in range(6)}

        assert len(deals) > 1

    def test_it_is_said_in_a_line(self):
        assignment = dict(de.assign(self.FLIGHTS), extra=de.LEFT_OUT)

        assert de.describe(assignment) == \
            "7 train, 2 validation, 1 test, 1 left out"


def test_flights_get_folder_names_a_file_system_takes():
    names = de.folder_names(["Meadow 1", "Meadow/1", "meadow_1", "  ", "Ok"])

    assert names == {"Meadow 1": "Meadow_1", "Meadow/1": "Meadow_1_2",
                     "meadow_1": "meadow_1_3", "  ": "flight", "Ok": "Ok"}


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

@pytest.fixture
def survey(tmp_path):
    """Three flights. Species ids are a flight's own: red deer is the first
    species of one flight and the second of another."""
    def flight(name, tracks):
        folder = make_flight(str(tmp_path / name), tracks)
        return {"name": name, "target_folder": folder, "epsg": 32633}

    return [
        dict(flight("North", [
            {"id": 1, "species": "roe deer", "at": (10, 10), "frames": (0, 2)},
            {"id": 2, "species": "red deer", "at": (30, 10),
             "frames": (1, 3)}]), split="train"),
        dict(flight("South", [
            {"id": 1, "species": "red deer", "at": (10, 10),
             "frames": (0, 2)}]), split="val"),
        dict(flight("East", [
            {"id": 1, "species": "wild boar", "at": (10, 10),
             "frames": (0, 1)}]), split="test"),
    ]


def exported(survey, tmp_path, key="coco", **more):
    output = str(tmp_path / "dataset")
    log = []
    summary = de.export(key, survey, "t", output, include_images=False,
                        log=log.append, **more)
    return output, summary, log


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def sized(monkeypatch):
    """The frames are not there to be measured: say how large they are."""
    from bambi_wildlife_detection.core.exporters import common

    monkeypatch.setattr(common, "require_frame_size",
                        lambda *_args, **_kwargs: (640, 512))


class TestExport:
    def test_each_flight_is_in_the_folder_of_its_split(
            self, survey, tmp_path, sized):
        output, summary, _log = exported(survey, tmp_path)

        assert sorted(os.listdir(os.path.join(output, "train"))) == [
            "North", de.COCO_FILE]
        assert os.path.isfile(os.path.join(
            output, "val", "South", "detections_coco.json"))
        assert os.path.isfile(os.path.join(
            output, "test", "East", "detections_coco.json"))
        assert summary["splits"] == {
            "train": ["North"], "val": ["South"], "test": ["East"]}

    def test_the_splits_are_written_down(self, survey, tmp_path, sized):
        output, summary, _log = exported(survey, tmp_path)

        assert read(output, de.SPLITS_FILE) == {
            "train": ["North"], "val": ["South"], "test": ["East"]}
        assert read(output, de.DATASET_FILE) == summary
        assert summary["format"] == "coco"
        assert summary["camera"] == "thermal"

    def test_no_flight_is_in_two_splits(self, survey, tmp_path, sized):
        output, _summary, _log = exported(survey, tmp_path)

        where = {}
        for split in de.SPLITS:
            for image in read(output, split, de.COCO_FILE)["images"]:
                where.setdefault(image["flight"], set()).add(split)
        assert where == {"North": {"train"}, "South": {"val"},
                         "East": {"test"}}

    def test_a_flight_left_out_is_not_exported(self, survey, tmp_path, sized):
        survey[2]["split"] = de.LEFT_OUT

        output, summary, _log = exported(survey, tmp_path)

        assert not os.path.isdir(os.path.join(output, "test", "East"))
        assert summary["left_out"] == ["East"]
        assert summary["splits"]["test"] == []

    def test_nothing_in_a_split_is_refused(self, survey, tmp_path):
        for flight in survey:
            flight["split"] = de.LEFT_OUT

        with pytest.raises(de.DatasetExportError, match="No flight is in"):
            exported(survey, tmp_path)

    def test_two_flights_of_one_name_are_refused(self, survey, tmp_path):
        survey[1]["name"] = "North"

        with pytest.raises(de.DatasetExportError, match="same name"):
            exported(survey, tmp_path)

    def test_an_unknown_format_is_refused(self, survey, tmp_path):
        with pytest.raises(de.DatasetExportError, match="Unknown export"):
            exported(survey, tmp_path, key="parquet")

    def test_a_flight_that_cannot_be_exported_is_named(
            self, survey, tmp_path, sized):
        def run(key, folder, *args, **kwargs):
            if folder.endswith("South"):
                raise exporters.ExportError("no detections")
            return exporters.run_export(key, folder, *args, **kwargs)

        output, summary, log = exported(survey, tmp_path, run_export=run)

        assert summary["not_exported"] == [
            {"name": "South", "split": "val", "error": "no detections"}]
        assert summary["splits"]["val"] == []
        assert not os.path.isdir(os.path.join(output, "val", "South"))
        assert any("South was not exported" in line for line in log)

    def test_no_flight_exported_at_all_is_an_error(self, survey, tmp_path):
        def run(*_args, **_kwargs):
            raise exporters.ExportError("nothing there")

        with pytest.raises(de.DatasetExportError,
                           match="No flight could be exported"):
            exported(survey, tmp_path, run_export=run)

    def test_it_can_be_cancelled_between_flights(self, survey, tmp_path):
        with pytest.raises(de.DatasetExportError, match="cancelled"):
            exported(survey, tmp_path, cancelled=lambda: True)

    def test_it_says_how_far_it_is(self, survey, tmp_path, sized):
        seen = []

        exported(survey, tmp_path, progress=seen.append)

        assert seen == [30, 60, 90, 100]


class TestClasses:
    def test_coco_names_the_classes_alike_in_every_flight(
            self, survey, tmp_path, sized):
        output, summary, _log = exported(survey, tmp_path)

        assert summary["classes"] == ["roe deer", "red deer", "wild boar"]
        for split, flight in (("train", "North"), ("val", "South"),
                              ("test", "East")):
            document = read(output, split, flight, "detections_coco.json")
            assert [c["name"] for c in document["categories"]] == \
                summary["classes"]
        # red deer was class 0 of South's own export; it is 1 everywhere now
        south = read(output, "val", de.COCO_FILE)
        assert {a["category_id"] for a in south["annotations"]} == {1}
        north = read(output, "train", de.COCO_FILE)
        assert {a["category_id"] for a in north["annotations"]} == {0, 1}

    def test_the_joined_file_finds_the_images_of_each_flight(
            self, survey, tmp_path, sized):
        output, _summary, _log = exported(survey, tmp_path)

        joined = read(output, "train", de.COCO_FILE)
        assert all(image["file_name"].startswith("North/images/")
                   for image in joined["images"])
        ids = [image["id"] for image in joined["images"]]
        assert ids == list(range(len(ids)))
        assert {a["image_id"] for a in joined["annotations"]} <= set(ids)
        assert len(joined["annotations"]) == 4

    def test_yolo_numbers_the_classes_alike(self, survey, tmp_path, sized):
        output, summary, _log = exported(survey, tmp_path, key="yolo")

        assert summary["classes"] == ["roe deer", "red deer", "wild boar"]

        def classes_of(split, flight):
            folder = os.path.join(output, split, flight, "labels")
            found = set()
            for name in os.listdir(folder):
                with open(os.path.join(folder, name)) as handle:
                    found |= {line.split()[0] for line in handle}
            return found

        assert classes_of("train", "North") == {"0", "1"}
        assert classes_of("val", "South") == {"1"}
        assert classes_of("test", "East") == {"2"}

    def test_yolo_has_one_yaml_over_all_flights(self, survey, tmp_path,
                                                sized):
        output, _summary, _log = exported(survey, tmp_path, key="yolo")

        with open(os.path.join(output, "data.yaml")) as handle:
            text = handle.read()
        assert "train:\n  - train/North/images\n" in text
        assert "val:\n  - val/South/images\n" in text
        assert "test:\n  - test/East/images\n" in text
        assert "nc: 3\n" in text
        assert "  1: red deer\n" in text

    def test_other_formats_are_a_folder_per_flight(self, survey, tmp_path):
        output, summary, _log = exported(survey, tmp_path, key="mot")

        assert os.path.isfile(os.path.join(
            output, "train", "North", "gt", "gt.txt"))
        assert summary["classes"] == []


# ---------------------------------------------------------------------------
# Exporting again
# ---------------------------------------------------------------------------

class TestAgain:
    def test_the_same_splits_may_be_exported_again(self, survey, tmp_path,
                                                   sized):
        exported(survey, tmp_path)

        _output, summary, _log = exported(survey, tmp_path)

        assert summary["splits"]["train"] == ["North"]

    def test_a_flight_that_changed_its_split_is_refused(
            self, survey, tmp_path, sized):
        exported(survey, tmp_path)
        survey[1]["split"] = "train"

        with pytest.raises(de.DatasetExportError,
                           match=r"South \(val -> train\)"):
            exported(survey, tmp_path)

    def test_a_flight_now_left_out_is_refused_too(self, survey, tmp_path,
                                                  sized):
        exported(survey, tmp_path)
        survey[2]["split"] = de.LEFT_OUT

        with pytest.raises(de.DatasetExportError,
                           match=r"East \(test -> leave out\)"):
            exported(survey, tmp_path)

    def test_split_folders_of_unknown_origin_are_refused(
            self, survey, tmp_path):
        os.makedirs(tmp_path / "dataset" / "train" / "something")

        with pytest.raises(de.DatasetExportError,
                           match="not an export of this tool"):
            exported(survey, tmp_path)

    def test_a_folder_with_other_things_in_it_is_fine(self, survey, tmp_path,
                                                      sized):
        os.makedirs(tmp_path / "dataset")
        (tmp_path / "dataset" / "notes.txt").write_text("mine")

        exported(survey, tmp_path)

        assert (tmp_path / "dataset" / "notes.txt").read_text() == "mine"


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

class TestStep:
    def config(self, survey, tmp_path, **settings):
        from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

        described = Survey(
            folder=str(tmp_path / "survey"),
            flights=[SurveyFlight(flight["name"], flight["target_folder"],
                                  32633) for flight in survey])
        export = {"format": "coco", "output": str(tmp_path / "dataset"),
                  "splits": {flight["name"]: flight["split"]
                             for flight in survey},
                  "include_images": False}
        export.update(settings)
        return {"survey": described.to_dict(), "dataset_export": export}

    def test_it_exports_the_surveys_flights(self, survey, tmp_path, sized):
        from bambi_wildlife_detection.bambi_processing import BambiProcessor

        log = []
        summary = BambiProcessor().run_dataset_export(
            self.config(survey, tmp_path), log_fn=log.append)

        assert summary["splits"] == {
            "train": ["North"], "val": ["South"], "test": ["East"]}
        assert read(str(tmp_path / "dataset"), de.SPLITS_FILE) == \
            summary["splits"]
        assert any("1 train, 1 validation, 1 test" in line for line in log)

    def test_a_flight_nobody_placed_is_left_out(self, survey, tmp_path,
                                                sized):
        from bambi_wildlife_detection.bambi_processing import BambiProcessor

        summary = BambiProcessor().run_dataset_export(self.config(
            survey, tmp_path, splits={"North": "train"}))

        assert summary["left_out"] == ["East", "South"]

    def test_what_is_wrong_is_said(self, survey, tmp_path):
        from bambi_wildlife_detection.bambi_processing import BambiProcessor

        with pytest.raises(RuntimeError, match="No flight is in a split"):
            BambiProcessor().run_dataset_export(
                self.config(survey, tmp_path, splits={}))

    def test_cancelling_is_cancelling(self, survey, tmp_path):
        from bambi_wildlife_detection.bambi_processing import (
            BambiProcessor, CancelledException)

        with pytest.raises(CancelledException):
            BambiProcessor().run_dataset_export(
                self.config(survey, tmp_path), cancel_check=lambda: True)
