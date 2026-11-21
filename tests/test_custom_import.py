# -*- coding: utf-8 -*-
"""Detections and tracks made by another tool, imported by the user's own
code (core.custom_models, kind "importer"): what the TRex import does for
TRex, for any format."""
import os

import pytest

from bambi_wildlife_detection.core import custom_models as cm

IMPORTER = '''
BAMBI_API = 1


class Mine:
    BAMBI_KIND = "importer"

    def setup(self, context):
        self.options = context.options
        return dict(context.options.get("info") or {})

    def run(self, source, frames):
        rows = self.options.get("rows")
        if rows is None:
            rows = [(frame.index, 200.0 + 2 * frame.index, 250.0,
                     220.0 + 2 * frame.index, 270.0, 0.9, "roe deer", 3)
                    for frame in frames]
        if self.options.get("raw_size"):
            return {"detections": rows,
                    "raw_size": self.options["raw_size"]}
        return rows
'''


def write(tmp_path, source=IMPORTER, name="mine.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------

class TestRows:
    def test_a_dict_per_box(self):
        found = cm.normalise_imported([
            {"frame": 2, "x1": 10, "y1": 20, "x2": 30, "y2": 45,
             "confidence": 0.8, "class": "roe deer", "track_id": 7}])

        assert found["rows"] == [{
            "frame": 2, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 45.0,
            "confidence": 0.8, "class": "roe deer", "track_id": 7}]
        assert (found["raw_size"], found["left_out"], found["empty"]) == \
            (None, 0, 0)

    def test_a_tuple_per_box_with_what_is_optional_left_out(self):
        rows = cm.normalise_imported([
            (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 0.5),
            (0, 1, 2, 3, 4, 0.5, "boar"), (0, 1, 2, 3, 4, 0.5, "boar", 9),
        ])["rows"]

        assert [row["confidence"] for row in rows] == [1.0, 0.5, 0.5, 0.5]
        assert [row["class"] for row in rows] == [
            "animal", "animal", "boar", "boar"]
        assert [row["track_id"] for row in rows] == [None, None, None, 9]

    def test_a_box_may_be_given_as_one(self):
        rows = cm.normalise_imported([
            {"frame": 0, "box": (30, 40, 10, 20), "class_name": "fox"}])["rows"]

        # and its corners are put in order
        assert (rows[0]["x1"], rows[0]["y1"], rows[0]["x2"],
                rows[0]["y2"]) == (10.0, 20.0, 30.0, 40.0)
        assert rows[0]["class"] == "fox"

    def test_the_rows_come_in_the_order_of_the_frames(self):
        rows = cm.normalise_imported(
            [(5, 0, 0, 1, 1), (1, 0, 0, 1, 1), (3, 0, 0, 1, 1)])["rows"]

        assert [row["frame"] for row in rows] == [1, 3, 5]

    def test_a_class_index_is_named(self):
        rows = cm.normalise_imported(
            [(0, 0, 0, 1, 1, 1.0, 1)], class_names=["deer", "boar"])["rows"]

        assert rows[0]["class"] == "boar"
        with pytest.raises(cm.CustomModelError, match="class index 5"):
            cm.normalise_imported([(0, 0, 0, 1, 1, 1.0, 5)],
                                  class_names=["deer", "boar"])

    def test_no_track_is_no_track(self):
        rows = cm.normalise_imported([
            (0, 0, 0, 1, 1, 1.0, "x", None), (0, 0, 0, 1, 1, 1.0, "x", -1),
        ])["rows"]

        assert [row["track_id"] for row in rows] == [None, None]

    def test_boxes_on_frames_that_were_not_extracted_are_left_out(self):
        found = cm.normalise_imported(
            [(0, 0, 0, 1, 1), (2, 0, 0, 1, 1), (3, 0, 0, 1, 1)], n_frames=3)

        assert [row["frame"] for row in found["rows"]] == [0, 2]
        assert found["left_out"] == 1

    def test_boxes_without_area_are_left_out(self):
        found = cm.normalise_imported([(0, 5, 5, 5, 9), (0, 0, 0, 1, 1)])

        assert len(found["rows"]) == 1
        assert found["empty"] == 1

    def test_the_size_of_the_raw_video(self):
        found = cm.normalise_imported(
            {"detections": [(0, 0, 0, 1, 1)], "raw_size": [1920, 1080]})

        assert found["raw_size"] == (1920, 1080)
        assert len(found["rows"]) == 1

    def test_nothing_is_nothing(self):
        assert cm.normalise_imported(None)["rows"] == []
        assert cm.normalise_imported([])["rows"] == []

    @pytest.mark.parametrize("result, complaint", [
        ([(0, 1, 2, 3)], "row 0 must be a dict or a tuple"),
        (["nonsense"], "row 0 must be a dict or a tuple"),
        ([{"frame": 0, "x1": 1}], "row 0 has no y1, x2, y2"),
        ([(0, "left", 2, 3, 4)], "not a number"),
        ([(0, float("nan"), 2, 3, 4)], "not finite"),
        ([(-1, 1, 2, 3, 4)], "frames count from 0"),
        ([(0, 1, 2, 3, 4, 1.0, "x", "seven")], "track_id"),
        ({"rows": []}, "without 'detections'"),
        ({"detections": [], "raw_size": "big"}, "raw_size"),
        ({"detections": [], "raw_size": (0, 10)}, "raw_size"),
        (5, "list of rows"),
    ])
    def test_what_is_no_row_is_refused(self, result, complaint):
        with pytest.raises(cm.CustomModelError, match=complaint):
            cm.normalise_imported(result)

    def test_confidence_stays_between_nought_and_one(self):
        rows = cm.normalise_imported(
            [(0, 0, 0, 1, 1, 1.7), (0, 0, 0, 1, 1, -0.2)])["rows"]

        assert [row["confidence"] for row in rows] == [1.0, 0.0]


class TestInterface:
    def opened(self, path, **options):
        return cm.open_implementation(
            {"path": path, "class": "", "options": options},
            cm.IMPORTER, cm.Context(kind=cm.IMPORTER, camera=cm.THERMAL))

    def test_an_importer_is_a_kind_of_its_own(self, tmp_path):
        path = write(tmp_path)

        assert cm.IMPORTER in cm.KINDS
        assert cm.find_classes(path, cm.IMPORTER) == ["Mine"]
        assert cm.find_classes(path, cm.DETECTOR) == []

    def test_its_boxes_are_on_the_frames_unless_it_says_raw(self, tmp_path):
        path = write(tmp_path)

        assert self.opened(path).import_space == cm.IMPORT_FRAMES
        assert self.opened(
            path, info={"space": "RAW"}).import_space == cm.IMPORT_RAW
        with pytest.raises(cm.CustomModelError, match="'frames' or 'raw'"):
            self.opened(path, info={"space": "world"})

    def test_it_is_handed_the_source_and_the_frames(self, tmp_path):
        impl = self.opened(write(tmp_path))

        found = impl.import_rows("somewhere", [cm.Frame(0), cm.Frame(1)])

        assert [row["frame"] for row in found["rows"]] == [0, 1]
        assert found["rows"][0]["track_id"] == 3


# ---------------------------------------------------------------------------
# The sample
# ---------------------------------------------------------------------------

class TestCsvSample:
    PATH = os.path.join(cm.samples_folder(), "importer_csv.py")

    def run(self, tmp_path, text, frames=(), **options):
        source = tmp_path / "boxes.csv"
        source.write_text(text, encoding="utf-8")
        impl = cm.open_implementation(
            {"path": self.PATH, "class": "", "options": options},
            cm.IMPORTER, cm.Context(kind=cm.IMPORTER))
        return impl.import_rows(str(source), list(frames)), impl

    def test_it_is_shipped(self):
        assert "importer_csv.py" in cm.sample_files()
        assert cm.find_classes(self.PATH, cm.IMPORTER) == ["CsvImporter"]

    def test_tracks_from_a_csv(self, tmp_path):
        found, _impl = self.run(
            tmp_path,
            "frame,x1,y1,x2,y2,confidence,class,track_id\n"
            "0,10,20,30,40,0.9,roe deer,1\n"
            "1,12,20,32,40,0.8,roe deer,1\n")

        assert [(row["frame"], row["track_id"], row["class"])
                for row in found["rows"]] == [
            (0, 1, "roe deer"), (1, 1, "roe deer")]

    def test_detections_with_corner_and_size_and_semicolons(self, tmp_path):
        found, _impl = self.run(tmp_path, "frame;x;y;w;h\n4;10;20;5;8\n")

        row = found["rows"][0]
        assert (row["x1"], row["y1"], row["x2"], row["y2"]) == \
            (10.0, 20.0, 15.0, 28.0)
        assert (row["track_id"], row["class"], row["confidence"]) == \
            (None, "animal", 1.0)

    def test_frames_named_by_their_file(self, tmp_path):
        frames = [cm.Frame(0, path="/f/frame_000000.jpg"),
                  cm.Frame(1, path="/f/frame_000001.jpg")]

        found, _impl = self.run(
            tmp_path, "frame,x1,y1,x2,y2\nframe_000001.jpg,1,2,3,4\n",
            frames)

        assert found["rows"][0]["frame"] == 1

    def test_boxes_on_the_raw_video(self, tmp_path):
        found, impl = self.run(
            tmp_path, "frame,x1,y1,x2,y2\n0,1,2,3,4\n",
            space="raw", raw_size=[1920, 1080])

        assert impl.import_space == cm.IMPORT_RAW
        assert found["raw_size"] == (1920, 1080)
