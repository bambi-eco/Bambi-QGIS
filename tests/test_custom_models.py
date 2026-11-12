# -*- coding: utf-8 -*-
"""Custom detector / tracker / classifier implementations: loading a class
from a .py file, reading what setup() declares, checking what run() returns
(core.custom_models, 9.0.0)."""

import textwrap

import numpy as np
import pytest

from bambi_wildlife_detection.core import custom_models as cm


def _file(tmp_path, source, name="impl.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


DETECTOR = """
    class Det:
        def setup(self, context):
            self.offset = context.options.get("offset", 0)
            context.log("ready")
            return {"class_names": ["deer", "boar"], "cameras": ["thermal"]}
        def run(self, image, frame):
            return [(10 + self.offset, 10, 30, 40, 0.9, 1),
                    (0, 0, 5, 5, 0.1, "deer")]
"""

CLASSIFIER = """
    class Frames:
        def setup(self, context):
            return {"input": "embeddings", "class_names": ["a", "b"]}
        def run(self, batch, items):
            return [(1, 0.75) for _ in items]

    class Tracks:
        class_names = ["a", "b", "c"]
        input = "crops"
        level = "track"
        cameras = ["rgb", "matched"]
        def setup(self, context):
            self.task = context.task
        def run_track(self, batch, items):
            return (2, 0.6) if len(items) > 1 else None
"""

TRACKER = """
    class Trk:
        def setup(self, context):
            return {"space": "pixel"}
        def run(self, detections, frames):
            return {d["detection_id"]: d["frame"] // 2 for d in detections}
"""


def _ctx(kind, camera="thermal", **kw):
    return cm.Context(kind=kind, camera=camera, **kw)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_the_only_candidate_is_found_without_a_name(tmp_path):
    path = _file(tmp_path, DETECTOR)
    assert cm.find_classes(path, cm.DETECTOR) == ["Det"]
    assert cm.load_class(path, "", cm.DETECTOR).__name__ == "Det"


def test_several_candidates_need_a_name(tmp_path):
    path = _file(tmp_path, CLASSIFIER)
    assert cm.find_classes(path, cm.CLASSIFIER) == ["Frames", "Tracks"]
    with pytest.raises(cm.CustomModelError, match="several"):
        cm.load_class(path, "", cm.CLASSIFIER)
    assert cm.load_class(path, "Tracks", cm.CLASSIFIER).__name__ == "Tracks"
    with pytest.raises(cm.CustomModelError, match="Nope"):
        cm.load_class(path, "Nope", cm.CLASSIFIER)


def test_a_file_without_a_fitting_class_says_what_is_needed(tmp_path):
    path = _file(tmp_path, "class Thing:\n    def setup(self, c): pass\n")
    with pytest.raises(cm.CustomModelError, match="setup, run"):
        cm.load_class(path, "", cm.DETECTOR)


def test_imported_classes_are_not_candidates(tmp_path):
    _file(tmp_path, DETECTOR, "helper_det.py")
    path = _file(tmp_path, "from helper_det import Det\n" + TRACKER.replace(
        "    class", "class").replace("\n    ", "\n"))
    # the sibling module is importable, but only Trk is defined here (a
    # tracker and a detector have the same method names, so it fits both)
    assert cm.find_classes(path, cm.TRACKER) == ["Trk"]
    assert cm.find_classes(path, cm.DETECTOR) == ["Trk"]


def test_missing_files_and_packages_are_named(tmp_path):
    with pytest.raises(cm.CustomModelError, match="not found"):
        cm.load_module(str(tmp_path / "nope.py"))
    # opening one hashes the file first; that must not leak an OSError
    with pytest.raises(cm.CustomModelError, match="not found"):
        cm.open_implementation({"path": str(tmp_path / "nope.py")},
                               cm.DETECTOR, _ctx(cm.DETECTOR))
    path =_file(tmp_path, "import a_package_nobody_has\n")
    with pytest.raises(cm.CustomModelError, match="not installed"):
        cm.load_module(path)
    path = _file(tmp_path, "raise ValueError('boom')\n", "broken.py")
    with pytest.raises(cm.CustomModelError, match="boom"):
        cm.load_module(path)


def test_an_edited_file_is_reloaded(tmp_path):
    path = _file(tmp_path, DETECTOR)
    first = cm.file_hash(path)
    _file(tmp_path, DETECTOR.replace("Det", "Other"))
    assert cm.find_classes(path, cm.DETECTOR) == ["Other"]
    assert cm.file_hash(path) != first


def test_options_are_a_json_object():
    assert cm.parse_options("") == {}
    assert cm.parse_options('{"a": 1}') == {"a": 1}
    assert cm.parse_options({"a": 1}) == {"a": 1}
    with pytest.raises(cm.CustomModelError):
        cm.parse_options("[1, 2]")
    with pytest.raises(cm.CustomModelError):
        cm.parse_options("{not json")


def test_which_specs_are_custom():
    assert cm.is_custom({"source": "implementation"})
    assert cm.is_custom({"model": "implementation"})
    assert not cm.is_custom({"model": "default"})
    assert not cm.is_custom(None)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def test_a_detector_runs_and_its_boxes_are_checked(tmp_path):
    log = []
    spec = {"path": _file(tmp_path, DETECTOR), "options": '{"offset": 5}'}
    impl = cm.open_implementation(
        spec, cm.DETECTOR, _ctx(cm.DETECTOR, min_confidence=0.5,
                                log=log.append))
    assert log == ["ready"]
    assert impl.class_names == ["deer", "boar"]
    assert impl.label.startswith("custom:impl.py:Det@")

    boxes = impl.detect(np.zeros((32, 64, 3), np.uint8), cm.Frame(3))
    assert len(boxes) == 1                       # the 0.1 box is dropped
    assert boxes[0]["class_name"] == "boar"
    assert boxes[0]["x1"] == 15.0
    assert boxes[0]["y2"] == 32.0                # clipped to the image


def test_a_detector_that_does_not_support_the_camera_is_refused(tmp_path):
    spec = {"path": _file(tmp_path, DETECTOR)}
    with pytest.raises(cm.CustomModelError, match="does not support"):
        cm.open_implementation(spec, cm.DETECTOR, _ctx(cm.DETECTOR, "rgb"))


def test_detection_rows_are_validated():
    names = ["deer"]
    assert cm.normalise_detections(None, 10, 10, names) == []
    assert cm.normalise_detections([(8, 8, 2, 2)], 10, 10, names)[0] == {
        "x1": 2.0, "y1": 2.0, "x2": 8.0, "y2": 8.0, "confidence": 1.0,
        "class_index": 0, "class_name": "deer"}
    assert cm.normalise_detections([(20, 20, 30, 30, 0.9, 0)], 10, 10,
                                   names) == []     # outside the image
    assert cm.normalise_detections([(0, 0, 5, 5, 0.9, "fox")], 10, 10,
                                   names)[0]["class_name"] == "fox"
    with pytest.raises(cm.CustomModelError):
        cm.normalise_detections([(1, 2)], 10, 10, names)
    with pytest.raises(cm.CustomModelError):
        cm.normalise_detections([(0, 0, float("nan"), 5)], 10, 10, names)
    with pytest.raises(cm.CustomModelError):
        cm.normalise_detections([5], 10, 10, names)


def test_errors_in_user_code_name_the_file(tmp_path):
    path = _file(tmp_path, """
        class Det:
            def setup(self, context):
                pass
            def run(self, image, frame):
                raise KeyError("weights")
    """)
    impl = cm.open_implementation({"path": path}, cm.DETECTOR,
                                  _ctx(cm.DETECTOR))
    with pytest.raises(cm.CustomModelError, match=r"impl\.py.*run\(\)"):
        impl.detect(np.zeros((4, 4, 3), np.uint8), cm.Frame(0))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def test_a_frame_level_classifier_on_embeddings(tmp_path):
    spec = {"path": _file(tmp_path, CLASSIFIER), "class": "Frames"}
    impl = cm.open_implementation(
        spec, cm.CLASSIFIER, _ctx(cm.CLASSIFIER, "matched", task="species"))
    assert (impl.input, impl.level) == ("embeddings", "frame")
    assert impl.cameras == ("thermal", "rgb", "matched")
    calls = impl.classify_frames(np.zeros((3, 8), np.float32),
                                 [{"detection_id": i} for i in range(3)])
    assert calls == [(1, 0.75)] * 3


def test_a_track_level_classifier_on_crops(tmp_path):
    spec = {"path": _file(tmp_path, CLASSIFIER), "class": "Tracks"}
    impl = cm.open_implementation(
        spec, cm.CLASSIFIER, _ctx(cm.CLASSIFIER, "rgb", task="sex"))
    assert (impl.input, impl.level) == ("crops", "track")
    assert impl.instance.task == "sex"
    assert impl.classify_track([None, None], [{}, {}]) == (2, 0.6)
    assert impl.classify_track([None], [{}]) is None
    with pytest.raises(cm.CustomModelError, match="does not support"):
        cm.open_implementation(spec, cm.CLASSIFIER,
                               _ctx(cm.CLASSIFIER, "thermal"))


@pytest.mark.parametrize("returns, message", [
    ("{}", "must say what the classifier reads"),
    ("{'input': 'pixels', 'class_names': ['a']}", "embeddings"),
    ("{'input': 'crops'}", "name its classes"),
    ("{'input': 'crops', 'class_names': ['a'], 'level': 'animal'}", "level"),
    ("{'input': 'crops', 'class_names': ['a'], 'level': 'track'}",
     "run_track"),
    ("{'input': 'crops', 'class_names': ['a'], 'cameras': ['lidar']}",
     "unknown camera"),
    ("[1, 2]", "must return a dict"),
])
def test_setup_has_to_declare_the_classifier(tmp_path, returns, message):
    path = _file(tmp_path, f"""
        class C:
            def setup(self, context):
                return {returns}
            def run(self, batch, items):
                return []
    """)
    with pytest.raises(cm.CustomModelError, match=message):
        cm.open_implementation({"path": path}, cm.CLASSIFIER,
                               _ctx(cm.CLASSIFIER, "rgb"))


def test_frame_calls_are_validated():
    assert cm.normalise_calls([(0, 0.5), (1, 1.0)], 2, 2) == \
        [(0, 0.5), (1, 1.0)]
    matrix = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
    assert cm.normalise_calls(matrix, 3, 2) == [(1, 0.8), (0, 0.9), (0, 0.5)]
    with pytest.raises(cm.CustomModelError, match="exactly one per row"):
        cm.normalise_calls([(0, 0.5)], 2, 2)
    with pytest.raises(cm.CustomModelError, match="outside"):
        cm.normalise_calls([(5, 0.5)], 1, 2)
    with pytest.raises(cm.CustomModelError, match="between 0 and 1"):
        cm.normalise_calls([(0, 7.0)], 1, 2)
    with pytest.raises(cm.CustomModelError, match="finite"):
        cm.normalise_calls([(0, float("nan"))], 1, 2)
    with pytest.raises(cm.CustomModelError, match="one column per class"):
        cm.normalise_calls(np.zeros((2, 3)), 2, 2)
    with pytest.raises(cm.CustomModelError):
        cm.normalise_calls(None, 1, 2)


def test_track_calls_are_validated():
    assert cm.normalise_track_call(None, 2) is None
    assert cm.normalise_track_call((1, 0.4), 2) == (1, 0.4)
    with pytest.raises(cm.CustomModelError):
        cm.normalise_track_call(3, 2)
    with pytest.raises(cm.CustomModelError):
        cm.normalise_track_call((9, 0.4), 2)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def test_a_tracker_assigns_detections_to_tracks(tmp_path):
    impl = cm.open_implementation(
        {"path": _file(tmp_path, TRACKER)}, cm.TRACKER, _ctx(cm.TRACKER))
    assert impl.space == "pixel"
    detections = [{"detection_id": 10 + i, "frame": i} for i in range(4)]
    assert impl.track(detections, []) == {10: 0, 11: 0, 12: 1, 13: 1}


def test_assignments_are_validated():
    ids = [7, 8, 9]
    assert cm.normalise_assignment({7: 1, 8: None, 9: -1}, ids) == {7: 1}
    assert cm.normalise_assignment([(7, 1), (9, 2)], ids) == {7: 1, 9: 2}
    assert cm.normalise_assignment([4, 4, 5], ids) == {7: 4, 8: 4, 9: 5}
    with pytest.raises(cm.CustomModelError, match="was not given"):
        cm.normalise_assignment({99: 1}, ids)
    with pytest.raises(cm.CustomModelError, match="track id"):
        cm.normalise_assignment([4, 4], ids)
    with pytest.raises(cm.CustomModelError):
        cm.normalise_assignment({7: "x"}, ids)
    with pytest.raises(cm.CustomModelError):
        cm.normalise_assignment(None, ids)


# ---------------------------------------------------------------------------
# The samples shipped with the plugin
# ---------------------------------------------------------------------------

def _sample(name):
    import os
    return os.path.join(cm.samples_folder(), name)


def test_every_sample_holds_exactly_one_class_of_its_kind():
    names = cm.sample_files()
    assert {n.split("_")[0] for n in names} == \
        {"detector", "tracker", "classifier"}
    for name in names:
        kind = name.split("_")[0]
        # imports of torch / ultralytics / transformers happen in setup(),
        # so a sample loads (and can be checked) without them
        assert len(cm.find_classes(_sample(name), kind)) == 1, name


def test_the_samples_cover_what_the_user_asked_for():
    names = set(cm.sample_files())
    assert {"detector_ultralytics_yolo.py", "detector_owl.py",
            "detector_dronemegadetector.py", "tracker_iou.py",
            "classifier_embeddings_head.py",
            "classifier_crops_torchscript.py",
            "classifier_track_level.py"} <= names


def test_samples_are_copied_without_overwriting(tmp_path):
    written = cm.copy_samples(str(tmp_path))
    assert len(written) == len(cm.sample_files())
    (tmp_path / "tracker_iou.py").write_text("# mine", encoding="utf-8")
    assert cm.copy_samples(str(tmp_path)) == []
    assert (tmp_path / "tracker_iou.py").read_text(encoding="utf-8") == "# mine"


def test_the_sample_tracker_follows_two_animals():
    impl = cm.open_implementation(
        {"path": _sample("tracker_iou.py"), "options": '{"max_age": 2}'},
        cm.TRACKER, _ctx(cm.TRACKER))
    assert impl.space == "pixel"
    detections = []
    for frame in range(5):
        if frame != 2:                      # one missed frame is bridged
            detections.append({"detection_id": frame * 2, "frame": frame,
                               "x1": 10 + frame, "y1": 10, "x2": 30 + frame,
                               "y2": 30})
        detections.append({"detection_id": frame * 2 + 1, "frame": frame,
                           "x1": 100, "y1": 100 - frame, "x2": 120,
                           "y2": 120 - frame})
    assert impl.online                      # called frame by frame
    tracks = {}
    for frame in range(5):
        tracks.update(impl.track_frame(
            cm.Frame(frame), [d for d in detections if d["frame"] == frame]))
    assert impl.track_finish([d["detection_id"] for d in detections]) == \
        ({}, set())                         # min_length 1: nothing to revise
    first = {tracks[i] for i in (0, 2, 6, 8)}
    second = {tracks[i] for i in (1, 3, 5, 7, 9)}
    assert len(first) == 1 and len(second) == 1 and first != second


def test_the_sample_track_level_classifier_runs():
    impl = cm.open_implementation(
        {"path": _sample("classifier_track_level.py")}, cm.CLASSIFIER,
        _ctx(cm.CLASSIFIER, "thermal", task="life_stage"))
    assert (impl.input, impl.level, impl.crop_size) == ("crops", "track", 96)
    small = np.zeros((96, 96, 3), np.uint8)
    small[40:50, 40:50] = 255
    large = np.zeros((96, 96, 3), np.uint8)
    large[10:80, 10:80] = 255
    items = [{}] * 3
    assert impl.classify_track([small] * 3, items)[0] == 0
    assert impl.classify_track([large] * 3, items)[0] == 1
    assert impl.classify_track([large], [{}]) is None
    with pytest.raises(cm.CustomModelError, match="does not support"):
        cm.open_implementation(
            {"path": _sample("classifier_track_level.py")}, cm.CLASSIFIER,
            _ctx(cm.CLASSIFIER, "matched"))


def test_the_dronemegadetector_sample_merges_tiles():
    module = cm.load_module(_sample("detector_dronemegadetector.py"))
    assert module._starts(1000, 1280, 1180) == [0]
    assert module._starts(3000, 1280, 1180) == [0, 1180, 1720]
    rows = [[0, 0, 10, 10, 0.9, 1], [1, 1, 11, 11, 0.5, 1],
            [50, 50, 60, 60, 0.4, 1], [0, 0, 10, 10, 0.3, 0]]
    kept = module._merge(rows)
    assert [r[4] for r in kept] == [0.9, 0.4, 0.3]


# ---------------------------------------------------------------------------
# Interpolation of track gaps
# ---------------------------------------------------------------------------

def test_gaps_in_a_track_are_filled_linearly():
    rows = [{"frame": 0, "track_id": 1, "x1": 0.0, "detection_id": 5},
            {"frame": 3, "track_id": 1, "x1": 30.0, "detection_id": 6},
            {"frame": 1, "track_id": 2, "x1": 7.0, "detection_id": 7}]
    out = cm.interpolate_gaps(rows, ("x1",))
    track = [r for r in out if r["track_id"] == 1]
    assert [r["frame"] for r in track] == [0, 1, 2, 3]
    assert [r["x1"] for r in track] == [0.0, 10.0, 20.0, 30.0]
    assert [r["interpolated"] for r in track] == [0, 1, 1, 0]
    assert track[1]["detection_id"] is None
    assert len([r for r in out if r["track_id"] == 2]) == 1


def test_a_tracker_says_whether_interpolation_applies(tmp_path):
    impl = cm.open_implementation(
        {"path": _file(tmp_path, TRACKER)}, cm.TRACKER, _ctx(cm.TRACKER))
    assert impl.interpolate is True
    path = _file(tmp_path, TRACKER.replace(
        '{"space": "pixel"}', '{"space": "pixel", "interpolate": False}'),
        "no_interp.py")
    impl = cm.open_implementation({"path": path}, cm.TRACKER,
                                  _ctx(cm.TRACKER))
    assert impl.interpolate is False


# ---------------------------------------------------------------------------
# Camera extrinsics
# ---------------------------------------------------------------------------

def test_a_pose_without_the_camera_stack_still_has_its_position():
    meta = {"location": [10.0, 20.0, 80.0], "rotation": [5.0, 0.0, 90.0],
            "fovy": [48.0], "lat": 48.1}
    pose = cm.make_pose(
        meta, {"translation": {"x": 1.0, "z": -2.0}}, origin=(1000, 2000, 300))
    assert pose.location == (10.0, 20.0, 80.0)
    assert pose.position == (11.0, 20.0, 78.0)
    assert pose.world == (1011.0, 2020.0, 378.0)
    assert pose.rotation == (5.0, 0.0, 90.0)
    assert pose.fovy == 48.0
    assert pose.quaternion is None and pose.matrix is None
    assert pose.raw["lat"] == 48.1


def test_a_corrected_camera_is_passed_through():
    class Quaternion(list):
        matrix33 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    pose = cm.make_pose({"location": [0, 0, 50]}, {}, None,
                        corrected=([1.0, 2.0, 3.0],
                                   Quaternion([0.0, 0.0, 0.0, 1.0])))
    assert pose.position == (1.0, 2.0, 3.0)
    assert pose.world is None                       # no DEM origin known
    assert pose.quaternion == (0.0, 0.0, 0.0, 1.0)
    assert pose.matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_an_entry_without_a_pose_does_not_fail():
    pose = cm.make_pose({"imagefile": "a.jpg"})
    assert pose.position == (0.0, 0.0, 0.0) and pose.fovy == 50.0


# ---------------------------------------------------------------------------
# Online trackers
# ---------------------------------------------------------------------------

ONLINE = """
    class Online:
        def setup(self, context):
            self.seen = []
            return {"space": "pixel", "images": True}
        def update(self, frame, detections):
            self.seen.append((frame.index, len(detections)))
            return {d["detection_id"]: 1 for d in detections}
        def finish(self):
            return {11: None, 12: 5}
"""


def test_a_tracker_with_update_is_online(tmp_path):
    impl = cm.open_implementation(
        {"path": _file(tmp_path, ONLINE)}, cm.TRACKER, _ctx(cm.TRACKER))
    assert impl.online and impl.wants_images
    assert impl.track_frame(cm.Frame(0), []) == {}
    assert impl.track_frame(cm.Frame(1), [{"detection_id": 11},
                                           {"detection_id": 12}]) == \
        {11: 1, 12: 1}
    assert impl.instance.seen == [(0, 0), (1, 2)]
    # finish() revises: 12 moves to track 5, 11 is taken out of its track
    assert impl.track_finish([11, 12, 13]) == ({12: 5}, {11})


def test_a_tracker_with_run_is_offline(tmp_path):
    impl = cm.open_implementation(
        {"path": _file(tmp_path, TRACKER)}, cm.TRACKER, _ctx(cm.TRACKER))
    assert not impl.online and not impl.wants_images
    assert impl.track_finish([1]) == ({}, set())


def test_a_tracker_needs_update_or_run(tmp_path):
    path = _file(tmp_path, "class T:\n    def setup(self, c): pass\n")
    with pytest.raises(cm.CustomModelError, match="update"):
        cm.load_class(path, "", cm.TRACKER)


def test_files_next_to_the_implementation_are_within_reach(tmp_path):
    """Helper modules by name, data through __file__."""
    (tmp_path / "labels.txt").write_text("deer\nboar\n", encoding="utf-8")
    _file(tmp_path, "def scale():\n    return 2\n", "my_helpers_xyz.py")
    path = _file(tmp_path, """
        import os
        import my_helpers_xyz

        class Det:
            def setup(self, context):
                here = os.path.dirname(__file__)
                with open(os.path.join(here, "labels.txt")) as fh:
                    return {"class_names": fh.read().split()}
            def run(self, image, frame):
                return [(0, 0, 2 * my_helpers_xyz.scale(), 4, 1.0, 1)]
    """)
    impl = cm.open_implementation({"path": path}, cm.DETECTOR,
                                  _ctx(cm.DETECTOR))
    assert impl.class_names == ["deer", "boar"]
    box = impl.detect(np.zeros((8, 8, 3), np.uint8), cm.Frame(0))[0]
    assert (box["x2"], box["class_name"]) == (4.0, "boar")


# ---------------------------------------------------------------------------
# Declared requirements and status reporting
# ---------------------------------------------------------------------------

def test_requirements_are_read_without_running_the_file(tmp_path):
    path = _file(tmp_path, """
        REQUIRES = ["json", "a_package_nobody_has", "os.path"]
        raise SystemExit("this file must not be executed")
    """)
    assert cm.requirements(path) == ["json", "a_package_nobody_has",
                                     "os.path"]
    assert cm.check_requirements(path) == (["json", "os.path"],
                                           ["a_package_nobody_has"])


def test_a_file_without_the_list_declares_nothing(tmp_path):
    assert cm.requirements(_file(tmp_path, DETECTOR)) == []
    assert cm.requirements(str(tmp_path / "nope.py")) == []
    assert cm.requirements(_file(tmp_path, "REQUIRES = compute()\n",
                                 "dyn.py")) == []
    assert cm.requirements(_file(tmp_path, 'REQUIRES = "numpy"\n',
                                 "one.py")) == ["numpy"]


def test_every_sample_declares_its_packages():
    for name in cm.sample_files():
        with open(_sample(name), encoding="utf-8") as fh:
            assert "REQUIRES = [" in fh.read(), name
    assert cm.requirements(_sample("detector_dronemegadetector.py")) == \
        ["torch", "transformers", "PIL", "cv2"]
    assert cm.requirements(_sample("tracker_iou.py")) == []


def test_opening_reports_what_it_is_doing(tmp_path):
    steps = []
    cm.open_implementation(
        {"path": _file(tmp_path, DETECTOR)}, cm.DETECTOR,
        _ctx(cm.DETECTOR), status=steps.append)
    assert steps[0].startswith("Importing impl.py")
    assert steps[1].startswith("Running Det.setup()")
