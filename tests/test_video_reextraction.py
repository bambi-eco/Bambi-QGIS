# -*- coding: utf-8 -*-
"""Extracting the frames of a video a second time.

The engine keeps a frame image that already exists unless it is told to
overwrite it, and the plugin never told it. With another calibration or
"preserve aspect ratio" the same frame is a different picture - so a
re-extraction wrote new poses and a new mask around the old pixels. And with
another sampling rate or range the frames are named differently, so the old
ones stayed in the folder beside the new ones.

The engine is not installed in the unit image; a stand-in records what the
plugin asks of it and writes what an extraction writes.
"""
import json
import os
import sys
import types

import pytest

from bambi_wildlife_detection import bambi_processing
from bambi_wildlife_detection.core import calibration_check, poses


# ---------------------------------------------------------------------------
# Frames an earlier extraction left behind
# ---------------------------------------------------------------------------

class TestRemoveUnlistedFrames:
    def folder(self, tmp_path, names):
        for name in names:
            (tmp_path / name).write_bytes(b"x")
        return str(tmp_path)

    def left(self, tmp_path):
        return sorted(os.listdir(tmp_path))

    def test_frames_the_new_list_does_not_name_go(self, tmp_path):
        folder = self.folder(tmp_path, [
            "0-0-1.jpg", "1-5-6.jpg", "1-1-2.jpg", "2-2-3.jpg"])
        images = [{"imagefile": "0-0-1.jpg"}, {"imagefile": "1-5-6.jpg"}]

        assert poses.remove_unlisted_frames(folder, images) == 2
        assert self.left(tmp_path) == ["0-0-1.jpg", "1-5-6.jpg"]

    def test_only_what_the_extractor_names_is_touched(self, tmp_path):
        others = ["DJI_0001_T.JPG", "notes.txt", "poses.json", "mask_T.png",
                  "1-2.jpg", "a-1-2.jpg", "1-2-3.jpg.bak", "1-2-3-4.jpg"]
        folder = self.folder(tmp_path, others + ["7-7-8.jpg", "9-9-9.PNG"])

        assert poses.remove_unlisted_frames(
            folder, [{"imagefile": "0-0-1.jpg"}]) == 2
        assert self.left(tmp_path) == sorted(others)

    def test_an_empty_list_removes_nothing(self, tmp_path):
        """Whatever went wrong there, it is no reason to empty the folder."""
        folder = self.folder(tmp_path, ["0-0-1.jpg", "1-1-2.jpg"])
        for images in ([], None, [{}], [{"imagefile": ""}]):
            assert poses.remove_unlisted_frames(folder, images) == 0
        assert self.left(tmp_path) == ["0-0-1.jpg", "1-1-2.jpg"]

    def test_a_path_in_the_list_counts_by_its_name(self, tmp_path):
        folder = self.folder(tmp_path, ["0-0-1.jpg", "1-1-2.jpg"])
        images = [{"imagefile": os.path.join("frames_t", "0-0-1.jpg")}]
        assert poses.remove_unlisted_frames(folder, images) == 1
        assert self.left(tmp_path) == ["0-0-1.jpg"]

    def test_a_missing_folder(self, tmp_path):
        assert poses.remove_unlisted_frames(
            str(tmp_path / "nowhere"), [{"imagefile": "0-0-1.jpg"}]) == 0


# ---------------------------------------------------------------------------
# What the plugin asks of the engine
# ---------------------------------------------------------------------------

class _Engine:
    """Stands in for ``TimedPoseExtractor``: writes the frames it is told to
    (keeping existing ones unless asked to overwrite, as the engine does),
    the frame list and the mask."""

    def __init__(self, camera):
        self.camera = camera
        self.calls = []
        self.frames = ["0-0-1.jpg", "1-1-2.jpg", "2-2-3.jpg"]
        self.pixels = b"first"
        self.fail_after = None

    def extract(self, target_folder, airdata_path, video_paths, srt_paths,
                **arguments):
        self.calls.append(arguments)
        for number, name in enumerate(self.frames):
            if self.fail_after is not None and number >= self.fail_after:
                raise bambi_processing.CancelledException("stopped")
            path = os.path.join(target_folder, name)
            if arguments.get("no_images"):
                continue
            if not os.path.exists(path) or arguments.get("overwrite_existing"):
                with open(path, "wb") as fh:
                    fh.write(self.pixels)
        with open(os.path.join(target_folder, "poses.json"), "w") as fh:
            json.dump({"images": [{"imagefile": name}
                                  for name in self.frames]}, fh)
        with open(os.path.join(target_folder,
                               f"mask_{self.camera}.png"), "wb") as fh:
            fh.write(b"mask")


def _module(name, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


@pytest.fixture(params=["t", "w"])
def extraction(request, monkeypatch, tmp_path, processor):
    """``(run, engine, frames folder, config, log)`` for one camera."""
    modality = request.param
    target = tmp_path / "flight"
    target.mkdir()
    engine = _Engine("T" if modality == "t" else "W")
    log = []

    class Callback:
        def __call__(self, idx, img):
            return True

    class Camera:
        @staticmethod
        def from_string(name):
            return name

    stubs = {
        "bambi": _module("bambi"),
        "bambi.video": _module("bambi.video"),
        "bambi.video.calibrated_video_frame_accessor": _module(
            "bambi.video.calibrated_video_frame_accessor",
            CalibratedVideoFrameAccessor=lambda *a, **k: object()),
        "bambi.webgl": _module("bambi.webgl"),
        "bambi.webgl.timed_pose_extractor": _module(
            "bambi.webgl.timed_pose_extractor",
            TimedPoseExtractor=object, TimedFrameExtractorCallback=Callback),
        "bambi.domain": _module("bambi.domain"),
        "bambi.domain.camera": _module("bambi.domain.camera", Camera=Camera),
    }
    stubs["bambi"].webgl = stubs["bambi.webgl"]
    stubs["bambi.webgl"].timed_pose_extractor = \
        stubs["bambi.webgl.timed_pose_extractor"]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(type(processor), "_get_extraction_prerequisites",
                        lambda self, config: (str(target), None, None))
    monkeypatch.setattr(bambi_processing, "make_timed_pose_extractor",
                        lambda *a, **k: engine)
    monkeypatch.setattr(bambi_processing, "resolve_margin_ms",
                        lambda *a, **k: 0)
    monkeypatch.setattr(bambi_processing, "count_srt_frames", lambda paths: 3)
    monkeypatch.setattr(calibration_check, "enforce_calibration_resolution",
                        lambda *a, **k: None)

    camera = "thermal" if modality == "t" else "rgb"
    config = {
        "target_folder": str(target), "input_mode": "video",
        f"{camera}_video_paths": ["flight.mp4"],
        f"{camera}_srt_paths": ["flight.srt"],
        f"{camera}_calibration_data": {"any": "calibration"},
    }
    step = (processor.extract_thermal_frames if modality == "t"
            else processor.extract_rgb_frames)

    def run(**changes):
        return step(dict(config, **changes), log_fn=log.append)

    return run, engine, str(target / f"frames_{modality}"), modality, log


def _pixels(folder):
    return {name: open(os.path.join(folder, name), "rb").read()
            for name in sorted(os.listdir(folder))}


def test_a_second_extraction_replaces_the_images(extraction):
    run, engine, folder, _modality, _log = extraction
    run()
    assert set(_pixels(folder).values()) == {b"first"}

    engine.pixels = b"second"          # another calibration, say
    run()

    assert engine.calls[-1]["overwrite_existing"] is True
    assert _pixels(folder) == {name: b"second" for name in engine.frames}


def test_frames_of_the_earlier_extraction_do_not_stay(extraction):
    run, engine, folder, modality, log = extraction
    run()
    with open(os.path.join(folder, "notes.txt"), "w") as fh:
        fh.write("mine")

    engine.frames = ["0-0-1.jpg", "1-5-6.jpg"]       # every fifth frame now
    engine.pixels = b"second"
    run()

    assert _pixels(folder) == {"0-0-1.jpg": b"second",
                               "1-5-6.jpg": b"second", "notes.txt": b"mine"}
    assert any("Removed 2 frame image(s)" in line for line in log)
    target = os.path.dirname(folder)
    with open(os.path.join(target, f"poses_{modality}.json")) as fh:
        assert [image["imagefile"] for image in json.load(fh)["images"]] == \
            engine.frames


def test_a_first_extraction_removes_nothing(extraction):
    run, engine, folder, _modality, log = extraction
    run()
    assert sorted(_pixels(folder)) == engine.frames
    assert not any("Removed" in line for line in log)


def test_poses_only_leaves_the_images_alone(extraction):
    """"No images" re-reads the poses; the images are those of before, and
    so is what is in the folder."""
    run, engine, folder, _modality, log = extraction
    run()

    engine.frames = ["0-0-1.jpg", "1-5-6.jpg"]
    engine.pixels = b"second"
    run(no_images=True)

    assert _pixels(folder) == {
        "0-0-1.jpg": b"first", "1-1-2.jpg": b"first", "2-2-3.jpg": b"first"}
    assert not any("Removed" in line for line in log)


def test_an_interrupted_reextraction_says_what_it_left(extraction, monkeypatch):
    run, engine, folder, modality, log = extraction
    run()
    marked = []
    monkeypatch.setattr(bambi_processing, "_frames_reextracted",
                        lambda config, m, log_fn=None: marked.append(m))

    engine.pixels = b"second"
    engine.fail_after = 1                 # Abort after the first frame
    with pytest.raises(bambi_processing.CancelledException):
        run()

    # new pixels under the previous frame list: said, and flagged
    assert _pixels(folder)["0-0-1.jpg"] == b"second"
    assert _pixels(folder)["1-1-2.jpg"] == b"first"
    assert any("did not finish" in line for line in log)
    assert marked == [modality]


def test_an_interrupted_first_extraction_has_nothing_to_warn_about(
        extraction, monkeypatch):
    run, engine, _folder, _modality, log = extraction
    marked = []
    monkeypatch.setattr(bambi_processing, "_frames_reextracted",
                        lambda config, m, log_fn=None: marked.append(m))

    engine.fail_after = 1
    with pytest.raises(bambi_processing.CancelledException):
        run()

    assert not any("did not finish" in line for line in log)
    assert marked == []
