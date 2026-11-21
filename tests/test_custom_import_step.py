# -*- coding: utf-8 -*-
"""The custom import step on a small flight over flat ground: imported
boxes and tracks end up in the store like any other flight's results."""
import os

import pytest

pytest.importorskip("cv2")
pytest.importorskip("trimesh")
pytest.importorskip("alfspy")
pytest.importorskip("bambi")
pytest.importorskip("gltflib")

from bambi_wildlife_detection.bambi_processing import BambiProcessor  # noqa: E402
from bambi_wildlife_detection.core import (  # noqa: E402
    analytics_source, custom_models as cm, detection_store,
    pipeline_outputs, stages, store, track_store, trex_tracklets)

from .survey_flights import species_id  # noqa: E402
from .test_custom_import import IMPORTER, write  # noqa: E402
from .test_trex_import import (  # noqa: E402,F401 - flight is a fixture
    DISTORTION, GSD, HEIGHT, MATRIX, N_FRAMES, WIDTH, _frames, flight)


def run(flight, tmp_path, source="", **options):  # noqa: F811
    config = dict(flight["config"])
    config.pop("trex_npz_dir")
    more = {key: options.pop(key) for key in list(options)
            if key.endswith("_data")}
    config.update(more, custom_import_source=source, custom_implementations={
        "importer": {"path": write(tmp_path), "class": "",
                     "options": options}})
    log = []
    BambiProcessor().run_custom_import(config, log_fn=log.append)
    return log


def boxes(root):
    return [(row["frame"], row["x1"], row["y1"], row["x2"], row["y2"])
            for row in track_store.load_detections(root, "t")]


def test_imported_tracks_have_positions(flight, tmp_path):  # noqa: F811
    run(flight, tmp_path)
    root = flight["root"]

    tracks = pipeline_outputs.geo_tracks(root, "t")
    assert len(tracks) == 1
    (points,) = tracks.values()
    assert len(points) == N_FRAMES
    # 2 px to the right per frame, and the drone a metre further each time
    centres = [(p["x1"] + p["x2"]) / 2 for p in points]
    assert centres[1] - centres[0] == pytest.approx(1.0 + 2 * GSD, abs=0.02)


def test_the_survey_counts_the_animal(flight, tmp_path):  # noqa: F811
    run(flight, tmp_path)

    records, facts = analytics_source.load_animals(flight["root"], "t")

    assert len(records) == 1
    assert records[0]["n_boxes"] == N_FRAMES
    assert facts["unplaced"] == 0


def test_the_boxes_are_stored_as_the_importers(flight, tmp_path):  # noqa: F811
    run(flight, tmp_path)
    root = flight["root"]

    assert boxes(root)[0] == (0, 200.0, 250.0, 220.0, 270.0)
    assert detection_store.detection_counts(root, "t") == {
        detection_store.IMPORT: N_FRAMES}
    run_ = track_store.active_run(root, "t")
    assert run_["kind"] == detection_store.IMPORT
    assert run_["tracker"].startswith("custom:mine.py:Mine@")


def test_a_class_named_as_a_species_is_that_species(flight, tmp_path):  # noqa: F811
    root = flight["root"]
    roe = species_id(root, "roe deer")

    log = run(flight, tmp_path)

    assert {row["species_id"] for row in
            track_store.load_detections(root, "t")} == {roe}
    assert any("roe deer" in line and "by name" in line for line in log)


def test_a_class_nobody_knows_is_an_animal(flight, tmp_path):  # noqa: F811
    rows = [(0, 10, 10, 30, 30, 0.9, "wolpertinger", 1)]

    run(flight, tmp_path, rows=rows)

    assert {row["species_id"] for row in track_store.load_detections(
        flight["root"], "t")} == {store.FALLBACK_SPECIES_ID}


def test_boxes_without_tracks_are_detections_only(flight, tmp_path):  # noqa: F811
    rows = [(0, 10, 10, 30, 30), (1, 12, 10, 32, 30)]

    log = run(flight, tmp_path, rows=rows)
    root = flight["root"]

    assert len(boxes(root)) == 2
    assert track_store.active_run(root, "t") is None
    assert any("only detections were imported" in line for line in log)
    done = set(stages.completed_stages(root, "t"))
    assert {"detection", "georeference"} <= done
    assert "tracking" not in done


def test_the_steps_are_done_and_none_is_out_of_date(flight, tmp_path):  # noqa: F811
    run(flight, tmp_path)
    root = flight["root"]

    assert {"detection", "georeference", "tracking"} <= set(
        stages.completed_stages(root, "t"))
    assert stages.stale_stages(root, "t") == []


def test_importing_again_replaces_the_import(flight, tmp_path):  # noqa: F811
    run(flight, tmp_path)
    run(flight, tmp_path, rows=[(0, 10, 10, 30, 30, 1.0, "animal", 5)])

    root = flight["root"]
    assert boxes(root) == [(0, 10.0, 10.0, 30.0, 30.0)]
    assert len(analytics_source.load_animals(root, "t")[0]) == 1


def test_the_detectors_boxes_stay(flight, tmp_path):  # noqa: F811
    root = flight["root"]
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1, "y1": 1, "x2": 9, "y2": 9, "confidence": 0.5}])

    run(flight, tmp_path)

    assert detection_store.detection_counts(root, "t") == {
        detection_store.DETECTOR: 1, detection_store.IMPORT: N_FRAMES}


def test_boxes_beyond_the_frames_are_left_out_and_said(flight, tmp_path):  # noqa: F811
    rows = [(0, 10, 10, 30, 30), (N_FRAMES + 4, 10, 10, 30, 30)]

    log = run(flight, tmp_path, rows=rows)

    assert len(boxes(flight["root"])) == 1
    assert any("Left out 1 box(es) on frames beyond" in line for line in log)


def test_nothing_to_import_is_an_error(flight, tmp_path):  # noqa: F811
    with pytest.raises(RuntimeError, match="no box to import"):
        run(flight, tmp_path, rows=[])


def test_boxes_on_the_raw_video_are_undistorted(flight, tmp_path):  # noqa: F811
    root = flight["root"]
    for name in os.listdir(os.path.join(root, "frames_t")):
        os.remove(os.path.join(root, "frames_t", name))
    _frames(root, 512, 512)

    run(flight, tmp_path, info={"space": "raw"}, raw_size=[WIDTH, HEIGHT],
        thermal_calibration_data={"mtx": MATRIX, "dist": DISTORTION})

    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (WIDTH, HEIGHT), (512, 512))
    expected = undistorter.box(200.0, 250.0, 220.0, 270.0)
    frame, *box = boxes(root)[0]
    assert frame == 0
    assert box == pytest.approx([round(value, 2) for value in expected])


def test_raw_boxes_need_the_videos_size(flight, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="did not say how large"):
        run(flight, tmp_path, info={"space": "raw"})


def test_without_an_importer_nothing_runs(flight):  # noqa: F811
    config = dict(flight["config"], custom_implementations={})

    with pytest.raises(ValueError, match="No importer is chosen"):
        BambiProcessor().run_custom_import(config)


def test_a_source_that_is_not_there_is_refused(flight, tmp_path):  # noqa: F811
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run(flight, tmp_path, source=str(tmp_path / "nowhere.csv"))


def test_the_sample_imports_a_csv(flight, tmp_path):  # noqa: F811
    source = tmp_path / "tracks.csv"
    source.write_text(
        "frame,x1,y1,x2,y2,confidence,class,track_id\n"
        "0,200,250,220,270,0.9,animal,1\n"
        "1,202,250,222,270,0.9,animal,1\n"
        "0,300,250,320,270,0.8,animal,2\n"
        "frame_000002.png,304,250,324,270,0.8,animal,2\n", encoding="utf-8")
    config = dict(flight["config"], custom_import_source=str(source),
                  custom_implementations={"importer": {
                      "path": os.path.join(cm.samples_folder(),
                                           "importer_csv.py"),
                      "class": "", "options": ""}})
    config.pop("trex_npz_dir")

    BambiProcessor().run_custom_import(config)

    records, _facts = analytics_source.load_animals(flight["root"], "t")
    assert sorted(record["n_boxes"] for record in records) == [2, 2]
    assert IMPORTER and HEIGHT
