# -*- coding: utf-8 -*-
"""The Position Uncertainty step: how far off the places of a survey's
detections may be (bambi_survey_position.py on top of
core/position_uncertainty.py)."""
import csv
import json
import math
import os

import pytest

from bambi_wildlife_detection.bambi_processing import (
    BambiProcessor, ProcessingWorker)
from bambi_wildlife_detection.core import (
    agl_profile, position_uncertainty as pu, survey_area)
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight

#: The drone flies due east along y = 0 at 5 m per frame, 80 m above the
#: DEM's origin (:func:`survey_flights.write_poses`). The helper places the
#: animals at height nought, and so is the origin: the camera is 80 m above
#: them.
#: One animal straight below the line, one 80 m to the side.
TRACKS = [
    {"id": 1, "species": "roe deer", "path": [
        (20 + n, 100.0 + 5.0 * n, 0.0) for n in range(5)]},
    {"id": 2, "species": "red deer", "path": [
        (60 + n, 300.0 + 5.0 * n, 80.0) for n in range(5)]},
]


def _flight(folder, tracks=TRACKS, dem=True, mesh=False, **more):
    make_flight(folder, tracks, n_frames=120, **more)
    if not dem:
        return ""
    path = os.path.join(folder, "dem.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"origin": [ORIGIN[0], ORIGIN[1], 0.0]}, handle)
    if mesh:
        with open(os.path.join(folder, "dem.glb"), "wb") as handle:
            handle.write(b"glTF")
    return path


def _survey(tmp_path, *flights, area=None, **more):
    survey_folder = str(tmp_path / "survey")
    if area is not None:
        from shapely.geometry import box
        os.makedirs(survey_folder, exist_ok=True)
        x0, y0 = ORIGIN
        survey_area.create(survey_folder, EPSG, plots=[
            (box(x0 + area[0], y0 + area[1], x0 + area[2], y0 + area[3]),
             "plot")])
    return Survey(folder=survey_folder, modality="t",
                  area_enabled=area is not None, flights=[
                      SurveyFlight(name=os.path.basename(folder),
                                   target_folder=folder, epsg=EPSG, dem=dem)
                      for folder, dem in flights], **more)


def _run(survey, **settings):
    log = []
    config = dict(survey=survey.to_dict(), **settings)
    document = BambiProcessor().run_position_uncertainty(
        config, log_fn=log.append)
    return document, log, os.path.join(survey.folder, "analytics_t")


def _boxes(out, source="tracks"):
    with open(os.path.join(out, f"position_uncertainty_{source}_boxes.csv"),
              encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def meadow(tmp_path):
    folder = str(tmp_path / "Meadow")
    return folder, _flight(folder)


def test_every_detection_gets_its_error(tmp_path, meadow):
    document, log, out = _run(_survey(tmp_path, meadow))
    rows = _boxes(out)

    assert len(rows) == 10
    assert list(rows[0]) == pu.BOX_COLUMNS
    below = [row for row in rows if row["track_id"] == "1"]
    beside = [row for row in rows if row["track_id"] == "2"]
    # the first animal is under the drone on frame 20: straight below
    first = below[0]
    assert (first["flight"], first["frame"]) == ("Meadow", "20")
    assert float(first["height_m"]) == pytest.approx(80.0)
    assert float(first["off_nadir_deg"]) == pytest.approx(0.0, abs=0.01)
    assert float(beside[0]["off_nadir_deg"]) == pytest.approx(45.0, abs=0.01)
    assert float(beside[0]["azimuth_deg"]) == pytest.approx(0.0, abs=0.01)
    assert float(beside[0]["error_m"]) > 1.3 * float(first["error_m"])
    assert first["main_source"] == pu.GPS
    assert beside[0]["main_source"] == pu.ANGLE
    assert log[0] == ("Working out the position uncertainty (Thermal, "
                      "tracks) of Meadow...")
    assert "Meadow: 10 detection(s)" in log


def test_the_figures_are_the_budgets(tmp_path, meadow):
    document, _log, out = _run(_survey(tmp_path, meadow))
    first = [row for row in _boxes(out) if row["track_id"] == "1"][0]

    # no frame on disk, so no pixel; the drone at 5 m/s, straight over it
    expected = pu.budget(80.0, 0.0, pu.Accuracy(), speed_ms=5.0)
    assert float(first["error_m"]) == pytest.approx(expected["error_m"],
                                                    abs=1e-3)
    assert float(first["error95_m"]) == pytest.approx(
        2 * expected["error_m"], abs=2e-3)
    assert document["accuracy"] == {
        "gps_horizontal_m": 1.5, "gps_vertical_m": 1.0,
        "camera_angle_deg": 1.0, "terrain_m": 1.0, "pixel_px": 2.0,
        "sync_s": 0.1}


def test_the_detections_together_and_the_animals(tmp_path, meadow):
    document, log, out = _run(_survey(tmp_path, meadow))

    summary = document["summary"]
    assert summary["boxes"] == 10
    assert summary["median_m"] < summary["p90_m"] <= summary["max_m"]
    assert summary["main_sources"] == {pu.GPS: 0.5, pu.ANGLE: 0.5}
    assert summary["median_height_m"] == pytest.approx(80.0)
    assert document["by_flight"] == {}
    animals = {row["track_id"]: row for row in document["tracks"]}
    assert (animals[1]["species"], animals[1]["boxes"]) == ("roe deer", 5)
    assert animals[2]["error_m"] > animals[1]["error_m"]
    assert animals[1]["x"] == pytest.approx(ORIGIN[0] + 110.0)
    with open(os.path.join(out, "position_uncertainty_tracks_tracks.csv"),
              encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert list(written[0]) == pu.TRACK_COLUMNS and len(written) == 2
    assert any(line.startswith("10 detections: typically ") for line in log)
    assert log[-1] == "Position uncertainty written to " + os.path.join(
        out, "position_uncertainty_tracks.json")
    with open(os.path.join(out, "position_uncertainty_tracks.json"),
              encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["summary"]["boxes"] == 10 and "boxes" not in stored


def test_the_accuracies_can_be_set(tmp_path, meadow):
    standard, _log, _out = _run(_survey(tmp_path, meadow))
    rtk, _log, out = _run(_survey(tmp_path, meadow),
                          position_gps_horizontal_m=0.05,
                          position_gps_vertical_m=0.1)

    assert rtk["accuracy"]["gps_horizontal_m"] == 0.05
    assert rtk["summary"]["median_m"] < standard["summary"]["median_m"]
    assert rtk["summary"]["main_sources"] == {pu.ANGLE: 1.0}


def test_an_accuracy_that_is_none_is_refused(tmp_path, meadow):
    with pytest.raises(RuntimeError, match="not an accuracy"):
        _run(_survey(tmp_path, meadow), position_camera_angle_deg=-1)


def test_the_slope_is_read_from_the_dem_mesh(tmp_path, monkeypatch):
    """Ground that falls away to the north at 1 in 2: the animal to the
    north of the line is seen across ground falling away from the
    camera."""
    folder = str(tmp_path / "Slope")
    dem = _flight(folder, mesh=True)
    asked = []

    def ground_heights(mesh_path, xy, errors=None):
        asked.append((os.path.basename(mesh_path), len(xy)))
        return [-0.5 * y for _x, y in xy]

    monkeypatch.setattr(agl_profile, "ground_heights", ground_heights)

    document, log, out = _run(_survey(tmp_path, (folder, dem)))

    beside = [row for row in _boxes(out) if row["track_id"] == "2"][0]
    below = [row for row in _boxes(out) if row["track_id"] == "1"][0]
    assert float(beside["slope_deg"]) == pytest.approx(
        -math.degrees(math.atan(0.5)), abs=0.01)
    expected = pu.budget(80.0, 80.0, pu.Accuracy(), slope=-0.5, speed_ms=5.0,
                         flight_angle_rad=math.pi / 2)
    assert float(beside["error_m"]) == pytest.approx(expected["error_m"],
                                                     abs=1e-2)
    # straight below, a slope changes nothing
    assert float(below["error_m"]) == pytest.approx(
        pu.budget(80.0, 0.0, pu.Accuracy(), speed_ms=5.0)["error_m"],
        abs=1e-2)
    assert document["level_terrain"] == []
    # one call for the whole flight, four heights per square of 5 m
    assert asked[0][0] == "dem.glb" and asked[0][1] % 4 == 0
    assert len(asked) == 1


def test_without_the_mesh_the_terrain_counts_as_level(tmp_path, meadow):
    document, log, out = _run(_survey(tmp_path, meadow))

    assert document["level_terrain"] == ["Meadow"]
    assert ("Meadow: no DEM mesh found - the terrain is taken as level"
            in log)
    assert "The terrain was taken as level for Meadow" in document["notes"]
    assert all(float(row["slope_deg"]) == 0.0 for row in _boxes(out))


def test_a_mesh_that_gives_no_heights_counts_as_level_too(tmp_path,
                                                          monkeypatch):
    folder = str(tmp_path / "Hole")
    dem = _flight(folder, mesh=True)

    def ground_heights(mesh_path, xy, errors=None):
        errors.append("no ray hit the mesh")
        return [None for _ in xy]

    monkeypatch.setattr(agl_profile, "ground_heights", ground_heights)

    document, log, _out = _run(_survey(tmp_path, (folder, dem)))

    assert document["level_terrain"] == ["Hole"]
    assert ("Hole: the DEM mesh gave no heights (no ray hit the mesh) - the "
            "terrain is taken as level") in log


def test_the_size_of_a_pixel_comes_from_the_frames(tmp_path, meadow):
    pytest.importorskip("PIL")
    from PIL import Image

    folder, dem = meadow
    frames = os.path.join(folder, "frames_t")
    os.makedirs(frames)
    Image.new("L", (640, 512)).save(os.path.join(frames, "frame_000000.jpg"))

    document, _log, out = _run(_survey(tmp_path, meadow),
                               position_gps_horizontal_m=0,
                               position_gps_vertical_m=0,
                               position_camera_angle_deg=0,
                               position_terrain_m=0, position_sync_s=0)

    first = [row for row in _boxes(out) if row["track_id"] == "1"][0]
    # the poses name no field of view: 50 degrees, over 512 rows
    per_pixel = pu.pixel_angle(50.0, 512)
    assert float(first["error_m"]) == pytest.approx(
        80 * 2 * per_pixel * 2 ** 0.5, abs=1e-3)
    assert first["main_source"] == pu.PIXEL
    assert "pixel is not known" not in document["notes"]


def test_without_a_frame_the_pixel_is_left_out_and_said(tmp_path, meadow):
    document, _log, _out = _run(_survey(tmp_path, meadow))

    assert ("The size of a pixel is not known for Meadow (no frame on "
            "disk)") in document["notes"]


def test_without_the_dem_nobody_knows_the_height(tmp_path):
    folder = str(tmp_path / "NoDem")
    _flight(folder, dem=False)

    with pytest.raises(RuntimeError, match="NoDem: the DEM origin is not "
                                           "known"):
        _run(_survey(tmp_path, (folder, "")))


def test_counting_boxes_judges_every_box(tmp_path):
    folder = str(tmp_path / "Loose")
    dem = _flight(folder, loose_boxes=[
        {"species": "roe deer", "at": (200.0, 30.0), "frame": 40}])

    document, _log, out = _run(_survey(tmp_path, (folder, dem),
                                       source="detections"))

    assert document["source"] == "detections"
    assert document["summary"]["boxes"] == 11
    assert document["tracks"] == []
    assert len(_boxes(out, "detections")) == 11
    assert not os.path.exists(os.path.join(
        out, "position_uncertainty_detections_tracks.csv"))


def test_only_what_the_survey_counts(tmp_path, meadow):
    species, _log, _out = _run(_survey(tmp_path, meadow,
                                       species=["red deer"]))
    area, _log, _out = _run(_survey(tmp_path, meadow,
                                    area=(0, -50, 250, 50)))

    assert [row["track_id"] for row in species["tracks"]] == [2]
    assert species["summary"]["boxes"] == 5
    # the plot holds the first animal only
    assert [row["track_id"] for row in area["tracks"]] == [1]


def test_several_flights(tmp_path, meadow):
    folder = str(tmp_path / "Forest")
    dem = _flight(folder, TRACKS[1:])

    document, log, _out = _run(_survey(tmp_path, meadow, (folder, dem)))

    assert list(document["by_flight"]) == ["Meadow", "Forest"]
    assert document["by_flight"]["Forest"]["boxes"] == 5
    assert document["summary"]["boxes"] == 15
    assert [(row["flight"], row["track_id"])
            for row in document["tracks"]] == [
        ("Meadow", 1), ("Meadow", 2), ("Forest", 1)]
    assert any(line.startswith("  Forest: median ") for line in log)


def test_a_survey_with_nothing_placed(tmp_path):
    folder = str(tmp_path / "Unplaced")
    dem = _flight(folder, [{"id": 1, "species": "roe deer", "at": None,
                            "frames": (3, 8)}])

    with pytest.raises(RuntimeError, match="No detection of the survey has "
                                           "a place"):
        _run(_survey(tmp_path, (folder, dem)))


def test_cancelling_stops_before_anything_is_written(tmp_path, meadow):
    from bambi_wildlife_detection.bambi_processing import CancelledException

    survey = _survey(tmp_path, meadow)

    with pytest.raises(CancelledException):
        BambiProcessor().run_position_uncertainty(
            {"survey": survey.to_dict()}, cancel_check=lambda: True)

    assert not os.path.isdir(os.path.join(survey.folder, "analytics_t"))


def test_the_worker_knows_the_step(tmp_path, meadow):
    survey = _survey(tmp_path, meadow)
    worker = ProcessingWorker(BambiProcessor(), "position_uncertainty",
                              {"survey": survey.to_dict()})
    ended, failed = [], []
    worker.finished.connect(lambda step, ok: ended.append((step, ok)))
    worker.error.connect(lambda *told: failed.append(told))

    worker.run()

    assert failed == []
    assert ended == [("position_uncertainty", True)]
