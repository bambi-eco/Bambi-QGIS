# -*- coding: utf-8 -*-
"""The Reaction to the Drone step: what the tracked animals of a survey did
while the drone was over them (bambi_survey_reaction.py on top of
core/drone_reaction.py)."""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import (
    BambiProcessor, ProcessingWorker)
from bambi_wildlife_detection.core import drone_reaction, survey_area
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight

pytest.importorskip("scipy")

#: The drone flies due east along y = 0, 5 m per frame and one frame per
#: second. An animal is in view for 20 frames.
IN_VIEW = 20


def animal(ident, x, y, velocity=(0.0, 0.0), species="roe deer", frames=None,
           **more):
    """An animal first seen at (x, y) - when the drone is 50 m short of it -
    moving at *velocity* (m/s)."""
    first = max(0, int(x / 5) - 10)
    path = [(first + n, x + velocity[0] * n, y + velocity[1] * n)
            for n in range(frames or IN_VIEW)]
    return dict({"id": ident, "species": species, "path": path}, **more)


def _flight(folder, tracks, dem=True, n_frames=300, **more):
    make_flight(folder, tracks, n_frames=n_frames, **more)
    if not dem:
        return ""
    path = os.path.join(folder, "dem.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"origin": [ORIGIN[0], ORIGIN[1], 0.0]}, handle)
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
    document = BambiProcessor().run_drone_reaction(config,
                                                   log_fn=log.append)
    return document, log, os.path.join(survey.folder, "analytics_t")


#: Eight animals either side of the line. Six walk away from it as the
#: drone comes, one stands, one walks towards it.
FLEEING = [
    *(animal(n + 1, 100 + 120 * n, side * 15.0, (0.0, side * 0.6))
      for n, side in enumerate((1, -1, 1, -1, 1, -1))),
    animal(7, 820, 25.0), animal(8, 940, -30.0, (0.0, 0.5))]


@pytest.fixture
def fleeing(tmp_path):
    folder = str(tmp_path / "Meadow")
    return folder, _flight(folder, FLEEING)


def _by_track(document):
    return {row["track_id"]: row for row in document["tracks"]}


def test_what_every_animal_did(tmp_path, fleeing):
    document, log, _out = _run(_survey(tmp_path, fleeing))
    rows = _by_track(document)

    assert len(rows) == 8
    walker = rows[1]
    assert (walker["flight"], walker["species"], walker["status"]) == (
        "Meadow", "roe deer", drone_reaction.MOVED)
    assert walker["direction"] == drone_reaction.AWAY
    assert walker["speed_ms"] == pytest.approx(0.6)
    assert walker["duration_s"] == pytest.approx(IN_VIEW - 1)
    assert walker["displacement_m"] == pytest.approx(0.6 * (IN_VIEW - 1))
    assert walker["away_m"] == pytest.approx(0.6 * (IN_VIEW - 1))
    assert walker["along_m"] == pytest.approx(0.0, abs=1e-6)
    assert walker["line_distance_m"] == pytest.approx(15.0)
    assert walker["heading_deg"] == pytest.approx(0.0)
    assert (walker["start_x"], walker["start_y"]) == pytest.approx(
        (ORIGIN[0] + 100.0, ORIGIN[1] + 15.0))
    # on the south side away is south
    assert rows[2]["direction"] == drone_reaction.AWAY
    assert rows[2]["heading_deg"] == pytest.approx(180.0)
    assert (rows[7]["status"], rows[7]["direction"]) == (
        drone_reaction.STAYED, None)
    assert rows[8]["direction"] == drone_reaction.TOWARDS
    assert rows[8]["away_m"] == pytest.approx(-0.5 * (IN_VIEW - 1))
    assert log[0] == ("Reading the reaction to the drone (Thermal tracks) "
                      "of Meadow...")
    assert ("Meadow: 8 animal(s), 8 in view long enough to tell, 7 moved"
            in log)


def test_the_drone_passes_while_the_animal_is_in_view(tmp_path, fleeing):
    document, _log, _out = _run(_survey(tmp_path, fleeing))

    # the standing animal 25 m off the line: the drone comes that near
    assert _by_track(document)[7]["drone_distance_m"] == pytest.approx(25.0)


def test_the_animals_together(tmp_path, fleeing):
    document, log, _out = _run(_survey(tmp_path, fleeing))
    summary = document["summary"][drone_reaction.ALL]

    assert list(document["summary"]) == [drone_reaction.ALL]
    assert (summary["tracks"], summary["judged"], summary["moved"]) == (
        8, 8, 7)
    assert summary["directions"][drone_reaction.AWAY] == 6
    assert summary["directions"][drone_reaction.TOWARDS] == 1
    assert summary["mean_away_ms"] == pytest.approx((6 * 0.6 - 0.5) / 8)
    assert summary["mean_away_ms_low"] > 0
    assert summary["verdict"] == drone_reaction.FLED
    assert any(line.startswith("All species: 7 of 8 animals moved while in "
                               "view; 6 went away from the line and 1 "
                               "towards it") for line in log)
    assert document["by_flight"] == {}
    assert document["settings"] == {
        "min_displacement_m": 3.0, "min_speed_ms": 0.0, "per_species": {},
        "min_duration_s": 2.0,
        "min_positions": 4, "sigmas": 3.0}
    assert (document["source"], document["camera"], document["crs"]) == (
        "tracks", "Thermal", "EPSG:32633")


def test_the_results_are_written(tmp_path, fleeing):
    document, log, out = _run(_survey(tmp_path, fleeing))

    with open(os.path.join(out, "drone_reaction.json"),
              encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["summary"][drone_reaction.ALL]["verdict"] == \
        drone_reaction.FLED
    assert len(stored["tracks"]) == 8
    with open(os.path.join(out, "drone_reaction_tracks.csv"),
              encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == drone_reaction.TRACK_COLUMNS
    assert (rows[0]["flight"], rows[0]["track_id"], rows[0]["status"]) == (
        "Meadow", "1", "moved")
    assert log[-1] == "Reaction to the drone written to " + os.path.join(
        out, "drone_reaction.json")


def test_how_far_counts_as_moving_can_be_set(tmp_path, fleeing):
    strict, _log, _out = _run(_survey(tmp_path, fleeing),
                              reaction_min_displacement_m=10.0)

    # 0.5 m/s for 19 s is 9.5 m: no longer far enough
    rows = _by_track(strict)
    assert rows[8]["status"] == drone_reaction.STAYED
    assert rows[1]["status"] == drone_reaction.MOVED
    assert strict["settings"]["min_displacement_m"] == 10.0
    assert "at least 10 m" in strict["notes"]


def test_how_fast_counts_as_moving_can_be_set(tmp_path, fleeing):
    loose, _log, _out = _run(_survey(tmp_path, fleeing))
    strict, _log, _out = _run(_survey(tmp_path, fleeing),
                              reaction_min_speed_ms=0.6)

    # 0.5 m/s is far enough over 19 s, and no longer fast enough
    assert _by_track(loose)[8]["status"] == drone_reaction.MOVED
    assert _by_track(strict)[8]["status"] == drone_reaction.STAYED
    assert _by_track(strict)[8]["speed_ms"] == pytest.approx(0.5, abs=0.05)
    assert strict["settings"]["min_speed_ms"] == 0.6
    assert "at least 3 m, at 0.6 m/s or faster, during" in strict["notes"]
    assert "m/s or faster" not in loose["notes"]


def test_a_species_is_judged_by_its_own_thresholds(tmp_path, fleeing):
    """Red deer and roe deer that did the same: 0.6 m/s away from the
    line. With a speed of its own, only the one species stayed."""
    folder = str(tmp_path / "Forest")
    dem = _flight(folder, [animal(n + 1, 100 + 120 * n, 15.0, (0.0, 0.6),
                                  species="red deer") for n in range(6)])
    survey = _survey(tmp_path, fleeing, (folder, dem))

    document, _log, _out = _run(survey, reaction_species_thresholds={
        "Red Deer": {"min_speed_ms": 1.0}})

    by_species = {}
    for row in document["tracks"]:
        by_species.setdefault(row["species"], set()).add(row["status"])
    assert by_species["red deer"] == {drone_reaction.STAYED}
    assert drone_reaction.MOVED in by_species["roe deer"]
    assert document["settings"]["per_species"] == {
        "red deer": {"min_speed_ms": 1.0}}
    assert document["notes"].endswith(
        "Species with thresholds of their own - red deer: at least 3 m at "
        "1 m/s or faster.")


def test_a_distance_of_a_species_own(tmp_path, fleeing):
    document, _log, _out = _run(
        _survey(tmp_path, fleeing),
        reaction_species_thresholds={"roe deer": {"min_displacement_m": 10}})

    # 0.5 m/s for 19 s is 9.5 m: far enough for everyone, not for roe deer
    assert _by_track(document)[8]["status"] == drone_reaction.STAYED
    assert document["settings"]["min_displacement_m"] == 3.0


def test_a_species_threshold_that_is_none_is_refused(tmp_path, fleeing):
    with pytest.raises(RuntimeError, match="speed for 'roe deer'"):
        _run(_survey(tmp_path, fleeing), reaction_species_thresholds={
            "roe deer": {"min_speed_ms": -1}})


@pytest.mark.parametrize("value", [-0.1, "fast"])
def test_a_speed_that_is_none_is_refused(tmp_path, fleeing, value):
    with pytest.raises(RuntimeError, match="speed an animal has to"):
        _run(_survey(tmp_path, fleeing), reaction_min_speed_ms=value)


@pytest.mark.parametrize("value", [0, -2, "far"])
def test_a_distance_that_is_none_is_refused(tmp_path, fleeing, value):
    with pytest.raises(RuntimeError, match="distance an animal has to"):
        _run(_survey(tmp_path, fleeing), reaction_min_displacement_m=value)


def test_a_glimpse_is_not_judged(tmp_path):
    folder = str(tmp_path / "Glimpse")
    dem = _flight(folder, [animal(1, 100, 15.0, (0.0, 0.8)),
                           animal(2, 300, 15.0, (0.0, 0.8), frames=2),
                           {"id": 3, "species": "roe deer", "at": (500, 9),
                            "frames": (98, 102)}])

    document, _log, _out = _run(_survey(tmp_path, (folder, dem)))
    rows = _by_track(document)

    assert [rows[n]["status"] for n in (1, 2, 3)] == [
        drone_reaction.MOVED, drone_reaction.NOT_JUDGED,
        drone_reaction.NOT_JUDGED]
    assert document["summary"][drone_reaction.ALL]["not_judged"] == 2
    assert document["summary"][drone_reaction.ALL]["verdict"] == \
        drone_reaction.UNKNOWN


def test_without_the_dem_the_line_is_not_known(tmp_path):
    folder = str(tmp_path / "NoDem")
    _flight(folder, FLEEING, dem=False)

    document, log, _out = _run(_survey(tmp_path, (folder, "")))
    rows = _by_track(document)

    assert rows[1]["status"] == drone_reaction.MOVED
    assert rows[1]["direction"] is None and rows[1]["away_m"] is None
    assert document["summary"][drone_reaction.ALL]["moved"] == 7
    assert document["summary"][drone_reaction.ALL]["verdict"] != \
        drone_reaction.FLED
    assert "The DEM of NoDem is not known" in document["notes"]
    assert any("how far and how fast the animals moved is read, where to "
               "is not" in line for line in log)


def test_frames_without_a_time_cannot_say_how_fast(tmp_path):
    folder = str(tmp_path / "Untimed")
    dem = _flight(folder, FLEEING, timed=False)

    with pytest.raises(RuntimeError, match="Untimed: the frames carry no "
                                           "capture time"):
        _run(_survey(tmp_path, (folder, dem)))


def test_several_flights_and_species(tmp_path, fleeing):
    folder = str(tmp_path / "Forest")
    dem = _flight(folder, [animal(n + 1, 100 + 120 * n, 20.0,
                                  species="red deer") for n in range(6)])

    document, _log, _out = _run(_survey(tmp_path, fleeing, (folder, dem)))

    assert list(document["summary"]) == [
        drone_reaction.ALL, "roe deer", "red deer"]
    assert document["summary"]["roe deer"]["verdict"] == drone_reaction.FLED
    assert document["summary"]["red deer"]["verdict"] == drone_reaction.QUIET
    assert list(document["by_flight"]) == ["Meadow", "Forest"]
    assert document["by_flight"]["Forest"]["moved"] == 0
    assert document["by_flight"]["Meadow"]["moved"] == 7
    assert document["flights"] == ["Meadow", "Forest"]


def test_the_species_of_the_survey(tmp_path, fleeing):
    folder = str(tmp_path / "Forest")
    dem = _flight(folder, [animal(n + 1, 100 + 120 * n, 20.0,
                                  species="red deer") for n in range(6)])

    document, _log, _out = _run(
        _survey(tmp_path, fleeing, (folder, dem), species=["red deer"]))

    assert {row["species"] for row in document["tracks"]} == {"red deer"}
    assert document["species_filter"] == ["red deer"]


def test_only_the_animals_in_the_survey_area(tmp_path, fleeing):
    # the plot ends at x = 500 m: the first four animals stand in it
    document, _log, _out = _run(
        _survey(tmp_path, fleeing, area=(0, -200, 500, 200)))

    assert sorted(_by_track(document)) == [1, 2, 3, 4]
    assert document["survey_area"]["adjusted_ha"] == pytest.approx(20.0)


def test_a_survey_without_tracked_animals(tmp_path):
    folder = str(tmp_path / "Empty")
    dem = _flight(folder, [animal(1, 100, 15.0)])

    with pytest.raises(RuntimeError, match="no tracked animals"):
        _run(_survey(tmp_path, (folder, dem), species=["wild boar"]))


def test_cancelling_stops_before_anything_is_written(tmp_path, fleeing):
    from bambi_wildlife_detection.bambi_processing import CancelledException

    survey = _survey(tmp_path, fleeing)

    with pytest.raises(CancelledException):
        BambiProcessor().run_drone_reaction(
            {"survey": survey.to_dict()}, cancel_check=lambda: True)

    assert not os.path.isdir(os.path.join(survey.folder, "analytics_t"))


def test_the_worker_knows_the_step(tmp_path, fleeing):
    survey = _survey(tmp_path, fleeing)
    worker = ProcessingWorker(BambiProcessor(), "drone_reaction",
                              {"survey": survey.to_dict()})
    ended, failed = [], []
    worker.finished.connect(lambda step, ok: ended.append((step, ok)))
    worker.error.connect(lambda *told: failed.append(told))

    worker.run()

    assert failed == []
    assert ended == [("drone_reaction", True)]
    assert os.path.isfile(os.path.join(
        survey.folder, "analytics_t", "drone_reaction.json"))
