# -*- coding: utf-8 -*-
"""The Compare Visits step: a survey's flights, visit by visit
(bambi_survey_trend.py on top of core/survey_trend.py)."""
import csv
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from bambi_wildlife_detection.bambi_processing import (
    BambiProcessor, ProcessingWorker)
from bambi_wildlife_detection.core import survey_area, survey_trend
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight

shapely = pytest.importorskip("shapely")

MAY = datetime(2024, 5, 1, 5, 0, tzinfo=timezone(timedelta(hours=2)))
#: The meadow: 400 m x 250 m, seen whole by a flight - ten hectares.
SEEN = [(0, 0, 200, 250), (200, 0, 400, 250)]


def deer(count, species="roe deer", start=1, row=0, **more):
    """*count* animals in a row, 30 m apart."""
    return [dict({"id": start + n, "species": species,
                  "at": (20 + 30 * n, 40 + 60 * row),
                  "frames": (5 * n, 5 * n + 3)}, **more)
            for n in range(count)]


def _survey(tmp_path, *folders, area=None, **more):
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
                                   target_folder=folder, epsg=EPSG)
                      for folder in folders], **more)


def _run(survey, **settings):
    log = []
    config = dict(survey=survey.to_dict(), **settings)
    document = BambiProcessor().run_survey_trend(config, log_fn=log.append)
    return document, log, os.path.join(survey.folder, "analytics_t")


@pytest.fixture
def visits(tmp_path):
    """The meadow in May, in June, and a year on: three, five and eight
    roe deer - and in May a group of two red deer."""
    return [
        make_flight(str(tmp_path / "May"), deer(3) + deer(
            2, "red deer", start=4, row=1), seen=SEEN, start=MAY),
        make_flight(str(tmp_path / "June"), deer(5), seen=SEEN,
                    start=MAY + timedelta(days=31)),
        make_flight(str(tmp_path / "Next May"), deer(8), seen=SEEN,
                    start=MAY + timedelta(days=366)),
    ]


def test_the_flights_are_compared_visit_by_visit(tmp_path, visits):
    document, log, _out = _run(_survey(tmp_path, *visits))

    assert [(v["label"], v["flights"]) for v in document["visits"]] == [
        ("2024-05-01", ["May"]), ("2024-06-01", ["June"]),
        ("2025-05-02", ["Next May"])]
    assert [v["looked_at_ha"] for v in document["visits"]] == pytest.approx(
        [10.0, 10.0, 10.0])
    roe = document["series"]["roe deer"]
    assert [entry["animals"] for entry in roe] == [3, 5, 8]
    assert [entry["density_per_km2"] for entry in roe] == pytest.approx(
        [30.0, 50.0, 80.0])
    red = document["series"]["red deer"]
    assert [entry["animals"] for entry in red] == [2, 0, 0]
    # the two red deer stood 30 m apart: one group, one sighting
    assert (red[0]["groups"], red[0]["largest_group"]) == (1, 2)
    assert document["species"][-1] == survey_trend.ALL
    assert document["source"] == "tracks" and document["camera"] == "Thermal"
    assert document["crs"] == "EPSG:32633"
    assert (document["gap_days"], document["ground"], document["basis"]) == (
        0, "seen", "ground")
    assert log[0] == ("Comparing the visits (Thermal, tracks) of May, June, "
                      "Next May...")
    assert "3 visits: 2024-05-01, 2024-06-01, 2025-05-02" in log
    assert any(line.startswith("roe deer: 30.00 -> 50.00 -> 80.00 /km². +")
               for line in log)


def test_the_local_day_of_the_frames_decides(tmp_path):
    """Two flights on the first of May, summer time: at half past midnight
    and in the evening - two days by the clock of the flight log."""
    night = MAY.replace(hour=0, minute=30)
    folders = [
        make_flight(str(tmp_path / "Night"), deer(2), seen=SEEN, start=night),
        make_flight(str(tmp_path / "Evening"), deer(4), seen=SEEN,
                    start=MAY.replace(hour=20)),
        make_flight(str(tmp_path / "Later"), deer(3), seen=SEEN,
                    start=MAY + timedelta(days=20)),
    ]

    document, _log, _out = _run(_survey(tmp_path, *folders))

    assert [v["flights"] for v in document["visits"]] == [
        ["Night", "Evening"], ["Later"]]
    assert document["visits"][0]["time_of_day"] == "00:30-20:00"
    # two passes over the same ground: six animals on twenty hectares
    assert document["series"]["roe deer"][0]["density_per_km2"] == \
        pytest.approx(30.0)


def test_the_gap_between_visits_can_be_set(tmp_path, visits):
    document, _log, _out = _run(_survey(tmp_path, *visits),
                                trend_gap_days=40)

    assert [v["flights"] for v in document["visits"]] == [
        ["May", "June"], ["Next May"]]
    assert document["visits"][0]["label"] == "2024-05-01 to 2024-06-01"
    assert document["notes"].startswith(
        "A visit is the flights flown no more than 40 day(s) after the one "
        "before.")


def test_the_results_are_written(tmp_path, visits):
    document, log, out = _run(_survey(tmp_path, *visits))

    with open(os.path.join(out, "trend_tracks.json"),
              encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["species"] == document["species"]
    assert stored["trends"]["roe deer"]["verdict"] == \
        document["trends"]["roe deer"]["verdict"]
    with open(os.path.join(out, "trend_tracks_visits.csv"),
              encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert (rows[0]["date"], rows[0]["species"], rows[0]["animals"]) == (
        "2024-05-01", "roe deer", "3")
    assert rows[0]["flights"] == "May"
    assert rows[1]["change_pct"] == "66.6667"
    assert log[-1] == "Visits written to " + os.path.join(
        out, "trend_tracks.json")


def test_a_survey_area_gives_the_number_in_it(tmp_path, visits):
    # twice the meadow: the flights saw half of the plot
    survey = _survey(tmp_path, *visits, area=(0, 0, 400, 500))

    document, _log, _out = _run(survey)

    may = document["series"]["roe deer"][0]
    assert document["plot_ha"] == pytest.approx(20.0)
    assert may["density_per_km2"] == pytest.approx(30.0)
    assert may["number"] == pytest.approx(6.0)
    assert document["survey_area"]["adjusted_ha"] == pytest.approx(20.0)


def test_only_the_ground_every_visit_saw(tmp_path, visits):
    # June saw the western half only - and what stood there
    short = make_flight(str(tmp_path / "June short"), deer(5),
                        seen=[(0, 0, 200, 250)],
                        start=MAY + timedelta(days=31))

    document, _log, _out = _run(
        _survey(tmp_path, visits[0], short, visits[2]),
        trend_ground="common")

    assert document["common_ha"] == pytest.approx(5.0)
    assert [v["looked_at_ha"] for v in document["visits"]] == pytest.approx(
        [5.0, 5.0, 5.0])
    # the animals stand 30 m apart from x = 20 m: six of eight west of 200 m
    assert [entry["animals"]
            for entry in document["series"]["roe deer"]] == [3, 5, 6]
    assert "Only the ground every visit saw is compared" in document["notes"]


def test_flights_without_footprints_and_no_area_compare_the_counts(
        tmp_path, visits):
    bare = make_flight(str(tmp_path / "Bare"), deer(4),
                       start=MAY + timedelta(days=100))

    document, log, _out = _run(_survey(tmp_path, *visits, bare))

    assert document["basis"] == "count"
    assert [entry["animals"]
            for entry in document["series"]["roe deer"]] == [3, 5, 4, 8]
    assert document["series"]["roe deer"][0]["density_per_km2"] is None
    assert ("Bare: no Thermal field-of-view footprints and no survey area "
            "- the ground this flight saw is not known") in log
    assert "the counts are compared as they are" in document["notes"]
    assert any(line.startswith("roe deer: 3 -> 5 -> 4 -> 8. ")
               for line in log)


def test_a_survey_area_counts_as_seen_by_a_flight_without_footprints(
        tmp_path, visits):
    bare = make_flight(str(tmp_path / "Bare"), deer(4),
                       start=MAY + timedelta(days=100))
    survey = _survey(tmp_path, visits[0], bare, area=(0, 0, 400, 250))

    document, _log, _out = _run(survey)

    assert document["basis"] == "ground"
    assert [v["looked_at_ha"] for v in document["visits"]] == pytest.approx(
        [10.0, 10.0])


def test_one_day_is_nothing_to_compare(tmp_path, visits):
    with pytest.raises(RuntimeError, match="that is one visit"):
        _run(_survey(tmp_path, visits[0]))


def test_a_flight_without_a_date_is_named(tmp_path, visits):
    undated = make_flight(str(tmp_path / "Undated"), deer(2), seen=SEEN,
                          timed=False)

    with pytest.raises(RuntimeError, match="when Undated was flown"):
        _run(_survey(tmp_path, visits[0], undated))


def test_the_species_of_the_survey(tmp_path, visits):
    document, _log, _out = _run(
        _survey(tmp_path, *visits, species=["red deer"]))

    assert document["species"] == ["red deer", survey_trend.ALL]
    assert [entry["animals"]
            for entry in document["series"][survey_trend.ALL]] == [2, 0, 0]
    assert document["species_filter"] == ["red deer"]


def test_counting_boxes_says_what_that_means(tmp_path, visits):
    document, _log, out = _run(_survey(tmp_path, *visits, source="detections"))

    assert document["source"] == "detections"
    # two boxes per animal
    assert [entry["animals"]
            for entry in document["series"]["roe deer"]] == [6, 10, 16]
    assert "Boxes are counted here, not animals" in document["notes"]
    assert os.path.isfile(os.path.join(out, "trend_detections.json"))


def test_the_weather_of_the_visits(tmp_path, visits):
    from bambi_wildlife_detection.core import environment

    environment.write_metadata(visits[0], environment.WEATHER, {
        "at_flight": {"temperature_2m": 9.5, "wind_speed_10m": 1.5}})

    document, _log, out = _run(_survey(tmp_path, *visits))

    assert document["visits"][0]["weather"] == {
        "air_temperature_c": 9.5, "wind_speed_ms": 1.5}
    assert document["visits"][1]["weather"] == {}
    with open(os.path.join(out, "trend_tracks_visits.csv"),
              encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert (rows[0]["air_temperature_c"], rows[1]["air_temperature_c"]) == (
        "9.5", "")


def test_cancelling_stops_before_anything_is_written(tmp_path, visits):
    from bambi_wildlife_detection.bambi_processing import CancelledException

    survey = _survey(tmp_path, *visits)

    with pytest.raises(CancelledException):
        BambiProcessor().run_survey_trend(
            {"survey": survey.to_dict()}, cancel_check=lambda: True)

    assert not os.path.isdir(os.path.join(survey.folder, "analytics_t"))


def test_progress_runs_to_the_end(tmp_path, visits):
    seen = []

    BambiProcessor().run_survey_trend(
        {"survey": _survey(tmp_path, *visits).to_dict()},
        progress_fn=seen.append)

    assert seen == sorted(seen) and seen[0] == 2 and seen[-1] == 100


def test_the_worker_knows_the_step(tmp_path, visits):
    survey = _survey(tmp_path, *visits)
    worker = ProcessingWorker(BambiProcessor(), "survey_trend",
                              {"survey": survey.to_dict()})
    ended = []
    worker.finished.connect(lambda step, ok: ended.append((step, ok)))
    failed = []
    worker.error.connect(lambda *told: failed.append(told))

    worker.run()

    assert failed == []
    assert ended == [("survey_trend", True)]
    assert os.path.isfile(os.path.join(
        survey.folder, "analytics_t", "trend_tracks.json"))
