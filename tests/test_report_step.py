# -*- coding: utf-8 -*-
"""The Survey Report step: what a page about a survey needs that is in no
result file, and the page written (bambi_survey_report.py on top of
core/survey_report.py)."""
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest

from bambi_wildlife_detection import bambi_survey_report
from bambi_wildlife_detection.bambi_processing import (
    BambiProcessor, CancelledException, ProcessingWorker)
from bambi_wildlife_detection.core import (
    environment, survey_area, survey_report)
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight

shapely = pytest.importorskip("shapely")
pytest.importorskip("scipy")

MAY = datetime(2024, 5, 3, 5, 10, tzinfo=timezone(timedelta(hours=2)))
#: The meadow: 400 m x 250 m, seen whole by a flight - ten hectares.
SEEN = [(0, 0, 200, 250), (200, 0, 400, 250)]
MEADOW = (0, 0, 400, 250)
#: The smallest PNG there is: one pixel.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000001e221bc330000000049454e44ae426082")


def deer(count, species="roe deer", start=1, row=0, **more):
    """*count* animals in a row, 30 m apart."""
    return [dict({"id": start + n, "species": species,
                  "at": (20 + 30 * n, 40 + 60 * row),
                  "frames": (5 * n, 5 * n + 3)}, **more)
            for n in range(count)]


def _flight(folder, tracks, dem=True, **more):
    """A flight in *folder*; with *dem*, the origin its poses are shifted
    by is known. Returns ``(folder, dem path)``."""
    make_flight(folder, tracks, seen=SEEN, **more)
    if not dem:
        return folder, ""
    path = os.path.join(folder, "dem.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"origin": [ORIGIN[0], ORIGIN[1], 0.0]}, handle)
    return folder, path


def _survey(tmp_path, *flights, area=MEADOW, **more):
    survey_folder = str(tmp_path / "Hintertal")
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
    """``(page, log, path)`` of the report of *survey*."""
    log = []
    config = dict(survey=survey.to_dict(), **settings)
    path = BambiProcessor().run_survey_report(config, log_fn=log.append)
    with open(path, encoding="utf-8") as handle:
        return handle.read(), log, path


def _drawing(page, cls="map"):
    """The parsed SVG of the page's map."""
    found = re.search(rf"<svg[^>]*class='{cls}'.*?</svg>", page, flags=re.S)
    assert found, "no such drawing in the page"
    return ET.fromstring(found.group(0))


def _of_class(root, cls):
    return [el for el in root.iter() if el.get("class") == cls]


@pytest.fixture
def may(tmp_path):
    """Three roe deer and a group of two red deer, in May."""
    return _flight(str(tmp_path / "May"), deer(3) + deer(
        2, "red deer", start=4, row=1), start=MAY)


@pytest.fixture
def june(tmp_path):
    return _flight(str(tmp_path / "June"), deer(5),
                   start=MAY + timedelta(days=31))


# ---------------------------------------------------------------------------
# The page of a survey
# ---------------------------------------------------------------------------

def test_the_report_is_one_file_in_the_surveys_analytics(tmp_path, may, june):
    survey = _survey(tmp_path, may, june)

    page, log, path = _run(survey)

    assert path == os.path.join(survey.folder, "analytics_t",
                                "survey_report.html")
    assert page.startswith("<!DOCTYPE html>")
    assert "<script" not in page and "<link" not in page
    assert log[0] == ("Writing the survey report (Thermal, tracks) of May, "
                      "June...")
    assert log[-1] == f"Survey report written to {path}"


def test_it_says_what_was_flown_and_when(tmp_path, may, june):
    page, _log, _path = _run(_survey(tmp_path, may, june))

    assert "<h1>Survey report - Hintertal</h1>" in page
    assert "<td>Camera</td><td>Thermal</td>" in page
    assert "<td>Coordinate system</td><td>EPSG:32633</td>" in page
    assert "<td>Survey area</td><td>10.00 ha</td>" in page
    # the local day and hour of the frames
    may_row = re.search(r"<tr><td>May</td>.*?</tr>", page).group(0)
    assert may_row.startswith(
        "<tr><td>May</td><td>2024-05-03</td><td>05:10-05:1")
    assert "<td>June</td><td>2024-06-03</td>" in page
    # five animals in May, five in June
    assert re.search(r"<td>May</td>.*?<td class='n'>5</td></tr>", page)
    version = bambi_survey_report.plugin_version()
    assert re.match(r"\d+\.\d+\.\d+$", version)
    assert f"Made with BAMBI - QGIS Integration {version} on " in page


def test_the_census_is_built_where_it_is_missing(tmp_path, may, june):
    survey = _survey(tmp_path, may, june)

    page, log, _path = _run(survey)

    assert os.path.isfile(os.path.join(survey.folder, "analytics_t",
                                       "census_summary_tracks.json"))
    assert any(line.startswith("The census has not been built for this "
                               "survey yet") for line in log)
    assert "<h2>Counts</h2>" in page
    assert "<tr class='total'><td>All animals</td><td class='n'>10</td>" \
        in page
    assert "<td>roe deer</td><td class='n'>8</td>" in page
    assert "<td>red deer</td><td class='n'>2</td>" in page


def test_a_census_that_is_there_is_taken_as_it_is(tmp_path, may):
    survey = _survey(tmp_path, may)
    processor = BambiProcessor()
    processor.run_census_summary({"survey": survey.to_dict()})
    census_file = os.path.join(survey.folder, "analytics_t",
                               "census_summary_tracks.json")
    with open(census_file, encoding="utf-8") as handle:
        document = json.load(handle)
    document["notes"] = "As it was written before."
    with open(census_file, "w", encoding="utf-8") as handle:
        json.dump(document, handle)

    page, log, _path = _run(survey)

    assert not any("has not been built" in line for line in log)
    assert "As it was written before." in page


def test_a_census_of_other_flights_is_built_anew(tmp_path, may, june):
    """The census was run when only May was ticked. Its five animals beside
    a table of two flights would be a report that contradicts itself."""
    BambiProcessor().run_census_summary(
        {"survey": _survey(tmp_path, may).to_dict()})

    page, log, _path = _run(_survey(tmp_path, may, june))

    assert ("The census that is there was made from other flights: May - "
            "it is built first, since the counts are what a report starts "
            "from.") in log
    assert "<tr class='total'><td>All animals</td><td class='n'>10</td>" \
        in page
    assert "class='warn'" not in page


def test_a_census_for_other_species_is_built_anew(tmp_path, may):
    BambiProcessor().run_census_summary(
        {"survey": _survey(tmp_path, may, species=["red deer"]).to_dict()})

    page, log, _path = _run(_survey(tmp_path, may))

    assert any(line.startswith("The census that is there was made for "
                               "other species: red deer - it is built first")
               for line in log)
    assert "<td>roe deer</td><td class='n'>3</td>" in page


def test_a_census_that_cannot_be_read_is_built_anew(tmp_path, may):
    survey = _survey(tmp_path, may)
    out = os.path.join(survey.folder, "analytics_t")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "census_summary_tracks.json"), "w") as handle:
        handle.write("{ not json")

    page, log, _path = _run(survey)

    assert any(line.startswith("The census that is there cannot be read - "
                               "it is built first") for line in log)
    assert "<tr class='total'><td>All animals</td><td class='n'>5</td>" \
        in page


def test_another_result_of_other_flights_is_named_not_made_anew(
        tmp_path, may, june):
    """Only the census is the report's to build. A reaction to the drone
    worked out for May alone stays what it is - and is said to be."""
    BambiProcessor().run_drone_reaction(
        {"survey": _survey(tmp_path, may).to_dict()})

    page, log, _path = _run(_survey(tmp_path, may, june))

    assert ("Note: Reaction to the drone was made from other flights: May - "
            "not what this report lists. Run it again, then write the "
            "report anew.") in log
    assert "<p class='warn'>Not of one piece: Reaction to the drone was " \
        "made when the survey had other flights or species" in page
    assert "<td>drone_reaction.json</td>" in page
    assert "<td>made from other flights: May</td>" in page


def test_what_was_run_is_in_it_and_what_was_not_is_named(tmp_path, may, june):
    survey = _survey(tmp_path, may, june)
    processor = BambiProcessor()
    config = {"survey": survey.to_dict()}
    processor.run_survey_trend(config)
    processor.run_drone_reaction(config)

    page, log, _path = _run(survey)

    headings = re.findall(r"<h2>(.*?)</h2>", page)
    assert "Over time" in headings and "Reaction to the drone" in headings
    assert "How good the places are" not in headings
    included = [line for line in log if line.startswith("In the report: ")]
    assert included == ["In the report: Census summary, Groups, Compare "
                        "visits, Reaction to the drone"]
    not_run = [line for line in log if line.startswith("Not run")][0]
    assert not_run == ("Not run, and so not in it: Grid density, Distance "
                       "sampling, Population estimation (transects), "
                       "Habitat context, Position uncertainty")
    assert "Not run, and so not in this report: Grid density, Distance " \
        "sampling" in page
    assert "<td>Compare visits</td><td>trend_tracks.json</td>" in page


def test_boxes_have_a_report_of_their_own(tmp_path, may):
    survey = _survey(tmp_path, may)

    page, log, _path = _run(survey, report_source="detections")

    assert log[0].startswith("Writing the survey report (Thermal, "
                             "detections)")
    assert "boxes - an animal counts once per frame it is on" in page
    assert os.path.isfile(os.path.join(survey.folder, "analytics_t",
                                       "census_summary_detections.json"))
    assert "<td>Census summary</td><td>census_summary_detections.json</td>" \
        in page


def test_the_species_the_survey_is_about(tmp_path, may):
    survey = _survey(tmp_path, may, species=["roe deer"])

    page, _log, _path = _run(survey)

    assert "<td>Species</td><td>roe deer</td>" in page
    assert "<tr class='total'><td>All animals</td><td class='n'>3</td>" \
        in page
    assert "red deer" not in page


def test_a_survey_without_an_area(tmp_path, may):
    page, _log, _path = _run(_survey(tmp_path, may, area=None))

    assert "none drawn: the whole flights count" in page
    assert "<th class='n'>Counted</th>" in page
    assert _of_class(_drawing(page), "plot") == []


# ---------------------------------------------------------------------------
# The title and the picture of the map
# ---------------------------------------------------------------------------

def test_the_title_can_be_given(tmp_path, may):
    page, _log, _path = _run(_survey(tmp_path, may),
                             report_title="  Rehe im Hintertal <2024>  ")

    assert "<title>Rehe im Hintertal &lt;2024&gt;</title>" in page
    assert "<h1>Rehe im Hintertal &lt;2024&gt;</h1>" in page


def test_the_picture_of_the_map_goes_into_the_page(tmp_path, may):
    picture = tmp_path / "canvas.png"
    picture.write_bytes(PNG)

    page, _log, _path = _run(_survey(tmp_path, may),
                             report_canvas_png=str(picture))

    import base64
    assert ("<img alt='The map as it was shown in QGIS' "
            f"src='data:image/png;base64,{base64.b64encode(PNG).decode()}'"
            ) in page


def test_a_picture_that_is_not_there_is_left_out(tmp_path, may):
    page, _log, _path = _run(_survey(tmp_path, may),
                             report_canvas_png=str(tmp_path / "none.png"))

    assert "<img" not in page


def test_a_picture_too_large_for_a_page_is_left_out(tmp_path, may,
                                                    monkeypatch):
    picture = tmp_path / "canvas.png"
    picture.write_bytes(PNG + b"\0" * 2000)
    monkeypatch.setattr(bambi_survey_report, "MAX_CANVAS_BYTES", 1000)

    page, log, _path = _run(_survey(tmp_path, may),
                            report_canvas_png=str(picture))

    assert "<img" not in page
    assert ("The picture of the map is 0.0 MB: too large for the page, "
            "which keeps the drawn map only") in log


# ---------------------------------------------------------------------------
# The drawn map
# ---------------------------------------------------------------------------

def test_the_map_has_the_area_the_ground_seen_the_routes_and_the_animals(
        tmp_path, may, june):
    page, _log, _path = _run(_survey(tmp_path, may, june))

    drawing = _drawing(page)
    assert len(_of_class(drawing, "plot")) == 1
    assert len(_of_class(drawing, "seen")) == 1
    routes = _of_class(drawing, "route")
    assert [route[0].text for route in routes] == ["May", "June"]
    animals = _of_class(drawing, "animal")
    assert len(animals) == 10
    assert sorted({animal[0].text for animal in animals}) == [
        "red deer", "roe deer"]
    # each species in one colour, and the two in different ones
    colours = {animal[0].text: animal.get("fill") for animal in animals}
    assert len(set(colours.values())) == 2
    for sign in ("roe deer", "red deer", "route flown", "survey area"):
        assert f"</i>{sign}</span>" in page


def test_the_route_is_where_the_drone_flew(tmp_path, may):
    """The drone flies due east from the origin, 5 m a frame: a line from
    the left of the map to its right, level."""
    page, _log, _path = _run(_survey(tmp_path, may))

    route = _of_class(_drawing(page), "route")[0]
    points = [tuple(float(v) for v in pair.split(","))
              for pair in route.get("points").split()]
    assert points[0][0] < points[-1][0]
    assert {round(y) for _x, y in points} == {round(points[0][1])}
    km = re.search(r"<td>May</td><td>.*?</td><td>.*?</td>"
                   r"<td class='n'>(\d+)</td><td class='n'>([\d.]+)</td>",
                   page)
    frames, flown = int(km.group(1)), float(km.group(2))
    assert flown == pytest.approx((frames - 1) * 5 / 1000.0, abs=0.006)


def test_a_flight_whose_origin_is_not_known_has_no_route_on_the_map(
        tmp_path):
    flight = _flight(str(tmp_path / "May"), deer(3), dem=False, start=MAY)

    page, log, _path = _run(_survey(tmp_path, flight))

    assert _of_class(_drawing(page), "route") == []
    # how far it flew is known all the same
    assert re.search(r"<td>May</td>.*?<td class='n'>[1-9]\d*</td>"
                     r"<td class='n'>\d\.\d\d</td>", page)
    assert "</i>route flown</span>" not in page
    assert ("Not on the map: the route of May. A flight&#x27;s camera "
            "positions are placed by its elevation model, which is not set "
            "for it.") in page
    assert ("The route of May is left off the map: its camera positions "
            "are placed by its elevation model, which is not set") in log
    # the animals and the ground seen are placed without it
    assert len(_of_class(_drawing(page), "animal")) == 3
    assert len(_of_class(_drawing(page), "seen")) == 1


def test_a_long_route_is_drawn_with_fewer_points(tmp_path, may, monkeypatch):
    monkeypatch.setattr(bambi_survey_report, "MAX_ROUTE_POINTS", 4)

    page, _log, _path = _run(_survey(tmp_path, may))

    points = _of_class(_drawing(page), "route")[0].get("points").split()
    assert 4 <= len(points) <= 6


def test_too_many_animals_for_a_page_are_thinned_out(tmp_path, may,
                                                     monkeypatch):
    monkeypatch.setattr(bambi_survey_report, "MAX_MAP_ANIMALS", 2)

    page, log, _path = _run(_survey(tmp_path, may))

    assert len(_of_class(_drawing(page), "animal")) == 2
    assert ("The map shows one in 3 of the 5 animals: all of them would "
            "make the page slow to open") in log
    # the counts are those of all of them
    assert "<tr class='total'><td>All animals</td><td class='n'>5</td>" \
        in page


def test_animals_outside_the_survey_area_are_not_on_the_map(tmp_path, may):
    """An area around the first two of the three roe deer - they stand
    at 20, 50 and 80 m; the red deer are a row further north."""
    survey = _survey(tmp_path, may, area=(0, 0, 60, 70))

    page, _log, _path = _run(survey)

    assert len(_of_class(_drawing(page), "animal")) == 2
    assert re.search(r"<td>May</td>.*?<td class='n'>2</td></tr>", page)


# ---------------------------------------------------------------------------
# Weather and settings
# ---------------------------------------------------------------------------

def test_the_weather_of_the_flights_that_have_it(tmp_path, may, june):
    environment.write_metadata(may[0], environment.WEATHER, {
        "at_flight": {"temperature_2m": 7.5, "wind_speed_10m": 2.25,
                      "cloud_cover": 40},
        "previous_24h": {"precipitation_sum": 0.0}})

    page, _log, _path = _run(_survey(tmp_path, may, june))

    assert "<th class='n'>Air °C</th><th class='n'>Wind m/s</th>" \
        "<th class='n'>Cloud %</th><th class='n'>Rain 24 h mm</th>" in page
    assert re.search(r"<td>May</td>.*?<td class='n'>7.5</td>"
                     r"<td class='n'>2.2</td><td class='n'>40</td>"
                     r"<td class='n'>0.0</td></tr>", page)
    assert re.search(r"<td>June</td>.*?<td class='n'>–</td>"
                     r"<td class='n'>–</td><td class='n'>–</td>"
                     r"<td class='n'>–</td></tr>", page)
    assert "No weather is known" not in page


def test_without_weather_the_page_says_how_to_have_it(tmp_path, may):
    page, _log, _path = _run(_survey(tmp_path, may))

    assert "No weather is known for these flights" in page


def test_what_made_each_flights_data(tmp_path, may):
    page, _log, _path = _run(_survey(tmp_path, may))

    settings = page[page.index("<h2>Settings</h2>"):]
    assert "<h3>May</h3>" in settings
    assert "<td>Elevation model</td><td>dem.json</td>" in settings
    assert "<td>Tracker</td>" in settings
    assert re.search(r"<td>Boxes of the \w+</td><td>\d+ - ", settings)


# ---------------------------------------------------------------------------
# What can go wrong
# ---------------------------------------------------------------------------

def test_a_result_that_cannot_be_read_costs_its_part_only(tmp_path, may):
    survey = _survey(tmp_path, may)
    out = os.path.join(survey.folder, "analytics_t")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "census_summary_tracks.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"rows": [{"flight": "May", "species": "roe deer"}],
                   "flights": ["May"], "plot_ha": 10.0}, handle)

    page, log, path = _run(survey)

    assert os.path.isfile(path)
    assert ("Note: 'Counts' could not be made from its result file "
            "(KeyError: 'inside'). Run its analysis again, then write the "
            "report anew.") in log
    assert "<h2>Counts</h2><p class='note'>This part could not be made" \
        in page
    assert "<h2>The survey</h2>" in page and "<h2>Settings</h2>" in page


def test_cancelled_before_anything_is_written(tmp_path, may):
    survey = _survey(tmp_path, may)

    with pytest.raises(CancelledException):
        BambiProcessor().run_survey_report(
            {"survey": survey.to_dict()}, cancel_check=lambda: True)

    assert not os.path.isfile(os.path.join(
        survey.folder, "analytics_t", survey_report.FILE_NAME))


def test_progress_runs_to_the_end(tmp_path, may, june):
    survey = _survey(tmp_path, may, june)
    told = []

    BambiProcessor().run_survey_report({"survey": survey.to_dict()},
                                       progress_fn=told.append)

    assert told == sorted(told) and told[0] == 3 and told[-1] == 100


def test_the_worker_knows_the_step(tmp_path, may):
    survey = _survey(tmp_path, may)
    worker = ProcessingWorker(BambiProcessor(), "survey_report",
                              {"survey": survey.to_dict()})
    ended, failed = [], []
    worker.finished.connect(lambda step, ok: ended.append((step, ok)))
    worker.error.connect(lambda *told: failed.append(told))

    worker.run()

    assert failed == []
    assert ended == [("survey_report", True)]
    assert os.path.isfile(os.path.join(
        survey.folder, "analytics_t", "survey_report.html"))
