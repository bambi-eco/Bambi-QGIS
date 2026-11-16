# -*- coding: utf-8 -*-
"""The survey in one page (core/survey_report.py): which results it finds,
what each section says of them, and that the page needs nothing but
itself."""
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest

from bambi_wildlife_detection.core import (
    census, drone_reaction, position_uncertainty, survey_report as sr,
    survey_trend)

MAY = datetime(2024, 5, 3, 5, 10, tzinfo=timezone(timedelta(hours=2)))
#: The flights of the survey, in the order they were flown.
NAMES = ("May", "June", "Next May")


def _records(species, count, group=None, **more):
    return [dict({"species": species, "x": 100.0 + 20 * n, "y": 50.0,
                  "sex": None, "age": None, "approved": False,
                  "group": group}, **more) for n in range(count)]


def census_of(flights=1, covered_m2=300000.0, with_area=True, **more):
    """A census of the first *flights* of :data:`NAMES` over a plot of
    30 ha."""
    class Plot:
        adjusted_m2 = 300000.0

        @staticmethod
        def contains(x, y):
            return True

    red_deer = _records("red deer", 2, group=1, age="adult")
    entries = [{"name": NAMES[n], "covered_m2": covered_m2,
                "records": [*_records("roe deer", 6 + n, sex="female"),
                            *red_deer]}
               for n in range(flights)]
    document = census.build(entries, Plot() if with_area else None)
    document.update({"source": "tracks", "notes": "Minimum counts.",
                     "group_distances_m": {
                         entry["name"]: {"roe deer": 50.0, "red deer": 50.0}
                         for entry in entries}}, **more)
    return document


def trend_of():
    from shapely.geometry import box

    ground = box(0, 0, 1000, 300)
    flights = [{"name": name, "start": MAY + timedelta(days=days),
                "end": MAY + timedelta(days=days, minutes=20),
                "seen": ground, "records": _records("roe deer", count)}
               for name, days, count in zip(NAMES, (0, 31, 366), (6, 9, 15))]
    document = survey_trend.analyse(flights)
    document["notes"] = "A visit is the flights flown on the same day."
    return document


GRID = {
    "grid_text": "hexagons of 1 ha (edge 62.0 m)", "min_seen": 0.5,
    "summary": {"cells": 42, "counted": 27},
    "statistics": {
        "all": {"animals": 52,
                "density_per_km2": {"estimate": 67.98, "low": 50.09,
                                    "high": 87.71},
                "total_in_survey_area": {"estimate": 20.4, "low": 15.0,
                                         "high": 26.3},
                "dispersion": {"index": 1.07, "pattern": "random"},
                "occupancy": {"share": 0.89}, "hot_spots": 4},
        "roe deer": {"animals": 37,
                     "density_per_km2": {"estimate": 47.06, "low": 33.28,
                                         "high": 63.09},
                     "total_in_survey_area": {}, "dispersion": {},
                     "occupancy": {}, "hot_spots": 0}},
    "notes": "A cell's density is its animals over the ground seen in it."}
DISTANCE = {
    "source": "tracks", "n": 54, "best_model": "half-normal + canopy cover",
    "truncation_m": 60.0, "effective_strip_width_m": 31.4,
    "density_per_km2": 92.4, "density_ci95": [61.0, 140.1],
    "abundance_in_survey_area": 27.7,
    "abundance_survey_area_ci95": [18.3, 42.0],
    "notes": "The detection function depends on canopy cover."}
POPULATION = {
    "n_transects": 12, "total_ha": 41.5, "total_count": 30,
    "estimates": {"ZINB": {"density_per_100ha": 71.2, "ci95": [40.0, 126.0]},
                  "Poisson": {"density_per_100ha": None, "error": "failed"}}}
HABITAT = {
    "source": "tracks", "radius_m": 30.0, "n_tracks": 54,
    "layers": {"tree_cover": {
        "title": "Tree cover", "unit": "%",
        "bin_edges": [0, 25, 50, 75, 100],
        "available_counts": [40, 30, 20, 10], "used_counts": [2, 6, 9, 4],
        "available": {"mean": 37.5}, "used": {"mean": 55.8}}},
    "landcover": {"n_tracks": 54, "classes": [
        {"name": "Tree cover", "available_share": 0.46, "used_share": 0.52,
         "selection_ratio": 1.13},
        {"name": "Cropland", "available_share": 0.14, "used_share": None,
         "selection_ratio": None}]},
    "note": "Track values are the mean of each layer around the track."}


def reaction_of():
    rows = [{"flight": "May", "track_id": n, "species": "roe deer",
             "status": drone_reaction.MOVED, "direction": drone_reaction.AWAY,
             "away_ms": 0.6, "line_distance_m": 10.0 + n, "speed_ms": 0.6,
             "displacement_m": 11.0} for n in range(8)]
    rows += [{"flight": "May", "track_id": 20 + n, "species": "roe deer",
              "status": drone_reaction.STAYED, "direction": None,
              "away_ms": 0.01 * (n - 3), "line_distance_m": 40.0 + n,
              "speed_ms": 0.02, "displacement_m": 0.3} for n in range(6)]
    return {"summary": drone_reaction.summarise(rows),
            "settings": {"min_displacement_m": 3.0},
            "notes": "A constant velocity is fitted."}


def position_of():
    rows, _ = position_uncertainty.box_rows(
        "May", [{"detection_id": n, "track_id": 1, "frame": n, "x": 5.0 * n,
                 "y": 20.0, "z": 0.0} for n in range(6)],
        {n: (5.0 * n, 0.0, 80.0) for n in range(6)},
        position_uncertainty.Accuracy())
    from dataclasses import asdict
    return {"summary": position_uncertainty.summarise(rows),
            "accuracy": asdict(position_uncertainty.Accuracy()),
            "notes": "A detection's place is where the line of sight meets "
                     "the terrain model."}


def results_of(**documents):
    """Results as :func:`collect` hands them on; a list gives one entry
    per ``(species, document)``."""
    found = {}
    for key in sr.RESULTS:              # in the order collect() finds them
        if key not in documents:
            continue
        value = documents[key]
        entries = value if isinstance(value, list) else [("", value)]
        found[key] = [{"species": species, "document": document,
                       "file": f"{key}.json", "written": "2026-11-16 09:30"}
                      for species, document in entries]
    missing = [title for key, (_file, title) in sr.RESULTS.items()
               if key not in found]
    return {"found": found, "missing": missing}


def report_of(results=None, flights=2, **more):
    """A report of the first *flights* of :data:`NAMES`."""
    report = {
        "title": "Survey report - Hintertal", "created": "2026-11-16 09:31",
        "plugin_version": "9.1.0", "camera": "Thermal", "source": "tracks",
        "crs": "EPSG:32633", "species_filter": None,
        "survey_area": {"adjusted_ha": 30.0, "exclusions": []},
        "flights": [
            {"name": "May", "date": "2024-05-03", "time": "05:10-05:30",
             "frames": 220, "route_km": 1.09, "animals": 8,
             "weather": {"air_temperature_c": 8.0, "wind_speed_ms": 2.1},
             "settings": {"Tracker": "builtin",
                          "Elevation model": "dem.glb"}},
            {"name": "June", "date": "2024-06-03", "time": "05:10-05:30",
             "frames": 230, "route_km": 1.12, "animals": 9, "weather": {},
             "settings": {}},
            {"name": "Next May", "date": "2025-05-04",
             "time": "05:10-05:30", "frames": 225, "route_km": 1.10,
             "animals": 15, "weather": {}, "settings": {}}][:flights],
        "map": {
            "area": [[[0, 0], [1000, 0], [1000, 300], [0, 300], [0, 0]]],
            "exclusions": [[[400, 100], [500, 100], [500, 200], [400, 100]]],
            "seen": [[[0, 50], [1000, 50], [1000, 250], [0, 250], [0, 50]]],
            "routes": [{"flight": "May",
                        "points": [[0, 150], [500, 150], [1000, 150]]}],
            "animals": [{"x": 200.0, "y": 120.0, "species": "roe deer"},
                        {"x": 700.0, "y": 180.0, "species": "red deer"}],
            "colours": {"roe deer": "#0072b2", "red deer": "#e69f00"}},
        "canvas_png": None,
        "results": results if results is not None else results_of(),
    }
    report.update(more)
    return report


def svg_root(text):
    """The parsed SVG - which it has to be, to be drawn at all."""
    assert text.startswith("<svg") and text.endswith("</svg>")
    return ET.fromstring(text)


def tags(root, name):
    return [el for el in root.iter() if el.tag.split("}")[-1] == name]


# ---------------------------------------------------------------------------
# What the analytics wrote
# ---------------------------------------------------------------------------

class TestCollect:
    @pytest.fixture
    def folder(self, tmp_path):
        out = tmp_path / "analytics_t"
        out.mkdir()
        for name, document in (
                ("census_summary_tracks.json", census_of()),
                ("grid_density_tracks.json", GRID),
                ("census_summary_detections.json", {"rows": []}),
                ("distance_sampling_tracks_roe-deer.json",
                 dict(DISTANCE, species="roe deer")),
                ("distance_sampling_tracks_red-deer.json", DISTANCE),
                ("population_estimate.json", POPULATION),
                ("habitat.json", HABITAT)):
            (out / name).write_text(json.dumps(document))
        (out / "trend_tracks.json").write_text("{ not json")
        (out / "habitat_tracks.json").write_text("[]")
        return str(out)

    def test_what_is_there_is_found(self, folder):
        results = sr.collect(folder, "tracks")

        assert list(results["found"]) == [
            "census", "grid", "distance", "population", "habitat"]
        entry = results["found"]["census"][0]
        assert (entry["species"], entry["file"]) == (
            "", "census_summary_tracks.json")
        assert entry["document"]["source"] == "tracks"
        assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d$", entry["written"])

    def test_what_was_not_run_is_named(self, folder):
        assert sr.collect(folder, "tracks")["missing"] == [
            "Groups", "Compare visits", "Reaction to the drone",
            "Position uncertainty"]

    def test_results_per_species_are_listed_one_by_one(self, folder):
        entries = sr.collect(folder, "tracks")["found"]["distance"]

        # the name from the file where it has one, else from its file name
        assert [entry["species"] for entry in entries] == [
            "red-deer", "roe deer"]
        assert sr.first({"found": {"distance": entries}}, "distance") is None

    def test_the_results_of_what_the_survey_counts(self, folder):
        results = sr.collect(folder, "detections")

        assert results["found"]["census"][0]["file"] == (
            "census_summary_detections.json")
        assert "grid" not in results["found"]
        # the habitat file is the same whatever is counted
        assert "habitat" in results["found"]

    def test_a_file_that_cannot_be_read_counts_as_not_run(self, folder):
        assert "trend" not in sr.collect(folder, "tracks")["found"]

    def test_so_does_one_that_is_no_result(self, folder):
        for name in ("habitat.json", "population_estimate.json",
                     "distance_sampling_tracks_red-deer.json"):
            with open(os.path.join(folder, name), "w") as handle:
                handle.write("[1, 2, 3]")

        results = sr.collect(folder, "tracks")

        assert "habitat" not in results["found"]
        assert "population" not in results["found"]
        assert [entry["species"] for entry
                in results["found"]["distance"]] == ["roe deer"]
        assert "Habitat context" in results["missing"]

    def test_a_folder_that_is_not_there(self, tmp_path):
        results = sr.collect(str(tmp_path / "nowhere"))

        assert results["found"] == {}
        assert len(results["missing"]) == len(sr.RESULTS)

    def test_the_one_of_all_animals(self):
        results = results_of(census=census_of(), distance=[
            ("roe deer", DISTANCE), ("", dict(DISTANCE, n=99))])

        assert sr.first(results, "distance")["n"] == 99
        assert sr.first(results, "grid") is None
        assert sr.first({}, "census") is None


# ---------------------------------------------------------------------------
# Whether the results are of the survey the report is about
# ---------------------------------------------------------------------------

class TestOfOnePiece:
    @pytest.mark.parametrize("document, flights", [
        ({"flights": ["May", "June"]}, ["May", "June"]),
        # the habitat summary: an entry per flight
        ({"flights": [{"flight": "May", "n_tracks": 3},
                      {"flight": "June", "n_tracks": 0}]}, ["May", "June"]),
        # the comparison of visits: the flights of each visit
        ({"visits": [{"flights": ["May"]}, {"flights": ["June", "July"]}]},
         ["May", "June", "July"]),
        ({"flights": []}, None), ({"visits": []}, None), ({}, None),
        ({"flights": "May"}, None)])
    def test_the_flights_a_result_says_it_was_made_from(self, document,
                                                        flights):
        assert sr.flights_of(document) == flights

    def test_the_trend_of_the_fixture_says_its_flights(self):
        assert sr.flights_of(trend_of()) == list(NAMES)

    def test_a_result_of_other_flights(self):
        assert sr.not_of_survey({"flights": ["May"]}, ["May", "June"]) == (
            "made from other flights: May")
        assert sr.not_of_survey({"flights": ["June", "May"]},
                                ["May", "June"]) == ""

    def test_a_result_for_other_species(self):
        document = {"flights": ["May"], "species_filter": ["Roe Deer"]}

        assert sr.not_of_survey(document, ["May"]) == (
            "made for other species: Roe Deer")
        assert sr.not_of_survey(document, ["May"], ["lynx"]) == (
            "made for other species: Roe Deer")
        assert sr.not_of_survey(document, ["May"], ["roe deer"]) == ""
        assert sr.not_of_survey({"species_filter": None}, ["May"],
                                ["lynx"]) == "made for other species: all"
        assert sr.not_of_survey({"species_filter": []}, ["May"]) == ""

    def test_a_result_that_does_not_say_is_not_judged(self):
        assert sr.not_of_survey({"density_per_km2": 3.0}, ["May"],
                                ["lynx"]) == ""

    def test_a_result_for_one_species_is_not_held_to_the_filter(self):
        document = {"flights": ["May"], "species_filter": ["roe deer"]}

        assert sr.not_of_survey(document, ["May"], ["roe deer", "lynx"],
                                per_species=True) == ""
        # its flights are held to the survey's all the same
        assert sr.not_of_survey(document, ["June"], per_species=True) == (
            "made from other flights: May")

    def test_a_report_without_flights_judges_no_flights(self):
        assert sr.not_of_survey({"flights": ["May"]}, []) == ""

    def test_the_results_of_a_report_that_are_out_of_step(self):
        report = report_of(results_of(
            census=census_of(flights=2), trend=trend_of(), grid=GRID,
            population=[("roe deer", dict(POPULATION, flights=["May"])),
                        ("", dict(POPULATION, flights=["May", "June"],
                                  species_filter=["lynx"]))]))

        assert sr.out_of_step(report) == [
            ("Population estimation (transects) (roe deer)",
             "made from other flights: May"),
            ("Population estimation (transects)",
             "made for other species: lynx"),
            ("Compare visits",
             "made from other flights: May, June, Next May")]

    def test_a_survey_of_one_piece_has_nothing_to_say(self):
        assert sr.out_of_step(everything()) == []
        assert sr.cautions(everything()) == []
        assert "class='warn'" not in sr.html(everything())

    def test_one_result_out_of_step_is_said_at_the_top(self):
        report = report_of(results_of(census=census_of(flights=2),
                                      trend=trend_of()))

        assert sr.cautions(report) == [
            "Not of one piece: Compare visits was made when the survey had "
            "other flights or species than this report lists (see 'What the "
            "report was made from' under Settings). Run it again and write "
            "the report anew, for figures that belong together."]
        page = sr.html(report)
        assert page.index("class='cards'") < page.index(
            "<p class='warn'>Not of one piece: Compare visits was made") \
            < page.index("<h2>The survey</h2>")

    def test_several_are_named_together(self):
        report = report_of(results_of(
            census=census_of(flights=3), trend=trend_of()))

        assert sr.cautions(report)[0].startswith(
            "Not of one piece: Census summary, Compare visits were made "
            "when")
        assert "Run them again" in sr.cautions(report)[0]

    def test_the_table_of_results_says_which_and_why(self):
        html = sr.section_settings(report_of(results_of(
            census=census_of(flights=2), trend=trend_of())))

        assert "<th>Written</th><th>Not of this survey</th>" in html
        assert "<td>census.json</td><td>2026-11-16 09:30</td><td></td>" \
            in html
        assert "<td>trend.json</td><td>2026-11-16 09:30</td>" \
            "<td>made from other flights: May, June, Next May</td>" in html

    def test_the_table_has_no_such_column_when_all_is_well(self):
        html = sr.section_settings(report_of(results_of(
            census=census_of(flights=2))))

        assert "Not of this survey" not in html
        assert "<td>census.json</td><td>2026-11-16 09:30</td></tr>" in html

    def test_a_result_that_cannot_be_judged_does_not_cost_the_page(self):
        report = everything()
        sr.first(report["results"], "census")["flights"] = [["May"], 7]
        problems = []

        page = sr.html(report, problems)

        assert "<h2>The survey</h2>" in page
        assert not any(part == "The survey" for part, _why in problems)


# ---------------------------------------------------------------------------
# What a page is made of
# ---------------------------------------------------------------------------

class TestPieces:
    def test_a_table_escapes_what_it_is_given(self):
        html = sr._table(["Flight", "n"], [["<b>May</b> & June", 3]],
                         numeric=(1,))

        assert "<td>&lt;b&gt;May&lt;/b&gt; &amp; June</td>" in html
        assert "<th class='n'>n</th>" in html
        assert "<td class='n'>3</td>" in html

    def test_a_column_that_is_html_already_is_left_alone(self):
        html = sr._table(["a", "b"], [["x", "1&ndash;2"]], raw=(1,))

        assert "<td>1&ndash;2</td>" in html

    def test_a_table_of_facts_has_no_heading(self):
        html = sr._table(["", ""], [("Camera", "Thermal")])

        assert "<thead>" not in html and "<td>Camera</td>" in html

    def test_the_first_row_can_be_the_total(self):
        html = sr._table(["a"], [["all"], ["one"]], bold_first=True)

        assert html.count("class='total'") == 1
        assert html.index("class='total'") < html.index("all")

    @pytest.mark.parametrize("value, digits, text", [
        (None, 1, "&ndash;"), (3.14159, 2, "3.14"), (1234.5, 0, "1,234"),
        (float("nan"), 1, "&ndash;"), ("n/a", 1, "n/a"), (7, 1, "7.0")])
    def test_a_number(self, value, digits, text):
        assert sr._num(value, digits) == text

    def test_a_number_with_its_interval(self):
        assert sr._interval(92.4, 61.0, 140.1) == "92.4 (61.0&ndash;140.1)"
        assert sr._interval(92.4, None, None) == "92.4"
        assert sr._interval(None, 1, 2) == "&ndash;"

    def test_a_species_keeps_the_colour_the_project_gave_it(self):
        colours = sr.species_colours(
            ["roe deer", "lynx", "red deer", "wolf"],
            {"Roe Deer": "#123456", "red deer": "#abcdef"})

        assert colours == {"roe deer": "#123456", "lynx": sr.PALETTE[0],
                           "red deer": "#abcdef", "wolf": sr.PALETTE[1]}

    def test_details_and_notes_are_escaped_too(self):
        assert sr._details("Grid <1>", "a < b") == (
            "<details><summary>Grid &lt;1&gt;</summary>"
            "<p class='note'>a &lt; b</p></details>")
        assert sr._details("Grid", "") == "" and sr._note("") == ""


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

class TestCharts:
    def test_bars_one_per_group_and_series(self):
        root = svg_root(sr.svg_bars(
            ["roe deer", "red deer"],
            [("female", [4, 1], "#0072b2"), ("male", [2, 0], "#e69f00")],
            unit="animals"))

        bars = [r for r in tags(root, "rect") if float(r.get("height")) > 10]
        assert len(bars) == 3                 # the nought is no bar to see
        heights = sorted(float(r.get("height")) for r in bars)
        assert heights[2] == pytest.approx(4 * heights[0])
        texts = [t.text for t in tags(root, "text")]
        assert {"roe deer", "red deer", "female", "male",
                "animals"} <= set(texts)
        assert root.get("width") == "640"

    def test_the_axis_has_round_marks(self):
        root = svg_root(sr.svg_bars(["a"], [("n", [37], "#000")]))

        marks = [t.text for t in tags(root, "text")
                 if "end" in (t.get("class") or "")]
        assert marks[:5] == ["0", "10", "20", "30", "40"]

    def test_shares_are_marked_in_percent(self):
        root = svg_root(sr.svg_bars(["a"], [("n", [0.46], "#000")],
                                    percent=True))

        assert "50%" in [t.text for t in tags(root, "text")]

    def test_nothing_to_draw_is_no_chart(self):
        assert sr.svg_bars([], []) == ""
        assert sr.svg_histogram({"bin_edges": [1]}) == ""
        assert sr.svg_trend({"values": [None], "days": [0]}) == ""
        assert sr.svg_map({}) == ""

    def test_a_histogram_of_ground_and_animals(self):
        root = svg_root(sr.svg_histogram(HABITAT["layers"]["tree_cover"]))

        fills = [r.get("fill") for r in tags(root, "rect")]
        assert fills.count(sr.GROUND) == 4 and fills.count(sr.ANIMALS) == 4
        assert [t.text for t in tags(root, "text")][:5] == [
            "0", "25", "50", "75", "100"]

    def test_a_trend_with_its_intervals_and_its_line(self):
        root = svg_root(sr.svg_trend({
            "days": [0, 31, 366], "labels": ["2024-05-03", "2024-06-03",
                                             "2025-05-04"],
            "values": [20.0, 30.0, 50.0], "lows": [8.0, 14.0, 28.0],
            "highs": [41.0, 55.0, 82.0], "fitted": [22.0, 24.0, 50.0],
            "unit": "animals / km²"}))

        assert len(tags(root, "circle")) == 3
        assert len([el for el in tags(root, "line")
                    if el.get("class") == "bar"]) == 3
        assert len(tags(root, "polyline")) == 1
        texts = [t.text for t in tags(root, "text")]
        assert "animals / km²" in texts and "2025-05-04" in texts
        # the dates that would run into each other are left out
        assert "2024-06-03" not in texts

    def test_a_trend_without_a_fitted_rate_has_no_line(self):
        root = svg_root(sr.svg_trend({
            "days": [0, 10], "labels": ["a", "b"], "values": [3, 0],
            "lows": [1, 0], "highs": [9, 3.7], "fitted": None, "unit": ""}))

        assert tags(root, "polyline") == []

    def test_the_map_draws_what_it_is_given(self):
        report = report_of()
        root = svg_root(sr.svg_map(report["map"]))

        classes = [el.get("class") for el in root.iter()]
        assert classes.count("plot") == 1 and classes.count("seen") == 1
        assert classes.count("excluded") == 1 and classes.count("route") == 1
        animals = tags(root, "circle")
        assert [c.get("fill") for c in animals] == ["#0072b2", "#e69f00"]
        assert "N" in [t.text for t in tags(root, "text")]
        # 1000 m wide: a scale bar of 250 m
        assert "250 m" in [t.text for t in tags(root, "text")]

    def test_north_is_up_and_east_is_right(self):
        root = svg_root(sr.svg_map({"animals": [
            {"x": 0.0, "y": 0.0, "species": "a"},
            {"x": 100.0, "y": 100.0, "species": "a"}]}))

        south_west, north_east = tags(root, "circle")
        assert float(north_east.get("cx")) > float(south_west.get("cx"))
        assert float(north_east.get("cy")) < float(south_west.get("cy"))

    def test_a_long_survey_is_marked_in_kilometres(self):
        root = svg_root(sr.svg_map({"routes": [{
            "flight": "x", "points": [[0, 0], [12000, 300]]}]}))

        assert "5 km" in [t.text for t in tags(root, "text")]
        bar = [el for el in tags(root, "line")
               if el.get("class") == "scale"][0]
        length = float(bar.get("x2")) - float(bar.get("x1"))
        # 12 km and a margin of 4 % on either side are 760 px wide
        assert length == pytest.approx(760 * 5000 / (12000 * 1.08), abs=0.2)

    def test_a_name_cannot_break_a_drawing(self):
        text = sr.svg_map({"animals": [
            {"x": 0.0, "y": 0.0, "species": "<script>x</script>"}],
            "colours": {"<script>x</script>": "red' onload='x"}})

        assert "<script>" not in text and "onload='x" not in text
        svg_root(text)


# ---------------------------------------------------------------------------
# The sections
# ---------------------------------------------------------------------------

class TestSurveySection:
    def test_what_was_flown(self):
        html = sr.section_survey(report_of())

        assert "<td>What is counted</td><td>tracked animals, each once</td>" \
            in html
        assert "<td>Survey area</td><td>30.00 ha</td>" in html
        assert "<td>May</td><td>2024-05-03</td><td>05:10-05:30</td>" in html
        # only the weather somebody knows
        assert "<th class='n'>Air °C</th>" in html
        assert "Cloud %" not in html
        assert "<td class='n'>8.0</td>" in html

    def test_without_weather_it_says_how_to_have_it(self):
        report = report_of()
        for flight in report["flights"]:
            flight["weather"] = {}

        html = sr.section_survey(report)

        assert "Air °C" not in html
        assert "run &#x27;Fetch Weather&#x27; of the Environmental Context" \
            in html

    def test_boxes_species_and_no_area(self):
        html = sr.section_survey(report_of(
            source="detections", species_filter=["roe deer", "lynx"],
            survey_area=None))

        assert "an animal counts once per frame it is on" in html
        assert "<td>Species</td><td>roe deer, lynx</td>" in html
        assert "none drawn: the whole flights count" in html

    def test_the_facts_are_escaped_like_everything_else(self):
        """They went into the page as they came - and a species called
        'deer <1 year' took the rest of the table with it."""
        html = sr.section_survey(report_of(
            camera="Thermal <b>", crs="EPSG:<1>",
            species_filter=["deer <1 year", "a & b"]))

        assert "<td>Camera</td><td>Thermal &lt;b&gt;</td>" in html
        assert "<td>Species</td><td>deer &lt;1 year, a &amp; b</td>" in html
        assert "<td>Coordinate system</td><td>EPSG:&lt;1&gt;</td>" in html

    def test_an_area_of_exclusion_zones_only(self):
        html = sr.section_survey(report_of(survey_area={
            "adjusted_ha": None, "exclusions": [{}]}))

        assert "<td>Survey area</td><td>exclusion zones only, 1 exclusion " \
            "zone(s) taken off</td>" in html

    def test_exclusion_zones_are_said(self):
        html = sr.section_survey(report_of(survey_area={
            "adjusted_ha": 27.5, "exclusions": [{}, {}]}))

        assert "27.50 ha, 2 exclusion zone(s) taken off" in html


class TestMapSection:
    def test_the_drawn_map_and_what_its_signs_mean(self):
        html = sr.section_map(report_of())

        assert html.startswith("<h2>Map</h2><figure><svg")
        for sign in ("roe deer", "red deer", "route flown",
                     "ground the cameras saw", "survey area",
                     "exclusion zone"):
            assert f"</i>{sign}</span>" in html
        assert "<img" not in html

    def test_the_picture_of_the_map_goes_in_as_it_is(self):
        html = sr.section_map(report_of(canvas_png="aGVsbG8="))

        assert "src='data:image/png;base64,aGVsbG8='" in html
        assert "as it was shown in QGIS" in html

    def test_only_the_signs_that_are_on_the_map_are_explained(self):
        report = report_of()
        report["map"].update(routes=[], exclusions=[], seen=[])

        html = sr.section_map(report)

        assert "</i>survey area</span>" in html
        for sign in ("route flown", "ground the cameras saw",
                     "exclusion zone"):
            assert sign not in html

    def test_a_route_that_could_not_be_placed_is_said(self):
        report = report_of()
        report["map"].update(routes=[], routes_missing=["June", "July"])

        html = sr.section_map(report)

        assert ("<p class='note'>Not on the map: the route of June, July. "
                "A flight&#x27;s camera positions are placed by its "
                "elevation model") in html
        assert "Not on the map" not in sr.section_map(report_of())

    def test_nothing_to_show(self):
        assert sr.section_map(report_of(map={})) == ""


class TestCounts:
    def test_one_flight(self):
        html = sr.section_counts(report_of(results_of(census=census_of()),
                                           flights=1))

        assert "<tr class='total'><td>All animals</td><td class='n'>8</td>" \
            in html
        assert "<td>roe deer</td><td class='n'>6</td>" in html
        # the two red deer were together
        assert re.search(r"<td>red deer</td><td class='n'>2</td>.*?"
                         r"<td class='n'>1</td><td class='n'>2.00</td>"
                         r"<td class='n'>2</td></tr>", html)
        assert "Per flight" not in html
        assert "different days" not in html

    def test_sex_and_age_only_where_somebody_knows(self):
        known = sr.section_counts(report_of(results_of(census=census_of())))
        plain = census.build([{"name": "May", "covered_m2": None,
                               "records": _records("roe deer", 3)}])
        unknown = sr.section_counts(report_of(results_of(census=plain)))

        assert "<h3>Sex</h3>" in known and "<h3>Age</h3>" in known
        assert "<th class='n'>female</th>" in known
        assert "<h3>Sex</h3>" not in unknown and "<h3>Age</h3>" not in unknown

    def test_what_is_not_known_is_drawn_grey(self):
        html = sr.section_counts(report_of(results_of(census=census_of()),
                                           flights=1))

        sex_chart = html[html.index("<h3>Sex</h3>"):html.index("<h3>Age")]
        fills = re.findall(r"fill='(#\w+)'/><text[^>]*>(\w+)</text>",
                           sex_chart)
        assert dict((name, colour) for colour, name in fills) == {
            "female": sr.PALETTE[0], "unknown": sr.GROUND}

    def test_several_flights_are_the_survey_and_each_flight(self):
        html = sr.section_counts(report_of(
            results_of(census=census_of(flights=2))))

        # 6 + 2 and 7 + 2
        assert "<td>All animals</td><td class='n'>17</td>" in html
        assert "<h3>Per flight</h3>" in html
        assert "<td>June</td><td class='n'>9</td>" in html

    def test_flights_on_several_days_are_said_to_be(self):
        html = sr.section_counts(report_of(
            results_of(census=census_of(flights=2))))

        assert ("The flights were flown on 2 different days, and these are "
                "the animals of all of them together") in html

    def test_ground_seen_twice_on_one_day_is_said_too(self):
        report = report_of(results_of(census=census_of(flights=2)))
        for flight in report["flights"]:
            flight["date"] = "2024-05-03"

        html = sr.section_counts(report)

        assert "part of it was seen more than once" in html
        assert "different days" not in html

    def test_without_an_area_everything_counted_counts(self):
        html = sr.section_counts(report_of(
            results_of(census=census_of(with_area=False))))

        assert "<th class='n'>Counted</th>" in html
        assert "Outside" not in html

    def test_no_census_no_counts(self):
        assert sr.section_counts(report_of()) == ""


class TestDensity:
    def test_every_density_with_what_it_rests_on(self):
        rows = sr.density_rows(results_of(
            census=census_of(), grid=GRID, distance=DISTANCE,
            population=POPULATION))

        assert [(row[0], row[1]) for row in rows] == [
            ("Census, over the ground seen", "All animals"),
            ("Census, over the ground seen", "roe deer"),
            ("Census, over the ground seen", "red deer"),
            ("Census, over the survey area", "All animals"),
            ("Census, over the survey area", "roe deer"),
            ("Census, over the survey area", "red deer"),
            ("Grid density", "All animals"), ("Grid density", "roe deer"),
            ("Distance sampling", "All animals"),
            ("Transects (ZINB)", "All animals")]
        census_row, grid_row = rows[0], rows[6]
        assert census_row[2] == "26.67" and census_row[3] == "&ndash;"
        assert "8 animals on 30.0 ha looked at; no interval" in census_row[4]
        assert grid_row[2] == "67.98 (50.09&ndash;87.71)"
        assert grid_row[3] == "20 (15&ndash;26)"
        assert rows[8][2] == "92.40 (61.00&ndash;140.10)"
        assert "half-normal + canopy cover" in rows[8][4]
        assert rows[9][2] == "71.20 (40.00&ndash;126.00)"

    def test_an_area_seen_more_than_once_is_not_one_census_of_it(self):
        """Three visits to 30 ha: the animals of all three over 30 ha is
        three times too many."""
        repeated = census_of(flights=3)

        rows = sr.density_rows(results_of(census=repeated))

        assert sr.seen_more_than_once(repeated)
        assert {row[0] for row in rows} == {"Census, over the ground seen"}

    def test_flights_side_by_side_are(self):
        side_by_side = census_of(flights=2, covered_m2=150000.0)

        assert not sr.seen_more_than_once(side_by_side)
        assert "Census, over the survey area" in {
            row[0] for row in sr.density_rows(
                results_of(census=side_by_side))}

    def test_without_knowing_the_ground_seen_the_area_has_to_do(self):
        unknown = census_of(covered_m2=None)

        rows = sr.density_rows(results_of(census=unknown))

        assert not sr.seen_more_than_once(unknown)
        assert {row[0] for row in rows} == {"Census, over the survey area"}

    def test_per_species_results_keep_their_species(self):
        rows = sr.density_rows(results_of(distance=[
            ("roe deer", DISTANCE), ("red deer", dict(
                DISTANCE, density_per_km2=None))]))

        assert [(row[0], row[1]) for row in rows] == [
            ("Distance sampling", "roe deer")]

    def test_what_a_file_says_cannot_break_the_table(self):
        """The columns of figures are HTML already - what comes out of a
        file is escaped before it goes into them."""
        rows = sr.density_rows(results_of(
            distance=dict(DISTANCE, n="<i>54</i>", best_model="a<b"),
            population=dict(POPULATION, n_transects="<12>", estimates={
                "<ZINB>": {"density_per_100ha": 7.0, "ci95": [1.0, 9.0]}})))

        assert rows[0][4].startswith("&lt;i&gt;54&lt;/i&gt; observations, "
                                     "a&lt;b, effective strip")
        assert rows[1][0] == "Transects (&lt;ZINB&gt;)"
        assert rows[1][4].startswith("&lt;12&gt; transects")

    def test_the_section(self):
        html = sr.section_density(report_of(results_of(distance=DISTANCE)))

        assert html.startswith("<h2>Density</h2><table>")
        assert "Each method answers a slightly different question" in html
        assert sr.section_density(report_of()) == ""


class TestWhere:
    def test_the_grid_and_the_habitat(self):
        html = sr.section_where(report_of(results_of(grid=GRID,
                                                     habitat=HABITAT)))

        assert "27 of 42 cells seen well enough to count" in html
        assert "<td>All animals</td><td class='n'>52</td>" in html
        assert "<td>random</td>" in html and "89 %" in html
        assert "What the 54 animals had around them (within 30 m)" in html
        assert "<b>Tree cover</b>" in html
        assert "ground: mean 37.50% &middot; animals: mean 55.80%" in html
        assert "<td>Cropland</td><td class='n'>14 %</td>" \
            "<td class='n'>–</td>" in html
        assert html.count("<svg") == 2        # land cover, and the layer

    def test_boxes_are_called_boxes(self):
        html = sr.section_where(report_of(results_of(
            habitat=dict(HABITAT, source="detections"))))

        assert "What the 54 boxes had around them" in html
        assert "Share of the boxes" in html

    def test_one_of_the_two_is_enough(self):
        assert "Habitat" not in sr.section_where(
            report_of(results_of(grid=GRID)))
        assert "Over the grid" not in sr.section_where(
            report_of(results_of(habitat=HABITAT)))
        assert sr.section_where(report_of()) == ""


class TestTrend:
    def test_the_visits_of_every_species_all_animals_first(self):
        html = sr.section_trend(report_of(results_of(trend=trend_of())))

        assert html.index("<h3>All animals</h3>") < html.index(
            "<h3>roe deer</h3>")
        assert "<td>1</td><td>2024-05-03</td><td>May</td>" \
            "<td class='n'>6</td>" in html
        assert "per year" in html
        assert html.count("<svg") == 2
        assert "2024-05-03 to 2024-06-03: +50 %" in html

    def test_without_the_ground_the_counts_are_listed(self):
        trend = trend_of()
        trend["basis"] = survey_trend.BASIS_COUNT

        html = sr.section_trend(report_of(results_of(trend=trend)))

        assert "Ground looked at" not in html
        assert "animals counted" in html

    def test_no_trend_no_section(self):
        assert sr.section_trend(report_of()) == ""


class TestReactionAndPosition:
    def test_how_the_animals_took_the_drone(self):
        html = sr.section_reaction(report_of(results_of(
            reaction=reaction_of())))

        assert "<p><b>All animals:</b> 8 of 14 animals moved while in " \
            "view" in html
        assert "<td>All animals</td><td class='n'>14</td>" \
            "<td class='n'>8 (57 %)</td><td class='n'>8</td>" \
            "<td class='n'>0</td>" in html
        assert sr.section_reaction(report_of()) == ""

    def test_how_good_the_places_are(self):
        html = sr.section_position(report_of(results_of(
            position=position_of())))

        assert "<p>6 detections: typically " in html
        assert "<td>Camera angle</td><td>1 °</td>" in html
        assert "<td>Timing</td><td>0.1 s</td>" in html
        assert sr.section_position(report_of()) == ""


class TestSettings:
    def test_what_the_survey_was_made_with(self):
        html = sr.section_settings(report_of(results_of(
            census=census_of(flights=2), grid=GRID, distance=DISTANCE,
            trend=trend_of(), reaction=reaction_of())))

        # the same for every flight: said once
        assert html.count("Group distance") == 1
        assert "<td>Group distance</td><td>red deer 50 m, roe deer 50 m</td>" \
            in html
        assert "a cell counts when 50 % of it was seen" in html
        assert "truncated at 60.0 m, model half-normal + canopy cover" in html
        assert "<td>Visits</td><td>the flights of one day</td>" in html
        assert "<td>Moved</td><td>went at least 3 m while in view</td>" in html

    def test_flights_with_their_own_group_distances_are_named(self):
        document = census_of(flights=2)
        document["group_distances_m"]["June"]["roe deer"] = 80.0

        html = sr.section_settings(report_of(results_of(census=document)))

        assert "Group distance (May)" in html
        assert "<td>Group distance (June)</td>" \
            "<td>red deer 50 m, roe deer 80 m</td>" in html

    def test_what_made_each_flights_data(self):
        html = sr.section_settings(report_of())

        assert "<h3>May</h3><table><tbody><tr><td>Tracker</td>" \
            "<td>builtin</td>" in html
        assert "<h3>June</h3>" not in html      # nothing known of it

    def test_what_the_report_was_made_from_and_what_not(self):
        html = sr.section_settings(report_of(results_of(
            census=census_of(), distance=[("roe deer", DISTANCE)])))

        assert "<td>Census summary</td><td>census.json</td>" \
            "<td>2026-11-16 09:30</td>" in html
        assert "<td>Distance sampling (roe deer)</td>" in html
        assert "Not run, and so not in this report: Groups, Grid density" \
            in html

    def test_the_notes_of_every_result_can_be_unfolded(self):
        html = sr.section_notes(report_of(results_of(
            census=census_of(), grid=GRID, habitat=HABITAT,
            population=POPULATION)))

        assert html.startswith("<h2>How the figures were made</h2>")
        assert "<summary>Census summary</summary>" \
            "<p class='note'>Minimum counts.</p>" in html
        assert "<summary>Habitat context</summary>" in html
        assert "Population" not in html         # it has no notes
        assert sr.section_notes(report_of()) == ""


class TestHeadline:
    def test_the_figures_a_reader_looks_for_first(self):
        cards = sr.headline(report_of(results_of(
            census=census_of(), distance=DISTANCE, grid=GRID)))

        assert cards == [
            ("8", "animals counted in the survey area, over 2 days of "
                  "flying"),
            ("2", "species"),
            ("92.4", "animals per km² by distance sampling (61.0–140.1)"),
            ("2", "flights"), ("2024-05-03 to 2024-06-03", "flown"),
            ("30.0 ha", "survey area")]

    def test_without_distance_sampling_the_grid_has_the_density(self):
        cards = dict((what, figure) for figure, what in sr.headline(
            report_of(results_of(grid=GRID))))

        assert cards["animals per km² over the grid (50.1–87.7)"] == "68.0"

    def test_one_flight_on_one_day(self):
        cards = sr.headline(report_of(results_of(census=census_of()),
                                      flights=1, survey_area=None))

        assert ("8", "animals counted in the survey area") in cards
        assert ("1", "flight") in cards and ("2024-05-03", "flown") in cards
        assert not any(what == "survey area" for _f, what in cards)

    def test_nothing_known(self):
        assert sr.headline({"results": results_of()}) == []


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def everything():
    """A survey of three flights with every result there is."""
    return report_of(results_of(
        census=census_of(flights=3), grid=GRID, distance=DISTANCE,
        population=POPULATION, habitat=HABITAT, trend=trend_of(),
        reaction=reaction_of(), position=position_of()),
        flights=3, canvas_png="aGVsbG8=")


class TestPage:
    def test_it_has_all_its_sections_in_order(self):
        page = sr.html(everything())

        headings = re.findall(r"<h2>(.*?)</h2>", page)
        assert headings == [
            "The survey", "Map", "Counts", "Density",
            "Where the animals were", "Over time", "Reaction to the drone",
            "How good the places are", "Settings",
            "How the figures were made"]
        assert page.startswith("<!DOCTYPE html>\n<html lang='en'>")
        assert "<title>Survey report - Hintertal</title>" in page
        assert "<h1>Survey report - Hintertal</h1>" in page
        assert "Made with BAMBI - QGIS Integration 9.1.0 on 2026-11-16 " \
            "09:31" in page

    def test_it_needs_nothing_but_itself(self):
        page = sr.html(everything())

        assert "<script" not in page and "<link" not in page
        assert "@import" not in page and "url(" not in page
        # nothing is fetched: the only addresses are the name of SVG and
        # the picture that is in the page itself
        addresses = set(re.findall(r"""(?:src|href)=['"]([^'"]+)""", page))
        assert addresses == {"data:image/png;base64,aGVsbG8="}
        assert set(re.findall(r"https?://[^'\"\s<]+", page)) == {
            "http://www.w3.org/2000/svg"}

    def test_every_drawing_in_it_can_be_drawn(self):
        page = sr.html(everything())

        drawings = re.findall(r"<svg.*?</svg>", page, flags=re.S)
        assert len(drawings) >= 8
        for drawing in drawings:
            ET.fromstring(drawing)

    def test_a_survey_with_nothing_run_is_still_a_page(self):
        page = sr.html(report_of())

        assert re.findall(r"<h2>(.*?)</h2>", page) == [
            "The survey", "Map", "Settings"]
        assert "Not run, and so not in this report: Census summary" in page

    def test_names_cannot_break_the_page(self):
        report = everything()
        report["title"] = "<script>alert(1)</script>"
        report["flights"][0]["name"] = "<img src=x onerror=alert(1)>"
        report["camera"] = "Thermal\"><script>"

        page = sr.html(report)

        assert "<script" not in page and "<img src=x" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page

    def test_a_result_it_cannot_read_costs_one_part_not_the_page(self):
        """A census of another make - its rows without their counts - used
        to end the report in a KeyError."""
        report = everything()
        census_document = sr.first(report["results"], "census")
        for row in census_document["rows"]:
            del row["inside"]
        problems = []

        page = sr.html(report, problems)

        assert [part for part, _why in problems] == [
            "The figures at the top", "Counts", "Density"]
        assert problems[1][1] == "KeyError: 'inside'"
        headings = re.findall(r"<h2>(.*?)</h2>", page)
        assert headings[:5] == ["The survey", "Map", "Counts", "Density",
                                "Where the animals were"]
        assert page.count("This part could not be made: a result file it "
                          "is read from is not as this version") == 2
        assert "<h2>Over time</h2><h3>All animals</h3>" in page

    def test_nobody_has_to_ask_for_the_problems(self):
        report = everything()
        sr.first(report["results"], "position")["accuracy"] = {
            "gps_horizontal_m": "about two"}

        page = sr.html(report)

        assert "<h2>How good the places are</h2><p class='note'>This part " \
            "could not be made" in page

    def test_a_page_without_problems_reports_none(self, tmp_path):
        problems = []

        sr.write(everything(), str(tmp_path / "report.html"), problems)

        assert problems == []

    def test_an_empty_report_does_not_break_it(self):
        page = sr.html({})

        assert "<h1>Survey report</h1>" in page
        assert "<h2>Settings</h2>" in page

    def test_it_is_written_as_utf8(self, tmp_path):
        path = sr.write(everything(), str(tmp_path / sr.FILE_NAME))

        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "animals per km²" in text and "Air °C" in text
        assert os.path.basename(path) == "survey_report.html"
