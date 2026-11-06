# -*- coding: utf-8 -*-
"""Survey model (core.survey) and census summary (core.census + the step).

The census is the plain survey figure - animals counted, inside the plot,
per km² - so the tests pin the two area bases and how several flights add
up: adjacent flights are one census of the plot, repeat flights are summed
effort on the covered basis.
"""
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import census, survey_area, track_store
from bambi_wildlife_detection.core.survey import (
    Survey, SurveyError, SurveyFlight, route_suffix)

from .test_analytics_strata import _flight

EPSG = 32633


# ---------------------------------------------------------------------------
# core.survey
# ---------------------------------------------------------------------------

class TestSurvey:
    def test_round_trip_through_the_config(self, tmp_path):
        survey = Survey(folder=str(tmp_path), modality="w",
                        flights=[SurveyFlight("A", str(tmp_path), 32633,
                                              dem="d.json")],
                        species=["roe deer"], per_species=True)
        again = Survey.from_config({"survey": survey.to_dict()})
        assert again == survey

    def test_no_survey_in_the_config(self):
        with pytest.raises(SurveyError, match="Tick the flights"):
            Survey.from_config({})

    def test_a_survey_needs_a_folder(self, tmp_path):
        survey = Survey(folder="", flights=[SurveyFlight("A", str(tmp_path))])
        with pytest.raises(SurveyError, match="survey folder"):
            survey.validate()

    def test_the_same_flight_twice_is_refused(self, tmp_path):
        flight = SurveyFlight("A", str(tmp_path), EPSG)
        survey = Survey(folder=str(tmp_path), flights=[flight, flight])
        with pytest.raises(SurveyError, match="twice"):
            survey.validate()

    def test_results_go_per_camera(self, tmp_path):
        survey = Survey(folder=str(tmp_path), modality="w")
        assert survey.analytics_folder == os.path.join(str(tmp_path),
                                                       "analytics_w")

    def test_species_names_are_translated_per_flight(self, tmp_path):
        root = str(tmp_path / "f")
        os.makedirs(root)
        _flight(root)
        survey = Survey(folder=str(tmp_path),
                        flights=[SurveyFlight("f", root, EPSG)])
        assert survey.species_ids(survey.flights[0], ["roe deer"]) == [1]
        assert survey.species_ids(survey.flights[0], ["wolverine"]) == []
        assert survey.species_ids(survey.flights[0]) is None
        assert survey.species_names()[:2] == ["roe deer", "chamois"] or \
            {"roe deer", "chamois"} <= set(survey.species_names())

    def test_route_prefers_the_survey_camera(self, tmp_path):
        for suffix in ("t", "w"):
            folder = tmp_path / f"flight_route_{suffix}"
            folder.mkdir()
            (folder / "flight_route.geojson").write_text("{}")
        assert route_suffix(str(tmp_path), "w") == "w"
        os.remove(tmp_path / "flight_route_w" / "flight_route.geojson")
        assert route_suffix(str(tmp_path), "w") == "t"
        assert route_suffix(str(tmp_path), "t", "missing.json") is None


# ---------------------------------------------------------------------------
# core.census
# ---------------------------------------------------------------------------

def _records(*specs):
    return [{"species": sp, "sex": sex, "age": age, "x": x, "y": y,
             "approved": approved}
            for sp, sex, age, x, y, approved in specs]


@pytest.fixture
def plot(tmp_path):
    """A 1 km² plot (1000 x 1000 m) with a 10 ha exclusion zone."""
    from shapely.geometry import box

    survey_area.create(str(tmp_path), EPSG,
                       plots=[(box(0, 0, 1000, 1000), "Revier")],
                       exclusions=[(box(0, 0, 100, 1000), "Wood", "forest")])
    return survey_area.load(str(tmp_path), EPSG)


class TestCensus:
    def test_counts_inside_outside_and_unplaced(self, plot):
        records = _records(
            ("roe deer", "female", "adult", 500, 500, True),
            ("roe deer", "male", "", 600, 600, False),
            ("roe deer", "", "juvenile", 50, 500, False),     # excluded zone
            ("roe deer", "female", "", 5000, 5000, False),     # outside
            ("roe deer", "", "", None, None, False),           # unplaced
        )
        rows = census.build([{"name": "F1", "records": records,
                              "covered_m2": None}], plot)["rows"]
        roe = next(r for r in rows if r["species"] == "roe deer")
        assert roe["counted"] == 5
        assert roe["inside"] == 2
        assert roe["outside"] == 2
        assert roe["unplaced"] == 1
        assert roe["approved"] == 1
        assert roe["sex"] == {"female": 1, "male": 1}
        assert roe["age"] == {"adult": 1, "unknown": 1}

    def test_plot_density_uses_the_adjusted_area(self, plot):
        records = _records(*[("roe deer", "", "", 500, 500, False)] * 9)
        rows = census.build([{"name": "F1", "records": records,
                              "covered_m2": None}], plot)["rows"]
        roe = rows[0]
        assert roe["plot_ha"] == pytest.approx(90.0)
        assert roe["per_km2_plot"] == pytest.approx(9 / 0.9)
        assert roe["per_km2_covered"] is None

    def test_without_a_plot_everything_counts(self):
        records = _records(("fox", "", "", 1, 1, False),
                           ("fox", "", "", None, None, False))
        result = census.build([{"name": "F1", "records": records,
                                "covered_m2": 2e5}])
        fox = result["rows"][0]
        assert fox["inside"] == 2
        assert fox["unplaced"] == 1
        assert fox["per_km2_plot"] is None
        assert fox["per_km2_covered"] == pytest.approx(2 / 0.2)
        assert result["plot_ha"] is None

    def test_flights_are_listed_and_totalled(self, plot):
        a = _records(*[("roe deer", "", "", 500, 500, False)] * 3,
                     ("fox", "", "", 500, 500, False))
        b = _records(*[("roe deer", "", "", 700, 700, False)] * 5)
        result = census.build([
            {"name": "A", "records": a, "covered_m2": 3e5},
            {"name": "B", "records": b, "covered_m2": 5e5},
        ], plot, species_order=["roe deer", "fox"])
        by = {(r["flight"], r["species"]): r for r in result["rows"]}
        assert by[("A", census.ALL_SPECIES)]["inside"] == 4
        total_roe = by[(census.ALL_FLIGHTS, "roe deer")]
        assert total_roe["inside"] == 8
        # One census of the plot: summed counts over the plot area once.
        assert total_roe["per_km2_plot"] == pytest.approx(8 / 0.9)
        # Effort basis: summed counts over the summed covered ground.
        assert total_roe["per_km2_covered"] == pytest.approx(8 / 0.8)
        grand = by[(census.ALL_FLIGHTS, census.ALL_SPECIES)]
        assert grand["inside"] == 9

    def test_a_single_flight_has_no_total_rows(self, plot):
        result = census.build([{"name": "A", "records": _records(
            ("fox", "", "", 500, 500, False)), "covered_m2": None}], plot)
        assert {r["flight"] for r in result["rows"]} == {"A"}

    def test_csv_spreads_the_breakdown(self, plot, tmp_path):
        result = census.build([{"name": "A", "records": _records(
            ("fox", "male", "adult", 500, 500, False),
            ("fox", "", "", 500, 500, False)), "covered_m2": None}], plot)
        path = census.write_csv(result, str(tmp_path / "c.csv"))
        header = open(path, encoding="utf-8").readline().strip().split(",")
        assert header[:3] == ["flight", "species", "counted"]
        assert "sex_male" in header and "sex_unknown" in header
        assert header.index("sex_male") < header.index("sex_unknown")
        assert "age_adult" in header


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def _config(survey_folder, *folders, source="tracks", area=False):
    survey = Survey(folder=survey_folder, modality="t", area_enabled=area,
                    flights=[SurveyFlight(os.path.basename(f), f, EPSG)
                             for f in folders])
    return {"survey": survey.to_dict(), "census_source": source}


@pytest.fixture
def flights(tmp_path):
    folders = []
    for name in ("a", "b"):
        root = str(tmp_path / name)
        os.makedirs(root)
        _flight(root)
        folders.append(root)
    track_store.set_track_attribute(folders[0], "t", 1, "sex", "female")
    track_store.set_track_attribute(folders[0], "t", 1, "approved", True)
    return folders


def _read(survey_folder, source="tracks"):
    with open(os.path.join(survey_folder, "analytics_t",
                           f"census_summary_{source}.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


class TestCensusStep:
    def test_tracks_of_two_flights(self, flights, tmp_path):
        folder = str(tmp_path / "survey")
        BambiProcessor().run_census_summary(_config(folder, *flights))
        result = _read(folder)
        # A track's species is the inventory's: its own label, then the
        # classifier's vote - the fixture's tracks carry the generic label.
        by = {(r["flight"], r["species"]): r for r in result["rows"]}
        assert by[("a", "animal")]["counted"] == 2
        assert by[("a", "animal")]["sex"] == {"female": 1, "unknown": 1}
        assert by[("a", "animal")]["approved"] == 1
        assert by[("b", "animal")]["sex"] == {"unknown": 2}
        assert by[(census.ALL_FLIGHTS, census.ALL_SPECIES)]["counted"] == 4
        assert os.path.isfile(os.path.join(
            folder, "analytics_t", "census_summary_tracks.csv"))

    def test_detections_count_every_box(self, flights, tmp_path):
        folder = str(tmp_path / "survey")
        BambiProcessor().run_census_summary(
            _config(folder, flights[0], source="detections"))
        rows = _read(folder, "detections")["rows"]
        everything = next(r for r in rows
                          if r["species"] == census.ALL_SPECIES)
        assert everything["counted"] == 3
        assert {r["species"] for r in rows} == {
            "roe deer", "chamois", census.ALL_SPECIES}

    def test_the_survey_area_limits_the_count(self, flights, tmp_path):
        from shapely.geometry import box

        folder = str(tmp_path / "survey")
        # Track 1's centroid is at (500010, 5300010); track 2's further out.
        survey_area.create(folder, EPSG, plots=[
            (box(500000, 5300000, 500020, 5300020), "")])
        BambiProcessor().run_census_summary(
            _config(folder, flights[0], area=True))
        result = _read(folder)
        everything = next(r for r in result["rows"]
                          if r["species"] == census.ALL_SPECIES)
        assert everything["inside"] == 1
        assert everything["outside"] == 1
        assert result["plot_ha"] == pytest.approx(0.04)
        assert everything["per_km2_plot"] == pytest.approx(1 / 0.0004)

    def test_species_filter_by_name(self, flights, tmp_path):
        folder = str(tmp_path / "survey")
        config = _config(folder, *flights, source="detections")
        config["survey"]["species"] = ["chamois"]
        BambiProcessor().run_census_summary(config)
        species = {r["species"] for r in _read(folder, "detections")["rows"]}
        assert species == {"chamois", census.ALL_SPECIES}
