# -*- coding: utf-8 -*-
"""The census counts the groups the animals formed (core.census, core.groups
and the Census Summary step)."""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import census, groups, store, survey_area
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight
from .test_groups import HERD


def _record(species, group, x=0.0, y=0.0, **more):
    return dict({"species": species, "group": group, "x": x, "y": y,
                 "sex": None, "age": None, "approved": False}, **more)


class _Area:
    """A survey area: everything with x up to *limit*."""

    adjusted_m2 = 1.0e6

    def __init__(self, limit):
        self.limit = limit

    def contains(self, x, y):
        return x <= self.limit


# ---------------------------------------------------------------------------
# core.census
# ---------------------------------------------------------------------------

class TestRows:
    def test_a_row_says_how_its_animals_were_grouped(self):
        records = [_record("red deer", 1), _record("red deer", 1),
                   _record("red deer", 1), _record("red deer", 4),
                   _record("roe deer", 6), _record("roe deer", 7)]
        rows = {(row["flight"], row["species"]): row for row in census.build(
            [{"name": "A", "records": records, "covered_m2": None}])["rows"]}

        red = rows[("A", "red deer")]
        assert (red["inside"], red["groups"], red["largest_group"]) == (4, 2, 3)
        assert red["mean_group_size"] == pytest.approx(2.0)
        roe = rows[("A", "roe deer")]
        assert (roe["groups"], roe["mean_group_size"],
                roe["largest_group"]) == (2, 1.0, 1)

    def test_all_species_is_all_their_groups(self):
        records = [_record("red deer", 1), _record("red deer", 1),
                   _record("roe deer", 6)]
        result = census.build(
            [{"name": "A", "records": records, "covered_m2": None}])
        total = next(row for row in result["rows"]
                     if row["species"] == census.ALL_SPECIES)

        assert (total["groups"], total["largest_group"]) == (2, 2)
        assert total["mean_group_size"] == pytest.approx(1.5)

    def test_groups_of_two_flights_add_up_although_their_ids_agree(self):
        """Group ids are a flight's own: group 1 here is not group 1 there."""
        flights = [
            {"name": "A", "covered_m2": None,
             "records": [_record("red deer", 1), _record("red deer", 1)]},
            {"name": "B", "covered_m2": None,
             "records": [_record("red deer", 1), _record("red deer", 1),
                         _record("red deer", 1), _record("red deer", 9)]}]
        result = census.build(flights)
        total = next(row for row in result["rows"]
                     if row["flight"] == census.ALL_FLIGHTS
                     and row["species"] == "red deer")

        assert (total["inside"], total["groups"],
                total["largest_group"]) == (6, 3, 3)
        assert total["mean_group_size"] == pytest.approx(2.0)

    def test_a_group_counts_with_its_members_inside_the_area(self):
        """Nine deer, six of them in the plot: a group of six."""
        records = [_record("red deer", 1, x=float(x)) for x in range(9)]
        rows = census.build(
            [{"name": "A", "records": records, "covered_m2": None}],
            area=_Area(5.0))["rows"]

        assert (rows[0]["inside"], rows[0]["groups"],
                rows[0]["largest_group"]) == (6, 1, 6)

    def test_a_group_wholly_outside_is_no_group_here(self):
        records = [_record("red deer", 1, x=1.0), _record("red deer", 2, x=50.0),
                   _record("red deer", 2, x=51.0)]
        rows = census.build(
            [{"name": "A", "records": records, "covered_m2": None}],
            area=_Area(5.0))["rows"]

        assert (rows[0]["inside"], rows[0]["groups"]) == (1, 1)

    def test_no_animals_inside_no_groups(self):
        rows = census.build(
            [{"name": "A", "covered_m2": None,
              "records": [_record("red deer", 1, x=50.0)]}],
            area=_Area(5.0))["rows"]

        assert rows[0]["groups"] == 0
        assert rows[0]["mean_group_size"] is None
        assert rows[0]["largest_group"] is None

    def test_records_without_a_group_leave_the_columns_empty(self, tmp_path):
        records = [{"species": "red deer", "x": 0.0, "y": 0.0}]
        result = census.build(
            [{"name": "A", "records": records, "covered_m2": None}])

        assert result["rows"][0]["groups"] is None
        assert result["rows"][0]["mean_group_size"] is None
        path = census.write_csv(result, str(tmp_path / "census.csv"))
        with open(path, newline="", encoding="utf-8") as handle:
            written = list(csv.DictReader(handle))
        assert written[0]["groups"] == "" and written[0]["inside"] == "1"

    def test_the_columns_are_in_the_csv(self, tmp_path):
        records = [_record("red deer", 1), _record("red deer", 1),
                   _record("red deer", 3)]
        result = census.build(
            [{"name": "A", "records": records, "covered_m2": None}])
        path = census.write_csv(result, str(tmp_path / "census.csv"))
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            written = list(reader)

        assert reader.fieldnames[:14] == census.COLUMNS
        assert census.COLUMNS[-3:] == [
            "groups", "mean_group_size", "largest_group"]
        assert (written[0]["groups"], written[0]["mean_group_size"],
                written[0]["largest_group"]) == ("2", "1.5", "2")

    def test_who_counts_inside(self):
        assert census.counts_inside({"x": None, "y": None})
        assert not census.counts_inside({"x": None, "y": None}, _Area(5.0))
        assert census.counts_inside({"x": 5.0, "y": 0.0}, _Area(5.0))
        assert not census.counts_inside({"x": 5.1, "y": 0.0}, _Area(5.0))


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def _census(tmp_path, folder, source="tracks", species=None,
            area_enabled=False):
    survey_folder = str(tmp_path / "survey")
    survey = Survey(folder=survey_folder, modality="t", source=source,
                    species=species, area_enabled=area_enabled, flights=[
                        SurveyFlight(name="Meadow", target_folder=folder,
                                     epsg=EPSG)])
    log = []
    BambiProcessor().run_census_summary(
        {"survey": survey.to_dict()}, log_fn=log.append)
    out = os.path.join(survey_folder, "analytics_t")
    with open(os.path.join(out, f"census_summary_{source}.json")) as handle:
        result = json.load(handle)
    with open(os.path.join(out, f"groups_{source}.json")) as handle:
        table = json.load(handle)
    return result, table, log, out


@pytest.fixture
def herd(tmp_path):
    return make_flight(str(tmp_path / "flight"), HERD)


def _row(result, species, flight="Meadow"):
    return next(row for row in result["rows"]
                if row["species"] == species and row["flight"] == flight)


def test_the_census_reports_the_groups(tmp_path, herd):
    result, _table, log, _out = _census(tmp_path, herd)

    red = _row(result, "red deer")
    assert (red["inside"], red["groups"], red["largest_group"]) == (5, 2, 3)
    assert red["mean_group_size"] == pytest.approx(2.5)
    assert _row(result, "roe deer")["groups"] == 1
    assert _row(result, "lynx")["groups"] == 1          # never placed: alone
    everything = _row(result, census.ALL_SPECIES)
    assert (everything["inside"], everything["groups"]) == (8, 4)
    assert "seen within 60 s of each other" in result["notes"]
    assert result["group_distances_m"] == {
        "Meadow": {"lynx": 50.0, "red deer": 50.0, "roe deer": 50.0}}
    assert any(line.startswith("Groups: 4 - 7 animal(s) in 3 group(s) of two "
                               "or more (largest 3), 1 alone")
               for line in log)


def test_the_groups_are_written_one_row_each(tmp_path, herd):
    _result, table, _log, out = _census(tmp_path, herd)

    assert [(row["flight"], row["group"], row["species"], row["size"],
             row["counted"], row["members"]) for row in table["groups"]] == [
        ("Meadow", 1, "red deer", 3, 3, [1, 2, 3]),
        ("Meadow", 4, "red deer", 2, 2, [4, 5]),
        ("Meadow", 6, "roe deer", 2, 2, [6, 7]),
        ("Meadow", 8, "lynx", 1, 1, [8])]
    first = table["groups"][0]
    assert first["x"] == pytest.approx(ORIGIN[0] + 30.0)
    assert first["y"] == pytest.approx(ORIGIN[1] + 100.0 / 3)
    assert first["first_time"] == "2024-05-01T05:00:30Z"
    assert first["last_time"] == "2024-05-01T05:00:55Z"
    assert len(first["positions"]) == 3
    assert table["groups"][3]["x"] is None and table["groups"][3]["positions"] == []
    assert table["summary"] == {
        "groups": 4, "animals": 8, "mean_size": 2.0,
        "se_size": pytest.approx(0.408248, abs=1e-5), "largest": 3,
        "alone": 1}
    assert table["source"] == "tracks" and table["crs"] == "EPSG:32633"
    with open(os.path.join(out, "groups_tracks.csv"), newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == groups.TABLE_COLUMNS
    assert rows[0]["members"] == "1 2 3" and rows[0]["size"] == "3"
    assert rows[3]["x"] == "" and rows[3]["members"] == "8"


def test_a_species_that_is_not_counted_has_no_groups_in_the_table(
        tmp_path, herd):
    result, table, _log, _out = _census(tmp_path, herd, species=["roe deer"])

    assert [row["species"] for row in table["groups"]] == ["roe deer"]
    assert _row(result, census.ALL_SPECIES)["groups"] == 1


def test_the_schemas_distance_reaches_the_census(tmp_path, herd):
    conn = store.open_store(store.project_path(herd), store.PROJECT)
    with store.transaction(conn):
        row = conn.execute("SELECT species_id FROM species "
                           "WHERE name = 'red deer'").fetchone()
        groups.store_distance(conn, row["species_id"], 0)
    conn.close()

    result, _table, _log, _out = _census(tmp_path, herd)

    red = _row(result, "red deer")
    assert (red["groups"], red["largest_group"]) == (5, 1)
    assert result["group_distances_m"]["Meadow"]["red deer"] == 0.0


def test_a_group_cut_by_the_survey_area_counts_with_what_is_inside(
        tmp_path, herd):
    """The plot ends between the herd's first member (x = 10) and the other
    two (x = 40)."""
    survey_folder = str(tmp_path / "survey")
    os.makedirs(survey_folder)
    x0, y0 = ORIGIN
    shapely = pytest.importorskip("shapely.geometry")
    survey_area.create(survey_folder, EPSG, plots=[
        (shapely.box(x0 + 25, y0 - 50, x0 + 400, y0 + 100), "plot")])

    result, table, _log, _out = _census(tmp_path, herd, area_enabled=True)

    red = _row(result, "red deer")
    assert (red["counted"], red["inside"], red["outside"]) == (5, 3, 2)
    # tracks 2 and 3 of the herd of three, track 5 of the second crossing
    assert (red["groups"], red["largest_group"]) == (2, 2)
    by_group = {row["group"]: row for row in table["groups"]}
    assert (by_group[1]["size"], by_group[1]["counted"]) == (3, 2)
    assert (by_group[4]["size"], by_group[4]["counted"]) == (2, 1)
    assert 8 not in by_group                      # the lynx has no place


def test_boxes_are_grouped_by_frame(tmp_path):
    folder = make_flight(str(tmp_path / "photos"), [], loose_boxes=[
        {"species": "red deer", "at": (0, 0), "frame": 3},
        {"species": "red deer", "at": (20, 0), "frame": 3},
        {"species": "red deer", "at": (0, 1), "frame": 4},
    ])

    result, table, _log, _out = _census(tmp_path, folder, "detections")

    red = _row(result, "red deer")
    assert (red["inside"], red["groups"], red["largest_group"]) == (3, 2, 2)
    assert "in the same frame" in result["notes"]
    assert sorted(row["size"] for row in table["groups"]) == [1, 2]
