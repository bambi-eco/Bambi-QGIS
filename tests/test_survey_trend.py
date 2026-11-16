# -*- coding: utf-8 -*-
"""Trends across visits (core/survey_trend.py): which flights are one visit,
what a visit counted and how sure that is, what changed between two visits,
and the rate of change through all of them."""
import math
from datetime import datetime, timedelta, timezone

import pytest

from bambi_wildlife_detection.core import census, survey_trend as st

UTC = timezone.utc
CEST = timezone(timedelta(hours=2))


def at(day, hour=5, minute=0, tz=UTC, month=5, year=2024):
    return datetime(year, month, day, hour, minute, tzinfo=tz)


def flight(name, start, minutes=30, **more):
    return dict({"name": name, "start": start,
                 "end": start + timedelta(minutes=minutes)}, **more)


def names(visits):
    return [visit["flights"] for visit in visits]


# ---------------------------------------------------------------------------
# Which flights are one visit
# ---------------------------------------------------------------------------

class TestVisits:
    def test_the_flights_of_one_day_are_one_visit(self):
        visits = st.group_visits([
            flight("north", at(1, 5)), flight("south", at(1, 19)),
            flight("again", at(2, 5))])

        assert names(visits) == [["north", "south"], ["again"]]
        assert [visit["visit"] for visit in visits] == [1, 2]
        assert [visit["label"] for visit in visits] == [
            "2024-05-01", "2024-05-02"]

    def test_in_order_of_time_whatever_order_they_come_in(self):
        visits = st.group_visits([
            flight("late", at(20)), flight("early", at(1)),
            flight("middle", at(10))])

        assert names(visits) == [["early"], ["middle"], ["late"]]

    def test_a_gap_of_days_chains_the_flights(self):
        flights = [flight("a", at(1)), flight("b", at(2)), flight("c", at(3)),
                   flight("d", at(6))]

        assert names(st.group_visits(flights, 1)) == [["a", "b", "c"], ["d"]]
        assert names(st.group_visits(flights, 3)) == [["a", "b", "c", "d"]]
        assert names(st.group_visits(flights, 0)) == [
            ["a"], ["b"], ["c"], ["d"]]

    def test_a_visit_of_several_days_says_from_when_to_when(self):
        visit = st.group_visits(
            [flight("a", at(1, 5)), flight("b", at(2, 7), minutes=60)], 1)[0]

        assert visit["label"] == "2024-05-01 to 2024-05-02"
        assert (visit["start"], visit["end"]) == (at(1, 5), at(2, 8))
        assert visit["mid"] == at(1, 18, 30)

    def test_a_night_survey_does_not_end_at_midnight(self):
        visits = st.group_visits([
            flight("before", at(1, 23, 10)), flight("after", at(2, 0, 20)),
            flight("next morning", at(2, 8, 0))])

        assert names(visits) == [["before", "after", "next morning"]]
        # ... the morning flight is the same day as the one after midnight;
        # without that one it is another day, more than six hours on
        assert names(st.group_visits([
            flight("before", at(1, 23, 10)),
            flight("next morning", at(2, 8, 0))])) == [
            ["before"], ["next morning"]]

    def test_the_day_is_the_local_one(self):
        """00:30 and 20:00 on the first of May, summer time: two days by
        the clock of the flight log (UTC), one day where the drone flew."""
        visits = st.group_visits([
            flight("night", at(1, 0, 30, tz=CEST)),
            flight("evening", at(1, 20, 0, tz=CEST))])

        assert names(visits) == [["night", "evening"]]
        assert visits[0]["label"] == "2024-05-01"

    def test_a_stamp_without_an_offset_is_taken_as_utc(self):
        visits = st.group_visits([
            flight("a", datetime(2024, 5, 1, 5)), flight("b", at(1, 9))])

        assert names(visits) == [["a", "b"]]

    def test_a_flight_without_a_date_is_named(self):
        with pytest.raises(st.TrendError, match="Meadow.*no capture time"):
            st.group_visits([flight("a", at(1)),
                             {"name": "Meadow", "start": None, "end": None}])

    @pytest.mark.parametrize("gap", [-1, 4000, "soon"])
    def test_a_gap_that_is_none_is_refused(self, gap):
        with pytest.raises(st.TrendError):
            st.group_visits([flight("a", at(1))], gap)


# ---------------------------------------------------------------------------
# How sure a count is
# ---------------------------------------------------------------------------

class TestCountInterval:
    def test_animals_seen_one_by_one_get_the_exact_poisson_interval(self):
        from scipy import stats

        low, high = st.count_interval(5, 5, 1)

        assert low == pytest.approx(stats.chi2.ppf(0.025, 10) / 2)
        assert high == pytest.approx(stats.chi2.ppf(0.975, 12) / 2)
        assert (low, high) == pytest.approx((1.6235, 11.6683), abs=1e-4)

    def test_nothing_counted_still_leaves_room(self):
        assert st.count_interval(0, 0, 1) == pytest.approx(
            (0.0, -math.log(0.025)))

    def test_a_herd_is_one_piece_of_luck(self):
        singles = st.count_interval(10, 10, 1)
        herd = st.count_interval(10, 100, 10)

        assert herd[0] < singles[0] and herd[1] > singles[1]
        assert herd[0] < 1 and herd[1] > 50

    def test_the_level_can_be_set(self):
        wide = st.count_interval(8, 8, 1, level=0.99)
        narrow = st.count_interval(8, 8, 1, level=0.80)

        assert wide[0] < narrow[0] < 8 < narrow[1] < wide[1]


# ---------------------------------------------------------------------------
# One visit against another
# ---------------------------------------------------------------------------

class TestCompare:
    def test_the_same_count_on_the_same_ground_is_no_change(self):
        found = st.compare(12, 12, 2.0, 12, 12, 2.0)

        assert found["ratio"] == 1.0 and found["change_pct"] == 0.0
        assert found["p_value"] == pytest.approx(1.0)
        assert found["verdict"] == st.NO_CHANGE
        assert found["ratio_low"] < 1 < found["ratio_high"]

    def test_three_times_as_many_is_an_increase(self):
        found = st.compare(10, 10, 1.0, 30, 30, 1.0)

        assert found["ratio"] == pytest.approx(3.0)
        assert found["change_pct"] == pytest.approx(200.0)
        # 30 of 40 at even odds
        assert found["p_value"] == pytest.approx(0.002221, abs=1e-6)
        assert found["verdict"] == st.INCREASE
        assert 1 < found["ratio_low"] < 3 < found["ratio_high"]

    def test_and_a_third_a_decrease(self):
        found = st.compare(30, 30, 1.0, 10, 10, 1.0)

        assert found["verdict"] == st.DECREASE
        assert found["ratio"] == pytest.approx(1 / 3)
        assert found["ratio_high"] < 1

    def test_more_animals_on_more_ground_is_no_change(self):
        found = st.compare(10, 10, 1.0, 30, 30, 3.0)

        assert found["ratio"] == pytest.approx(1.0)
        assert found["verdict"] == st.NO_CHANGE
        assert found["p_value"] > 0.9

    def test_the_test_is_the_exact_binomial_one(self):
        stats = pytest.importorskip("scipy.stats")
        if not hasattr(stats, "binomtest"):
            pytest.skip("this scipy has no binomtest")

        found = st.compare(7, 7, 1.5, 15, 15, 2.5)

        assert found["p_value"] == pytest.approx(
            stats.binomtest(15, 22, 2.5 / 4.0).pvalue)

    def test_from_none_to_some(self):
        found = st.compare(0, 0, 1.0, 6, 6, 1.0)

        assert found["ratio"] is None and found["change_pct"] is None
        assert found["p_value"] == pytest.approx(2 * 0.5 ** 6)
        assert found["verdict"] == st.INCREASE
        assert found["ratio_low"] > 1 and found["ratio_high"] is None

    def test_from_some_to_none(self):
        found = st.compare(6, 6, 1.0, 0, 0, 1.0)

        assert found["ratio"] == 0.0 and found["change_pct"] == -100.0
        assert found["verdict"] == st.DECREASE
        assert found["ratio_low"] == 0.0 and found["ratio_high"] < 1

    def test_two_animals_more_is_not_a_change(self):
        found = st.compare(0, 0, 1.0, 2, 2, 1.0)

        assert found["p_value"] == pytest.approx(0.5)
        assert found["verdict"] == st.NO_CHANGE

    def test_nothing_on_either_visit_says_nothing(self):
        found = st.compare(0, 0, 1.0, 0, 0, 1.0)

        assert found["p_value"] is None and found["ratio"] is None
        assert found["verdict"] == st.NO_CHANGE

    def test_herds_are_worth_fewer_sightings_than_animals(self):
        """Thirty animals in three herds against ten on their own: three
        times the density - on the strength of four sightings."""
        singles = st.compare(10, 10, 1.0, 30, 30, 1.0)
        herds = st.compare(10, 10, 1.0, 30, 300, 1.0)

        assert herds["ratio"] == singles["ratio"]
        assert herds["dispersion"] == pytest.approx(310 / 40)
        assert herds["p_value"] == pytest.approx(0.625)     # 3 of 4
        assert herds["verdict"] == st.NO_CHANGE
        assert herds["ratio_high"] > singles["ratio_high"]


# ---------------------------------------------------------------------------
# The rate of change through the visits
# ---------------------------------------------------------------------------

class TestLogLinear:
    def test_a_constant_rate_is_found_again(self):
        days = [0, 365, 730, 1095]
        ground = [1.0, 2.0, 1.5, 1.0]
        counts = [km2 * 5 * math.exp(0.001 * day)
                  for km2, day in zip(ground, days)]

        trend = st.log_linear(days, counts, ground, counts)

        assert trend["reason"] is None
        assert trend["slope_per_day"] == pytest.approx(0.001)
        assert trend["change_per_year_pct"] == pytest.approx(
            (math.exp(0.36525) - 1) * 100)
        assert trend["change_over_span_pct"] == pytest.approx(
            (math.exp(1.095) - 1) * 100)
        assert trend["fitted"] == pytest.approx(
            [5 * math.exp(0.001 * day) for day in days])
        # the visits lie on the line: counting is all the doubt there is
        assert trend["dispersion_from"] == "counting"
        assert trend["dispersion"] == 1.0
        assert trend["verdict"] == st.INCREASE

    def test_two_visits_give_the_ratio_of_the_two(self):
        trend = st.log_linear([0, 30], [10, 20], [1.0, 1.0], [10, 20])

        assert trend["slope_per_day"] == pytest.approx(math.log(2) / 30)
        assert trend["slope_se"] == pytest.approx(
            math.sqrt(1 / 10 + 1 / 20) / 30)
        assert trend["change_over_span_pct"] == pytest.approx(100.0)
        assert trend["change_over_span_pct_low"] < 0     # ten and twenty
        assert trend["verdict"] == st.NO_CHANGE
        assert trend["fitted"] == pytest.approx([10.0, 20.0])

    def test_the_ground_looked_at_is_the_exposure(self):
        same = st.log_linear([0, 100], [10, 30], [1.0, 3.0], [10, 30])

        assert same["slope_per_day"] == pytest.approx(0.0, abs=1e-12)
        assert same["change_over_span_pct"] == pytest.approx(0.0, abs=1e-9)
        assert same["p_value"] == pytest.approx(1.0)

    def test_a_clear_decrease(self):
        trend = st.log_linear([0, 200, 400], [200, 100, 50], [1, 1, 1],
                              [200, 100, 50])

        assert trend["verdict"] == st.DECREASE
        assert trend["change_over_span_pct"] == pytest.approx(-75.0, abs=0.5)
        assert trend["change_over_span_pct_high"] < 0
        assert trend["change_per_year_pct"] == pytest.approx(
            (0.25 ** (365.25 / 400) - 1) * 100, abs=0.5)

    def test_below_a_year_no_rate_per_year_is_given(self):
        short = st.log_linear([0, 364], [10, 20], [1, 1], [10, 20])
        year = st.log_linear([0, 366], [10, 20], [1, 1], [10, 20])

        assert short["change_per_year_pct"] is None
        assert year["change_per_year_pct"] == pytest.approx(100.0, abs=0.5)
        assert year["change_per_year_pct_low"] < 100 < \
            year["change_per_year_pct_high"]

    def test_groups_widen_the_interval(self):
        singles = st.log_linear([0, 30], [40, 80], [1, 1], [40, 80])
        herds = st.log_linear([0, 30], [40, 80], [1, 1], [400, 800])

        assert singles["verdict"] == st.INCREASE
        assert herds["dispersion_from"] == "groups"
        assert herds["dispersion"] == pytest.approx(10.0)
        assert herds["slope_se"] == pytest.approx(
            singles["slope_se"] * math.sqrt(10))
        assert herds["slope_per_day"] == pytest.approx(
            singles["slope_per_day"])
        assert herds["verdict"] == st.NO_CHANGE

    def test_visits_that_scatter_about_the_line_widen_it_too(self):
        from scipy import stats

        days = [0, 100, 200, 300, 400]
        counts = [100, 30, 160, 40, 210]

        trend = st.log_linear(days, counts, [1] * 5, counts)

        assert trend["dispersion_from"] == "visits"
        assert trend["dispersion"] > 10
        # Student's, on the three degrees of freedom five visits leave
        half = stats.t.ppf(0.975, 3) * trend["slope_se"] * 400
        middle = trend["slope_per_day"] * 400
        assert trend["change_over_span_pct_high"] == pytest.approx(
            (math.exp(middle + half) - 1) * 100)
        assert trend["p_value"] == pytest.approx(2 * stats.t.sf(
            abs(trend["slope_per_day"]) / trend["slope_se"], 3))
        assert trend["verdict"] == st.NO_CHANGE

    def test_no_animals_at_all(self):
        trend = st.log_linear([0, 10, 20], [0, 0, 0], [1, 1, 1], [0, 0, 0])

        assert trend["reason"] == "No animals were counted on any visit."
        assert trend["slope_per_day"] is None and trend["fitted"] is None

    @pytest.mark.parametrize("counts, end, how", [
        ([0, 0, 5], "last", "starts"), ([5, 0, 0], "first", "ends")])
    def test_all_the_animals_at_one_end_have_no_rate(self, counts, end, how):
        trend = st.log_linear([0, 30, 60], counts, [1, 1, 1], counts)

        assert f"on the {end} visit" in trend["reason"]
        assert f"{how} with nothing" in trend["reason"]
        assert trend["verdict"] == st.NO_CHANGE

    def test_all_the_animals_in_the_middle_do(self):
        trend = st.log_linear([0, 30, 60], [0, 5, 0], [1, 1, 1], [0, 5, 0])

        assert trend["reason"] is None
        assert trend["slope_per_day"] == pytest.approx(0.0, abs=1e-9)
        assert trend["verdict"] == st.NO_CHANGE

    def test_one_visit_is_no_trend(self):
        assert st.log_linear([0], [5], [1], [5])["reason"] == (
            "A trend needs at least two visits.")


# ---------------------------------------------------------------------------
# The survey, visit by visit
# ---------------------------------------------------------------------------

def box(x0, y0, x1, y1):
    from shapely.geometry import box as _box
    return _box(x0, y0, x1, y1)


class Plot:
    """A survey area: a rectangle."""

    def __init__(self, x0, y0, x1, y1):
        self.bounds = (x0, y0, x1, y1)
        self.geometry = box(x0, y0, x1, y1)
        self.adjusted_m2 = self.geometry.area

    def contains(self, x, y):
        x0, y0, x1, y1 = self.bounds
        return x0 <= x <= x1 and y0 <= y <= y1


def animals(species, places, group=None, **more):
    """One record per place; *group* gives them all one group."""
    return [dict({"species": species, "x": x, "y": y, "sex": None,
                  "age": None, "approved": False, "group": group}, **more)
            for x, y in places]


#: A square kilometre, and its western and eastern halves.
KM = (0, 0, 1000, 1000)
WEST = (0, 0, 500, 1000)
EAST = (500, 0, 1000, 1000)


def spread(n, within=KM, row=0):
    x0, y0, x1, y1 = within
    return [(x0 + (i + 0.5) * (x1 - x0) / n, y0 + 100 + 50 * row)
            for i in range(n)]


@pytest.fixture
def survey():
    """Three visits to one square kilometre: in May two flights side by
    side, in June one over all of it, a year on one over all of it."""
    return [
        flight("May west", at(1, 5), seen=box(*WEST),
               records=animals("roe deer", spread(2, WEST))),
        flight("May east", at(1, 6), seen=box(*EAST), records=[
            *animals("roe deer", spread(2, EAST)),
            *animals("red deer", spread(3, EAST, 1), group=7)]),
        flight("June", at(1, 5, month=6), seen=box(*KM),
               records=animals("roe deer", spread(6))),
        flight("Next May", at(3, 5, year=2025), seen=box(*KM), records=[
            *animals("roe deer", spread(9)),
            *animals("red deer", spread(2, row=1))]),
    ]


class TestAnalyse:
    def test_the_visits(self, survey):
        document = st.analyse(survey)

        assert [(v["visit"], v["label"], v["flights"])
                for v in document["visits"]] == [
            (1, "2024-05-01", ["May west", "May east"]),
            (2, "2024-06-01", ["June"]),
            (3, "2025-05-03", ["Next May"])]
        assert [v["looked_at_ha"] for v in document["visits"]] == [
            100.0, 100.0, 100.0]
        assert document["visits"][0]["time_of_day"] == "05:00-06:30"
        assert document["visits"][0]["day"] == 0.0
        assert document["visits"][2]["day"] == pytest.approx(367.0, abs=0.05)
        assert (document["basis"], document["ground"]) == (
            st.BASIS_GROUND, st.SEEN)

    def test_what_each_visit_counted(self, survey):
        document = st.analyse(survey)

        assert document["species"] == ["roe deer", "red deer", st.ALL]
        roe = document["series"]["roe deer"]
        assert [entry["animals"] for entry in roe] == [4, 6, 9]
        assert [entry["density_per_km2"] for entry in roe] == [4.0, 6.0, 9.0]
        red = document["series"]["red deer"]
        assert [entry["animals"] for entry in red] == [3, 0, 2]
        assert [entry["groups"] for entry in red] == [1, 0, 2]
        assert [entry["largest_group"] for entry in red] == [3, None, 1]
        assert [entry["animals"] for entry in document["series"][st.ALL]] == [
            7, 6, 11]

    def test_the_interval_knows_the_herd(self, survey):
        document = st.analyse(survey)
        red, roe = (document["series"][name][0]
                    for name in ("red deer", "roe deer"))

        # three red deer in one group: far less sure than four roe deer
        assert (red["density_low"], red["density_high"]) == pytest.approx(
            st.count_interval(3, 9, 3))
        assert (roe["density_low"], roe["density_high"]) == pytest.approx(
            st.count_interval(4, 4, 1))
        assert red["density_high"] / 3 > roe["density_high"] / 4
        # none in June: nought, with room for some
        none = document["series"]["red deer"][1]
        assert (none["density_per_km2"], none["density_low"]) == (0.0, 0.0)
        assert none["density_high"] == pytest.approx(3.689, abs=1e-3)

    def test_two_flights_over_the_same_ground_are_the_mean_of_the_two(self):
        document = st.analyse([
            flight("first pass", at(1, 5), seen=box(*KM),
                   records=animals("roe deer", spread(4))),
            flight("second pass", at(1, 7), seen=box(*KM),
                   records=animals("roe deer", spread(6))),
            flight("later", at(20, 5), seen=box(*KM),
                   records=animals("roe deer", spread(5))),
        ])

        first, later = document["series"]["roe deer"]
        assert first["animals"] == 10
        assert document["visits"][0]["looked_at_ha"] == 200.0
        assert (first["density_per_km2"], later["density_per_km2"]) == (
            5.0, 5.0)
        assert document["changes"]["roe deer"][0]["ratio"] == 1.0

    def test_the_changes_from_visit_to_visit_and_first_to_last(self, survey):
        document = st.analyse(survey)

        changes = document["changes"]["roe deer"]
        assert [(c["from"], c["to"]) for c in changes] == [
            (1, 2), (2, 3), (1, 3)]
        assert [c["change_pct"] for c in changes] == pytest.approx(
            [50.0, 50.0, 125.0])
        assert changes[0]["days"] == pytest.approx(31.0, abs=0.05)
        assert all(c["verdict"] == st.NO_CHANGE for c in changes)
        red = document["changes"]["red deer"]
        assert (red[0]["ratio"], red[1]["ratio"]) == (0.0, None)

    def test_two_visits_have_one_change(self, survey):
        document = st.analyse(survey[:3])

        assert [(c["from"], c["to"])
                for c in document["changes"]["roe deer"]] == [(1, 2)]

    def test_the_trend(self, survey):
        document = st.analyse(survey)
        trend = document["trends"]["roe deer"]

        assert trend["visits"] == 3
        assert trend["span_days"] == pytest.approx(367.0, abs=0.05)
        assert trend["slope_per_day"] > 0
        assert trend["change_per_year_pct"] is not None
        assert len(trend["fitted"]) == 3
        assert trend["verdict"] == st.NO_CHANGE       # 4, 6 and 9 animals
        assert document["trends"]["red deer"]["reason"] is None

    def test_a_survey_area_counts_what_is_in_it_and_gives_a_number(
            self, survey):
        plot = Plot(0, 0, 2000, 1000)                 # two square kilometres
        survey[2]["records"] += animals("roe deer", [(5000, 5000)])

        document = st.analyse(survey, area=plot)

        june = document["series"]["roe deer"][1]
        assert (june["counted"], june["animals"], june["outside"]) == (
            7, 6, 1)
        assert document["plot_ha"] == 200.0
        # six on the square kilometre the cameras saw: twelve on two
        assert june["number"] == pytest.approx(12.0)
        assert june["number_low"] == pytest.approx(2 * june["density_low"])
        assert june["number_high"] == pytest.approx(2 * june["density_high"])

    def test_without_an_area_no_number(self, survey):
        entry = st.analyse(survey)["series"]["roe deer"][0]

        assert entry["number"] is None

    def test_only_the_ground_every_visit_saw(self, survey):
        """June was cut short: only the western half was flown."""
        survey[2]["seen"] = box(*WEST)
        survey[2]["records"] = animals("roe deer", spread(3, WEST))

        document = st.analyse(survey, ground=st.COMMON)

        assert document["common_ha"] == 50.0
        assert [v["looked_at_ha"] for v in document["visits"]] == [
            50.0, 50.0, 50.0]
        roe = document["series"]["roe deer"]
        # May: the two in the west; next May: the western half of nine
        assert [entry["animals"] for entry in roe] == [2, 3, 4]
        assert [entry["density_per_km2"] for entry in roe] == [4.0, 6.0, 8.0]
        # May's red deer stood in the east, which June never saw; of next
        # May's two, one stood in the west
        assert [entry["animals"]
                for entry in document["series"]["red deer"]] == [0, 0, 1]
        assert roe[0]["number"] is None

    def test_all_the_ground_each_visit_saw_takes_the_short_visit_as_it_is(
            self, survey):
        survey[2]["seen"] = box(*WEST)
        survey[2]["records"] = animals("roe deer", spread(3, WEST))

        document = st.analyse(survey)

        assert [v["looked_at_ha"] for v in document["visits"]] == [
            100.0, 50.0, 100.0]
        assert [entry["density_per_km2"]
                for entry in document["series"]["roe deer"]] == [
            4.0, 6.0, 9.0]

    def test_an_animal_without_a_place_is_not_on_common_ground(self, survey):
        survey[0]["records"] += animals("roe deer", [(None, None)])

        seen = st.analyse(survey)["series"]["roe deer"][0]
        common = st.analyse(survey, ground=st.COMMON)["series"]["roe deer"][0]

        assert (seen["animals"], seen["unplaced"]) == (5, 1)
        assert common["animals"] == 4

    def test_visits_with_no_ground_in_common(self, survey):
        survey[2]["seen"] = box(5000, 5000, 6000, 6000)

        with pytest.raises(st.TrendError, match="no ground in common"):
            st.analyse(survey, ground=st.COMMON)

    def test_common_ground_needs_to_know_what_was_seen(self, survey):
        survey[3]["seen"] = None

        with pytest.raises(st.TrendError, match="Next May saw is not known"):
            st.analyse(survey, ground=st.COMMON)

    def test_without_knowing_the_ground_the_counts_are_compared(self, survey):
        survey[3]["seen"] = None

        document = st.analyse(survey)

        assert document["basis"] == st.BASIS_COUNT
        assert [v["looked_at_ha"] for v in document["visits"]] == [
            None, None, None]
        roe = document["series"]["roe deer"]
        assert [entry["animals"] for entry in roe] == [4, 6, 9]
        assert all(entry["density_per_km2"] is None for entry in roe)
        assert roe[0]["animals_low"] == pytest.approx(
            st.count_interval(4, 4, 1)[0])
        assert document["changes"]["roe deer"][0]["change_pct"] == \
            pytest.approx(50.0)
        assert document["trends"]["roe deer"]["fitted"] is not None

    def test_a_visit_that_saw_none_of_the_ground(self, survey):
        from shapely.geometry import Polygon

        survey[2]["seen"] = Polygon()

        with pytest.raises(st.TrendError, match="2024-06-01.*saw none"):
            st.analyse(survey)

    def test_one_visit_is_nothing_to_compare(self, survey):
        with pytest.raises(st.TrendError) as told:
            st.analyse(survey[:2])

        assert "all flown on 2024-05-01: that is one visit" in str(told.value)
        assert "fewer days" not in str(told.value)
        with pytest.raises(st.TrendError, match="allow fewer days"):
            st.analyse(survey, gap_days=400)

    def test_no_flights(self):
        with pytest.raises(st.TrendError, match="no flights"):
            st.analyse([])

    def test_an_unknown_ground(self, survey):
        with pytest.raises(st.TrendError):
            st.analyse(survey, ground="everything")

    def test_the_species_come_in_the_order_of_the_survey(self, survey):
        document = st.analyse(survey,
                              species_order=["wild boar", "red deer"])

        # wild boar was never seen; roe deer follows what was asked for
        assert document["species"] == ["red deer", "roe deer", st.ALL]

    def test_sex_and_age_of_each_visit(self, survey):
        survey[0]["records"] = animals(
            "roe deer", spread(2, WEST), sex="Female", age="juvenile")
        survey[2]["records"] = animals("roe deer", spread(6), sex="male")

        document = st.analyse(survey)

        assert document["sex_labels"] == ["female", "male", census.UNKNOWN]
        assert document["age_labels"] == ["juvenile", census.UNKNOWN]
        may, june, _next = document["series"]["roe deer"]
        assert may["sex"] == {"female": 2, census.UNKNOWN: 2}
        assert june["sex"] == {"male": 6}
        assert may["age"] == {"juvenile": 2, census.UNKNOWN: 2}

    def test_the_weather_of_a_visit_is_that_of_its_flights(self, survey):
        survey[0]["weather"] = {"air_temperature_c": 8.0, "wind_speed_ms": 2}
        survey[1]["weather"] = {"air_temperature_c": 12.0,
                                "wind_speed_ms": None}
        survey[2]["weather"] = {"air_temperature_c": None}

        visits = st.analyse(survey)["visits"]

        assert visits[0]["weather"] == {"air_temperature_c": 10.0,
                                        "wind_speed_ms": 2.0}
        assert visits[1]["weather"] == {"air_temperature_c": None}
        assert visits[2]["weather"] == {}


# ---------------------------------------------------------------------------
# Written down
# ---------------------------------------------------------------------------

class TestWritten:
    def test_one_row_per_visit_and_species(self, survey):
        survey[0]["weather"] = {"air_temperature_c": 8.0}
        document = st.analyse(survey)

        rows = st.flat_rows(document)

        assert len(rows) == 9
        assert list(rows[0])[:len(st.VISIT_COLUMNS)] == st.VISIT_COLUMNS
        first, second = rows[0], rows[1]
        assert (first["visit"], first["date"], first["species"]) == (
            1, "2024-05-01", "roe deer")
        assert first["flights"] == "May west; May east"
        assert first["start"] == "2024-05-01T05:00:00+00:00"
        assert (first["animals"], first["looked_at_ha"]) == (4, 100.0)
        assert first["change_pct"] is None            # nothing came before
        assert second["change_pct"] == pytest.approx(50.0)
        assert second["change_p_value"] == pytest.approx(
            document["changes"]["roe deer"][0]["p_value"])
        assert first["time_of_day"] == "05:00-06:30"
        assert first["air_temperature_c"] == 8.0
        assert second["air_temperature_c"] is None
        assert first[f"sex_{census.UNKNOWN}"] == 4

    def test_the_table_and_the_document_can_be_read_again(self, survey,
                                                          tmp_path):
        import csv

        document = st.analyse(survey)
        table = st.write_table(st.flat_rows(document),
                               str(tmp_path / "visits.csv"))
        stored = st.write_json(document, str(tmp_path / "trend.json"))

        with open(table, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 9
        assert rows[0]["density_per_km2"] == "4.0"
        assert rows[0]["change_pct"] == ""
        again = st.read_json(stored)
        assert again["species"] == document["species"]
        assert again["trends"]["roe deer"]["slope_per_day"] == pytest.approx(
            document["trends"]["roe deer"]["slope_per_day"])

    def test_a_table_of_nothing_still_has_its_header(self, tmp_path):
        path = st.write_table([], str(tmp_path / "none.csv"))

        with open(path, encoding="utf-8") as handle:
            assert handle.read().strip() == ",".join(st.VISIT_COLUMNS)

    def test_in_a_few_lines(self, survey):
        lines = st.describe(st.analyse(survey))

        assert lines[0] == ("3 visits: 2024-05-01 (2 flights), 2024-06-01, "
                            "2025-05-03")
        assert lines[1].startswith("roe deer: 4.00 -> 6.00 -> 9.00 /km². +")
        assert "over 12 months" in lines[1] and "per year" in lines[1]
        assert lines[1].endswith(
            "no clear trend - the interval includes no change.")
        assert len(lines) == 4

    def test_in_a_few_lines_without_the_ground(self, survey):
        survey[3]["seen"] = None

        lines = st.describe(st.analyse(survey))

        assert lines[1].startswith("roe deer: 4 -> 6 -> 9. ")


class TestWords:
    @pytest.mark.parametrize("percent, text", [
        (None, "n/a"), (12.4, "+12 %"), (-40.0, "-40 %"), (3.21, "+3.2 %"),
        (-0.04, "no change"), (0.0, "no change"), (1500.0, "+1,500 %"),
        (1e21, "more than +9,999 %")])
    def test_a_change_in_percent(self, percent, text):
        assert st.signed(percent) == text

    @pytest.mark.parametrize("days, text", [
        (0.2, "less than a day"), (1.0, "1 day"), (14.0, "14 days"),
        (31.0, "4 weeks"), (120.0, "4 months"), (367.0, "12 months"),
        (800.0, "2.2 years")])
    def test_a_length_of_time(self, days, text):
        assert st.span_text(days) == text

    def test_a_bound_of_nought_is_a_number(self):
        assert st.signed(0.0, "0 %") == "0 %"

    def test_a_trend_in_a_sentence(self):
        up = st.log_linear([0, 400], [50, 150], [1, 1], [50, 150])
        down = st.log_linear([0, 40], [150, 50], [1, 1], [150, 50])
        none = st.log_linear([0, 40], [0, 0], [1, 1], [0, 0])

        assert st.trend_sentence(up).startswith("+200 % over 13 months (95 %")
        assert st.trend_sentence(up).endswith("per year: an increase.")
        assert st.trend_sentence(down).startswith("-67 % over 6 weeks")
        assert st.trend_sentence(down).endswith(": a decrease.")
        assert st.trend_sentence(none) == (
            "No animals were counted on any visit.")

    def test_a_change_in_a_sentence(self):
        labels = {1: "2024-05-01", 2: "2024-06-01"}

        def said(*counts):
            found = st.compare(*counts)
            found.update({"from": 1, "to": 2})
            return st.change_sentence(found, labels)

        assert said(10, 10, 1, 30, 30, 1) == (
            "2024-05-01 to 2024-06-01: +200 % (p = 0.002): an increase.")
        assert said(30, 30, 1, 10, 10, 1).endswith(
            "-67 % (p = 0.002): a decrease.")
        assert said(10, 10, 1, 12, 12, 1).endswith(
            "+20 % (p = 0.832): within what chance explains.")
        assert said(0, 0, 1, 8, 8, 1).endswith(
            "from none to some (p = 0.008): an increase.")
        assert said(60, 60, 1, 0, 0, 1).endswith(
            "from some to none (p < 0.001): a decrease.")
        assert said(0, 0, 1, 0, 0, 1).endswith(
            "nothing counted on either visit.")
