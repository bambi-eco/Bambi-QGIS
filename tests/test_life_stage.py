# -*- coding: utf-8 -*-
"""Life stage from box area (paper §3.3, Eq. 4).

The fixtures follow the paper's flights: A2 holds a juvenile that must be
flagged, while A1 and C hold none and must not produce a false one. That last
property is the harder half — in any herd someone is smallest.

One departure worth knowing: on a flight with only a handful of individuals the
candidate sits in the lower half of the distribution and so inflates the very
interquartile range its size gap is measured against, and the rule declines to
call it. That is the cautious answer, which is the one a census wants, but it
is reported rather than silent.
"""
import pytest

from bambi_wildlife_detection.core import life_stage
from bambi_wildlife_detection.core.life_stage import Config


def _areas(values):
    """``{track_id: {...}}`` from a list of areas."""
    return {index + 1: {"area": float(area), "frames": 20}
            for index, area in enumerate(values)}


def _labels(assessments):
    return {item.track_id: item.label for item in assessments}


# ---------------------------------------------------------------------------
# The paper's flights
# ---------------------------------------------------------------------------

class TestPaperCohorts:

    def test_a2_juvenile_is_flagged(self):
        """A2 D5 sits at 0.53x the cohort median (Fig. 7)."""
        # Ten adults clustered around 1000, and the juvenile far below.
        areas = _areas([530, 980, 1000, 1010, 1020, 1040, 1050, 1060, 1090,
                        1120])
        found = life_stage.assess(areas)

        assert _labels(found)[1] == life_stage.JUVENILE
        assert all(item.label == life_stage.ADULT
                   for item in found if item.track_id != 1)

    def test_a_juvenile_in_a_tightly_clustered_herd_is_flagged(self):
        """B D2 sits at 0.60x its cohort median, among adults of a size."""
        areas = _areas([673, 1080, 1100, 1110, 1120, 1130, 1150])
        found = life_stage.assess(areas)
        assert _labels(found)[1] == life_stage.JUVENILE

    def test_a_small_flight_is_answered_cautiously(self):
        """With few individuals the outlier sits in the lower half and so
        inflates the very interquartile range its gap is tested against. The
        rule then declines — the right failure mode for a census, where a
        false juvenile is worse than a missing one — but it must say so."""
        areas = _areas([673, 1050, 1120, 1180, 1210])
        found = life_stage.assess(areas)

        assert all(item.label == life_stage.ADULT for item in found)
        # The candidate is still reported as a low outlier, so the user can
        # tell "nothing was small" from "the cohort was too thin to confirm".
        assert _labels(found)[1] == life_stage.ADULT
        assert found[0].low_outlier is True
        assert "not wide enough" in life_stage.explain(found)

    def test_a1_has_no_juvenile_to_find(self):
        """A1's smallest animals lie on a smooth size continuum, so the test
        must not over-fire — this is the half that matters."""
        areas = _areas([900, 950, 1000, 1050, 1100, 1150, 1200, 1250])
        found = life_stage.assess(areas)
        assert all(item.label == life_stage.ADULT for item in found)

    def test_c_has_no_juvenile_to_find(self):
        areas = _areas([820, 900, 970, 1040, 1110])
        found = life_stage.assess(areas)
        assert all(item.label == life_stage.ADULT for item in found)

    def test_the_cohort_ratio_matches_the_published_figure(self):
        areas = _areas([530, 980, 1000, 1010, 1020, 1040, 1050, 1060, 1090,
                        1120])
        ratios = life_stage.cohort_ratio(life_stage.assess(areas))
        assert ratios[1] == pytest.approx(0.53, abs=0.02)


# ---------------------------------------------------------------------------
# The two conditions, separately
# ---------------------------------------------------------------------------

class TestConditions:

    def test_a_low_z_alone_is_not_enough(self):
        """Several small animals together are a size continuum, not juveniles:
        each one's gap to the next is tiny even though all sit low."""
        areas = _areas([500, 520, 540, 1000, 1010, 1020, 1030])
        found = life_stage.assess(areas)
        assert all(item.label == life_stage.ADULT for item in found)

    def test_a_large_gap_alone_is_not_enough(self):
        """A gap high up the range is not a juvenile — z must also be low."""
        areas = _areas([1000, 1010, 1020, 1030, 3000])
        found = life_stage.assess(areas)
        assert _labels(found)[5] == life_stage.ADULT

    def test_both_conditions_together_flag_the_outlier(self):
        areas = _areas([400, 1000, 1010, 1020, 1030, 1040, 1050])
        assert _labels(life_stage.assess(areas))[1] == life_stage.JUVENILE

    def test_the_z_threshold_is_configurable(self):
        areas = _areas([700, 1000, 1010, 1020, 1030])
        strict = life_stage.assess(areas, Config(z_threshold=-20.0))
        assert all(item.label == life_stage.ADULT for item in strict)

    def test_the_gap_factor_is_configurable(self):
        areas = _areas([400, 1000, 1010, 1020, 1030, 1040, 1050])
        strict = life_stage.assess(areas, Config(iqr_factor=100.0))
        assert all(item.label == life_stage.ADULT for item in strict)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:

    def test_too_few_individuals_declines_to_answer(self):
        """A robust z-score over three animals is not robust."""
        assert life_stage.assess(_areas([400, 1000, 1010])) == []

    def test_the_minimum_is_configurable(self):
        found = life_stage.assess(_areas([400, 1000, 1010]),
                                  Config(min_individuals=3))
        assert len(found) == 3

    def test_identical_sizes_produce_no_outlier(self):
        """A zero MAD means nobody is unusual, not that everyone is."""
        found = life_stage.assess(_areas([1000] * 6))
        assert all(item.label == life_stage.ADULT for item in found)
        assert all(item.z == 0.0 for item in found)

    def test_a_known_male_is_an_adult_whatever_its_box_says(self):
        """The cue that marked him is antlers, not size."""
        areas = _areas([400, 1000, 1010, 1020, 1030, 1040, 1050])
        found = life_stage.assess(areas, adults=[1])
        assert _labels(found)[1] == life_stage.ADULT

    def test_the_largest_individual_is_never_a_juvenile(self):
        areas = _areas([1000, 1010, 1020, 5000])
        assert _labels(life_stage.assess(areas))[4] == life_stage.ADULT

    def test_every_individual_gets_an_answer(self):
        areas = _areas([400, 1000, 1010, 1020, 1030, 1040, 1050])
        found = life_stage.assess(areas)
        # Non-outliers are adults, not unknowns: the test is a low-outlier
        # flag, and every non-outlier in a surveyed herd is an adult.
        assert len(found) == len(areas)
        assert {item.label for item in found} == {life_stage.JUVENILE,
                                                  life_stage.ADULT}


# ---------------------------------------------------------------------------
# Areas from tracks
# ---------------------------------------------------------------------------

class TestTrackAreas:

    def _rows(self):
        return [
            {"track_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
             "gx1": 0, "gy1": 0, "gx2": 1, "gy2": 2},
            {"track_id": 1, "x1": 0, "y1": 0, "x2": 20, "y2": 10,
             "gx1": 0, "gy1": 0, "gx2": 2, "gy2": 2},
            {"track_id": 1, "x1": 0, "y1": 0, "x2": 30, "y2": 10,
             "gx1": 0, "gy1": 0, "gx2": 3, "gy2": 2},
            {"track_id": 2, "x1": 0, "y1": 0, "x2": 5, "y2": 5,
             "gx1": 0, "gy1": 0, "gx2": 1, "gy2": 1},
        ]

    def test_the_median_area_is_taken_per_track(self):
        areas = life_stage.track_areas(self._rows())
        assert areas[1]["area"] == 200.0      # median of 100, 200, 300
        assert areas[1]["frames"] == 3

    def test_one_bad_box_does_not_move_a_track(self):
        """The median is chosen over the mean precisely for this."""
        rows = self._rows() + [
            {"track_id": 1, "x1": 0, "y1": 0, "x2": 900, "y2": 900,
             "gx1": 0, "gy1": 0, "gx2": 90, "gy2": 90}]
        assert life_stage.track_areas(rows)[1]["area"] == 250.0

    def test_geo_areas_are_metric(self):
        areas = life_stage.track_areas(self._rows(), geo=True)
        assert areas[1]["area"] == 4.0        # median of 2, 4, 6 m²

    def test_zero_area_boxes_are_ignored(self):
        rows = [{"track_id": 1, "x1": 5, "y1": 5, "x2": 5, "y2": 5,
                 "gx1": 0, "gy1": 0, "gx2": 0, "gy2": 0}]
        assert life_stage.track_areas(rows) == {}

    def test_reversed_corners_are_tolerated(self):
        rows = [{"track_id": 1, "x1": 10, "y1": 10, "x2": 0, "y2": 0,
                 "gx1": 0, "gy1": 0, "gx2": 0, "gy2": 0}]
        assert life_stage.track_areas(rows)[1]["area"] == 100.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestExplain:

    def test_a_juvenile_is_named_with_its_z(self):
        found = life_stage.assess(_areas([400, 1000, 1010, 1020, 1030, 1040, 1050]))
        line = life_stage.explain(found)
        assert "1 juvenile(s)" in line and "track 1" in line

    def test_no_juvenile_says_so_plainly(self):
        found = life_stage.assess(_areas([900, 1000, 1010, 1020, 1030]))
        assert "no juvenile" in life_stage.explain(found)

    def test_too_few_individuals_is_explained(self):
        line = life_stage.explain([], Config())
        assert "too few individuals" in line

    def test_the_area_source_is_named(self):
        """Metric and pixel areas are not comparable, so which was used has to
        travel with the answer."""
        found = life_stage.assess(_areas([400, 1000, 1010, 1020, 1030, 1040, 1050]))
        assert "orthorectified" in life_stage.explain(
            found, source=life_stage.AREA_GEO)


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

class TestStatistics:

    def test_mad_of_a_symmetric_set(self):
        assert life_stage.median_absolute_deviation([1, 2, 3, 4, 5]) == 1.0

    def test_mad_ignores_a_wild_outlier(self):
        """Which is the whole reason for using it over a standard deviation."""
        assert life_stage.median_absolute_deviation(
            [1, 2, 3, 4, 5, 1000]) == pytest.approx(1.5)

    def test_quartiles_of_an_even_set(self):
        assert life_stage._quartiles([1, 2, 3, 4]) == (1.5, 3.5)

    def test_quartiles_of_an_odd_set_exclude_the_median(self):
        assert life_stage._quartiles([1, 2, 3, 4, 5]) == (1.5, 4.5)
