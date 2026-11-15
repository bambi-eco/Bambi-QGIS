# -*- coding: utf-8 -*-
"""The statistics of counts over cells (core/spatial_stats.py)."""
import math
import random

import pytest

from bambi_wildlife_detection.core import spatial_stats as ss

np = pytest.importorskip("numpy")


def _square_neighbours(side):
    """Neighbours (shared edge) of the cells of a *side* x *side* square,
    numbered row by row."""
    near = []
    for row in range(side):
        for col in range(side):
            around = []
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = row + d_row, col + d_col
                if 0 <= r < side and 0 <= c < side:
                    around.append(r * side + c)
            near.append(around)
    return near


# ---------------------------------------------------------------------------
# Density with an interval
# ---------------------------------------------------------------------------

class TestRatioInterval:
    def test_the_estimate_is_the_ratio_of_the_sums(self):
        result = ss.ratio_interval([2, 0, 5, 1], [1.0, 0.5, 2.0, 0.5])

        assert result["estimate"] == pytest.approx(8 / 4.0)
        assert result["low"] <= result["estimate"] <= result["high"]
        assert result["n"] == 4 and result["n_boot"] == 2000

    def test_the_same_data_give_the_same_interval(self):
        counts = [0, 0, 1, 3, 0, 7, 2, 0, 0, 1]
        ground = [1.0] * 10

        assert ss.ratio_interval(counts, ground) == \
            ss.ratio_interval(counts, ground)
        assert ss.ratio_interval(counts, ground, seed=1)["se"] != \
            ss.ratio_interval(counts, ground, seed=2)["se"]

    def test_cells_that_all_agree_leave_no_doubt(self):
        result = ss.ratio_interval([3, 3, 3, 3], [1.0, 1.0, 1.0, 1.0])

        assert result["low"] == result["high"] == 3.0
        assert result["se"] == pytest.approx(0.0, abs=1e-12)

    def test_the_interval_is_about_what_theory_says(self):
        """Many equal cells with Poisson counts: the standard error of the
        mean is sqrt(mean / n)."""
        rng = np.random.default_rng(11)
        counts = rng.poisson(4.0, size=400)
        result = ss.ratio_interval(counts, np.ones(400))

        expected = math.sqrt(counts.var(ddof=1) / 400)
        assert result["se"] == pytest.approx(expected, rel=0.1)
        assert result["high"] - result["low"] == pytest.approx(
            2 * 1.96 * expected, rel=0.12)

    def test_clumped_counts_make_it_wider(self):
        even = ss.ratio_interval([2] * 50, [1.0] * 50)
        clumped = ss.ratio_interval([0] * 45 + [20] * 5, [1.0] * 50)

        assert even["estimate"] == clumped["estimate"] == 2.0
        assert clumped["high"] - clumped["low"] > 1.5
        assert even["high"] - even["low"] == pytest.approx(0.0, abs=1e-12)

    def test_ground_weighs_a_cell(self):
        """A cell seen in full says more than one seen by a tenth."""
        result = ss.ratio_interval([10, 1], [1.0, 0.1])

        assert result["estimate"] == pytest.approx(10.0)

    def test_one_cell_has_no_interval(self):
        result = ss.ratio_interval([4], [2.0])

        assert result["estimate"] == 2.0
        assert result["low"] is None and result["se"] is None

    def test_no_ground_no_estimate(self):
        assert ss.ratio_interval([], [])["estimate"] is None
        assert ss.ratio_interval([0, 0], [0.0, 0.0])["estimate"] is None

    def test_many_cells_are_resampled_in_batches(self):
        counts = np.arange(30000) % 3
        result = ss.ratio_interval(counts, np.ones(30000), n_boot=300)

        assert result["estimate"] == pytest.approx(1.0, abs=1e-3)
        assert result["low"] < 1.0 < result["high"]
        assert result["n_boot"] == 300


# ---------------------------------------------------------------------------
# Clumped, random or regular
# ---------------------------------------------------------------------------

class TestDispersion:
    def test_it_is_the_variance_to_mean_ratio_for_equal_cells(self):
        counts = [0, 2, 1, 4, 0, 3, 5, 1]
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / (len(counts) - 1)

        result = ss.dispersion(counts)

        assert result["index"] == pytest.approx(variance / mean)
        assert result["df"] == 7 and result["n"] == 8
        assert result["chi2"] == pytest.approx(7 * variance / mean)

    def test_animals_scattered_at_random_are_called_random(self):
        pytest.importorskip("scipy")
        rng = np.random.default_rng(5)
        result = ss.dispersion(rng.poisson(3.0, size=300))

        assert result["index"] == pytest.approx(1.0, abs=0.2)
        assert result["pattern"] == ss.RANDOM

    def test_clumped_animals_are_called_clustered(self):
        result = ss.dispersion([0] * 40 + [12, 9, 15, 11])

        assert result["index"] > 5
        assert result["p_clustered"] < 0.001
        assert result["pattern"] == ss.CLUSTERED

    def test_evenly_spaced_animals_are_called_regular(self):
        result = ss.dispersion([3] * 30 + [4] * 30)

        assert result["index"] < 0.2
        assert result["p_regular"] < 0.001
        assert result["pattern"] == ss.REGULAR

    def test_ground_is_taken_into_account(self):
        """Twice the ground, twice the animals: that is no clumping."""
        counts = [2, 4, 2, 4, 2, 4, 2, 4]
        ground = [1.0, 2.0] * 4

        assert ss.dispersion(counts, ground)["index"] == pytest.approx(0.0)
        assert ss.dispersion(counts)["index"] > 0.3

    def test_cells_without_ground_are_left_out(self):
        result = ss.dispersion([1, 5, 2], [1.0, 0.0, 1.0])

        assert result["n"] == 2 and result["df"] == 1

    def test_too_little_says_nothing(self):
        assert ss.dispersion([3])["index"] is None
        assert ss.dispersion([0, 0, 0])["index"] is None
        assert ss.dispersion([])["pattern"] is None

    def test_the_approximation_is_close_to_the_distribution(self):
        stats = pytest.importorskip("scipy.stats")
        import builtins

        real_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name.startswith("scipy"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        for x, df in ((3.0, 5), (20.0, 10), (45.0, 30), (150.0, 100)):
            exact = float(stats.chi2.sf(x, df))
            builtins.__import__ = no_scipy
            try:
                close = ss._chi2_sf(x, df)
            finally:
                builtins.__import__ = real_import
            assert close == pytest.approx(exact, abs=0.01)


# ---------------------------------------------------------------------------
# Hot spots
# ---------------------------------------------------------------------------

def _gi_star_by_the_book(values, near):
    """Ord and Getis (1995), eq. 14, with weights of one."""
    n = len(values)
    mean = sum(values) / n
    s = math.sqrt(sum(v * v for v in values) / n - mean * mean)
    scores = []
    for i in range(n):
        weights = [1.0 if (j == i or j in near[i]) else 0.0 for j in range(n)]
        w_sum = sum(weights)
        w_squares = sum(w * w for w in weights)
        top = sum(w * v for w, v in zip(weights, values)) - mean * w_sum
        bottom = s * math.sqrt((n * w_squares - w_sum ** 2) / (n - 1))
        scores.append(top / bottom)
    return scores


class TestGiStar:
    def test_it_is_the_formula(self):
        rng = random.Random(4)
        values = [rng.uniform(0, 10) for _ in range(36)]
        near = _square_neighbours(6)

        assert ss.gi_star(values, near) == pytest.approx(
            _gi_star_by_the_book(values, near))

    def test_a_patch_of_high_values_is_a_hot_spot(self):
        side = 9
        values = [0.5] * (side * side)
        for row in (3, 4, 5):
            for col in (3, 4, 5):
                values[row * side + col] = 9.0
        scores = ss.gi_star(values, _square_neighbours(side))

        middle = scores[4 * side + 4]
        assert middle > 2.58 and ss.spot(middle) == "hot spot (99 %)"
        corner = scores[0]
        assert corner < 0 and ss.spot(corner) == ""

    def test_a_cell_is_judged_with_its_neighbours(self):
        """The middle of a patch of high cells scores higher than a high
        cell that has only low ones around it - and that one scores the
        same as the low cell next to it: they share most of what is
        summed."""
        side = 9
        patch = [1.0] * (side * side)
        for row in (3, 4, 5):
            for col in (3, 4, 5):
                patch[row * side + col] = 9.0
        alone = [1.0] * (side * side)
        alone[4 * side + 4] = 9.0
        near = _square_neighbours(side)
        lone = ss.gi_star(alone, near)

        assert ss.gi_star(patch, near)[4 * side + 4] > 1.5 * lone[4 * side + 4]
        assert lone[4 * side + 5] == pytest.approx(lone[4 * side + 4])
        assert lone[0] < 0

    def test_a_patch_of_low_values_is_a_cold_spot(self):
        side = 9
        values = [8.0 + (i % 3) for i in range(side * side)]
        for row in (3, 4, 5):
            for col in (3, 4, 5):
                values[row * side + col] = 0.0
        scores = ss.gi_star(values, _square_neighbours(side))

        assert ss.spot(scores[4 * side + 4]).startswith("cold spot")

    def test_cells_that_are_all_alike_have_no_score(self):
        assert ss.gi_star([2.0] * 9, _square_neighbours(3)) == [None] * 9
        assert ss.gi_star([5.0], [[]]) == [None]
        assert ss.gi_star([], []) == []

    def test_a_neighbourhood_that_is_everything_has_no_score(self):
        assert ss.gi_star([1.0, 3.0], [[1], [0]]) == [None, None]

    def test_what_a_score_is_called(self):
        assert ss.spot(None) == "" and ss.spot(0.0) == ""
        assert ss.spot(1.64) == "" and ss.spot(1.65) == "hot spot (90 %)"
        assert ss.spot(1.97) == "hot spot (95 %)"
        assert ss.spot(2.58) == "hot spot (99 %)"
        assert ss.spot(-1.7) == "cold spot (90 %)"
        assert ss.spot(-2.0) == "cold spot (95 %)"
        assert ss.spot(-3.0) == "cold spot (99 %)"


# ---------------------------------------------------------------------------
# Are neighbours alike
# ---------------------------------------------------------------------------

def _morans_i_by_the_book(values, near):
    n = len(values)
    mean = sum(values) / n
    links = sum(len(others) for others in near)
    top = sum((values[i] - mean) * (values[j] - mean)
              for i in range(n) for j in near[i])
    bottom = sum((v - mean) ** 2 for v in values)
    return n / links * top / bottom


class TestMoransI:
    def test_it_is_the_formula(self):
        rng = random.Random(8)
        values = [rng.uniform(0, 10) for _ in range(49)]
        near = _square_neighbours(7)

        result = ss.morans_i(values, near, permutations=0)

        assert result["i"] == pytest.approx(_morans_i_by_the_book(values, near))
        assert result["expected"] == pytest.approx(-1 / 48)
        assert result["links"] == 2 * 7 * 6
        assert result["p_value"] is None

    def test_a_gradient_is_clustered(self):
        side = 8
        values = [row + col for row in range(side) for col in range(side)]

        result = ss.morans_i(values, _square_neighbours(side))

        assert result["i"] > 0.7
        assert result["p_value"] == pytest.approx(1 / 1000)
        assert result["pattern"] == ss.CLUSTERED and result["z"] > 5

    def test_a_chess_board_is_the_opposite(self):
        side = 8
        values = [(row + col) % 2 for row in range(side)
                  for col in range(side)]

        result = ss.morans_i(values, _square_neighbours(side))

        assert result["i"] == pytest.approx(-1.0)
        assert result["pattern"] == ss.REGULAR

    def test_shuffled_values_are_random(self):
        rng = random.Random(21)
        values = [rng.uniform(0, 10) for _ in range(100)]

        result = ss.morans_i(values, _square_neighbours(10))

        assert abs(result["i"]) < 0.15
        assert result["p_value"] > 0.05 and result["pattern"] == ss.RANDOM

    def test_the_same_data_give_the_same_answer(self):
        values = [(7 * i) % 11 for i in range(36)]
        near = _square_neighbours(6)

        assert ss.morans_i(values, near) == ss.morans_i(values, near)

    def test_where_it_cannot_be_worked_out(self):
        assert ss.morans_i([1.0, 2.0], [[1], [0]])["i"] is None
        assert ss.morans_i([1.0, 2.0, 3.0], [[], [], []])["i"] is None
        assert ss.morans_i([4.0] * 9, _square_neighbours(3))["i"] is None


# ---------------------------------------------------------------------------
# A share
# ---------------------------------------------------------------------------

class TestWilson:
    def test_half(self):
        low, high = ss.wilson(5, 10)

        assert low == pytest.approx(0.2366, abs=1e-4)
        assert high == pytest.approx(0.7634, abs=1e-4)

    def test_none_and_all_stay_inside_zero_and_one(self):
        low, high = ss.wilson(0, 20)
        assert low == 0.0 and 0.1 < high < 0.2
        low, high = ss.wilson(20, 20)
        assert high == 1.0 and 0.8 < low < 0.9

    def test_more_cells_less_doubt(self):
        few = ss.wilson(3, 10)
        many = ss.wilson(300, 1000)

        assert many[1] - many[0] < (few[1] - few[0]) / 5

    def test_no_cells_no_share(self):
        assert ss.wilson(0, 0) == (None, None)


def test_neighbours_by_name_become_neighbours_by_position():
    near = {"a": ["b", "x"], "b": ["a", "c"], "c": ["b"]}

    assert ss.index_neighbours(["c", "a", "b"], near) == [[2], [2], [1, 0]]
    assert ss.index_neighbours(["a", "z"], near) == [[], []]
