# -*- coding: utf-8 -*-
"""Detection functions whose width depends on the conditions an animal was
seen in (core/detection_function.py): the fit, the density that goes with
it, and what the covariates do in the units of the survey."""
import math

import numpy as np
import pytest

from bambi_wildlife_detection.core import detection_function as df
from bambi_wildlife_detection.core import distance_effort

pytest.importorskip("scipy")

W = 60.0


def survey(seed=7, length=400_000.0, per_km2=30.0, canopy_effect=-0.012,
           height_effect=0.0, shape=None):
    """A line-transect survey of animals whose detectability depends on the
    canopy over them (0-90 %) and the height they were flown over at.

    :return: what was seen - distances, canopy, height - and the true
        density per km² of the animals that were there
    """
    rng = np.random.default_rng(seed)
    there = rng.poisson(per_km2 / 1e6 * 2 * W * length)
    x = rng.uniform(0, W, there)
    canopy = rng.uniform(0, 90, there)
    height = rng.choice([60.0, 80.0, 100.0], there)
    of_canopy = canopy_effect * (canopy - 45)
    of_height = height_effect * (height - 80)
    sigma = np.exp(math.log(30.0) + of_canopy + of_height)
    if shape is None:
        chance = np.exp(-x * x / (2 * sigma * sigma))
    else:
        chance = 1 - np.exp(-(x / sigma) ** -shape)
    seen = rng.uniform(size=there) < chance
    return {"x": x[seen], df.CANOPY: canopy[seen], df.HEIGHT: height[seen],
            "true": there / (2 * W * length) * 1e6, "length": length}


@pytest.fixture(scope="module")
def data():
    return survey()


# ---------------------------------------------------------------------------
# Names and candidates
# ---------------------------------------------------------------------------

def test_a_model_is_named_by_its_key_and_what_it_depends_on():
    assert df.model_name(df.HALF_NORMAL) == "half-normal"
    assert df.model_name(df.HAZARD_RATE, [df.CANOPY, df.HEIGHT]) == (
        "hazard-rate + canopy cover + flight height")
    assert (df.label(df.HEIGHT), df.unit(df.HEIGHT)) == ("flight height", "m")
    assert (df.label("wind_speed"), df.unit("wind_speed")) == (
        "wind speed", "")


def test_every_way_of_using_some_of_the_covariates():
    assert df.subsets([]) == []
    assert df.subsets(["a", "b"]) == [("a",), ("b",), ("a", "b")]
    assert len(df.subsets(["a", "b", "c"])) == 7


class TestUsable:
    def test_a_covariate_known_for_everything_that_varies_is_used(self):
        kept, refused = df.usable(
            {df.CANOPY: [float(i) for i in range(25)]}, 25)

        assert list(kept) == [df.CANOPY] and refused == {}
        assert kept[df.CANOPY][:3] == [0.0, 1.0, 2.0]

    def test_one_that_is_not_known_for_every_observation_is_not(self):
        values = [10.0] * 20 + [None] * 4 + [float("nan")]

        kept, refused = df.usable({df.CANOPY: values}, 25)

        assert kept == {}
        assert refused == {
            df.CANOPY: "known for 20 of 25 observations only"}

    def test_one_that_is_the_same_everywhere_says_nothing(self):
        kept, refused = df.usable({df.HEIGHT: [80.0] * 30}, 30)

        assert kept == {}
        assert refused == {
            df.HEIGHT: "the same for every observation (80 m)"}

    def test_a_handful_of_observations_is_too_few(self):
        kept, refused = df.usable(
            {df.CANOPY: [float(i) for i in range(12)]}, 12)

        assert kept == {}
        assert refused[df.CANOPY] == (
            "12 observations are too few to tell what it does (at least 20)")

    def test_each_is_judged_on_its_own(self):
        kept, refused = df.usable(
            {df.CANOPY: [float(i) for i in range(30)],
             df.HEIGHT: [80.0] * 29 + [None]}, 30)

        assert list(kept) == [df.CANOPY] and list(refused) == [df.HEIGHT]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_without_covariates_it_is_the_conventional_half_normal(
            self, data):
        """The plain model, against the textbook: the likelihood is at its
        maximum, and the strip width follows from sigma in closed form."""
        from scipy.special import erf

        model = df.fit(data["x"], W, df.HALF_NORMAL)
        sigma = math.exp(model.theta[0])
        width = sigma * math.sqrt(math.pi / 2) * erf(W / (sigma * 2 ** 0.5))

        assert model.name == "half-normal" and model.k == 1
        assert np.allclose(model.sigma, sigma)
        assert np.allclose(model.mu, width)
        # no other sigma does better
        best = -model.log_likelihood
        for other in (sigma * 0.97, sigma * 1.03):
            mu = other * math.sqrt(math.pi / 2) * erf(W / (other * 2 ** 0.5))
            nll = np.sum(data["x"] ** 2 / (2 * other ** 2) + math.log(mu))
            assert nll > best
        assert model.aic == pytest.approx(2 * 1 + 2 * best)
        assert model.shape is None and model.coefficients() == []

    def test_it_agrees_with_the_step_that_was_there_before(self, data):
        from bambi_wildlife_detection.bambi_processing import BambiProcessor

        for key in df.KEYS:
            before = BambiProcessor()._fit_detection_function(
                key, data["x"], W)
            model = df.fit(data["x"], W, key)

            assert model.aic == pytest.approx(before["aic"], abs=0.02)
            assert float(model.mu[0]) == pytest.approx(before["esw"],
                                                       rel=2e-3)

    def test_what_the_canopy_does_is_found_again(self, data):
        model = df.fit(data["x"], W, df.HALF_NORMAL,
                       {df.CANOPY: data[df.CANOPY]})
        row = model.coefficients()[0]

        assert model.name == "half-normal + canopy cover" and model.k == 2
        assert (row["name"], row["label"], row["unit"]) == (
            df.CANOPY, "canopy cover", "%")
        # the truth is -0.012 per percent of canopy
        assert abs(row["coefficient"] - (-0.012)) < 2.5 * row["se"]
        assert row["coefficient"] < 0 and row["p_value"] < 0.001
        assert row["z"] == pytest.approx(row["coefficient"] / row["se"])
        # under open sky an animal is seen from further away
        order = np.argsort(data[df.CANOPY])
        assert model.mu[order[0]] > 1.5 * model.mu[order[-1]]

    def test_a_covariate_that_matters_is_worth_its_parameter(self, data):
        plain = df.fit(data["x"], W, df.HALF_NORMAL)
        canopy = df.fit(data["x"], W, df.HALF_NORMAL,
                        {df.CANOPY: data[df.CANOPY]})
        height = df.fit(data["x"], W, df.HALF_NORMAL,
                        {df.HEIGHT: data[df.HEIGHT]})

        assert canopy.aic < plain.aic - 10
        # the height did nothing in this survey: a parameter for nothing
        assert height.aic > plain.aic - 2
        assert height.coefficients()[0]["p_value"] > 0.05

    def test_two_covariates_at_once(self):
        seen = survey(seed=11, height_effect=0.012)

        model = df.fit(seen["x"], W, df.HALF_NORMAL,
                       {df.CANOPY: seen[df.CANOPY],
                        df.HEIGHT: seen[df.HEIGHT]})
        canopy, height = model.coefficients()

        assert model.name == "half-normal + canopy cover + flight height"
        assert abs(canopy["coefficient"] + 0.012) < 2.5 * canopy["se"]
        assert abs(height["coefficient"] - 0.012) < 2.5 * height["se"]
        assert height["coefficient"] > 0 and height["p_value"] < 0.01

    def test_the_hazard_rate_takes_covariates_too(self):
        seen = survey(seed=5, shape=3.0)

        model = df.fit(seen["x"], W, df.HAZARD_RATE,
                       {df.CANOPY: seen[df.CANOPY]})
        row = model.coefficients()[0]

        assert model.name == "hazard-rate + canopy cover" and model.k == 3
        assert abs(row["coefficient"] + 0.012) < 2.5 * row["se"]
        assert 2.0 < model.shape < 4.5
        assert df.fit(seen["x"], W, df.HALF_NORMAL,
                      {df.CANOPY: seen[df.CANOPY]}).aic > model.aic

    def test_the_curve_starts_at_one_and_falls(self, data):
        model = df.fit(data["x"], W, df.HALF_NORMAL,
                       {df.CANOPY: data[df.CANOPY]})

        curve = model.mean_curve(np.linspace(0, W, 7))

        assert curve[0] == pytest.approx(1.0)
        assert np.all(np.diff(curve) < 0) and 0 < curve[-1] < 0.5

    def test_too_little_to_fit(self):
        assert df.fit([5.0], W) is None
        assert df.fit([5.0, 9.0, 12.0], W, df.HALF_NORMAL,
                      {df.CANOPY: [40.0, 40.0, 40.0]}) is None

    def test_an_unknown_key(self):
        with pytest.raises(ValueError):
            df.fit([1.0, 2.0, 3.0], W, "uniform")


# ---------------------------------------------------------------------------
# The density
# ---------------------------------------------------------------------------

class TestEstimate:
    def _lines(self, n, k=10, seed=1):
        return np.random.default_rng(seed).integers(0, k, n)

    def test_the_density_that_was_there_is_found_again(self, data):
        covariates = {df.CANOPY: data[df.CANOPY]}
        model = df.fit(data["x"], W, df.HALF_NORMAL, covariates)
        n = data["x"].size

        found = df.estimate(model, np.ones(n), self._lines(n),
                            [data["length"] / 10] * 10, covariates)

        density = found["density_m2"] * 1e6
        assert abs(density - data["true"]) < 2.5 * found["cv_density"] * \
            density
        assert density == pytest.approx(data["true"], rel=0.12)
        assert found["group_density_m2"] == pytest.approx(found["density_m2"])
        assert 0.4 < found["detection_probability"] < 0.75
        assert found["esw_m"] == pytest.approx(
            found["detection_probability"] * W)

    def test_without_covariates_it_is_n_over_two_l_mu(self, data):
        model = df.fit(data["x"], W, df.HALF_NORMAL)
        n = data["x"].size

        found = df.estimate(model, np.ones(n), self._lines(n),
                            [data["length"] / 10] * 10)

        assert found["density_m2"] == pytest.approx(
            n / (2 * data["length"] * float(model.mu[0])))
        assert found["esw_m"] == pytest.approx(float(model.mu[0]))

    def test_groups_count_with_their_animals(self, data):
        covariates = {df.CANOPY: data[df.CANOPY]}
        model = df.fit(data["x"], W, df.HALF_NORMAL, covariates)
        n = data["x"].size
        sizes = np.where(np.arange(n) % 4 == 0, 5, 1)

        found = df.estimate(model, sizes, self._lines(n),
                            [data["length"] / 10] * 10, covariates)

        assert found["density_m2"] == pytest.approx(
            float(np.sum(sizes / model.mu)) / (2 * data["length"]))
        assert found["density_m2"] == pytest.approx(
            2 * found["group_density_m2"], rel=0.1)

    def test_the_uncertainty_has_two_parts(self, data):
        covariates = {df.CANOPY: data[df.CANOPY]}
        model = df.fit(data["x"], W, df.HALF_NORMAL, covariates)
        n = data["x"].size
        lines = self._lines(n)
        lengths = [data["length"] / 10] * 10

        found = df.estimate(model, np.ones(n), lines, lengths, covariates)

        per_line = np.bincount(lines, weights=1 / model.mu, minlength=10)
        between, how = distance_effort.encounter_rate_cv(
            per_line.tolist(), lengths)
        assert how == distance_effort.BETWEEN_LINES
        assert found["encounter_variance"] == how
        assert found["cv_encounter"] == pytest.approx(between)
        assert 0 < found["cv_detection"] < 0.2
        assert found["cv_density"] == pytest.approx(math.hypot(
            found["cv_encounter"], found["cv_detection"]))

    def test_a_single_line_falls_back_on_the_sightings_themselves(self, data):
        covariates = {df.CANOPY: data[df.CANOPY]}
        model = df.fit(data["x"], W, df.HALF_NORMAL, covariates)
        n = data["x"].size

        found = df.estimate(model, np.ones(n), np.zeros(n, dtype=int),
                            [data["length"]], covariates)

        weights = 1 / model.mu
        assert found["encounter_variance"] == distance_effort.POISSON
        assert found["cv_encounter"] == pytest.approx(
            math.sqrt(np.sum(weights ** 2)) / weights.sum())
        # more than 1/sqrt(n): the sightings do not all stand for the same
        assert found["cv_encounter"] > 1 / math.sqrt(n)

    def test_a_tight_fit_is_surer_than_a_loose_one(self):
        few, many = survey(seed=3, length=40_000.0), survey(seed=3)
        cvs = []
        for seen in (few, many):
            covariates = {df.CANOPY: seen[df.CANOPY]}
            model = df.fit(seen["x"], W, df.HALF_NORMAL, covariates)
            n = seen["x"].size
            cvs.append(df.estimate(
                model, np.ones(n), self._lines(n),
                [seen["length"] / 10] * 10, covariates)["cv_detection"])

        assert cvs[0] > 2 * cvs[1]


# ---------------------------------------------------------------------------
# In the units of the survey
# ---------------------------------------------------------------------------

class TestEffects:
    def test_the_strip_width_at_little_and_at_much_canopy(self, data):
        covariates = {df.CANOPY: data[df.CANOPY]}
        model = df.fit(data["x"], W, df.HALF_NORMAL, covariates)

        row = df.effects(model, covariates)[0]

        low, high = np.percentile(data[df.CANOPY], [10, 90])
        assert (row["low"], row["high"]) == pytest.approx((low, high))
        assert row["esw_low_m"] > 1.4 * row["esw_high_m"]
        # the truth: sigma 30 m at 45 % canopy, falling 1.2 % per percent
        sigma = 30 * math.exp(-0.012 * (low - 45))
        from scipy.special import erf
        true = sigma * math.sqrt(math.pi / 2) * erf(W / (sigma * 2 ** 0.5))
        assert row["esw_low_m"] == pytest.approx(true, rel=0.12)
        assert row["min"] <= row["low"] <= row["mean"] <= row["high"] <= \
            row["max"]

    def test_the_other_covariate_is_held_at_its_mean(self):
        seen = survey(seed=11, height_effect=0.012)
        covariates = {df.CANOPY: seen[df.CANOPY], df.HEIGHT: seen[df.HEIGHT]}
        model = df.fit(seen["x"], W, df.HALF_NORMAL, covariates)

        canopy, height = df.effects(model, covariates)

        assert canopy["esw_low_m"] > canopy["esw_high_m"]
        assert height["esw_low_m"] < height["esw_high_m"]
        assert (height["low"], height["high"]) == (60.0, 100.0)

    def test_in_a_sentence(self):
        row = {"label": "canopy cover", "unit": "%", "coefficient": -0.015,
               "p_value": 0.0002, "low": 8.4, "high": 75.2,
               "esw_low_m": 48.52, "esw_high_m": 22.61}

        assert df.effect_sentence(row) == (
            "The more canopy cover, the less far animals are seen: the "
            "effective strip half-width is 48.5 m at 8 % and 22.6 m at "
            "75 % (p < 0.001).")
        assert df.effect_sentence(dict(
            row, label="flight height", unit="m", coefficient=0.01,
            p_value=0.012)).startswith(
            "The more flight height, the further away animals are seen: ")

    def test_an_effect_nobody_can_tell_is_not_called_one(self):
        row = {"label": "flight height", "unit": "m", "coefficient": -0.0005,
               "p_value": 0.884, "low": 60.0, "high": 100.0,
               "esw_low_m": 37.5, "esw_high_m": 37.0}

        assert df.effect_sentence(row) == (
            "Flight height: no clear effect - the effective strip "
            "half-width is 37.5 m at 60 m and 37.0 m at 100 m (p = 0.884).")
