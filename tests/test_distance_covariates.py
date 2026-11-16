# -*- coding: utf-8 -*-
"""Distance sampling that lets canopy cover and flight height explain how
far away an animal is seen: the step (bambi_survey_analytics.py on top of
core/detection_function.py), on flights in the stores."""
import json
import math
import os

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    detection_function, environment, groups, store)
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight

pytest.importorskip("scipy")
pytest.importorskip("rasterio")

#: The drone flies due east along y = 0, 5 m per frame: a leg over open
#: ground (x 0-1000 m, frames 0-200) and one over forest (1000-2000 m).
LEG = 1000.0
W = 60.0
#: The DEM's origin: the poses are 80 m above it, the animals stand 300 m
#: above sea level - 100 m below the camera.
DEM_ORIGIN = (ORIGIN[0], ORIGIN[1], 320.0)
GROUND = 300.0


def _half_normal(rng, sigma, count):
    """*count* distances from the line as a half-normal curve leaves them."""
    found = []
    while len(found) < count:
        distance = abs(rng.normal(0.0, sigma))
        if distance <= W:
            found.append(distance)
    return found


def _animals(open_sigma=32.0, forest_sigma=9.0, per_leg=(70, 45), seed=3,
             alike=False):
    """(id, x along the line, y off it): far out over open ground, only
    close to the line under the trees. With *alike* the animals under the
    trees are as far from the line as those in the open, one for one."""
    rng = np.random.default_rng(seed)
    animals = []
    drawn = None
    for leg, (sigma, count) in enumerate(zip((open_sigma, forest_sigma),
                                             per_leg)):
        if drawn is None or not alike:
            drawn = _half_normal(rng, sigma, count)
        for n, distance in enumerate(drawn):
            x = leg * LEG + 10.0 + (LEG - 20.0) * (n + 0.5) / count
            side = 1.0 if n % 2 else -1.0
            animals.append((len(animals) + 1, x, side * distance))
    return animals


def _flight(folder, animals, canopy=True, dem=True, heights=None):
    """A flight over *animals*; *heights* gives the camera's height above
    the DEM origin per leg (80 m unless given)."""
    tracks = []
    for ident, x, y in animals:
        frame = int(x / 5)
        tracks.append({"id": ident, "species": "roe deer", "at": (x, y),
                       "frames": (max(0, frame - 2), frame + 2)})
    make_flight(folder, tracks, n_frames=410)
    conn = store.open_store(store.project_path(folder), store.PROJECT)
    with store.transaction(conn):               # every animal on its own
        row = conn.execute(
            "SELECT species_id FROM species WHERE name = 'roe deer'"
        ).fetchone()
        groups.store_distance(conn, row["species_id"], 0)
    conn.close()

    if heights is not None:
        path = os.path.join(folder, "poses_t.json")
        with open(path, encoding="utf-8") as handle:
            poses = json.load(handle)
        for frame, image in enumerate(poses["images"]):
            image["location"][2] = heights[0 if frame < 200 else 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(poses, handle)

    os.makedirs(os.path.join(folder, "transects_t"))
    with open(os.path.join(folder, "transects_t", "transects.json"),
              "w") as handle:
        json.dump({"version": 1, "modality": "t", "transects": [
            {"id": 1, "name": "open", "start_frame": 0, "end_frame": 200},
            {"id": 2, "name": "forest", "start_frame": 200,
             "end_frame": 400}]}, handle)
    route = os.path.join(folder, "flight_route_t")
    os.makedirs(route)
    entries = [{"track_id": ident, "class_id": 1, "distance_m": abs(y),
                "last_frame": int(x / 5) + 2,
                "detection_center": [ORIGIN[0] + x, ORIGIN[1] + y, GROUND],
                "foot_point": [ORIGIN[0] + x, ORIGIN[1]]}
               for ident, x, y in animals]
    with open(os.path.join(route, "perpendicular_tracks_t.json"),
              "w") as handle:
        json.dump({"tracks": entries}, handle)

    if canopy:
        # 10 m cells: none over the first leg, 80 % over the second
        grid = environment.Grid(EPSG, ORIGIN[0] - 100.0, ORIGIN[1] + 200.0,
                                220, 40, 10.0)
        cover = np.zeros(grid.shape, dtype=np.float32)
        cover[:, 110:] = 80.0
        environment.ensure_folder(folder)
        environment.write_raster(
            environment.layer_path(folder, "tree_cover"), cover, grid)
    dem_json = ""
    if dem:
        dem_json = os.path.join(folder, "dem.json")
        with open(dem_json, "w", encoding="utf-8") as handle:
            json.dump({"origin": list(DEM_ORIGIN)}, handle)
    return dem_json


def _run(tmp_path, *flights, **settings):
    """Run the step over ``(name, folder, dem json)`` flights."""
    survey_folder = str(tmp_path / "survey")
    survey = Survey(folder=survey_folder, modality="t", flights=[
        SurveyFlight(name=name, target_folder=folder, epsg=EPSG, dem=dem)
        for name, folder, dem in flights])
    log = []
    config = dict({"survey": survey.to_dict(), "target_epsg": EPSG,
                   "ds_truncation": W}, **settings)
    BambiProcessor().run_distance_sampling(config, log_fn=log.append)
    with open(os.path.join(survey_folder, "analytics_t",
                           "distance_sampling_tracks.json")) as handle:
        return json.load(handle), log


@pytest.fixture(scope="module")
def forest(tmp_path_factory):
    """Open ground and forest, flown at one height."""
    folder = str(tmp_path_factory.mktemp("forest") / "Forest")
    return "Forest", folder, _flight(folder, _animals())


def test_unasked_the_result_is_the_conventional_one(tmp_path, forest):
    result, log = _run(tmp_path, forest)

    assert result["covariates"] is None
    assert result["best_model"] in ("half-normal", "hazard-rate")
    assert [m["name"] for m in result["models"]] == [
        "half-normal", "hazard-rate"]
    assert all(m["covariates"] == [] for m in result["models"])
    assert "Canopy cover" not in result["notes"]
    assert not any("canopy" in line for line in log)


def test_the_canopy_explains_how_far_animals_are_seen(tmp_path, forest):
    result, log = _run(tmp_path, forest, ds_covariates=True)

    covariates = result["covariates"]
    assert result["best_model"].endswith("+ canopy cover")
    assert covariates["in_best_model"] == [detection_function.CANOPY]
    assert covariates["considered"] == {detection_function.CANOPY: {
        "label": "canopy cover", "unit": "%", "mean": pytest.approx(
            80 * 45 / 115, abs=2.0), "min": 0.0, "max": 80.0}}
    effect = covariates["effects"][0]
    assert effect["coefficient"] < 0 and effect["p_value"] < 0.001
    # sigma 32 m in the open and 9 m under 80 % of canopy
    assert effect["coefficient"] == pytest.approx(
        math.log(9 / 32) / 80, abs=3 * effect["se"])
    assert (effect["low"], effect["high"]) == (0.0, 80.0)
    assert effect["esw_low_m"] > 2.5 * effect["esw_high_m"]
    assert any(line.startswith("  The more canopy cover, the less far "
                               "animals are seen") for line in log)
    assert "The detection function depends on canopy cover" in result["notes"]


def test_the_plain_models_are_candidates_all_the_same(tmp_path, forest):
    result, log = _run(tmp_path, forest, ds_covariates=True)

    names = [m["name"] for m in result["models"]]
    assert names == ["half-normal", "hazard-rate",
                     "half-normal + canopy cover",
                     "hazard-rate + canopy cover"]
    by_name = {m["name"]: m for m in result["models"]}
    assert by_name["half-normal + canopy cover"]["aic"] < \
        by_name["half-normal"]["aic"] - 10
    assert by_name["half-normal + canopy cover"]["covariates"] == [
        detection_function.CANOPY]
    assert "per_canopy_cover_pct" in \
        by_name["half-normal + canopy cover"]["params"]
    assert "  half-normal + canopy cover: AIC=" in "\n".join(log)


def test_a_height_that_never_changed_explains_nothing(tmp_path, forest):
    result, log = _run(tmp_path, forest, ds_covariates=True)

    assert result["covariates"]["not_used"] == {
        detection_function.HEIGHT: "the same for every observation (100 m)"}
    assert ("  flight height is not used: the same for every observation "
            "(100 m)") in log


def test_each_observation_stands_for_the_animals_its_strip_says(tmp_path,
                                                                forest):
    plain, _log = _run(tmp_path, forest)
    result, _log = _run(tmp_path, forest, ds_covariates=True)

    # 70 animals seen over a wide strip and 45 over a narrow one: the
    # density is the sum of what each stands for
    best = result["best_model"]
    assert result["n"] == plain["n"] == 115
    effect = result["covariates"]["effects"][0]
    by_hand = (70 / effect["esw_low_m"] + 45 / effect["esw_high_m"]) / (
        2 * 2000.0) * 1e6
    assert result["density_per_km2"] == pytest.approx(by_hand, rel=0.02)
    assert result["effective_strip_width_m"] == pytest.approx(
        115 / (70 / effect["esw_low_m"] + 45 / effect["esw_high_m"]),
        rel=0.02)
    assert result["detection_probability"] == pytest.approx(
        result["effective_strip_width_m"] / W)
    assert result["abundance_in_covered_area"] == pytest.approx(
        result["density_per_km2"] * result["covered_area_km2"])
    low, high = result["density_ci95"]
    assert low < result["density_per_km2"] < high
    assert result["cv_density"] == pytest.approx(math.hypot(
        result["encounter_rate_cv"], result["detection_cv"]))
    assert best != plain["best_model"]


def test_the_curve_drawn_is_the_average_over_the_observations(tmp_path,
                                                              forest):
    result, _log = _run(tmp_path, forest, ds_covariates=True)
    curve = result["detection_function_curve"]

    assert curve["g"][0] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(curve["g"], curve["g"][1:]))
    assert len(curve["x"]) == len(curve["g"]) == 60


def test_without_a_tree_cover_layer_the_canopy_is_left_out(tmp_path):
    folder = str(tmp_path / "Bare")
    dem = _flight(folder, _animals(), canopy=False)

    result, log = _run(tmp_path, ("Bare", folder, dem), ds_covariates=True)

    reason = result["covariates"]["not_used"][detection_function.CANOPY]
    assert reason == (
        "known for 0 of 115 observations only: Bare has no tree-cover layer "
        "- run 'Land Cover' of the Environmental Context")
    assert result["covariates"]["considered"] == {}
    assert result["best_model"] in ("half-normal", "hazard-rate")
    assert ("Canopy cover and flight height were asked for but could not be "
            "offered to the detection function") in result["notes"]
    assert f"  canopy cover is not used: {reason}" in log


def test_the_height_the_legs_were_flown_at_explains_too(tmp_path):
    """Open ground only: one leg at 60 m above the animals, one at 140 m -
    from where they are found only close to the line."""
    folder = str(tmp_path / "Heights")
    dem = _flight(folder, _animals(open_sigma=34.0, forest_sigma=11.0),
                  canopy=False, heights=(40.0, 120.0))

    result, log = _run(tmp_path, ("Heights", folder, dem),
                       ds_covariates=True)

    covariates = result["covariates"]
    assert covariates["in_best_model"] == [detection_function.HEIGHT]
    considered = covariates["considered"][detection_function.HEIGHT]
    assert (considered["min"], considered["max"]) == (60.0, 140.0)
    effect = covariates["effects"][0]
    assert (effect["label"], effect["unit"]) == ("flight height", "m")
    assert effect["coefficient"] < 0 and effect["p_value"] < 0.001
    assert effect["esw_low_m"] > 2 * effect["esw_high_m"]
    assert any("The more flight height, the less far animals are seen"
               in line for line in log)


def test_without_the_dem_the_height_is_not_known(tmp_path):
    folder = str(tmp_path / "NoDem")
    _flight(folder, _animals(), dem=False)

    result, _log = _run(tmp_path, ("NoDem", folder, ""), ds_covariates=True)

    assert result["covariates"]["not_used"][detection_function.HEIGHT] == (
        "known for 0 of 115 observations only: the DEM origin of NoDem is "
        "not known - set the DEM on the flight's Input tab")
    assert result["covariates"]["in_best_model"] == [
        detection_function.CANOPY]


def test_a_covariate_one_of_the_flights_cannot_tell_is_left_out(
        tmp_path, forest):
    folder = str(tmp_path / "Bare")
    dem = _flight(folder, _animals(seed=9), canopy=False)

    result, _log = _run(tmp_path, forest, ("Bare", folder, dem),
                        ds_covariates=True)

    assert result["n"] == 230
    assert result["covariates"]["not_used"][detection_function.CANOPY] == (
        "known for 115 of 230 observations only: Bare has no tree-cover "
        "layer - run 'Land Cover' of the Environmental Context")


def test_a_covariate_that_explains_nothing_is_not_used(tmp_path):
    """The same curve over open ground and under the trees: the plain model
    describes the distances as well, with a parameter less."""
    folder = str(tmp_path / "Same")
    dem = _flight(folder, _animals(open_sigma=25.0, per_leg=(60, 60),
                                   alike=True))

    result, _log = _run(tmp_path, ("Same", folder, dem), ds_covariates=True)

    by_name = {m["name"]: m for m in result["models"]}
    # the same likelihood - but for the few animals at the edge of the
    # forest, whose canopy is neither - and two for the parameter that
    # bought nothing
    assert by_name["half-normal + canopy cover"]["aic"] == pytest.approx(
        by_name["half-normal"]["aic"] + 2.0, abs=0.05)
    assert result["covariates"]["in_best_model"] == []
    assert result["covariates"]["effects"] == []
    assert list(result["covariates"]["considered"]) == [
        detection_function.CANOPY]
    assert result["best_model"] in ("half-normal", "hazard-rate")
    assert ("a model without them describes the distances as well (lower "
            "AIC): they are not used") in result["notes"]


def test_a_handful_of_animals_is_too_few_to_ask(tmp_path):
    folder = str(tmp_path / "Few")
    dem = _flight(folder, _animals(per_leg=(8, 6)))

    result, _log = _run(tmp_path, ("Few", folder, dem), ds_covariates=True)

    assert result["covariates"]["not_used"][detection_function.CANOPY] == (
        "14 observations are too few to tell what it does (at least 20)")
    assert result["covariates"]["considered"] == {}


def test_the_canopy_is_taken_around_the_animal(tmp_path):
    """An animal at the forest edge: with a wide radius half its
    surroundings are trees."""
    folder = str(tmp_path / "Edge")
    dem = _flight(folder, _animals())
    survey = Survey(folder=str(tmp_path / "survey"), modality="t", flights=[
        SurveyFlight(name="Edge", target_folder=folder, epsg=EPSG, dem=dem)])
    entries = [{"last_frame": 200, "detection_center": [
        ORIGIN[0] + LEG, ORIGIN[1] + 5.0, GROUND]}]

    def canopy(radius):
        conditions, hints = BambiProcessor()._distance_sampling_conditions(
            survey.flights[0], survey, entries, "last_frame", radius)
        assert hints == {}
        return conditions[0][detection_function.CANOPY]

    assert canopy(50.0) == pytest.approx(40.0, abs=4.0)
    assert canopy(8.0) in (0.0, 80.0) or 0.0 < canopy(8.0) < 80.0


def test_the_height_is_above_where_the_animal_stood(tmp_path):
    folder = str(tmp_path / "Slope")
    dem = _flight(folder, _animals())
    survey = Survey(folder=str(tmp_path / "survey"), modality="t", flights=[
        SurveyFlight(name="Slope", target_folder=folder, epsg=EPSG, dem=dem)])
    entries = [
        {"last_frame": 10, "detection_center": [ORIGIN[0], ORIGIN[1], 300.0]},
        {"last_frame": 10, "detection_center": [ORIGIN[0], ORIGIN[1], 350.0]},
        {"last_frame": 10, "detection_center": [ORIGIN[0], ORIGIN[1], 450.0]},
        {"last_frame": 9999, "detection_center": [ORIGIN[0], ORIGIN[1], 300]},
        {"last_frame": 10, "detection_center": [ORIGIN[0], ORIGIN[1]]},
        {"detection_center": [ORIGIN[0], ORIGIN[1], 300.0]},
    ]

    conditions, _hints = BambiProcessor()._distance_sampling_conditions(
        survey.flights[0], survey, entries, "last_frame", 30.0)

    heights = [row[detection_function.HEIGHT] for row in conditions]
    # the camera at 80 m above an origin at 320 m: 400 m above sea level.
    # Above it (450 m) nothing can have been seen from there; a frame that
    # is not there, a place without a height and no frame tell nothing
    assert heights == [100.0, 50.0, None, None, None, None]
