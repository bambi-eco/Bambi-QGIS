# -*- coding: utf-8 -*-
"""Distance sampling with canopy cover and flight height: the tick box on
the Survey Analytics tab and what the result window says
(dock/analytics_tab.py). The models themselves are covered without QGIS in
tests/test_detection_function.py and tests/test_distance_covariates.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
_RESULT = {
    "source": "tracks", "n": 115, "n_before_truncation": 115,
    "transect_length_m": 2000.0, "n_lines": 2, "truncation_m": 60.0,
    "best_model": "half-normal + canopy cover",
    "effective_strip_width_m": 21.3, "detection_probability": 0.355,
    "density_per_km2": 1349.0, "density_ci95": [900.0, 2020.0],
    "abundance_in_covered_area": 323.8, "abundance_ci95": [216.0, 485.0],
    "covered_area_km2": 0.24, "notes": "",
    "models": [
        {"name": "half-normal", "esw_m": 27.9, "aic": 915.4},
        {"name": "half-normal + canopy cover", "esw_m": 21.3, "aic": 868.1,
         "covariates": ["canopy_cover_pct"]}],
}
_EFFECT = {
    "name": "canopy_cover_pct", "label": "canopy cover", "unit": "%",
    "coefficient": -0.0159, "se": 0.0021, "z": -7.6, "p_value": 2e-14,
    "low": 0.0, "high": 80.0, "esw_low_m": 38.2, "esw_high_m": 11.3}


def test_the_group_has_the_tick_box_and_it_is_off(dock):
    assert dock.ds_covariates_check.text() == (
        "Let canopy cover and flight height explain detectability")
    assert not dock.ds_covariates_check.isChecked()
    assert "'Land Cover'" in dock.ds_covariates_check.toolTip()
    assert dock.get_config()["ds_covariates"] is False


def test_ticked_the_step_is_told(dock):
    dock.ds_covariates_check.setChecked(True)

    config = dock.get_config()

    assert config["ds_covariates"] is True
    assert config["environment_radius_m"] > 0     # what "around" means


def test_the_result_says_what_explains_detectability(dock):
    result = dict(_RESULT, covariates={
        "considered": {"canopy_cover_pct": {}},
        "not_used": {"flight_height_m":
                     "the same for every observation (100 m)"},
        "in_best_model": ["canopy_cover_pct"], "effects": [_EFFECT]})

    html = dock._distance_sampling_html(result, "result.json")

    assert "<h4>What explains detectability</h4>" in html
    assert ("<li>The more canopy cover, the less far animals are seen: the "
            "effective strip half-width is 38.2 m at 0 % and 11.3 m at "
            "80 % (p &lt; 0.001).</li>") in html
    assert ("<li>Flight height is not used: the same for every observation "
            "(100 m).</li>") in html
    assert "<td>half-normal + canopy cover</td>" in html
    assert html.index("What explains detectability") < html.index(
        "Model comparison")


def test_what_a_flight_is_called_cannot_break_the_page(dock):
    result = dict(_RESULT, covariates={
        "considered": {}, "in_best_model": [], "effects": [],
        "not_used": {"canopy_cover_pct":
                     "known for 0 of 9 observations only: <b>Wood & "
                     "Meadow</b> has no tree-cover layer"}})

    html = dock._distance_sampling_html(result, "result.json")

    assert "&lt;b&gt;Wood &amp; Meadow&lt;/b&gt; has no tree-cover" in html


def test_a_condition_that_explained_nothing_is_said_to(dock):
    result = dict(_RESULT, best_model="half-normal", covariates={
        "considered": {"canopy_cover_pct": {}, "flight_height_m": {}},
        "not_used": {}, "in_best_model": ["flight_height_m"],
        "effects": [dict(_EFFECT, name="flight_height_m",
                         label="flight height", unit="m", p_value=0.4)]})

    html = dock._distance_sampling_html(result, "result.json")

    assert "<li>Flight height: no clear effect - " in html
    assert ("<li>Canopy cover: offered to the model, and the distances are "
            "described as well without it.</li>") in html


def test_unasked_the_window_reads_as_it_always_did(dock):
    for covariates in (None, {}):
        html = dock._distance_sampling_html(
            dict(_RESULT, covariates=covariates), "result.json")
        assert "What explains detectability" not in html
    assert "What explains detectability" not in \
        dock._distance_sampling_html(_RESULT, "result.json")


def test_asked_with_nothing_to_say_there_is_no_empty_heading(dock):
    html = dock._distance_sampling_html(dict(_RESULT, covariates={
        "considered": {}, "not_used": {}, "in_best_model": [],
        "effects": []}), "result.json")

    assert "What explains detectability" not in html
