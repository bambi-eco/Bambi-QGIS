# -*- coding: utf-8 -*-
"""Effort and encounter rate for distance sampling.

The whole flight log used to count as surveyed line - take-off, the transit
to the first transect, the way home - which makes the line too long and the
density too low. And the encounter rate's variance was a Poisson one, which
ignores how unevenly animals are spread over the lines.
"""
import json
import math
import os

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import distance_effort as de
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight
from bambi_wildlife_detection.core.transects import Transect


def _images(points):
    return [{"imagefile": f"{i}.jpg", "location": [x, y, 80.0]}
            for i, (x, y) in enumerate(points)]


#: A flight of 1700 m: 300 m of approach, a 500 m leg east, a 100 m turn, a
#: 500 m leg back west, and 300 m home - one frame every 10 m.
APPROACH = [(-300.0 + 10.0 * i, 0.0) for i in range(30)]
LEG_A = [(10.0 * i, 0.0) for i in range(51)]              # frames 30..80
TURN = [(500.0, 10.0 * i) for i in range(1, 10)]          # frames 81..89
LEG_B = [(500.0 - 10.0 * i, 100.0) for i in range(51)]    # frames 90..140
HOME = [(-10.0 * i, 100.0) for i in range(1, 31)]         # frames 141..170
FLIGHT = APPROACH + LEG_A + TURN + LEG_B + HOME


def _transects():
    return [Transect(transect_id=1, name="A", start_frame=30, end_frame=80),
            Transect(transect_id=2, name="", start_frame=90, end_frame=140)]


class _Area:
    """A survey area: everything with x between *low* and *high*."""

    def __init__(self, low, high):
        self.low, self.high = low, high

    def clipped_length(self, coordinates):
        length = 0.0
        for (x1, y1), (x2, y2) in zip(coordinates, coordinates[1:]):
            lo, hi = sorted((x1, x2))
            if x1 == x2:
                if self.low <= x1 <= self.high:
                    length += abs(y2 - y1)
                continue
            inside = max(0.0, min(hi, self.high) - max(lo, self.low))
            length += inside / (hi - lo) * math.hypot(x2 - x1, y2 - y1)
        return length


# ---------------------------------------------------------------------------
# What counts as line
# ---------------------------------------------------------------------------

def test_the_transect_legs_are_the_lines():
    lines, kind = de.flight_lines(_images(FLIGHT), _transects())

    assert kind == de.TRANSECTS
    assert [(line["name"], line["first_frame"], line["last_frame"])
            for line in lines] == [("A", 30, 80), ("Transect 2", 90, 140)]
    assert [line["length_m"] for line in lines] == pytest.approx([500.0, 500.0])
    # ... of a flight 1700 m long
    whole, _kind = de.flight_lines(_images(FLIGHT))
    assert whole[0]["length_m"] == pytest.approx(1700.0)


def test_without_transects_the_recorded_route_is_the_line():
    lines, kind = de.flight_lines(_images(LEG_A))
    assert kind == de.RECORDED_ROUTE
    assert len(lines) == 1
    assert lines[0]["length_m"] == pytest.approx(500.0)
    assert (lines[0]["first_frame"], lines[0]["last_frame"]) == (0, 50)


def test_a_line_is_as_long_as_its_part_inside_the_survey_area():
    lines, _kind = de.flight_lines(
        _images(FLIGHT), _transects(), area=_Area(0.0, 200.0))
    assert [line["length_m"] for line in lines] == pytest.approx([200.0, 200.0])


def test_the_dem_origin_places_the_lines_in_the_survey_area():
    """The frame positions are mesh-local, the area is in the world CRS."""
    area = _Area(1000.0, 1200.0)
    here, _kind = de.flight_lines(_images(FLIGHT), _transects(), area=area)
    assert [line["length_m"] for line in here] == [0.0, 0.0]
    shifted, _kind = de.flight_lines(
        _images(FLIGHT), _transects(), area=area, offset=(1000.0, 0.0))
    assert [line["length_m"] for line in shifted] == pytest.approx(
        [200.0, 200.0])


def test_frames_without_a_position_do_not_break_a_line():
    images = _images(LEG_A)
    images[20]["location"] = None
    lines, _kind = de.flight_lines(images)
    assert lines[0]["length_m"] == pytest.approx(500.0)


def test_no_frames_no_lines():
    assert de.flight_lines([]) == ([], de.RECORDED_ROUTE)


def test_an_animal_belongs_to_the_line_its_frame_is_on():
    lines, _kind = de.flight_lines(_images(FLIGHT), _transects())
    assert de.line_of_frame(lines, 30) == 0
    assert de.line_of_frame(lines, 80) == 0
    assert de.line_of_frame(lines, 120) == 1
    assert de.line_of_frame(lines, "95") == 1


@pytest.mark.parametrize("frame", [0, 29, 85, 141, 170, None, "x"])
def test_an_animal_seen_in_transit_is_on_no_line(frame):
    lines, _kind = de.flight_lines(_images(FLIGHT), _transects())
    assert de.line_of_frame(lines, frame) is None


# ---------------------------------------------------------------------------
# How sure the encounter rate is
# ---------------------------------------------------------------------------

def test_evenly_spread_animals_give_the_poisson_figure():
    rng = np.random.default_rng(3)
    counts = rng.poisson(4.0, 400)
    cv, method = de.encounter_rate_cv(counts, [500.0] * 400)
    assert method == de.BETWEEN_LINES
    assert cv == pytest.approx(1.0 / math.sqrt(counts.sum()), rel=0.1)


def test_clumped_animals_make_the_encounter_rate_less_sure():
    """Most lines empty, a few with a herd: the same n, far less certain."""
    counts = [0] * 18 + [20, 20]
    cv, method = de.encounter_rate_cv(counts, [500.0] * 20)
    assert method == de.BETWEEN_LINES
    assert cv > 3.0 * (1.0 / math.sqrt(40))


def test_the_formula():
    """Buckland et al. 2001, eq. 3.4, by hand for three lines."""
    counts, lengths = [2.0, 5.0, 3.0], [100.0, 300.0, 200.0]
    total_n, total_length, k = 10.0, 600.0, 3
    rate = total_n / total_length
    variance = k / (total_length ** 2 * (k - 1)) * sum(
        length ** 2 * (n / length - rate) ** 2
        for n, length in zip(counts, lengths))
    cv, _method = de.encounter_rate_cv(counts, lengths)
    assert cv == pytest.approx(math.sqrt(variance) / rate)


def test_lines_that_all_have_the_same_rate_have_no_spread():
    cv, method = de.encounter_rate_cv([2, 4, 6], [100.0, 200.0, 300.0])
    assert (cv, method) == (0.0, de.BETWEEN_LINES)


def test_a_single_line_falls_back_on_poisson():
    assert de.encounter_rate_cv([25], [800.0]) == (0.2, de.POISSON)
    # a line outside the survey area is no second line
    assert de.encounter_rate_cv([25, 0], [800.0, 0.0]) == (0.2, de.POISSON)


def test_nothing_seen_has_no_figure():
    assert de.encounter_rate_cv([0, 0], [100.0, 100.0]) == (None, de.POISSON)
    assert de.encounter_rate_cv([], []) == (None, de.POISSON)


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def _flight_folder(folder, transects=True, poses=True, frames_of=None):
    """*frames_of*: ``{frame: [distances]}`` of the animals seen."""
    os.makedirs(folder, exist_ok=True)
    if poses:
        with open(os.path.join(folder, "poses_t.json"), "w") as fh:
            json.dump({"images": _images(FLIGHT)}, fh)
    if transects:
        os.makedirs(os.path.join(folder, "transects_t"))
        with open(os.path.join(folder, "transects_t", "transects.json"),
                  "w") as fh:
            json.dump({"version": 1, "modality": "t", "transects": [
                {"id": 1, "name": "A", "start_frame": 30, "end_frame": 80},
                {"id": 2, "name": "", "start_frame": 90, "end_frame": 140}]},
                fh)
    route = os.path.join(folder, "flight_route_t")
    os.makedirs(route)
    with open(os.path.join(route, "flight_route.geojson"), "w") as fh:
        json.dump({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "LineString",
                         "coordinates": [list(p) for p in FLIGHT]}}]}, fh)
    entries, tracks = [], []
    for frame, distances in (frames_of or {}).items():
        x, y = FLIGHT[frame]
        for distance in distances:
            entry = {"class_id": 1, "distance_m": float(distance),
                     "detection_center": [x, y + distance, 0.0]}
            entries.append(dict(entry, frame=frame))
            tracks.append(dict(entry, last_frame=frame,
                               track_id=len(tracks) + 1))
    with open(os.path.join(route, "perpendicular_t.json"), "w") as fh:
        json.dump({"perpendiculas": entries}, fh)
    with open(os.path.join(route, "perpendicular_tracks_t.json"), "w") as fh:
        json.dump({"tracks": tracks}, fh)
    return folder


#: Animals on both legs, and three seen on the approach and the way home.
SEEN = {35: [2.0, 14.0], 50: [5.0], 70: [9.0, 1.0], 100: [3.0], 120: [7.0],
        135: [4.0, 11.0], 10: [6.0], 20: [2.0], 160: [8.0]}


def _run(tmp_path, source="detections", **flight):
    folder = _flight_folder(str(tmp_path / "flight"), **flight)
    survey_folder = str(tmp_path / "survey")
    survey = Survey(folder=survey_folder, modality="t", flights=[
        SurveyFlight(name="flight", target_folder=folder, epsg=32633)])
    log = []
    BambiProcessor().run_distance_sampling({
        "survey": survey.to_dict(), "target_epsg": 32633,
        "ds_source": source, "ds_truncation": 20.0}, log_fn=log.append)
    with open(os.path.join(survey_folder, "analytics_t",
                           f"distance_sampling_{source}.json")) as fh:
        return json.load(fh), log


@pytest.mark.parametrize("source", ["detections", "tracks"])
def test_with_transects_only_the_legs_count(tmp_path, source):
    pytest.importorskip("scipy")
    result, log = _run(tmp_path, source, frames_of=SEEN)

    assert result["effort"] == ["transects"]
    assert result["transect_length_m"] == pytest.approx(1000.0)   # not 1700
    assert result["n"] == 9                                        # not 12
    assert result["n_off_effort"] == 3
    assert result["n_lines"] == 2
    assert [(line["name"], line["n"]) for line in result["lines"]] == [
        ("A", 5), ("Transect 2", 4)]
    assert any("3 seen outside the transect legs" in line for line in log)
    assert "in transit" in result["notes"]


def test_without_transects_the_recorded_route_counts(tmp_path):
    pytest.importorskip("scipy")
    result, _log = _run(tmp_path, transects=False, frames_of=SEEN)

    assert result["effort"] == ["recorded route"]
    assert result["transect_length_m"] == pytest.approx(1700.0)
    assert result["n"] == 12
    assert result["n_off_effort"] == 0
    assert result["encounter_rate_variance"] == "poisson"
    assert result["encounter_rate_cv"] == pytest.approx(1 / math.sqrt(12))
    assert "single line" in result["notes"]


def test_the_density_is_no_longer_diluted_by_the_transit(tmp_path):
    """Same animals on the legs: counting 1700 m of line instead of 1000 m
    gave 59 % of the density."""
    pytest.importorskip("scipy")
    on_legs = {frame: d for frame, d in SEEN.items() if 30 <= frame <= 140}
    with_legs, _log = _run(tmp_path / "a", frames_of=on_legs)
    whole, _log = _run(tmp_path / "b", transects=False, frames_of=on_legs)

    assert with_legs["n"] == whole["n"] == 9
    assert with_legs["density_per_km2"] == pytest.approx(
        whole["density_per_km2"] * 1700.0 / 1000.0, rel=1e-6)


def test_the_variance_is_measured_between_the_lines(tmp_path):
    pytest.importorskip("scipy")
    result, _log = _run(tmp_path, frames_of=SEEN)

    assert result["encounter_rate_variance"] == "between lines"
    expected, _method = de.encounter_rate_cv([5, 4], [500.0, 500.0])
    assert result["encounter_rate_cv"] == pytest.approx(expected)
    assert "between the 2 lines" in result["notes"]


def test_without_frame_positions_the_flight_route_is_all_there_is(tmp_path):
    pytest.importorskip("scipy")
    result, log = _run(tmp_path, poses=False, frames_of=SEEN)

    assert result["effort"] == ["flight route"]
    assert result["transect_length_m"] == pytest.approx(1700.0)
    assert result["n"] == 12
    assert any("take-off and transit included" in line for line in log)


def test_several_flights_are_several_lines(tmp_path):
    """Without transects each flight is one line - and two lines are enough
    to measure the variance between them."""
    pytest.importorskip("scipy")
    one = _flight_folder(str(tmp_path / "one"), transects=False,
                         frames_of={40: [2.0, 5.0, 9.0], 100: [4.0]})
    two = _flight_folder(str(tmp_path / "two"), transects=False,
                         frames_of={60: [3.0]})
    survey_folder = str(tmp_path / "survey")
    survey = Survey(folder=survey_folder, modality="t", flights=[
        SurveyFlight(name="one", target_folder=one, epsg=32633),
        SurveyFlight(name="two", target_folder=two, epsg=32633)])
    BambiProcessor().run_distance_sampling({
        "survey": survey.to_dict(), "target_epsg": 32633,
        "ds_source": "detections", "ds_truncation": 20.0})
    with open(os.path.join(survey_folder, "analytics_t",
                           "distance_sampling_detections.json")) as fh:
        result = json.load(fh)

    assert result["n_lines"] == 2
    assert result["encounter_rate_variance"] == "between lines"
    assert [(line["flight"], line["n"]) for line in result["lines"]] == [
        ("one", 4), ("two", 1)]
    assert result["transect_length_m"] == pytest.approx(2 * 1700.0)
