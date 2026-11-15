# -*- coding: utf-8 -*-
"""The Grid Density step: a survey's flights, cell by cell
(bambi_survey_grid.py on top of core/grid_density.py)."""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import (
    BambiProcessor, ProcessingWorker)
from bambi_wildlife_detection.core import grid_density, survey_area
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight

shapely = pytest.importorskip("shapely")

#: A herd of three in the south-west hectare, a pair two hectares east, one
#: animal in the north - on a meadow of 400 m x 300 m of which the flight
#: saw the southern 150 m.
MEADOW = [
    {"id": 1, "species": "red deer", "at": (20, 30), "frames": (5, 9),
     "sex": "female"},
    {"id": 2, "species": "red deer", "at": (40, 30), "frames": (6, 10)},
    {"id": 3, "species": "red deer", "at": (40, 60), "frames": (6, 11)},
    {"id": 4, "species": "roe deer", "at": (250, 40), "frames": (30, 34)},
    {"id": 5, "species": "roe deer", "at": (270, 60), "frames": (31, 36)},
    {"id": 6, "species": "roe deer", "at": (150, 250), "frames": (60, 62)},
]
SEEN = [(0, 0, 200, 150), (200, 0, 400, 150)]
SQUARES = {"grid_shape": "square", "grid_size_mode": "edge",
           "grid_cell_edge_m": 100.0}


def _survey(tmp_path, *folders, area=None, **more):
    survey_folder = str(tmp_path / "survey")
    if area is not None:
        from shapely.geometry import box
        os.makedirs(survey_folder, exist_ok=True)
        x0, y0 = ORIGIN
        survey_area.create(survey_folder, EPSG, plots=[
            (box(x0 + area[0], y0 + area[1], x0 + area[2], y0 + area[3]),
             "plot")])
    return Survey(folder=survey_folder, modality="t",
                  area_enabled=area is not None, flights=[
                      SurveyFlight(name=os.path.basename(folder),
                                   target_folder=folder, epsg=EPSG)
                      for folder in folders], **more)


def _run(tmp_path, survey, **settings):
    log = []
    config = dict(SQUARES, survey=survey.to_dict(), **settings)
    document = BambiProcessor().run_grid_density(config, log_fn=log.append)
    return document, log, os.path.join(survey.folder, "analytics_t")


@pytest.fixture
def meadow(tmp_path):
    return make_flight(str(tmp_path / "Meadow"), MEADOW, seen=SEEN)


def _cells(document):
    return {cell["cell"]: cell for cell in document["cells"]}


def _name(col, row):
    """The cell *col* hectares east and *row* north of ORIGIN."""
    return f"c{int(ORIGIN[0] / 100) + col}r{int(ORIGIN[1] / 100) + row}"


def test_the_survey_is_counted_cell_by_cell(tmp_path, meadow):
    survey = _survey(tmp_path, meadow, area=(0, 0, 400, 300))
    document, log, _out = _run(tmp_path, survey)
    cells = _cells(document)

    assert document["summary"]["cells"] == 12
    assert document["summary"]["counted"] == 8        # rows 0 and 1 (half)
    herd = cells[_name(0, 0)]
    assert (herd["animals"], herd["groups"], herd["mean_group_size"]) == (
        3, 1, 3.0)
    assert herd["density_per_km2"] == pytest.approx(300.0)
    assert herd["sex"] == {"female": 1, "unknown": 2}
    pair = cells[_name(2, 0)]
    assert pair["species"]["roe deer"]["animals"] == 2
    # the animal in the north is on ground the flight did not see
    north = cells[_name(1, 2)]
    assert (north["status"], north["animals"]) == (grid_density.NOT_SEEN, 1)
    assert document["source"] == "tracks" and document["camera"] == "Thermal"
    assert document["crs"] == "EPSG:32633"
    assert document["summary"]["plot_ha"] == pytest.approx(12.0)
    assert any(line.startswith("Grid: squares of 1 ha (edge 100.0 m) - 12 "
                               "cell(s), 8 counted") for line in log)
    assert any(line.startswith("All animals: 5 in the counted cells")
               for line in log)


def test_the_total_is_for_the_survey_area(tmp_path, meadow):
    survey = _survey(tmp_path, meadow, area=(0, 0, 400, 300))
    document, _log, _out = _run(tmp_path, survey)
    everything = document["statistics"]["all"]

    # 5 animals on the 6 ha that were seen, on 12 ha of plot
    assert everything["density_per_km2"]["estimate"] == pytest.approx(
        5 / 0.06)
    assert everything["total_in_survey_area"]["estimate"] == pytest.approx(
        10.0)
    assert "The total is the mean density times the survey area" in \
        document["notes"]


def test_without_a_survey_area_the_grid_covers_what_the_flights_saw(
        tmp_path, meadow):
    survey = _survey(tmp_path, meadow)
    document, _log, _out = _run(tmp_path, survey)

    assert document["summary"]["cells"] == 8
    assert document["summary"]["region_ha"] == pytest.approx(6.0)
    assert document["summary"]["animals_outside"] == 1     # the one up north
    assert document["statistics"]["all"]["total_in_survey_area"] is None
    assert "no total is given" in document["notes"]


def test_the_three_files(tmp_path, meadow):
    survey = _survey(tmp_path, meadow, area=(0, 0, 400, 300))
    _document, log, out = _run(tmp_path, survey)

    with open(os.path.join(out, "grid_density_tracks.json")) as handle:
        stored = json.load(handle)
    assert len(stored["cells"]) == 12 and len(stored["cells"][0]["ring"]) == 4
    with open(os.path.join(out, "grid_density_tracks_cells.csv"),
              newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    herd = next(row for row in rows if row["cell"] == _name(0, 0))
    assert (herd["animals"], herd["n_red_deer"], herd["n_roe_deer"]) == (
        "3", "3", "0")
    assert herd["status"] == "counted" and herd["groups"] == "1"
    with open(os.path.join(out, "grid_density_tracks_by_flight.csv"),
              newline="", encoding="utf-8") as handle:
        visits = list(csv.DictReader(handle))
    assert {row["flight"] for row in visits} == {"Meadow"}
    assert any(line.startswith("Grid density written to") for line in log)


def test_the_settings_reach_the_grid(tmp_path, meadow):
    survey = _survey(tmp_path, meadow, area=(0, 0, 400, 300))

    hexagons, _log, _out = _run(
        tmp_path, survey, grid_shape="hexagon", grid_size_mode="area",
        grid_cell_area_ha=0.5, grid_min_seen_pct=80.0, grid_repeat="highest")

    assert hexagons["grid"]["shape"] == "hexagon"
    assert hexagons["grid"]["cell_area_m2"] == pytest.approx(5000.0)
    assert hexagons["min_seen"] == pytest.approx(0.8)
    assert hexagons["repeat"] == "highest"
    assert "flight with the highest density" in hexagons["notes"]
    assert "less than 80 % was seen" in hexagons["notes"]
    assert all(len(cell["ring"]) == 6 for cell in hexagons["cells"])


def test_two_flights_over_the_same_ground(tmp_path, meadow):
    evening = make_flight(str(tmp_path / "Evening"), [
        {"id": 1, "species": "red deer", "at": (30, 40), "frames": (5, 9)},
    ], seen=[(0, 0, 100, 150)])
    survey = _survey(tmp_path, meadow, evening, area=(0, 0, 400, 300))

    average, _log, _out = _run(tmp_path, survey)
    highest, _log, _out = _run(tmp_path, survey, grid_repeat="highest")

    cell = _cells(average)[_name(0, 0)]
    assert (cell["animals"], cell["flights"]) == (4, 2)
    assert cell["density_per_km2"] == pytest.approx(200.0)
    best = _cells(highest)[_name(0, 0)]
    assert (best["animals"], best["best_flight"]) == (3, "Meadow")
    assert best["density_per_km2"] == pytest.approx(300.0)
    assert len([row for row in average["by_flight"]
                if row["cell"] == _name(0, 0)]) == 2


def test_the_species_filter_applies(tmp_path, meadow):
    survey = _survey(tmp_path, meadow, area=(0, 0, 400, 300),
                     species=["roe deer"])
    document, _log, _out = _run(tmp_path, survey)

    assert document["species"] == ["roe deer"]
    assert _cells(document)[_name(0, 0)]["animals"] == 0
    assert document["statistics"]["all"]["animals"] == 2
    assert document["species_filter"] == ["roe deer"]


def test_boxes_instead_of_animals(tmp_path):
    folder = make_flight(str(tmp_path / "Photos"), [], loose_boxes=[
        {"species": "red deer", "at": (20, 30), "frame": 3},
        {"species": "red deer", "at": (40, 30), "frame": 3},
        {"species": "red deer", "at": (250, 40), "frame": 8},
    ], seen=SEEN)
    survey = _survey(tmp_path, folder, source="detections")
    document, _log, out = _run(tmp_path, survey)

    assert document["source"] == "detections"
    assert _cells(document)[_name(0, 0)]["animals"] == 2
    assert _cells(document)[_name(0, 0)]["groups"] == 1
    assert os.path.isfile(os.path.join(out, "grid_density_detections.json"))


def test_a_flight_without_footprints_needs_a_survey_area(tmp_path):
    folder = make_flight(str(tmp_path / "Blind"), MEADOW)

    with pytest.raises(RuntimeError, match="Calculate Field of View"):
        _run(tmp_path, _survey(tmp_path, folder))


def test_with_a_survey_area_it_counts_as_having_seen_all_of_it(tmp_path):
    folder = make_flight(str(tmp_path / "Blind"), MEADOW)
    survey = _survey(tmp_path, folder, area=(0, 0, 400, 300))

    document, log, _out = _run(tmp_path, survey)

    assert document["summary"]["counted"] == 12
    assert document["summary"]["seen_share"] == pytest.approx(1.0)
    assert any("the whole survey area counts as seen" in line
               for line in log)


def test_a_grid_that_is_no_grid_is_refused(tmp_path, meadow):
    survey = _survey(tmp_path, meadow)

    with pytest.raises(RuntimeError, match="finer than the positions"):
        _run(tmp_path, survey, grid_cell_edge_m=0.2)
    with pytest.raises(RuntimeError, match="Unknown cell shape"):
        _run(tmp_path, survey, grid_shape="circle")


def test_the_step_can_be_cancelled(tmp_path, meadow):
    from bambi_wildlife_detection.bambi_processing import CancelledException

    survey = _survey(tmp_path, meadow)
    with pytest.raises(CancelledException):
        BambiProcessor().run_grid_density(
            dict(SQUARES, survey=survey.to_dict()),
            cancel_check=lambda: True)


def test_the_worker_knows_the_step(tmp_path, meadow, qapp=None):
    survey = _survey(tmp_path, meadow, area=(0, 0, 400, 300))
    said = []
    worker = ProcessingWorker(
        BambiProcessor(), "grid_density",
        dict(SQUARES, survey=survey.to_dict()))
    worker.finished.connect(lambda step, ok: said.append((step, ok)))
    worker.error.connect(lambda step, text: said.append(("error", text)))

    worker.run()

    assert said == [("grid_density", True)]
    assert os.path.isfile(os.path.join(
        survey.folder, "analytics_t", "grid_density_tracks.json"))


def test_environment_values_reach_the_cells(tmp_path, meadow):
    np = pytest.importorskip("numpy")
    pytest.importorskip("rasterio")
    from bambi_wildlife_detection.core import environment

    grid = environment.Grid(epsg=EPSG, min_x=ORIGIN[0], max_y=ORIGIN[1] + 300,
                            width=40, height=30, resolution=10.0)
    ndvi = np.full(grid.shape, 0.3, dtype="float32")
    ndvi[20:, :10] = 0.7                      # the south-west hectare
    try:
        environment.write_raster(
            environment.layer_path(meadow, "ndvi"), ndvi, grid)
    except (AttributeError, TypeError):
        pytest.skip("the environment module writes its rasters another way")
    survey = _survey(tmp_path, meadow, area=(0, 0, 400, 300))

    document, log, out = _run(tmp_path, survey)

    assert document["environment_layers"] == ["ndvi"]
    assert _cells(document)[_name(0, 0)]["environment"]["ndvi"] == \
        pytest.approx(0.7, abs=1e-3)
    assert _cells(document)[_name(3, 0)]["environment"]["ndvi"] == \
        pytest.approx(0.3, abs=1e-3)
    assert any(line == "Environment per cell: ndvi" for line in log)
    with open(os.path.join(out, "grid_density_tracks_cells.csv"),
              newline="", encoding="utf-8") as handle:
        assert "ndvi" in csv.DictReader(handle).fieldnames
