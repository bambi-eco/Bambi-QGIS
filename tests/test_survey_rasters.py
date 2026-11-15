# -*- coding: utf-8 -*-
"""A value, a zero, or nothing (core.survey_rasters) - and the density and
coverage steps that write their rasters with it.

Both maps wrote their zeros as "no data": ground that had been surveyed and
was empty looked like ground that was never surveyed, and every statistic
taken from the raster left it out.
"""
import json
import os

import pytest

np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")

from rasterio.transform import Affine, from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from bambi_wildlife_detection.bambi_processing import BambiProcessor  # noqa: E402
from bambi_wildlife_detection.core import (  # noqa: E402
    fov_store, survey_area, survey_rasters)
from bambi_wildlife_detection.core.survey import (  # noqa: E402
    Survey, SurveyFlight)

from .test_analytics_source import survey as source_survey  # noqa: E402,F401

EPSG = 32633
NODATA = survey_rasters.DENSITY_NODATA
GAP = survey_rasters.COVERAGE_NODATA


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

def test_surveyed_and_empty_is_a_zero_density():
    density = np.array([[0.0, 2.0], [0.0, 0.0]])
    signal = density > 0
    looked_at = np.array([[True, True], [False, False]])

    out = survey_rasters.density_raster(density, signal, looked_at)

    assert out.dtype == np.float32
    assert out.tolist() == [[0.0, 2.0], [NODATA, NODATA]]


def test_a_kernel_tail_outside_the_surveyed_ground_is_not_shown():
    density = np.array([[0.5, 2.0]])
    looked_at = np.array([[False, True]])
    out = survey_rasters.density_raster(density, density > 0, looked_at)
    assert out.tolist() == [[NODATA, 2.0]]


def test_without_knowing_the_ground_only_the_signal_is_shown():
    """"No animal" cannot be told from "not looked at" - so it is not
    claimed."""
    density = np.array([[0.0, 2.0], [0.0, 0.0]])
    out = survey_rasters.density_raster(density, density > 0)
    assert out.tolist() == [[NODATA, 2.0], [NODATA, NODATA]]


def test_a_gap_in_the_plot_is_a_zero_coverage():
    count = np.array([[0, 3], [1, 0]])
    inside = np.array([[True, True], [True, False]])

    out = survey_rasters.coverage_raster(count, inside)

    assert out.dtype == np.uint16
    assert out.tolist() == [[0, 3], [1, GAP]]


def test_frames_outside_the_plot_do_not_count():
    count = np.array([[4, 4]])
    inside = np.array([[True, False]])
    assert survey_rasters.coverage_raster(count, inside).tolist() == \
        [[4, GAP]]


def test_without_a_plot_unseen_ground_is_no_data():
    count = np.array([[0, 3]])
    assert survey_rasters.coverage_raster(count).tolist() == [[GAP, 3]]


def test_exclusion_zones_without_a_plot_are_no_data():
    count = np.array([[2, 3, 0]])
    excluded = np.array([[True, False, False]])
    assert survey_rasters.coverage_raster(
        count, excluded=excluded).tolist() == [[GAP, 3, GAP]]


def test_a_count_cannot_run_into_the_no_data_value():
    count = np.array([[70000.0, 5.0]])
    out = survey_rasters.coverage_raster(count, np.array([[True, True]]))
    assert out.tolist() == [[GAP - 1, 5]]


def test_the_bounds_are_the_grids_own():
    """1003 m at 5 m is 201 cells = 1005 m: written over 1003 m, every cell
    was a little narrower than the one the animals were counted into."""
    assert survey_rasters.snap_bounds(100.0, 900.0, 201, 10, 5.0) == \
        (100.0, 850.0, 1105.0, 900.0)


# -- growing a raster to a plot ------------------------------------------------

def _grid():
    """3 x 2 cells of 10 m with their top left corner at (100, 200)."""
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    return array, from_origin(100.0, 200.0, 10.0, 10.0)


def test_a_raster_that_covers_the_bounds_is_left_alone():
    array, transform = _grid()
    grown, moved = survey_rasters.grow_to_bounds(
        array, transform, (100.0, 180.0, 130.0, 200.0))
    assert grown is array and moved == transform


def test_a_raster_grows_by_whole_cells_on_every_side():
    array, transform = _grid()

    grown, moved = survey_rasters.grow_to_bounds(
        array, transform, (85.0, 155.0, 131.0, 212.0))

    # 2 cells west, 1 east, 2 north, 3 south
    assert grown.shape == (2 + 2 + 3, 3 + 2 + 1)
    assert isinstance(moved, Affine)
    assert (moved.c, moved.f) == (80.0, 220.0)
    assert (moved.a, moved.e) == (10.0, -10.0)
    assert grown[2:4, 2:5].tolist() == array.tolist()
    assert grown.sum() == array.sum()              # the rest is zeros


def test_a_cell_keeps_its_place_on_the_ground():
    array, transform = _grid()
    grown, moved = survey_rasters.grow_to_bounds(
        array, transform, (85.0, 155.0, 131.0, 212.0))
    # the cell at row 1, column 2 of the original: its centre
    x, y = rasterio.transform.xy(transform, 1, 2)
    row, column = rasterio.transform.rowcol(moved, x, y)
    assert grown[row, column] == array[1, 2]


# -- the ground that was looked at ---------------------------------------------

def test_the_surveyed_ground_is_every_flights():
    union = survey_rasters.looked_at_geometry(
        [box(0, 0, 10, 10), box(20, 0, 30, 10)])
    assert union.area == pytest.approx(200.0)


def test_one_flight_without_known_ground_makes_it_unknown():
    assert survey_rasters.looked_at_geometry(
        [box(0, 0, 10, 10), None]) is None
    assert survey_rasters.looked_at_geometry([]) is None


# ---------------------------------------------------------------------------
# The density heatmap
# ---------------------------------------------------------------------------

def _survey(folder, *flights, area_enabled=True):
    return Survey(folder=folder, modality="t", area_enabled=area_enabled,
                  flights=[SurveyFlight(name=os.path.basename(f),
                                        target_folder=f, epsg=EPSG)
                           for f in flights])


def _density(survey_folder, flight, **options):
    config = {"survey": _survey(survey_folder, flight,
                                **options).to_dict(),
              "target_epsg": EPSG, "density_source": "detections",
              "density_cell_size": 1.0, "density_bandwidth": 2.0}
    log = []
    BambiProcessor().run_density_heatmap(config, log_fn=log.append)
    folder = os.path.join(survey_folder, "analytics_t")
    with open(os.path.join(folder, "density_detections.json"),
              encoding="utf-8") as handle:
        stats = json.load(handle)
    with rasterio.open(os.path.join(folder, "density_detections.tif")) as src:
        return src.read(1), src.transform, src.bounds, stats, log


def test_the_empty_part_of_a_survey_area_is_zero_not_blank(
        source_survey, tmp_path):  # noqa: F811
    """The animals are in the south-west corner of a 100 m plot: the rest of
    the plot was surveyed too, and holds no animal."""
    survey_folder = str(tmp_path / "survey")
    survey_area.create(survey_folder, EPSG, plots=[(box(0, 0, 100, 100), "")])

    data, transform, bounds, stats, _log = _density(survey_folder,
                                                    source_survey)

    # far from every animal
    row, column = rasterio.transform.rowcol(transform, 90.5, 90.5)
    assert data[row, column] == 0.0
    inside = data != NODATA
    assert int(inside.sum()) == 100 * 100            # the plot, cell for cell
    assert int((data[inside] == 0.0).sum()) > 5000   # most of it is empty
    assert bounds.left <= 0.0 and bounds.right >= 100.0
    assert bounds.bottom <= 0.0 and bounds.top >= 100.0


def test_the_mean_is_the_mean_of_the_survey_area(source_survey, tmp_path):  # noqa: F811
    """It was the mean of the cells with animals - how dense they stood,
    not how many there are per hectare of the area."""
    survey_folder = str(tmp_path / "survey")
    survey_area.create(survey_folder, EPSG, plots=[(box(0, 0, 100, 100), "")])

    data, _transform, _bounds, stats, _log = _density(survey_folder,
                                                      source_survey)

    assert stats["mean_density_over"] == "surveyed_ground"
    assert stats["surveyed_area_ha"] == pytest.approx(1.0)
    # 3 boxes on one hectare, less what the kernel spreads over the edge
    assert stats["points_per_ha_in_survey_area"] == pytest.approx(3.0)
    assert 2.0 < stats["mean_density"] <= 3.0
    # ... and what the raster itself says is the same
    assert float(data[data != NODATA].mean()) == pytest.approx(
        stats["mean_density"], rel=1e-4)


def test_outside_the_survey_area_stays_blank(source_survey, tmp_path):  # noqa: F811
    survey_folder = str(tmp_path / "survey")
    survey_area.create(survey_folder, EPSG, plots=[(box(0, 0, 15, 30), "")])

    data, transform, _bounds, _stats, _log = _density(survey_folder,
                                                      source_survey)

    rows, columns = np.nonzero(data != NODATA)
    xs, ys = rasterio.transform.xy(transform, rows, columns)
    assert 0.0 <= min(xs) and max(xs) <= 15.0
    assert 0.0 <= min(ys) and max(ys) <= 30.0


def test_the_cameras_footprints_say_where_was_looked(source_survey,  # noqa: F811
                                                     tmp_path):
    """No survey area, but the field of view was calculated."""
    fov_store.record_fov(source_survey, "t", {
        0: [(-10.0, -10.0, 0.0), (60.0, -10.0, 0.0), (60.0, 60.0, 0.0),
            (-10.0, 60.0, 0.0)]})
    survey_folder = str(tmp_path / "survey")

    data, transform, _bounds, stats, _log = _density(
        survey_folder, source_survey, area_enabled=False)

    # seen, and nothing there
    row, column = rasterio.transform.rowcol(transform, 50.5, 5.5)
    assert data[row, column] == 0.0
    assert stats["mean_density_over"] == "surveyed_ground"
    assert stats["surveyed_area_ha"] == pytest.approx(0.49, abs=0.01)


def test_without_either_only_the_animals_are_shown(source_survey, tmp_path):  # noqa: F811
    survey_folder = str(tmp_path / "survey")

    data, _transform, _bounds, stats, log = _density(
        survey_folder, source_survey, area_enabled=False)

    assert stats["mean_density_over"] == "cells_with_animals"
    assert "surveyed_area_ha" not in stats
    assert not (data == 0.0).any()
    assert any("not looked at" in line for line in log)


def test_the_raster_has_the_cells_the_animals_were_counted_into(
        source_survey, tmp_path):  # noqa: F811
    survey_folder = str(tmp_path / "survey")
    config = {"survey": _survey(survey_folder, source_survey,
                                area_enabled=False).to_dict(),
              "target_epsg": EPSG, "density_source": "detections",
              "density_cell_size": 0.7, "density_bandwidth": 2.0}
    BambiProcessor().run_density_heatmap(config)
    with rasterio.open(os.path.join(survey_folder, "analytics_t",
                                    "density_detections.tif")) as src:
        assert src.transform.a == pytest.approx(0.7)
        assert -src.transform.e == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# The coverage map
# ---------------------------------------------------------------------------

def _frame(folder, index, left, top, size=10):
    """A frame GeoTIFF of *size* m (1 m cells) with its top left corner at
    (*left*, *top*)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{index:08d}.tiff")
    with rasterio.open(
            path, "w", driver="GTiff", height=size, width=size, count=1,
            dtype="uint8", nodata=0, crs=f"EPSG:{EPSG}",
            transform=from_origin(left, top, 1.0, 1.0)) as dst:
        dst.write(np.full((size, size), 200, dtype=np.uint8), 1)
    return path


@pytest.fixture
def flown(tmp_path):
    """A flight whose two frames cover x 0..15, y 0..10 - 5 m of it twice."""
    flight = str(tmp_path / "flight")
    _frame(os.path.join(flight, "geotiffs_t"), 0, 0.0, 10.0)
    _frame(os.path.join(flight, "geotiffs_t"), 1, 5.0, 10.0)
    return flight


def _coverage(survey_folder, flight, **options):
    config = {"survey": _survey(survey_folder, flight, **options).to_dict(),
              "target_epsg": EPSG, "coverage_cell_size": 1.0}
    BambiProcessor().run_coverage_map(config)
    folder = os.path.join(survey_folder, "analytics_t")
    with open(os.path.join(folder, "coverage_map.json"),
              encoding="utf-8") as handle:
        stats = json.load(handle)
    with rasterio.open(os.path.join(folder, "coverage_map.tif")) as src:
        return src.read(1), src.transform, src.nodata, src.bounds, stats


def _at(data, transform, x, y):
    row, column = rasterio.transform.rowcol(transform, x, y)
    return int(data[row, column])


def test_a_gap_in_the_survey_area_is_on_the_map(flown, tmp_path):
    """The plot is 30 m wide and the frames reach 15 m: the other half was
    not flown. It was not on the map at all."""
    survey_folder = str(tmp_path / "survey")
    survey_area.create(survey_folder, EPSG, plots=[(box(0, 0, 30, 10), "")])

    data, transform, nodata, bounds, stats = _coverage(survey_folder, flown)

    assert nodata == GAP
    assert (bounds.left, bounds.right) == (0.0, 30.0)
    assert _at(data, transform, 2.5, 5.5) == 1          # one frame
    assert _at(data, transform, 7.5, 5.5) == 2          # both
    assert _at(data, transform, 20.5, 5.5) == 0         # the gap: a value
    assert stats["gap_area_ha"] == pytest.approx(150 / 10000.0)
    assert stats["covered_area_ha"] == pytest.approx(150 / 10000.0)
    assert stats["survey_area_coverage_fraction"] == pytest.approx(0.5)


def test_the_gap_counts_in_what_the_raster_says(flown, tmp_path):
    """Mean coverage of the plot, read off the raster as any GIS would: the
    gap is half of it."""
    survey_folder = str(tmp_path / "survey")
    survey_area.create(survey_folder, EPSG, plots=[(box(0, 0, 30, 10), "")])

    data, _transform, nodata, _bounds, stats = _coverage(survey_folder, flown)

    values = data[data != nodata]
    assert values.size == 300
    assert float(values.mean()) == pytest.approx((100 * 1 + 50 * 2) / 300)
    # the step's own figure stays "over the covered ground"
    assert stats["mean_overlap"] == pytest.approx(200 / 150)


def test_frames_outside_the_survey_area_are_blank(flown, tmp_path):
    survey_folder = str(tmp_path / "survey")
    survey_area.create(survey_folder, EPSG, plots=[(box(0, 0, 8, 10), "")])

    data, transform, nodata, _bounds, stats = _coverage(survey_folder, flown)

    assert _at(data, transform, 12.5, 5.5) == nodata
    assert _at(data, transform, 2.5, 5.5) == 1
    assert stats["gap_area_ha"] == 0.0


def test_without_a_survey_area_unflown_ground_is_blank(flown, tmp_path):
    """Nothing says it was to be surveyed."""
    _frame(os.path.join(flown, "geotiffs_t"), 2, 40.0, 10.0)
    survey_folder = str(tmp_path / "survey")

    data, transform, nodata, _bounds, stats = _coverage(
        survey_folder, flown, area_enabled=False)

    assert nodata == GAP
    assert _at(data, transform, 30.5, 5.5) == nodata    # between the frames
    assert not (data == 0).any()
    assert "gap_area_ha" not in stats
    assert stats["covered_area_ha"] == pytest.approx(250 / 10000.0)


# ---------------------------------------------------------------------------
# How the coverage map is coloured
# ---------------------------------------------------------------------------

def test_a_gap_has_a_colour_of_its_own():
    from bambi_wildlife_detection.core import raster_styles

    classes = raster_styles.coverage_classes(0, 3)

    assert [(bound, label) for bound, _colour, label in classes] == \
        [(0.0, "not seen"), (1.0, "1"), (2.0, "2"), (3.0, "3")]
    assert classes[0][1] == raster_styles.GAP_COLOUR
    assert classes[0][1][:3] not in [colour[:3] for _b, colour, _l
                                     in classes[1:]]


def test_a_map_without_gaps_has_no_gap_class():
    from bambi_wildlife_detection.core import raster_styles

    labels = [label for _b, _c, label
              in raster_styles.coverage_classes(1, 5)]
    assert labels == ["1", "2", "3", "4", "5"]


def test_many_frames_are_grouped_and_end_at_the_highest_count():
    from bambi_wildlife_detection.core import raster_styles

    classes = raster_styles.coverage_classes(0, 40)
    assert [label for _b, _c, label in classes][-3:] == \
        ["9-16", "17-32", "33-40"]
    bounds = [bound for bound, _c, _l in classes]
    assert bounds == sorted(bounds) and bounds[-1] == 40.0


def test_more_frames_are_darker():
    from bambi_wildlife_detection.core import raster_styles

    colours = [colour for _b, colour, label
               in raster_styles.coverage_classes(1, 8)]
    brightness = [sum(colour[:3]) for colour in colours]
    assert brightness == sorted(brightness, reverse=True)
