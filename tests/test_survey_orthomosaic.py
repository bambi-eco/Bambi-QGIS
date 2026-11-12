# -*- coding: utf-8 -*-
"""The survey orthomosaic: the flight's orthomosaic, over several flights
(9.0.0)."""

import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

EPSG = 32633


def _geotiff(folder, frame, x0, y0, value, size=20):
    """A *size* m square of 1 m pixels with its upper-left at (x0, y0)."""
    os.makedirs(folder, exist_ok=True)
    data = np.full((3, size, size), value, np.uint8)
    with rasterio.open(
            os.path.join(folder, f"{frame:08d}.tiff"), "w", driver="GTiff",
            height=size, width=size, count=3, dtype="uint8", nodata=0,
            crs=f"EPSG:{EPSG}", transform=from_origin(x0, y0, 1.0, 1.0)) as dst:
        dst.write(data)


@pytest.fixture
def flights(tmp_path):
    """Two flights side by side, overlapping by 10 m."""
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _geotiff(os.path.join(a, "geotiffs_t"), 0, 500000, 5300020, 50)
    _geotiff(os.path.join(a, "geotiffs_t"), 1, 500010, 5300020, 60)
    _geotiff(os.path.join(b, "geotiffs_t"), 0, 500020, 5300020, 200)
    return a, b


def _survey(folders, folder):
    return Survey(
        folder=folder, modality="t",
        flights=[SurveyFlight(name=os.path.basename(f), target_folder=f,
                              epsg=EPSG) for f in folders])


def _run(survey, **config):
    logs = []
    BambiProcessor().run_survey_orthomosaic(
        dict({"survey": survey.to_dict(), "target_epsg": EPSG,
              "ortho_method": "first", "ortho_nodata": 0}, **config),
        log_fn=logs.append)
    path = os.path.join(survey.analytics_folder, "orthomosaic.tif")
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs, logs


def test_the_flights_end_up_in_one_mosaic(flights, tmp_path):
    a, b = flights
    data, transform, crs, logs = _run(_survey([a, b], str(tmp_path / "s")))

    assert data.shape == (3, 20, 40)          # x 500000..500040
    assert (transform.c, transform.f) == (500000.0, 5300020.0)
    assert crs.to_epsg() == EPSG
    assert data[0, 0, 0] == 50                # flight a, frame 0
    assert data[0, 0, 15] == 50               # a's frames overlap: first wins
    assert data[0, 0, 25] == 60               # a's frame 1 before flight b
    assert data[0, 0, 35] == 200              # only flight b
    assert any("a: 2 GeoTIFF(s)" in line for line in logs)


def test_the_merge_mode_and_flight_order_are_those_of_the_orthomosaic(
        flights, tmp_path):
    a, b = flights
    last, *_ = _run(_survey([a, b], str(tmp_path / "s")), ortho_method="last")
    assert last[0, 0, 25] == 200              # flight b is merged last
    swapped, *_ = _run(_survey([b, a], str(tmp_path / "s2")))
    assert swapped[0, 0, 25] == 200           # ... or first, when listed first
    mean, *_ = _run(_survey([a, b], str(tmp_path / "s3")),
                    ortho_method="average")
    assert mean[0, 0, 25] == 130              # (60 + 200) / 2


def test_one_flight_gives_exactly_its_own_orthomosaic(flights, tmp_path):
    a, _b = flights
    survey_data, survey_transform, *_ = _run(_survey([a], str(tmp_path / "s")))

    BambiProcessor().run_orthomosaic({
        "target_folder": a, "ortho_camera": "T", "ortho_method": "first",
        "ortho_nodata": 0, "target_epsg": EPSG})
    with rasterio.open(
            os.path.join(a, "orthomosaic_t", "orthomosaic.tif")) as src:
        assert np.array_equal(src.read(), survey_data)
        assert src.transform == survey_transform


def test_what_went_in_is_recorded(flights, tmp_path):
    a, b = flights
    survey = _survey([a, b], str(tmp_path / "s"))
    _run(survey)
    with open(os.path.join(survey.analytics_folder, "orthomosaic.json"),
              encoding="utf-8") as fh:
        info = json.load(fh)
    assert info["flights"] == ["a", "b"]
    assert info["frames_per_flight"] == {"a": 2, "b": 1}
    assert (info["n_frames"], info["merge_mode"]) == (3, "first")
    assert (info["width"], info["height"]) == (40, 20)


def test_a_flight_without_geotiffs_is_named(flights, tmp_path):
    a, _b = flights
    empty = str(tmp_path / "c")
    os.makedirs(empty)
    with pytest.raises(FileNotFoundError, match="c: no frame GeoTIFFs"):
        _run(_survey([a, empty], str(tmp_path / "s")))
