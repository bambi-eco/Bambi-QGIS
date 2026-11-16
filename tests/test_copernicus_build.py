# -*- coding: utf-8 -*-
"""The elevation model of a flight made from the Copernicus DEM
(copernicus_dem_downloader.py): the GeoTIFF, the mesh and the metadata -
with the heights handed in, so that no network is needed."""
import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection import copernicus_dem_downloader as cdd
from bambi_wildlife_detection.core import copernicus_dem, environment

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

#: A flight near Hagenberg, Upper Austria: UTM zone 33 north.
FLIGHT = (14.508, 48.365, 14.518, 48.372)
SOURCE = {"product": "Copernicus DEM GLO-30", "editions": ["GLO-30"],
          "tiles": ["Copernicus_DSM_COG_10_N48_00_E014_00_DEM"],
          "attribution": copernicus_dem.ATTRIBUTION,
          "vertical_datum": "EGM2008", "surface_model": True,
          "sea_share": 0.0, "min_m": 400.0, "max_m": 500.0}


def slope(grid):
    """Heights rising to the east: 400 m at the western edge, a metre per
    cell."""
    return np.tile(400.0 + np.arange(grid.width, dtype=np.float32),
                   (grid.height, 1))


def heights_of(make=slope, source=None, seen=None):
    def fetch(grid, log=None, cancelled=None, progress=None):
        if seen is not None:
            seen.append(grid)
        if progress:
            progress(90)
        return make(grid).astype(np.float32), dict(source or SOURCE)
    return fetch


def test_the_three_files_are_written(tmp_path):
    grids, log = [], []

    built = cdd.build_dem(FLIGHT, str(tmp_path / "out"), "EPSG:32633",
                          log=log.append, fetch=heights_of(seen=grids))

    assert sorted(os.listdir(tmp_path / "out")) == [
        "copernicus_dem.glb", "copernicus_dem.json", "copernicus_dem.tif"]
    assert built["mesh"].endswith("copernicus_dem.glb")
    assert built["metadata"].endswith("copernicus_dem.json")
    assert built["grid"] == grids[0]
    assert log[0].startswith("Elevation for 1.")
    assert log[0].endswith("km around the flight, in EPSG:32633")


def test_the_grid_is_the_flight_and_the_ground_around_it(tmp_path):
    from pyproj import Transformer

    grids = []
    cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                  fetch=heights_of(seen=grids))

    grid = grids[0]
    to_utm = Transformer.from_crs(4326, 32633, always_xy=True)
    x0, y0 = to_utm.transform(FLIGHT[0], FLIGHT[1])
    x1, y1 = to_utm.transform(FLIGHT[2], FLIGHT[3])
    west, south, east, north = grid.bounds
    assert (grid.epsg, grid.resolution) == (32633, 30.0)
    for edge, flight in ((x0 - west, 150), (east - x1, 150),
                         (y0 - south, 150), (north - y1, 150)):
        assert flight <= edge < flight + 45       # 150 m, on whole cells


def test_more_ground_can_be_asked_for(tmp_path):
    grids = []
    for padding in (150.0, 600.0):
        cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                      padding_m=padding, fetch=heights_of(seen=grids))

    assert grids[1].width == grids[0].width + 30      # 450 m either side


def test_the_geotiff_holds_the_heights(tmp_path):
    import rasterio

    built = cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                          fetch=heights_of())

    grid = built["grid"]
    with rasterio.open(built["geotiff"]) as src:
        assert (src.width, src.height) == (grid.width, grid.height)
        assert src.crs.to_epsg() == 32633
        assert (src.transform.c, src.transform.f) == (grid.min_x, grid.max_y)
        assert np.array_equal(src.read(1), slope(grid))


def test_the_metadata_says_where_the_heights_came_from(tmp_path):
    built = cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                          fetch=heights_of())

    with open(built["metadata"], encoding="utf-8") as handle:
        document = json.load(handle)
    grid = built["grid"]
    assert document["crs"] == "EPSG:32633"
    assert document["origin"][:2] == [grid.min_x, grid.bounds[1]]
    assert document["origin"][2] == 400.0             # the lowest height
    assert document["heights_at"] == "cell centres"
    assert document["source"] == {
        "product": "Copernicus DEM GLO-30", "editions": ["GLO-30"],
        "tiles": ["Copernicus_DSM_COG_10_N48_00_E014_00_DEM"],
        "attribution": copernicus_dem.ATTRIBUTION,
        "vertical_datum": "EGM2008", "surface_model": True,
        "sea_share": 0.0, "resolution_m": 30.0}


def test_the_mesh_has_the_heights_where_they_are(tmp_path):
    trimesh = pytest.importorskip("trimesh")

    built = cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                          fetch=heights_of())

    grid = built["grid"]
    with open(built["metadata"], encoding="utf-8") as handle:
        origin = np.array(json.load(handle)["origin"])
    mesh = trimesh.load(built["mesh"], force="mesh", process=False)
    world = np.asarray(mesh.vertices, dtype=np.float64) + origin
    assert len(world) == grid.width * grid.height
    # a metre per cell of 30 m, from 400 m at the middle of the first cell
    expected = 400.0 + (world[:, 0] - grid.min_x - 15.0) / 30.0
    assert np.abs(world[:, 2] - expected).max() < 0.01


def test_the_sea_is_ground_not_a_hole(tmp_path):
    """The mesh generator takes exactly nought for "no data": a flight
    along a shore would get a model with the sea cut out of it."""
    trimesh = pytest.importorskip("trimesh")
    import rasterio

    def shore(grid):
        heights = slope(grid)
        heights[:, :10] = 0.0
        return heights

    built = cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                          fetch=heights_of(shore))

    grid = built["grid"]
    mesh = trimesh.load(built["mesh"], force="mesh", process=False)
    assert len(mesh.vertices) == grid.width * grid.height
    with rasterio.open(built["geotiff"]) as src:
        sea = src.read(1)[:, :10]
    assert np.allclose(sea, cdd.SEA_LEVEL_M) and cdd.SEA_LEVEL_M <= 0.01


def test_the_southern_hemisphere(tmp_path):
    grids = []

    built = cdd.build_dem((18.47, -34.36, 18.48, -34.35), str(tmp_path),
                          "EPSG:32734", fetch=heights_of(seen=grids))

    assert grids[0].epsg == 32734
    with open(built["metadata"], encoding="utf-8") as handle:
        document = json.load(handle)
    assert document["crs"] == "EPSG:32734"
    assert document["origin_wgs84"]["latitude"] == pytest.approx(-34.36,
                                                                 abs=0.01)
    assert document["origin_wgs84"]["longitude"] == pytest.approx(18.47,
                                                                  abs=0.01)


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:31287", "", "somewhere"])
def test_the_mesh_needs_a_utm_crs(tmp_path, crs):
    with pytest.raises(environment.EnvironmentFailure, match="not a UTM CRS"):
        cdd.build_dem(FLIGHT, str(tmp_path), crs, fetch=heights_of())

    assert os.listdir(tmp_path) == []


def test_no_heights_no_files(tmp_path):
    def fetch(grid, **more):
        raise copernicus_dem.NoElevation("open sea")

    with pytest.raises(copernicus_dem.NoElevation):
        cdd.build_dem(FLIGHT, str(tmp_path / "out"), "EPSG:32633",
                      fetch=fetch)

    assert not os.path.exists(tmp_path / "out")


def test_cancelled_after_the_heights_nothing_is_written(tmp_path):
    state = {"stop": False}

    def fetch(grid, log=None, cancelled=None, progress=None):
        state["stop"] = True                  # the user presses cancel
        return slope(grid), dict(SOURCE)

    with pytest.raises(environment.Cancelled):
        cdd.build_dem(FLIGHT, str(tmp_path / "out"), "EPSG:32633",
                      fetch=fetch, cancelled=lambda: state["stop"])

    assert not os.path.exists(tmp_path / "out")


def test_progress_runs_to_the_end(tmp_path):
    seen = []

    cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633", fetch=heights_of(),
                  progress=seen.append)

    assert seen == sorted(seen)
    assert seen[0] > 0 and seen[-1] == 100


def test_a_mesh_that_cannot_be_made_says_so(tmp_path, monkeypatch):
    from bambi_wildlife_detection import austria_dem_downloader

    monkeypatch.setattr(austria_dem_downloader.GLTFMeshGenerator,
                        "generate_mesh", lambda self, *a, **k: False)

    with pytest.raises(environment.EnvironmentFailure,
                       match="mesh could not be made"):
        cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                      fetch=heights_of())


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

class TestWorker:
    def _run(self, monkeypatch, build):
        monkeypatch.setattr(cdd, "build_dem", build)
        worker = cdd.CopernicusDEMWorker(FLIGHT, "/out", "EPSG:32633", 200.0)
        ended, log = [], []
        worker.finished.connect(lambda ok, text: ended.append((ok, text)))
        worker.log.connect(log.append)
        worker.run()
        return worker, ended, log

    def test_it_hands_back_the_mesh(self, monkeypatch):
        asked = []

        def build(bounds, folder, crs, padding, log=None, progress=None,
                  cancelled=None):
            asked.append((bounds, folder, crs, padding, cancelled()))
            return {"mesh": "/out/copernicus_dem.glb",
                    "source": {"product": "Copernicus DEM GLO-30"}}

        _worker, ended, log = self._run(monkeypatch, build)

        assert asked == [(FLIGHT, "/out", "EPSG:32633", 200.0, False)]
        assert ended == [(True, "/out/copernicus_dem.glb")]
        assert log == [
            "Fetching elevation from the Copernicus DEM...",
            "Elevation model written: /out/copernicus_dem.glb (Copernicus "
            "DEM GLO-30)"]

    def test_no_heights_is_told_as_it_is(self, monkeypatch):
        def build(*args, **more):
            raise copernicus_dem.NoElevation("Open sea: use 'Flat Surface'.")

        _worker, ended, log = self._run(monkeypatch, build)

        assert ended == [(False, "Open sea: use 'Flat Surface'.")]
        assert log[-1] == "Open sea: use 'Flat Surface'."

    def test_cancelled(self, monkeypatch):
        def build(*args, cancelled=None, **more):
            raise environment.Cancelled()

        worker, ended, _log = self._run(monkeypatch, build)
        worker.cancel()

        assert ended == [(False, "Cancelled")]
        assert worker._cancelled is True

    def test_anything_else_is_told_whole(self, monkeypatch):
        def build(*args, **more):
            raise ValueError("disk full")

        _worker, ended, _log = self._run(monkeypatch, build)

        assert ended == [(False, "The elevation model could not be made: "
                                 "disk full")]
