# -*- coding: utf-8 -*-
"""The heights of an elevation mesh are where the elevation model has them.

A raster cell's height is the height at the cell's centre. The mesh put it
at the cell's south-west corner, which moved the whole terrain half a cell:
half a metre with the Austrian one-metre model, five metres with a ten-metre
one - and on a slope that is a height error at every position.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from bambi_wildlife_detection.core import dem_mesh

WEST, NORTH = 500000.0, 5200000.0


def surface(east, north):
    """A slope: 30 cm up per metre east, 10 cm per metre north."""
    return 300.0 + 0.30 * (east - WEST) + 0.10 * (north - NORTH)


# ---------------------------------------------------------------------------
# Where a cell's height is
# ---------------------------------------------------------------------------

class TestCellPosition:
    def test_it_is_the_cells_centre(self):
        # 4 rows of 10 m cells; row 0 is the northernmost
        assert dem_mesh.cell_position(3, 0, 4, 10.0, 10.0) == (5.0, 5.0)
        assert dem_mesh.cell_position(0, 0, 4, 10.0, 10.0) == (5.0, 35.0)
        assert dem_mesh.cell_position(0, 7, 4, 10.0, 10.0) == (75.0, 35.0)

    def test_cells_need_not_be_square(self):
        assert dem_mesh.cell_position(1, 2, 2, 10.0, 4.0) == (25.0, 2.0)


class TestOlderMeshes:
    NEW = {"transform": [10.0, 0.0, WEST, 0.0, -10.0, NORTH, 0, 0, 1],
           "heights_at": "cell centres"}
    OLD = {"transform": [10.0, 0.0, WEST, 0.0, -10.0, NORTH, 0, 0, 1]}
    FLAT = {"origin": [WEST, NORTH, 400.0], "crs": "EPSG:32633"}

    def test_a_mesh_made_since_is_where_it_belongs(self):
        assert dem_mesh.stale_shift(self.NEW) is None
        assert dem_mesh.stale_note(self.NEW) is None

    def test_a_mesh_made_before_is_half_a_cell_off(self):
        assert dem_mesh.stale_shift(self.OLD) == (5.0, 5.0)
        note = dem_mesh.stale_note(self.OLD)
        assert "5 m too far west and 5 m too far south" in note
        assert "Create the mesh again" in note

    def test_the_one_metre_model(self):
        metre = {"transform": [1.0, 0.0, WEST, 0.0, -1.0, NORTH, 0, 0, 1]}
        assert dem_mesh.stale_shift(metre) == (0.5, 0.5)
        assert "0.5 m too far west" in dem_mesh.stale_note(metre)

    def test_a_flat_surface_has_no_cells(self):
        assert dem_mesh.stale_shift(self.FLAT) is None
        assert dem_mesh.stale_note(self.FLAT) is None

    @pytest.mark.parametrize("metadata", [
        None, {}, {"transform": None}, {"transform": [1.0]},
        {"transform": ["a", 0, 0, 0, "b", 0]}, {"transform": [0, 0, 0, 0, 0, 0]},
    ])
    def test_metadata_that_does_not_say(self, metadata):
        assert dem_mesh.cell_size(metadata) is None
        assert dem_mesh.stale_note(metadata) is None

    def test_the_steps_say_so_once_in_their_log(self):
        from bambi_wildlife_detection.bambi_processing import BambiProcessor

        said = []
        BambiProcessor._say_if_the_mesh_is_shifted(self.OLD, said.append)
        BambiProcessor._say_if_the_mesh_is_shifted(self.NEW, said.append)
        BambiProcessor._say_if_the_mesh_is_shifted(self.FLAT, said.append)
        BambiProcessor._say_if_the_mesh_is_shifted(self.OLD, None)

        assert len(said) == 1 and said[0].startswith("Note: this elevation mesh")


# ---------------------------------------------------------------------------
# The mesh that is made
# ---------------------------------------------------------------------------

@pytest.fixture
def make_mesh(tmp_path):
    """Write a GeoTIFF of :func:`surface` and make its mesh; returns the
    mesh's vertices in world coordinates and its metadata."""
    rasterio = pytest.importorskip("rasterio")
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin

    from bambi_wildlife_detection.austria_dem_downloader import (
        GLTFMeshGenerator)

    def make(cell_x=10.0, cell_y=10.0, rows=30, cols=40, simplify=1,
             heights=None):
        east = WEST + (np.arange(cols) + 0.5) * cell_x
        north = NORTH - (np.arange(rows) + 0.5) * cell_y
        E, N = np.meshgrid(east, north)
        values = surface(E, N) if heights is None else heights(E, N)
        tif = tmp_path / "dem.tif"
        with rasterio.open(
                str(tif), "w", driver="GTiff", height=rows, width=cols,
                count=1, dtype="float32", crs="EPSG:32633",
                transform=from_origin(WEST, NORTH, cell_x, cell_y)) as dataset:
            dataset.write(values.astype("float32"), 1)

        glb, meta = tmp_path / "dem.glb", tmp_path / "dem.json"
        assert GLTFMeshGenerator(simplify_factor=simplify).generate_mesh(
            Path(tif), Path(glb), Path(meta), source_crs="EPSG:32633")
        metadata = json.loads(meta.read_text())
        mesh = trimesh.load(str(glb), force="mesh", process=False)
        return (np.asarray(mesh.vertices, dtype=np.float64)
                + np.array(metadata["origin"]), metadata)

    return make


class TestTheMesh:
    def test_its_heights_are_the_surfaces_heights_where_they_are(self, make_mesh):
        """On this slope every height used to be 2 m above the surface at
        the place it was put: 5 m west and 5 m south of where it is."""
        world, _metadata = make_mesh()

        off = world[:, 2] - surface(world[:, 0], world[:, 1])
        assert np.abs(off).max() < 0.01

    def test_cells_that_are_not_square(self, make_mesh):
        world, _metadata = make_mesh(cell_x=10.0, cell_y=4.0)
        off = world[:, 2] - surface(world[:, 0], world[:, 1])
        assert np.abs(off).max() < 0.01

    def test_a_thinned_out_mesh(self, make_mesh):
        """Every third cell: each still at its own centre."""
        world, _metadata = make_mesh(simplify=3)
        off = world[:, 2] - surface(world[:, 0], world[:, 1])
        assert np.abs(off).max() < 0.01
        assert len(world) == 10 * 14

    def test_a_summit_is_at_its_cell(self, make_mesh):
        def summit(east, north):
            height = np.full(east.shape, 500.0)
            height[12, 25] = 900.0
            return height

        world, _metadata = make_mesh(heights=summit)
        top = world[np.argmax(world[:, 2])]

        assert top[0] == pytest.approx(WEST + 25.5 * 10.0)
        assert top[1] == pytest.approx(NORTH - 12.5 * 10.0)

    def test_it_covers_the_cell_centres(self, make_mesh):
        world, _metadata = make_mesh()
        assert world[:, 0].min() == pytest.approx(WEST + 5.0)
        assert world[:, 0].max() == pytest.approx(WEST + 395.0)
        assert world[:, 1].min() == pytest.approx(NORTH - 295.0)
        assert world[:, 1].max() == pytest.approx(NORTH - 5.0)

    def test_it_says_where_its_heights_are(self, make_mesh):
        _world, metadata = make_mesh()
        assert metadata["heights_at"] == "cell centres"
        assert dem_mesh.stale_note(metadata) is None
        # and the origin is the raster's corner, as it was
        assert metadata["origin"][:2] == [WEST, NORTH - 300.0]
