# -*- coding: utf-8 -*-
"""Unit tests for the flat DEM surface mesh generator (BambiProcessor)."""
import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pyproj")
pytest.importorskip("gltflib")

from pyproj import Transformer  # noqa: E402

from bambi_wildlife_detection.bambi_processing import BambiProcessor  # noqa: E402

LAT, LON = 47.5, 13.0  # Austria
MSL = 500.0
EXTENT = 100.0


@pytest.fixture
def mesh_outputs(tmp_path):
    glb = str(tmp_path / "flat.glb")
    meta = str(tmp_path / "flat.json")
    BambiProcessor.generate_flat_surface_mesh(
        LAT, LON, MSL, EXTENT, glb, meta, epsg=32633)
    return tmp_path


class TestGenerateFlatSurfaceMesh:
    def test_glb_file_has_gltf_magic(self, mesh_outputs):
        data = (mesh_outputs / "flat.glb").read_bytes()
        assert data[:4] == b"glTF"

    def test_json_origin_matches_projected_centre(self, mesh_outputs):
        meta = json.loads((mesh_outputs / "flat.json").read_text())
        x_expected, y_expected = Transformer.from_crs(
            "EPSG:4326", "EPSG:32633", always_xy=True).transform(LON, LAT)
        origin = meta["origin"]
        assert origin[0] == pytest.approx(x_expected)
        assert origin[1] == pytest.approx(y_expected)
        assert origin[2] == MSL
        assert meta["crs"] == "EPSG:32633"
        assert meta["origin_wgs84"]["latitude"] == LAT
        assert meta["origin_wgs84"]["longitude"] == LON

    def test_geojson_ring_is_closed_quad(self, mesh_outputs):
        geojson = json.loads((mesh_outputs / "flat.geojson").read_text())
        ring = geojson["features"][0]["geometry"]["coordinates"][0]
        assert len(ring) == 5
        assert ring[0] == ring[-1]
        # All corners must lie within ~2x the extent of the centre
        # (100 m is well below 0.01 degrees at this latitude).
        for lon, lat in ring[:4]:
            assert abs(lon - LON) < 0.01
            assert abs(lat - LAT) < 0.01

    def test_auto_epsg_detects_utm_zone(self, tmp_path):
        glb = str(tmp_path / "auto.glb")
        meta_path = str(tmp_path / "auto.json")
        BambiProcessor.generate_flat_surface_mesh(
            LAT, LON, MSL, EXTENT, glb, meta_path, epsg=0)
        meta = json.loads((tmp_path / "auto.json").read_text())
        # Longitude 13°E, northern hemisphere → UTM zone 33N.
        assert meta["crs"] == "EPSG:32633"

    def test_southern_hemisphere_epsg(self, tmp_path):
        BambiProcessor.generate_flat_surface_mesh(
            -33.9, 18.4, 10.0, EXTENT,
            str(tmp_path / "s.glb"), str(tmp_path / "s.json"), epsg=0)
        meta = json.loads((tmp_path / "s.json").read_text())
        # Cape Town → UTM zone 34S.
        assert meta["crs"] == "EPSG:32734"
