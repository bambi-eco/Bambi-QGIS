# -*- coding: utf-8 -*-
"""Unit tests for :meth:`BambiProcessor._get_extraction_prerequisites`.

The method resolves the WGS84 origin that every extracted camera pose is
expressed relative to.  Getting it wrong silently poisons the whole pipeline
(camera locations become ``-Infinity`` or land far off the DEM), so the
degree-vs-metres disambiguation is pinned down here.
"""
import json
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_airdata(monkeypatch):
    """Provide the ``bambi.airdata.air_data_frame`` import used by the method.

    The upstream ``bambi`` package is not installed in the unit test image;
    only the plain attribute container is needed here.
    """
    if "bambi.airdata.air_data_frame" in sys.modules:
        return

    class AirDataFrame:
        latitude = 0.0
        longitude = 0.0
        altitude = 0.0

    bambi_mod = types.ModuleType("bambi")
    airdata_mod = types.ModuleType("bambi.airdata")
    frame_mod = types.ModuleType("bambi.airdata.air_data_frame")
    frame_mod.AirDataFrame = AirDataFrame
    airdata_mod.air_data_frame = frame_mod
    bambi_mod.airdata = airdata_mod

    monkeypatch.setitem(sys.modules, "bambi", bambi_mod)
    monkeypatch.setitem(sys.modules, "bambi.airdata", airdata_mod)
    monkeypatch.setitem(sys.modules, "bambi.airdata.air_data_frame", frame_mod)


def _config(tmp_path, dem_json, target_epsg=32633):
    dem_path = tmp_path / "dem.json"
    dem_path.write_text(json.dumps(dem_json))
    return {
        "target_folder": str(tmp_path / "target"),
        "target_epsg": target_epsg,
        "alfs_dem_metadata_path": str(dem_path),
    }


class TestOriginResolution:
    def test_uses_stored_origin_wgs84(self, processor, tmp_path):
        config = _config(tmp_path, {
            "origin": [432000.0, 5262000.0, 500.0],
            "origin_wgs84": {"latitude": 47.5, "longitude": 13.4, "altitude": 500.0},
        })

        _, _, ad_origin = processor._get_extraction_prerequisites(config)

        assert ad_origin.latitude == pytest.approx(47.5)
        assert ad_origin.longitude == pytest.approx(13.4)
        assert ad_origin.altitude == pytest.approx(500.0)

    def test_equatorial_dem_keeps_metric_origin(self, processor, tmp_path):
        """Near-equator surveys must not trip the degree fallback.

        A correct ``origin_wgs84.latitude`` of -0.005 is "essentially zero",
        but ``origin`` here holds UTM metres.  Treating the northing
        (9_999_428) as a latitude pushes it outside the projection domain,
        pyproj returns ``inf`` and every pose location becomes ``-Infinity``.
        """
        config = _config(tmp_path, {
            "origin": [264862.53909388673, 9999428.819025926, 1797.9703369140625],
            "origin_wgs84": {
                "latitude": -0.005164117243387438,
                "longitude": 36.887361102737735,
                "altitude": 1797.9703369140625,
            },
        }, target_epsg=32737)

        _, transformer, ad_origin = processor._get_extraction_prerequisites(config)

        assert ad_origin.latitude == pytest.approx(-0.005164117243387438)
        assert ad_origin.longitude == pytest.approx(36.887361102737735)

        easting, northing = transformer.transform(ad_origin.latitude, ad_origin.longitude)
        assert easting == pytest.approx(264862.539, abs=1.0)
        assert northing == pytest.approx(9999428.819, abs=1.0)

    def test_degree_origin_fallback_still_applies(self, processor, tmp_path):
        """The DJI mislabeled-CRS case: ``origin`` really is in degrees."""
        config = _config(tmp_path, {
            "origin": [72.814, 3.887, 12.0],
            "origin_wgs84": {"latitude": 0.0, "longitude": 70.5, "altitude": 0.0},
        }, target_epsg=32643)

        _, _, ad_origin = processor._get_extraction_prerequisites(config)

        assert ad_origin.longitude == pytest.approx(72.814)
        assert ad_origin.latitude == pytest.approx(3.887)
        assert ad_origin.altitude == pytest.approx(12.0)

    def test_invalid_origin_raises(self, processor, tmp_path):
        """An unusable origin fails loudly instead of producing infinities."""
        config = _config(tmp_path, {
            "origin": [264862.0, 9999428.0, 1797.0],
            "origin_wgs84": {"latitude": 9999428.0, "longitude": 264862.0, "altitude": 0.0},
        }, target_epsg=32737)

        with pytest.raises(RuntimeError, match="invalid WGS84 coordinate"):
            processor._get_extraction_prerequisites(config)
