# -*- coding: utf-8 -*-
"""The survey-area GeoPackage in real QGIS.

``core.survey_area`` writes the feature GeoPackage with plain ``sqlite3`` and
reads it back without GDAL, so the two questions only a real QGIS answers are:
does GDAL accept the file as an editable polygon layer in the right CRS, and
do features QGIS writes come back through the core reader? The panel's import
(reprojecting another layer into the plot) is exercised on top.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.core import (
    QgsFeature, QgsGeometry, QgsVectorLayer, QgsWkbTypes,
)

from bambi_wildlife_detection.core import survey_area

EPSG = 32633


def _layer(path, table):
    return QgsVectorLayer(f"{path}|layername={table}", table, "ogr")


def test_both_layers_open_as_polygon_layers_in_the_target_crs(tmp_path):
    path = survey_area.create(str(tmp_path), EPSG)
    for table in survey_area.LAYERS:
        layer = _layer(path, table)
        assert layer.isValid(), table
        assert layer.geometryType() == \
            QgsWkbTypes.GeometryType.PolygonGeometry
        assert layer.crs().authid() == f"EPSG:{EPSG}"
        assert {"name", "reason"} <= set(layer.fields().names())


def test_a_polygon_drawn_in_qgis_is_read_by_the_analytics(tmp_path):
    folder = str(tmp_path)
    path = survey_area.create(folder, EPSG)
    layer = _layer(path, survey_area.PLOT_LAYER)
    assert layer.startEditing()
    feature = QgsFeature(layer.fields())
    feature.setGeometry(QgsGeometry.fromWkt(
        "POLYGON((500000 5300000, 500100 5300000, 500100 5300100, "
        "500000 5300100, 500000 5300000))"))
    feature["name"] = "District"
    assert layer.addFeature(feature)
    assert layer.commitChanges(), layer.commitErrors()

    area = survey_area.load(folder, EPSG)
    assert area is not None
    assert area.plot_m2 == pytest.approx(10000.0)
    assert area.describe()["plots"] == ["District"]


def test_polygons_written_by_core_show_up_in_qgis(tmp_path):
    from shapely.geometry import box

    path = survey_area.create(
        str(tmp_path), EPSG,
        exclusions=[(box(0, 0, 10, 20), "Wood", "thick forest cover")])
    layer = _layer(path, survey_area.EXCLUSION_LAYER)
    features = list(layer.getFeatures())
    assert len(features) == 1
    assert features[0]["reason"] == "thick forest cover"
    assert features[0].geometry().area() == pytest.approx(200.0)


def test_the_panel_imports_and_reprojects_a_layer(dock, tmp_path,
                                                  clean_project):
    flight = tmp_path / "flight"
    flight.mkdir()
    survey_folder = tmp_path / "survey"
    dock.target_folder_edit.setText(str(flight))
    dock.target_crs_edit.setText(f"EPSG:{EPSG}")
    dock.survey_panel.folder_edit.setText(str(survey_folder))

    # A district outline in WGS84, as a shapefile of it would usually be.
    source = QgsVectorLayer("Polygon?crs=EPSG:4326&field=name:string",
                            "district", "memory")
    feature = QgsFeature(source.fields())
    feature.setGeometry(QgsGeometry.fromWkt(
        "POLYGON((15 47, 15.01 47, 15.01 47.01, 15 47.01, 15 47))"))
    feature["name"] = "Revier Nord"
    source.dataProvider().addFeatures([feature])
    clean_project.addMapLayer(source)

    panel = dock.survey_area_panel
    count = panel._copy_features(source, survey_area.PLOT_LAYER,
                                 selected_only=False, replace=True)
    assert count == 1

    # The plot belongs to the survey, so it lands in the survey folder.
    area = survey_area.load(str(survey_folder), EPSG)
    assert area.describe()["plots"] == ["Revier Nord"]
    # ~760 m x 1110 m at 47° N, in metres now.
    assert 70 < area.plot_m2 / 10000.0 < 95
    panel.refresh()
    assert panel.enabled_check.isEnabled()
    assert panel.enabled_check.isChecked()
    assert dock.current_survey().area_enabled is True
