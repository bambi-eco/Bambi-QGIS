# -*- coding: utf-8 -*-
"""Reviewing in the inspector: the details panel, and deleting with the map
layers following (``bambi_feature_viewer``, ``bambi_layer_sync``,
``bambi_track_inventory_dialog``).

The store part of a deletion is unit-tested in ``tests/test_review.py``;
this tier checks the pieces that need real QGIS objects: memory layers
tagged the way the dock widget tags them, the layer tree, and the widgets.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.core import (
    QgsFeature, QgsField, QgsGeometry, QgsLayerTreeGroup, QgsPointXY,
    QgsProject, QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant

from bambi_wildlife_detection import bambi_layer_sync
from bambi_wildlife_detection.core import (
    detection_store, review, track_store)


# ---------------------------------------------------------------------------
# A flight in the store, and its layers on the map
# ---------------------------------------------------------------------------

@pytest.fixture
def flight(tmp_path):
    """Two thermal tracks, geo-referenced, as the pipeline leaves them."""
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 1, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.7, "source_class": "0"},
        {"frame": 2, "x1": 14.0, "y1": 24.0, "x2": 34.0, "y2": 44.0,
         "confidence": 0.8, "source_class": "0"},
        {"frame": 1, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
         "confidence": 0.6, "source_class": "0"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": ids[i], "gx1": 100.0 + 10 * i, "gy1": 200.0,
         "gz1": 0.0, "gx2": 102.0 + 10 * i, "gy2": 202.0, "gz2": 0.0}
        for i in range(4)])
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 1, "detection_id": ids[2]},
        {"track_id": 2, "detection_id": ids[3]},
    ])
    return root


def _tag(layer, layer_type, root):
    layer.setCustomProperty("bambi_layer_type", layer_type)
    layer.setCustomProperty("bambi_target_folder", root)
    layer.setCustomProperty("bambi_detection_camera", "T")
    return layer


def _polygon(x1, y1, x2, y2):
    return QgsGeometry.fromPolygonXY([[
        QgsPointXY(x1, y1), QgsPointXY(x2, y1), QgsPointXY(x2, y2),
        QgsPointXY(x1, y2), QgsPointXY(x1, y1)]])


def _track_layers(root, track_id, points):
    """A Final Position and a Path layer as add_tracks_to_qgis draws them."""
    final = QgsVectorLayer("Polygon?crs=EPSG:32633", "Final Position", "memory")
    final.dataProvider().addAttributes([
        QgsField("track_id", QVariant.Int), QgsField("frame", QVariant.Int),
        QgsField("confidence", QVariant.Double),
        QgsField("class_id", QVariant.Int)])
    final.updateFields()
    last = points[-1]
    feature = QgsFeature(final.fields())
    feature.setGeometry(_polygon(last["x1"], last["y1"], last["x2"], last["y2"]))
    feature.setAttributes([track_id, last["frame"], last["confidence"], 0])
    final.dataProvider().addFeatures([feature])
    _tag(final, "track_final", root)

    path = None
    if len(points) >= 2:
        path = QgsVectorLayer("LineString?crs=EPSG:32633", "Path", "memory")
        path.dataProvider().addAttributes([
            QgsField("track_id", QVariant.Int),
            QgsField("start_frame", QVariant.Int),
            QgsField("end_frame", QVariant.Int),
            QgsField("num_detections", QVariant.Int),
            QgsField("avg_confidence", QVariant.Double)])
        path.updateFields()
        feature = QgsFeature(path.fields())
        feature.setGeometry(QgsGeometry.fromPolylineXY([
            QgsPointXY((p["x1"] + p["x2"]) / 2, (p["y1"] + p["y2"]) / 2)
            for p in points]))
        feature.setAttributes([track_id, points[0]["frame"], points[-1]["frame"],
                               len(points), 0.8])
        path.dataProvider().addFeatures([feature])
        _tag(path, "track_path", root)
    return final, path


def _detection_layer(root, detections):
    layer = QgsVectorLayer("Polygon?crs=EPSG:32633", "Detections_Frame", "memory")
    layer.dataProvider().addAttributes([
        QgsField("det_id", QVariant.Int), QgsField("frame", QVariant.Int),
        QgsField("confidence", QVariant.Double),
        QgsField("class_id", QVariant.Int)])
    layer.updateFields()
    features = []
    for index, det in enumerate(detections):
        feature = QgsFeature(layer.fields())
        feature.setGeometry(_polygon(det["x1"], det["y1"], det["x2"], det["y2"]))
        # A running index, as layers written before this release carry.
        feature.setAttributes([index, det["frame"], det["confidence"], 0])
        features.append(feature)
    layer.dataProvider().addFeatures(features)
    return _tag(layer, "detection", root)


@pytest.fixture
def mapped(flight, clean_project):
    """The flight's layers in the layer tree, grouped as the dock does."""
    from bambi_wildlife_detection.core import pipeline_outputs

    root = clean_project.layerTreeRoot()
    tracks_group = QgsLayerTreeGroup("BAMBI Wildlife Tracks (Thermal)")
    root.addChildNode(tracks_group)
    layers = {}
    for track_id in (1, 2):
        points = pipeline_outputs.geo_track_points(flight, "t", track_id)
        final, path = _track_layers(flight, track_id, points)
        group = tracks_group.addGroup(f"Track {track_id}")
        for layer in (final, path):
            if layer is None:
                continue
            clean_project.addMapLayer(layer, False)
            group.addLayer(layer)
        layers[track_id] = (final, path, group)

    dets = pipeline_outputs.load_georef_detections_by_frame(
        flight + "/georeferenced_t/georeferenced.txt")
    det_layer = _detection_layer(flight, [d for f in sorted(dets)
                                          for d in dets[f]])
    clean_project.addMapLayer(det_layer, False)
    root.addLayer(det_layer)
    return {"tracks": layers, "detections": det_layer, "group": tracks_group}


# ---------------------------------------------------------------------------
# bambi_layer_sync
# ---------------------------------------------------------------------------

class TestLayerSync:

    def test_a_deleted_track_leaves_the_map_with_its_group(self, flight, mapped):
        final, path, group = mapped["tracks"][1]
        ids = {final.id(), path.id()}
        result = bambi_layer_sync.remove_track(flight, "t", 1)
        assert result["layers"] == 2
        assert not ids & set(QgsProject.instance().mapLayers())
        assert [g.name() for g in mapped["group"].children()] == ["Track 2"]
        # The other track is untouched.
        other_final, _path, _group = mapped["tracks"][2]
        assert other_final.id() in QgsProject.instance().mapLayers()

    def test_a_deleted_detection_leaves_its_layer(self, flight, mapped):
        layer = mapped["detections"]
        assert layer.featureCount() == 4
        ids = [d["detection_id"] for d in track_store.load_detections(flight, "t")]
        removed = bambi_layer_sync.remove_detection(
            flight, "t", ids[1], frame=1, confidence=0.7, class_id=0)
        assert removed == 1
        assert layer.featureCount() == 3
        assert sorted(f["confidence"] for f in layer.getFeatures()) == [0.6, 0.8, 0.9]

    def test_other_cameras_and_folders_are_left_alone(self, flight, mapped):
        layer = mapped["detections"]
        assert bambi_layer_sync.remove_detection(
            flight, "w", 1, frame=1, confidence=0.7, class_id=0) == 0
        assert bambi_layer_sync.remove_detection(
            flight + "_other", "t", 1, frame=1, confidence=0.7, class_id=0) == 0
        assert layer.featureCount() == 4
        assert bambi_layer_sync.remove_track(flight, "w", 1)["layers"] == 0

    def test_a_track_that_lost_a_box_is_redrawn(self, flight, mapped):
        ids = [d["detection_id"] for d in track_store.load_detections(flight, "t")]
        # Delete the last box in the store, then redraw.
        review.delete_detections(flight, "t", [ids[2]])
        result = bambi_layer_sync.redraw_track(flight, "t", 1)
        assert result["points"] == 2
        final, path, _group = mapped["tracks"][1]
        feature = next(final.getFeatures())
        assert feature["frame"] == 1
        assert feature.geometry().boundingBox().xMinimum() == pytest.approx(110.0)
        feature = next(path.getFeatures())
        assert (feature["start_frame"], feature["end_frame"],
                feature["num_detections"]) == (0, 1, 2)
        assert len(feature.geometry().asPolyline()) == 2

    def test_a_track_down_to_one_box_loses_its_path(self, flight, mapped):
        ids = [d["detection_id"] for d in track_store.load_detections(flight, "t")]
        review.delete_detections(flight, "t", ids[1:3])
        final, path, group = mapped["tracks"][1]
        # Removing a layer deletes the C++ object, so remember the ids.
        final_id, path_id = final.id(), path.id()
        bambi_layer_sync.redraw_track(flight, "t", 1)
        assert path_id not in QgsProject.instance().mapLayers()
        assert final_id in QgsProject.instance().mapLayers()
        assert next(final.getFeatures())["frame"] == 0
        assert [n.name() for n in group.children()] == ["Final Position"]


# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------

class TestFeatureViewer:

    def test_track_view_shows_the_inventory_facts(self, flight, monkeypatch):
        from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog
        from bambi_wildlife_detection.core import inspection

        monkeypatch.setattr(FeatureViewerDialog, "_show_and_raise",
                            lambda self: None)
        viewer = FeatureViewerDialog()
        frames = inspection.track_frames(flight, "t", 1)
        viewer.show_track("Track 1", frames, 0, target_folder=flight,
                          track_id=1, modality="t", epsg=32633)
        assert viewer.details_widget.isVisible() or viewer.details_widget.isVisibleTo(viewer)
        text = viewer.details_label.text()
        assert "3 box(es)" in text and "0 - 2" in text
        assert "animal" in text
        assert viewer.approved_check.isEnabled()
        assert not viewer.approved_check.isChecked()
        assert viewer.delete_track_btn.text() == "Delete track 1"
        assert viewer.delete_detection_btn.isEnabled()
        viewer.close()

    def test_approving_in_the_viewer_reaches_the_store(self, flight, monkeypatch):
        from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog
        from bambi_wildlife_detection.core import inspection

        monkeypatch.setattr(FeatureViewerDialog, "_show_and_raise",
                            lambda self: None)
        viewer = FeatureViewerDialog()
        heard = []
        viewer.trackApproved.connect(lambda *args: heard.append(args))
        viewer.show_track("Track 1", inspection.track_frames(flight, "t", 1), 0,
                          target_folder=flight, track_id=1, modality="t")
        viewer.approved_check.setChecked(True)
        assert track_store.track_labels(flight, "t")[1]["attributes"] == {
            "approved": True}
        assert heard == [(flight, "t", 1, True)]
        viewer.close()

    def test_fov_views_are_not_reviewable(self, flight, monkeypatch):
        from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog

        monkeypatch.setattr(FeatureViewerDialog, "_show_and_raise",
                            lambda self: None)
        viewer = FeatureViewerDialog()
        viewer.show_track("FoV", [{
            "frame_idx": 0, "image_path_t": "", "image_path_w": "",
            "boxes_modality": "t", "boxes_green": [], "boxes_blue": [],
            "target_folder": flight, "dem_path": "", "correction_path": ""}],
            0, target_folder=flight)
        assert not viewer.review_widget.isVisibleTo(viewer)
        assert not viewer.details_widget.isVisibleTo(viewer)
        viewer.close()

    def test_deleting_a_detection_reloads_the_track(self, flight, mapped,
                                                    monkeypatch):
        from qgis.PyQt.QtWidgets import QMessageBox
        from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog
        from bambi_wildlife_detection.core import inspection

        monkeypatch.setattr(FeatureViewerDialog, "_show_and_raise",
                            lambda self: None)
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.StandardButton.Yes)
        viewer = FeatureViewerDialog()
        heard = []
        viewer.detectionDeleted.connect(lambda *args: heard.append(args))
        viewer.show_track("Track 1", inspection.track_frames(flight, "t", 1), 2,
                          target_folder=flight, track_id=1, modality="t")
        ids = [d["detection_id"] for d in track_store.load_detections(flight, "t")]
        viewer._delete_current_detection()

        assert heard == [(flight, "t", ids[2], 1)]
        assert ids[2] not in [d["detection_id"] for d in
                              track_store.load_detections(flight, "t")]
        assert len(viewer._frames) == 2 and viewer._current_idx == 1
        assert "2 box(es)" in viewer.details_label.text()
        # The map followed: the box is gone and the track was redrawn.
        assert mapped["detections"].featureCount() == 3
        final, _path, _group = mapped["tracks"][1]
        assert next(final.getFeatures())["frame"] == 1
        viewer.close()

    def test_deleting_the_track_closes_the_viewer(self, flight, mapped, monkeypatch):
        from qgis.PyQt.QtWidgets import QMessageBox
        from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog
        from bambi_wildlife_detection.core import inspection

        monkeypatch.setattr(FeatureViewerDialog, "_show_and_raise",
                            lambda self: None)
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.StandardButton.Yes)
        viewer = FeatureViewerDialog()
        heard = []
        viewer.trackDeleted.connect(lambda *args: heard.append(args))
        viewer.show_track("Track 2", inspection.track_frames(flight, "t", 2), 0,
                          target_folder=flight, track_id=2, modality="t")
        final, _path, _group = mapped["tracks"][2]
        final_id = final.id()
        viewer._delete_current_track()

        assert heard == [(flight, "t", 2)]
        assert inspection.track_frames(flight, "t", 2) == []
        assert track_store.track_orphans(flight, "t") == []
        assert final_id not in QgsProject.instance().mapLayers()
        assert [g.name() for g in mapped["group"].children()] == ["Track 1"]
        # Its box left the detection layer as well.
        assert mapped["detections"].featureCount() == 3
        assert not viewer._frames


# ---------------------------------------------------------------------------
# The report follows
# ---------------------------------------------------------------------------

class TestInventoryDialogFollows:

    def _dialog(self):
        from bambi_wildlife_detection.bambi_track_inventory_dialog import (
            BambiTrackInventoryDialog)
        blank = {"sex": "", "age": "", "matched_track": None, "start_x": None}
        rows = [
            {"track_id": 1, "approved": False, "species": "red deer",
             "n_boxes": 3, "start_frame": 0, "end_frame": 2, **blank},
            {"track_id": 2, "approved": False, "species": "roe deer",
             "n_boxes": 1, "start_frame": 1, "end_frame": 1, **blank},
        ]
        recorded = []
        return BambiTrackInventoryDialog(
            rows, "Thermal", set_approved=lambda t, a: recorded.append((t, a)),
            columns=["track_id", "species", "n_boxes"]), recorded

    def _column(self, dialog, name):
        return dialog.columns.index(name)

    def test_remove_track_drops_the_row(self):
        dialog, _recorded = self._dialog()
        assert dialog.table.rowCount() == 2
        assert dialog.remove_track(1)
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, self._column(dialog, "track_id")).text() == "2"
        assert not dialog.remove_track(42)
        assert "1 tracked individual" in dialog.summary.text()

    def test_update_track_replaces_its_cells(self):
        dialog, _recorded = self._dialog()
        from qgis.PyQt.QtCore import Qt
        dialog.table.sortByColumn(self._column(dialog, "n_boxes"),
                                  Qt.SortOrder.AscendingOrder)
        dialog.update_track({"track_id": 1, "approved": False,
                             "species": "red deer", "n_boxes": 2,
                             "start_frame": 0, "end_frame": 1, "sex": "",
                             "age": "", "matched_track": None, "start_x": None})
        cells = {dialog.table.item(r, self._column(dialog, "track_id")).text():
                 dialog.table.item(r, self._column(dialog, "n_boxes")).text()
                 for r in range(dialog.table.rowCount())}
        assert cells == {"1": "2", "2": "1"}
        # The sort the user chose survived the rebuild.
        assert dialog.table.horizontalHeader().sortIndicatorSection() == \
            self._column(dialog, "n_boxes")

    def test_set_track_approved_ticks_without_recording(self):
        dialog, recorded = self._dialog()
        dialog.set_track_approved(2, True)
        column = self._column(dialog, "approved")
        ticked = {dialog.table.item(r, self._column(dialog, "track_id")).text():
                  dialog.table.item(r, column).checkState()
                  for r in range(dialog.table.rowCount())}
        from qgis.PyQt.QtCore import Qt
        assert ticked["2"] == Qt.CheckState.Checked
        assert ticked["1"] == Qt.CheckState.Unchecked
        assert recorded == []
        assert "1 of 2 approved" in dialog.summary.text()
