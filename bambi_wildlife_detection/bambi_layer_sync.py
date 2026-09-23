# -*- coding: utf-8 -*-
"""Keeping the QGIS layers in step with the store after a deletion.

The map layers are built from the store, but they are copies: a detection
polygon or a track's path drawn into a GeoPackage under ``qgis_layers``.
Deleting the detection in the store would leave that polygon on the map -
an orphan box nobody can click through to anything - so this module takes
it off the map too, or redraws a track from the members it still has.

Layers are found by the custom properties the inspector already relies on
(``bambi_layer_type``, ``bambi_target_folder``, ``bambi_detection_camera``),
never by name, so renamed layers and several flights in one project work.

Derived analytics rasters (density, coverage) are not touched: they are
recomputed by their steps, which the store marks stale.
"""
import os
from typing import Callable, Dict, Iterable, List, Optional

from qgis.core import (
    QgsFeature, QgsGeometry, QgsLayerTreeGroup, QgsPointXY, QgsProject,
    QgsVectorLayer,
)

TRACK_TYPES = ("track_final", "track_path")
DETECTION_TYPE = "detection"

#: Confidence is stored with four decimals in the layers; this is the same
#: tolerance the inspector identifies a clicked box with.
_CONFIDENCE_TOLERANCE = 0.0015


def _same_folder(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _layers(target_folder: str, types: Iterable[str],
            modality: Optional[str] = None) -> List[QgsVectorLayer]:
    """BAMBI vector layers of *types* built from *target_folder*; with a
    *modality*, only those of that camera."""
    wanted = set(types)
    camera = None if modality is None else ("T" if modality == "t" else "W")
    found = []
    for layer in QgsProject.instance().mapLayers().values():
        if not isinstance(layer, QgsVectorLayer):
            continue
        if layer.customProperty("bambi_layer_type", "") not in wanted:
            continue
        if not _same_folder(layer.customProperty("bambi_target_folder", ""),
                            target_folder):
            continue
        if camera is not None and \
                layer.customProperty("bambi_detection_camera", "T") != camera:
            continue
        found.append(layer)
    return found


def _attr(feature: QgsFeature, name: str):
    try:
        value = feature[name]
    except KeyError:
        return None
    # QGIS hands NULL back as a QVariant; anything without a plain value is None.
    if value is None or (hasattr(value, "isNull") and value.isNull()):
        return None
    return value


def _delete_where(layer: QgsVectorLayer,
                  predicate: Callable[[QgsFeature], bool]) -> int:
    fids = [f.id() for f in layer.getFeatures() if predicate(f)]
    if fids:
        layer.dataProvider().deleteFeatures(fids)
        layer.updateExtents()
        layer.triggerRepaint()
    return len(fids)


def _remove_layers(layers: List[QgsVectorLayer]) -> None:
    """Remove layers from the project, and the group nodes they leave empty.

    A track's subgroup holds only its Final Position and Path; once those
    are gone the group is an empty heading, which is confusing rather than
    useful.
    """
    if not layers:
        return
    root = QgsProject.instance().layerTreeRoot()
    parents = []
    for layer in layers:
        node = root.findLayer(layer.id())
        if node is not None and node.parent() is not None \
                and node.parent() not in parents:
            parents.append(node.parent())
    QgsProject.instance().removeMapLayers([layer.id() for layer in layers])
    for group in parents:
        if isinstance(group, QgsLayerTreeGroup) and not group.children() \
                and group.parent() is not None:
            group.parent().removeChildNode(group)


def _refresh_canvas() -> None:
    try:
        from qgis.utils import iface
        if iface is not None:
            iface.mapCanvas().refresh()
    except Exception:  # nosec B110 - no canvas outside QGIS
        pass


def _track_layers(target_folder: str, modality: str,
                  track_id: int) -> Dict[str, List[QgsVectorLayer]]:
    """The Final Position and Path layers drawn for *track_id*."""
    found: Dict[str, List[QgsVectorLayer]] = {t: [] for t in TRACK_TYPES}
    for layer in _layers(target_folder, TRACK_TYPES, modality):
        for feature in layer.getFeatures():
            if _attr(feature, "track_id") == track_id:
                found[layer.customProperty("bambi_layer_type")].append(layer)
                break
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_track(target_folder: str, modality: str, track_id: int) -> dict:
    """Take a deleted track off the map.

    Its Final Position and Path layers go (with their subgroup), the pair
    drawn for it in the matched-pairs layer goes, and so does its point in
    the inventory layer. Returns what was removed, for the log.
    """
    layers = _track_layers(target_folder, modality, track_id)
    to_remove = layers["track_final"] + layers["track_path"]
    _remove_layers(to_remove)

    column = "track_t" if modality == "t" else "track_w"
    pairs = sum(_delete_where(layer, lambda f: _attr(f, column) == track_id)
                for layer in _layers(target_folder, ["matched_pairs"]))

    camera = "thermal" if modality == "t" else "rgb"

    def _its_point(feature) -> bool:
        same_track = _attr(feature, "track_id") == track_id
        return same_track and _attr(feature, "camera") == camera

    points = sum(_delete_where(layer, _its_point)
                 for layer in _layers(target_folder, ["track_inventory"]))

    _refresh_canvas()
    return {"layers": len(to_remove), "pairs": pairs, "inventory_points": points}


def remove_detection(target_folder: str, modality: str, detection_id: int,
                     frame: int, confidence: float, class_id: int) -> int:
    """Take a deleted detection's polygon out of the detection layers.

    Layers written before this release carry a running index in ``det_id``
    rather than the store id, so the box is identified the way the
    inspector identifies a clicked one: by frame, confidence and class.
    When that still leaves several candidates (two boxes of one class with
    the same rounded confidence on one frame), ``det_id`` decides.
    Returns the number of polygons removed.
    """
    def _same_box(feature) -> bool:
        if _attr(feature, "frame") != frame or _attr(feature, "class_id") != class_id:
            return False
        found = _attr(feature, "confidence")
        if found is None:
            return False
        return abs(float(found) - float(confidence)) < _CONFIDENCE_TOLERANCE

    removed = 0
    for layer in _layers(target_folder, [DETECTION_TYPE], modality):
        candidates = [f for f in layer.getFeatures() if _same_box(f)]
        if len(candidates) > 1:
            by_id = [f for f in candidates if _attr(f, "det_id") == detection_id]
            if by_id:
                candidates = by_id
        fids = [f.id() for f in candidates]
        if fids:
            layer.dataProvider().deleteFeatures(fids)
            layer.updateExtents()
            layer.triggerRepaint()
            removed += len(fids)
    if removed:
        _refresh_canvas()
    return removed


def _polygon(point: dict) -> QgsGeometry:
    corners = [
        QgsPointXY(point["x1"], point["y1"]),
        QgsPointXY(point["x2"], point["y1"]),
        QgsPointXY(point["x2"], point["y2"]),
        QgsPointXY(point["x1"], point["y2"]),
        QgsPointXY(point["x1"], point["y1"]),
    ]
    return QgsGeometry.fromPolygonXY([corners])


def _set_feature(layer: QgsVectorLayer, geometry: QgsGeometry,
                 values: Dict[str, object]) -> None:
    """Rewrite the (single) feature of a track layer in place."""
    provider = layer.dataProvider()
    fields = layer.fields()
    feature = next(layer.getFeatures(), None)
    if feature is None:
        new = QgsFeature(fields)
        new.setGeometry(geometry)
        new.setAttributes([values.get(field.name()) for field in fields])
        provider.addFeatures([new])
    else:
        provider.changeGeometryValues({feature.id(): geometry})
        provider.changeAttributeValues({feature.id(): {
            fields.indexFromName(name): value for name, value in values.items()
            if fields.indexFromName(name) >= 0}})
    layer.updateExtents()
    layer.triggerRepaint()


def redraw_track(target_folder: str, modality: str, track_id: int) -> dict:
    """Redraw a track that lost a member from the members it still has.

    The Path becomes the polyline through the remaining ground positions
    (and is removed when fewer than two remain, as it is never drawn for a
    single box), and the Final Position moves to the last remaining one.
    A track with no ground position left loses both. Returns what changed.
    """
    from .core import pipeline_outputs

    layers = _track_layers(target_folder, modality, track_id)
    points = pipeline_outputs.geo_track_points(target_folder, modality, track_id)
    if not points:
        _remove_layers(layers["track_final"] + layers["track_path"])
        _refresh_canvas()
        return {"removed": len(layers["track_final"] + layers["track_path"]),
                "points": 0}

    last = points[-1]
    for layer in layers["track_final"]:
        _set_feature(layer, _polygon(last), {
            "track_id": track_id, "frame": last["frame"],
            "confidence": last["confidence"], "class_id": last["class_id"]})

    if len(points) < 2:
        _remove_layers(layers["track_path"])
    else:
        centres = [QgsPointXY((p["x1"] + p["x2"]) / 2.0, (p["y1"] + p["y2"]) / 2.0)
                   for p in points]
        avg_conf = sum(p["confidence"] for p in points) / len(points)
        for layer in layers["track_path"]:
            _set_feature(layer, QgsGeometry.fromPolylineXY(centres), {
                "track_id": track_id,
                "start_frame": points[0]["frame"],
                "end_frame": points[-1]["frame"],
                "num_detections": len(points),
                "avg_confidence": round(avg_conf, 4)})

    _refresh_canvas()
    return {"removed": 0 if len(points) >= 2 else len(layers["track_path"]),
            "points": len(points)}
