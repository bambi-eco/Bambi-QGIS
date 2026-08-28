# -*- coding: utf-8 -*-
"""The engine surface the plugin builds on: bambi-detection >= 1.0.0.

The plugin is the GUI over the ``bambi`` engine; every processing step will
call the array-in / array-out functions listed here (Phase 3 of the layer
split swaps the step bodies onto them, family by family). This test pins
that surface *from the plugin's side*: the symbols must exist, their
signatures must carry the parameters the plugin passes, and the version must
be at least the floor in ``core.dependency_ops``. An engine release that
renames or drops one of these fails here before it can fail in QGIS.

Skipped when ``bambi`` is not importable (the plugin's unit tier runs without
the engine installed); CI with the engine present runs it in full.
"""
import inspect

import pytest

bambi = pytest.importorskip("bambi")

from bambi_wildlife_detection.core.dependency_ops import _VERSION_RANGES  # noqa: E402

# module -> {function: [parameter names the plugin passes]}
CONTRACT = {
    "bambi.geo.poses": {
        "make_origin": ["latitude", "longitude", "altitude", "epsg"],
        "geographic_to_local": ["lla", "origin"],
        "local_to_geographic": ["positions", "origin"],
        "gimbal_to_rotation": [], "rotation_to_gimbal": [], "grid_convergence": [],
        "poses_from_geographic": [], "poses_to_geographic": [],
        "Origin": [], "Poses": [],
    },
    "bambi.geo.calibration": {
        "check_resolution": ["mtx", "width", "height"],
        "fovy_after_undistortion": ["mtx", "dist", "src_size", "new_size"],
        "new_camera_matrix": ["mtx", "dist", "src_size", "new_size"],
        "undistort_points": ["points", "mtx", "dist", "src_size", "new_size"],
        "undistort_boxes": ["boxes", "mtx", "dist", "src_size", "new_size"],
        "implied_resolution": ["mtx"],
    },
    "bambi.geo.camera": {
        "ray_convention": [],
        "camera_from_pose": ["position", "rotation", "fovy", "aspect_ratio", "translation_correction",
                             "rotation_correction"],
        "cameras_from_poses": ["poses", "fovy", "aspect_ratio", "translation_corrections", "rotation_corrections",
                               "indices"],
        "world_to_pixel": ["points", "camera", "width", "height"],
    },
    "bambi.geo.georef": {
        "pixels_to_world": ["pixels", "camera", "width", "height", "mesh"],
        "boxes_to_world": ["boxes", "camera", "width", "height", "mesh"],
        "boxes_to_world_by_frame": ["frames", "boxes", "poses", "fovy", "width", "height", "mesh",
                                    "translation_corrections", "rotation_corrections", "legacy"],
        "corners_to_extent": ["corners"],
        "footprint": ["camera", "width", "height", "mesh", "samples_per_edge"],
        "pixels_to_world_legacy": [],
    },
    "bambi.geo.dem": {"heightfield_mesh": ["elevation", "cell_size"], "sample_heightfield": []},
    "bambi.tracking.iou": {
        "track_boxes": ["frames", "boxes", "classes", "mode", "iou_threshold", "class_aware", "max_age",
                        "max_center_distance"],
        "interpolate_tracks": ["frames", "track_ids", "boxes", "confidences", "classes", "fill", "confidence"],
        "iou_matrix": [], "Tracks": [],
    },
    "bambi.tracking.matching": {
        "detections": ["frames", "track_ids", "boxes", "confidences", "detection_ids"],
        "match_frames_by_time": ["src_epochs", "dst_epochs", "max_dt"],
        "frame_pairs_from_map": ["frame_map"],
        "match_tracks": ["det_t", "det_w", "frame_pairs", "config", "frame_size_t", "frame_size_w"],
        "estimate_affine": [], "candidates": [], "assign": [], "rejection_reasons": [],
        "Affine": [], "MatchConfig": [], "MatchResult": [], "Candidates": [],
    },
    "bambi.survey.transects": {
        "cumulative_distances": ["positions"], "path_length": ["cum", "frame_a", "frame_b"],
        "frame_after_distance": ["cum", "start_frame", "meters"], "centerline": ["positions", "first_frame", "last_frame"],
        "split_by_distance": [], "split_into": [], "transect_lengths": [],
    },
    "bambi.survey.perpendicular": {
        "nearest_on_polyline": ["points", "line"],
        "perpendicular_to_route": ["points", "line", "camera_positions"],
        "flat_footprint": ["position", "rotation", "fovy", "aspect_ratio"],
        "route_length": ["line"],
    },
    "bambi.survey.density": {
        "kde_grid": ["points", "cell_size", "bandwidth", "nodata"],
        "coverage_grid": ["footprints", "cell_size", "bounds"], "Grid": [], "NODATA": [],
    },
    "bambi.survey.distance_sampling": {
        "estimate_density": ["distances", "transect_length", "truncation", "models"],
        "fit_detection_function": ["distances", "truncation", "model"],
        "lognormal_ci": [], "DistanceSamplingResult": [],
    },
    "bambi.survey.population": {
        "assign_to_transects": ["points", "centerlines", "truncation", "inside"],
        "points_in_geometries": ["points", "geometries"],
        "merged_footprint_area": ["footprints"], "geometry_rings": [],
        "estimate_naive": ["counts", "areas_ha"], "estimate_bootstrap": ["counts", "areas_ha", "n_boot", "seed"],
        "estimate_zinb": ["counts", "areas_ha"],
        "estimate_population": ["counts", "areas_ha", "methods", "n_boot", "seed", "study_area_ha"],
        "PopulationEstimate": [],
    },
    "bambi.render.masks": {
        "mask_polygon": ["mask", "simplify_epsilon"], "default_mask_polygon": ["width", "height"],
        "polygon_to_world": ["polygon", "camera", "width", "height", "mesh", "legacy"], "polygon_bounds": [],
    },
    "bambi.render.ortho": {
        "render_size": ["width_m", "height_m", "ground_resolution", "fixed", "max_dim"],
        "make_shot": ["ctx", "image", "position", "rotation", "fovy", "aspect_ratio", "translation_correction",
                      "rotation_correction", "lazy", "convention"],
        "make_shots": [], "ortho_camera": ["bounds", "height"],
        "render_orthographic": ["ctx", "mesh_data", "texture_data", "shots", "bounds", "size", "mask", "integral"],
        "tiles": ["size", "max_tile"], "tile_camera": [], "erode_valid_mask": ["valid", "erosion_px"],
        "frame_orthophoto": ["ctx", "mesh_data", "texture_data", "mesh", "image", "position", "rotation", "fovy",
                             "width", "height", "polygon", "ground_resolution", "translation_correction",
                             "rotation_correction", "bounds_correction_scale", "mask", "edge_erosion_px",
                             "convention"],
        "mask_texture": [], "crop_to_content": [],
    },
    "bambi.util.render_context": {"make_render_context": [], "render_backend": []},
    "bambi.io.poses": {"read_poses": [], "to_local_poses": [], "write_poses": [], "epochs_from_timestamps": []},
    "bambi.io.calibration": {"load_calibration": [], "media_resolution": [], "enforce_calibration": []},
    "bambi.io.corrections": {"read_corrections": [], "parse_corrections": [], "corrections_for_frames": []},
    "bambi.io.dem": {"read_dem_mesh": [], "write_dem_mesh": [], "geotiff_to_dem": [], "read_render_data": []},
    "bambi.io.tracks": {"read_detections": [], "write_detections": [], "read_georeferenced": [],
                        "write_georeferenced": [], "read_tracks_csv": [], "write_tracks_csv": [],
                        "read_pixel_tracks": [], "write_pixel_tracks": [], "read_mot": [], "write_mot": []},
    "bambi.io.trex": {"read_trex_tracklets": []},
    "bambi.io.survey": {"read_route_geojson": [], "write_route_geojson": [], "write_points_geojson": [],
                        "read_fov_polygons": [], "write_fov_polygons": [], "write_perpendicular_json": [],
                        "read_perpendicular_json": [], "read_transects": [], "write_transects": []},
    "bambi.io.raster": {"write_single_band": [], "read_single_band": []},
    "bambi.io.geotiff": {"write_frame_geotiff": [], "read_geotiff": [], "merge_average": [], "count_overlaps": []},
}


def test_installed_engine_meets_the_floor():
    packaging = pytest.importorskip("packaging.version")
    installed = getattr(bambi, "__version__", None)
    if installed is None:
        pytest.skip("bambi.__version__ missing (pre-1.0 engine)")
    assert packaging.Version(installed) >= packaging.Version(_VERSION_RANGES["bambi-detection"][0])


@pytest.mark.parametrize("module", sorted(CONTRACT))
def test_engine_module_exposes_the_contract(module):
    import importlib
    mod = importlib.import_module(module)
    missing = [name for name in CONTRACT[module] if not hasattr(mod, name)]
    assert not missing, f"{module} lacks {missing}"
    for name, params in CONTRACT[module].items():
        obj = getattr(mod, name)
        if not params or not callable(obj) or inspect.isclass(obj):
            continue
        have = set(inspect.signature(obj).parameters)
        lost = [p for p in params if p not in have]
        assert not lost, f"{module}.{name} lost parameters {lost}"
