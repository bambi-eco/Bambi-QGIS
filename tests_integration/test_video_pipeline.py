# -*- coding: utf-8 -*-
"""End-to-end pipeline integration tests on raw BAMBI dataset flight 6.

Stages are chained through session-scoped fixtures, mirroring how the plugin
runs them: extract frames → (fabricated) detections → geo-reference →
GeoTIFF export → orthomosaic / ALFS. Detection itself is not run (no model
inference in CI); instead, synthetic detections at the image centre exercise
the geo-referencing math against the real DEM and real camera poses.
"""
import json
import math
import os

import pytest

GEOREF_FRAMES = 10   # frames that get a synthetic detection
GEOTIFF_FRAMES = 3   # frames exported as GeoTIFF / merged into products


def _load_poses(target_folder, suffix):
    with open(os.path.join(target_folder, f"poses_{suffix}.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Stage fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def thermal_extraction(processor, base_config, target_folder):
    processor.extract_thermal_frames(base_config, log_fn=print)
    return _load_poses(target_folder, "t")


@pytest.fixture(scope="session")
def synthetic_detections(thermal_extraction, base_config, target_folder):
    """Write a centre-of-image detection for the first frames."""
    import cv2

    images = thermal_extraction["images"]
    first_frame = os.path.join(
        target_folder, "frames_t", images[0]["imagefile"])
    frame = cv2.imread(first_frame)
    assert frame is not None, f"Cannot read extracted frame {first_frame}"
    height, width = frame.shape[:2]

    det_folder = os.path.join(target_folder, "detections_t")
    os.makedirs(det_folder, exist_ok=True)
    det_file = os.path.join(det_folder, "detections.txt")
    half = 15
    n = min(GEOREF_FRAMES, len(images))
    with open(det_file, "w") as fh:
        fh.write(f"# video_size {width} {height}\n")
        for i in range(n):
            fh.write(
                f"{i} {width / 2 - half:.1f} {height / 2 - half:.1f} "
                f"{width / 2 + half:.1f} {height / 2 + half:.1f} 0.90 0\n"
            )
    return {"file": det_file, "count": n, "width": width, "height": height}


@pytest.fixture(scope="session")
def georeferenced(processor, base_config, synthetic_detections, target_folder):
    config = dict(base_config, georeference_camera="T")
    processor.run_georeference(config, log_fn=print)
    georef_file = os.path.join(target_folder, "georeferenced_t", "georeferenced.txt")
    assert os.path.isfile(georef_file)
    rows = []
    with open(georef_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            assert len(parts) >= 10, f"Unexpected georeferenced.txt row: {line}"
            rows.append({
                "frame": int(parts[1]),
                "x1": float(parts[2]), "y1": float(parts[3]), "z1": float(parts[4]),
                "x2": float(parts[5]), "y2": float(parts[6]), "z2": float(parts[7]),
                "confidence": float(parts[8]), "class_id": int(parts[9]),
            })
    return rows


@pytest.fixture(scope="session")
def geotiff_export(processor, base_config, thermal_extraction, target_folder):
    config = dict(
        base_config,
        geotiff_camera="T",
        alfs_use_all_frames=False,
        alfs_start_frame=0,
        alfs_end_frame=GEOTIFF_FRAMES - 1,
        alfs_ground_resolution=0.25,
    )
    processor.run_export_geotiffs(config, log_fn=print)
    return os.path.join(target_folder, "geotiffs_t")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

class TestThermalExtraction:
    def test_frames_and_poses_written(self, thermal_extraction, base_config, target_folder):
        images = thermal_extraction["images"]
        assert 0 < len(images) <= base_config["extract_limit"]
        for info in images:
            frame_path = os.path.join(target_folder, "frames_t", info["imagefile"])
            assert os.path.isfile(frame_path), f"Missing extracted frame {frame_path}"

    def test_poses_have_valid_geometry(self, thermal_extraction):
        for info in thermal_extraction["images"]:
            location = info["location"]
            rotation = info["rotation"]
            fovy = info["fovy"]
            assert len(location) == 3 and len(rotation) == 3
            # DEM-relative altitude: the drone flies above the terrain origin.
            assert -100 < location[2] < 1000
            fovy_value = fovy[0] if isinstance(fovy, list) else fovy
            assert 10 < float(fovy_value) < 120

    def test_camera_positions_form_a_track(self, thermal_extraction):
        # Consecutive 1 fps samples of a moving drone: distinct positions,
        # plausible speed (< 30 m/s).
        images = thermal_extraction["images"]
        assert len(images) >= 2
        steps = []
        for a, b in zip(images, images[1:]):
            dx = b["location"][0] - a["location"][0]
            dy = b["location"][1] - a["location"][1]
            steps.append(math.hypot(dx, dy))
        assert max(steps) < 30.0
        assert any(s > 0.01 for s in steps)


class TestRgbExtraction:
    def test_rgb_frames_and_poses(self, processor, base_config, target_folder):
        if not base_config["rgb_video_paths"]:
            pytest.skip("Flight has no RGB videos")
        config = dict(base_config, extract_limit=5)
        processor.extract_rgb_frames(config, log_fn=print)
        poses = _load_poses(target_folder, "w")
        images = poses["images"]
        assert 0 < len(images) <= 5
        first = os.path.join(target_folder, "frames_w", images[0]["imagefile"])
        assert os.path.isfile(first)


# ---------------------------------------------------------------------------
# Geo-referencing (real DEM ray casting via alfspy)
# ---------------------------------------------------------------------------

class TestGeoreference:
    def test_all_detections_hit_the_dem(self, georeferenced, synthetic_detections):
        assert len(georeferenced) == synthetic_detections["count"]

    def test_world_coordinates_are_near_the_camera(
            self, georeferenced, thermal_extraction, dem_files):
        with open(dem_files["json"]) as fh:
            origin = json.load(fh)["origin"]

        images = thermal_extraction["images"]
        for row in georeferenced:
            cam = images[row["frame"]]["location"]
            cam_x = cam[0] + origin[0]
            cam_y = cam[1] + origin[1]
            cx = (row["x1"] + row["x2"]) / 2
            cy = (row["y1"] + row["y2"]) / 2
            # A nadir-ish centre detection lands close to the ground point
            # below the camera.
            dist = math.hypot(cx - cam_x, cy - cam_y)
            assert dist < 300.0, (
                f"Frame {row['frame']}: georeferenced point {dist:.1f} m "
                f"from camera ground position"
            )

    def test_ground_elevation_is_plausible_for_austria(self, georeferenced):
        for row in georeferenced:
            assert 100.0 < row["z1"] < 2500.0

    def test_confidence_and_class_survive(self, georeferenced):
        for row in georeferenced:
            assert row["confidence"] == pytest.approx(0.90, abs=0.02)
            assert row["class_id"] == 0


# ---------------------------------------------------------------------------
# GeoTIFF export, orthomosaic, ALFS (headless OpenGL rendering)
# ---------------------------------------------------------------------------

class TestGeotiffExport:
    def test_one_geotiff_per_frame(self, geotiff_export):
        rasterio = pytest.importorskip("rasterio")  # noqa: F841
        names = sorted(
            f for f in os.listdir(geotiff_export) if f.endswith((".tif", ".tiff")))
        assert len(names) == GEOTIFF_FRAMES

    def test_geotiffs_are_georeferenced_in_target_crs(
            self, geotiff_export, dem_files, base_config):
        import numpy as np
        import rasterio

        with rasterio.open(dem_files["tif"]) as dem:
            dem_bounds = dem.bounds

        for name in sorted(os.listdir(geotiff_export)):
            if not name.endswith((".tif", ".tiff")):
                continue
            with rasterio.open(os.path.join(geotiff_export, name)) as ds:
                assert ds.crs is not None
                assert ds.crs.to_epsg() == base_config["target_epsg"]
                # The frame footprint must lie inside the DEM coverage.
                assert ds.bounds.left >= dem_bounds.left - 50
                assert ds.bounds.right <= dem_bounds.right + 50
                assert ds.bounds.bottom >= dem_bounds.bottom - 50
                assert ds.bounds.top <= dem_bounds.top + 50
                data = ds.read(1)
                assert np.count_nonzero(data) > 0, f"{name} is empty"


class TestOrthomosaic:
    def test_merges_frame_geotiffs(self, processor, base_config, geotiff_export, target_folder):
        import rasterio

        config = dict(base_config, ortho_camera="T", ortho_method="first")
        processor.run_orthomosaic(config, log_fn=print)

        output = os.path.join(target_folder, "orthomosaic_t", "orthomosaic.tif")
        assert os.path.isfile(output)
        with rasterio.open(output) as ds:
            assert ds.crs.to_epsg() == base_config["target_epsg"]
            assert ds.width > 0 and ds.height > 0


class TestAlfs:
    def test_renders_alfs_geotiff(self, processor, base_config, thermal_extraction, target_folder):
        import numpy as np
        import rasterio

        config = dict(
            base_config,
            alfs_camera="T",
            alfs_use_all_frames=False,
            alfs_start_frame=0,
            alfs_end_frame=GEOTIFF_FRAMES - 1,
            alfs_ground_resolution=0.25,
        )
        processor.run_alfs(config, log_fn=print)

        # alfspy writes either a single alfs.tif or (in the tiled code path)
        # tile_RR_CC.tif files into the alfs folder.
        alfs_folder = os.path.join(target_folder, "alfs_t")
        outputs = sorted(
            f for f in os.listdir(alfs_folder) if f.endswith((".tif", ".tiff")))
        assert outputs, f"No ALFS GeoTIFF written to {alfs_folder}"
        for name in outputs:
            with rasterio.open(os.path.join(alfs_folder, name)) as ds:
                assert ds.width > 0 and ds.height > 0
                data = ds.read(1)
                assert np.count_nonzero(data) > 0, f"ALFS output {name} is empty"
