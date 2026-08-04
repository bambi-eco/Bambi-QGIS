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
    # Mirror into the 6.0 store so run_georeference takes the store path and
    # keys its results on detection_id (EXCHANGE_FORMAT_PLAN.md phase 3).
    from bambi_wildlife_detection.core import detection_store

    detection_store.record_detections(
        target_folder, "t",
        [{"frame": i,
          "x1": round(width / 2 - half, 2), "y1": round(height / 2 - half, 2),
          "x2": round(width / 2 + half, 2), "y2": round(height / 2 + half, 2),
          "confidence": 0.90, "source_class": "0"} for i in range(n)],
        log_fn=print)

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


# ---------------------------------------------------------------------------
# 6.0 migration on real pipeline output (EXCHANGE_FORMAT_PLAN.md §9, §12.2)
# ---------------------------------------------------------------------------

class TestMigrationOnRealOutput:
    """Migrate the folder this run just produced, and check it accounts for
    every row the legacy files hold.

    The unit tier proves the mapping against synthetic fixtures; this proves it
    against text a real pipeline wrote, including whatever the geo-referencing
    stage decided to drop.
    """

    @pytest.fixture(scope="class")
    def legacy_copy(self, georeferenced, target_folder, tmp_path_factory):
        """The legacy text outputs alone, in a folder with no 6.0 store.

        The pipeline now writes the store as it runs, and migration refuses to
        touch a folder that already has one — so migrating a real 5.x project
        means migrating the text files without it.
        """
        import shutil

        root = str(tmp_path_factory.mktemp("legacy_project"))
        for name in os.listdir(target_folder):
            if name.startswith(("detections_", "georeferenced_", "tracks_",
                                "fov_", "labels_", "segmentation_")):
                shutil.copytree(os.path.join(target_folder, name),
                                os.path.join(root, name))
        return root

    @pytest.fixture(scope="class")
    def migrated(self, legacy_copy):
        from bambi_wildlife_detection.core import migration
        return migration.migrate_project(legacy_copy, log_fn=print)

    def test_detections_match_the_text_file(
            self, migrated, legacy_copy, synthetic_detections):
        from bambi_wildlife_detection.core import store

        conn = store.open_store(
            store.stage_path(legacy_copy, store.DETECTIONS, "t"),
            store.DETECTIONS, "t")
        try:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM detections").fetchone()["n"]
        finally:
            conn.close()
        assert n == synthetic_detections["count"]

    def test_every_detection_is_accounted_for(self, migrated, legacy_copy):
        """Either geo-referenced or explicitly failed — never silently absent."""
        from bambi_wildlife_detection.core import store

        det = store.open_store(
            store.stage_path(legacy_copy, store.DETECTIONS, "t"),
            store.DETECTIONS, "t")
        geo = store.open_store(
            store.stage_path(legacy_copy, store.GEOREFERENCED, "t"),
            store.GEOREFERENCED, "t")
        try:
            all_ids = {r["detection_id"] for r in det.execute(
                "SELECT detection_id FROM detections")}
            resolved = {r["detection_id"] for r in geo.execute(
                "SELECT detection_id FROM detections_geo")}
            failed = {r["detection_id"] for r in geo.execute(
                "SELECT detection_id FROM georef_failures")}
        finally:
            det.close()
            geo.close()

        unaccounted = all_ids - resolved - failed
        assert not unaccounted, (
            f"{len(unaccounted)} detection(s) neither geo-referenced nor "
            "recorded as failures")

    def test_geo_coordinates_survive_the_round_trip(
            self, migrated, legacy_copy, georeferenced):
        from bambi_wildlife_detection.core import store

        geo = store.open_store(
            store.stage_path(legacy_copy, store.GEOREFERENCED, "t"),
            store.GEOREFERENCED, "t")
        try:
            rows = [dict(r) for r in geo.execute(
                "SELECT gx1, gy1 FROM detections_geo ORDER BY detection_id")]
        finally:
            geo.close()

        expected = [(r["x1"], r["y1"]) for r in georeferenced
                    if r["x1"] >= 0 and r["y1"] >= 0]
        assert [(r["gx1"], r["gy1"]) for r in rows] == expected

    def test_vocabulary_is_seeded(self, migrated, legacy_copy):
        from bambi_wildlife_detection.core import store

        conn = store.open_store(store.project_path(legacy_copy), store.PROJECT)
        try:
            base = {r["name"]: r["species_id"] for r in conn.execute(
                "SELECT species_id, name FROM species WHERE protected = 1")}
        finally:
            conn.close()
        assert base == {"animal": 0, "unknown": -1, "not-an-animal": -2}

    def test_legacy_files_are_untouched(self, migrated, synthetic_detections):
        with open(synthetic_detections["file"]) as fh:
            content = fh.read()
        assert "# video_size" in content


# ---------------------------------------------------------------------------
# Phase 3: geo-referencing against the store (EXCHANGE_FORMAT_PLAN.md §3.2, §12.2)
# ---------------------------------------------------------------------------

class TestGeoreferenceStore:
    """The live geo-referencing stage, checked through the store.

    The unit tier proves the accounting logic on synthetic data; this proves it
    on a real DEM ray-cast, where detections genuinely can fail to project.
    """

    def test_every_detection_is_accounted_for(self, georeferenced, target_folder):
        from bambi_wildlife_detection.core import track_store

        report = track_store.accounting(target_folder, "t")
        assert report["detections"] > 0
        assert report["unaccounted"] == [], (
            f"{len(report['unaccounted'])} detection(s) neither geo-referenced "
            "nor recorded as failures")
        assert report["both"] == []

    def test_resolved_and_failed_sum_to_the_detections(
            self, georeferenced, target_folder):
        from bambi_wildlife_detection.core import track_store

        report = track_store.accounting(target_folder, "t")
        assert report["resolved"] + report["failed"] == report["detections"]

    def test_geo_rows_carry_a_real_detection_id(self, georeferenced, target_folder):
        from bambi_wildlife_detection.core import track_store

        rows = track_store.load_georeferenced(target_folder, "t")
        assert rows, "no geo-referenced detections in the store"
        detection_ids = {row["detection_id"] for row in rows}
        assert len(detection_ids) == len(rows)     # 1:1, no duplicates

    def test_store_agrees_with_the_text_output(
            self, georeferenced, target_folder):
        """Dual-write parity for the geo-referencing stage."""
        from bambi_wildlife_detection.core import track_store

        rows = track_store.load_georeferenced(target_folder, "t")
        assert len(rows) == len(georeferenced)
        for stored, text in zip(
                sorted(rows, key=lambda r: (r["frame"], r["gx1"])),
                sorted(georeferenced, key=lambda r: (r["frame"], r["x1"]))):
            assert stored["frame"] == text["frame"]
            assert abs(stored["gx1"] - text["x1"]) < 1e-6
            assert abs(stored["gy1"] - text["y1"]) < 1e-6

    def test_failures_carry_a_known_reason(self, georeferenced, target_folder):
        from bambi_wildlife_detection.core import track_store

        summary = track_store.failure_summary(target_folder, "t")
        assert set(summary) <= set(track_store.GEOREF_REASONS)

    def test_no_track_orphans(self, georeferenced, target_folder):
        from bambi_wildlife_detection.core import track_store

        assert track_store.track_orphans(target_folder, "t") == []
