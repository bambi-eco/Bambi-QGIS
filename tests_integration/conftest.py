# -*- coding: utf-8 -*-
"""Fixtures for the BAMBI integration test suite.

These tests run the real processing pipeline (frame extraction,
geo-referencing, GeoTIFF export, ALFS, orthomosaic) against a publicly
available raw dataset flight:

* Flight 6 of the BAMBI raw dataset (DJI M30T, red deer, Austria) is
  downloaded from Zenodo once and cached — mirroring the logic of
  https://github.com/bambi-eco/Dataset/blob/main/download_from_zenodo.py.
* The matching DEM is downloaded once from the Austrian BEV ATOM service via
  the plugin's own ``austria_dem_downloader`` module and also cached.

The cache location is taken from the ``BAMBI_TEST_DATA_DIR`` environment
variable. In the Docker setup this is a named volume mounted at ``/data``,
so both downloads survive container restarts. Without the variable the whole
suite is skipped.
"""
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.qgis_stub import install_qgis_stubs  # noqa: E402

install_qgis_stubs()


# ---------------------------------------------------------------------------
# Optional output persistence
# ---------------------------------------------------------------------------
# By default the pipeline outputs go to an ephemeral pytest tmp dir and are
# deleted with the container. Set BAMBI_TEST_KEEP_OUTPUT=1 (or point
# BAMBI_TEST_OUTPUT_DIR at a path) to keep them for manual/visual inspection.
# The default kept location lives under the repo (bind-mounted at /workspace),
# so the frames / GeoTIFFs / orthomosaic / ALFS renders appear on the host.

def _persistent_output_dir() -> "str | None":
    """Return the directory to keep pipeline outputs in, or None (ephemeral)."""
    explicit = os.environ.get("BAMBI_TEST_OUTPUT_DIR", "").strip()
    if explicit:
        return explicit
    if os.environ.get("BAMBI_TEST_KEEP_OUTPUT", "").strip().lower() in (
            "1", "true", "yes", "on"):
        return str(REPO_ROOT / "reports" / "integration_output")
    return None


def pytest_report_header(config):
    keep = _persistent_output_dir()
    if keep:
        return f"integration outputs: KEPT at {os.path.join(keep, 'pipeline_out')}"
    return ("integration outputs: ephemeral "
            "(set BAMBI_TEST_KEEP_OUTPUT=1 to keep them for inspection)")


# ---------------------------------------------------------------------------
# Test flight selection (raw BAMBI dataset)
# ---------------------------------------------------------------------------
FLIGHT_ID = "6"
# Deposition holding flight_6.zip, from zenodo_upload_summary_raw.json
# (part 42) in https://github.com/bambi-eco/Dataset.
ZENODO_DEPOSITION_ID = 19155449
ZENODO_API = "https://zenodo.org/api"

TARGET_EPSG = 32633  # flight 6 was recorded in Austria (UTM 33N)


def _data_dir() -> Path:
    root = os.environ.get("BAMBI_TEST_DATA_DIR")
    if not root:
        pytest.skip(
            "BAMBI_TEST_DATA_DIR is not set — integration tests need a cache "
            "directory (run via: docker compose run --rm integration)"
        )
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Zenodo flight download (mirrors download_from_zenodo.py)
# ---------------------------------------------------------------------------

def _get_deposition_files(deposition_id: int) -> dict:
    """Fetch the file listing for a deposition. Returns {filename: url}."""
    import requests

    r = requests.get(f"{ZENODO_API}/records/{deposition_id}", timeout=60)
    r.raise_for_status()
    data = r.json()

    file_map = {}
    for f in data.get("files", []):
        name = f.get("filename") or f.get("key")
        url = f.get("links", {}).get("download") or f.get("links", {}).get("self")
        if name and url:
            file_map[name] = url
    return file_map


def _download_file(url: str, dest: Path) -> None:
    """Stream-download a file in 8 MB chunks (same as the official script)."""
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r     {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


@pytest.fixture(scope="session")
def flight_folder():
    """Download and extract flight 6 into the cache (once), return its folder."""
    folder = _data_dir() / f"flight_{FLIGHT_ID}"
    marker = folder / ".extracted"
    if marker.exists():
        return folder

    folder.mkdir(parents=True, exist_ok=True)
    zip_name = f"flight_{FLIGHT_ID}.zip"
    zip_path = folder / zip_name

    if not zip_path.exists():
        print(f"\nFetching Zenodo deposition {ZENODO_DEPOSITION_ID} file list...")
        file_map = _get_deposition_files(ZENODO_DEPOSITION_ID)
        if zip_name not in file_map:
            pytest.fail(f"{zip_name} not found in deposition {ZENODO_DEPOSITION_ID}")
        print(f"Downloading {zip_name} (~12 GB, cached for future runs)...")
        _download_file(file_map[zip_name], zip_path)

    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(folder)
    zip_path.unlink()  # free ~12 GB; the extracted files are what we cache
    marker.touch()
    return folder


@pytest.fixture(scope="session")
def flight_files(flight_folder):
    """Discover the raw flight's videos, SRTs, AirData log and calibrations."""
    def _find(patterns, required=True):
        hits = sorted(
            p for p in flight_folder.rglob("*")
            if p.is_file() and any(p.match(g) for g in patterns)
        )
        if required and not hits:
            pytest.fail(f"No file matching {patterns} in {flight_folder}")
        return hits

    videos = _find(["*.MP4", "*.mp4"])
    thermal_videos = [p for p in videos if "_T" in p.stem.upper()]
    rgb_videos = [p for p in videos if "_W" in p.stem.upper()]
    srts = _find(["*.SRT", "*.srt"])
    thermal_srts = [p for p in srts if "_T" in p.stem.upper()]
    rgb_srts = [p for p in srts if "_W" in p.stem.upper()]
    csvs = _find(["*.csv", "*.CSV"])

    if not thermal_videos or len(thermal_videos) != len(thermal_srts):
        pytest.fail(
            f"Unexpected flight layout: thermal videos {thermal_videos} "
            f"vs SRTs {thermal_srts}"
        )

    # Undistortion parameters shipped with the raw flight (any JSON containing
    # an OpenCV camera matrix); fall back to the plugin's built-in presets.
    from bambi_wildlife_detection.bambi_calibrations import (
        RGB_CALIBRATIONS, THERMAL_CALIBRATIONS)

    thermal_calib = THERMAL_CALIBRATIONS["DJI M30T (T;Video)"]
    rgb_calib = RGB_CALIBRATIONS["DJI M30T (W;Video)"]
    for path in _find(["*.json", "*.JSON"], required=False):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict) or "mtx" not in data:
            continue
        # Camera marker as a separate name token (handles "T_calib.json",
        # "6_T.json", "..._W_calib.json", ...).
        tokens = re.split(r"[^A-Za-z0-9]+", path.stem.upper())
        if "T" in tokens:
            thermal_calib = data
        elif "W" in tokens or "V" in tokens:
            rgb_calib = data

    return {
        "thermal_videos": [str(p) for p in thermal_videos],
        "thermal_srts": [str(p) for p in thermal_srts],
        "rgb_videos": [str(p) for p in rgb_videos],
        "rgb_srts": [str(p) for p in rgb_srts],
        "airdata_csv": str(csvs[0]),
        "thermal_calibration": thermal_calib,
        "rgb_calibration": rgb_calib,
    }


# ---------------------------------------------------------------------------
# DEM download via the plugin's Austrian DEM downloader
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dem_files(flight_files):
    """Build the DEM (GeoTIFF + GLB mesh + metadata) for the flight area.

    Replicates DEMDownloadWorker.run() headless: AirData bbox → BEV tile
    download → merge/clip → reproject → GLTF mesh. Both the raw BEV tiles
    and the final DEM products are cached in the data volume.
    """
    data_dir = _data_dir()
    dem_dir = data_dir / f"dem_flight_{FLIGHT_ID}"
    dem_tif = dem_dir / "dem.tif"
    dem_glb = dem_dir / "dem.glb"
    dem_json = dem_dir / "dem.json"
    if dem_tif.exists() and dem_glb.exists() and dem_json.exists():
        return {"tif": str(dem_tif), "glb": str(dem_glb), "json": str(dem_json)}

    import tempfile

    from bambi_wildlife_detection.austria_dem_downloader import (
        BEVDownloader, BoundingBox, DEMProcessor, GLTFMeshGenerator)

    dem_dir.mkdir(parents=True, exist_ok=True)
    log = print

    bbox = BoundingBox.from_airdata_csv(flight_files["airdata_csv"], padding_meters=200)
    assert bbox is not None, "Could not read GPS bounding box from AirData CSV"
    log(f"Flight bbox: SW {bbox.min_lat:.5f},{bbox.min_lon:.5f} "
        f"NE {bbox.max_lat:.5f},{bbox.max_lon:.5f}")

    downloader = BEVDownloader(cache_dir=data_dir / "bev_tiles", log_callback=log)
    tile_paths = downloader.download_tiles_for_bbox(bbox)
    assert tile_paths, "No BEV DEM tiles downloaded — area outside Austria?"

    processor = DEMProcessor(output_crs=f"EPSG:{TARGET_EPSG}", log_callback=log)
    with tempfile.TemporaryDirectory() as temp_dir:
        clipped = Path(temp_dir) / "clipped.tif"
        assert processor.merge_and_clip(tile_paths, clipped, bbox), \
            "Failed to merge and clip DEM tiles"
        assert processor.reproject_geotiff(clipped, dem_tif), \
            "Failed to reproject DEM"

    mesh_generator = GLTFMeshGenerator(simplify_factor=2, log_callback=log)
    assert mesh_generator.generate_mesh(
        dem_tif, dem_glb, dem_json, source_crs=f"EPSG:{TARGET_EPSG}"), \
        "Failed to generate DEM mesh"

    return {"tif": str(dem_tif), "glb": str(dem_glb), "json": str(dem_json)}


# ---------------------------------------------------------------------------
# Pipeline scaffolding
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def target_folder(tmp_path_factory):
    """Fresh pipeline output folder shared by all stages of the session.

    Ephemeral by default. When BAMBI_TEST_KEEP_OUTPUT / BAMBI_TEST_OUTPUT_DIR
    request persistence, the folder is created under a host-visible path and
    NOT deleted after the run, so the frames / GeoTIFFs / orthomosaic / ALFS
    renders can be inspected manually.
    """
    keep = _persistent_output_dir()
    if not keep:
        return str(tmp_path_factory.mktemp("pipeline_out"))

    folder = os.path.join(keep, "pipeline_out")
    # Start from a clean slate so stale outputs from a previous run cannot
    # mask a regression; do NOT remove it on teardown.
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    return folder


@pytest.fixture(scope="session")
def base_config(flight_files, dem_files, target_folder):
    """Plugin configuration dict as the dock widget would assemble it."""
    return {
        "target_folder": target_folder,
        "target_epsg": TARGET_EPSG,
        "dem_path": dem_files["glb"],
        "alfs_dem_metadata_path": dem_files["json"],
        "airdata_path": flight_files["airdata_csv"],
        "thermal_video_paths": flight_files["thermal_videos"],
        "thermal_srt_paths": flight_files["thermal_srts"],
        "thermal_calibration_data": flight_files["thermal_calibration"],
        "rgb_video_paths": flight_files["rgb_videos"],
        "rgb_srt_paths": flight_files["rgb_srts"],
        "rgb_calibration_data": flight_files["rgb_calibration"],
        # Keep runs short: one frame per second of video, capped.
        "extract_sampling_rate": 30,
        "extract_limit": 25,
        # Flight 6 was recorded 2023-09-20 in Austria: CEST = UTC+2. Both
        # extraction methods apply this offset when matching SRT timestamps
        # against the (UTC) AirData log; the dock widget normally
        # auto-detects it.
        "timezone_offset_hours": 2.0,
        # Correction factors determined for flight 6 with the Correction
        # Wizard (translation in metres, rotation in radians).
        "translation": {"x": 0.0, "y": 0.0, "z": -2.825},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.011},
    }


@pytest.fixture(scope="session")
def processor():
    from bambi_wildlife_detection.bambi_processing import BambiProcessor
    return BambiProcessor()
