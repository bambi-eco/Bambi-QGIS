# BAMBI QGIS Plugin

[![QGIS Plugin](https://img.shields.io/badge/QGIS-Plugin-589632?logo=qgis&logoColor=white)](https://plugins.qgis.org/plugins/bambi_wildlife_detection/)
[![QGIS](https://img.shields.io/badge/QGIS-3.22+-green.svg)](https://qgis.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

Detect, track, and geo-reference wildlife in aerial drone recordings, directly in QGIS. BAMBI processes drone **videos** or **still photo** collections from DJI Enterprise drones (thermal and/or RGB), projects every detection onto a Digital Elevation Model, and loads all results as styled map layers.

![Plugin Overview](images/plugin_overview.png)

## What it does

- **Detect & track animals**: YOLO-based detection (model auto-downloaded from HuggingFace) and multi-object tracking with built-in, BoxMOT, or geo-referenced backends; pre-computed [TRex](https://trex.run) tracklets can be imported instead of running a tracker
- **Geo-reference everything**: detections, tracks, and SAM3 segmentations are projected onto a DEM to real-world UTM coordinates
- **Label manually**: key-frame based labelling tool to review results on the extracted frames (thermal or RGB) and to create or correct MOT-style track annotations with species/sex/age/occlusion classes, interpolation between key frames, geo-referenced box propagation, and export back into the detection pipeline
- **Survey analytics**: perpendicular distance sampling from detections/tracks to the flight route, per-frame camera field-of-view footprints, coverage areas, kernel-density heatmaps of animal locations, and line-transect distance-sampling density/abundance estimation with confidence intervals
- **Map products**: georeferenced per-frame GeoTIFFs, true orthomosaics, and Airborne Light Field Sampling (ALFS) mosaics
- **Calibration wizards**: guided workflows for camera intrinsics (single-camera SfM or stereo RGB+thermal) and per-flight positional/rotational corrections
- **Results that carry their own meaning**: every detection has an identity the later steps refer to, so a track knows which detections it is made of and a geo-referencing failure is recorded rather than silently dropped; user-defined fields travel through the whole pipeline
- **Export & publish**: COCO, YOLO, MOT, TRex tracklets, GeoJSON, Camtrap DP, and Darwin Core Archive for GBIF
- **Several flights per project**: each with its own target folder, configuration and QGIS layer group
- **Companion tools**: video creator for shareable result videos, radiometric thermal image viewer, randomized transect flight planner, interactive map-canvas inspectors, and a built-in dependency manager

## Installation

1. In QGIS, go to **Plugins → Manage and Install Plugins…**, search for **BAMBI**, and install it directly from the [official QGIS Plugin Repository](https://plugins.qgis.org/plugins/bambi_wildlife_detection/)
2. Open the plugin's **Dependency Manager** (toolbar button) and install the two required packages: **BAMBI Detection Framework** and **ALFS-PY**
3. **Restart QGIS**: newly installed Python packages are only picked up at startup

Alternatively, install from a ZIP file ([GitHub releases](https://github.com/bambi-eco/Bambi-QGIS/releases) or a zipped development version of this repository) or copy the plugin folder manually; see the [Installation Guide](docs/installation.md).

Optional capabilities (GPU inference, BoxMOT trackers, camera calibration via SfM, flight planning, DJI Thermal SDK) can be added the same way. See the [Installation Guide](docs/installation.md) for details and manual installation methods.

## Quick start

1. Click the **BAMBI** icon in the toolbar to open the dock widget
2. In the **Input** tab, choose **Video** or **Photo** mode and select your recordings, camera calibration, AirData flight log, DEM, target CRS, and output folder
3. Adjust per-step settings in the **Configuration** tab (detection confidence, tracker backend, thermal colormaps, …)
4. Run the steps in the **Pre-Processing** tab (**P1. Extract Frames** to
   **P6. Generate Orthomosaic**) and the **Processing** tab (**A1. Detect
   Animals** to **A4. Geo-Reference Segmentation**). Pre-Processing derives
   from the drone poses and the DEM and is independent of any animal;
   Processing depends on the detections, so the two can be run separately
5. Use the **→ Add … to QGIS** buttons after each step to load the results as styled layers, grouped under the flight's name
6. Export the results — COCO, YOLO, MOT, GeoJSON, Camtrap DP or a Darwin Core Archive for GBIF — from the **Export** box on the Processing tab

Upgrading from 5.x? Open your existing project and use **Migrate 5.x…** beside the target folder; the existing files are only read, never modified. See [Results, Flights and Export](docs/results-and-export.md).

What you need as input: MP4 videos + SRT subtitle files (video mode) *or* a folder of still images (photo mode), an AirData flight log CSV, a camera calibration JSON, and a DEM. The plugin can download DEMs automatically (Austria), convert any GeoTIFF DEM, or generate a flat surface mesh for aquatic surveys, and the calibration JSON can be created with the built-in Camera Calibration Wizard.

## Documentation

| Guide | Contents |
|-------|----------|
| [Installation & Dependencies](docs/installation.md) | Install methods, Dependency Manager, optional packages, GPU support, DJI Thermal SDK |
| [Processing Pipeline](docs/pipeline.md) | Input modes and file formats, configuration, all processing steps, output folder structure |
| [Results, Flights & Export](docs/results-and-export.md) | Where results are stored, migrating a 5.x project, several flights per project, the project schema (species, enums, custom fields), and the export formats |
| [DEM Import & Conversion](docs/dem.md) | Automatic download (Austria), GeoTIFF conversion, manual DEMs, flat surface meshes for aquatic surveys |
| [Correction Wizard](docs/correction-wizard.md) | Finding per-flight positional/rotational correction factors |
| [Camera Calibration Wizard](docs/camera-calibration.md) | Estimating camera intrinsics for single cameras (SfM) and stereo RGB+thermal setups |
| [Tools](docs/tools.md) | Video Creator, Thermal Image Viewer, Labelling Tool, interactive map-canvas inspectors |
| [Flight Strategy Planner](docs/flight-planner.md) | Randomized transect-based survey route generation |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and how to resolve them |

## Requirements

- **QGIS** 3.22+ (Windows, Linux, or macOS) with its bundled Python 3.9+
- **Drone data** from DJI Enterprise drones (M30T, M3T, M4T, M300, …) with thermal and/or RGB cameras; GPS RTK recordings recommended for best geo-referencing accuracy
- **Video mode** needs SRT subtitle files and an AirData flight log CSV; **photo mode** needs still images with EXIF timestamps and an AirData CSV

## Development

### Running everything (combined coverage)

The three test tiers below run in separate images (they need mutually
incompatible environments), so a combined coverage figure is produced by
running each and merging the results with `coverage combine`:

```bash
bash run_all_tests.sh
```

This runs the unit, integration and real-QGIS smoke tiers in turn, then
prints one combined report (also written to `reports/coverage/`, with an
HTML view under `reports/coverage/html/`). Each tier still runs on its own
via the individual commands below when you only want one part. The smoke
tier contributes coverage of the GUI modules the QGIS-free unit suite cannot
reach, so the combined figure is substantially higher than any single tier.

### Unit tests

The repository ships a QGIS-free unit test suite (`tests/`) covering the pure
processing logic: geometry and FOV footprints, per-frame corrections, track
interpolation, distance-sampling statistics, world/prj file output, box
projection, flat surface mesh generation, and tracker configuration. The
`qgis` package is stubbed by `tests/conftest.py`, so no QGIS installation is
needed. Run the suite in Docker (reports land in `reports/`):

```bash
docker compose run --rm tests
```

Or locally in any Python 3.9+ environment with
`pytest pytest-cov numpy scipy pyproj gltflib requests` installed:

```bash
pytest tests
```

### Integration tests

`tests_integration/` runs the real pipeline — thermal/RGB frame extraction,
DEM ray-casting geo-reference, per-frame GeoTIFF export, orthomosaic, and
ALFS rendering — against **flight 6 of the public BAMBI raw dataset**
([Zenodo](https://zenodo.org/record/19155449), DJI M30T, red deer, Austria)
with the matching DEM fetched through the plugin's Austrian BEV downloader.
Detection inference is skipped; synthetic detections exercise the
geo-referencing math against real poses and terrain.

```bash
docker compose run --rm integration
```

The first run downloads ~12 GB (flight ZIP + DEM tiles) into the
`bambi-test-data` Docker volume; subsequent runs reuse the cache and only
re-execute the pipeline. The image installs the BAMBI Detection Framework
and ALFS-PY from GitHub (CPU-only torch) and renders through Xvfb/Mesa —
no GPU or QGIS required. To reclaim the cache: `docker volume rm
bambi-qgis_bambi-test-data`.

By default the pipeline outputs (extracted frames, geo-referenced GeoTIFFs,
orthomosaic, ALFS renders) live in an ephemeral tmp dir and vanish with the
container. To keep them for manual/visual inspection, set
`BAMBI_TEST_KEEP_OUTPUT=1` — they are written under
`reports/integration_output/pipeline_out/` on the host (the repo is
bind-mounted at `/workspace`):

```bash
docker compose run --rm -e BAMBI_TEST_KEEP_OUTPUT=1 integration
```

Point `BAMBI_TEST_OUTPUT_DIR` at a path to keep them somewhere else. The
folder is wiped clean at the start of each kept run (so stale outputs never
mask a regression) but is not deleted afterwards.

### Real-QGIS smoke tests

`tests_qgis/` constructs the plugin's Qt widgets inside a headless QGIS
(`Dockerfile.qgis`, `QT_QPA_PLATFORM=offscreen`): the dock widget and every
dialog are instantiated, and the project configuration is round-tripped
through a real `QgsProject` (save → reset → load). This catches the
renamed-attribute / dead-signal breakage the QGIS-free unit suite cannot
see. The image is based on `qgis/qgis` (multi-GB); the tests need no GPU,
no plugin data, and no heavy pipeline dependencies.

```bash
docker compose run --rm qgis-tests
```

### Linting & security checks

Bandit, detect-secrets, and flake8 run in their own image (`Dockerfile`;
the unit tests use `Dockerfile.unit`, the integration tests
`Dockerfile.integration`):

```bash
docker compose run --rm checks
```

### Qt6 / QGIS 4 compatibility check

The same `pyqt5_to_pyqt6.py` dry run that plugins.qgis.org applies to every
uploaded plugin version, using the official
[`pyqgis4-checker`](https://github.com/qgis/pyqgis4-checker) image:

```bash
docker compose run --rm qt6-check
```

## Citation

If you use this plugin in your research, please cite:

```bibtex
@Article{praschlGeoReferencedTracking,
    author  = {Praschl, Christoph and Coucke, Vincent and Maschek, Anna and Schedl, David},
    title   = {Stay Local or Go Global: Geo-Referenced Bounding Boxes for
               Tracking Wildlife in Thermal Drone Videos},
}
```

## License & Credits

Released under the **MIT License**; see [LICENSE](LICENSE).

**Authors:** Christoph Praschl, Anna Maschek, David C. Schedl
**Organization:** University of Applied Sciences Upper Austria, Hagenberg Campus

Built on the [BAMBI Detection Framework](https://github.com/bambi-eco/bambi_detection), [ALFS-PY](https://github.com/bambi-eco/alfs_py), [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), and optionally [BoxMOT](https://github.com/mikel-brostrom/boxmot), [Geo-Referenced Tracking](https://github.com/bambi-eco/Geo-Referenced-Tracking), the [DJI Thermal SDK](https://www.dji.com/at/downloads/softwares/dji-thermal-sdk), [Fiona](https://github.com/Toblerity/Fiona), and [simplekml](https://simplekml.readthedocs.io/).

Contributions are welcome; please submit issues or pull requests.

![BAMBI Logo](bambi_wildlife_detection/icons/icon.png)
