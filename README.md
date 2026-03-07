# BAMBI QGIS Plugin

[![QGIS](https://img.shields.io/badge/QGIS-3.22+-green.svg)](https://qgis.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

A comprehensive QGIS plugin for detecting and tracking wildlife in aerial drone recordings — supporting both **video** and **still photo** workflows — with full geo-referencing, flight route visualization, and perpendicular distance sampling.

![Plugin Overview](images/plugin_overview.png)

---

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Processing Pipeline](#processing-pipeline)
- [Configuration](#configuration)
- [Input File Formats](#input-file-formats)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Credits](#credits)

---

## Features

- **Dual Input Mode**: Process drone **videos** (with SRT timestamps) or **still photo collections** from a single mission
- **Frame Extraction**: Extract and undistort thermal and RGB frames; camera calibration applied automatically
- **Flight Route Visualization**: Generate the drone's GPS flight path from the AirData log as a vector line layer, plus per-frame camera position markers
- **Animal Detection**: YOLO-based wildlife detection with automatic model download from HuggingFace
- **Prompt-based Segmentation**: Object segmentation using Roboflow's SAM3 API
- **Geo-referencing**: Project detections and segmentations onto a Digital Elevation Model (DEM) for real-world UTM coordinates
- **Perpendicular Distance Sampling**: For each geo-referenced detection (or the last bounding box of each track), automatically find the nearest point on the flight route and measure the perpendicular distance
- **Multi-Object Tracking**: Track animals across frames with multiple backend options (built-in, BoxMOT, Geo-Referenced Tracking)
- **Field of View Calculation**: Calculate camera footprints for each frame with optional custom mask support
- **Orthomosaic Generation**: Create georeferenced orthomosaics by projecting all frames onto the DEM
- **GeoTIFF Export**: Export individual frames as georeferenced GeoTIFFs for detailed analysis
- **DEM Import**: Automatically download DEMs (currently limited to Austria), or convert any GeoTIFF DEM to the required GLB format — including source CRS override for files with incorrect embedded CRS metadata
- **Full QGIS Integration**: All outputs are automatically added as styled layers to the QGIS layer panel

---

## Screenshots

### Plugin Interface

![Plugin Overview](images/plugin_overview.png)

![Configuration Tab](images/config_tab.png)

![Processing Tab](images/processing_tab.png)

#### Video Input

![Input Tab](images/input_tab_video.png)

#### Photo Input

![Input Tab](images/input_tab.png)


### Flight Route and Camera Positions

#### Video Route

![Flight Route](images/route.png)

#### Photo Route

![Flight Route](images/route_photo.png)


### Detections and Geo-referencing

![Detection Results](images/detection_results.png)

#### Perpendicular distances

![Perpendicular Distances](/images/perpendicular_distances.png)

### Track Visualization

![Track Visualization](images/tracks_qgis.png)

### Field of View

![Field of View per Frame](images/fov.png)

![Merged Flight FoV](images/flight_fov.png)

### Orthomosaic and GeoTIFF Export

![Orthomosaic Output](images/orthomosaic.png)

![GeoTIFF Export](images/geotiff.png)

### Segmentation

![Segmentation Output](images/segmentations.png)

---

## Requirements

### Data Requirements

- The plugin is designed for DJI Enterprise drones (M30T, M3T, M4T, M300, etc.) with thermal and/or RGB cameras
- GPS RTK recordings are recommended for best geo-referencing accuracy
- **Video mode**: Requires SRT subtitle files (frame timestamps) and an AirData flight log CSV
- **Photo mode**: Requires a directory of still images and an AirData flight log CSV for GPS positions; SRT files are not needed
- Camera calibration (intrinsic parameters) must be provided for lens undistortion

### System Requirements

- **QGIS**: Version 3.22 or higher
- **Python**: Version 3.9 or higher (included with QGIS)
- **Operating System**: Windows, Linux, or macOS

### Required Python Packages

The plugin requires the **BAMBI Detection Framework** and the **ALFS-PY** framework. Install them using pip within the **OSGeo4W Shell** (Windows) or your QGIS Python environment:

```bash
# Install BAMBI Detection Framework (required)
pip install git+https://github.com/bambi-eco/bambi_detection.git

# Install ALFS-PY Framework (required)
pip install git+https://github.com/bambi-eco/alfs_py.git
```

### Optional: Extended Tracking Capabilities

The plugin includes simple geo-based tracking strategies out of the box. For advanced tracking algorithms, install the following packages in the **QGIS OSGeo4W Shell**:

![Tracker options](images/trackers.png)

#### BoxMOT Trackers

For state-of-the-art multi-object tracking algorithms (DeepOCSORT, BoTSORT, StrongSORT, ByteTrack, etc.):

```bash
pip install boxmot
```

Or install from source: [https://github.com/mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)

#### Geo-Referenced Tracking Extension

For tracking algorithms that operate natively in geo-referenced coordinates (recommended for wildlife surveys):

```bash
pip install git+https://github.com/bambi-eco/Geo-Referenced-Tracking.git
```

Or visit: [https://github.com/bambi-eco/Geo-Referenced-Tracking](https://github.com/bambi-eco/Geo-Referenced-Tracking)

---

## Installation

### Method 1: Install from ZIP (Recommended)

1. Download the plugin by either
    - Getting the zipped plugin from a [released version](https://github.com/bambi-eco/Bambi-QGIS/releases) (recommended)
    - Or downloading this repository and zipping the `bambi_wildlife_detection` subfolder for the current development version
2. Open QGIS
3. Go to **Plugins** → **Manage and Install Plugins...**
4. Select the **Install from ZIP** tab
5. Browse to the downloaded ZIP file
6. Click **Install Plugin**

![Install from ZIP](images/install_zip.png)

### Method 2: Manual Installation

1. Download and extract the plugin folder
2. Copy the `bambi_wildlife_detection` folder to your QGIS plugins directory:

   | Platform | Path |
   |----------|------|
   | **Windows** | `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\` |
   | **Linux** | `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` |
   | **macOS** | `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/` |

3. Restart QGIS
4. Enable the plugin via **Plugins** → **Manage and Install Plugins...**

---

## Usage

### Quick Start

1. Click the **BAMBI** icon in the toolbar to open the dock widget
2. Select your **input mode** (Video or Photo) in the **Input** tab
3. Configure inputs and the target folder
4. Adjust parameters in the **Configuration** tab
5. Run processing steps sequentially in the **Processing** tab

### Input Modes

#### Video Mode

Used when the drone recorded continuous video (thermal and/or RGB). Each camera requires:

| File Type   | Extension | Description                              |
|-------------|-----------|------------------------------------------|
| Video files | `.MP4`    | Drone thermal and/or RGB video recordings |
| SRT files   | `.SRT`    | Subtitle files with per-frame timestamps |
| Calibration | `.json`   | Camera intrinsic parameters              |

#### Photo Mode

Used when the drone captured a series of still images (e.g. single-shot mapping flights). Each camera requires:

| File Type        | Extension | Description                               |
|------------------|-----------|-------------------------------------------|
| Photo directory  | folder    | Directory containing the still images     |
| Calibration      | `.json`   | Camera intrinsic parameters               |

In photo mode, GPS positions are matched to images via timestamps in the AirData log and the image EXIF data. SRT files are not required.

### Common Inputs (Both Modes)

| File Type     | Extension         | Description                                                                                                                                                             |
|---------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AirData CSV   | `.csv`            | Flight log exported from AirData (GPS positions, altitude, timestamps)                                                                                                  |
| Calibration   | `.json`           | Camera intrinsic parameters for undistortion                                                                                                                            |
| Correction    | `.json`           | JSON file with positional/rotational corrections (can also be adjusted interactively in the plugin)                                                                     |
| DEM           | `.gltf` or `.glb` | Digital Elevation Model + metadata JSON. Either provide manually, convert a GeoTIFF, or download automatically (Austria only)                                           |
| Target CRS    | EPSG code         | UTM-based CRS for the output (e.g. `EPSG:32633` for UTM Zone 33N)                                                                                                      |
| Target Folder | folder path       | Output directory for all generated files                                                                                                                                |

![Input Tab](images/input_tab.png)

---

## Configuration

Before starting processing, configure per-step settings in the **Configuration** tab:

- **Frame sampling**: Start/end frame, sample rate
- **Detection**: Confidence threshold, model path
- **Tracking**: Backend selection, IoU threshold, interpolation
- **Orthomosaic**: Resolution, tile size
- **Correction factors**: Translation and rotation offsets for geo-referencing
- **Flight route**: Frame marker interval, distance marker interval
- **Field of View**: Custom mask, simplification

![Configuration Tab](images/config_tab.png)

---

## Processing Pipeline

The plugin provides a sequential processing pipeline. Execute steps in order using the **Processing** tab.

![Processing Tab](images/processing_tab.png)

---

### Step 1: Extract Frames

Extracts, undistorts, and geo-tags frames from the input data.

- **Video mode**: Decodes frames from thermal/RGB video files; timestamps from SRT logs are matched with AirData GPS positions to compute camera poses
- **Photo mode**: Reads still images from the specified directories; image timestamps (EXIF) are matched with AirData GPS positions to compute camera poses

**Outputs:**
```
frames_t/          # Undistorted thermal frames
frames_w/          # Undistorted RGB frames
poses_t.json       # Camera pose for every thermal frame
poses_w.json       # Camera pose for every RGB frame
mask_T.png         # Undistortion mask (thermal)
mask_W.png         # Undistortion mask (RGB)
```

> **Note:** For large datasets or long videos this step may take several minutes. Progress is updated between frames.

---

### Step 2: Generate Flight Route

Builds two complementary vector layers from the mission data:

- **Flight route line**: GPS positions recorded in the AirData CSV are projected to the target UTM CRS and connected as a `LineString`. This represents the true GPS-recorded flight path.
- **Camera position points**: The position of the drone at every extracted frame (from `poses.json`) is added as a separate point layer — these are the positions at which images were actually captured.

**Outputs:**
```
flight_route/
├── flight_route.geojson       # GPS-based flight path (LineString)
└── camera_positions.geojson   # Per-frame camera positions (Points)
```

Use **→ Add Flight Route to QGIS** to load both layers with styling applied.

![Flight Route](images/route.png)

---

### Step 3: Detect Animals

Runs YOLO-based detection on all extracted frames. The default thermal wildlife detection model is downloaded automatically from HuggingFace on first use.

**Outputs:**
```
detections/
└── detections.txt    # Bounding box detections (frame, x1, y1, x2, y2, confidence, class)
```

---

### Step 4: Geo-Reference Detections

Projects pixel-space bounding boxes to real-world UTM coordinates by ray-casting against the DEM mesh. Each detection's four corners are projected and the result is stored as a world-space bounding box.

**Outputs:**
```
georeferenced/
└── georeferenced.txt    # Detections with UTM bounding box coordinates
```

Use **→ Add Detections to QGIS** to load per-frame detection layers.

![Detection Results](images/detection_results.png)

---

### → Calculate Perpendicular (Detection Distances)

For each geo-referenced detection, computes the **perpendicular distance** to the flight route:

1. The 2D center of each detection's bounding box is computed
2. The nearest point on the AirData GPS `LineString` is found by projecting onto each segment
3. The Euclidean distance between the detection center and the foot point is recorded

This is particularly useful for **transect-based wildlife surveys**, where the perpendicular offset from the flight line is a key sampling variable.

**Outputs:**
```
flight_route/
├── perpendicular.json             # Flat list (used by QGIS layer)
└── perpendicular_by_image.json    # Per-image keyed results
```

`perpendicular_by_image.json` structure:
```json
{
  "frame_000042.jpg": {
    "0": {
      "center": [UTM_x, UTM_y, altitude],
      "perpendicular": [foot_x, foot_y, altitude],
      "distance": 12.34
    }
  }
}
```

Use **→ Add Perpendicular Lines to QGIS** to visualize the connections as line features with `distance_m` attribute.

![Perpendicular Distances](/images/perpendicular_distances.png)

---

### Step 5: Track Animals

Associates detections across frames into continuous tracks using the selected tracking backend.

- Tracks with a single detection (no movement) are fully supported and appear as a bounding box without a movement line
- Multi-detection tracks show both the movement path and the final bounding box

**Outputs:**
```
tracks/
├── tracks_pixel.csv    # Tracks in pixel coordinates
└── tracks.csv          # Geo-referenced tracks (UTM)
```

Use **→ Add Tracks to QGIS** to load each track as a grouped layer (movement path + final bounding box).

![Track Visualization](images/tracks_qgis.png)

---

### → Calculate Track Perpendicular (Track Distances)

For each track, takes the **last bounding box** (the animal's final recorded position) and calculates the perpendicular distance to the flight route, using the same method as the detection perpendicular step.

**Outputs:**
```
flight_route/
├── perpendicular_tracks.json             # Flat list (used by QGIS layer)
└── perpendicular_tracks_by_track.json    # Per-track keyed results
```

`perpendicular_tracks_by_track.json` structure:
```json
{
  "42": {
    "last_frame": 123,
    "last_image": "frame_000123.jpg",
    "center": [UTM_x, UTM_y, altitude],
    "perpendicular": [foot_x, foot_y, altitude],
    "distance": 35.7
  }
}
```

Use **→ Add Track Perpendicular Lines to QGIS** to visualize as line features.

![Perpendicular Distances](/images/perpendicular_distances.png)

---

### Step 6: Calculate Field of View

Computes the camera footprint polygon on the ground for each frame (or a subset), using the DEM for accurate terrain-following projection.

**Outputs:**
```
fov/
├── fov_polygons.txt      # Per-frame FoV polygon coordinates
└── merged_fov.geojson    # Union of all footprints (coverage area)
```

![Field of View per Frame](images/fov.png)

![Merged Flight FoV](images/flight_fov.png)

---

### Step 7: Generate Orthomosaic

Creates a georeferenced orthomosaic by projecting all (or a subset of) frames onto the DEM surface and blending them together. Output is a Cloud-Optimized GeoTIFF.

**Outputs:**
```
orthomosaic/
└── orthomosaic.tif    # Georeferenced mosaic (COG GeoTIFF)
```

![Orthomosaic Output](images/orthomosaic.png)

---

### Step 8: Export Frames as GeoTIFF

Exports individual frames as georeferenced GeoTIFFs, suitable for import into GIS tools or for detailed per-frame analysis.

**Outputs:**
```
geotiffs/
├── frame_000000.tiff
└── ...
```

![GeoTIFF Export](images/geotiff.png)

---

### Step 9: Object Segmentation

Segments individual detected objects from aerial images using Roboflow's SAM3 API. Recommended for RGB imagery.

**Outputs:**
```
segmentation/
└── segmentation_pixel.json    # Pixel-space segmentation masks
```

---

### Step 10: Geo-Reference Object Segmentation

Projects SAM3 pixel-space segmentation masks to world coordinates using the DEM.

**Outputs:**
```
segmentation/
└── segmentation_georef.json    # UTM-coordinate segmentation polygons
```

![Segmentation Output](images/segmentations.png)

---

## DEM Import and Conversion

The plugin supports three ways to provide a DEM:

### 1. Automatic Download (Austria only)

Use the **DEM** tab to download and tile Austrian elevation data from the BEV ATOM service automatically.

### 2. Convert a GeoTIFF DEM

Any GeoTIFF DEM can be converted to the required GLB + metadata JSON format using the **GeoTIFF Input** panel in the DEM tab:

- **Input file**: Path to the GeoTIFF (`.tif` / `.tiff`)
- **Output folder**: Where the `.glb` and `.json` files will be saved
- **Output CRS**: Target UTM CRS for the mesh (e.g. `EPSG:32633`)
- **Simplification**: Mesh vertex reduction factor (1 = full resolution)
- **Source CRS** *(optional)*: Override the CRS embedded in the file. Use this when the GeoTIFF has incorrect CRS metadata — for example, a file containing SWEREF99TM (EPSG:3006) data but tagged as EPSG:32634. Leave empty to auto-detect.

![GeoTiff Conversion](/images/geotiff_conversion.png)

### 3. Manual DEM

Provide a `.gltf` or `.glb` file and its companion `.json` metadata file directly.

---

## Input File Formats

### Calibration JSON

```json
{
    "mtx": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist": [k1, k2, p1, p2, k3]
}
```

### DEM Metadata JSON

```json
{
    "origin": [UTM_x, UTM_y, UTM_z],
    "origin_wgs84": {
        "latitude": 47.xxx,
        "longitude": 14.xxx,
        "altitude": 500.0
    }
}
```

### Correction JSON (Optional)

```json
{
    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
    "additional": [
        {
            "start": 10,
            "end": 100,
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}
        }
    ]
}
```

---

## Output Structure

A complete run produces the following folder structure:

```
target_folder/
├── frames_t/                                  # Extracted/undistorted thermal frames
│   ├── frame_000000.jpg
│   └── ...
├── frames_w/                                  # Extracted/undistorted RGB frames
│   ├── frame_000000.jpg
│   └── ...
├── poses_t.json                               # Camera pose per thermal frame
├── poses_w.json                               # Camera pose per RGB frame
├── mask_T.png                                 # Thermal undistortion mask
├── mask_W.png                                 # RGB undistortion mask
├── flight_route/
│   ├── flight_route.geojson                   # AirData GPS flight path (LineString)
│   ├── camera_positions.geojson               # Per-frame camera positions (Points)
│   ├── perpendicular.json                     # Detection perpendicular distances (flat)
│   ├── perpendicular_by_image.json            # Detection perpendicular distances (by image)
│   ├── perpendicular_tracks.json              # Track perpendicular distances (flat)
│   └── perpendicular_tracks_by_track.json     # Track perpendicular distances (by track)
├── detections/
│   └── detections.txt                         # Raw YOLO detections
├── georeferenced/
│   └── georeferenced.txt                      # Geo-referenced detections (UTM)
├── tracks/
│   ├── tracks_pixel.csv                       # Tracks in pixel coordinates
│   └── tracks.csv                             # Geo-referenced tracks (UTM)
├── fov/
│   ├── fov_polygons.txt                       # Per-frame FoV polygon coordinates
│   └── merged_fov.geojson                     # Combined coverage area
├── orthomosaic/
│   └── orthomosaic.tif                        # Georeferenced mosaic (COG GeoTIFF)
├── geotiffs/
│   ├── frame_000000.tiff
│   └── ...
├── segmentation/
│   ├── segmentation_pixel.json                # Pixel-space segmentation masks
│   └── segmentation_georef.json               # Geo-referenced segmentation polygons
└── models/
    ├── thermal_animal_detector.pt             # BAMBI thermal detection model
    ├── osnet_x0_5_bambi_thermal_omni.pt       # BAMBI Re-ID model
    └── osnet_x0_25_msmt17.pt                  # BoxMOT default Re-ID model
```

---

## Troubleshooting

### Model Download Fails

- Check your internet connection
- Download manually from HuggingFace: [Detection (thermal_animal_detector.pt)](https://huggingface.co/cpraschl/bambi-thermal-detection), [Re-ID (osnet_x0_5_bambi_thermal_omni.pt)](https://huggingface.co/cpraschl/bambi-thermal-omni)
- Place the `.pt` file in `target_folder/models/`

### Geo-referencing Issues

- Ensure the DEM covers your entire flight area
- Verify the DEM metadata JSON has correct origin coordinates
- Check that the target CRS matches your DEM projection

### GeoTIFF Conversion: Wrong Origin Location

- If the converted DEM places the origin in the wrong country, the GeoTIFF likely has incorrect CRS metadata
- Enter the correct CRS in the **Source CRS** field (e.g. `EPSG:3006` for SWEREF99TM) before converting
- Verify the correct CRS by loading the file in QGIS and checking its reported coordinate system

### Photo Mode: Images Not Matched to GPS

- Ensure image timestamps (EXIF) are in UTC or that the correct timezone offset is configured in the **Configuration** tab
- Verify the AirData CSV covers the same time range as the photo session

### No Tracks Generated

- Lower the detection confidence threshold
- Adjust the IoU threshold for tracking
- Verify that detections were successfully generated in Step 3

### Perpendicular Distances Not Calculated

- Run **Generate Flight Route** before **Calculate Perpendicular** — the AirData GPS `LineString` is required
- Ensure the AirData CSV contains enough GPS records (at least 2 valid positions)
- For track perpendicular: ensure **Track Animals** has been completed first

### BoxMOT Import Errors

- Ensure you're installing in the correct Python environment (OSGeo4W Shell)
- Try: `pip install --upgrade boxmot`
- Check for conflicting package versions

### Memory Issues with Large Datasets

- Reduce the number of frames processed at each step; back up intermediate results and rerun with different settings
- For orthomosaics and GeoTIFFs: reduce resolution or decrease the max tile size parameter

### Problems Installing Dependencies

If you are installing dependencies using the commands above, Git must be installed and available in the OSGeo4W Shell's PATH. On Windows, edit `OSGeo4W.bat` to add:

```text
@echo off
call "%~dp0\bin\o4w_env.bat"
set "PATH=C:\Users\<username>\AppData\Local\Programs\Git\cmd\;%PATH%"
@echo on
@if [%1]==[] (echo run o-help for a list of available commands & cd /d "%~dp0" & cmd.exe /k) else (cmd /c "%*")
```

Alternatively, download the repositories and install from local paths:

```shell
pip install <path>/alfs_py
pip install <path>/bambi_detection
```

---

## Citation

If you use this plugin in your research, please cite:

```bibtex
@Article{praschlGeoReferencedTracking,
    author  = {Praschl, Christoph and Coucke, Vincent and Maschek, Anna and Schedl, David},
    title   = {Stay Local or Go Global: Geo-Referenced Bounding Boxes for
               Tracking Wildlife in Thermal Drone Videos},
}
```

---

## License

This plugin is released under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

## Credits

**Author:** Christoph Praschl, Anna Maschek, David C. Schedl
**Organization:** University of Applied Sciences Upper Austria, Hagenberg Campus

### Dependencies

- [ALFS-PY Framework](https://github.com/bambi-eco/alfs_py)
- [BAMBI Detection Framework](https://github.com/bambi-eco/bambi_detection)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) (optional)
- [Geo-Referenced Tracking](https://github.com/bambi-eco/Geo-Referenced-Tracking) (optional)

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

![BAMBI Logo](bambi_wildlife_detection/icons/icon.png)
