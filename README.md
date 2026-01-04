# BAMBI Wildlife Detection QGIS Plugin

[![QGIS](https://img.shields.io/badge/QGIS-3.22+-green.svg)](https://qgis.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

A comprehensive QGIS plugin for detecting and tracking wildlife in drone thermal videos with full geo-referencing support.

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
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Credits](#credits)

---

## Features

- **Frame Extraction** — Extract frames from drone thermal videos with configurable sample rate
- **Flight Route Visualization** — Generate and display the drone flight path as a polyline layer
- **Animal Detection** — YOLO-based wildlife detection with automatic model download from HuggingFace
- **Geo-referencing** — Project detections onto a Digital Elevation Model (DEM) for real-world coordinates
- **Multi-Object Tracking** — Track animals across frames with multiple backend options
- **Field of View Calculation** — Calculate camera footprints for each frame with custom mask support
- **Orthomosaic Generation** — Create georeferenced orthomosaic from extracted frames
- **GeoTIFF Export** — Export individual frames as georeferenced GeoTIFFs
- **Full QGIS Integration** — Automatically add all outputs as styled layers to QGIS

---

## Screenshots

### Plugin Interface

![Input Tab](images/input_tab.png)

![Configuration Tab](images/config_tab.png)

![Processing Tab](images/processing_tab.png)

### Output Examples

![Detection Results](images/detection_results.png)
*<!-- TODO: Add screenshot showing detection bounding boxes on thermal image -->*

![Track Visualization](images/tracks_qgis.png)
*<!-- TODO: Add screenshot of animal tracks displayed in QGIS -->*

![Orthomosaic Output](images/orthomosaic.png)

---

## Requirements

### Data Requirements

- The plugin is intended to be used with DJI Enterprise drones like the DJI M30T, M3T, M4T, M300, ...
- It is recommended to use GPS RTK for recordings
- The plugin needs DJI's SRT logs as well as the flight logs e.g. exported via AirData
- The cameras need to be calibrated and the distortion parameters need to be provided

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

Additional dependencies (typically installed automatically):

```bash
pip install ultralytics          # YOLO detection
pip install huggingface_hub      # Model downloads
pip install pyproj               # Coordinate transformations
pip install pyrr                 # 3D math utilities
pip install trimesh              # 3D mesh handling
pip install scipy                # Scientific computing
pip install rasterio             # GeoTIFF I/O
pip install moderngl             # GPU-accelerated rendering
```

### Optional: Extended Tracking Capabilities

The plugin includes simple geo-based tracking strategies out of the box. For advanced tracking algorithms, you can optionally install additional packages within the **QGIS OSGeo4W Shell**:

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

1. Download the repository and zip the "bambi_wlidlife_detection" subfolder
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
2. Configure inputs in the **Input** tab
3. Adjust parameters in the **Configuration** tab
4. Run processing steps sequentially in the **Processing** tab

### Input Data Requirements

You need the following files for a complete processing workflow:

| File Type | Extension | Description |
|-----------|-----------|-------------|
| Video Files | `.MP4` | Drone thermal video recordings |
| SRT Files | `.SRT` | Subtitle files with frame timestamps |
| AirData CSV | `.csv` | Flight log exported from AirData |
| DEM (GLTF) | `.gltf` | Digital Elevation Model with metadata JSON |
| Calibration | `.json` | Camera intrinsic parameters |

---

## Processing Pipeline

The plugin provides an 8-step processing pipeline. Execute steps in order using the **Processing** tab

![Processing Tab](images/processing_tab.png)

### Step 1: Extract Frames

Extracts and undistorts frames from drone videos based on the configured sample rate.

**Outputs:** `frames/` folder with JPG images and `poses.json`

### Step 2: Generate Flight Route

Creates a polyline showing the drone's flight path from extracted camera positions.

**Outputs:** `flight_route.geojson` — can be added to QGIS as a vector layer

### Step 3: Detect Animals

Runs YOLO-based detection on all extracted frames. The default model is automatically downloaded from HuggingFace.

**Outputs:** `detections/detections.txt`

### Step 4: Geo-Reference Detections

Projects pixel-space detections to real-world coordinates using the DEM.

**Outputs:** `georeferenced/georeferenced.txt` with world coordinates

### Step 5: Track Animals

Associates detections across frames into continuous tracks.

**Outputs:** `tracks/tracks.csv` with track IDs and trajectories

### Step 6: Calculate Field of View

Computes the camera footprint polygon for each frame.

**Outputs:** `fov/` folder with per-frame and merged FoV polygons

### Step 7: Generate Orthomosaic

Creates a georeferenced orthomosaic by projecting all frames onto the DEM.

**Outputs:** `orthomosaic/orthomosaic.tif` (Cloud-Optimized GeoTIFF)

### Step 8: Export Frames as GeoTIFF

Exports individual frames as georeferenced GeoTIFFs for detailed analysis.

**Outputs:** `geotiffs/` folder with per-frame GeoTIFFs

---

## Configuration

### Detection Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| Model Path | Custom YOLO model (auto-downloads if empty) | HuggingFace default |
| Confidence | Detection confidence threshold | 0.25 |
| Camera | Camera selector (for dual-camera drones) | T (thermal) |

### Tracking Backend Selection

The plugin supports multiple tracking backends with different capabilities:

| Backend | Description | Requirements |
|---------|-------------|--------------|
| **Built-in** | Simple geo-based IoU tracker | None (included) |
| **BoxMOT** | DeepOCSORT, BoTSORT, StrongSORT, etc. | `pip install boxmot` |
| **GeoRef Native** | Full tracking in geo-coordinates | Geo-Referenced-Tracking |
| **GeoRef Hybrid** | Standard tracking + geo-recovery | Geo-Referenced-Tracking |

### Built-in Tracker Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| IoU Threshold | Intersection over Union for matching | 0.3 |
| Max Age | Frames without detection before track ends (-1 = unlimited) | -1 |
| Max Center Distance | Maximum center distance in meters | 0.2 |
| Tracker Mode | GREEDY, HUNGARIAN, CENTER, HUNGARIAN_CENTER | HUNGARIAN |
| Class Aware | Only match same-class detections | True |
| Interpolate | Fill gaps with interpolated positions | True |

### ReID Model Selection (BoxMOT/GeoRef)

| Model | Description |
|-------|-------------|
| **osnet** | Standard OSNet model (general purpose) |
| **bambi** | Wildlife-specific ReID from HuggingFace |
| **custom** | Your own ReID weights file |

### Orthomosaic Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| Ground Resolution | Meters per pixel | 0.05 |
| Blend Mode | INTEGRAL, FIRST, LAST, CENTER | INTEGRAL |
| Frame Range | Subset of frames to include | All |
| Crop to Content | Remove empty borders | True |
| Create Overviews | Build pyramid levels | True |

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
    "origin": [x_offset, y_offset, z_offset],
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
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}
}
```

---

## Troubleshooting

### Model Download Fails

- Check your internet connection
- Download manually from [HuggingFace](https://huggingface.co/cpraschl/bambi-thermal-detection)
- Place the `.pt` file in `target_folder/models/`

### Geo-referencing Issues

- Ensure the DEM covers your entire flight area
- Verify the DEM metadata JSON has correct origin coordinates
- Check that the target CRS matches your DEM projection

### No Tracks Generated

- Lower the detection confidence threshold
- Adjust the IoU threshold for tracking
- Verify that detections were successfully generated in Step 3

### BoxMOT Import Errors

- Ensure you're installing in the correct Python environment (OSGeo4W Shell)
- Try: `pip install --upgrade boxmot`
- Check for conflicting package versions

### Memory Issues with Large Datasets

- Reduce the orthomosaic resolution
- Use a smaller frame range
- Decrease the max tile size parameter

---

## Project Structure

```
target_folder/
├── frames/                    # Extracted video frames
│   ├── frame_000000.jpg
│   ├── frame_000010.jpg
│   └── ...
├── poses.json                 # Camera poses for each frame
├── flight_route.geojson       # Drone flight path
├── detections/
│   └── detections.txt         # Raw YOLO detections
├── georeferenced/
│   └── georeferenced.txt      # Geo-referenced detections
├── tracks/
│   └── tracks.csv             # Animal tracks with IDs
├── fov/
│   ├── fov_polygons.geojson   # Per-frame FoV polygons
│   └── merged_fov.geojson     # Combined coverage area
├── orthomosaic/
│   └── orthomosaic.tif        # Georeferenced mosaic
├── geotiffs/
│   ├── frame_000000.tiff      # Per-frame GeoTIFFs
│   └── ...
└── models/
    └── thermal_animal_detector.pt
```

---
## Citation

If you use this framework in your research, please cite:

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

- [BAMBI Detection Framework](https://github.com/bambi-eco/bambi_detection)
- [ALFS-PY Framework](https://github.com/bambi-eco/alfs_py)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) (optional)
- [Geo-Referenced Tracking](https://github.com/bambi-eco/Geo-Referenced-Tracking) (optional)

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

![BAMBI Logo](bambi_wildlife_detection/icons/icon.png)
