# Processing Pipeline

This guide covers the main workflow of the plugin: configuring inputs, running the processing steps in order, and understanding the outputs.

- [Input](#input)
- [Configuration](#configuration)
- [Processing steps](#processing-steps)
- [Input file formats](#input-file-formats)
- [Output structure](#output-structure)

## Input

Open the dock widget via the **BAMBI** toolbar icon and select the **input mode** (Video or Photo) in the **Input** tab.

### Video mode

Used when the drone recorded continuous video (thermal and/or RGB). Each camera requires:

| File Type   | Extension | Description                              |
|-------------|-----------|------------------------------------------|
| Video files | `.MP4`    | Drone thermal and/or RGB video recordings |
| SRT files   | `.SRT`    | Subtitle files with per-frame timestamps |
| Calibration | `.json`   | Camera intrinsic parameters              |

![Video Input](../images/input_tab_video.png)

### Photo mode

Used when the drone captured a series of still images (e.g. single-shot mapping flights). Each camera requires:

| File Type        | Extension | Description                               |
|------------------|-----------|-------------------------------------------|
| Photo directory  | folder    | Directory containing the still images     |
| Calibration      | `.json`   | Camera intrinsic parameters               |

In photo mode, GPS positions are matched to images via timestamps in the AirData log and the image EXIF data. SRT files are not required.

![Photo Input](../images/input_tab.png)

### Common inputs (both modes)

| Input         | Format            | Description |
|---------------|-------------------|-------------|
| AirData CSV   | `.csv`            | Flight log exported from AirData (GPS positions, altitude, timestamps) |
| Calibration   | `.json`           | Camera intrinsic parameters for undistortion — can be created with the [Camera Calibration Wizard](camera-calibration.md) |
| Correction    | `.json`           | Positional/rotational corrections — can be found interactively with the [Correction Wizard](correction-wizard.md) |
| DEM           | `.gltf` / `.glb`  | Digital Elevation Model + metadata JSON. Provide manually, convert a GeoTIFF, download automatically (Austria), or generate a flat surface mesh — see [DEM Import & Conversion](dem.md) |
| Target CRS    | EPSG code         | UTM-based CRS for the output (e.g. `EPSG:32633` for UTM Zone 33N). The **Auto** button derives a suitable UTM zone from the flight data |
| Target folder | folder path       | Output directory for all generated files |

## Configuration

Before starting processing, configure per-step settings in the **Configuration** tab:

![Configuration Tab](../images/config_tab.png)

- **Extraction**: Frame skip, limit, sampling rate, and thermal visualisation (see below)
- **Detection**: Confidence threshold, model path
- **Tracking**: Backend selection, IoU threshold, interpolation, TRex tracklet import
- **SAM3**: Roboflow API key for segmentation
- **ALFS**: Resolution, tile size
- **Correction factors**: Translation and rotation offsets for geo-referencing
- **Flight route**: Frame marker interval, distance marker interval
- **Field of View**: Custom mask, simplification

### Extraction

The **Extraction** sub-tab provides fine-grained control over which frames are processed and how thermal images are rendered.

![Extraction Configuration](../images/config_extraction.png)

| Setting | Description |
|---------|-------------|
| **Skip first** | Skip this many frames/images at the start before processing begins. Useful for ignoring take-off frames. |
| **Limit** | Cap the total number of frames processed (enable the checkbox to activate). |
| **Sampling rate** *(video only)* | Take every N-th frame (e.g. 5 = every 5th frame). Reduces processing time for long recordings. |

**Thermal visualisation** (requires the DJI Thermal SDK — install via the [Dependency Manager](installation.md#dependency-manager); the group is greyed out when the SDK is not detected):

| Setting | Description |
|---------|-------------|
| **Colormap** | Apply a false-colour map to the exported thermal frames (e.g. `plasma`, `inferno`, `jet`, `white-hotspot`). Choose `(none)` to keep raw 8-bit grey values. |
| **Lower threshold** | Pixels below this temperature (°C) are rendered black. Enable the checkbox to activate. |
| **Upper threshold** | Pixels above this temperature (°C) are rendered black. Enable the checkbox to activate. |

### TRex tracklet import

If you have already tracked animals externally with [TRex](https://trex.run), the plugin can import those tracklets instead of running its own tracker. In the **Tracking** sub-tab, set the **NPZ Folder** to the directory containing the TRex `*.npz` tracklet files.

- Check **Labels already in undistorted frame space** if TRex was run on already-undistorted BAMBI frames; leave it unchecked (default) if TRex was run on the original raw video
- When an NPZ folder is set, step **4. Track Animals Or Import** imports and geo-references the TRex tracklets instead of running the configured tracking backend

## Processing steps

Execute the steps in order using the **Processing** tab. The **→** buttons below each step load the corresponding results into QGIS as styled layers, or compute derived products.

![Processing Tab](../images/processing_tab.png)

### 1. Extract Frames

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

### 2. Generate Flight Route

Builds two complementary vector layers from the mission data:

- **Flight route line**: GPS positions recorded in the AirData CSV are projected to the target UTM CRS and connected as a `LineString`. This represents the true GPS-recorded flight path.
- **Camera position points**: The position of the drone at every extracted frame (from `poses.json`) is added as a separate point layer — these are the positions at which images were actually captured.

**Outputs:**
```
flight_route_t/    # or flight_route_w/ depending on camera selection
├── flight_route.geojson       # GPS-based flight path (LineString)
└── camera_positions.geojson   # Per-frame camera positions (Points)
```

Use **→ Add Flight Route to QGIS** to load both layers with styling applied.

![Flight Route](../images/route.png)

### 3. Detect Animals

Runs YOLO-based detection on all extracted frames. The default thermal wildlife detection model is downloaded automatically from HuggingFace on first use.

**Outputs:**
```
detections_t/    # or detections_w/ depending on camera selection
└── detections.txt    # Bounding box detections (frame, x1, y1, x2, y2, confidence, class)
```

#### → Geo-Reference Detections

Projects pixel-space bounding boxes to real-world UTM coordinates by ray-casting against the DEM mesh. Each detection's four corners are projected and the result is stored as a world-space bounding box.

**Outputs:**
```
georeferenced_t/    # or georeferenced_w/ — follows the detection camera selection
└── georeferenced.txt    # Detections with UTM bounding box coordinates
```

Use **→ Add Detections to QGIS** to load per-frame detection layers.

![Detection Results](../images/detection_results.png)

#### → Calculate Perpendicular

For each geo-referenced detection, computes the **perpendicular distance** to the flight route:

1. The 2D center of each detection's bounding box is computed
2. The nearest point on the AirData GPS `LineString` is found by projecting onto each segment
3. The Euclidean distance between the detection center and the foot point is recorded

This is particularly useful for **transect-based wildlife surveys**, where the perpendicular offset from the flight line is a key sampling variable.

**Outputs:**
```
flight_route_t/    # or flight_route_w/ — uses the flight route camera selection
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

Use **→ Add Perpendicular Lines to QGIS** to visualize the connections as line features with a `distance_m` attribute.

![Perpendicular Distances](../images/perpendicular_distances.png)

### 4. Track Animals Or Import

Associates detections across frames into continuous tracks using the selected tracking backend — or, if a TRex NPZ folder is configured (see [TRex tracklet import](#trex-tracklet-import)), imports and geo-references the pre-computed TRex tracklets instead.

- Tracks with a single detection (no movement) are fully supported and appear as a bounding box without a movement line
- Multi-detection tracks show both the movement path and the final bounding box

**Outputs:**
```
tracks_t/    # or tracks_w/ — follows the detection camera selection
├── tracks_pixel.csv    # Tracks in pixel coordinates
└── tracks.csv          # Geo-referenced tracks (UTM)
```

Use **→ Add Tracks to QGIS** to load each track as a grouped layer (movement path + final bounding box).

![Track Visualization](../images/tracks_qgis.png)

#### → Calculate Track Perpendicular

For each track, takes the **last bounding box** (the animal's final recorded position) and calculates the perpendicular distance to the flight route, using the same method as the detection perpendicular step.

**Outputs:**
```
flight_route_t/    # or flight_route_w/ — uses the flight route camera selection
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

### 5. Calculate Field of View

Computes the camera footprint polygon on the ground for each frame (or a subset), using the DEM for accurate terrain-following projection.

**Outputs:**
```
fov_t/    # or fov_w/ depending on camera selection
├── fov_polygons.txt      # Per-frame FoV polygon coordinates
└── merged_fov.geojson    # Union of all footprints (coverage area)
```

![Field of View per Frame](../images/fov.png)

![Merged Flight FoV](../images/flight_fov.png)

### 6. Generate ALFS

Creates a georeferenced Airborne Light Field Sampling (ALFS) by projecting all (or a subset of) frames onto the DEM surface and blending them together. Output is a Cloud-Optimized GeoTIFF.

**Outputs:**
```
alfs_t/    # or alfs_w/ depending on camera selection
└── alfs.tif    # Georeferenced mosaic (COG GeoTIFF)
```

![ALFS Output](../images/alfs.png)

### 7. Export Frames as GeoTIFF

Exports individual frames as georeferenced GeoTIFFs, suitable for import into GIS tools or for detailed per-frame analysis.

**Outputs:**
```
geotiffs_t/    # or geotiffs_w/ depending on camera selection
├── frame_000000.tiff
└── ...
```

![GeoTIFF Export](../images/geotiff.png)

### 8. Generate Orthomosaic

Builds a **true orthomosaic** by mosaicking the individually orthorectified frame GeoTIFFs from step 7 into a single georeferenced raster. Unlike the ALFS product (an integral light-field image), this merges the exported GeoTIFFs with `rasterio`, so step 7 must be run first for the selected camera.

Like the other steps you can use **all** exported GeoTIFFs or restrict to a **frame-index range**, choose the **camera** (Thermal / RGB), and pick the **merge mode** that resolves overlapping pixels:

- **First** — the first frame covering a pixel wins (default)
- **Last** — the last frame covering a pixel wins
- **Min** — the darkest overlapping value wins
- **Max** — the brightest overlapping value wins
- **Average** — the mean of all overlapping frames (smooths seams; slower, two-pass)

**Outputs:**
```
orthomosaic_t/    # or orthomosaic_w/ depending on camera selection
└── orthomosaic.tif    # Merged georeferenced orthomosaic (LZW GeoTIFF with overviews)
```

### 9. Run SAM3 Segmentation

Segments individual detected objects from aerial images using Roboflow's SAM3 API. Recommended for RGB imagery.

> **Requires a Roboflow API key** — enter it in the **Configuration → SAM3** tab (Roboflow API Configuration group). The key is masked by default; use the "Show API key" checkbox to verify it.

**Outputs:**
```
segmentation_t/    # or segmentation_w/ depending on camera selection
└── segmentation_pixel.json    # Pixel-space segmentation masks
```

### 10. Geo-Reference Segmentation

Projects SAM3 pixel-space segmentation masks to world coordinates using the DEM.

**Outputs:**
```
segmentation_t/    # or segmentation_w/ depending on camera selection
└── segmentation_georef.json    # UTM-coordinate segmentation polygons
```

![Segmentation Output](../images/segmentations.png)

## Input file formats

### Calibration JSON

```json
{
    "mtx": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist": [k1, k2, p1, p2, k3]
}
```

### DEM metadata JSON

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

### Correction JSON (optional)

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

## Output structure

Each stage writes to a camera-specific subfolder (`_t` for thermal, `_w` for RGB), so thermal and RGB results coexist without overwriting each other. A complete run produces:

```
target_folder/
├── frames_t/                                # Extracted/undistorted thermal frames
│   ├── frame_000000.jpg
│   └── ...
├── frames_w/                                # Extracted/undistorted RGB frames
│   └── ...
├── poses_t.json                             # Camera pose per thermal frame
├── poses_w.json                             # Camera pose per RGB frame
├── mask_T.png                               # Thermal undistortion mask
├── mask_W.png                               # RGB undistortion mask
├── flight_route_t/                          # Flight route (thermal camera poses)
│   ├── flight_route.geojson                 # AirData GPS flight path (LineString)
│   ├── camera_positions.geojson             # Per-frame camera positions (Points)
│   ├── perpendicular.json                   # Detection perpendicular distances (flat)
│   ├── perpendicular_by_image.json          # Detection perpendicular distances (by image)
│   ├── perpendicular_tracks.json            # Track perpendicular distances (flat)
│   └── perpendicular_tracks_by_track.json   # Track perpendicular distances (by track)
├── flight_route_w/                          # Flight route (RGB camera poses)
│   └── ...
├── detections_t/                            # Detections on thermal frames
│   └── detections.txt                       # Raw YOLO detections
├── detections_w/
│   └── detections.txt
├── georeferenced_t/                         # Geo-referenced thermal detections
│   └── georeferenced.txt                    # Detections with UTM bounding box coordinates
├── georeferenced_w/
│   └── georeferenced.txt
├── tracks_t/                                # Tracks from thermal detections
│   ├── tracks_pixel.csv                     # Tracks in pixel coordinates
│   └── tracks.csv                           # Geo-referenced tracks (UTM)
├── tracks_w/
│   └── ...
├── fov_t/                                   # Field of View (thermal camera)
│   ├── fov_polygons.txt                     # Per-frame FoV polygon coordinates
│   └── merged_fov.geojson                   # Combined coverage area
├── fov_w/
│   └── ...
├── alfs_t/                                  # ALFS from thermal frames
│   └── alfs.tif                             # Georeferenced mosaic (COG GeoTIFF)
├── alfs_w/
│   └── alfs.tif
├── geotiffs_t/                              # GeoTIFFs from thermal frames
│   ├── frame_000000.tiff
│   └── ...
├── geotiffs_w/
│   └── ...
├── orthomosaic_t/                           # Orthomosaic merged from thermal GeoTIFFs
│   └── orthomosaic.tif
├── orthomosaic_w/
│   └── orthomosaic.tif
├── segmentation_t/                          # SAM3 segmentation on thermal frames
│   ├── segmentation_pixel.json              # Pixel-space segmentation masks
│   └── segmentation_georef.json             # Geo-referenced segmentation polygons
└── segmentation_w/
    └── ...
```

> **Note:** Only the subfolders for camera/stage combinations you actually run will be created. Running a stage for only one camera leaves the other subfolder absent.

Detection and Re-ID models are stored globally in the QGIS profile directory and shared across all projects:

```
%APPDATA%\QGIS\QGIS3\profiles\default\bambi_deps\models\
├── thermal_animal_detector.pt               # YOLO detection model
├── osnet_x0_5_bambi_thermal_omni.pt         # BAMBI Re-ID model
└── osnet_x0_25_msmt17.pt                    # BoxMOT default Re-ID model
```
