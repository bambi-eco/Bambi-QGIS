# Processing Pipeline

This guide covers the main workflow of the plugin: configuring inputs, running the processing steps in order, and understanding the outputs.

- [Input](#input)
- [Configuration](#configuration)
- [Processing steps](#processing-steps)
- [Input file formats](#input-file-formats)
- [Output structure](#output-structure)

> **New in 6.0.** Results are stored in GeoPackages rather than text files, a
> QGIS project can hold several flights, and the steps are split across two
> tabs. See [Results, Flights and Export](results-and-export.md) for the
> concepts; this page covers the steps themselves.

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
| Calibration   | `.json`           | Camera intrinsic parameters for undistortion; can be created with the [Camera Calibration Wizard](camera-calibration.md) |
| Correction    | `.json`           | Positional/rotational corrections; can be found interactively with the [Correction Wizard](correction-wizard.md) |
| DEM           | `.gltf` / `.glb`  | Digital Elevation Model + metadata JSON. Provide manually, convert a GeoTIFF, download automatically (Austria), or generate a flat surface mesh; see [DEM Import & Conversion](dem.md) |
| Target CRS    | EPSG code         | UTM-based CRS for the output (e.g. `EPSG:32633` for UTM Zone 33N). The **Auto** button derives a suitable UTM zone from the flight data |
| Flight name   | text (required)   | Name of the flight: titles its layer group and identifies it in the survey tools |
| Target folder | folder path       | Output directory for all generated files |

## Configuration

Before starting processing, configure per-step settings in the **Configuration** tab:

![Configuration Tab](../images/config_tab.png)

The tab has one sub-tab per topic:

| Sub-tab | Settings |
|---|---|
| **Project** | The project schema: species (with their map colour), enums and custom fields ([Results, Flights and Export](results-and-export.md#the-project-schema)); **Map Symbols**: pins in place of bounding boxes when zoomed out, and the map scale at which they switch ([How animals are drawn](results-and-export.md#how-animals-are-drawn)) |
| **Extraction** | Frame skip, limit, sampling rate, aspect ratio, heading source, AirData margin, **duplicated thermal frames**, timezone, thermal visualisation (see below) |
| **Flight Route** | Frame, distance and time marker intervals |
| **Detection** | Thermal / RGB model, confidence, frame range, **Projection** (camera frames, GeoTIFFs or ALFS integrals), detector class mapping, **custom detector** per camera ([Custom Implementations](custom-implementations.md)) |
| **Position Correction** | Translation and rotation offsets for geo-referencing, default and per frame range ([Correction Wizard](correction-wizard.md)) |
| **Tracking** | Backend, IoU threshold, interpolation, Re-ID model, RGB↔thermal matching, TRex tracklet import, **custom tracker** |
| **Classification** | Projection, crop settings, backbone, the classifier per task (default, model file or **custom implementation**), class mapping, voting, the size-based age estimate ([Classifier configuration](tools.md#classifier-configuration)) |
| **Field of View** | Custom mask, simplification, frame range and sample rate |
| **ALFS/GeoTiff** | Ground resolution or fixed render size, frame range and step, crop to content, overviews, tile size, sampling mode; edge erosion of the frame GeoTIFFs |
| **Orthomosaic** | Merge mode, frame range |
| **Environment** | Sentinel-2 scene window and required clear share, buffer around the flight, whether to extend to the survey area ([Environmental Context](environment.md#configuration)) |


### Extraction

The **Extraction** sub-tab provides fine-grained control over which frames are processed and how thermal images are rendered.

![Extraction Configuration](../images/config_extraction.png)

| Setting | Description |
|---------|-------------|
| **Skip first** | Skip this many frames/images at the start before processing begins. Useful for ignoring take-off frames. |
| **Limit** | Cap the total number of frames processed (enable the checkbox to activate). |
| **Sampling rate** *(video only)* | Take every N-th frame (e.g. 5 = every 5th frame). Reduces processing time for long recordings. |
| **Keep duplicated thermal frames (flagged)** *(thermal video only, on by default)* | A thermal camera closes its shutter now and then to recalibrate; while it is closed the video repeats the previous image. Ticked, these frozen frames stay in `poses_t.json`, marked `"duplicated": true`, so the frame index equals the video's frame number (one video, no skip or sampling). Unticked, they are dropped as before 9.0.0 and all later frames move up. See [Duplicated frames](#duplicated-frozen-thermal-frames). |
| **Turn headings from true north to the map's north** *(on by default)* | A drone reports where its camera looks as a heading from **true north**. The north of the map - of the project's CRS - is true north on the projection's central meridian only; away from it the two differ by the *grid convergence*, about (longitude - central meridian) x sin(latitude): 1.4 degrees at Salzburg and 3.9 degrees in Vorarlberg in UTM zone 33, up to 2.2 degrees at the edge of a UTM zone in Austria. Ticked, every pose in `poses_*.json` gets the angle at its position (`"grid_convergence"`) and its heading is used as what it is; the log of the extraction names the angle. Unticked, the heading is used as a bearing on the map as up to 9.0, which turns everything by that angle about the point below the drone (1.4 m at the corner of a frame taken from 80 m up, for two degrees). See [Headings and the map's north](#headings-and-the-maps-north). |

**Thermal visualisation** (requires the DJI Thermal SDK, installable via the [Dependency Manager](installation.md#dependency-manager); the group is greyed out when the SDK is not detected):

| Setting | Description |
|---------|-------------|
| **Colormap** | Apply a false-colour map to the exported thermal frames (e.g. `plasma`, `inferno`, `jet`, `white-hotspot`). Choose `(none)` to keep raw 8-bit grey values. |
| **Tone mapping** | `Thresholds (lower/upper)` for a linear stretch with optional black clipping, or `Curve (custom mapping)` for a fine-granular, style curve. |
| **Lower threshold** | *(threshold mode)* Pixels below this temperature (°C) are rendered black. Enable the checkbox to activate. |
| **Upper threshold** | *(threshold mode)* Pixels above this temperature (°C) are rendered black. Enable the checkbox to activate. |
| **Curve** | *(curve mode)* Opens the curve editor: a tone curve maps a fixed temperature range to display intensity via draggable control points (left-click adds/drags, right-click removes). Because a flight's temperature bounds are usually unknown, the range can be entered manually or filled by **Auto Detect**, which scans all images in the thermal photo directory for the actual minimum/maximum (optionally ignoring the extreme 1 % to suppress dead/hot pixels). Temperatures outside the range clamp to the curve's endpoint values instead of turning black. The curve is saved with the project. |

### Duplicated (frozen) thermal frames

A frozen frame has a **valid pose** (position, rotation, timestamp) but the
**wrong image** - it shows what the camera saw before the shutter closed.
The frame list is never shortened, so frame numbers stay what they are;
instead each step decides what to do with a flagged frame:

| Skipped (the image would give a wrong result) | Kept, only marked (the pose is what counts) |
|---|---|
| Detection, thermal↔RGB frame matching and track matching, field of view (and with it transect area), ALFS, GeoTIFFs (and with them orthomosaic and coverage map), segmentation, embeddings and classification crops, COCO / YOLO / Camtrap DP export, label boxes (interpolated or propagated), the Correction Wizard, and in the 3D Flight Viewer the trail, live ALFS, auto-refine, frame comparison and the thermal reconstruction texture | Flight route and camera positions (with a `duplicated` attribute), transects, the height profile, track times; the click, transect, labelling and segmentation tools, the 3D Flight Viewer and the video creator show a ❄ "frozen frame" note |

Detection simply has no boxes on a frozen frame; the track interpolation
closes the gap like any other missed frame.

**Changing the setting renumbers the frames.** Re-extracting with the other
setting asks for confirmation, and everything computed from the previous
frames is marked out of date. Labels and frame-range corrections refer to the
old numbers and have to be checked by hand. Flights extracted before 9.0.0
had their frozen frames dropped.

### Headings and the map's north

The drone's heading is an angle from true north; the plugin's results lie in a projected map whose north is true north only on the projection's central meridian. The difference is the grid convergence of the place in the project's CRS:

| Place | CRS | The map's north lies |
|-------|-----|----------------------|
| Vienna | UTM 33N (EPSG:32633) | 1.0 degree east of true north |
| Salzburg | UTM 33N | 1.4 degrees west |
| Bregenz | UTM 33N | 3.9 degrees west |
| Bregenz | UTM 32N (EPSG:32632) | 0.6 degrees east |
| Salzburg | Austria Lambert (EPSG:31287) | 0.2 degrees west |

Frames extracted with **Turn headings from true north to the map's north** carry the angle in their poses, and every tool - geo-referencing, field of view, ALFS, GeoTIFFs, the wizards, the 3D viewer, the box projector - subtracts it from the heading through the plugin's one camera model. The extraction's log says how large it is.

**Flights extracted with 9.0 or earlier** have no such entry and are used exactly as before: their results do not change by themselves. The geo-referencing steps write a note with the angle into the log. To include it, extract the frames again and run the steps that follow.

**Rotation corrections.** A rotation correction about the vertical (*z*) that was calibrated in the Correction Wizard on frames extracted with 9.0 or earlier absorbed the convergence together with the compass's own error. After extracting again it would count twice: reduce the correction by the angle (the extraction's log names the value it should then have) or calibrate it again. The same holds for correction values carried over from another flight.

### TRex tracklet import

If you have already tracked animals externally with [TRex](https://trex.run), the plugin can import those tracklets instead of running its own tracker. In the **Tracking** sub-tab, set the **NPZ Folder** to the directory containing the TRex `*.npz` tracklet files.

- Check **Labels already in undistorted frame space** if TRex was run on already-undistorted BAMBI frames; leave it unchecked (default) if TRex was run on the original raw video
- **Frame numbers:** TRex counts frames as the video does, and that number is used as the frame index. Extract the thermal frames with **Keep duplicated thermal frames** ticked (the default), from one video, without skip or sampling - otherwise every box after the first frozen frame is placed with a later pose. The import warns when the frames were extracted with duplicates dropped, and leaves out boxes on frozen frames.
- When an NPZ folder is set, step **4. Track Animals Or Import** imports and geo-references the TRex tracklets instead of running the configured tracking backend

What the import records is what the plugin's own steps record: every tracklet is a **track**, each of its frames a **box** - under a source of their own, so detecting again leaves them alone. The boxes are stored in the pixels of the **extracted frames**: boxes made on the raw video are undistorted with the camera's calibration first, the way the extraction undistorts the frames. Their **positions** are then computed by *Geo-Reference Detections* itself, as part of the import and for every box of the camera, so a box that cannot be placed (no pose for its frame, off the DEM) is recorded with the reason rather than dropped. The text files (`detections.txt`, `tracks_pixel.csv`, `tracks.csv`, …) are written only while **Also write legacy text outputs** is on.

> Before 9.1 the import wrote the positions to the text files only, which nothing reads since 6.0 - imported tracklets could not be added to the map or analysed. Import them again to get their positions.

## Processing steps

The steps live on two tabs, split by what they depend on:

| Tab | Steps | Depends on |
|---|---|---|
| **Pre-Processing** | P1 Extract Frames, P2 Generate Flight Route, P3 Calculate Field of View, P4 Generate ALFS, P5 Export Frames as GeoTIFF, P6 Generate Orthomosaic | the drone poses and the DEM - **no animals involved** |
| **Pre-Processing** (*Environmental Context*) | E1 Satellite Indices, E2 Land Cover, E3 Terrain Wetness, E4 Weather - see [Environmental Context](environment.md) | the flight's extent and date; **camera-independent**, each source its own sub-step |
| **Processing** | Three sections: *Detection and Tracking* (A1 Detect Animals with geo-referencing and perpendicular distances, A2 Track Animals Or Import, A3 Match RGB ↔ Thermal Tracks with label sync), *Classification* (C1 embeddings, C2 occlusion, C3 species, C4 sex, C5 age), and *Segmentation* (S1 opens the [Segmentation Tool](tools.md#segmentation-tool)) | the detections |

The numbering is prefixed so a bare number cannot mean two different steps.
They are *not* one sequence: only P1 is a prerequisite for the Processing tab.
Re-running detection marks the Processing steps out of date and leaves the ALFS
and orthomosaic alone.

The **→** buttons below each step load the corresponding results into QGIS as styled layers, or compute derived products. Layers are grouped under the active flight's name, per camera (see [Layer groups and order](results-and-export.md#layer-groups-and-order)).

**Camera selection.** Each stage has one selector at its top, instead of one
per step (since 9.0.0):

| Stage | Selector | Applies to |
|-------|----------|------------|
| Pre-Processing steps | **Camera** (Thermal / RGB) | P1 extraction, flight route, FoV, ALFS, GeoTIFFs, orthomosaic |
| Detection and Tracking | **Camera** (Thermal / RGB) | detection, geo-referencing, perpendicular distances, tracking (matching always uses both) |
| Classification | **Input** (Thermal / RGB / Matched) | C1 embeddings and all classifiers |

The step statuses show the selected camera's outputs. To process both
cameras, run the stage once per selection.

Progress, the abort button and the log sit below the tabs, so they are visible whichever tab a step was started from.

![Processing Tab](../images/processing_tab.png)

### P1. Extract Frames

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

**Square or not.** Frames are written square (1:1) unless **Preserve original aspect ratio** is ticked. Every step and tool reads the shape off the frames themselves, so both work - since 9.1 also in the inspector (the boxes shown in the other camera, the cross hair of a map click) and in the Correction Wizard, which until then assumed square frames and placed everything on a 16:9 frame too far from the image centre.

**Extracting again** replaces the frames: every image is written anew, so a changed calibration or "preserve aspect ratio" shows in the pictures and not only in the poses and the mask, and video frames of the earlier extraction that the new one does not have (another sampling rate or range) are removed from the folder. Other files in the folder are left alone. Everything computed from the previous frames is marked out of date. An extraction that is aborted half-way leaves some new images under the previous frame list - the log says so; run it again before using the frames.

### P2. Generate Flight Route

Builds two complementary vector layers from the mission data:

- **Flight route line**: GPS positions recorded in the AirData CSV are projected to the target UTM CRS and connected as a `LineString`. This represents the true GPS-recorded flight path.
- **Camera position points**: The position of the drone at every extracted frame (from `poses.json`) is added as a separate point layer; these are the positions at which images were actually captured.

**Outputs:**
```
flight_route_t/    # or flight_route_w/ depending on camera selection
├── flight_route.geojson       # GPS-based flight path (LineString)
└── camera_positions.geojson   # Per-frame camera positions (Points)
```

Use **→ Add Flight Route to QGIS** to load both layers with styling applied.

> **Note:** Frame extraction (P1) is optional for this step. If no `poses.json` exists yet, only the AirData-based flight route line is generated (a warning is shown), which is useful for a quick overview of the flight before running any extraction. Camera positions, frame/distance markers and image labels require P1; re-run this step after extraction to add them.

![Flight Route](../images/route.png)

### A1. Detect Animals

Runs YOLO-based detection on all extracted frames. Separate model paths can be configured for the thermal and the RGB modality; the model matching the selected camera is used. The default wildlife detection model for each modality is downloaded automatically from HuggingFace on first use - the version this plugin release was tested with, fetched at the commit of its repository and checked against its published size and SHA-256 (so is every other default model: the Re-ID weights, the classifiers, the DINOv3 backbone and SAM3). A model therefore does not change under a survey unless the plugin is updated or you choose another one.

**Projection** (Configuration → Detection): detection normally looks at the
extracted camera frames (*Perspective*). With *Orthorectified (GeoTIFF)* it
looks at the per-frame GeoTIFFs instead - for detectors trained on orthophotos,
built-in or [custom](custom-implementations.md). *Export Frames as GeoTIFF* has
to have run for the camera (frames without a GeoTIFF are skipped) and the DEM
is needed. The detector's boxes are then raster pixels, and their world
position follows directly from the GeoTIFF; what the plugin adds is the box's
place in the **camera frame** (through the DEM and the frame's corrected
camera), because a detection is stored as a box in the extracted frame - the
picture the labelling tool, the inspector, the video creator, the classifiers'
crops and the image exports draw it on. From there on it is a detection like
any other: geo-referencing, tracking and everything after work unchanged. A
box that falls outside the DEM or its camera frame is dropped and counted in
the log. The default BAMBI models were trained on perspective frames.

*ALFS (light-field integrals)* works the same way on the integrals of **ALFS
sampling mode** (*Generate ALFS* with sampling enabled): one integral per
central frame, focused on the DEM, in which occluding vegetation is blurred
away. Only the central frames have an integral, so only those are searched;
the boxes are placed in the central frame's camera image. It is there for
detectors trained on ALFS imagery ([custom](custom-implementations.md) ones
today). The Classification tab's **Projection** has the same third option
for the crops of the embedding step and of custom classifiers - a frame
between two central frames then uses the nearest integral; no default
classifiers have been published for ALFS imagery yet.

**Outputs:**
```
detections_t/    # or detections_w/ depending on camera selection
└── detections.txt    # Bounding box detections (frame, x1, y1, x2, y2, confidence, class)
```

Manual annotations created with the [Labelling Tool](tools.md#labelling-tool) can be added alongside the detector's output (via its **Add detections to project** button) - or can replace it entirely (via **Replace detections in project**, after confirmation). They are geo-referenced like regular detections. A label track is a track already, so the trackers leave its boxes out and associate only the detector's: a labelled animal is one track, not two.

Since 6.0 the two never interfere: each producer owns its own rows, so
re-running the detector leaves manual labels untouched and vice versa. Editing
a label re-uses the detections it already produced, so only the boxes that
actually moved need geo-referencing again.

**Detecting again replaces the detector's boxes of that camera** - also when another model is selected. The class mapping is kept per model, but only the boxes of the latest run exist; they do not pile up next to those of the model that ran before.

#### → Geo-Reference Detections

Projects pixel-space bounding boxes to real-world UTM coordinates by ray-casting against the DEM mesh. Each detection's four corners are projected and the result is stored as a world-space bounding box.

**Outputs:**
```
georeferenced_t/    # or georeferenced_w/, follows the detection camera selection
└── georeferenced.txt    # Detections with UTM bounding box coordinates
```

Use **→ Add Detections to QGIS** to load per-frame detection layers. Zoomed in, each detection is its bounding box, in the colour of the track it belongs to; zoomed out it becomes a pin in the colour of its species, with sex and age on it - see [How animals are drawn](results-and-export.md#how-animals-are-drawn).

![Detection Results](../images/detection_results.png)

Detections whose ray never reaches the DEM - sky above the horizon, ground
beyond the DEM edge, a frame with no pose - are recorded with the reason rather
than dropped, so every detection is accounted for afterwards.

#### → Calculate Perpendicular

For each geo-referenced detection, computes the **perpendicular distance** to the flight route:

1. The 2D center of each detection's bounding box is computed
2. The nearest point on the AirData GPS `LineString` is found by projecting onto each segment
3. The Euclidean distance between the detection center and the foot point is recorded

This is particularly useful for **transect-based wildlife surveys**, where the perpendicular offset from the flight line is a key sampling variable.

**Outputs:**
```
flight_route_t/    # or flight_route_w/, uses the flight route camera selection
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

### A2. Track Animals Or Import

Associates detections across frames into continuous tracks using the selected tracking backend. If a TRex NPZ folder is configured (see [TRex tracklet import](#trex-tracklet-import)), the pre-computed TRex tracklets are imported and geo-referenced instead.

- Tracks with a single detection (no movement) are fully supported and appear as a bounding box without a movement line
- Multi-detection tracks show both the movement path and the final bounding box

**The built-in tracker** works on the geo-referenced boxes, frame by frame: each box of a frame either continues a track or opens a new one. How boxes and tracks are matched is the **Tracker Mode** (Configuration → Tracking):

| Mode | Boxes and tracks are matched |
|------|------------------------------|
| `GREEDY` | by overlap (IoU), the best pair first |
| `HUNGARIAN` (default) | by overlap, taking the assignment with the most overlap in all |
| `CENTER` | by the distance between the box centres, nearest first |
| `HUNGARIAN_CENTER` | as `HUNGARIAN`, then as `CENTER` for what overlap could not pair |

**IoU Threshold** is the smallest overlap that joins a box to a track (the three overlap modes); **Max Center Distance** the largest distance between the centres, in metres on the ground (the two centre modes) - the form enables the one the mode uses. The centre modes are for animals whose consecutive boxes do not overlap: a small animal, a low frame rate, a jump in the position. **Max Age** ends a track that went unmatched for more than that many frames; only frames that have boxes count, so the thermal camera's shutter does not end tracks. **Class Aware** never joins boxes of different species, and **Interpolate Missing** fills the frames a track skipped with boxes in between (in the text output; they are not detections).

**Abort** stops a tracker between two frames, before anything is written: the previous tracks stay.

**Outputs:**
```
tracks_t/    # or tracks_w/, follows the detection camera selection
├── tracks_pixel.csv    # Tracks in pixel coordinates
└── tracks.csv          # Geo-referenced tracks (UTM)
```

Use **→ Add Tracks to QGIS** to load each track as a grouped layer (movement path + final bounding box). Every track has a colour of its own for its box and path; zoomed out, the box becomes a pin in the colour of the track's species, with the sex in its middle and the age as a badge - see [How animals are drawn](results-and-export.md#how-animals-are-drawn).

The tracks are read from the store (`bambi_t/tracks.gpkg`), so the button works whether or not the two text files above were written - they are only there while **Also write legacy text outputs** is on - and it shows animals that were labelled by hand as well, which are never written to them.

![Track Visualization](../images/tracks_qgis.png)

#### → Calculate Track Perpendicular

For each track, takes the **last bounding box** (the animal's final recorded position) and calculates the perpendicular distance to the flight route, using the same method as the detection perpendicular step.

**Outputs:**
```
flight_route_t/    # or flight_route_w/, uses the flight route camera selection
├── perpendicular_tracks_t.json             # Flat list (used by QGIS layer); _w for the RGB tracks
└── perpendicular_tracks_by_track_t.json    # Per-track keyed results
```

`perpendicular_tracks_by_track_t.json` structure:
```json
{
  "42": {
    "detection_id": 1873,
    "last_frame": 123,
    "last_image": "frame_000123.jpg",
    "center": [UTM_x, UTM_y, altitude],
    "perpendicular": [foot_x, foot_y, altitude],
    "distance": 35.7
  }
}
```

Use **→ Add Track Perpendicular Lines to QGIS** to visualize as line features.

The distance is also the **Distance to the flight line** column of the [track inventory](#track-inventory) (`perpendicular_distance_m`). A distance is shown for a track only when it was measured from one of that track's own boxes - after tracking again the tracks are numbered anew, so run this step again to fill the column. (Before 9.1 the column was always empty: the inventory misread the file.)

### P3. Calculate Field of View

Computes the camera footprint polygon on the ground for each frame (or a subset), using the DEM for accurate terrain-following projection.

**Outputs:**
```
fov_t/    # or fov_w/ depending on camera selection
├── fov_polygons.txt      # Per-frame FoV polygon coordinates
└── merged_fov.geojson    # Union of all footprints (coverage area)
```

![Field of View per Frame](../images/fov.png)

![Merged Flight FoV](../images/flight_fov.png)

### P4. Generate ALFS

Creates a georeferenced Airborne Light Field Sampling (ALFS) by projecting all (or a subset of) frames onto the DEM surface and blending them together. Output is a Cloud-Optimized GeoTIFF.

The mosaic spans the whole DEM and is rendered in tiles of at most the *tile size*. A tile gets the frames that can see it - worked out from each frame's position, direction and field of view - and a tile no frame sees is not rendered at all; the log says how many those were. (Up to 9.0 such a tile had **every** frame of the flight rendered into it before it was found empty, which made the step slower the larger the DEM was compared to the flight; and a camera tilted away from straight down could be left out of tiles its frames reached.)

**Outputs:**
```
alfs_t/    # or alfs_w/ depending on camera selection
└── alfs.tif    # Georeferenced mosaic (COG GeoTIFF)
```

![ALFS Output](../images/alfs.png)

### P5. Export Frames as GeoTIFF

Exports individual frames as georeferenced GeoTIFFs, suitable for import into GIS tools or for detailed per-frame analysis.

**Outputs:**
```
geotiffs_t/    # or geotiffs_w/ depending on camera selection
├── frame_000000.tiff
└── ...
```

![GeoTIFF Export](../images/geotiff.png)

### P6. Generate Orthomosaic

Builds a **true orthomosaic** by mosaicking the individually orthorectified frame GeoTIFFs from P5 into a single georeferenced raster. Unlike the ALFS product (an integral light-field image), this merges the exported GeoTIFFs with `rasterio`, so P5 must be run first for the selected camera.

Like the other steps you can use **all** exported GeoTIFFs or restrict to a **frame-index range**, choose the **camera** (Thermal / RGB), and pick the **merge mode** that resolves overlapping pixels:

- **First**: the first frame covering a pixel wins (default)
- **Last**: the last frame covering a pixel wins
- **Min**: the darkest overlapping value wins
- **Max**: the brightest overlapping value wins
- **Average**: the mean of all overlapping frames (smooths seams; slower, two-pass)

**Outputs:**
```
orthomosaic_t/    # or orthomosaic_w/ depending on camera selection
└── orthomosaic.tif    # Merged georeferenced orthomosaic (LZW GeoTIFF with overviews)
```

### Classification

Works out **what** each tracked animal is: whether a frame shows it clearly, what species it is, its sex, and whether it is a juvenile. Implements the pipeline of *When One Modality Is Not Enough: Multimodal Sex and Life-Stage Classification of Red Deer from Aerial RGB–Thermal Video*.

> **Tracks you annotated in the labelling tool are never changed by these steps.** A hand annotation outranks a model: the classifiers fill in what is missing and leave your own work alone.

The reason this is worth doing in two sensors rather than one is that they fail in opposite conditions. In colour a deer under canopy blends into the ground; in thermal it is an unmistakable bright blob but the fur colour is gone, and antlers only show while they are still growing and warm. Which sensor carries the sex cue therefore changes with the season - which is why the *matched* option, reading both at once, is the default.

> **Requires a Hugging Face token.** The DINOv3 model these classifiers read is *gated*: request access at [huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m), then paste a read token into **Configuration → Classification** and press **Check access**. See [Installation](installation.md#optional-classification).

#### A3. Match RGB ↔ Thermal Tracks

Decides which thermal track and which RGB track are the same animal, by registering the two views onto each other and comparing where the boxes sit. Needs both cameras tracked; there is no camera selector, because the step is inherently about the pair.

Confirmation is what keeps a census honest. An animal both cameras saw is a real animal; a track only one camera saw is either an animal the other sensor cannot make out, or noise. In the paper's four test flights, of 34 tracks without a partner only *one* was a real animal - admitting them all would have added six individuals that did not exist.

Each track is paired with at most one track of the other camera, and the pairing is chosen for all tracks together: the one with the smallest total distance, where a track that stays without a partner counts as half the gate. A pair is therefore taken exactly when it is closer than the gate, and a clearly right pair is not given up to squeeze in two that barely clear it. (Up to 9.0 the pairing matched as many tracks as it could before it looked at a distance, and in a herd could drop the right pair for two marginal ones.)

The run log reports how many pairs were confirmed out of how many raw tracks. If nothing matches, it names which gate rejected everything and how far the closest candidate was, so you can tell "there were no animals" from "the gate is wrong for this resolution".

**→ Add Matched Pairs to QGIS** draws a line between each pair's positions, attributed with the shared frame count and median distance - the quickest way to see whether the gate is set sensibly.

**→ Sync Labels Across the Match** copies species, sex and age from one camera's animals onto their matched partners on the other (direction selectable: *Thermal → RGB* or *RGB → Thermal*), and the per-frame occlusion onto the paired detections. Only matched pairs are touched; an animal that already carries an answer keeps it, hand annotations from the labelling tool are never overwritten, and what a person decided in the [review queue](#review-queue) stays - also where they decided that the species, sex or age cannot be told. (Up to 9.0 the per-box part - occlusion, and the species on the boxes - looked the pairs up the wrong way round: it was right only where both cameras happened to number their boxes alike, and otherwise copied nothing or took the values of an unrelated box. Run the sync again on flights where it was used.) It is what lets a classifier that only works on one camera label the animals of both.

The step sits in the *Detection and Tracking* section (it needs both cameras tracked, nothing else); the classification stage builds on it whenever its **Input** is *Matched*.

**Outputs:**
```
matches.gpkg    # beside project.gpkg: a match belongs to neither camera
```

#### C1. Compute DINOv3 Embeddings

Describes every tracked animal's crop as a feature vector, once, so all three classifiers can reuse it. This is the expensive step - the model is large, and on CPU it takes minutes per hundred crops.

Vectors are written beside the frames, one file per frame, so they are reusable outside the plugin. A re-run only embeds what is missing: an interrupted run resumes rather than starting again, and changing the crop settings starts a new set without discarding the old one.

**Outputs:**
```
embeddings_t/            # or embeddings_w/
└── non_geo/             # or geo_1k/ geo_2k/, per the chosen projection
    └── frame_000123.npz # one array per detection, named det_<id>
bambi_t/classification.gpkg   # which detections are embedded, and by which run
```

#### C2–C4. Occlusion, Species and Sex Classification

One step each, run in that order, because each depends on the one before:

1. **Occlusion**, per frame, labels a crop *clear* or *occluded*. It is a quality filter, not a verdict about the animal, so it produces no per-animal answer - and it is **optional**.
2. **Species** votes across the frames that are worth trusting, and its majority fixes what the animal is. The published model knows red deer, roe deer and wild boar - anything else is forced into one of the three, so on a survey with other animals check the class mapping or leave the task off.
3. **Sex** reuses *exactly those frames*, and picks its model from the species just assigned.

Voting is what makes a noisy per-frame call safe. An antler only resolves from some angles, so many frames of a true male look female; the majority still recovers him. The margin behind every call is kept - "male, 106 of 115 frames" is what lets you judge a borderline animal - and you can re-vote at a different quorum without re-running anything.

An animal the classifier cannot call is **left unknown, never discarded**: it keeps its place in the census with the attribute blank.

**The animals follow the latest run.** Running a classifier again - with another model, or re-voting at another quorum - replaces what the earlier run wrote, on the track and on its boxes (which is where the exports and the analytics read the species), and takes a call back from an animal the new run no longer calls. Only what a classifier wrote is replaced or taken back: a species from the detector's class mapping, a value synced across from the other camera after the call, hand labels, and a species, sex or age that a person set in the [review queue](#review-queue) stay.

**Frames used** decides what may vote. *Visible frames only* is decided frame by frame, by the best that is known about that frame: the occlusion classifier's call where it has one, otherwise the occlusion value stored on the box (what the labelling tool writes), and otherwise the frame votes - nobody called it occluded. The run log says which of these the tracks went by, and how many frames voted without a label. (Up to 9.0 a frame without a label counted as occluded as soon as any box of the flight had one: a single hand-labelled track left every other track without a frame to vote with, and so without a species.)

**Unmatched animals** decides what happens to an animal only one camera saw when a classifier's **Input** is *Matched* - such a classifier reads both cameras side by side and cannot take a blank for the missing one. *Skip the animal* leaves it uncalled. *Fall back to RGB / thermal* hands it to the published classifier of that one camera for the same task, and the stored result of each frame names which of the two called it; the log says how many frames went that way. A model file of your own has no single-camera counterpart, so its unmatched animals are skipped (the log says so). (Up to 9.0 the fallback handed the matched classifier the one camera's vector, which is half as wide as its input: the classifier refused it and the whole run stopped.)

**Applying the classifications to tracks and detections** copies the answers onto the animals themselves, which is what makes them visible to the exports, the map layers, the survey analytics and the labelling tool. It runs automatically unless switched off, and is safe to repeat.

#### C5. Age Classification

Flags juveniles by body size - the **fallback** where no classifier called the animal.

Life stage comes from whatever you chose for that species. Under the life-stage classifier's **Species…** button each species is set to **Size-based**, a model, or **Off** - one decision in one place, rather than a model choice plus a switch somewhere else. Size-based is the default, because no life-stage model has been published yet; a species set to a model is called by it in C3 and left alone here, and one set to Off is not called at all.

Needs no models - only tracks, and geo-referencing if the metric areas are to be used.

A juvenile cannot be told from an adult female by appearance at survey resolution, which is exactly why the sex classifier's second class is *female/juvenile*. Size settles it: a juvenile sits far below its cohort **and** a clear gap separates it from the herd - directly above it, or above another animal as far below the median as it is. Both conditions are required, because in any herd someone is smallest and that alone is not evidence. Twins are both juveniles: the smaller of two fawns of a size is a hair below its sibling, and the gap that counts is the one over the two of them (up to 9.0 only the larger twin, the one next to the gap, was called).

Sizes are only ever compared **within one flight** - how tightly boxes fit varies between recordings, enough that the paper's own juvenile from one flight lands inside another flight's adult range. There is deliberately no absolute threshold.

... and **within one species**. A roe deer is half the size of a red deer at any age; measured against a herd of red deer an adult roe deer is exactly the low outlier the rule looks for, and up to 9.0 - when all animals of a flight were one cohort - it was called a juvenile. An animal's species is the species classifier's call, else the one on its track, else what the detector called most of its boxes. The animals nobody identified are compared with each other (on a flight without any species that is everyone, as before), a species with fewer animals than **Min individuals** is not judged at all - the other species' are no measure for it - and a track that is `not-an-animal` is neither judged nor part of anyone's cohort. The log has one line per species.

Areas come from the **geo-referenced** boxes where geo-referencing has run, which makes them metric; otherwise from the camera-frame boxes, which still work because the comparison never leaves the flight. Which was used is recorded with every verdict, so metric and pixel figures are never mixed. Animals a classifier already called still count towards the cohort statistics - excluding them would shift everyone else's score - they simply do not take a verdict from it.

On a flight with only a handful of animals the test declines to call one, because the candidate sits in the lower half of the distribution and so widens the very spread it is measured against. The log says so rather than reporting a bare "no juvenile found": a cautious answer and an empty one look identical otherwise.

Each camera is measured as **its own cohort** - a thermal sensor and an RGB sensor frame the same animal differently, so their areas are never pooled. Where tracks have been matched across cameras (A3), a juvenile found on one camera is **carried onto its partner track** on the other. The thermal cohort is often the one that cannot confirm a calf: it holds every warm blob the detector tracked, species-less and frequently one frame long, which pads out the lower end of the distribution until an animal at half the median is nothing special - while among the RGB deer the same animal is a clear outlier. Only size verdicts move: an animal a classifier called keeps that answer, *adult* is never carried because it is the absence of an outlier rather than a finding, and a camera the stage did not measure receives nothing. The carried verdict records which partner it came from, and the track inventory shows it as `matched RGB track N` in the `age_source` column.

**Outputs:**
```
bambi_t/classification.gpkg
├── frame_predictions    # per crop: label, probability, model
└── track_predictions    # per animal: the call, and the vote behind it
```

### Accuracy against hand labels

A flight somebody went through in the [Labelling Tool](tools.md#labelling-tool) carries its own answer key: every animal that person drew is known, frame by frame, with its species, sex and age. **→ Compare with Hand Labels…** (Processing tab, *Accuracy*) compares the pipeline's output on the selected flight with it - which says how far the numbers of the flights nobody went through can be trusted, and where the pipeline fails. Nothing is changed by it.

What is compared is the pipeline's **own** output: the detector's boxes (hand-drawn ones are left out), the tracker's tracks - also those a label has replaced since - and the species, sex and age the survey tools count each track as. The report answers three questions in turn, each needing the one before:

- **Were the animals found?** Frame by frame, the detector's boxes are paired with the labelled ones - the best overlapping pairs first, a box used once. A labelled animal with a box on it is *found*, one without is *missed*, a box on nothing is a *false alarm*. From these come **precision** (the share of boxes that are on an animal) and **recall** (the share of labelled boxes that got one), overall, per labelled species and per species the detector called its boxes - and a table of what both would have been **had only boxes of at least a given confidence been kept**, with the confidence at which their balance (F1) is best.
- **Was each animal one track?** A census counts tracks. Per labelled animal: whether a track follows it, for how much of the time, and in how many pieces; per track: whether it follows a labelled animal at all, or more than one. The number of tracks against the number of labelled animals is what a census would have got wrong (`+2`: two animals too many).
- **Was it named right?** For the animals that say what they are, a confusion table - rows what was labelled, columns what the survey tools count the track as that follows the animal for the longest - for species, sex and age; where a classifier made the call, also the classifier's own call per model.

| Setting | Description |
|---------|-------------|
| **Camera** | Thermal or RGB; starts on the camera the animals are tracked on. |
| **Overlap (IoU) at least** | How much a box of the pipeline and a labelled box have to overlap to be the same animal: the area they share over the area they cover together. Labels are drawn on key frames and filled in between, and a thermal animal is a few pixels across - the default of 0.3 allows for that; benchmark data sets use 0.5. |
| **First to last labelled frame** | Which frames were gone through. A frame without a label only says "no animal here" if somebody looked at it, so by default the frames from the first to the last label count. Untick to name the frames - the whole flight, if all of it was gone through. |

A label of the species **`not-an-animal`** is the reviewer saying "this is none": a box on it is a *confirmed* false alarm, and it is not an animal that could be missed. Labels made with *Import as label track* are compared like any other; where nobody moved a box it matches itself, which flatters the figures a little, and the report says how many such labels there are. Detections and tracks deleted in the inspector are gone from the project and cannot be counted against the pipeline.

**Outputs** (written on every comparison):
```
analytics_t/    # or analytics_w/, in the flight's folder
├── accuracy.html        # the report
├── accuracy.json        # all figures, per animal and per track
└── accuracy_boxes.csv   # every labelled box and every box of the pipeline: found, missed, false alarm, ruled out
```

### S1. Open Segmentation Tool

Segmentation is interactive - a prompt, a look at the mask, another click - so it is a window of its own rather than a step: the [Segmentation Tool](tools.md#segmentation-tool), also on the toolbar. It prompts SAM3 / SAM 3.1 with **text** ("deer") or **clicked points** on the extracted frames of either camera, on single frames or **tracked across a sequence**, through the Roboflow API, `transformers`, or Meta's official `sam3` package. The window also geo-references the masks onto the DEM, adds them as QGIS layers and exports them as GeoJSON.

**Outputs:**
```
segmentation_t/    # or segmentation_w/ depending on the camera
├── segmentation_pixel.json     # Pixel-space mask polygons per frame and prompt
└── segmentation_georef.json    # The same polygons in UTM coordinates (after geo-referencing)
```

![Segmentation Output](../images/segmentations.png)

## Survey analytics

The **Survey Analytics** tab (next to *Processing*) turns geo-referenced
detections, tracks and exported frames into population-level products. It is
**not tied to the flight selected on the other tabs**: a survey is a set of
processed flights analysed together.

### The survey: flights, camera, folder

The box at the top of the tab defines the survey every tool below works on:

| Setting | |
|---|---|
| **Flights** | Tick the processed flights of this QGIS project that make up the survey. Their counts, areas and distances are pooled; the census summary also lists them one by one. A flight that has not been processed, or whose folder is missing, cannot be ticked. |
| **Camera** | Thermal or RGB - whose detections, tracks, footprints and transects are analysed. The flights must have been processed for it. (This is the survey's own selector; it is independent of the stage selectors on the processing tabs.) |
| **Count** | What every tool on the tab counts. **Animals (tracks)** - the default - counts each tracked animal once, at the centre of its boxes, with the species, sex and age of the track. **Boxes (detections)** counts every geo-referenced box: what a still-photo survey without tracking has - an animal on two photos counts twice. |
| **Survey folder** | Where the survey results (`analytics_t/` / `analytics_w/`) and the survey area are stored. Empty = `bambi_survey` next to the QGIS project - next to the first ticked flight while the project has not been saved. **A survey folder that holds results stays where it is**: when saving the project (for the first time, or somewhere else) or ticking another flight first would move the default, the folder used so far is entered in the field instead, and the log says so. Empty the field to go to the new default, or enter another folder, to move the survey yourself. |

Four things hold for every tool:

- **One CRS.** Pooled points, rasters and areas need the ticked flights to
  share their target CRS; this is checked, not assumed.
- **Species by name.** Species ids are assigned per flight, so the species
  filter works on names. A name missing from a flight is not an error - that
  flight simply has none of it.
- **What was counted is recorded.** Each result notes the tracking run used,
  whether manual tracks were included, the species filter, and how many
  detections labelled `not-an-animal` were excluded. One tracker run
  contributes (built-in, BoxMOT and TRex describe the same animals, so pooling
  two would double-count), and manual tracks are *added* to it. A label created
  with *Import as label track* replaces the tracker track it came from. See
  [Results, Flights and Export](results-and-export.md#what-the-analytics-counted).
- **An animal is one thing in every tool.** A tracked animal has one
  species: the track's own where it was identified - by a classifier, by
  hand, or synced across from the other camera - and otherwise what most of
  its boxes were called by the detector's class mapping. It has one position,
  the centre of all its boxes, and the sex and age of its track. A track that
  is `not-an-animal` by that rule is not counted. Up to 9.0 the tools asked
  different things - the census and the inventory the track (`animal` for
  every track no classifier had named), the heatmap each box (an animal
  called roe deer on two frames and chamois on a third counted once for
  each), distance sampling and the population estimate the last box - and so
  disagreed about the same animals. A box, when the survey counts boxes, is
  what the box itself was called, with the sex and age of the animal it
  belongs to.

**One setting decides what is counted.** Up to 9.0 the census, the density
heatmap and distance sampling each had a selector of their own, with
different defaults - the census counted animals, the other two boxes - so the
same survey was three different things depending on the tool, and a density
of boxes is not a density of animals. The **Count** setting of the survey box
now applies to all of them, and is saved with the QGIS project. The
population estimate and habitat use, which only knew tracks, follow it too,
so a still-photo survey without tracking can use every tool. Census, heatmap
and distance-sampling results are still written per choice (`..._tracks` /
`..._detections`), so switching does not overwrite the other set; the
population estimate and the habitat summary record what they counted
(`source`).

### Survey area and exclusion zones

A census is usually reported for a defined area - a hunting district, a
reserve, a forest compartment - not for whatever ground the flights happened
to see. The **Survey Area** box holds that area:

- **Survey areas** - one or more census polygons. An animal counts when it is
  inside any of them, and only the ground inside them is surveyed area
  (overlapping areas count once).
- **Exclusion zones** - ground removed again, each with an optional reason
  (thick forest cover, a building, a road). Animals there are not counted and
  their ground is not part of the area.

`effective area = union(survey areas) − union(exclusion zones)`. Exclusion
zones may be defined without a survey area; they are then cut out of the
whole flights.

**Draw Survey Area** / **Draw Exclusion Zone** add the layer, switch it to
editing and start the polygon tool (click the vertices, right-click to finish,
save the layer edits). **Import…** copies polygons from a layer already in the
project - a shapefile of the hunting districts, a KML, a GeoJSON. The polygons
live in `<survey folder>/survey_area/survey_area.gpkg`, an ordinary
GeoPackage edited with the normal QGIS tools.

With **Limit analytics to the survey area** ticked, the census, density
heatmap, coverage map, distance sampling and population estimation count only
animals inside a survey area and outside every exclusion zone, and use only
that ground as the surveyed area. Unticked, they cover the whole flights. The
survey orthomosaic is never clipped.

### Groups

Many animals do not stand alone - a herd of red deer, a sounder of wild boar,
a doe with her fawns. The survey tools therefore know which animals were
together:

- **Same species, close enough.** Two animals of one species no further apart
  than the species' **group distance** belong together, and so does everything
  joined to them in the same way: a herd strung out along a forest edge is one
  herd although its ends are far apart. The distance is **50 m** unless the
  [Project Schema](results-and-export.md#species) sets another one for the
  species; `0` means the species is never grouped.
- **Seen together.** A flight crosses the same ground more than once -
  neighbouring legs overlap - and an animal near the border of two legs is
  counted on both. Two sightings minutes apart are not a group of two: tracked
  animals are only joined when the drone saw them within **60 seconds** of each
  other, and boxes (when the survey's **Count** setting is *Boxes*) only
  within the same frame. Where a frame has no capture time, distance alone
  decides.

Animals whose species is not determined (`animal`, `unknown`) are grouped among
themselves, like any species; an animal without a geo-referenced position is on
its own. The animals of a flight are divided once, all of them, so a group is
the same in every tool: the [census](#census-summary) counts the groups, the
[track inventory](#track-inventory) says which group each animal was in, and
[distance sampling](#distance-sampling) takes a group as one observation.

### Census summary

The plain census figures a survey report starts from: how many of each
species were counted, how many of them inside the survey area, how many per
km², how they split by sex and age, and how many [groups](#groups) they formed
- per flight and for the survey as a whole. No detection-function model is
involved; these are minimum counts.

What is counted - animals or boxes - is the survey's **Count** setting.

Two densities are reported, because which is right depends on the survey:

- **per plot** - the survey area minus its exclusion zones. Right when the
  flights together cover the area completely: adjacent flights add up to one
  census, `sum(counts) / plot area`.
- **per covered ground** - the ground the frames actually saw inside the area
  (the union of the field-of-view footprints). It does not assume full
  coverage, and across flights it is `sum(counts) / sum(covered areas)` -
  right for repeat flights over the same plot as well.

An animal without a geo-referenced position is counted, but with a survey
area it is reported as *unplaced* rather than inside. **→ Show Census Table**
opens the result as a table.

The last three columns are the **groups** of the animals counted inside, their
**mean size** and the **largest**. A group counts with its members that are
counted: nine deer of which six stand inside the survey area are a group of
six. The groups themselves are listed one row each - flight, species, size,
members counted, position, radius, first and last time seen, and the track
(or detection) ids of the members - and **→ Add Groups to QGIS** draws every
group of two or more as an outline around its members, in the colour of the
species and with the number of animals in it.

**Outputs:**
```
analytics_t/    # or analytics_w/, in the survey folder
├── census_summary_tracks.csv    # or census_summary_detections.csv
├── census_summary_tracks.json
├── groups_tracks.csv            # or groups_detections.csv: one row per group
└── groups_tracks.json
```

### Grid density

Transects describe a survey along the way it was flown. A grid describes it by the ground: the survey's ground is cut into **equal cells**, and what was found is looked at cell by cell - how dense, how clumped, and where the hot spots are. The grid covers the **survey area**; without one it covers the ground the ticked flights saw.

| Setting | Description |
|---------|-------------|
| **Cells** | The shape of a cell: **hexagons** (the default), **squares**, **triangles** or **rectangles**. Every neighbour of a hexagon shares a whole edge with it and is equally far away, which is what the hot-spot statistics assume; rectangles have a width and a height of their own. |
| **Size by** | Give a cell's size by its **area** (default 1 ha) or by the **length of its edge** - two ways to say the same: switching keeps the cells as they are. |
| **Seen at least** | The share of a cell the cameras have to have seen for it to count (default 50 %). |
| **Repeated flights** | How a cell that several flights saw is read: **Average over the flights** - all sightings over all the ground looked at, the mean density over the visits - or **Highest flight** - the flight with the highest density, among those that saw enough of the cell on their own: what was there at least. |

The settings are saved with the QGIS project.

**Density is animals over ground that was seen.** A cell the drone crossed at one corner holds few animals because little of it was looked at, not because few are there. A cell's density is therefore its animals divided by the ground the cameras saw *in the cell* - the field-of-view footprints inside the survey area - not by the size of the cell. Run **Calculate Field of View** for the survey's camera on every flight first; a flight without footprints counts as having seen the whole survey area, and cannot be analysed without one.

**A cell that was hardly seen says nothing.** Cells of which less than the set share was seen stay on the map as *too little seen*, with what was found there, and are left out of every statistic. The share is of the whole cell, so a cell the edge of the survey area cuts down to a corner is treated the same way. Cells no flight saw are *not seen*.

**The same ground is the same cell, in every survey.** The grid is laid from the origin of the map's coordinate system, not from a corner of what is analysed today: a square cell of 100 m always runs from a multiple of 100 to the next. Two surveys of an area - or one survey analysed again after a flight was added - have the same cells with the same names (`c5000r53000`: column and row counted from the origin), so they can be compared cell by cell.

**Per cell** (`grid_density_{source}_cells.csv`, and the attributes of the layer): its status, the animals and their density per km², the same per species, the counts by sex and age, the [groups](#groups) among its animals and their mean size, how much of it was seen and by how many flights, the [environment](environment.md) it lies in - the mean of each layer over the cell and the most common land-cover class with its share - and whether it is a hot or cold spot, overall and per species. A second table has one row per cell **and flight**.

**Over the cells that count** - for all animals and for each species; **→ Show Grid Statistics** opens them as a table:

- the **mean density** with a 95 % interval from *resampling the cells* - they are drawn again and again with replacement, and the middle 95 % of the densities that gives is the interval. It asks nothing of how the counts are distributed, but takes the cells as independent: with clumped animals it is somewhat too narrow;
- the **total for the survey area** - the mean density times the area minus its exclusion zones (not given without a survey area);
- the **variance-to-mean ratio** - 1 for animals scattered at random, above 1 for clumped, below 1 for evenly spread ones - with its chi-square test. Cells that were seen to different extents are compared with what their ground would hold at the overall density;
- **Moran's I** - whether neighbouring cells are more alike than cells in general, with a permutation test;
- the **share of cells with animals**, with its interval. A minimum: an animal that was there and not found makes a cell look empty;
- the number of **hot-spot and cold-spot cells**: cells that hold more (or less), together with their neighbours, than cells picked at random would (Getis-Ord Gi\*, at 90, 95 and 99 %; not corrected for the number of cells tested).

Resampling and permutations use a fixed seed: the same survey gives the same numbers every time.

**→ Add Grid to QGIS** loads the cells as a layer in the BAMBI Survey group, coloured by density, with the number of animals written in. Cells that were too little seen are hatched, cells nobody saw left empty. The layer comes with a second style, **Hot spots**, in its style menu (right-click → Styles).

**Outputs:**
```
analytics_t/    # or analytics_w/, in the survey folder
├── grid_density_tracks.json            # or grid_density_detections...: everything, with the cells' corners
├── grid_density_tracks_cells.csv       # one row per cell
└── grid_density_tracks_by_flight.csv   # one row per cell and flight
```

### Compare visits

One survey of an area is a count. Several - weeks or years apart - say what is happening to the animals, if the visits are compared fairly. Tick the flights of **every** visit to the area in the Survey box; **→ Compare Visits** divides them into visits by the day they were flown, counts each visit, and says what changed.

| Setting | Description |
|---------|-------------|
| **One visit: flights up to … day(s) apart** | A flight flown no more than this many days after the one before belongs to the same visit. The default, 0, makes the flights of one day one visit; a survey that takes three mornings is one visit with 1. |
| **Ground compared** | **All the ground each visit saw** - every visit is taken as it was flown, its animals over its ground. **Only the ground every visit saw** - the like-for-like comparison where the visits did not cover the same part of the area: only the ground all of them saw is compared, and only the animals that stood on it. |

Both are saved with the QGIS project.

**The day is the local one** the frames were stamped with, so two flights at 00:30 and at 20:00 of the same day are one visit although the flight log's clock (UTC) has them on two. Flights less than six hours apart are one visit whatever the calendar says: a night survey does not end at midnight. A flight whose frames carry no capture time cannot be placed and is named.

**What a visit counted** - per species and for all animals, as the [census](#census-summary) counts them (inside the survey area, if one is drawn): the animals, the [groups](#groups) they formed and the largest, sex and age, and the **density**: the animals over the ground the cameras looked at during the visit, summed over its flights. Two flights side by side are one count of both grounds; two flights over the same ground on one day give the mean of the two. With a survey area the density is also given as the **number in the survey area** (density times the area minus its exclusion zones; not where only common ground is compared). A flight without field-of-view footprints counts as having seen the whole survey area; without footprints *and* without an area the ground is not known, and the counts are compared as they are - fair only where every visit covered the same ground.

**How sure a count is.** The 95 % interval of a density takes a group of animals as *one sighting*: ten deer in one herd are one piece of luck, not ten. (It is the gamma interval of Fay and Feuer for a sum of group sizes; for animals seen one by one that is the exact Poisson interval. A visit that found nothing still has an upper limit.)

**From visit to visit** - the column *Since the visit before*, and below the table the first visit against the last: the change in percent and how likely a difference at least as large is between two counts of one and the same density (an exact test: given the animals of both visits together, how they would fall to the two by their shares of the ground looked at). Clumped animals are worth fewer sightings than there are animals, so thirty deer in three herds against ten singles is *three times as many, on the strength of four sightings* - within what chance explains.

**The trend** - from two visits on, a constant rate of change is fitted to the counts (Poisson regression with the ground looked at as exposure), and given as the change over the time the visits span - and per year, once they span a year - with a 95 % interval: an *increase*, a *decrease*, or *no clear trend* where the interval includes no change. The interval allows for the groups and, from three visits on, for visits that lie further from the line than counting explains (it is then Student's, on the visits less two). A series that starts or ends with no animals at all has no rate; the visit-to-visit comparison says what there is to say.

**→ Show Visits and Trend** opens the result: for the species chosen at the top a chart of every visit's density with its interval and the fitted trend, the table of visits, and the sentences above. Each visit is listed with its time of day and - where the [weather step](environment.md) was run - the air temperature, for a reason:

> A count is of the animals **seen**. Canopy, season, time of day and weather change how many of the animals present that are, so a trend in counts is a trend in animals only as far as the visits were flown alike. Compare spring with spring and dawn with dawn; a leaf-off count against a leaf-on one mostly measures the leaves.

Counting **boxes** instead of animals (the survey's *Count* setting) counts an animal once per frame it is on: the intervals are then far too narrow, and only the direction of a change means anything.

**Outputs:**
```
analytics_t/    # or analytics_w/, in the survey folder
├── trend_tracks.json           # or trend_detections...: the visits, the series per species, changes, trends
└── trend_tracks_visits.csv     # one row per visit and species, with sex, age and weather
```

### Density heatmap

Generates a **kernel-density estimate raster** of animal locations. Points are binned to a grid and smoothed with a Gaussian kernel; each output pixel is the estimated density in **animals per hectare**.

| Setting | Description |
|---------|-------------|
| **Cell (m)** | Output raster cell size in metres (default 5 m) |
| **Bandwidth (m)** | Gaussian smoothing radius in metres (default 25 m) - larger values give smoother, broader hotspots |

**Outputs:**
```
analytics_t/    # or analytics_w/
├── density_tracks.tif    # or density_detections.tif - float32 GeoTIFF (points/hectare)
└── density_tracks.json   # stats: peak/mean density, point count, parameters
```

Use **→ Add Density Heatmap to QGIS** to load the raster with a blue→red colour ramp scaled to the data.

**A zero is a result, and no data is something else.** The raster tells three things apart:

- ground that is known to have been surveyed carries its density - **`0` where no animal was**. "Known" is the survey area where one is switched on, else the cameras' footprints (run **Calculate Field of View** for the survey's camera); with both, the footprints inside the area. The raster reaches as far as that ground does.
- everything outside it is **no data** (transparent).
- where neither a survey area nor footprints say what was looked at, only the cells the kernel reaches carry a value and the rest is blank - "no animal" cannot be told from "not looked at" then, and the log says so.

`mean_density` in the JSON is the mean of what the raster holds, and `mean_density_over` says which: `surveyed_ground` (all of it, its empty parts included - the figure to quote, with `surveyed_area_ha`) or `cells_with_animals` (how dense the animals stood where they were, which says nothing about the survey as a whole).

Before 9.1 every cell without animals was no data. Surveyed ground that was empty looked like ground that was never surveyed, and a mean taken from the raster - by the plugin, or by QGIS' zonal statistics - was the mean over the places where animals were. The raster's cells are now exactly the cells the animals were counted into; they used to be written a little narrower, sliding by up to one cell towards the top left.

### Distance sampling

Estimates **density and abundance with 95% confidence intervals** using conventional line-transect distance sampling. It reuses the perpendicular distances of what the survey counts: those of **Calculate Track Perpendicular** when it counts animals - one observation per animal, which is what distance sampling assumes - and those of **Calculate Perpendicular** when it counts boxes. Run that step first.

The tool fits both a **half-normal** and a **hazard-rate** detection function by maximum likelihood, selects the better fit by **AIC**, and computes the effective strip width (ESW), average detection probability, density (per km²) and abundance for the covered strip. Uncertainty combines the encounter-rate variance with the detection-function variance via the delta method, reported as a lognormal confidence interval.

**What counts as surveyed line.** Only line that was searched, and only the animals seen from it:

- where a flight has **transects** ([Transect Splitting Tool](tools.md#transect-splitting-tool)), its lines are the transect legs, and an animal counts when its frame lies on one of them - animals seen in transit are reported and left out;
- otherwise the line is the camera's path **from the first to the last recorded frame**;
- with a **survey area**, a line is as long as the part of it inside the area.

Up to 9.0 the whole flight log counted - take-off, the transit to the first transect, the way home - which made the line too long and the density too low.

**What is observed: groups.** Whoever sees one animal of a herd sees the herd, so animals that were [together](#groups) are **one observation**: the detection function is fitted to the groups, each at the distance of its middle from the line (two animals either side of the line have their middle on it), and the density of animals is the density of groups times the **mean group size** - whose own uncertainty is part of the confidence interval (Buckland et al. 2001, section 3.5). Twelve deer in one meadow counted as twelve sightings would claim a precision the survey does not have. A group seen from a line counts with all its members. With nothing but animals on their own - or a group distance of `0` - the result is the one for individual animals. If the larger groups were the ones seen from further away, the result says so: near the limit of the strip a large group is noticed more easily than a small one, which makes the mean group size, and with it the density, too high.

**How sure the encounter rate is.** Animals are clumped and lines cross different ground, so the number seen per metre varies from line to line by more than chance alone. With two or more lines - transect legs, or several flights - that variation is measured between them (Buckland et al. 2001); a Poisson figure, which assumes evenly spread animals and is too small otherwise, is used only when there is a single line. The report says which was used, and lists every line with its length and count.

**What explains detectability: canopy cover and flight height.** One detection function for everything says that the chance of seeing an animal falls off with its distance from the line the same way everywhere. From a drone it does not: under closed canopy an animal a few metres off the line is already hidden, and from twice the height it is half as many pixels. With **Let canopy cover and flight height explain detectability** ticked, the *width* of the detection function may depend on the conditions each animal was seen in (multiple-covariate distance sampling, Marques and Buckland 2003):

- **canopy cover** - the tree cover (percent) around the animal, within the *sampling radius* of the [habitat settings](environment.md#habitat-context-survey-analytics). Needs **E2 Land Cover** of the Environmental Context on every flight;
- **flight height** - how high the camera was above the animal on the frame its distance was measured on. Needs the flight's DEM (it is the frame's altitude less the ground the animal stood on).

Beside the two plain models, each of them is fitted with canopy cover, with flight height and with both, and the model with the **lowest AIC** is used - so a condition that explains nothing is left out, and ticking the box can never make the result worse than the plain one by more than chance. A condition is only offered where it is known for **every** observation and differs between them, and only from **20 observations** on; the result says which were used, which were not and why (a flight without the land-cover step, a height that never changed).

With a model that depends on the conditions, every observation stands for the animals its *own* strip width says it does - an animal found under the trees for more than one found in the open - and the density is their sum over the line surveyed. The effective strip width and the detection probability reported are the averages over the observations. The uncertainty is what the totals vary by between the lines (group sizes included) plus what not knowing the model's coefficients adds. The result window says what each condition does in the units of the survey - *"The more canopy cover, the less far animals are seen: the effective strip half-width is 38 m at 0 % and 11 m at 80 %"* - which is also what the next flight can be planned by: how close the lines have to be over forest, and what flying higher costs.

| Setting | Description |
|---------|-------------|
| **Truncation (m)** | Discard observations beyond this perpendicular distance. `0` = automatic (95th percentile) |
| **Let canopy cover and flight height explain detectability** | Off by default: the two plain detection functions. On: also the models whose width depends on the canopy over an animal and on the height it was flown over at; the lowest AIC decides (see above) |

On completion a results dialog summarises n (the number of groups, with the animals in them and the mean group size where animals were together), the length of surveyed line, truncation, the selected model, ESW, detection probability, density and abundance with CIs, plus the model-comparison (AIC) table.

**Outputs:**
```
analytics_t/    # or analytics_w/
└── distance_sampling_tracks.json   # or distance_sampling_detections.json
```

The JSON also stores the fitted detection-function curve (with conditions in the model: the average over the observations) and the distance histogram for external plotting, what the observations were: `n_groups`, `n_animals`, `mean_group_size` with its standard error and CV, `largest_group`, `group_density_per_km2` and the size-bias check (`group_size_bias`), and - where they were asked for - `covariates`: the conditions offered (`considered`), those left out and why (`not_used`), those in the chosen model (`in_best_model`) and for each of these its coefficient per unit with standard error and p, and the strip half-width at a low and a high value (`effects`). Abundance is reported for the covered strip area (2·w·L); multiply the density estimate by your study-area size for a study-area abundance.

### Coverage map

Combines the exported per-frame GeoTIFFs on the same grid as the **Orthomosaic**, but instead of merging image content it counts, per output pixel, how many frames contain valid (non-nodata) data at that position. The result is a single-band raster where `1` means the ground was imaged once and `N` that it was seen in `N` overlapping frames - a map of survey effort/overlap. Run **Export Frames as GeoTIFF** for the chosen camera first.

**Inside a survey area, `0` is a value: ground no frame saw - a gap in the survey.** The map reaches as far as the survey plot does, also where no frame went, and everything outside the plot (and inside an exclusion zone) is no data. Without a survey plot, ground no frame saw is no data: nothing says it was to be surveyed. (Before 9.1 `0` was the no-data value, so a gap inside the survey area was transparent like the world outside it, was left out of every statistic taken from the raster, and was not on the map at all where the plot reached beyond the frames.)

| Setting | Description |
|---------|-------------|
| **Cell (m)** | Output raster cell size in metres. `0` = native resolution of the exported GeoTIFFs (larger output) |

**Outputs:**
```
analytics_t/    # or analytics_w/
├── coverage_map.tif    # uint16 GeoTIFF - overlapping frame count per pixel (no data: 65535)
└── coverage_map.json   # stats: frame count, max/mean overlap, covered & multi-covered area (ha), gap area and covered share of the survey area
```

Use **→ Add Coverage Map to QGIS** to load the raster in classes from light (seen once) to dark (seen most often) - one class per count up to eight frames, doubling from there. A gap in the survey area has a colour of its own, grey, and is listed as **not seen**. `mean_overlap` in the JSON is the mean over the covered ground; `gap_area_ha` and `survey_area_coverage_fraction` say how much of the survey area was not, and was, seen.

### Survey Orthomosaic

One orthomosaic over every flight ticked in the survey. It is the flight's
**Orthomosaic** step (Pre-Processing tab) across flights: the same merge of the
exported per-frame GeoTIFFs, with the **merge mode** of
*Configuration → Orthomosaic*, written with the same compression and
overviews - a survey of one flight gives exactly that flight's orthomosaic.

- The camera is the survey's camera, and *Export Frames as GeoTIFF* has to
  have run for it in every ticked flight.
- The flights are merged in the order of the flight list and each flight's
  frames in frame order; that is what the *first* and *last* merge modes refer
  to where flights overlap.
- The frame range of the Orthomosaic tab belongs to a single flight and is not
  applied: all exported frames of every flight are used.
- The survey plot and exclusion zones do not clip it - it is an image of what
  was flown, not an analytic of the census area.
- All flights have to share one target CRS (as for every survey tool). The
  mosaic is built in memory at the GeoTIFFs' resolution, so several large
  flights need correspondingly more RAM than one.

**Outputs:**
```
analytics_t/    # or analytics_w/
├── orthomosaic.tif     # the mosaic over all ticked flights
└── orthomosaic.json    # flights, frames per flight, merge mode, size
```

Use **→ Add Survey Orthomosaic to QGIS** to load it into the *BAMBI Survey*
layer group.

### Population estimation (transects)

Density from the **transects** of the ticked flights, pooled into one
count / area table - the analysis of Praschl et al. 2026. Every transect
contributes one row: a *count* (the tracks assigned to it) and an *area* in
hectares (the ground it monitored).

- **Counts:** each track is assigned to the transect nearest in perpendicular
  distance. **Truncation (m)** drops tracks farther than that from every
  transect (`0` = assign every track to its nearest transect).
- **Areas:** the union of the field-of-view footprints of the frames inside the
  transect's frame range.

Requires, for every ticked flight: transects defined with the
[Transect Splitting Tool](tools.md#transect-splitting-tool), *Calculate Field
of View*, *Calculate Track Perpendicular* (*Calculate Perpendicular* when the
survey counts boxes) and the flight's DEM. That step supplies the positions;
what species each animal counts as is read when the estimate runs, so
classifying or relabelling afterwards needs no new distances.

| Estimator | |
|---|---|
| **Naive** | `sum(count) / sum(ha) × 100` - animals per 100 ha (= per km²) |
| **Bootstrap** | transects resampled with replacement (*Bootstrap iterations*, reproducible with *Seed*); mean, SE and percentile 95 % CI of the naive density |
| **ZINB** | zero-inflated negative binomial regression `count ~ ha`, for surveys where most transects are empty; SE by the delta method, log-normal 95 % CI |

**How sure the ZINB density is.** The density is the sum of the transects' fitted counts over their summed area. Those fitted counts all come from the same four model parameters, so they rise and fall together, and the uncertainty of their sum is far larger than what one gets by adding up each transect's own uncertainty - which is what the reference R analysis does, and what this estimate reproduced up to 9.0. An interval built that way held the true density about one time in four in simulations with 40 transects; the standard error now reported is that of the sum itself and held it about nine times in ten. The density has not changed and still agrees with `glmmTMB`; its standard error and interval are wider than those in reports made with earlier versions (on the paper's St. Stefan data: 32.7 ± 9.2 per 100 ha, interval 19 to 56, where ± 2.5 and 28 to 38 used to be shown). The interval is log-normal, so it does not reach below zero. It needs transects to be right: with fewer than 30 the estimate says that its interval is likely too narrow still (it held the true density about 6 times in 10 with 12 transects and 8 in 10 with 20) and points to the bootstrap.

**A flight without animals counts.** A flight on which nothing was detected, or none of the selected species, or nothing inside the survey area, contributes its transects with zero animals - leaving it out would overstate the density. Tick it like any other flight; it only needs its transects and its field of view.

**Study area (ha)** turns the density into an abundance: enter it, or take
it from the flights' field-of-view area (**Use flight FoV area**) or from the
survey area (**Use survey area**).

**→ Add Transect Areas to QGIS** loads each transect's monitored area with
its tracks, route and footprint, labelled with name and count. **→ Show
Report** opens the estimates.

**Outputs:**
```
analytics_t/    # or analytics_w/, in the survey folder
├── population_estimate.json    # estimates per method, the transect table
├── population_tracks.csv       # which track (or box) was assigned to which transect
└── transect_areas.geojson      # monitored area per transect
```

### Track inventory

One row per tracked individual of the chosen camera, with everything the project knows about it side by side: species (as every survey tool counts it), sex and age with their votes, the [group](#groups) it was in (named after the lowest track number among its members) and that group's size, box and occlusion counts, confidences, first and last frame with their capture times, start and end position (project CRS and WGS84), path length, the matched track on the other camera, labelling-tool annotations and the distance to the flight line. Two columns say where an answer came from: **Detector species** (`detector_species`) is what the detector's own classes called the animal, whatever a classifier or a person made of it since, and **Set by hand** (`by_hand`) lists what a person decided on the track - `species`, `sex`, `age` - in the [review queue](#review-queue). **→ List Tracked Individuals** writes it as `analytics_t/track_inventory.csv` and `.json`; **→ Add to QGIS** loads it as a point layer at each animal's last position.

**→ Show Report** opens the inventory as a sortable, filterable table built from the store as it stands now. Its first column, **Approved**, is your own verdict - a checkmark written to the track, so it survives, exports and appears in every later inventory. A click on a track's start or end frame opens the animal in the inspector at that frame, where the same facts are shown under the image and a wrong detection or track can be [deleted from the project](tools.md#detection--track-selection); the report follows what is done there. **Columns…** chooses which columns the table shows (saved with the QGIS project); the written CSV and JSON keep every column regardless.

### Review queue

Nobody looks at every track of a survey. Most are plain - a confident detector, a long track, a classifier in no doubt, both cameras agreeing - and a few are not. **→ Review Queue…** (next to *Show Report*) lists the few for the flights and the camera chosen above, the most doubtful first, each with why it is in doubt:

| Why | A track is listed when |
|-----|------------------------|
| boxes of low confidence | the mean confidence of its boxes is below **confidence below** (default 0.40) |
| only *n* boxes | it has fewer boxes than **fewer boxes than** (default 4) - a few boxes are what a false alarm looks like |
| no species | nobody knows what it is: no species, `animal` or `unknown` |
| species / sex by *n* % of the votes | a classifier called it by less than **votes below** (default 70 %) |
| the detector calls it … | the detector's own classes say another species than the track carries |
| the other camera says … | the track it was [matched](#a3-match-rgb--thermal-tracks) with carries another species or sex |
| not seen by the other camera | the cameras of this flight were matched and this track found no partner |
| hidden on *n* of *m* frames | it was occluded on more than half of its frames |

A track somebody **approved**, a track that was **labelled by hand** and one already ruled out as `not-an-animal` are never listed. Doubts that change a count (a faint or very short track, one camera only) come before doubts that change a label (a narrow vote), and a track with several doubts before a track with one. The three limits can be changed at the top of the window; **Refresh** builds the queue again from the project as it is now.

Selecting a row opens the track in the [inspector](tools.md#detection--track-selection) at its first frame (untick *Show in the inspector* to stop that). Then say what it is:

- **✓ Approve** (key **A**) - the track is right as it stands. It is ticked as approved, as in the inventory report, and leaves the queue.
- **✗ Not an animal** (key **X**) - rules it out: the track and its boxes become `not-an-animal`, which no survey tool counts. Nothing is deleted, so the verdict can be taken back by choosing another species.
- **Species**, **Sex**, **Age** - correct what the track carries; sex and age can also be set to *(not known)*. The track stays in the list so that you can approve it once it is right.
- **Skip** (key **N**) - leaves the track as it is and goes on; it is listed again next time.

Each of these is written to the project at once and logged. A track approved or deleted in the inspector leaves the queue as well.

**What a person decided stands.** A species, sex or age set here is marked on the track as set by hand (the inventory's *Set by hand* column), and from then on a classifier run, a re-vote at another quorum and *Sync Labels Across the Match* leave that answer alone - on the track and on its boxes - while still filling in the ones nobody decided. (Before 9.1 only tracks drawn in the labelling tool were protected: a classifier run replaced whatever stood on a track the tracker had made.) The written inventory, census, density and population results are marked stale after a correction, since their counts may no longer hold; run them again when the review is done.

### Habitat context

Environment values (satellite indices, land cover, tree cover, terrain wetness, weather) at the tracked animals, compared with the ground the flights monitored: a table per animal and per transect, and charts. Needs the *Environmental Context* steps of the Pre-Processing tab on each flight. See [Environmental Context](environment.md#habitat-context-survey-analytics).

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

Each stage writes to a camera-specific subfolder (`_t` for thermal, `_w` for RGB), so thermal and RGB results coexist without overwriting each other.

Since 6.0 the results themselves live in GeoPackages, with the text files below
written alongside for compatibility - see
[Results, Flights and Export](results-and-export.md):

```
target_folder/
├── project.gpkg                             # Species, enums, custom fields,
│                                            # configuration, step status
├── bambi_t/                                 # Thermal results, one file per step
│   ├── detections.gpkg
│   ├── georeferenced.gpkg                   # …plus why a detection was dropped
│   ├── tracks.gpkg                          # Track membership by detection id
│   ├── fov.gpkg
│   ├── labels.gpkg
│   └── segmentation.gpkg
├── bambi_w/                                 # RGB, same set
│   └── ...
```

The text outputs below are still produced, and can be switched off under
**Output Configuration** once nothing of yours reads them:

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
├── labels_t/                                # Manual labels (Labelling Tool, thermal)
│   ├── labels.json                          # The labelling tool's copy of the label tracks (read from bambi_t/labels.gpkg)
│   └── labels.csv                           # Per-frame interpolated label export
├── labels_w/
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
├── segmentation_w/
│   └── ...
├── analytics_t/                             # Survey analytics (thermal camera)
│   ├── density_detections.tif              # Density heatmap raster (points/hectare)
│   ├── density_detections.json             # Density heatmap stats
│   ├── distance_sampling_detections.json   # Distance-sampling density/abundance estimate
│   ├── coverage_map.tif                    # Coverage map raster (overlapping frame count)
│   ├── coverage_map.json                   # Coverage map stats
│   ├── orthomosaic.tif                     # Survey orthomosaic (all ticked flights)
│   └── orthomosaic.json                    # Survey orthomosaic info
└── analytics_w/
    └── ...
```

> **Note:** Only the subfolders for camera/stage combinations you actually run will be created. Running a stage for only one camera leaves the other subfolder absent.

Detection and Re-ID models are stored globally in the QGIS profile directory and shared across all projects:

```
%APPDATA%\QGIS\QGIS3\profiles\default\bambi_deps\models\
├── thermal_animal_detector.pt               # YOLO detection model (thermal)
├── rgb_animal_detector.pt                   # YOLO detection model (RGB)
├── osnet_x0_5_bambi_thermal_omni.pt         # BAMBI Re-ID model
└── osnet_x0_25_msmt17.pt                    # BoxMOT default Re-ID model
```
