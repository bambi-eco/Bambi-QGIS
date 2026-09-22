# Tools

Besides the main processing pipeline, the plugin ships several companion tools available from the BAMBI toolbar and the plugin menu.

- [Video Creator](#video-creator)
- [Thermal Image Viewer](#thermal-image-viewer)
- [Labelling Tool](#labelling-tool)
- [Segmentation Tool](#segmentation-tool)
- [Classifier configuration](#classifier-configuration)
- [Interactive selection tools](#interactive-selection-tools)

## Video Creator

The Video Creator turns processed BAMBI results into a shareable MP4 video. Open it via the **Video Creator** toolbar button or the plugin menu.

![Video Creator](../images/video_creator.jpg)

Point it at a processing **target folder** (the one containing `frames_t/` / `frames_w/`) and compose a video from up to three side-by-side panels plus an info bar:

| Section | Options |
|---------|---------|
| **Frame source** | Extracted frames, or the orthographic projection (per-frame GeoTIFFs or the ALFS mosaic) |
| **Modality** | RGB, Thermal, Both (two video panels), or None (map only) |
| **Overlay** | None, detections, or tracks drawn onto the video panels |
| **Map view** | A panel rendered live from the QGIS project: flight path, current field of view, detections, tracks, perpendicular distances, merged FoV as background, a frame-by-frame accumulating FoV (area monitored so far), and an optional satellite/OSM background (needs internet) |
| **Info panel** | Bottom bar showing the current frame number, detection/track counts for the current frame, and monitored area (observed / total) |
| **Output** | MP4 file path, frames per second, panel height in pixels, and all frames or a frame range |

At least one video modality *or* the map panel is required. Rendering runs on the GUI thread (QGIS map rendering must happen there) but keeps the UI responsive; a Cancel button aborts cleanly.

## Thermal Image Viewer

The Thermal Image Viewer is a standalone non-modal dialog for inspecting individual DJI radiometric thermal images. Open it via the **Thermal Image Viewer** toolbar button or the plugin menu.

> **Requires DJI Thermal SDK**: install it via the [Dependency Manager](installation.md#dependency-manager).

![Thermal Image Viewer](../images/thermal_viewer.png)

Key features:

- Load any DJI radiometric thermal JPEG or TIFF directly
- Choose from multiple **colormaps** (`white-hotspot`, `black-hotspot`, `plasma`, `inferno`, `magma`, `viridis`, `jet`)
- Set optional **lower and upper temperature thresholds** (°C) that clip the display range, making it easier to isolate warm or cold targets
- Alternatively switch **Tone mapping** to **Curve (custom mapping)** for a fine-granular, tone curve over a fixed temperature range (with live histogram); the range can be typed in or auto-detected by scanning the opened image/folder
- Pixel temperature readout on hover

## Labelling Tool

The Labelling Tool is a key-frame based annotation editor for reviewing pipeline results and creating or correcting MOT-style track labels directly on the extracted frames. Open it via the **Labelling Tool** toolbar button or the plugin menu.

Point it at a processing **target folder** (the one containing `frames_t/` / `frames_w/`) - when opened from a configured plugin panel the folder, DEM, and correction paths are picked up automatically. Labelling is done per **modality** (thermal *or* RGB, never mixed); the selector at the top switches between the two and each modality keeps its own label set.

### Reviewing existing results

Detections and tracks from the pipeline are drawn as read-only overlays (toggleable via checkboxes). An existing pipeline track can be converted into an editable label track with **Import as label track**; the **Resample** value controls how many key frames are kept (`1` = every frame of the track becomes a key frame, `N` = only every N-th frame plus the first and last). **Import all as label tracks** converts every pipeline track at once, applying the same Resample setting to each.

**Copy labels from _&lt;other modality&gt;_** imports the label tracks you already made on the other camera (RGB ↔ thermal) into the current one - see [Cross-modality copy](#cross-modality-copy) below.

### Key frames & interpolation

Labels do not need to be drawn on every frame. A track stores boxes only on **key frames**; all frames in between are interpolated linearly and drawn with a dashed border. The typical workflow:

1. Press **New Track (N)** and drag a bounding box around the animal - this creates the track with a key frame at the current frame
2. Jump ahead with **Step >>** (stride set by the **Step** spinbox, e.g. 10 frames)
3. Drag/resize the interpolated box to fit - any edit automatically turns the frame into a key frame
4. Repeat; the timeline bar below the slider shows the track's range and its key frames (click it to jump). The key-frame list in the side panel is clickable too - each frame number is an anchor that jumps to that frame (stop frames red, current frame bold; long lists show a window around the current frame)

Each track carries a **species** (free-text combo with common defaults), **sex**, and **age** class (*Track classes* section - the identity of the animal, constant along the track). The **occlusion** level (*clear* / *occluded* - the values the occlusion classifier reports, so hand annotations and predictions share one vocabulary; extend it in the [Project Schema](results-and-export.md#the-project-schema) if you need finer grades), in contrast, is stored **per key frame** (it sits in the *Key frames* section next to the stop-frame toggle): an animal can be fully visible on one key frame and occluded on the next. Interpolated frames inherit the occlusion of the previous key frame; changing occlusion on an interpolated frame promotes that frame to a key frame. The current value is also shown in the status line while scrubbing.

Boxes can be moved/resized and classes changed at any time; **Set key frame (K)** freezes the current interpolated box (outside the track's range it copies the nearest key frame's box, extending the track), **Delete key frame** removes one (deleting the last key frame removes the track).

To place a key frame fully manually - including on frames before/after the track's current range - select the track and press **Draw key frame (B)**, then drag the new box on the canvas. This extends or corrects the track without geo-propagation.

When an animal temporarily disappears (e.g. under a tree) and reappears recognizably later, mark the last sighting as a **Stop frame (S)**: no boxes are interpolated between a stop frame and the next key frame - the track pauses and resumes there. Stop frames appear red in the timeline, the gap is left unshaded, and gap frames are excluded from all exports. To continue the track after the gap, place a key frame on the reappearance frame (**Draw key frame (B)**, or **K** to copy the nearest box, or geo-propagation).

### Geo-referenced propagation

Instead of manually re-drawing a box on a far-away frame, **Propagate box (geo)** ray-casts the current box onto the DEM (pixel → world) with the current frame's camera pose and back-projects it (world → pixel) into the frame at the configured **Frame offset**, creating a key frame there. The DEM mesh is loaded once on first use (may take a while). As with the geo-referenced FoV inspector, the result depends on calibration and correction accuracy - usually only small size adaptions are needed.

Several tracks can be propagated in one step: select them in the track list (Ctrl / Shift click, the same selection used for merging and deleting) and the button changes to **Propagate boxes (geo) - _n_ tracks**, projecting each selected track's box on the current frame. Tracks are handled independently - one whose box has no DEM intersection, or that has no box on the current frame at all, is skipped without aborting the others, and a summary afterwards lists per track what was created and what was skipped.

### Cross-modality copy

Wildlife is often easier to spot in one modality than the other (e.g. warm animals in thermal, or distinctive shapes in RGB). Once a modality is labelled, **Copy labels from _&lt;other modality&gt;_** (in the *Overlays* section, its caption follows the current modality) brings those tracks over instead of re-labelling from scratch.

Each key-frame box of every track in the other modality is projected onto this modality using the same geo-referenced propagation as above: it is ray-cast onto the DEM with the *source* camera and back-projected with *this* modality's camera. Because thermal and RGB frames are captured on the same clock but at different rates, the two are matched by **capture time** - for each source key frame the nearest-in-time frame of the current modality is used (source and target resolutions may differ).

The projected boxes are added as **new label tracks** (species/sex/age/occlusion and stop flags are carried over) rather than merged into existing ones, because - like single-frame propagation - they typically need small manual adaptions before export. A summary reports how many tracks and key frames were copied and how many were skipped (no DEM intersection, projected outside the target frame, or no time-matched frame). The other modality must already have been processed far enough to have `poses_<other>.json`, extracted frames, and `labels_<other>/labels.json`.

### Species, sex, age and custom fields

Since 6.0 the categorical inputs are **chosen from the project's vocabulary**,
not typed. Species, `sex`, `age` and `occlusion` come from the
[Project Schema](results-and-export.md#the-project-schema), and the **…** button
beside the species list - like the gear button - opens that editor without
leaving the tool. Adding a species there is a single step and does not disturb
anything already labelled.

This is what makes labels durable: species and enum values are referenced by
identity, so renaming one never changes the meaning of work already done. In
5.x a species typed into the box could renumber the others.

Free-text fields are still available - define a `string` custom field for
notes or collar ids. Unlike 5.x, custom fields are **not** confined to
`labels.json`: they travel through geo-referencing and tracking and reach the
exports.

### Saving & pipeline export

**Save Labels** (or the **Autosave** checkbox, which saves automatically shortly after every change) writes per modality:

| File                       | Contents                                                                                            |
|----------------------------|-----------------------------------------------------------------------------------------------------|
| `bambi_{t,w}/labels.gpkg`  | Key-frame label tracks and their attributes (source of truth since 6.0)                             |
| `labels_{t,w}/labels.json` | The same tracks in the 5.x format, still written                                                    |
| `labels_{t,w}/labels.csv`  | Per-frame interpolated export: `frame,track_id,x1,y1,x2,y2,species,sex,age,occlusion,keyframe`      |

**Add detections to project** turns the interpolated label boxes into detections the rest of the pipeline consumes. The detector's output is untouched - each producer owns its own rows - so the two can coexist and either can be re-run without disturbing the other.

Editing a label and exporting again **re-uses the detections it already produced**: a box you did not move keeps its identity, and only boxes that actually moved need geo-referencing again. Without that, one key-frame edit would invalidate the whole flight, which is why 5.x export was an all-or-nothing rewrite.

Label tracks are also written as **real tracks**, so labelled animals reach the survey analytics and the exports without re-running tracking. A track imported with *Import as label track* replaces the pipeline track it came from, so refining a detector track does not count the animal twice.

**Replace detections in project** is the destructive variant: after a confirmation dialog the detector's detections are removed and the labels remain. Re-run **Geo-Reference Detections** and **Track Animals** afterwards to rebuild the QGIS layers from the labels.

### Controls

| Input                   | Action                                     |
|-------------------------|--------------------------------------------|
| Mouse wheel             | Zoom                                       |
| Middle-button drag      | Pan                                        |
| Drag box edge / corner  | Resize (creates a key frame)               |
| Drag box interior       | Move (creates a key frame)                 |
| `←` / `→`               | Previous / next frame                      |
| `PageUp` / `PageDown`   | Jump forward / back by the step size       |
| `N`                     | Toggle New Track drawing mode              |
| `B`                     | Toggle Draw Key Frame mode                 |
| `K`                     | Set key frame from the interpolated box    |
| `S`                     | Toggle stop frame (pause interpolation)    |
| `Delete`                | Delete the key frame at the current frame  |
| `Esc`                   | Cancel drawing mode                        |

## Segmentation Tool

The Segmentation Tool prompts Meta's **SAM3** (or **SAM 3.1**) on the extracted frames and turns the masks into geo-referenced polygons. Open it via the **Segmentation Tool** toolbar button, the plugin menu, or **S1** on the Processing tab. Point it at a processing **target folder** (the one with `frames_t/` / `frames_w/`); opened from a configured plugin panel the folder and the DEM are picked up. The **Camera** selector switches between thermal and RGB; the **Source** selector between the extracted frames and the P6 **orthomosaic** (see below). Each camera keeps its own results in `segmentation_t/` or `segmentation_w/`; frame and orthomosaic results sit side by side there.

### Backends

| Backend | Prompts | Sequence tracking | Needs |
|---|---|---|---|
| **Local - transformers** (`facebook/sam3`) | text, points | yes (`Sam3VideoModel` / `Sam3TrackerVideoModel`) | `transformers >= 5.0`, a Hugging Face read token with access to the gated repository (the **HF token** field - one token shared with the Classification tab, entering it in either place fills both); CPU works, a GPU is much faster |
| **Local - Meta sam3 package** | text, points | yes - **SAM 3.1** with object multiplexing (~7x faster with many objects) | `pip install git+https://github.com/facebookresearch/sam3.git` (Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6) and the same token; the SAM 3.1 checkpoint exists only as a bare checkpoint on Hugging Face, so this is the only way to run it |
| **Remote - Roboflow API** | text | no | a Roboflow API key (kept in QSettings, never in the project) |

**Check access** asks Hugging Face whether the HF token may read the chosen checkpoint before anything is downloaded. Both local backends download their checkpoint (~3.4 GB) automatically on first use into the plugin's shared model cache (`<QGIS profile>/bambi_deps/models/hf_cache`, the same folder as the DINOv3 backbone), so one download serves every project; the Meta backend can also be pointed at a checkpoint file you already have.

### Prompts

- **Text**: one concept per line (`deer`, `wild boar`). SAM3 finds every instance of each concept; the result is one prompt entry per line.
- **Points**: left click *on* the object, right click *beside* it (a negative click). Every click belongs to the current **object**; press **New object** for the next animal, and name it if you like. One prompt entry per object is written, named after it.

The two kinds run separately - a text prompt finds all instances of a concept, a point prompt defines one object, and different heads of the model answer them.

### Frames

- **Current frame** segments the frame on the canvas.
- **Frame range** (from / to / step) segments every selected frame. With **Track across the sequence** ticked the frames are handed to the video tracker as one clip: an object found - or clicked on one frame - keeps its identity on the others and carries an `object_id`. Without it every frame is segmented on its own. For clips the frames are shrunk to SAM3's 1008 px working size in memory and the polygons scaled back, so a few hundred 4K frames stay feasible.

A run **merges** into the existing results: re-running `deer` replaces the deer masks on the frames it processed and leaves `boar` alone; the same holds for a clicked object run again with more points. **Delete results of this camera…** starts over.

### Orthomosaic

A canopy visible in ten frames is ten masks on the frames - and one on the orthomosaic, which shows every point on the ground once. With **Source: Orthomosaic** the tool reads `orthomosaic_{t|w}/orthomosaic.tif` (step P6), shows a decimated preview, and segments the mosaic in overlapping **tiles** (default 1024 px with 128 px overlap; the *Tiles* box shows how many, and how many metres a tile covers). Text-prompt masks from all tiles are then **merged**: a prompt's masks are painted into one raster at mosaic resolution (reduced to at most 8192 px a side) and its connected regions become the objects, so a tree cut by a tile border is one polygon with an `object_id`. Point prompts are answered on a tile centred on the clicks, one object each. Thermal mosaics are stretched between their 2nd and 98th percentile and shown as grey.

The mosaic is a GeoTIFF, so its pixels map to world coordinates by its own transform: there is no geo-referencing step, the geo file is written with the masks, and **Add to QGIS** puts one layer per prompt into `SAM3 Segmentation Orthomosaic (Thermal|RGB)`. The GeoJSON export marks these features with `source: orthomosaic` and `frame: -1`. Sequence mode does not apply; the frame range controls give way to the tile settings.

### Results

The masks are drawn over the frame, coloured per prompt, with the object id and confidence. Three follow-ups sit below:

- **Geo-reference onto the DEM** projects every polygon through the camera pose onto the DEM mesh (with the flight's correction factors), writing `segmentation_georef.json`. After a new run the geo file is marked *stale* until this is repeated.
- **Add to QGIS as layers** creates `SAM3 Segmentation (Thermal|RGB)` under the flight's group, one sub-group per frame and one polygon layer per prompt, with `prompt`, `prompt_type`, `object_id`, `frame` and `confidence` attributes.
- **Export GeoJSON…** writes WGS84 polygons (one feature per object and frame, `MultiPolygon` when a mask fell into several regions) with the same properties.

Backend, model, prompts, confidence and the Roboflow key are remembered in QSettings across sessions; they describe your machine rather than one flight, so they are not written into the project.

## Classifier configuration

The **Configuration → Classification** tab holds everything the [classification steps](pipeline.md#classification) read. Two dialogs open from it.

### Classifiers table

One row per task - occlusion, species, sex - in the order they run.

| Column | What it decides |
|--------|-----------------|
| **Input** | Which camera's features the classifier reads: *Thermal*, *RGB*, or *Matched* (both at once). Matched is the default, because it is where the two sensors complement each other most. |
| **Model** | *Default* (our published model, downloaded on first use), *Custom* (your own file), or *Off*. |
| **File / repository** | The model file, for a custom choice. |
| **Labels** | Opens the class mapping, below. |
| **Species…** | Sex and life stage: which classifier to use per species. |

All three have a published default: [occlusion](https://huggingface.co/cpraschl/bambi-occlusion-classifiers), [species](https://huggingface.co/cpraschl/bambi-species-classification) and [red deer sex](https://huggingface.co/cpraschl/bambi-red-deer-sex-classifiers). The species classifier knows **red deer, roe deer and wild boar** - anything else in a survey is forced into one of the three, so on a project with other animals check the class mapping, or leave the task off and label the species by hand. Its classes are named `red_deer`, `roe_deer` and `wild_boar`, which the mapping matches onto the project's `red deer`, `roe deer` and `wild boar` without any clicks.

**Download models** fetches every classifier set to *Default*, for the projection and input chosen above. Worth pressing first: with the files present, **Labels** can read each model's own class list and the mapping can be set up before anything is run. They are small - a few megabytes each. The DINOv3 model is *not* downloaded here; that happens on the first embedding run, and is much larger.

### Class mapping

Connects the classes a model returns to the values used in this project.

The mapping follows the class **order**, not the names. A model returns positions - "class 2 scored highest" - and the names are only there to read; renaming one here never re-points the mapping. That is also why a model does not have to name its classes at all.

**Read classes from the model** takes the list off the model. Where a model carries no names, it is asked how many classes it returns instead, and the rows appear as `class 0`, `class 1`… with the labels left for you to fill in. Where the model is not available locally yet, **Add class** defines them by hand.

With the project's own occlusion vocabulary (`clear` / `occluded`) the mapping fills itself in by name and needs no clicks. It exists for everything else: an older project whose occlusion values are `none` / `partially` / `fully`, or a third-party model that calls them something else entirely.

For occlusion the mapping additionally decides **which class means "this frame is usable"** - taken from whichever class you map onto the project's first occlusion value. Frame selection depends on it, so it is recorded rather than guessed from the word.

### Classifiers per species

Sex and life stage are not one problem across animals. The published sex classifier reads antlers on red deer; nothing about it transfers to a wild boar, and applying it anyway would produce confident nonsense rather than nothing.

So the choice is made per species. For **sex** the options are a model or **Off**, and Off means simply not sexed - the honest answer for a species nobody has a classifier for.

**Life stage** has a third option, **Size-based**: not a model at all, but the box-area measurement described under [C6](pipeline.md#c6-age-classification). It sits in the same column because it answers the same question, and because which of the two decides a species is one decision rather than a model choice plus a switch elsewhere.

Red deer is the only species with a released sex model, so it is the only sex row switched on by default. No life-stage model has been published yet, so every species there defaults to Size-based.

## Interactive selection tools

Three map-canvas click tools are available from the BAMBI toolbar to inspect layers that have already been loaded into QGIS.

### Detection / track selection

Click the **Select Detection or Track** tool in the toolbar, then click anywhere on the map canvas to highlight the detection or track bounding box nearest to the clicked point.

- Works on any layer that was added via **→ Add Detections to QGIS** or **→ Add Tracks to QGIS**
- The clicked feature is selected in the layer and its attributes are shown in the inspector panel
- If no detection or track layers are present in the QGIS layer hierarchy, a warning dialog is shown

![Detection Selection Tool](../images/selection_tool_detections.png)

### Field-of-view selection

Click the **Select Field of View** tool in the toolbar, then click on the canvas to select the FoV(s) that contain the clicked point.

- Works on any layer that was added via **→ Add FoV Layers to QGIS** (not the merged FoV!)
- If no Field of View layers are present, a warning dialog is shown

There are two versions of the FoV tool:

1. A fast version that just opens the related FoVs
2. A **geo-referenced** version that also geo-references your click and shows it as a yellow cross. This requires loading the digital elevation model, which takes some time, and the result depends heavily on your calibration parameters and correction factors.

![Field-of-View Selection Tool](../images/selection_tool_fov.png)

![Field-of-View Selection Tool Geo-referenced](../images/selection_tool_fov_geo.png)
