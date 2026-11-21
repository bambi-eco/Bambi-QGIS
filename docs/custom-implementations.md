# Custom Implementations

Detection, tracking, the embedding model and every classification stage can be
replaced by your own code (since 9.0.0; the embedding model since 9.1.0).
Detections or tracks made by another tool can be imported with it, and
detections, tracks and segmentation masks exported in a format of your own
(9.1.0). A custom implementation is **one `.py` file with a class**
that follows a small interface; the plugin loads it, hands it frames,
detections or crops, checks what it returns and keeps doing everything else -
frame selection, storage, geo-referencing, voting, class mapping, layers and
exports.

| Stage | Built-in | A custom class replaces |
|---|---|---|
| Detection (A1) | Ultralytics YOLO | "image → boxes" |
| Tracking (A2) | built-in / BoxMOT / geo-referenced trackers | "detections → tracks" |
| Embeddings (C1) | DINOv3 | "cut-out animals → vectors" |
| Classification (C2-C5) | triplet-net heads on DINOv3 embeddings | "frames of an animal → class" |
| Export | COCO, YOLO, MOT, GeoJSON, Camtrap DP, … | "detections, tracks and masks → your files" |
| Import (in place of A1 / A2) | the TRex tracklet import | "another tool's file → boxes, with or without tracks" |

## Getting started

1. **Configuration → Detection → Download sample implementations…** saves
   ready-to-edit files into a folder of your choice:

   | File | What it is |
   |---|---|
   | `detector_ultralytics_yolo.py` | the built-in detector as a custom class |
   | `detector_owl.py` | Microsoft's Overhead Wildlife Locator (MegaDetector-Overhead); locates points, turned into boxes of a size you set |
   | `detector_dronemegadetector.py` | DroneMegaDetector (RF-DETR; person / animal / vehicle), with tiling for large frames |
   | `tracker_iou.py` | a small greedy IoU tracker, no extra packages |
   | `embedder_colour_histogram.py` | the smallest embedding model - a colour histogram per crop, no extra packages - with a torchvision ResNet as an example in its header |
   | `exporter_csv.py` | exports detections, tracks and segmentation masks as CSV / JSON, no extra packages |
   | `importer_csv.py` | imports detections or tracks from a CSV file, no extra packages |
   | `classifier_embeddings_head.py` | the built-in kind of classifier: a TorchScript head on the DINOv3 embeddings |
   | `classifier_crops_torchscript.py` | an image classifier working on the cut-out boxes, no embeddings needed |
   | `classifier_track_level.py` | decides once per animal instead of per frame; runs without any model file |

   The OWL and DroneMegaDetector samples follow the projects' own inference
   code but **have not been run against the real models by the plugin
   authors** - treat them as a starting point and check the results.
2. Copy one, adapt it, and select it:
   - **Detection**: *Configuration → Detection → Custom detector*, per camera.
   - **Tracking**: *Configuration → Tracking → Custom tracker*.
   - **Embeddings**: *Configuration → Classification → Custom embedding
     model*.
   - **Export**: choose the format **Custom exporter…** under *Processing →
     Export* (one flight) or *Survey Analytics → Export* (several flights
     with splits); the file is picked in the **Custom exporter** box of
     *Processing → Export*.
   - **Import**: *Configuration → Tracking → Custom import*, with the file
     or folder to read as **Source**, then **Import now**.
   - **Classification**: in the model table of *Configuration →
     Classification* choose **Custom implementation…** as the model of a
     task, then **…** to pick the file. For sex and age the same entry is
     available per species (**Species…**).
3. Press **Check**: it tests the implementation step by step and lists
   what happens in a log below the button - the required packages, the
   import, `setup()`, for a detector a test run on an empty image and for an
   embedder one on two black crops - and
   then what the class declares. It runs in the background, so QGIS stays
   usable while a model loads or is downloaded (the button counts the
   seconds; **Stop waiting** gives up on it). The log can be selected and
   copied (**Copy log**); when your code raises, it contains the full
   traceback.

Each selection has a **File**, an optional **Class** (only needed when the
file holds several fitting classes) and free-form **Options**: a JSON object
that arrives in `setup()` as `context.options`, e.g.
`{"weights": "C:/models/my_model.pt"}`.

**The file's folder belongs to it.** One `.py` file is selected, but it does
not have to stand alone:

- other `.py` files in the same folder can be imported by name
  (`import my_helpers`, `from my_model import Net`) - so an implementation can
  be split into several modules;
- weights, label lists and other data next to it are reached relative to the
  file: `os.path.join(os.path.dirname(__file__), "weights.pt")`. That keeps an
  implementation a folder you can copy or hand on, without absolute paths in
  the options.

Mind two things: a helper module is found by its name, so do not give it the
name of an installed package (`torch.py`, `utils.py` next to another
implementation's `utils.py`); and the trust prompt covers the selected file
only - helper modules it imports run with it.

## Dependencies and safety

- **Say what the file needs.** A module-level list of import names,

  ```python
  REQUIRES = ["torch", "transformers"]
  ```

  lets **Check** look the packages up *before any of the file's code runs*
  and name the missing ones, instead of failing somewhere inside an import.
  All samples have it.

- **The plugin does not manage the dependencies of custom code.** Install
  what your file imports into the Python that QGIS uses: on Windows in the
  **OSGeo4W Shell**, on Linux / macOS in a terminal with QGIS' Python:
  `python -m pip install <package>`. A missing package is reported by name.
- **Installing packages for custom code can break other things.** QGIS, this
  plugin and every other plugin share one Python. `pip` happily replaces a
  package they rely on - torch, torchvision, transformers, numpy, opencv -
  with a version your new package asks for, and a step that worked yesterday
  then fails with an import error or a "procedure could not be found" from a
  mismatched library. To stay out of trouble:
  - install only what Check reports as missing, and avoid `--upgrade` unless
    an error message asks for a newer version;
  - before installing, look at what would change:
    `python -m pip install --dry-run <package>`;
  - prefer a model release that works with the versions you already have
    over upgrading torch;
  - afterwards open the **Dependency Manager**: it shows which of the
    plugin's packages are now out of range and can install its versions
    again;
  - a package that is installed but broken (for example a `torchaudio` that
    does not match `torch`) can make an unrelated import fail - removing it
    (`python -m pip uninstall <package>`) is often the fix.
- **A custom implementation is ordinary Python and runs inside QGIS with your
  rights.** Before a file runs for the first time on a computer - or the
  first time after it changed - the plugin asks. The approval is remembered
  on that computer only, never in the project: a flight folder you received
  may point at a file you have never looked at.
- Results record which implementation produced them (file, class and a hash
  of the file's content).

## The interface

Nothing has to be imported from the plugin. Every class has `setup(context)`,
called once per run, and may have `teardown()`.

`context` carries: `camera` (`"thermal"`, `"rgb"`, for classifiers also
`"matched"`), `device` (`"cuda"`, `"mps"`, `"cpu"`), `model_dir` (the plugin's
shared model folder), `options`, `log(message)`, `progress(fraction)`, and per
kind `min_confidence` (detector) or `task` (classifier).

**Progress bar.** For detectors and classifiers the plugin moves the bar
itself - per frame, per batch, per animal - so the class has nothing to do.
The same holds for a tracker called frame by frame (`update`). Only a tracker
that takes the whole flight in one call (`run`) has to report itself: it knows
how many frames and detections there are, so it calls
`context.progress(fraction)` (0.0 - 1.0) as it works through them.

**Batches.** A frame-level classifier is not given a whole flight at once:
`run` is called with up to 1024 embedding vectors or 64 crops per call, so
memory does not grow with the flight. A track-level classifier gets one
animal's frames per call - for a very long track on crops that is that
animal's crops in memory at once.

`setup` may return a dict (the same keys may be class attributes instead):

| Key | Meaning |
|---|---|
| `class_names` | what class index *i* means (required for classifiers) |
| `cameras` | the inputs it supports: any of `"thermal"`, `"rgb"` and, for classifiers, `"matched"`. Default: all. The plugin refuses to run it on an input it does not list. |

### Camera extrinsics

Every kind is told where the camera was, as a pose object - the frame's camera
**with the flight's correction factors applied**, i.e. the camera the plugin
itself geo-references and renders with:

| Where | |
|---|---|
| Detector | `frame.pose` |
| Tracker | `.pose` of every entry of `frames` |
| Classifier | `item["pose"]` of every row of `items` |

| Field | Meaning |
|---|---|
| `pose.position` | `(x, y, z)` in the mesh-local coordinates of the DEM |
| `pose.world` | the same in the project CRS (position + DEM origin); `None` when the DEM origin is unknown |
| `pose.quaternion`, `pose.matrix` | the camera rotation as `(x, y, z, w)` and as a 3 × 3 matrix (the camera looks along its −Z axis); `None` if it could not be computed |
| `pose.fovy` | vertical field of view in degrees |
| `pose.location`, `pose.rotation` | the uncorrected values of the poses file: `(x, y, z)` and `(tilt, roll, heading)` in degrees |
| `pose.correction` | the correction applied to this frame |
| `pose.raw` | the whole entry of the poses file (`lat` / `lng`, timestamp, … where the extraction wrote them) |

That is enough for geo-fencing ("only animals seen from inside this area"),
for weighting by viewing angle or height, or for a tracker's own motion model.

### Detector

```python
class MyDetector:
    def setup(self, context):
        return {"class_names": ["animal"], "cameras": ["rgb"]}

    def run(self, image, frame):
        # image: H x W x 3 BGR uint8 (as cv2.imread returns it)
        # frame.index, frame.path, frame.camera, frame.timestamp, frame.pose
        return [(x1, y1, x2, y2, confidence, class_index), ...]
```

Boxes are in pixels of the image passed in; they are clipped to it, and boxes
below the tab's *Min Confidence* are dropped. With the Detection tab's
**Projection** on *Orthorectified* or *ALFS* that image is the frame's GeoTIFF
or its light-field integral (`context.projection` is `"orthographic"` /
`"alfs"`); nothing changes for the class - it still returns pixels of the image
it was given, and the plugin places the boxes in the camera frame. A class name is accepted in
place of the index. Class names are mapped to the project's species with
**Edit Class Mapping…**, as for the built-in detector (spaces in a class name
become underscores).

### Tracker

A tracker is called **frame by frame** (recommended) or **once with the whole
flight**:

```python
class OnlineTracker:                       # frame by frame
    def setup(self, context):
        return {"space": "pixel",          # or "geo" (default)
                "interpolate": True,       # gaps may be interpolated (default)
                "images": False}           # True: frame.image is loaded

    def update(self, frame, detections):   # once per frame, in frame order
        return {detection_id: track_id, ...}

    def finish(self):                      # optional: revise earlier answers
        return None


class OfflineTracker:                      # the whole flight in one call
    def setup(self, context):
        self.progress = context.progress
        return {"space": "geo"}

    def run(self, detections, frames):
        return {detection_id: track_id, ...}
```

- **Frame by frame (`update`).** The plugin owns the loop: it calls `update`
  for every frame in order - also frames without detections, so tracks can
  age - moves the progress bar, can be cancelled between frames, and with
  `"images": True` loads the frame as `frame.image` (BGR) for the call.
  `finish()` may return a final `{detection_id: track_id}` that overrides the
  earlier answers; mapping a detection to `None` takes it out of its track.
  This is how BoxMOT-style trackers work, so wrapping one takes a few lines.
- **Whole flight (`run`).** For trackers that need to see everything at once
  (global association). The plugin cannot know how far it is, so the tracker
  reports it: `context.progress(fraction)`, 0.0 - 1.0.

A detection is a dict with `detection_id`, `frame`, the pixel box `x1 y1 x2
y2`, `confidence`, `class` and `world` (the box corners `(x1, y1, z1, x2, y2,
z2)` in world coordinates, or `None`). With `"space": "geo"` only
geo-referenced detections are passed and *Geo-Reference Detections* has to
have run; with `"pixel"` all of them. A frame has `index`, `path`,
`timestamp`, `duplicated` and `pose`. A detection left out (or mapped to
`None` / a negative id) stays untracked.

When a custom tracker is ticked it replaces the tracker selected on the tab,
and a configured TRex folder is not imported.

**Interpolation.** Frames a track skips can be filled with linearly
interpolated boxes, as for the other trackers (in the track files; the
interpolated boxes are not detections). Two things decide it:

- the tracker says whether it is *applicable*: `setup` may return
  `"interpolate": False` - e.g. for a tracker that fills gaps itself, or one
  that re-identifies animals after long absences, where a straight line
  between two sightings would be wrong. Default: applicable.
- the **Interpolate gaps in the tracks** tick box of the Custom tracker box
  switches it on or off where it is applicable.

### Exporter

An exporter writes the detections, tracks and segmentation masks of a flight
in a format of your own. The plugin reads the store and hands everything
over as plain Python - species and enum values as names, never ids - and the
class writes its files:

```python
class MyExporter:
    def setup(self, context):
        return {"name": "My format"}

    def run(self, data, output):
        # output: a folder that exists and is yours to fill
        for detection in data["detections"]:
            ...
```

| `data[...]` | |
|---|---|
| `flight` | `name`, `folder`, `camera` (`"thermal"` / `"rgb"`), `crs` (`"EPSG:32633"` or `None`) |
| `frames` | per extracted frame: `index`, `file`, `path`, `timestamp`, `duplicated` |
| `frame_size` | `(width, height)`, or `None` when it could not be read |
| `species` | the species names that occur |
| `detections` | `detection_id`, `frame`, `box` = `(x1, y1, x2, y2)` in pixels, `confidence`, `species`, `track_id` (or `None`), `attributes` (sex, age, approved, custom fields), `world` = the box corners `(x1, y1, z1, x2, y2, z2)` in the project CRS, `lonlat` = `(lon, lat)` of its centre |
| `tracks` | `track_id`, `species`, `attributes`, `detection_ids` (in frame order), `first_frame`, `last_frame` |
| `segmentations` | `segment_id`, `detection_id` (or `None`), `frame`, `polygon` (rings of `[x, y]` in pixels), `world` (the same in the project CRS), `lonlat` (in WGS84), `attributes` |
| `options` | `include_images`, `include_not_an_animal` - the export's tick boxes |

- `world` and `lonlat` are `None` for what has not been geo-referenced;
  `lonlat` also needs the target CRS to be set.
- **What is in `data` follows the export**, as for every format: detections
  labelled `not-an-animal` only when the box is ticked, and - in a survey
  with *Only animals confirmed by the other camera* - only the boxes, tracks
  and masks of confirmed animals. A mask of a detection that is not exported
  is left out with it.
- **Images** are the exporter's business: `options["include_images"]` says
  whether they are wanted, `frame["path"]` where each one is.
- **Several flights:** in *Survey Analytics → Export* the exporter runs once
  per flight, into `<dataset>/<split>/<flight>/`; `splits.json` and
  `dataset.json` are written by the plugin.
- The segmentation tool keeps its own GeoJSON export; a custom exporter gets
  the masks of the flight's camera with everything else.

### Importer

An importer reads detections - or whole tracks - that another tool made,
from whatever that tool wrote. It is to any format what the TRex tracklet
import is to TRex:

```python
class MyImporter:
    def setup(self, context):
        return {"space": "frames"}         # or "raw": see below

    def run(self, source, frames):
        # source: the file or folder chosen as "Source"
        # frames: the extracted frames (index, path, timestamp, pose)
        return [{"frame": 12, "x1": 100, "y1": 80, "x2": 140, "y2": 130,
                 "confidence": 0.9, "class": "roe deer", "track_id": 3},
                ...]
```

- **A row** is a dict as above, or a tuple `(frame, x1, y1, x2, y2,
  confidence, class, track_id)` with the last three optional. `frame` is the
  index of the extracted frame; `frames` is there to translate a tool's own
  frame names or times into indices. Defaults: confidence 1, class
  `"animal"`, no track.
- **Tracks or detections.** With a `track_id` on any row the tracks are
  imported too and become the flight's tracks, as a tracker's would (a box
  without one stays a box of no track). Without any, only detections are
  imported and *Track Animals* can run on them with any tracker.
- **Pixels.** By default the boxes are in the pixels of the extracted frames.
  A tool that worked on the raw video returns `{"space": "raw"}` from `setup`
  and `{"detections": rows, "raw_size": (width, height)}` from `run`; the
  plugin undistorts the boxes with the camera's calibration (and the square
  crop, unless the aspect ratio is preserved), as it does for TRex.
- **What the plugin does with them.** The boxes are stored under a source of
  their own, so a later detector run does not remove them - and an import
  does not remove the detector's boxes: use one or the other for a camera, or
  both knowingly. Importing again replaces the earlier import. The boxes are
  then placed on the ground by *Geo-Reference Detections*, which the import
  runs itself, so imported animals are on the map and counted by every survey
  tool.
- **Left out, and said in the log:** boxes on frames that were not extracted,
  boxes without area, and boxes on duplicated (frozen) thermal frames.
- **Species.** A class named exactly as a species of the project (whatever
  its case) is that species. Any other class is `animal` until it is mapped
  with **Edit Class Mapping…**.
- The import works on the camera of the *Track* step and needs the frames
  extracted and the DEM set.

### Embedder

An embedder turns each cut-out animal into the vector the classifiers read -
step C1, in place of DINOv3:

```python
class MyEmbedder:
    def setup(self, context):
        return {"dim": 768,            # width of the vectors (optional)
                "batch_size": 32}      # crops per call (optional)

    def run(self, crops, items):
        # crops: list of H x W x 3 RGB uint8 arrays
        return vectors                 # (N, D) array, one row per crop
```

- **Crops** are cut as the Classification tab says - crop size, padding,
  letterbox and **Projection** - and arrive in batches of `batch_size`
  (default: the tab's batch size). `items` holds one dict per crop with
  `detection_id`, `track_id`, `frame` and `box` (the pixel box).
- **Vectors**: an `(N, D)` NumPy array or torch tensor, every row finite and
  of the same width. Without `dim` the plugin finds the width out with one
  black crop before the run.
- **Storage** is DINOv3's: per frame, resumable, only what is missing is
  computed. The run is recorded under the file's name, class and hash, so a
  changed file starts a new run instead of mixing two models' vectors.
- **The classifiers have to match.** The published classifiers were trained
  on DINOv3 vectors and are refused on anything else. With a custom embedder
  every task needs a classifier trained on its vectors: a TorchScript model
  file, or a custom classifier with `"input": "embeddings"`.
- A *matched* classifier gets the thermal and the RGB vector of an animal
  concatenated, as with DINOv3; the embedder itself sees one camera at a
  time (`context.camera`).

### Classifier

`setup` **must** say what the classifier reads and what it decides:

| Key | Values |
|---|---|
| `input` | `"embeddings"` - the DINOv3 vectors of step C1, an `(N, D)` float32 array (for `matched`: RGB and thermal concatenated). `"crops"` - the cut-out boxes, a list of RGB uint8 arrays of `crop_size` × `crop_size` (for `matched`: `(thermal, rgb)` pairs). **Step C1 is not needed for crops.** |
| `level` | `"frame"` (default) - `run(batch, items)` scores each row; the plugin picks the frames (occlusion) and takes the vote per animal. `"track"` - `run_track(batch, items)` gets one animal's frames at once and returns its class itself. |
| `crop_size` | side of the crops in pixels (default 224) |

```python
class FrameLevel:
    def setup(self, context):
        return {"input": "crops", "level": "frame",
                "class_names": ["roe deer", "red deer"], "crop_size": 224}

    def run(self, batch, items):
        # one result per row, in order - or an (N, classes) probability array
        return [(class_index, probability), ...]


class TrackLevel:
    def setup(self, context):
        return {"input": "embeddings", "level": "track",
                "class_names": ["female", "male"]}

    def run_track(self, batch, items):
        return (class_index, probability)      # or None for "no call"
```

`items` holds one dict per row with `detection_id`, `track_id`, `frame`,
`box` (the pixel box), `world` (its geo-referenced corners `(x1, y1, z1, x2,
y2, z2)`, or `None` when it has not been geo-referenced) and `pose` (the
camera of that frame).

- Crops are cut as step C1 cuts them (padding and letterboxing of the
  Classification tab), from the imagery chosen as **Projection** there:
  *Perspective* crops the extracted camera frames; the *Orthorectified*
  options crop the per-frame GeoTIFFs and *ALFS* the light-field integrals
  of ALFS sampling mode, where a box is placed through its world coordinates
  (so the GeoTIFF export or ALFS sampling mode and geo-referencing have to
  have run, and a detection without world coordinates has no crop). For
  ALFS a frame without an integral of its own uses the nearest central
  frame's. `context.projection` tells the class which it gets
  (`"perspective"` / `"orthographic"` / `"alfs"`). Frozen thermal frames and missing images give no crop.
- A **track-level** classifier is called with the frames the plugin selected
  for the animal (those the occlusion step left usable, or all). No per-frame
  results are stored and no vote is taken; `None` leaves the animal
  undecided. Occlusion is a per-frame question, so it needs a frame-level
  classifier.
- For **matched** crops with the *unmatched* setting on "use RGB" / "use
  thermal", the missing side of the pair is `None`.
- Class names are connected to the project's species / sex / age values with
  the **Labels** button, as for any model ("Read classes from the model"
  loads the implementation to ask it).

## What is checked

The plugin validates what comes back and names the file in the message: a box
with a non-numeric coordinate, a result count that differs from the number of
rows, a class index outside the class names, a probability outside 0-1 or not
finite, embedding vectors of differing width or with NaN, an imported row without a
frame or a coordinate, a track id for a detection that was not passed in. Any exception in
your code stops the step with the exception's text.
