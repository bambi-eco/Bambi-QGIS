# Custom Implementations

Detection, tracking and every classification stage can be replaced by your own
code (since 9.0.0). A custom implementation is **one `.py` file with a class**
that follows a small interface; the plugin loads it, hands it frames,
detections or crops, checks what it returns and keeps doing everything else -
frame selection, storage, geo-referencing, voting, class mapping, layers and
exports.

| Stage | Built-in | A custom class replaces |
|---|---|---|
| Detection (A1) | Ultralytics YOLO | "image → boxes" |
| Tracking (A2) | built-in / BoxMOT / geo-referenced trackers | "detections → tracks" |
| Classification (C2-C5) | triplet-net heads on DINOv3 embeddings | "frames of an animal → class" |

## Getting started

1. **Configuration → Detection → Download sample implementations…** saves
   ready-to-edit files into a folder of your choice:

   | File | What it is |
   |---|---|
   | `detector_ultralytics_yolo.py` | the built-in detector as a custom class |
   | `detector_owl.py` | Microsoft's Overhead Wildlife Locator (MegaDetector-Overhead); locates points, turned into boxes of a size you set |
   | `detector_dronemegadetector.py` | DroneMegaDetector (RF-DETR; person / animal / vehicle), with tiling for large frames |
   | `tracker_iou.py` | a small greedy IoU tracker, no extra packages |
   | `classifier_embeddings_head.py` | the built-in kind of classifier: a TorchScript head on the DINOv3 embeddings |
   | `classifier_crops_torchscript.py` | an image classifier working on the cut-out boxes, no embeddings needed |
   | `classifier_track_level.py` | decides once per animal instead of per frame; runs without any model file |

   The OWL and DroneMegaDetector samples follow the projects' own inference
   code but **have not been run against the real models by the plugin
   authors** - treat them as a starting point and check the results.
2. Copy one, adapt it, and select it:
   - **Detection**: *Configuration → Detection → Custom detector*, per camera.
   - **Tracking**: *Configuration → Tracking → Custom tracker*.
   - **Classification**: in the model table of *Configuration →
     Classification* choose **Custom implementation…** as the model of a
     task, then **…** to pick the file. For sex and age the same entry is
     available per species (**Species…**).
3. Press **Check**: it tests the implementation step by step and lists
   what happens in a log below the button - the required packages, the
   import, `setup()`, and for a detector a test run on an empty image - and
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
finite, a track id for a detection that was not passed in. Any exception in
your code stops the step with the exception's text.
