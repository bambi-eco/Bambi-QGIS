# -*- coding: utf-8 -*-
"""Custom implementations of the detection, tracking, embedding and
classification stages, and of the import of detections and tracks made
elsewhere (9.0.0; embedding and import 9.1.0).

A custom implementation is **one ``.py`` file with a class** that follows
the interface below. The file does not have to stand alone: its folder is
importable, so helper modules next to it can be imported by name, and data
next to it (weights, label lists) is reached through
``os.path.dirname(__file__)``. Nothing has to be imported from the plugin - the class
is duck-typed, so it can be written and tested outside QGIS - and whatever
packages it needs are the user's to install into the Python QGIS runs
(the OSGeo4W Shell on Windows, or a terminal with the Python QGIS uses on
Linux / macOS: ``python -m pip install <package>``).

Interface (version 1)
---------------------

Every class has ``setup(context)``, called once per run before anything
else. It may return a dict describing itself; the same keys may instead be
class attributes::

    class_names   list of str - what class index i means
    cameras       which inputs it supports: any of "thermal", "rgb" and,
                  for classifiers, "matched" (both cameras of one animal).
                  Default: all.

``context`` (:class:`Context`) carries ``camera``, ``device``,
``model_dir``, ``options`` (the free-form JSON from the configuration),
``log(message)``, ``progress(fraction)`` and, per kind, ``min_confidence``
/ ``task``.

``context.progress(fraction)`` (0.0 - 1.0) moves the plugin's progress bar.
A tracker should call it as it works through the frames - it is handed
everything in one call, so the bar cannot move otherwise. Detectors and
classifiers need not: the plugin advances the bar per frame / batch /
animal itself.

**Detector** ::

    run(image, frame) -> [(x1, y1, x2, y2, confidence, class_index), ...]

``image`` is an ``H x W x 3`` BGR ``uint8`` array, ``frame`` a
:class:`Frame` (index, path, camera). A class *name* is accepted in place
of the index. With the Detection tab's "Projection" on Orthorectified or
ALFS the image is the frame's GeoTIFF or light-field integral
(``context.projection`` is ``"orthographic"`` / ``"alfs"``); the boxes are
still simply pixels of the image passed in - the plugin places them in the
camera frame.

**Tracker** - frame by frame (recommended), or the whole flight at once::

    update(frame, detections) -> {detection_id: track_id, ...}     online
    run(detections, frames)   -> {detection_id: track_id, ...}     offline

*Online*: the plugin calls ``update`` once per frame, in frame order, with
that frame's detections (also for frames without any, so tracks can age).
The plugin owns the loop, so the progress bar, cancelling and - with
``"images": True`` from ``setup`` - loading the frame image
(``frame.image``, BGR) are its business. An optional ``finish()`` may
return a final ``{detection_id: track_id}`` that overrides what ``update``
answered, for trackers that revise earlier decisions.

*Offline*: a class without ``update`` gets everything in one ``run`` call -
for trackers that need the whole flight (global association). It should
report its progress through ``context.progress``.

A detection is a dict with ``detection_id``, ``frame``, pixel box ``x1 y1
x2 y2``, ``confidence``, ``class`` and, where geo-referencing has run,
``world`` = the box corners ``(x1, y1, z1, x2, y2, z2)``; ``frames`` /
``frame`` are :class:`Frame` objects. A detection left out (or mapped to
``None`` / a negative id) stays untracked. ``setup`` may return ``space``:
``"geo"`` (default: only geo-referenced detections are passed) or
``"pixel"`` (all of them), and ``interpolate``: whether filling the gaps of
a track with interpolated boxes makes sense for it (default True; it then
follows the Tracking tab's setting).

**Classifier** - ``setup`` *must* say what it reads and what it decides::

    input   "embeddings"  the DINOv3 vectors of step C1 (an (N, D) float32
                          array; for "matched", thermal and RGB concatenated)
            "crops"       the cut-out bounding boxes themselves (a list of
                          RGB uint8 arrays; for "matched", (thermal, rgb)
                          pairs) - step C1 is not needed
    level   "frame"       run(batch, items) scores each row; the plugin
                          picks the frames and takes the vote per animal
            "track"       run_track(batch, items) sees one animal's frames
                          at once and returns its class itself

    run(batch, items)        -> [(class_index, probability), ...]  one per row
                                (or an (N, C) probability matrix)
    run_track(batch, items)  -> (class_index, probability), or None for
                                "no call"

``items`` is one dict per row: ``detection_id``, ``track_id``, ``frame``,
``box`` (the pixel box), ``world`` (its geo-referenced corners ``(x1, y1,
z1, x2, y2, z2)``, or None) and ``pose`` (the camera of that frame).
``setup`` may also return ``crop_size`` (crops are resized to it, default
224). Crops come from the imagery chosen as "Projection" on the
Classification tab - the perspective camera frames, the orthorectified
per-frame GeoTIFFs or the ALFS light-field integrals;
``context.projection`` says which ("perspective" / "orthographic" /
"alfs").

**Embedder** - what turns a cut-out animal into the vector the
classifiers read (step C1), in place of DINOv3::

    run(crops, items) -> an (N, D) float array, one row per crop

``crops`` is a list of ``H x W x 3`` RGB ``uint8`` arrays, cut and padded
as the Classification tab says (crop size, padding, letterbox, projection);
``items`` is one dict per crop with ``detection_id``, ``track_id``,
``frame`` and ``box`` (the pixel box). Every vector must have the same
width D and be finite. ``setup`` may return ``dim`` (D; otherwise the
plugin finds it out with one black crop) and ``batch_size`` (how many
crops ``run`` is handed at most; default: the tab's batch size).

The vectors are stored as DINOv3's are, per frame, and recorded under the
file's name and hash - so a changed file starts a new run instead of
mixing two models' vectors. The published classifiers were trained on
DINOv3 and cannot read anything else: with a custom embedder every
classifier has to be one of your own, trained on its vectors (a
TorchScript file, or a custom classifier with ``input: "embeddings"``).

**Importer** - detections, or whole tracks, made by another tool, read
from whatever that tool wrote (as the TRex import does for TRex)::

    run(source, frames) -> [row, ...]
                           or {"detections": [row, ...],
                               "raw_size": (width, height)}

``source`` is the file or folder chosen next to the importer; ``frames``
the extracted frames as :class:`Frame` objects (index, path, timestamp,
pose), to translate the other tool's own frame names or times into frame
indices. A row is a dict::

    frame        index of the extracted frame the box is on
    x1 y1 x2 y2  the box in pixels
    confidence   0 - 1 (default 1)
    class        what it is - a name (default "animal")
    track_id     the track it belongs to; leave out (or None) for a box
                 of no track

or the same as a tuple ``(frame, x1, y1, x2, y2, confidence, class,
track_id)``, the last three optional. With a ``track_id`` on any row the
import records the tracks as well - they become the flight's tracks, as a
tracker's would; without, only the boxes are imported and any tracker can
run on them. ``setup`` may return ``space``: ``"frames"`` (default: the
boxes are in the pixels of the extracted frames) or ``"raw"`` (in the
pixels of the raw video; the plugin undistorts them with the camera's
calibration, and ``run`` then has to say the video's size as
``raw_size``). The boxes are placed on the ground by the Geo-Reference
step, as every other box.

Camera extrinsics
-----------------
Every kind is told where the camera was: a detector's ``frame.pose``, each
entry of a tracker's ``frames``, and ``pose`` in a classifier's ``items``
are a :class:`Pose` - the frame's camera **with the flight's correction
factors applied**, i.e. the camera the plugin itself geo-references and
renders with:

    pose.position     (x, y, z) in the mesh-local coordinates of the DEM
    pose.world        the same in the project CRS (position + DEM origin),
                      or None when the DEM origin is unknown
    pose.quaternion   camera rotation (x, y, z, w), or None
    pose.matrix       the same as a 3 x 3 rotation matrix (camera looks
                      along its -Z axis), or None
    pose.fovy         vertical field of view in degrees
    pose.location / pose.rotation   the uncorrected values of the poses
                      file ((x, y, z) and (tilt, roll, heading) in degrees)
    pose.correction   the correction applied to this frame
    pose.raw          the whole entry of the poses file (lat / lng,
                      timestamp, ... where the extraction wrote them)

All classes may define ``teardown()``.

Required packages
-----------------
A file may list the packages it needs as import names at module level::

    REQUIRES = ["torch", "transformers"]

"Check" looks them up first - without running the file - and names the
missing ones, instead of failing somewhere inside an import.
"""

import hashlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from . import quiet

#: Version of the interface described above.
API_VERSION = 1

DETECTOR = "detector"
TRACKER = "tracker"
CLASSIFIER = "classifier"
EMBEDDER = "embedder"
IMPORTER = "importer"
KINDS = (DETECTOR, TRACKER, CLASSIFIER, EMBEDDER, IMPORTER)

THERMAL, RGB, MATCHED = "thermal", "rgb", "matched"
#: The inputs a kind can be asked to work on.
CAMERAS = {
    DETECTOR: (THERMAL, RGB),
    TRACKER: (THERMAL, RGB),
    CLASSIFIER: (THERMAL, RGB, MATCHED),
    EMBEDDER: (THERMAL, RGB),
    IMPORTER: (THERMAL, RGB),
}

INPUT_EMBEDDINGS, INPUT_CROPS = "embeddings", "crops"
LEVEL_FRAME, LEVEL_TRACK = "frame", "track"
SPACE_GEO, SPACE_PIXEL = "geo", "pixel"
#: Where an importer's boxes are: the extracted frames, or the raw video.
IMPORT_FRAMES, IMPORT_RAW = "frames", "raw"
#: What an imported box is when its row does not say.
IMPORT_CLASS = "animal"

#: Methods a class needs to count as an implementation of a kind (a
#: classifier needs ``run`` or ``run_track``, checked against its level).
_REQUIRED = {
    DETECTOR: ("setup", "run"),
    TRACKER: ("setup",),              # plus update (online) or run (offline)
    CLASSIFIER: ("setup",),
    EMBEDDER: ("setup", "run"),
    IMPORTER: ("setup", "run"),
}

#: The source value that selects a custom implementation in a model spec.
SOURCE = "implementation"

DEFAULT_CROP_SIZE = 224


class CustomModelError(RuntimeError):
    """A custom implementation could not be loaded, or broke the interface.
    The message names the file, so it is not mistaken for a plugin fault."""


# ---------------------------------------------------------------------------
# What the implementation is handed
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Camera extrinsics of one frame, correction factors applied."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # mesh-local
    world: Optional[Tuple[float, float, float]] = None        # project CRS
    quaternion: Optional[Tuple[float, float, float, float]] = None
    matrix: Optional[List[List[float]]] = None                # 3 x 3
    fovy: float = 50.0
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # uncorrected
    rotation: Tuple[float, ...] = (0.0, 0.0, 0.0)            # uncorrected
    correction: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def make_pose(meta: dict, correction: Optional[dict] = None,
              origin: Optional[Sequence[float]] = None,
              corrected=None) -> Pose:
    """A :class:`Pose` from a poses-file entry.

    :param correction: ``{"translation": {x, y, z}, "rotation": {x, y, z}}``
        of the frame
    :param origin: the DEM origin ``(x, y, z)``; gives ``pose.world``
    :param corrected: ``(position, quaternion)`` of the plugin's camera
        model (``core.camera_pose.corrected_pose``). Without it the
        position is location + translation correction and the rotation is
        left out.
    """
    meta = meta or {}
    correction = correction or {}
    location = tuple(float(v) for v in
                     (list(meta.get("location") or [0.0, 0.0, 0.0])
                      + [0.0, 0.0, 0.0])[:3])
    rotation = tuple(float(v) for v in (meta.get("rotation") or [0, 0, 0]))
    fovy = meta.get("fovy", 50.0)
    if isinstance(fovy, (list, tuple)):
        fovy = fovy[0] if fovy else 50.0
    try:
        fovy = float(fovy)
    except (TypeError, ValueError):
        fovy = 50.0

    quaternion = matrix = None
    if corrected is not None:
        position_raw, rotation_raw = corrected
        position = tuple(float(v) for v in list(position_raw)[:3])
        quaternion = tuple(float(v) for v in list(rotation_raw)[:4])
        rows = getattr(rotation_raw, "matrix33", None)
        if rows is not None:
            matrix = [[float(v) for v in row] for row in rows]
    else:
        shift = correction.get("translation") or {}
        position = tuple(location[i] + float(shift.get(axis, 0.0) or 0.0)
                         for i, axis in enumerate("xyz"))
    world = None
    if origin is not None:
        origin = (list(origin) + [0.0, 0.0, 0.0])[:3]
        world = tuple(position[i] + float(origin[i]) for i in range(3))
    return Pose(position=position, world=world, quaternion=quaternion,
                matrix=matrix, fovy=fovy, location=location,
                rotation=rotation, correction=dict(correction),
                raw=dict(meta))


@dataclass
class Frame:
    """One extracted frame."""
    index: int
    path: str = ""
    camera: str = THERMAL
    timestamp: str = ""
    duplicated: bool = False       # frozen thermal image (core.poses)
    pose: Optional[Pose] = None    # camera extrinsics, corrections applied
    #: the frame as a BGR array - only for an online tracker that asked
    #: for images, and only during its ``update`` call
    image: Any = field(default=None, repr=False)


@dataclass
class Context:
    """What ``setup`` receives."""
    kind: str
    camera: str = THERMAL
    task: str = ""                      # classifier: occlusion / species / ...
    device: str = "cpu"
    model_dir: str = ""                 # the plugin's shared model folder
    options: Dict[str, Any] = field(default_factory=dict)
    min_confidence: float = 0.0         # detector
    #: the imagery handed over - "perspective" (camera frames),
    #: "orthographic" (orthorectified GeoTIFFs) or "alfs" (light-field
    #: integrals): a detector's images, a classifier's crops
    projection: str = "perspective"
    target_folder: str = ""
    api_version: int = API_VERSION
    log: Callable[[str], None] = field(default=lambda message: None,
                                       repr=False)
    #: report progress of the current call as a fraction 0.0 - 1.0
    progress: Callable[[float], None] = field(default=lambda fraction: None,
                                              repr=False)


# ---------------------------------------------------------------------------
# Specs (what the configuration stores)
# ---------------------------------------------------------------------------

def is_custom(spec: Optional[dict]) -> bool:
    """True when a model spec selects a custom implementation."""
    return bool(spec) and (spec.get("source") == SOURCE
                           or spec.get("model") == SOURCE)


def parse_options(text) -> Dict[str, Any]:
    """The free-form options of an implementation: a JSON object, or
    nothing.

    :raises CustomModelError: when the text is not a JSON object
    """
    if isinstance(text, dict):
        return dict(text)
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise CustomModelError(
            f"The options are not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CustomModelError(
            'The options must be a JSON object, e.g. {"weights": "model.pt"}')
    return value


def file_hash(path: str) -> str:
    """sha256 of the implementation file - its identity for the trust
    check, the provenance record and "has it changed since the last run"."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label(path: str, class_name: str, sha: str = "") -> str:
    """How results name the implementation that produced them."""
    sha = sha or (file_hash(path) if os.path.isfile(path) else "")
    return f"custom:{os.path.basename(path)}:{class_name}@{sha[:8]}"


# ---------------------------------------------------------------------------
# Required packages
# ---------------------------------------------------------------------------

def requirements(path: str) -> List[str]:
    """The packages an implementation file says it needs: its module-level
    ``REQUIRES = ["torch", "transformers"]`` (import names).

    Read from the source **without running the file**, so it can be shown
    before any of its code - or a slow import - is touched. A file without
    the list has no declared requirements.
    """
    import ast

    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "REQUIRES"
                   for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(value, str):
            value = [value]
        return [str(name) for name in value if str(name).strip()]
    return []


def installed(name: str) -> bool:
    """True when the package *name* (an import name) can be found - looked
    up without importing it, so it is quick even for torch."""
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def check_requirements(path: str) -> Tuple[List[str], List[str]]:
    """``(present, missing)`` of the packages *path* declares."""
    names = requirements(path)
    present = [name for name in names if installed(name)]
    return present, [name for name in names if name not in present]


INSTALL_HINT = (
    "Install it yourself - the plugin does not manage the dependencies of "
    "custom code: in the OSGeo4W Shell on Windows, or a terminal with the "
    "Python QGIS uses on Linux / macOS: python -m pip install <package>. "
    "Take care: installing or upgrading packages can change ones the "
    "plugin, QGIS or other plugins rely on (torch, transformers, numpy, "
    "...) and break steps that worked before - install only what is "
    "missing and check the Dependency Manager afterwards.")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _candidates(module, kind: str) -> List[type]:
    """Classes defined in *module* that have the methods of *kind*."""
    found = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue                       # imported, not defined here
        if not all(callable(getattr(obj, m, None)) for m in _REQUIRED[kind]):
            continue
        if kind == CLASSIFIER and not (
                callable(getattr(obj, "run", None))
                or callable(getattr(obj, "run_track", None))):
            continue
        if kind == TRACKER and not (
                callable(getattr(obj, "update", None))
                or callable(getattr(obj, "run", None))):
            continue
        declared = getattr(obj, "BAMBI_KIND", None)
        if declared and declared != kind:
            continue
        found.append(obj)
    return found


def load_module(path: str):
    """Import the implementation file as a module of its own.

    Its folder is put on ``sys.path`` so helper modules next to it can be
    imported.
    """
    path = os.path.abspath(path or "")
    if not os.path.isfile(path):
        raise CustomModelError(f"Custom implementation not found: {path}")
    if not path.lower().endswith(".py"):
        raise CustomModelError(
            f"A custom implementation is a single .py file: {path}")
    folder = os.path.dirname(path)
    if folder not in sys.path:
        sys.path.append(folder)
    name = "bambi_custom_" + hashlib.sha256(
        path.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CustomModelError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered so dataclasses / pickling inside the file work, and
    # replaced on every load so an edited file takes effect.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise CustomModelError(
            f"{os.path.basename(path)} needs a package that is not "
            f"installed in QGIS' Python: {exc}. Install it yourself - the "
            "plugin does not manage the dependencies of custom code "
            "(in the OSGeo4W Shell on Windows, or a terminal with the "
            "Python QGIS uses on Linux / macOS: python -m pip install "
            "<package>).") from exc
    except Exception as exc:  # noqa: BLE001 - any error in user code
        raise CustomModelError(
            f"{os.path.basename(path)} failed while being imported: "
            f"{type(exc).__name__}: {exc}") from exc
    return module


def find_classes(path: str, kind: str) -> List[str]:
    """Names of the classes in *path* usable as *kind*."""
    return [cls.__name__ for cls in _candidates(load_module(path), kind)]


def load_class(path: str, class_name: str, kind: str) -> type:
    """The implementation class of *kind* in *path*.

    *class_name* may be empty when the file holds exactly one candidate.
    """
    if kind not in KINDS:
        raise CustomModelError(f"Unknown kind of implementation: {kind}")
    module = load_module(path)
    candidates = _candidates(module, kind)
    base = os.path.basename(path)
    if class_name:
        for cls in candidates:
            if cls.__name__ == class_name:
                return cls
        raise CustomModelError(
            f"{base} has no {kind} class called '{class_name}'"
            + (f" (found: {', '.join(c.__name__ for c in candidates)})"
               if candidates else f" - a {kind} needs the methods "
               f"{', '.join(_REQUIRED[kind])}"))
    if not candidates:
        raise CustomModelError(
            f"{base} contains no {kind} class. A {kind} needs the methods "
            f"{', '.join(_REQUIRED[kind])}"
            + (" and run or run_track" if kind == CLASSIFIER else "")
            + (" and update (frame by frame) or run (whole flight)"
               if kind == TRACKER else "") + ".")
    if len(candidates) > 1:
        raise CustomModelError(
            f"{base} contains several {kind} classes "
            f"({', '.join(c.__name__ for c in candidates)}) - enter the "
            "class name.")
    return candidates[0]


# ---------------------------------------------------------------------------
# An opened implementation
# ---------------------------------------------------------------------------

def _info_value(instance, info: dict, key: str, default=None):
    if key in info and info[key] is not None:
        return info[key]
    value = getattr(instance, key, None)
    if callable(value):
        value = value()
    return default if value is None else value


class Implementation:
    """A loaded, set-up custom class with its interface checked.

    Calls into the user's code go through :meth:`call`, which turns any
    exception into a :class:`CustomModelError` naming the file.
    """

    def __init__(self, path: str, class_name: str, kind: str,
                 context: Context, status=None):
        """:param status: ``callable(text)`` told what is happening - the
        import and ``setup`` of a model can take minutes"""
        status = status or (lambda text: None)
        self.path = os.path.abspath(path)
        self.kind = kind
        self.context = context
        if not os.path.isfile(self.path):
            # before hashing, which would raise a bare FileNotFoundError
            raise CustomModelError(
                f"Custom implementation not found: {self.path}")
        self.sha = file_hash(self.path)
        status(f"Importing {os.path.basename(self.path)} (and the packages "
               "it imports)...")
        cls = load_class(self.path, class_name, kind)
        self.class_name = cls.__name__
        self.label = label(self.path, self.class_name, self.sha)
        try:
            self.instance = cls()
        except Exception as exc:  # noqa: BLE001
            raise self._error("could not be created", exc) from exc

        status(f"Running {self.class_name}.setup() - loading (or, the first "
               "time, downloading) a model can take a while...")
        info = self.call("setup", context)
        if info is None:
            info = {}
        if not isinstance(info, dict):
            raise CustomModelError(
                f"{self.label}: setup() must return a dict (or nothing), "
                f"not {type(info).__name__}")
        self.info = info

        names = _info_value(self.instance, info, "class_names", [])
        self.class_names: List[str] = [str(n) for n in (names or [])]

        allowed = CAMERAS[kind]
        cameras = _info_value(self.instance, info, "cameras", allowed)
        if isinstance(cameras, str):
            cameras = [cameras]
        self.cameras = tuple(str(c).lower() for c in cameras)
        unknown = [c for c in self.cameras if c not in allowed]
        if unknown:
            raise CustomModelError(
                f"{self.label}: unknown camera(s) {', '.join(unknown)} - a "
                f"{kind} can support {', '.join(allowed)}")
        if context.camera and context.camera not in self.cameras:
            raise CustomModelError(
                f"{self.label} does not support the "
                f"{context.camera} input (it supports: "
                f"{', '.join(self.cameras) or 'nothing'}). Choose another "
                "camera for this stage, or another implementation.")

        self.input = INPUT_EMBEDDINGS
        self.level = LEVEL_FRAME
        self.space = SPACE_GEO
        self.interpolate = True
        self.crop_size = DEFAULT_CROP_SIZE
        #: tracker: called frame by frame (``update``) rather than once
        self.online = False
        #: online tracker: wants ``frame.image`` loaded
        self.wants_images = False
        if kind == TRACKER:
            self.interpolate = bool(_info_value(
                self.instance, info, "interpolate", True))
            self.online = callable(getattr(self.instance, "update", None))
            self.wants_images = self.online and bool(_info_value(
                self.instance, info, "images", False))
        #: embedder: the width of its vectors (0 = not said) and how many
        #: crops it takes at once (0 = as the plugin is set)
        self.dim = 0
        self.batch_size = 0
        if kind == EMBEDDER:
            self._read_embedder_info()
        #: importer: the pixel space of its boxes
        self.import_space = IMPORT_FRAMES
        if kind == IMPORTER:
            self.import_space = str(_info_value(
                self.instance, info, "space", IMPORT_FRAMES)).lower()
            if self.import_space not in (IMPORT_FRAMES, IMPORT_RAW):
                raise CustomModelError(
                    f"{self.label}: space must be 'frames' or 'raw', not "
                    f"'{self.import_space}'")
        if kind == CLASSIFIER:
            self._read_classifier_info()
        elif kind == TRACKER:
            self.space = str(_info_value(
                self.instance, info, "space", SPACE_GEO)).lower()
            if self.space not in (SPACE_GEO, SPACE_PIXEL):
                raise CustomModelError(
                    f"{self.label}: space must be 'geo' or 'pixel', not "
                    f"'{self.space}'")

    def _read_embedder_info(self):
        for key in ("dim", "batch_size"):
            value = _info_value(self.instance, self.info, key, 0)
            try:
                value = int(value or 0)
            except (TypeError, ValueError) as exc:
                raise CustomModelError(
                    f"{self.label}: {key} must be a whole number") from exc
            if value < 0:
                raise CustomModelError(
                    f"{self.label}: {key} must not be negative")
            setattr(self, key, value)

    def _read_classifier_info(self):
        raw_input = _info_value(self.instance, self.info, "input")
        if raw_input is None:
            raise CustomModelError(
                f"{self.label}: setup() must say what the classifier reads "
                "- return {'input': 'embeddings'} or {'input': 'crops'}")
        self.input = str(raw_input).lower()
        if self.input not in (INPUT_EMBEDDINGS, INPUT_CROPS):
            raise CustomModelError(
                f"{self.label}: input must be 'embeddings' or 'crops', not "
                f"'{raw_input}'")
        self.level = str(_info_value(
            self.instance, self.info, "level", LEVEL_FRAME)).lower()
        if self.level not in (LEVEL_FRAME, LEVEL_TRACK):
            raise CustomModelError(
                f"{self.label}: level must be 'frame' or 'track', not "
                f"'{self.level}'")
        method = "run_track" if self.level == LEVEL_TRACK else "run"
        if not callable(getattr(self.instance, method, None)):
            raise CustomModelError(
                f"{self.label}: a '{self.level}'-level classifier needs a "
                f"{method}(batch, items) method")
        if not self.class_names:
            raise CustomModelError(
                f"{self.label}: a classifier has to name its classes - "
                "return 'class_names' from setup() or define class_names()")
        try:
            self.crop_size = int(_info_value(
                self.instance, self.info, "crop_size", DEFAULT_CROP_SIZE))
        except (TypeError, ValueError) as exc:
            raise CustomModelError(
                f"{self.label}: crop_size must be a number") from exc
        if self.crop_size < 8:
            raise CustomModelError(f"{self.label}: crop_size is too small")

    # -- calling the user's code --------------------------------------------

    def _error(self, what: str, exc: Exception) -> CustomModelError:
        if isinstance(exc, CustomModelError):
            return exc
        return CustomModelError(
            f"{self.label} {what}: {type(exc).__name__}: {exc}")

    def call(self, method: str, *args):
        try:
            return getattr(self.instance, method)(*args)
        except ImportError as exc:
            raise CustomModelError(
                f"{self.label} needs a package that is not installed in "
                f"QGIS' Python: {exc}. Install it yourself (in the "
                "OSGeo4W Shell on Windows, or a terminal with the Python "
                "QGIS uses on Linux / macOS: python -m pip install "
                "<package>).") from exc
        except Exception as exc:  # noqa: BLE001 - anything from user code
            raise self._error(f"failed in {method}()", exc) from exc

    def close(self):
        if callable(getattr(self.instance, "teardown", None)):
            try:
                self.instance.teardown()
            except Exception:  # best effort
                quiet.problem("The custom model's teardown() failed")

    def class_name_of(self, index: int) -> str:
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"class {index}"

    # -- detector -----------------------------------------------------------

    def detect(self, image, frame: Frame) -> List[dict]:
        """Boxes of one frame, checked and clipped to the image."""
        height, width = image.shape[0], image.shape[1]
        return normalise_detections(
            self.call("run", image, frame), width, height, self.class_names,
            self.context.min_confidence, self.label)

    # -- tracker ------------------------------------------------------------

    def track(self, detections: List[dict],
              frames: List[Frame]) -> Dict[int, int]:
        """Offline: everything in one call."""
        return normalise_assignment(
            self.call("run", detections, frames),
            [d["detection_id"] for d in detections], self.label)

    def track_frame(self, frame: Frame,
                    detections: List[dict]) -> Dict[int, int]:
        """Online: one frame's detections."""
        result = self.call("update", frame, detections)
        if result is None:
            return {}
        return normalise_assignment(
            result, [d["detection_id"] for d in detections], self.label)

    def track_finish(self, detection_ids: Sequence[int]):
        """Online: the tracker's optional final word (``finish()``).

        :return: ``(assignment, untracked)`` - what ``finish`` assigned, and
            the detections it explicitly took out of their track (mapped to
            ``None`` or a negative id)
        """
        if not callable(getattr(self.instance, "finish", None)):
            return {}, set()
        result = self.call("finish")
        if result is None:
            return {}, set()
        final = normalise_assignment(result, detection_ids, self.label)
        if isinstance(result, dict):
            mentioned = set(result)
        else:
            rows = list(result)
            mentioned = ({row[0] for row in rows}
                         if rows and isinstance(rows[0], (tuple, list))
                         else set(detection_ids))
        return final, {int(i) for i in mentioned} - set(final)

    # -- classifier ---------------------------------------------------------

    def classify_frames(self, batch, items: List[dict]
                        ) -> List[Tuple[int, float]]:
        return normalise_calls(self.call("run", batch, items), len(items),
                               len(self.class_names), self.label)

    def classify_track(self, batch, items: List[dict]
                       ) -> Optional[Tuple[int, float]]:
        return normalise_track_call(
            self.call("run_track", batch, items), len(self.class_names),
            self.label)

    # -- importer -----------------------------------------------------------

    def import_rows(self, source: str, frames: List[Frame]) -> dict:
        """What the other tool made, checked: ``{"rows", "raw_size",
        "left_out"}`` (:func:`normalise_imported`)."""
        return normalise_imported(
            self.call("run", source, frames), len(frames), self.class_names,
            self.label)

    # -- embedder -----------------------------------------------------------

    def embed(self, crops, items: Optional[List[dict]] = None):
        """One float32 vector per crop, checked."""
        items = items if items is not None else [{} for _ in crops]
        vectors = normalise_embeddings(
            self.call("run", list(crops), items), len(crops), self.dim,
            self.label)
        if not self.dim and len(vectors):
            self.dim = int(vectors.shape[1])
        return vectors


class Embedder:
    """A custom embedder in the shape of ``classification.Backbone``, so
    the embedding step treats DINOv3 and a custom model alike."""

    def __init__(self, implementation: Implementation, crop_size: int = 0):
        self.implementation = implementation
        #: what the embedding run is recorded under: the file, the class
        #: and the file's hash - another file is another run
        self.model_id = implementation.label
        self.crop_size = int(crop_size or DEFAULT_CROP_SIZE)

    @property
    def batch_size(self) -> int:
        return self.implementation.batch_size

    def load(self) -> None:
        """Find out how wide the vectors are, where ``setup`` did not say:
        one black crop through the model. Idempotent."""
        if self.implementation.dim:
            return
        import numpy as np

        size = self.crop_size
        self.implementation.embed([np.zeros((size, size, 3), np.uint8)])
        if not self.implementation.dim:
            raise CustomModelError(
                f"{self.implementation.label}: run() returned no vector "
                "for a crop, so the width of its vectors is unknown")

    @property
    def dim(self) -> int:
        return int(self.implementation.dim)

    def embed(self, crops, items: Optional[List[dict]] = None):
        return self.implementation.embed(crops, items)

    def close(self) -> None:
        self.implementation.close()


def is_custom_backbone(name) -> bool:
    """True when an embedding run was made by a custom embedder
    (:func:`label`) rather than by a Hugging Face model."""
    return str(name or "").startswith("custom:")


class ClassifierHead:
    """A custom classifier in the shape of ``classification.Head``, so the
    classification step treats built-in and custom models alike."""

    def __init__(self, implementation: Implementation):
        self.implementation = implementation
        #: what results record as their model
        self.path = implementation.label
        self.input = implementation.input
        self.level = implementation.level
        self.crop_size = implementation.crop_size
        self.class_source = "custom implementation"

    @property
    def classes(self) -> List[str]:
        return list(self.implementation.class_names)

    def _batch(self, features):
        if self.input == INPUT_EMBEDDINGS:
            import numpy as np
            array = np.asarray(features, dtype=np.float32)
            return array[None, :] if array.ndim == 1 else array
        return list(features)

    def predict(self, features, items: Optional[List[dict]] = None
                ) -> List[Tuple[int, float]]:
        if not len(features):
            return []
        items = items if items is not None else [{} for _ in features]
        return self.implementation.classify_frames(
            self._batch(features), items)

    def predict_track(self, features, items: List[dict]
                      ) -> Optional[Tuple[int, float]]:
        if not len(features):
            return None
        return self.implementation.classify_track(
            self._batch(features), items)

    def close(self):
        self.implementation.close()


def open_implementation(spec: dict, kind: str, context: Context,
                        status=None) -> Implementation:
    """Load and set up the implementation a model spec points at."""
    path = (spec.get("path") or "").strip()
    if not path:
        raise CustomModelError(
            f"A custom {kind} is selected but no .py file was chosen.")
    context.options = parse_options(spec.get("options"))
    return Implementation(path, (spec.get("class") or "").strip(), kind,
                          context, status)


# ---------------------------------------------------------------------------
# Checking what comes back
# ---------------------------------------------------------------------------

def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def normalise_detections(result, width: int, height: int,
                         class_names: Sequence[str],
                         min_confidence: float = 0.0,
                         who: str = "The detector") -> List[dict]:
    """``run()`` output as ``{x1, y1, x2, y2, confidence, class_index,
    class_name}`` dicts: boxes clipped to the image, empty ones and those
    below *min_confidence* dropped."""
    if result is None:
        return []
    names = list(class_names)
    out = []
    for row in result:
        try:
            values = list(row)
        except TypeError as exc:
            raise CustomModelError(
                f"{who}: run() must return a list of (x1, y1, x2, y2, "
                "confidence, class) rows") from exc
        if len(values) < 4:
            raise CustomModelError(
                f"{who}: a detection needs at least x1, y1, x2, y2 "
                f"(got {values!r})")
        if not all(_finite(v) for v in values[:4]):
            raise CustomModelError(
                f"{who}: a box has a non-numeric coordinate ({values[:4]!r})")
        x1, y1, x2, y2 = (float(v) for v in values[:4])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
            continue                          # outside the image, or empty
        confidence = float(values[4]) if len(values) > 4 \
            and _finite(values[4]) else 1.0
        confidence = min(1.0, max(0.0, confidence))
        if confidence < min_confidence:
            continue
        raw = values[5] if len(values) > 5 else 0
        if isinstance(raw, str):
            name = raw
            index = names.index(raw) if raw in names else -1
        else:
            if not _finite(raw):
                raise CustomModelError(
                    f"{who}: a detection's class is neither an index nor a "
                    f"name ({raw!r})")
            index = int(raw)
            name = names[index] if 0 <= index < len(names) \
                else (f"class {index}" if names else str(index))
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": confidence, "class_index": index,
                    "class_name": name})
    return out


_IMPORT_KEYS = ("frame", "x1", "y1", "x2", "y2", "confidence", "class",
                "track_id")


def _imported_row(entry, number: int, class_names: Sequence[str],
                  who: str) -> dict:
    """One row of an importer's answer as a dict with every key."""
    if isinstance(entry, dict):
        row = dict(entry)
        if "box" in row and "x1" not in row:
            try:
                row["x1"], row["y1"], row["x2"], row["y2"] = row["box"]
            except (TypeError, ValueError) as exc:
                raise CustomModelError(
                    f"{who}: row {number}: 'box' must be (x1, y1, x2, "
                    "y2)") from exc
        if "class" not in row and "class_name" in row:
            row["class"] = row["class_name"]
    elif isinstance(entry, (tuple, list)) and 5 <= len(entry) <= 8:
        row = dict(zip(_IMPORT_KEYS, entry))
    else:
        raise CustomModelError(
            f"{who}: row {number} must be a dict or a tuple (frame, x1, "
            "y1, x2, y2[, confidence[, class[, track_id]]]), not "
            f"{type(entry).__name__}")
    missing = [key for key in _IMPORT_KEYS[:5] if row.get(key) is None]
    if missing:
        raise CustomModelError(
            f"{who}: row {number} has no {', '.join(missing)}")
    try:
        frame = int(row["frame"])
        box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        confidence = (1.0 if row.get("confidence") is None
                      else float(row["confidence"]))
    except (TypeError, ValueError) as exc:
        raise CustomModelError(
            f"{who}: row {number} has a value that is not a number "
            f"({exc})") from exc
    if not all(_finite(value) for value in box + [confidence]):
        raise CustomModelError(
            f"{who}: row {number} has a coordinate or confidence that is "
            "not finite")
    if frame < 0:
        raise CustomModelError(
            f"{who}: row {number} is on frame {frame}; frames count from 0")
    name = row.get("class")
    if name is None or name == "":
        name = IMPORT_CLASS
    elif isinstance(name, (int,)) and not isinstance(name, bool) \
            and class_names:
        if not 0 <= name < len(class_names):
            raise CustomModelError(
                f"{who}: row {number} has class index {name}, and the "
                f"importer names {len(class_names)} class(es)")
        name = class_names[name]
    track = row.get("track_id")
    if track is not None:
        try:
            track = int(track)
        except (TypeError, ValueError) as exc:
            raise CustomModelError(
                f"{who}: row {number} has a track_id that is no whole "
                "number") from exc
        if track < 0:
            track = None
    return {"frame": frame,
            "x1": min(box[0], box[2]), "y1": min(box[1], box[3]),
            "x2": max(box[0], box[2]), "y2": max(box[1], box[3]),
            "confidence": min(1.0, max(0.0, confidence)),
            "class": str(name), "track_id": track}


def normalise_imported(result, n_frames: int = 0,
                       class_names: Sequence[str] = (),
                       who: str = "The importer") -> dict:
    """What an importer's ``run`` returned, checked.

    :param n_frames: how many frames were extracted; a box on a later
        frame has no picture and no pose and is left out (0 = not known)
    :return: ``{"rows": [{frame, x1, y1, x2, y2, confidence, class,
        track_id}], "raw_size": (width, height) | None, "left_out": boxes
        on frames that were not extracted, "empty": boxes without area}``
        - the rows ordered by frame
    """
    raw_size = None
    if isinstance(result, dict):
        if result.get("raw_size") is not None:
            try:
                raw_size = (int(result["raw_size"][0]),
                            int(result["raw_size"][1]))
            except (TypeError, ValueError, IndexError) as exc:
                raise CustomModelError(
                    f"{who}: raw_size must be (width, height)") from exc
            if min(raw_size) <= 0:
                raise CustomModelError(
                    f"{who}: raw_size must be (width, height) in pixels")
        if "detections" not in result:
            raise CustomModelError(
                f"{who}: run() returned a dict without 'detections'")
        result = result["detections"]
    if result is None:
        result = []
    try:
        entries = list(result)
    except TypeError as exc:
        raise CustomModelError(
            f"{who}: run() must return a list of rows, not "
            f"{type(result).__name__}") from exc

    rows, left_out, empty = [], 0, 0
    for number, entry in enumerate(entries):
        row = _imported_row(entry, number, class_names, who)
        if n_frames and row["frame"] >= n_frames:
            left_out += 1
            continue
        if row["x2"] <= row["x1"] or row["y2"] <= row["y1"]:
            empty += 1
            continue
        rows.append(row)
    rows.sort(key=lambda row: row["frame"])
    return {"rows": rows, "raw_size": raw_size, "left_out": left_out,
            "empty": empty}


def normalise_embeddings(result, rows: int, dim: int = 0,
                         who: str = "The embedder"):
    """What an embedder's ``run`` returned as an ``(rows, D)`` float32
    array: one finite vector per crop, all of one width - *dim* where that
    is known already."""
    import numpy as np

    if hasattr(result, "detach"):               # a torch tensor
        result = result.detach().cpu().numpy()
    try:
        array = np.asarray(result, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise CustomModelError(
            f"{who}: run() must return one vector per crop, all of the "
            f"same width ({exc})") from exc
    if rows == 0 and array.size == 0:
        return np.zeros((0, int(dim or 0)), dtype=np.float32)
    if array.ndim == 1 and rows == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise CustomModelError(
            f"{who}: run() must return an (N, D) array - one vector per "
            f"crop - not one of shape {tuple(array.shape)}")
    if array.shape[0] != rows:
        raise CustomModelError(
            f"{who}: run() returned {array.shape[0]} vector(s) for "
            f"{rows} crop(s)")
    if array.shape[1] < 1:
        raise CustomModelError(f"{who}: run() returned empty vectors")
    if dim and array.shape[1] != int(dim):
        raise CustomModelError(
            f"{who}: run() returned {array.shape[1]}-d vectors, not the "
            f"{int(dim)}-d it returned (or declared) before")
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array).all(axis=1)).sum())
        raise CustomModelError(
            f"{who}: {bad} of {rows} vector(s) contain NaN or infinite "
            "values")
    return array


def normalise_calls(result, rows: int, classes: int,
                    who: str = "The classifier") -> List[Tuple[int, float]]:
    """``run()`` output of a frame-level classifier as ``(class_index,
    probability)`` per row. Accepts that list directly or an ``(N, C)``
    matrix of probabilities."""
    if result is None:
        raise CustomModelError(f"{who}: run() returned nothing")
    shape = getattr(result, "shape", None)
    if shape is not None and len(shape) == 2:
        # an array is always a probability matrix; pairs come as a list
        if shape[1] != classes:
            raise CustomModelError(
                f"{who}: run() returned a {shape[0]} x {shape[1]} matrix, "
                f"but it has {classes} class name(s) - one column per class")
        import numpy as np
        matrix = np.asarray(result, dtype=float)
        calls = [(int(np.argmax(row)), float(np.max(row))) for row in matrix]
    else:
        try:
            calls = [(row[0], row[1]) for row in result]
        except (TypeError, IndexError) as exc:
            raise CustomModelError(
                f"{who}: run() must return one (class_index, probability) "
                "per row, or an (N, classes) probability matrix") from exc
    if len(calls) != rows:
        raise CustomModelError(
            f"{who}: run() returned {len(calls)} result(s) for {rows} "
            "row(s) - it must return exactly one per row, in order")
    return [_check_call(index, probability, classes, who)
            for index, probability in calls]


def _check_call(index, probability, classes: int, who: str
                ) -> Tuple[int, float]:
    if not _finite(index) or not _finite(probability):
        raise CustomModelError(
            f"{who}: a result is not a (class_index, probability) pair of "
            f"finite numbers ({index!r}, {probability!r})")
    index = int(index)
    if not 0 <= index < classes:
        raise CustomModelError(
            f"{who}: class index {index} is outside its {classes} class "
            "name(s)")
    probability = float(probability)
    if not -1e-6 <= probability <= 1.0 + 1e-6:
        raise CustomModelError(
            f"{who}: a probability has to be between 0 and 1 (got "
            f"{probability})")
    return index, min(1.0, max(0.0, probability))


def normalise_track_call(result, classes: int, who: str = "The classifier"
                         ) -> Optional[Tuple[int, float]]:
    """``run_track()`` output: ``(class_index, probability)`` or None."""
    if result is None:
        return None
    try:
        index, probability = result[0], result[1]
    except (TypeError, IndexError) as exc:
        raise CustomModelError(
            f"{who}: run_track() must return (class_index, probability) or "
            "None") from exc
    return _check_call(index, probability, classes, who)


def normalise_assignment(result, detection_ids: Sequence[int],
                         who: str = "The tracker") -> Dict[int, int]:
    """``run()`` output of a tracker as ``{detection_id: track_id}``.

    Accepts that dict, a list of ``(detection_id, track_id)`` pairs, or a
    list of track ids aligned with the detections passed in. Unknown
    detection ids are an error; ``None`` or a negative track id means
    "not tracked".
    """
    if result is None:
        raise CustomModelError(f"{who}: run() returned nothing")
    known = set(detection_ids)
    if isinstance(result, dict):
        pairs = list(result.items())
    else:
        rows = list(result)
        if rows and not isinstance(rows[0], (tuple, list)):
            if len(rows) != len(detection_ids):
                raise CustomModelError(
                    f"{who}: run() returned {len(rows)} track id(s) for "
                    f"{len(detection_ids)} detection(s)")
            pairs = list(zip(detection_ids, rows))
        else:
            try:
                pairs = [(row[0], row[1]) for row in rows]
            except IndexError as exc:
                raise CustomModelError(
                    f"{who}: run() must return {{detection_id: track_id}} "
                    "or (detection_id, track_id) pairs") from exc
    out: Dict[int, int] = {}
    for detection_id, track_id in pairs:
        if detection_id not in known:
            raise CustomModelError(
                f"{who}: run() returned detection id {detection_id!r}, "
                "which it was not given")
        if track_id is None:
            continue
        if not _finite(track_id):
            raise CustomModelError(
                f"{who}: track id {track_id!r} is not a number")
        if int(track_id) < 0:
            continue
        out[int(detection_id)] = int(track_id)
    return out


def interpolate_gaps(rows: List[dict], keys: Sequence[str]) -> List[dict]:
    """Fill the frames a track skips with linearly interpolated rows.

    *rows* are dicts with ``frame``, ``track_id`` and the numeric *keys*;
    the result holds them plus one row per missing frame between two
    consecutive members of a track, marked ``interpolated = 1`` (and
    without a ``detection_id`` - no detection was made there).
    """
    by_track: Dict[int, List[dict]] = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row)
    out: List[dict] = []
    for track_id, members in by_track.items():
        members.sort(key=lambda r: r["frame"])
        for index, row in enumerate(members):
            out.append(dict(row, interpolated=row.get("interpolated", 0)))
            if index + 1 == len(members):
                break
            following = members[index + 1]
            gap = following["frame"] - row["frame"]
            for step in range(1, gap):
                alpha = step / gap
                filled = {key: row[key] + alpha * (following[key] - row[key])
                          for key in keys}
                filled.update(frame=row["frame"] + step, track_id=track_id,
                              interpolated=1, detection_id=None)
                for key, value in row.items():
                    filled.setdefault(key, value)
                out.append(filled)
    out.sort(key=lambda r: (r["frame"], r["track_id"]))
    return out


# ---------------------------------------------------------------------------
# Samples shipped with the plugin
# ---------------------------------------------------------------------------

def samples_folder() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "custom_samples")


def sample_files() -> List[str]:
    """The sample implementations shipped with the plugin (file names)."""
    folder = samples_folder()
    if not os.path.isdir(folder):
        return []
    return sorted(name for name in os.listdir(folder)
                  if name.endswith(".py") and not name.startswith("_"))


def copy_samples(destination: str,
                 names: Optional[Sequence[str]] = None) -> List[str]:
    """Copy the sample implementations into *destination*; existing files
    of the same name are kept and reported as skipped by omission."""
    import shutil

    os.makedirs(destination, exist_ok=True)
    written = []
    for name in (names or sample_files()):
        source = os.path.join(samples_folder(), name)
        target = os.path.join(destination, name)
        if not os.path.isfile(source) or os.path.exists(target):
            continue
        shutil.copyfile(source, target)
        written.append(target)
    return written
