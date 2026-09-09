# -*- coding: utf-8 -*-
"""Promptable segmentation of the extracted frames - the model behind the
Segmentation tool.

Three backends produce the same per-frame structure, so the geo-referencing
step, the QGIS layers and the GeoJSON export never know which one ran:

* :class:`RoboflowBackend` - Roboflow's hosted SAM3 endpoint. Text prompts on
  single images only; needs an API key and a network round trip per frame.
* :class:`TransformersBackend` - Meta's ``facebook/sam3`` checkpoint through
  ``transformers``: text prompts (``Sam3Model``), point prompts
  (``Sam3TrackerModel``), and both across a frame sequence with the video
  trackers (``Sam3VideoModel`` / ``Sam3TrackerVideoModel``).
* :class:`MetaSam3Backend` - the official ``sam3`` package from
  facebookresearch, which is the only route to the **SAM 3.1** checkpoint
  (``facebook/sam3.1``, object multiplexing for many-object video tracking).
  Hugging Face hosts SAM 3.1 as a bare checkpoint without a transformers
  integration, so it cannot be reached through the backend above.

Two prompt kinds. A *text* prompt ("deer") finds every instance of a concept;
its result is one prompt entry with N predictions. A *point* prompt is one
or more positive/negative clicks defining one object; its result is one
prompt entry per object, named after the object, with the one mask that
object produced on each frame.

Two modes. *image* segments every selected frame on its own; *sequence*
hands the selected frames to the video tracker as one clip, so an object
found (or clicked) on one frame keeps its identity on the others.

Per-frame result (the pixel JSON the tool writes)::

    {"frame_idx": 12, "imagefile": "frame_000012.jpg",
     "prompts": [{"prompt": "deer", "prompt_type": "text",
                  "predictions": [{"confidence": 0.91,
                                   "polygons": [[[x, y], …], …],
                                   "object_id": 3}]},
                 {"prompt": "object 1", "prompt_type": "point", "object_id": 1,
                  "predictions": [{"confidence": 1.0, "polygons": [...]}]}]}

Everything heavy (``torch``, ``transformers``, ``sam3``, ``cv2``, ``PIL``)
is imported lazily; this module imports in the headless test stub.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import hf_access, sam3_local

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

BACKEND_ROBOFLOW = "roboflow"
BACKEND_TRANSFORMERS = "transformers"
BACKEND_META = "meta"

CAP_TEXT = "text"
CAP_POINTS = "points"
CAP_SEQUENCE = "sequence"

PROMPT_TEXT = "text"
PROMPT_POINT = "point"

MODE_IMAGE = "image"
MODE_SEQUENCE = "sequence"

#: ``(key, label, capabilities)`` in the order the tool lists them.
BACKENDS: List[Tuple[str, str, frozenset]] = [
    (BACKEND_TRANSFORMERS, "Local - transformers (facebook/sam3)",
     frozenset({CAP_TEXT, CAP_POINTS, CAP_SEQUENCE})),
    (BACKEND_META, "Local - Meta sam3 package (SAM 3 / SAM 3.1)",
     frozenset({CAP_TEXT, CAP_POINTS, CAP_SEQUENCE})),
    (BACKEND_ROBOFLOW, "Remote - Roboflow API (SAM3, text only)",
     frozenset({CAP_TEXT})),
]

#: Versions the official package can build (``build_sam3_predictor``), and
#: the checkpoint file each one reads from its Hugging Face repository.
META_VERSIONS = ("sam3.1", "sam3")
META_CHECKPOINTS = {"sam3": "sam3.pt", "sam3.1": "sam3.1_multiplex.pt"}

#: SAM3 works at 1008 px; a 4K RGB frame gains nothing above that and a clip
#: of them would not fit in memory, so sequence frames are shrunk to this
#: longer side before the clip is assembled. Polygons are scaled back.
SEQUENCE_MAX_SIDE = 1008

PIXEL_FILE = "segmentation_pixel.json"
GEOREF_FILE = "segmentation_georef.json"

ROBOFLOW_ENDPOINT = "https://serverless.roboflow.com/sam3/concept_segment"


class SegmentationError(RuntimeError):
    """The segmentation could not be run as requested."""


class CancelledError(RuntimeError):
    """The user cancelled the run."""


def backend_capabilities(key: str) -> frozenset:
    for name, _label, caps in BACKENDS:
        if name == key:
            return caps
    raise SegmentationError(f"Unknown segmentation backend {key!r}")


def backend_label(key: str) -> str:
    for name, label, _caps in BACKENDS:
        if name == key:
            return label
    return key


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class PointPrompt:
    """One click. ``positive`` clicks are on the object, negatives beside it."""
    object_id: int
    frame_idx: int
    x: float
    y: float
    positive: bool = True

    def label(self) -> int:
        return 1 if self.positive else 0


@dataclass
class SegmentationRequest:
    """Everything one run needs, independent of any widget."""
    target_folder: str
    modality: str                       # "t" | "w"
    backend: str = BACKEND_TRANSFORMERS
    mode: str = MODE_IMAGE
    frames: List[int] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    points: List[PointPrompt] = field(default_factory=list)
    object_names: Dict[int, str] = field(default_factory=dict)
    confidence: float = 0.5
    model: str = ""                     # HF repo (transformers) / version (meta)
    checkpoint_path: str = ""           # meta backend: local .pt, optional
    api_key: str = ""                   # roboflow
    hf_token: str = ""                  # gated downloads
    device: str = "auto"
    models_dir: str = ""

    def prompt_names(self) -> List[str]:
        """Every prompt entry name this run will write (for the merge)."""
        names = [t for t in self.texts if t]
        for object_id in sorted({p.object_id for p in self.points}):
            names.append(object_name(object_id, self.object_names))
        return names

    def validate(self) -> None:
        caps = backend_capabilities(self.backend)
        if self.modality not in ("t", "w"):
            raise SegmentationError(f"Unknown camera {self.modality!r}")
        if not self.frames:
            raise SegmentationError("No frames selected.")
        if not self.texts and not self.points:
            raise SegmentationError(
                "Enter at least one text prompt, or click at least one point.")
        if self.texts and self.points:
            raise SegmentationError(
                "Run text prompts and point prompts separately: a text prompt "
                "finds every instance of a concept, a point prompt defines "
                "one object, and the two are answered by different models.")
        if self.points and CAP_POINTS not in caps:
            raise SegmentationError(
                f"{backend_label(self.backend)} accepts text prompts only. "
                "Choose a local backend for point prompts.")
        if self.mode == MODE_SEQUENCE and CAP_SEQUENCE not in caps:
            raise SegmentationError(
                f"{backend_label(self.backend)} segments single images only. "
                "Choose a local backend to track across a sequence.")
        if self.mode not in (MODE_IMAGE, MODE_SEQUENCE):
            raise SegmentationError(f"Unknown mode {self.mode!r}")
        if self.backend == BACKEND_ROBOFLOW and not self.api_key:
            raise SegmentationError(
                "Roboflow API key is required for SAM3 segmentation")
        if self.points:
            outside = sorted({p.frame_idx for p in self.points}
                             - set(self.frames))
            if outside:
                raise SegmentationError(
                    f"Points were clicked on frame(s) {outside}, which are "
                    "not in the selected frame range.")


def object_name(object_id: int, names: Optional[Dict[int, str]] = None) -> str:
    """The prompt entry name of a clicked object."""
    custom = (names or {}).get(object_id, "")
    return custom.strip() if custom and custom.strip() else f"object {object_id}"


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def pose_images(target_folder: str, modality: str) -> List[dict]:
    """``images`` of ``poses_{modality}.json`` - the frames and their files."""
    path = os.path.join(target_folder, f"poses_{modality}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"poses_{modality}.json not found - run frame extraction first")
    with open(path, "r", encoding="utf-8") as handle:
        poses = json.load(handle)
    images = poses.get("images", []) or []
    if not images:
        raise SegmentationError(f"No frames found in poses_{modality}.json")
    return images


def frame_path(target_folder: str, modality: str, image: dict) -> str:
    return os.path.join(target_folder, f"frames_{modality}",
                        image.get("imagefile", "") or "")


def select_frames(total: int, all_frames: bool = True, start: int = 0,
                  end: int = 999999, step: int = 1) -> List[int]:
    """Frame indices in range, then every *step*-th - the pipeline's rule."""
    if step < 1:
        raise SegmentationError("The frame step must be >= 1")
    if total <= 0:
        return []
    if all_frames:
        indices = list(range(total))
    else:
        start = max(0, int(start))
        end = min(int(end), total - 1)
        indices = list(range(start, end + 1))
    return indices[::step] if step > 1 else indices


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def scale_polygons(polygons, factor: float):
    """Polygons scaled by *factor* (``1.0`` returns them untouched)."""
    if factor == 1.0:
        return polygons
    return [[[float(x) * factor, float(y) * factor] for x, y in ring]
            for ring in polygons]


def text_entry(prompt: str, predictions: List[dict]) -> dict:
    return {"prompt": prompt, "prompt_type": PROMPT_TEXT,
            "predictions": predictions}


def point_entry(object_id: int, names: Optional[Dict[int, str]],
                predictions: List[dict]) -> dict:
    return {"prompt": object_name(object_id, names),
            "prompt_type": PROMPT_POINT, "object_id": int(object_id),
            "predictions": predictions}


def frame_result(frame_idx: int, imagefile: str, prompts: List[dict]) -> dict:
    return {"frame_idx": int(frame_idx), "imagefile": imagefile,
            "prompts": prompts}


def prediction_from_mask(mask, confidence: float, scale: float = 1.0,
                         object_id: Optional[int] = None,
                         box=None) -> Optional[dict]:
    """A prediction dict from one binary mask, or ``None`` when it is empty."""
    polygons = sam3_local.mask_to_polygons(mask)
    if not polygons:
        return None
    prediction = {"confidence": float(confidence),
                  "polygons": scale_polygons(polygons, scale)}
    if object_id is not None:
        prediction["object_id"] = int(object_id)
    if box is not None:
        prediction["box"] = [float(v) * scale for v in list(box)[:4]]
    return prediction


def count_predictions(results: Iterable[dict]) -> int:
    return sum(len(p.get("predictions", []))
               for r in results for p in r.get("prompts", []))


class SequencePace:
    """Progress and pace of a sequence run, for the bar and the log.

    A clip goes through three stretches the user cannot see into: reading
    the frames, handing them to the tracker (which resizes and normalises
    every one), and propagating. The bar used to sit at the start of that
    until the first tracked frame came back - minutes on a CPU - so each
    stretch now moves it: loading fills the first *load_share*, preparation
    a sliver after it, propagation the rest. Every *every* frames a log line
    states the rate and the time left, because "31 of 300, 4.2 s/frame,
    ~19 min left" is what makes a slow run tolerable.

    *progress_fn* receives a fraction 0..1 of the whole run.
    """

    def __init__(self, total: int, progress_fn=None, log_fn=None,
                 load_share: float = 0.15, prepare_share: float = 0.05,
                 every: int = 10, clock=None):
        import time
        self.total = max(1, int(total))
        self._progress = progress_fn
        self._log = log_fn
        self._load_share = load_share
        self._prepare_share = prepare_share
        self._every = max(1, every)
        self._clock = clock or time.monotonic
        self._started = None
        self._done = 0

    def _emit(self, fraction: float) -> None:
        if self._progress:
            self._progress(max(0.0, min(1.0, fraction)))

    def loading(self, done: int, total: Optional[int] = None) -> None:
        """*done* of *total* frames read from disk."""
        total = total or self.total
        self._emit(self._load_share * done / max(1, total))

    def preparing(self, message: str = "") -> None:
        """The tracker is ingesting the clip; nothing to count yet."""
        if message and self._log:
            self._log(message)
        self._emit(self._load_share)

    def propagating(self, message: str = "") -> None:
        """Propagation starts now; the frame clock starts with it."""
        if message and self._log:
            self._log(message)
        self._started = self._clock()
        self._done = 0
        self._emit(self._load_share + self._prepare_share)

    def frame(self) -> None:
        """One more frame tracked."""
        if self._started is None:
            self.propagating()
        self._done += 1
        rest = 1.0 - self._load_share - self._prepare_share
        self._emit(self._load_share + self._prepare_share
                   + rest * self._done / self.total)
        if self._log and (self._done % self._every == 0
                          or self._done == self.total):
            elapsed = max(1e-6, self._clock() - self._started)
            rate = elapsed / self._done
            left = rate * max(0, self.total - self._done)
            self._log(f"Tracked {self._done}/{self.total} frames, "
                      f"{rate:.1f} s/frame, ~{_minutes(left)} left")


def _minutes(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class SegmentationBackend:
    """What the run loop needs from a model. Subclasses fill in the two calls."""

    key = ""
    capabilities: frozenset = frozenset()

    def __init__(self, log_fn=None):
        self._log = log_fn

    def log(self, message: str) -> None:
        if self._log:
            self._log(message)

    def load(self) -> None:
        """Load the model(s) once, before the frames. Idempotent."""

    def prepare(self, texts: Sequence[str], points: Sequence[PointPrompt],
                mode: str) -> None:
        """Load exactly the heads this run needs, before any frame is read.

        A missing token, an old transformers or a gated repository fails
        here - in one message - rather than once per frame in the warnings.
        """

    def close(self) -> None:
        """Release what :meth:`load` took."""

    def segment_image(self, image_path: str, texts: Sequence[str],
                      points: Sequence[PointPrompt],
                      names: Dict[int, str], confidence: float) -> List[dict]:
        """Prompt entries for one frame on its own."""
        raise NotImplementedError

    def segment_sequence(self, image_paths: Sequence[str],
                         texts: Sequence[str], points: Sequence[PointPrompt],
                         names: Dict[int, str], confidence: float,
                         progress_fn=None, cancel_check=None
                         ) -> Dict[int, List[dict]]:
        """Prompt entries per *local* frame index (position in *image_paths*).

        Point prompts arrive with ``frame_idx`` already translated to that
        local index. *progress_fn* takes the fraction 0..1 of the run done -
        loading and preparing the clip included, not only tracked frames -
        because the bar must move while the tracker ingests the clip.
        """
        raise NotImplementedError


# -- Roboflow ------------------------------------------------------------------

class RoboflowBackend(SegmentationBackend):
    """Roboflow's hosted SAM3 concept segmentation. Text on single images."""

    key = BACKEND_ROBOFLOW
    capabilities = frozenset({CAP_TEXT})

    def __init__(self, api_key: str, output_format: str = "polygon",
                 endpoint: str = ROBOFLOW_ENDPOINT, log_fn=None):
        super().__init__(log_fn)
        self.api_key = api_key
        self.output_format = output_format
        self.endpoint = endpoint
        self._session = None

    def load(self) -> None:
        if self._session is None:
            import requests
            self._session = requests.Session()

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # nosec B110
                pass
            self._session = None

    def segment_image(self, image_path, texts, points, names, confidence):
        import base64

        if points:
            raise SegmentationError(
                "The Roboflow backend accepts text prompts only.")
        self.load()
        with open(image_path, "rb") as handle:
            image_b64 = base64.b64encode(handle.read()).decode("utf-8")
        payload = {
            "format": self.output_format,
            "output_prob_thresh": confidence,
            "image": {"type": "base64", "value": image_b64},
            "prompts": [{"type": "text", "text": p} for p in texts],
        }
        response = self._session.post(
            self.endpoint, params={"api_key": self.api_key},
            headers={"Content-Type": "application/json"}, json=payload,
            timeout=(10, 120))
        response.raise_for_status()
        return self.normalise(response.json(), self.output_format)

    @staticmethod
    def normalise(response: dict, output_format: str = "polygon") -> List[dict]:
        """The shared structure from one ``prompt_results`` response."""
        entries = []
        for prompt_result in response.get("prompt_results", []) or []:
            echo = prompt_result.get("echo", {}) or {}
            prompt_text = (echo.get("text") or prompt_result.get("prompt")
                           or prompt_result.get("text") or "")
            predictions = []
            for prediction in prompt_result.get("predictions", []) or []:
                data = {"confidence": float(prediction.get("confidence", 0.0))}
                masks = prediction.get("masks")
                if output_format == "polygon" and masks is not None:
                    data["polygons"] = masks
                elif masks is not None:
                    data["masks"] = masks
                predictions.append(data)
            entries.append(text_entry(prompt_text, predictions))
        return entries


# -- transformers -----------------------------------------------------------

def _group_points(points: Sequence[PointPrompt]) -> Dict[int, List[PointPrompt]]:
    grouped: Dict[int, List[PointPrompt]] = {}
    for point in points:
        grouped.setdefault(int(point.object_id), []).append(point)
    return dict(sorted(grouped.items()))


def _load_image(path: str, max_side: Optional[int] = None):
    """A PIL RGB image and the factor original/loaded for the coordinates."""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    scale = 1.0
    if max_side and max(image.size) > max_side:
        factor = max_side / float(max(image.size))
        new_size = (max(1, round(image.size[0] * factor)),
                    max(1, round(image.size[1] * factor)))
        image = image.resize(new_size, Image.BILINEAR)
        scale = 1.0 / factor
    return image, scale


def _sigmoid(value: float) -> float:
    import math
    try:
        return 1.0 / (1.0 + math.exp(-float(value)))
    except OverflowError:
        return 0.0


class TransformersBackend(SegmentationBackend):
    """``facebook/sam3`` through transformers: four heads, loaded on demand.

    Text on an image is answered by :class:`sam3_local.LocalSam3` exactly as
    the processing step did; the tracker heads share its checkpoint cache.
    """

    key = BACKEND_TRANSFORMERS
    capabilities = frozenset({CAP_TEXT, CAP_POINTS, CAP_SEQUENCE})

    def __init__(self, model_id: str = "", models_dir: str = "",
                 # "no token supplied" is a state, not a credential.
                 token: str = "",  # nosec B107
                 device: str = "auto", log_fn=None):
        super().__init__(log_fn)
        from .classification import split_revision

        self.model_id, self.revision = split_revision(
            model_id or sam3_local.DEFAULT_SAM3_REPO)
        self.models_dir = models_dir
        self.token = token
        self.device_preference = device
        self.device: Optional[str] = None
        self._text_image = None
        self._heads: Dict[str, tuple] = {}

    # -- loading ------------------------------------------------------------

    def load(self) -> None:
        """Resolve the device; the heads load in :meth:`prepare`."""
        if self.device is None:
            from .classification import resolve_device
            self.device = resolve_device(self.device_preference)

    def prepare(self, texts, points, mode) -> None:
        sequence = mode == MODE_SEQUENCE
        if texts:
            if sequence:
                self._head("Sam3Video")
            else:
                self._text_image_model()
        if points:
            self._head("Sam3TrackerVideo" if sequence else "Sam3Tracker")

    def close(self) -> None:
        self._text_image = None
        self._heads = {}
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # nosec B110
            pass

    def _text_image_model(self):
        if self._text_image is None:
            model = sam3_local.LocalSam3(
                model_id=self.model_id + (f"@{self.revision}" if self.revision else ""),
                models_dir=self.models_dir, token=self.token,
                device=self.device_preference, log_fn=self._log)
            model.load()
            self._text_image = model
        return self._text_image

    def _head(self, name: str):
        """``(model, processor)`` for ``Sam3Tracker``, ``Sam3Video``, …"""
        if name in self._heads:
            return self._heads[name]
        import time
        from .classification import (
            DownloadReporter, _load_failure_message, describe_cache,
            describe_device)

        self.load()
        model_cls, processor_cls = sam3_local.transformers_classes(
            f"{name}Model", f"{name}Processor")
        kwargs = sam3_local.pretrained_kwargs(
            self.models_dir, self.token, self.revision)
        cache_dir = kwargs.get("cache_dir")
        self.log(f"Loading {name} from {self.model_id} on {self.device}…")
        # All heads read one repository, so after the first head the cache
        # is complete and this only says so; on a first run it reports the
        # download as the cache folder grows.
        with hf_access.token_environment(self.token):
            cached = describe_cache(self.model_id, cache_dir, self.revision,
                                    self.token or None, log_fn=self._log)
        started = time.monotonic()
        try:
            # nosec B615 - the revision is user-configurable ('repo@commit').
            with hf_access.token_environment(self.token), DownloadReporter(
                    cache_dir, self.model_id, cached.expected_bytes,
                    log_fn=self._log, active=not cached.complete):
                processor = processor_cls.from_pretrained(
                    self.model_id, **kwargs)  # nosec B615
                model = model_cls.from_pretrained(
                    self.model_id, **kwargs)  # nosec B615
        except Exception as exc:
            raise sam3_local.Sam3LocalError(
                _load_failure_message(self.model_id, exc)) from exc
        model.to(self.device)
        model.eval()
        self.log(f"{name} ready on {describe_device(self.device)} in "
                 f"{time.monotonic() - started:.0f} s")
        self._heads[name] = (model, processor)
        return self._heads[name]

    # -- image ----------------------------------------------------------------

    def segment_image(self, image_path, texts, points, names, confidence):
        entries: List[dict] = []
        if texts:
            entries.extend(self._text_image_model().segment_path(
                image_path, list(texts), confidence))
        if points:
            entries.extend(self._points_image(image_path, points, names))
        return entries

    def _points_image(self, image_path, points, names) -> List[dict]:
        import torch

        model, processor = self._head("Sam3Tracker")
        image, _scale = _load_image(image_path)
        grouped = _group_points(points)
        input_points = [[[[p.x, p.y] for p in pts] for pts in grouped.values()]]
        input_labels = [[[p.label() for p in pts] for pts in grouped.values()]]
        inputs = processor(images=image, input_points=input_points,
                           input_labels=input_labels,
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = sam3_local._to_numpy(masks)
        scores = sam3_local._to_numpy(outputs.iou_scores)

        entries = []
        for index, object_id in enumerate(grouped):
            mask = masks[index]
            if getattr(mask, "ndim", 2) == 3:
                mask = mask[0]
            try:
                confidence = float(scores[0][index][0])
            except (IndexError, TypeError):
                confidence = 1.0
            prediction = prediction_from_mask(mask, confidence)
            entries.append(point_entry(
                object_id, names, [prediction] if prediction else []))
        return entries

    # -- sequence -----------------------------------------------------------

    def segment_sequence(self, image_paths, texts, points, names, confidence,
                         progress_fn=None, cancel_check=None):
        pace = SequencePace(len(image_paths), progress_fn, self._log)
        frames, scale = self._load_clip(image_paths, pace, cancel_check)
        if texts:
            return self._text_sequence(frames, scale, list(texts), confidence,
                                       pace, cancel_check)
        return self._points_sequence(frames, scale, points, names,
                                     pace, cancel_check)

    def _load_clip(self, image_paths, pace: SequencePace, cancel_check=None):
        frames = []
        scale = 1.0
        for index, path in enumerate(image_paths):
            if cancel_check and cancel_check():
                raise CancelledError("Segmentation cancelled")
            image, scale = _load_image(path, SEQUENCE_MAX_SIDE)
            frames.append(image)
            pace.loading(index + 1, len(image_paths))
        self.log(f"Loaded {len(frames)} frames"
                 + (f", shrunk to {SEQUENCE_MAX_SIDE} px (x{scale:.2f} back "
                    "on export) - SAM3 works at that size" if scale != 1.0
                    else ""))
        return frames, scale

    def _init_clip_session(self, processor, frames):
        kwargs = {"video": frames, "inference_device": self.device}
        try:
            return processor.init_video_session(
                processing_device="cpu", video_storage_device="cpu",
                **kwargs)
        except TypeError:
            return processor.init_video_session(**kwargs)

    def _text_sequence(self, frames, scale, texts, confidence,
                       pace: SequencePace, cancel_check) -> Dict[int, List[dict]]:
        import torch

        model, processor = self._head("Sam3Video")
        pace.preparing(f"Handing {len(frames)} frames to the video tracker "
                       "(each is resized and normalised first)…")
        session = self._init_clip_session(processor, frames)
        processor.add_text_prompt(inference_session=session, text=texts)
        pace.propagating(
            f"Tracking {texts} across {len(frames)} frames on {self.device}. "
            "The tracker holds its first results back for a few frames to "
            "weed out duplicates, so the first update takes a moment.")

        results: Dict[int, List[dict]] = {}
        with torch.no_grad():
            for output in model.propagate_in_video_iterator(
                    inference_session=session):
                if cancel_check and cancel_check():
                    raise CancelledError("Segmentation cancelled")
                processed = processor.postprocess_outputs(session, output)
                local_idx = int(output.frame_idx)
                results[local_idx] = self._video_entries(
                    processed, texts, scale, confidence)
                pace.frame()
        return results

    @staticmethod
    def _video_entries(processed: dict, texts: Sequence[str], scale: float,
                       confidence: float) -> List[dict]:
        object_ids = [int(v) for v in sam3_local._to_numpy(
            processed.get("object_ids", [])).reshape(-1).tolist()]
        scores = sam3_local._to_numpy(processed.get("scores", [])).reshape(-1)
        masks = processed.get("masks")
        masks = sam3_local._to_numpy(masks) if masks is not None else None
        boxes = processed.get("boxes")
        boxes = sam3_local._to_numpy(boxes) if boxes is not None else None
        prompt_of: Dict[int, str] = {}
        for prompt, ids in (processed.get("prompt_to_obj_ids") or {}).items():
            for obj in ids:
                prompt_of[int(obj)] = prompt
        by_prompt: Dict[str, List[dict]] = {t: [] for t in texts}
        for index, object_id in enumerate(object_ids):
            if masks is None or index >= len(masks):
                continue
            score = float(scores[index]) if index < len(scores) else 1.0
            if score < confidence:
                continue
            prediction = prediction_from_mask(
                masks[index], score, scale, object_id=object_id,
                box=boxes[index] if boxes is not None and index < len(boxes)
                else None)
            if prediction is None:
                continue
            prompt = prompt_of.get(object_id, texts[0] if texts else "")
            by_prompt.setdefault(prompt, []).append(prediction)
        return [text_entry(prompt, preds) for prompt, preds in by_prompt.items()]

    def _points_sequence(self, frames, scale, points, names,
                         pace: SequencePace, cancel_check) -> Dict[int, List[dict]]:
        import torch

        model, processor = self._head("Sam3TrackerVideo")
        pace.preparing(f"Handing {len(frames)} frames to the video tracker "
                       "(each is resized and normalised first)…")
        session = self._init_clip_session(processor, frames)
        height, width = frames[0].size[1], frames[0].size[0]

        # One call per annotated frame, all its objects at once.
        by_frame: Dict[int, Dict[int, List[PointPrompt]]] = {}
        for point in points:
            by_frame.setdefault(int(point.frame_idx), {}).setdefault(
                int(point.object_id), []).append(point)
        for local_idx, grouped in sorted(by_frame.items()):
            grouped = dict(sorted(grouped.items()))
            processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=local_idx,
                obj_ids=list(grouped.keys()),
                input_points=[[[[p.x / scale, p.y / scale] for p in pts]
                               for pts in grouped.values()]],
                input_labels=[[[p.label() for p in pts]
                               for pts in grouped.values()]])

        first = min(by_frame)
        results: Dict[int, List[dict]] = {}
        pace.propagating(
            f"Tracking {len(session.obj_ids)} clicked object(s) across "
            f"{len(frames)} frames on {self.device}…")

        def consume(output):
            if cancel_check and cancel_check():
                raise CancelledError("Segmentation cancelled")
            masks = processor.post_process_masks(
                [output.pred_masks], original_sizes=[[height, width]],
                binarize=True)[0]
            masks = sam3_local._to_numpy(masks)
            logits = getattr(output, "object_score_logits", None)
            logits = sam3_local._to_numpy(logits) if logits is not None else None
            entries = []
            for index, object_id in enumerate(session.obj_ids):
                if index >= len(masks):
                    break
                mask = masks[index]
                if getattr(mask, "ndim", 2) == 3:
                    mask = mask[0]
                score = 1.0
                if logits is not None:
                    try:
                        score = _sigmoid(float(logits.reshape(-1)[index]))
                    except (IndexError, ValueError):
                        score = 1.0
                prediction = prediction_from_mask(mask, score, scale)
                entries.append(point_entry(
                    int(object_id), names, [prediction] if prediction else []))
            results[int(output.frame_idx)] = entries
            pace.frame()

        with torch.no_grad():
            for output in model.propagate_in_video_iterator(
                    inference_session=session, start_frame_idx=first):
                consume(output)
            if first > 0:
                for output in model.propagate_in_video_iterator(
                        inference_session=session, start_frame_idx=first,
                        reverse=True):
                    consume(output)
        return results


# -- Meta's sam3 package ------------------------------------------------------

class MetaSam3Backend(SegmentationBackend):
    """The official ``sam3`` package: SAM 3 or SAM 3.1, image and video.

    Everything goes through one predictor session over a clip of PIL frames;
    a single image is a clip of one. The predictor needs a CUDA GPU (the
    package autocasts to bfloat16 on ``cuda``) and Python >= 3.12 - both are
    reported before any frame is read.
    """

    key = BACKEND_META
    capabilities = frozenset({CAP_TEXT, CAP_POINTS, CAP_SEQUENCE})

    def __init__(self, version: str = "sam3.1", checkpoint_path: str = "",
                 # "no token supplied" is a state, not a credential.
                 token: str = "",  # nosec B107
                 models_dir: str = "", log_fn=None):
        super().__init__(log_fn)
        self.version = version if version in META_VERSIONS else "sam3.1"
        self.checkpoint_path = checkpoint_path or ""
        self.token = token
        self.models_dir = models_dir
        self._predictor = None

    @property
    def repo_id(self) -> str:
        return f"facebook/{self.version}"

    def resolve_checkpoint(self) -> str:
        """The checkpoint file to load - downloaded on first use.

        A user-chosen file wins. Otherwise the checkpoint is fetched from
        the gated repository into the plugin's shared model cache - the
        same ``hf_cache`` the DINOv3 backbone and the transformers SAM3 use
        - rather than the package's default of ``~/.cache/huggingface``, so
        every model the plugin downloads lives in one place, and one
        download serves every project.
        """
        if self.checkpoint_path:
            if not os.path.isfile(self.checkpoint_path):
                raise SegmentationError(
                    f"Checkpoint not found: {self.checkpoint_path}")
            return self.checkpoint_path
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SegmentationError(
                "huggingface_hub is not installed. Install the Classification "
                "dependencies from the Dependency Manager.") from exc
        from .classification import DownloadReporter, _mb

        filename = META_CHECKPOINTS[self.version]
        kwargs = sam3_local.pretrained_kwargs(self.models_dir, self.token)
        cache_dir = kwargs.get("cache_dir")
        with hf_access.token_environment(self.token):
            cached_path, expected = self._checkpoint_state(filename, cache_dir)
        if cached_path:
            self.log(f"{self.repo_id}/{filename} is in the local cache "
                     f"({_mb(os.path.getsize(cached_path))} MB); no download needed")
            return cached_path
        self.log(f"Fetching {self.repo_id}/{filename} into the shared model "
                 f"cache ({'about ' + _mb(expected) + ' MB' if expected else 'several GB'}, "
                 "once; later runs load from disk)…")
        try:
            # nosec B615 - unpinned like the backbone: the repository moves
            # only when Meta publishes a new checkpoint, and that is wanted.
            with hf_access.token_environment(self.token), DownloadReporter(
                    cache_dir, self.repo_id, expected, log_fn=self._log):
                return hf_hub_download(  # nosec B615
                    repo_id=self.repo_id, filename=filename, **kwargs)
        except Exception as exc:
            from .classification import _load_failure_message
            raise SegmentationError(
                _load_failure_message(self.repo_id, exc)) from exc

    def load(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise SegmentationError(
                "PyTorch is not installed. Install the Classification "
                "dependencies from the Dependency Manager.") from exc
        if not torch.cuda.is_available():
            raise SegmentationError(
                "Meta's sam3 package runs on a CUDA GPU only; none is "
                "available to this QGIS. Use the transformers backend for "
                "CPU inference.")
        try:
            from sam3.model_builder import build_sam3_predictor
        except ImportError as exc:
            raise SegmentationError(
                "The 'sam3' package is not installed. It is Meta's official "
                "SAM 3 / SAM 3.1 code (Python >= 3.12, PyTorch >= 2.7, CUDA "
                ">= 12.6): pip install "
                "git+https://github.com/facebookresearch/sam3.git\n\n"
                f"Original error: {exc}") from exc

        checkpoint = self.resolve_checkpoint()
        self.log(f"Building the {self.version} predictor from {checkpoint}…")
        try:
            self._predictor = build_sam3_predictor(
                checkpoint_path=checkpoint, version=self.version)
        except Exception as exc:
            from .classification import _load_failure_message
            raise SegmentationError(
                _load_failure_message(self.repo_id, exc)) from exc

    def _checkpoint_state(self, filename: str, cache_dir: Optional[str]):
        """``(cached_path_or_"", expected_bytes_or_None)`` - both best effort.

        The cache holds a whole snapshot per repository, and ``config.json``
        being present says nothing about a 3 GB checkpoint beside it, so the
        checkpoint file itself is looked up; its size comes from one
        metadata call and is only for the progress percentage.
        """
        cached = ""
        expected = None
        try:
            from huggingface_hub import try_to_load_from_cache
            found = try_to_load_from_cache(
                self.repo_id, filename, cache_dir=cache_dir or None)
            if isinstance(found, str) and os.path.isfile(found):
                cached = found
        except Exception:  # nosec B110 - offline or old hub: download anyway
            cached = ""
        if not cached:
            try:
                from huggingface_hub import HfApi
                info = HfApi().model_info(self.repo_id, token=self.token or None,
                                          files_metadata=True)
                for sibling in info.siblings or []:
                    if getattr(sibling, "rfilename", "") == filename:
                        expected = getattr(sibling, "size", None)
            except Exception:  # nosec B110 - size is a nicety
                expected = None
        return cached, expected

    def close(self) -> None:
        predictor = self._predictor
        self._predictor = None
        if predictor is not None:
            shutdown = getattr(predictor, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:  # nosec B110
                    pass

    # -- both calls are one clip ------------------------------------------

    def segment_image(self, image_path, texts, points, names, confidence):
        local_points = [PointPrompt(p.object_id, 0, p.x, p.y, p.positive)
                        for p in points]
        return self.segment_sequence(
            [image_path], texts, local_points, names, confidence).get(0, [])

    def segment_sequence(self, image_paths, texts, points, names, confidence,
                         progress_fn=None, cancel_check=None):
        self.load()
        pace = SequencePace(len(image_paths), progress_fn, self._log)
        frames = []
        scale = 1.0
        max_side = SEQUENCE_MAX_SIDE if len(image_paths) > 1 else None
        for index, path in enumerate(image_paths):
            if cancel_check and cancel_check():
                raise CancelledError("Segmentation cancelled")
            image, scale = _load_image(path, max_side)
            frames.append(image)
            pace.loading(index + 1)
        width, height = frames[0].size

        if len(frames) > 1:
            pace.preparing(f"Handing {len(frames)} frames to the "
                           f"{self.version} predictor…")
        session_id = self._predictor.handle_request(
            request={"type": "start_session", "resource_path": frames}
        )["session_id"]
        results: Dict[int, List[dict]] = {}
        try:
            if texts:
                for text in texts:
                    self._predictor.handle_request(request={
                        "type": "add_prompt", "session_id": session_id,
                        "frame_index": 0, "text": text,
                        "output_prob_thresh": confidence})
                    per_frame = self._propagate(
                        session_id, len(frames), confidence, pace,
                        cancel_check, what=repr(text))
                    for local_idx, outputs in per_frame.items():
                        results.setdefault(local_idx, []).append(text_entry(
                            text, self._predictions(outputs, scale,
                                                    with_ids=True)))
            else:
                grouped_frames: Dict[int, Dict[int, List[PointPrompt]]] = {}
                for point in points:
                    grouped_frames.setdefault(int(point.frame_idx), {}) \
                        .setdefault(int(point.object_id), []).append(point)
                for local_idx, grouped in sorted(grouped_frames.items()):
                    for object_id, pts in sorted(grouped.items()):
                        self._predictor.handle_request(request={
                            "type": "add_prompt", "session_id": session_id,
                            "frame_index": local_idx, "obj_id": int(object_id),
                            "points": [[p.x / scale / width,
                                        p.y / scale / height] for p in pts],
                            "point_labels": [p.label() for p in pts],
                            "rel_coordinates": True})
                per_frame = self._propagate(
                    session_id, len(frames), confidence, pace,
                    cancel_check, what="the clicked objects")
                for local_idx, outputs in per_frame.items():
                    results[local_idx] = self._point_entries(
                        outputs, scale, names)
        finally:
            try:
                self._predictor.handle_request(request={
                    "type": "close_session", "session_id": session_id})
            except Exception:  # nosec B110
                pass
        return results

    def _propagate(self, session_id, total, confidence, pace: SequencePace,
                   cancel_check, what: str = "") -> Dict[int, dict]:
        per_frame: Dict[int, dict] = {}
        request = {"type": "propagate_in_video", "session_id": session_id,
                   "output_prob_thresh": confidence}
        if total == 1:
            request["propagation_direction"] = "forward"
        if total > 1:
            pace.propagating(f"Tracking {what} across {total} frames…")
        for response in self._predictor.handle_stream_request(request=request):
            if cancel_check and cancel_check():
                raise CancelledError("Segmentation cancelled")
            per_frame[int(response["frame_index"])] = response["outputs"]
            if total > 1:
                pace.frame()
        return per_frame

    @staticmethod
    def _predictions(outputs: dict, scale: float, with_ids: bool) -> List[dict]:
        ids = [int(v) for v in list(outputs.get("out_obj_ids", []))]
        masks = outputs.get("out_binary_masks")
        probs = outputs.get("out_probs")
        probs = list(probs) if probs is not None else []
        predictions = []
        for index, object_id in enumerate(ids):
            if masks is None or index >= len(masks):
                continue
            score = float(probs[index]) if index < len(probs) else 1.0
            prediction = prediction_from_mask(
                masks[index], score, scale,
                object_id=object_id if with_ids else None)
            if prediction is not None:
                predictions.append(prediction)
        return predictions

    @staticmethod
    def _point_entries(outputs: dict, scale: float,
                       names: Dict[int, str]) -> List[dict]:
        ids = [int(v) for v in list(outputs.get("out_obj_ids", []))]
        masks = outputs.get("out_binary_masks")
        probs = outputs.get("out_probs")
        probs = list(probs) if probs is not None else []
        entries = []
        for index, object_id in enumerate(ids):
            if masks is None or index >= len(masks):
                continue
            score = float(probs[index]) if index < len(probs) else 1.0
            prediction = prediction_from_mask(masks[index], score, scale)
            entries.append(point_entry(
                object_id, names, [prediction] if prediction else []))
        return entries


def make_backend(request: SegmentationRequest, log_fn=None) -> SegmentationBackend:
    """The backend the request names, configured from it."""
    if request.backend == BACKEND_ROBOFLOW:
        return RoboflowBackend(request.api_key, log_fn=log_fn)
    if request.backend == BACKEND_TRANSFORMERS:
        return TransformersBackend(
            model_id=request.model, models_dir=request.models_dir,
            token=request.hf_token, device=request.device, log_fn=log_fn)
    if request.backend == BACKEND_META:
        return MetaSam3Backend(
            version=request.model or "sam3.1",
            checkpoint_path=request.checkpoint_path, token=request.hf_token,
            models_dir=request.models_dir, log_fn=log_fn)
    raise SegmentationError(f"Unknown segmentation backend {request.backend!r}")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_segmentation(request: SegmentationRequest, progress_fn=None,
                     log_fn=None, cancel_check=None,
                     backend: Optional[SegmentationBackend] = None
                     ) -> List[dict]:
    """Segment the requested frames; per-frame results in frame order.

    *progress_fn* receives 0-100. *backend* is injectable for tests; by
    default :func:`make_backend` builds it and it is closed at the end.
    """
    request.validate()
    images = pose_images(request.target_folder, request.modality)
    camera_name = "Thermal" if request.modality == "t" else "RGB"

    frames = []
    for frame_idx in request.frames:
        if frame_idx < 0 or frame_idx >= len(images):
            continue
        image = images[frame_idx]
        path = frame_path(request.target_folder, request.modality, image)
        if not image.get("imagefile") or not os.path.isfile(path):
            if log_fn:
                log_fn(f"Warning: Frame not found: {path}")
            continue
        frames.append((frame_idx, image.get("imagefile", ""), path))
    if not frames:
        raise SegmentationError(
            f"None of the selected {camera_name} frames exist on disk - "
            f"extract the {camera_name} frames first.")

    owns_backend = backend is None
    if backend is None:
        backend = make_backend(request, log_fn)

    if log_fn:
        what = (f"{len(request.texts)} text prompt(s): {request.texts}"
                if request.texts else
                f"{len(request.points)} point(s) on "
                f"{len({p.object_id for p in request.points})} object(s)")
        how = ("local transformers model"
               if request.backend == BACKEND_TRANSFORMERS else
               backend_label(request.backend))
        log_fn(f"Starting SAM3 segmentation on {len(frames)} {camera_name} "
               f"frame(s) ({how}), {request.mode} mode, {what}")
        log_fn(f"Confidence threshold: {request.confidence}")
    if progress_fn:
        progress_fn(5)

    results: List[dict] = []
    try:
        backend.load()
        backend.prepare(request.texts, request.points,
                        request.mode if len(frames) > 1 else MODE_IMAGE)
        if progress_fn:
            progress_fn(10)
        if request.mode == MODE_SEQUENCE and len(frames) > 1:
            results = _run_sequence(request, backend, frames, progress_fn,
                                    log_fn, cancel_check)
        else:
            results = _run_images(request, backend, frames, progress_fn,
                                  log_fn, cancel_check)
    finally:
        if owns_backend:
            backend.close()

    if log_fn:
        log_fn(f"SAM3 segmentation complete: {len(results)} frames, "
               f"{count_predictions(results)} predictions")
    if progress_fn:
        progress_fn(100)
    return results


def _run_images(request, backend, frames, progress_fn, log_fn, cancel_check):
    results = []
    by_frame: Dict[int, List[PointPrompt]] = {}
    for point in request.points:
        by_frame.setdefault(int(point.frame_idx), []).append(point)
    if request.points:
        # Point prompts describe one frame each; only those frames run.
        frames = [f for f in frames if f[0] in by_frame]
    for index, (frame_idx, imagefile, path) in enumerate(frames):
        if cancel_check and cancel_check():
            if log_fn:
                log_fn("SAM3 segmentation cancelled by user")
            raise CancelledError("SAM3 segmentation cancelled")
        try:
            entries = backend.segment_image(
                path, request.texts, by_frame.get(frame_idx, []),
                request.object_names, request.confidence)
            results.append(frame_result(frame_idx, imagefile, entries))
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad frame is logged
            if log_fn:
                log_fn(f"Warning: Failed to process frame {frame_idx}: {exc}")
            continue
        if progress_fn:
            progress_fn(min(95, 10 + int((index + 1) / max(1, len(frames)) * 85)))
        if log_fn and (index + 1) % 10 == 0:
            log_fn(f"Processed {index + 1}/{len(frames)} frames")
    return results


def _run_sequence(request, backend, frames, progress_fn, log_fn, cancel_check):
    local_of = {frame_idx: i for i, (frame_idx, _f, _p) in enumerate(frames)}
    local_points = [PointPrompt(p.object_id, local_of[p.frame_idx], p.x, p.y,
                                p.positive)
                    for p in request.points if p.frame_idx in local_of]
    if request.points and not local_points:
        raise SegmentationError(
            "None of the clicked frames are among the frames on disk.")

    def _progress(fraction):
        """Backends report 0..1 of their run; the bar shows 10..95."""
        if progress_fn:
            progress_fn(min(95, 10 + int(max(0.0, fraction) * 85)))

    if log_fn:
        log_fn(f"Tracking across {len(frames)} frames as one sequence…")
    per_local = backend.segment_sequence(
        [path for _i, _f, path in frames], request.texts, local_points,
        request.object_names, request.confidence, progress_fn=_progress,
        cancel_check=cancel_check)
    results = []
    for local_idx, (frame_idx, imagefile, _path) in enumerate(frames):
        results.append(frame_result(frame_idx, imagefile,
                                    per_local.get(local_idx, [])))
    return results


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class SegmentationStore:
    """``segmentation_{modality}/`` - the pixel results and their geo twin."""

    def __init__(self, target_folder: str, modality: str):
        self.target_folder = target_folder
        self.modality = modality
        self.folder = os.path.join(target_folder, f"segmentation_{modality}")
        self.pixel_path = os.path.join(self.folder, PIXEL_FILE)
        self.georef_path = os.path.join(self.folder, GEOREF_FILE)

    def load_pixel(self) -> List[dict]:
        return self._read(self.pixel_path)

    def load_georef(self) -> List[dict]:
        return self._read(self.georef_path)

    def save_pixel(self, results: List[dict]) -> str:
        os.makedirs(self.folder, exist_ok=True)
        with open(self.pixel_path, "w", encoding="utf-8") as handle:
            json.dump(sorted(results, key=lambda r: r.get("frame_idx", 0)),
                      handle, indent=2)
        return self.pixel_path

    def has_pixel(self) -> bool:
        return os.path.isfile(self.pixel_path)

    def has_georef(self) -> bool:
        return os.path.isfile(self.georef_path)

    def georef_is_stale(self) -> bool:
        """The geo file predates the pixel file, so it no longer matches."""
        if not self.has_pixel() or not self.has_georef():
            return False
        try:
            return (os.path.getmtime(self.georef_path)
                    < os.path.getmtime(self.pixel_path))
        except OSError:
            return False

    @staticmethod
    def _read(path: str) -> List[dict]:
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []


def merge_results(existing: List[dict], new: List[dict],
                  replace_prompts: Iterable[str]) -> List[dict]:
    """*new* frames folded into *existing*, replacing the named prompts.

    A re-run of "deer" replaces the deer entries on the frames it processed
    and leaves "boar" (and every frame it did not touch) alone; the same
    holds for a clicked object re-run with more points.
    """
    names = set(replace_prompts)
    by_frame: Dict[int, dict] = {}
    for entry in existing:
        by_frame[int(entry.get("frame_idx", 0))] = {
            "frame_idx": int(entry.get("frame_idx", 0)),
            "imagefile": entry.get("imagefile", ""),
            "prompts": list(entry.get("prompts", []) or []),
        }
    for entry in new:
        frame_idx = int(entry.get("frame_idx", 0))
        current = by_frame.get(frame_idx)
        kept = [p for p in (current["prompts"] if current else [])
                if p.get("prompt") not in names]
        by_frame[frame_idx] = {
            "frame_idx": frame_idx,
            "imagefile": entry.get("imagefile", "")
            or (current["imagefile"] if current else ""),
            "prompts": kept + list(entry.get("prompts", []) or []),
        }
    return [by_frame[k] for k in sorted(by_frame)]


def prompt_summary(results: Iterable[dict]) -> Dict[str, Tuple[int, int]]:
    """``{prompt: (frames, predictions)}`` over *results*."""
    summary: Dict[str, List[int]] = {}
    for entry in results:
        for prompt in entry.get("prompts", []) or []:
            count = len(prompt.get("predictions", []) or [])
            if count == 0:
                continue
            row = summary.setdefault(prompt.get("prompt", ""), [0, 0])
            row[0] += 1
            row[1] += count
    return {k: (v[0], v[1]) for k, v in summary.items()}


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def _close_ring(ring: List[List[float]]) -> List[List[float]]:
    ring = [[float(p[0]), float(p[1])] for p in ring]
    if ring and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return ring


def export_geojson(georef_results: List[dict], output_path: str,
                   epsg: Optional[int], modality: str,
                   frames: Optional[Iterable[int]] = None,
                   log_fn=None) -> Dict[str, int]:
    """Write the geo-referenced masks as GeoJSON polygons (WGS84 lon/lat).

    One feature per prediction: a ``Polygon`` for a single ring, a
    ``MultiPolygon`` when the mask fell into several regions. Properties
    carry the prompt, its kind, the tracked object id, the frame and the
    confidence, so the file stands on its own in any GIS.
    """
    from .exporters import common

    # Called first even for an empty document, so the CRS check fires
    # before anything is written.
    common.to_wgs84([], epsg)

    wanted = set(frames) if frames is not None else None
    features = []
    skipped = 0
    for entry in georef_results:
        frame_idx = int(entry.get("frame_idx", 0))
        if wanted is not None and frame_idx not in wanted:
            continue
        for prompt in entry.get("prompts", []) or []:
            for pred_idx, prediction in enumerate(
                    prompt.get("predictions", []) or []):
                rings = []
                for polygon in prediction.get("world_polygons", []) or []:
                    if len(polygon) < 3:
                        continue
                    ring = _close_ring(common.to_wgs84(
                        [(p[0], p[1]) for p in polygon], epsg))
                    rings.append([ring])
                if not rings:
                    skipped += 1
                    continue
                geometry = ({"type": "Polygon", "coordinates": rings[0]}
                            if len(rings) == 1 else
                            {"type": "MultiPolygon", "coordinates": rings})
                properties = {
                    "prompt": prompt.get("prompt", ""),
                    "prompt_type": prompt.get("prompt_type", PROMPT_TEXT),
                    "object_id": prediction.get(
                        "object_id", prompt.get("object_id")),
                    "frame": frame_idx,
                    "imagefile": entry.get("imagefile", ""),
                    "prediction_idx": pred_idx,
                    "confidence": round(float(prediction.get("confidence", 0.0)), 4),
                    "camera": "thermal" if modality == "t" else "rgb",
                }
                features.append({"type": "Feature", "geometry": geometry,
                                 "properties": properties})
    document = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    if log_fn:
        log_fn(f"GeoJSON written: {len(features)} segmentation polygon(s) "
               f"to {output_path}"
               + (f" - {skipped} empty mask(s) skipped" if skipped else ""))
    return {"features": len(features), "skipped": skipped}
