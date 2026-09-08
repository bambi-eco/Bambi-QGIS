# -*- coding: utf-8 -*-
"""Crops, DINOv3 features and the classification heads.

Implements §3.3 of *When One Modality Is Not Enough*. This module is the part
that touches models; the persistence lives in :mod:`core.classification_store`
and :mod:`core.embedding_files`.

``torch`` and ``transformers`` are imported lazily inside functions, so a
project that never classifies anything neither needs them installed nor pays
their import cost - and the whole geometry half of this module stays testable
without either.
"""

import math
import os
import threading
import time
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from . import hf_access

#: Fill value for the parts of a crop window that fall outside the frame.
#: Mid-grey rather than black: an animal at the frame edge otherwise gets a
#: hard black border, which is a strong edge the backbone will happily encode.
EDGE_FILL = 114


class CropConfig(NamedTuple):
    """How a detection box becomes the square image the backbone sees."""

    #: Fraction of the box added on every side. Some context helps the
    #: backbone; too much and the animal is a speck in a field of grass.
    #: 0.25 on each side of the longer edge is the published heads' training
    #: convention - a square crop at 1.5x the longer box side, centred on the
    #: animal - and their feature space is locked to it: a different crop
    #: silently degrades accuracy rather than failing.
    padding: float = 0.25
    #: Side length of the crop handed to the backbone.
    size: int = 224
    #: Keep the aspect ratio and pad to square, rather than stretching. A deer
    #: seen from above is elongated, and stretching changes its proportions -
    #: which is part of what the sex head is reading.
    letterbox: bool = True


class Window(NamedTuple):
    """A crop rectangle in frame pixels. May extend beyond the frame."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


# ---------------------------------------------------------------------------
# Crop geometry - pure, and the part most worth pinning down
# ---------------------------------------------------------------------------

def crop_window(box: Sequence[float], config: CropConfig = CropConfig(),
                size_override: Optional[Tuple[float, float]] = None) -> Window:
    """The rectangle to cut for *box*, before any clamping to the frame.

    Deliberately allowed to fall outside the image: an animal at the edge of
    the frame still has a centred crop that way, with the missing part filled
    in. Clamping here instead would shift the animal off-centre exactly when it
    is already hardest to classify.

    *size_override* replaces the box's own dimensions while keeping its centre -
    this is how a matched RGB crop is sized from its thermal partner.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = abs(x2 - x1) if size_override is None else float(size_override[0])
    height = abs(y2 - y1) if size_override is None else float(size_override[1])

    width *= (1.0 + 2.0 * config.padding)
    height *= (1.0 + 2.0 * config.padding)

    if config.letterbox:
        # Squaring here rather than padding the resized image keeps real
        # surroundings in the crop instead of inventing grey bars for them.
        width = height = max(width, height)

    return Window(cx - width / 2.0, cy - height / 2.0,
                  cx + width / 2.0, cy + height / 2.0)


def anchored_size(thermal_box: Sequence[float],
                  inverse_scale: Tuple[float, float]) -> Tuple[float, float]:
    """Size an RGB crop from its thermal partner's box.

    The thermal box is the looser of the two and so more reliably encloses the
    whole animal - antlers included, which is the cue the sex head depends on.
    *inverse_scale* is how many RGB pixels one thermal pixel spans, i.e. the
    scale of the inverse of the registration affine.

    Only the crop is affected. The stored detections are never rewritten, so
    geo-referencing and tracking stay valid - the paper redraws the boxes, but
    doing that here would invalidate two upstream stages for a crop-quality
    gain.
    """
    x1, y1, x2, y2 = (float(v) for v in thermal_box)
    return (abs(x2 - x1) * inverse_scale[0], abs(y2 - y1) * inverse_scale[1])


def extract_crop(image, window: Window, size: int,
                 fill: int = EDGE_FILL):
    """Cut *window* out of *image* and resize it to ``size`` x ``size``.

    Parts of the window outside the image are filled rather than clipped, so
    the animal stays where the geometry says it is.
    """
    import numpy as np

    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    left, top = int(math.floor(window.x1)), int(math.floor(window.y1))
    right, bottom = int(math.ceil(window.x2)), int(math.ceil(window.y2))
    out_width, out_height = max(1, right - left), max(1, bottom - top)

    shape = ((out_height, out_width, channels) if image.ndim == 3
             else (out_height, out_width))
    canvas = np.full(shape, fill, dtype=image.dtype)

    # The overlap between the window and the image, in both coordinate frames.
    src_x1, src_y1 = max(0, left), max(0, top)
    src_x2, src_y2 = min(width, right), min(height, bottom)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[src_y1 - top:src_y2 - top, src_x1 - left:src_x2 - left] = \
            image[src_y1:src_y2, src_x1:src_x2]

    import cv2

    interpolation = (cv2.INTER_AREA if out_width > size else cv2.INTER_LINEAR)
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def to_rgb(image):
    """Three channels, however the frame was stored.

    Thermal frames are colourised to 8-bit RGB by the extraction step, but a
    single-channel frame is still possible, and DINOv3 wants three.
    """
    import numpy as np

    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        return image[:, :, :3]
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


# ---------------------------------------------------------------------------
# The backbone
# ---------------------------------------------------------------------------

class BackboneError(RuntimeError):
    """The DINOv3 backbone could not be loaded or run."""


def split_revision(model_id: str) -> Tuple[str, str]:
    """Split ``repo@revision`` into its parts; the revision may be empty.

    Hugging Face resolves an unpinned model to whatever ``main`` holds today,
    so a repository that changes changes the features under a user's saved
    results. Pinning is offered rather than imposed: the backbone is
    user-configurable, so a hardcoded revision here would be wrong for anyone
    using their own, and a released checkpoint like DINOv3 does not move in
    practice. Someone who needs reproducibility writes
    ``facebook/dinov3-vith16plus-pretrain-lvd1689m@<sha>``.
    """
    text = (model_id or "").strip()
    # A leading @ would be an empty repo name, and an org/repo never contains
    # one, so the last @ is unambiguous.
    if "@" in text[1:]:
        repo, _sep, revision = text.rpartition("@")
        return repo.strip(), revision.strip()
    return text, ""


def resolve_device(preference: str = "auto") -> str:
    """``"cuda"`` or ``"cpu"``, honouring the user's choice where possible."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise BackboneError(
            "PyTorch is not installed. Install the Classification "
            "dependencies from the Dependency Manager.") from exc

    available = torch.cuda.is_available()
    if preference == "cuda" and not available:
        raise BackboneError(
            "CUDA was requested but no CUDA-capable PyTorch build is "
            "available. Install GPU support from the Dependency Manager, or "
            "set the device to Auto.")
    return "cuda" if available else "cpu"


def reduced_precision_dtype():
    """The reduced-precision dtype to run the backbone in, or ``None``.

    bfloat16 where the GPU supports it: it trades mantissa for the full fp32
    exponent range, which is exactly what the ViT-H+ needs - fp16 overflows
    in its attention and returns NaN. ``None`` means "use ``half()``" (fp16),
    for older GPUs and for torch builds without the probe.
    """
    try:
        import torch
        probe = getattr(torch.cuda, "is_bf16_supported", None)
        if probe is not None and probe():
            return getattr(torch, "bfloat16", None)
    except Exception:  # nosec B110
        pass
    return None


class Backbone:
    """Frozen DINOv3, producing one CLS vector per crop.

    Loaded once and reused: the model is ~840 M parameters, so paying that
    twice would be the dominant cost of a run.
    """

    def __init__(self, model_id: str = hf_access.DEFAULT_BACKBONE,
                 models_dir: str = "",
                 # "no token supplied" is a state the caller reports, not a
                 # hardcoded credential.
                 token: str = "",  # nosec B107
                 device: str = "auto", fp16: bool = True, log_fn=None):
        self.model_id, self.revision = split_revision(
            model_id or hf_access.DEFAULT_BACKBONE)
        self.models_dir = models_dir
        self.token = token
        self.device = resolve_device(device)
        # Half precision is a GPU concern; on CPU it is slower, not faster.
        self.fp16 = bool(fp16) and self.device == "cuda"
        self._log = log_fn
        self._model = None
        self._processor = None
        self._reduced_dtype = None

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        """Fetch and instantiate the backbone. Idempotent."""
        if self._model is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise BackboneError(_transformers_import_message(exc)) from exc

        # cache_dir rather than HF_HOME: setting that environment variable
        # would relocate the cache for everything else in the QGIS process
        # that uses Hugging Face, which is not ours to do.
        cache_dir = (hf_access.backbone_cache_dir(self.models_dir)
                     if self.models_dir else None)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if self._log:
            self._log(f"Loading {self.model_id} on {self.device}"
                      f"{' (fp16)' if self.fp16 else ''}…")

        kwargs = {"token": self.token or None}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if self.revision:
            kwargs["revision"] = self.revision

        # The checkpoint is gigabytes, so the wait between "Loading" and
        # "ready" is minutes on a first run - say what the time is going on.
        cached = describe_cache(self.model_id, cache_dir, self.revision,
                                self.token or None, log_fn=self._log)
        started = time.monotonic()
        try:
            # nosec B615 - the revision is deliberately optional: the model id
            # is user-configurable, so a hardcoded pin would be wrong for a
            # custom backbone. See split_revision for how a user pins one.
            with DownloadReporter(cache_dir, self.model_id, cached.expected_bytes,
                                  log_fn=self._log, active=not cached.complete):
                self._processor = AutoImageProcessor.from_pretrained(
                    self.model_id, **kwargs)  # nosec B615
                if self._log:
                    self._log("Image processor ready; loading the weights…")
                self._model = AutoModel.from_pretrained(
                    self.model_id, **kwargs)  # nosec B615
        except Exception as exc:
            raise BackboneError(_load_failure_message(self.model_id, exc)) from exc

        if self._log:
            self._log(f"Weights loaded in {time.monotonic() - started:.0f} s"
                      f"{_parameter_note(self._model)}; moving to "
                      f"{self.device}…")
        self._model.to(self.device)
        if self.fp16:
            self._reduced_dtype = reduced_precision_dtype()
            if self._reduced_dtype is not None:
                self._model.to(self._reduced_dtype)
            else:
                self._model.half()
        self._model.eval()

        if self._log:
            self._log(f"Backbone ready ({self.dim}-d features) on "
                      f"{describe_device(self.device)}"
                      f"{' in ' + self.precision if self.fp16 else ''}, "
                      f"{time.monotonic() - started:.0f} s in total")

    @property
    def precision(self) -> str:
        """``"bf16"``, ``"fp16"`` or ``"fp32"`` - what the weights are in."""
        if not self.fp16:
            return "fp32"
        name = str(getattr(self._reduced_dtype, "__name__", None)
                   or self._reduced_dtype or "")
        return "bf16" if "bfloat16" in name else "fp16"

    def _demote_to_fp32(self, reason: str) -> None:
        """Leave reduced precision for the rest of the run.

        Tried on the current device first; when the fp32 weights do not fit
        (3.4 GB for the ViT-H+), the model moves to the CPU rather than
        failing - slower, but it produces numbers.
        """
        import torch

        self.fp16 = False
        self._reduced_dtype = None
        try:
            self._model.float()
            where = self.device
        except RuntimeError:
            # CUDA out of memory: the reduced-precision weights alone nearly
            # filled the card.
            self._model.to("cpu").float()
            self.device = "cpu"
            try:
                torch.cuda.empty_cache()
            except Exception:  # nosec B110
                pass
            where = "cpu (the fp32 weights do not fit on the GPU)"
        if self._log:
            self._log(f"{reason}; continuing in fp32 on {where}")

    @property
    def dim(self) -> int:
        """Width of the CLS vector this backbone produces."""
        if self._model is None:
            return hf_access.BACKBONE_DIM
        return int(getattr(self._model.config, "hidden_size",
                           hf_access.BACKBONE_DIM))

    # -- inference -------------------------------------------------------

    def embed(self, crops: Sequence["object"]):
        """CLS vectors for a batch of ``HxWx3`` uint8 crops, as float32."""
        import numpy as np
        import torch

        if not len(crops):
            return np.zeros((0, self.dim), dtype=np.float32)

        self.load()
        features = self._forward(crops)

        # fp16 has a range of 65 504, and the ViT-H+ exceeds it inside the
        # attention on some crops: the CLS vector comes out NaN and nothing
        # downstream can use it (2026-09-07: 3400 of 3428 vectors on a flight
        # were NaN, and the heads then wrote NaN probabilities). bf16 keeps
        # the fp32 range and is preferred where the GPU supports it; whatever
        # the precision, a non-finite batch is redone in fp32 rather than
        # written to disk.
        finite = np.isfinite(features).all(axis=1)
        if not finite.all():
            bad = int((~finite).sum())
            if self.fp16:
                self._demote_to_fp32(
                    f"{bad} of {len(features)} feature vector(s) came out "
                    f"non-finite in {self.precision}")
                features = self._forward(crops)
                finite = np.isfinite(features).all(axis=1)
            if not finite.all():
                raise BackboneError(
                    f"{int((~finite).sum())} of {len(features)} feature "
                    "vector(s) are non-finite even in fp32. The crops fed to "
                    "the backbone are probably empty or corrupt; check the "
                    "extracted frames.")
        return features

    def _forward(self, crops: Sequence["object"]):
        """One pass through the backbone, as float32 rows."""
        import torch

        inputs = self._processor(images=list(crops), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.fp16:
            inputs = {k: (self._to_reduced(v) if v.dtype == torch.float32
                          else v)
                      for k, v in inputs.items()}

        # no_grad throughout: without it the autograd graph grows across the
        # whole flight and the run dies on memory rather than on time.
        with torch.no_grad():
            outputs = self._model(**inputs)

        features = getattr(outputs, "pooler_output", None)
        if features is None:
            # Not every checkpoint exposes a pooler; the CLS token is the
            # first position of the sequence either way, and that is what the
            # published heads were trained on.
            features = outputs.last_hidden_state[:, 0]
        return features.detach().to("cpu").float().numpy()

    def _to_reduced(self, tensor):
        if self._reduced_dtype is not None:
            return tensor.to(self._reduced_dtype)
        return tensor.half()


def repo_cache_folder(cache_dir: Optional[str], model_id: str) -> str:
    """The folder the Hugging Face cache keeps *model_id* in, or ``""``.

    ``hf_cache/models--facebook--dinov3-.../`` - the hub's own naming, taken
    from ``huggingface_hub`` when it is importable and reproduced otherwise.
    """
    if not cache_dir:
        return ""
    try:
        from huggingface_hub.file_download import repo_folder_name
        name = repo_folder_name(repo_id=model_id, repo_type="model")
    except Exception:  # nosec B110 - the naming has been stable for years
        name = "models--" + model_id.replace("/", "--")
    return os.path.join(cache_dir, name)


def cached_bytes(repo_folder: str) -> int:
    """Bytes of *repo_folder*'s blobs on disk, partial downloads included.

    The hub streams each file into ``blobs/<hash>.incomplete`` and moves it
    into place when done, so the folder size grows with the download - which
    is the progress signal, without hooking the hub's own progress bars.

    Where the finished file ends up differs by platform: with symlinks the
    blob stays in ``blobs/`` and ``snapshots/<rev>/`` links to it; on Windows
    without symlink rights the file sits under ``snapshots/`` and ``blobs/``
    is left empty. Both are walked, links are not followed, so a file is
    counted exactly once either way.
    """
    if not repo_folder:
        return 0
    total = 0
    for base in ("blobs", "snapshots"):
        root = os.path.join(repo_folder, base)
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    pass
    return total


class CacheState(NamedTuple):
    """What is known about the backbone's local cache before loading."""

    complete: bool
    expected_bytes: Optional[int]


def describe_cache(model_id: str, cache_dir: Optional[str], revision: str,
                   token: Optional[str], log_fn=None) -> CacheState:
    """Log whether the weights are on disk already, and how big they are.

    Answers the question the log otherwise leaves open for minutes: is this
    downloading, or loading? The remote size comes from one metadata call
    and is skipped when the snapshot is already complete; every hub call is
    best-effort, so a missing or old ``huggingface_hub`` only costs detail.
    """
    folder = repo_cache_folder(cache_dir, model_id)
    on_disk = cached_bytes(folder)

    complete = False
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(
            model_id, "config.json", cache_dir=cache_dir or None,
            revision=revision or None)
        # A cached config.json plus no partial blobs means a finished snapshot.
        partial = any(
            name.endswith(".incomplete")
            for name in _listdir(os.path.join(folder, "blobs")))
        complete = isinstance(found, str) and on_disk > 0 and not partial
    except Exception:  # nosec B110
        complete = False

    if complete:
        if log_fn:
            log_fn(f"Weights are in the local cache ({_mb(on_disk)} MB at "
                   f"{folder}); no download needed")
        return CacheState(True, on_disk)

    expected = None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, revision=revision or None,
                                  token=token, files_metadata=True)
        sizes = [getattr(s, "size", None) for s in (info.siblings or [])]
        if any(sizes):
            expected = int(sum(s for s in sizes if s))
    except Exception:  # nosec B110 - offline, gated, old hub: size unknown
        expected = None

    if log_fn:
        size = f"about {_mb(expected)} MB" if expected else "several GB"
        have = f"; {_mb(on_disk)} MB already on disk" if on_disk else ""
        log_fn(f"Weights are not in the local cache yet - downloading "
               f"{size} from Hugging Face into {folder or 'the default cache'}"
               f"{have}. This happens once; later runs load from disk.")
    return CacheState(False, expected)


def _listdir(folder: str) -> List[str]:
    try:
        return os.listdir(folder)
    except OSError:
        return []


def _mb(value: Optional[int]) -> str:
    return f"{(value or 0) / (1024 * 1024):,.0f}"


class DownloadReporter:
    """Log the growth of the cache folder while ``from_pretrained`` runs.

    A thread samples :func:`cached_bytes` every *interval* seconds and logs a
    line whenever the number moved, with a percentage when the total is
    known. Inactive when the weights are already cached, so a warm start
    logs nothing extra.
    """

    def __init__(self, cache_dir: Optional[str], model_id: str,
                 expected_bytes: Optional[int], log_fn=None,
                 active: bool = True, interval: float = 5.0):
        self._folder = repo_cache_folder(cache_dir, model_id)
        self._expected = expected_bytes
        self._log = log_fn
        self._active = bool(active and log_fn and self._folder)
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last = -1

    def __enter__(self):
        if self._active:
            self._thread = threading.Thread(
                target=self._run, name="bambi-download-progress", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
        return False

    def _run(self):
        while not self._stop.wait(self._interval):
            self.report()

    def report(self):
        """One sample; public so a test can drive it without the thread."""
        current = cached_bytes(self._folder)
        if current == self._last or current <= 0:
            return
        self._last = current
        line = f"Downloading weights: {_mb(current)} MB"
        if self._expected:
            percent = min(100.0, 100.0 * current / self._expected)
            line += f" of {_mb(self._expected)} MB ({percent:.0f}%)"
        self._log(line)


def _parameter_note(model) -> str:
    """`` (840 M parameters)`` when the model can count them, else ``""``."""
    try:
        count = sum(int(p.numel()) for p in model.parameters())
    except Exception:  # nosec B110 - a fake or exotic model
        return ""
    return f" ({count / 1e6:,.0f} M parameters)" if count else ""


def describe_device(device: str) -> str:
    """``cuda (NVIDIA RTX A2000, 11.2 GB free of 12.0 GB)`` or just the name."""
    if not str(device).startswith("cuda"):
        return str(device)
    try:
        import torch
        index = torch.device(device).index or 0
        name = torch.cuda.get_device_name(index)
        free, total = torch.cuda.mem_get_info(index)
        return (f"{device} ({name}, {free / 2**30:.1f} GB free of "
                f"{total / 2**30:.1f} GB)")
    except Exception:  # nosec B110
        return str(device)


def _transformers_import_message(exc: ImportError) -> str:
    """Say what actually stopped ``transformers`` from importing.

    ``transformers`` imports lazily and pulls in a long tail (scikit-learn,
    pyarrow, ...), so an ``ImportError`` here is more often one of *those*
    breaking than transformers being absent. Reporting "not installed" for a
    package the Dependency Manager shows as installed sends the user to the
    wrong place - the real case was a user-site pyarrow that could not load
    its DLL inside QGIS.
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name == "transformers":
        return ("transformers is not installed. Install the Classification "
                "dependencies from the Dependency Manager.")
    # The innermost cause is the one that names the broken package.
    root = exc
    while root.__cause__ is not None or root.__context__ is not None:
        root = root.__cause__ or root.__context__
    culprit = getattr(root, "name", None) or ""
    module = f" while importing '{culprit}'" if culprit else ""
    message = (f"transformers is installed but could not be imported{module}: "
               f"{root}")
    if "dll load failed" in str(root).lower():
        message += (
            "\n\nA native library failed to load inside QGIS. This usually "
            "means a package in the user site-packages shadows the copy QGIS "
            "ships, and the two disagree on the DLLs QGIS has already loaded. "
            "Open the Dependency Manager and repair the shadowed packages, "
            "or uninstall the user-site copy of that package.")
    return message


def _load_failure_message(model_id: str, exc: Exception) -> str:
    """Turn a backbone load failure into something actionable."""
    text = str(exc)
    lowered = text.lower()
    if "gated" in lowered or "401" in text or "403" in text:
        return (
            f"{model_id} is gated and could not be downloaded. Request access "
            f"at https://huggingface.co/{model_id}, then enter a read token in "
            "the Classification tab and press 'Check access'.\n\n"
            f"Original error: {text}")
    if "connection" in lowered or "resolve" in lowered or "timed out" in lowered:
        return (f"Could not reach Hugging Face to download {model_id}. Check "
                f"the network connection or any proxy settings.\n\n"
                f"Original error: {text}")
    return f"Could not load {model_id}: {text}"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class HeadError(RuntimeError):
    """A classification head could not be loaded or run."""


#: What to do with a track that has no cross-modal partner when a head is
#: configured ``matched``. The matched heads were trained on real pairs and
#: cannot take a zero-filled stand-in for a missing modality, so this has to
#: be an explicit choice rather than a silent one.
UNMATCHED_SKIP = "skip"
UNMATCHED_RGB = "rgb"
UNMATCHED_THERMAL = "thermal"

#: A stage's Input answers one question, not two: which feature view its head
#: reads, and therefore which modalities' animals carry the answer. ``matched``
#: fuses both cameras, so both sides of a pair are the same animal and both
#: get the call; a single-camera view labels only that camera's tracks.
INPUT_TARGETS = {
    "thermal": ("t",),
    "rgb": ("w",),
    "matched": ("t", "w"),
}

#: Feature view name of a modality suffix, as a head's ``modality`` reads.
MODALITY_NAMES = {"t": "thermal", "w": "rgb"}


def targets_of(spec: dict) -> tuple:
    """Modalities a stage configured by *spec* records its answers on."""
    return INPUT_TARGETS.get(spec.get("modality", "matched"), ("t", "w"))


class FeatureResolver:
    """Assembles the feature vector a head expects, per detection.

    Classification is anchored on one modality - its tracks are what the
    answers are recorded against - but a ``matched`` head needs both sides.
    This turns "detection 4711 of the primary modality" into whichever vector
    the configured head was trained on, or ``None`` when the pieces are not
    all there.

    Concatenation is always ``[RGB, thermal]`` regardless of which modality is
    primary, because that is the order the published heads were trained on.
    """

    def __init__(self, primary: str,
                 vectors_t: Dict[int, object],
                 vectors_w: Dict[int, object],
                 partner_of_primary: Optional[Dict[int, int]] = None):
        if primary not in ("t", "w"):
            raise ValueError(f"Unknown modality: {primary!r}")
        self.primary = primary
        self.vectors_t = vectors_t or {}
        self.vectors_w = vectors_w or {}
        self.partner = partner_of_primary or {}

    def thermal(self, detection_id: int):
        if self.primary == "t":
            return self.vectors_t.get(detection_id)
        return self.vectors_t.get(self.partner.get(detection_id))

    def rgb(self, detection_id: int):
        if self.primary == "w":
            return self.vectors_w.get(detection_id)
        return self.vectors_w.get(self.partner.get(detection_id))

    def resolve(self, detection_id: int, modality_in: str,
                unmatched: str = UNMATCHED_SKIP):
        """The vector for *detection_id* under *modality_in*, or ``None``."""
        import numpy as np

        if modality_in == "thermal":
            return self.thermal(detection_id)
        if modality_in == "rgb":
            return self.rgb(detection_id)
        if modality_in != "matched":
            raise ValueError(f"Unknown input configuration: {modality_in!r}")

        rgb = self.rgb(detection_id)
        thermal = self.thermal(detection_id)
        if rgb is not None and thermal is not None:
            return np.concatenate([np.asarray(rgb), np.asarray(thermal)])

        # No partner. Zeros are not an option - the matched heads were trained
        # on real pairs - so fall back to a single-modality view or decline.
        if unmatched == UNMATCHED_RGB:
            return rgb
        if unmatched == UNMATCHED_THERMAL:
            return thermal
        return None

    def resolved_dim(self, modality_in: str, backbone_dim: int) -> int:
        """Feature width a head reading *modality_in* expects."""
        return backbone_dim * 2 if modality_in == "matched" else backbone_dim


#: Label used when a head reports no class names of its own. The mapping is
#: keyed on the class *index* anyway, so a name is only ever for the user.
UNNAMED_CLASS = "class %d"


def head_classes(module, feature_dim: int = 0) -> Tuple[List[str], str]:
    """The head's class list and where it came from.

    Three sources, in order of how much they tell us (see the plan, §5.3):

    ``classes``
        the ``m.classes`` attribute our own heads export.
    ``probe``
        a forward pass with a zero feature vector, which yields the class
        *count* but no names - enough structure that the user only has to
        supply labels.
    ``unknown``
        neither worked; the caller falls back to a mapping defined by hand.

    A head is not required to name its classes: what a prediction means is
    decided by the project's label mapping, and that is keyed on index.
    """
    names = getattr(module, "classes", None)
    if names:
        try:
            listed = [str(name) for name in names]
            if listed:
                return listed, "classes"
        except TypeError:
            pass

    count = probe_class_count(module, feature_dim)
    if count:
        return [UNNAMED_CLASS % i for i in range(count)], "probe"
    return [], "unknown"


def probe_class_count(module, feature_dim: int) -> int:
    """How many classes a head returns, found by running one zero vector.

    Costs a single forward pass through a head of a few hundred kilobytes,
    which is worth it to avoid asking the user for something discoverable.
    """
    if not feature_dim:
        return 0
    try:
        import torch

        with torch.no_grad():
            _embedding, probabilities = module(
                torch.zeros((1, int(feature_dim)), dtype=torch.float32))
        return int(probabilities.shape[1])
    except Exception:
        return 0


class Head:
    """One TorchScript classification head.

    The published heads are exported with BatchNorm folded in, so they are
    deterministic without ``.eval()`` and need only be called under
    ``no_grad``.
    """

    def __init__(self, path: str, feature_dim: int = 0, log_fn=None):
        self.path = path
        self.feature_dim = feature_dim
        self._log = log_fn
        self._module = None
        self._classes: List[str] = []
        self._class_source = "unknown"

    def load(self) -> None:
        """Load the head. Idempotent."""
        if self._module is not None:
            return
        if not os.path.isfile(self.path):
            raise HeadError(f"Classifier not found: {self.path}")

        try:
            import torch
        except ImportError as exc:
            raise HeadError(
                "PyTorch is not installed. Install the Classification "
                "dependencies from the Dependency Manager.") from exc

        try:
            self._module = torch.jit.load(self.path, map_location="cpu")
        except Exception as exc:
            raise HeadError(
                f"Could not load the classifier at {self.path}: {exc}") from exc

        self._classes, self._class_source = head_classes(
            self._module, self.feature_dim)
        if self._log:
            self._log(f"Loaded {os.path.basename(self.path)} "
                      f"({len(self._classes)} classes, from "
                      f"{self._class_source})")

    @property
    def classes(self) -> List[str]:
        self.load()
        return list(self._classes)

    @property
    def class_source(self) -> str:
        self.load()
        return self._class_source

    def predict(self, features) -> List[Tuple[int, float]]:
        """``(class_index, probability)`` for each row of *features*.

        The winning index rather than a name: a head's own labels are a
        convenience, and the project's mapping is keyed on position because
        that is the only thing ``probs[i]`` guarantees.
        """
        import numpy as np
        import torch

        self.load()
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if not array.size:
            return []

        expected = self.feature_dim
        if expected and array.shape[1] != expected:
            raise HeadError(
                f"{os.path.basename(self.path)} expects {expected}-d features "
                f"but was given {array.shape[1]}-d. Check that the classifier "
                "matches the configured modality - a 'matched' head takes both "
                "modalities concatenated.")
        finite = np.isfinite(array).all(axis=1)
        if not finite.all():
            # A NaN feature gives a NaN probability, which SQLite stores as
            # NULL - the NOT NULL constraint on frame_predictions.prob was
            # how this first surfaced. Name the cause instead.
            raise HeadError(
                f"{int((~finite).sum())} of {len(array)} feature vector(s) "
                "contain NaN or infinite values, so the classifier cannot "
                "score them. The embeddings were computed in a precision "
                "the backbone overflowed in; re-run 'Compute Embeddings' - "
                "it recomputes the unusable vectors - and then this step.")

        with torch.no_grad():
            _embedding, probabilities = self._module(torch.from_numpy(array))
        probabilities = probabilities.detach().cpu().numpy()

        return [(int(np.argmax(row)), float(np.max(row)))
                for row in probabilities]


# ---------------------------------------------------------------------------
# Which frames a track votes over (see the plan, §5.2a)
# ---------------------------------------------------------------------------

#: Where the set of votable frames came from, recorded per track so two runs
#: that differ can be explained.
FRAMES_FROM_HEAD = "occlusion-head"
FRAMES_FROM_ANNOTATIONS = "annotations"
FRAMES_FROM_ALL = "all-frames"
FRAMES_FROM_NOTHING = "all-frames-unfiltered"


def visible_detections(detection_ids: Sequence[int],
                       head_labels: Optional[Dict[int, str]] = None,
                       annotated: Optional[Dict[int, object]] = None,
                       clear_head_labels: Sequence[str] = ("clear",),
                       clear_annotations: Sequence[object] = (),
                       use_all: bool = False) -> Tuple[List[int], str]:
    """The frames a track's species and sex votes may use.

    Occlusion is a *quality filter, not a prerequisite*: species and sex have
    to work whether or not it ran. Three sources, in order - the occlusion
    head if it ran, otherwise the stored annotations (which is what the
    labelling tool writes, and a hand annotation is better evidence than a
    78 %-accurate head), otherwise every frame.

    The third case is reported rather than silent: voting over occluded frames
    while the UI claims "visible only" would be the wrong kind of quiet.
    """
    ids = [int(i) for i in detection_ids]
    if use_all:
        return ids, FRAMES_FROM_ALL

    if head_labels:
        clear = set(clear_head_labels)
        chosen = [i for i in ids if head_labels.get(i) in clear]
        return chosen, FRAMES_FROM_HEAD

    if annotated:
        clear = set(clear_annotations)
        chosen = [i for i in ids if annotated.get(i) in clear]
        return chosen, FRAMES_FROM_ANNOTATIONS

    return ids, FRAMES_FROM_NOTHING


# ---------------------------------------------------------------------------
# Quorum
# ---------------------------------------------------------------------------

class Vote(NamedTuple):
    """The outcome of a track-level vote."""

    label: str
    class_index: int
    votes: int
    n: int
    fraction: float


def quorum_vote(calls: Sequence[Tuple[int, str]], quorum: float = 0.5,
                min_frames: int = 1) -> Optional[Vote]:
    """Aggregate per-frame calls into one answer for the animal.

    *calls* are ``(class_index, label)`` pairs, one per frame that was allowed
    to vote. Returns ``None`` when there are too few frames or no class clears
    the quorum - an abstention, which the caller records as "unknown" rather
    than by discarding the animal. The paper's whole point is that voting makes
    a noisy per-frame call safe: an antler resolves only from some angles, so
    many frames of a true male look female, and the vote is what recovers him.
    """
    if len(calls) < max(1, int(min_frames)):
        return None

    counts: Dict[Tuple[int, str], int] = {}
    for class_index, label in calls:
        key = (int(class_index), str(label))
        counts[key] = counts.get(key, 0) + 1

    # Ties break towards the lower class index, so the answer does not depend
    # on the order frames happened to be processed in.
    (index, label), votes = max(
        counts.items(), key=lambda item: (item[1], -item[0][0]))
    fraction = votes / len(calls)
    if fraction <= quorum:
        return None
    return Vote(label=label, class_index=index, votes=votes,
                n=len(calls), fraction=fraction)


def batched(items: Sequence, size: int) -> Iterable[Sequence]:
    """Split *items* into chunks of at most *size*."""
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def group_by_frame(rows: Iterable[dict]) -> Dict[int, List[dict]]:
    """Detections grouped by frame, so each image is decoded once."""
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["frame"]), []).append(row)
    return grouped
