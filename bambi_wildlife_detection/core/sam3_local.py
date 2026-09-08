# -*- coding: utf-8 -*-
"""SAM3 concept segmentation run locally through ``transformers``.

The segmentation step has always gone through Roboflow's hosted SAM3 endpoint,
which needs a Roboflow key and a network round trip per frame. Meta publishes
the same model on Hugging Face (:data:`DEFAULT_SAM3_REPO`) - **gated**, exactly
like the DINOv3 backbone the classifiers read - so a user who already has a
token with access can run it on their own GPU instead.

The output is normalised to the very structure the Roboflow branch writes
(``prompt -> predictions -> {confidence, polygons}``), so the geo-referencing
step, the store migration and the exporters do not know or care which backend
produced a mask.

Everything heavy (``torch``, ``transformers``, ``cv2``) is imported lazily:
this module is importable in the headless test stub and in a QGIS that has
none of it installed.
"""

import os
import time
from typing import Dict, List, Optional, Sequence

from . import hf_access

#: Meta's SAM3 checkpoint on Hugging Face. Gated: request access there first.
DEFAULT_SAM3_REPO = "facebook/sam3"

#: ``transformers`` release that first shipped ``Sam3Model``. The DINOv3 floor
#: (4.56) is older, so a working classification setup can still be too old
#: for this.
SAM3_TRANSFORMERS_MIN = "5.0.0"


class Sam3LocalError(RuntimeError):
    """SAM3 could not be loaded or run locally."""


# ---------------------------------------------------------------------------
# Mask -> polygon
# ---------------------------------------------------------------------------

def mask_to_polygons(mask, min_area: float = 1.0) -> List[List[List[float]]]:
    """Outer rings of the connected regions in a binary *mask*.

    Returns ``[[[x, y], …], …]`` in pixel coordinates - the same shape the
    Roboflow ``polygon`` format delivers, so the downstream code is shared.
    Holes are dropped: a mask with a hole is still one animal, and the
    georeferencing only ever consumed outer rings.
    """
    import numpy as np

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[0]
    binary = (array > 0).astype(np.uint8)
    if not binary.any():
        return []

    try:
        import cv2
    except ImportError as exc:
        raise Sam3LocalError(
            "OpenCV (cv2) is required to turn SAM3 masks into polygons. "
            "Install the core dependencies from the Dependency Manager.") from exc

    contours, _hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < min_area:
            continue
        polygons.append([[float(x), float(y)] for x, y in contour[:, 0, :]])
    return polygons


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _sam3_classes():
    """``(Sam3Model, Sam3Processor)``, or a clear error about why not."""
    try:
        import transformers
    except ImportError as exc:
        from .classification import _transformers_import_message
        raise Sam3LocalError(_transformers_import_message(exc)) from exc

    model_cls = getattr(transformers, "Sam3Model", None)
    processor_cls = getattr(transformers, "Sam3Processor", None)
    if model_cls is None or processor_cls is None:
        installed = getattr(transformers, "__version__", "unknown")
        raise Sam3LocalError(
            f"transformers {installed} has no SAM3 support; local SAM3 needs "
            f"transformers >= {SAM3_TRANSFORMERS_MIN}. Re-install the "
            "Classification dependencies from the Dependency Manager to "
            "upgrade, or untick 'Run SAM3 locally' to use Roboflow.")
    return model_cls, processor_cls


class LocalSam3:
    """SAM3 loaded once and asked per frame. Mirrors ``classification.Backbone``.

    *models_dir* is the plugin's shared model folder; the checkpoint lands in
    the same Hugging Face cache the DINOv3 backbone uses, so one download
    serves every project.
    """

    def __init__(self, model_id: str = "", models_dir: str = "",
                 # "no token supplied" is a state, not a credential.
                 token: str = "",  # nosec B107
                 device: str = "auto", log_fn=None):
        from .classification import split_revision

        self.model_id, self.revision = split_revision(
            model_id or DEFAULT_SAM3_REPO)
        self.models_dir = models_dir
        self.token = token
        self.device_preference = device
        self.device: Optional[str] = None
        self._log = log_fn
        self._model = None
        self._processor = None

    # -- loading ----------------------------------------------------------

    def load(self) -> None:
        """Download (first run) and instantiate the model. Idempotent."""
        if self._model is not None:
            return
        from .classification import (
            _load_failure_message, describe_device, resolve_device)

        model_cls, processor_cls = _sam3_classes()
        self.device = resolve_device(self.device_preference)

        cache_dir = (hf_access.backbone_cache_dir(self.models_dir)
                     if self.models_dir else None)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        kwargs = {"token": self.token or None}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if self.revision:
            kwargs["revision"] = self.revision

        if self._log:
            self._log(f"Loading {self.model_id} on {self.device}… (the "
                      "checkpoint is ~3.4 GB, so a first run downloads for "
                      "a while)")
        started = time.monotonic()
        try:
            # nosec B615 - the revision is optional on purpose: the model id
            # is user-configurable, so a hardcoded pin would be wrong for a
            # fine-tuned SAM3. A user pins one with 'repo@commit'.
            self._processor = processor_cls.from_pretrained(
                self.model_id, **kwargs)  # nosec B615
            self._model = model_cls.from_pretrained(
                self.model_id, **kwargs)  # nosec B615
        except Exception as exc:
            raise Sam3LocalError(
                _load_failure_message(self.model_id, exc)) from exc

        self._model.to(self.device)
        self._model.eval()
        if self._log:
            self._log(f"SAM3 ready on {describe_device(self.device)} in "
                      f"{time.monotonic() - started:.0f} s")

    # -- inference --------------------------------------------------------

    def segment(self, image, prompts: Sequence[str],
                confidence: float = 0.5,
                mask_threshold: float = 0.5) -> List[Dict]:
        """Concept-segment one PIL *image* for every prompt.

        Returns ``[{"prompt": str, "predictions": [{"confidence": float,
        "polygons": [...]}, …]}, …]`` - the Roboflow-normalised structure.
        Prompts are asked one at a time: SAM3 scores each concept on its own,
        and a frame rarely carries more than a handful of prompts.
        """
        self.load()
        import torch

        results = []
        for prompt in prompts:
            inputs = self._processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            processed = self._processor.post_process_instance_segmentation(
                outputs, threshold=confidence, mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist())[0]
            results.append({
                "prompt": prompt,
                "predictions": normalise_predictions(processed),
            })
        return results

    def segment_path(self, image_path: str, prompts: Sequence[str],
                     confidence: float = 0.5) -> List[Dict]:
        """:meth:`segment` on an image file."""
        from PIL import Image

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        return self.segment(image, prompts, confidence)


def normalise_predictions(processed: Dict) -> List[Dict]:
    """Roboflow-shaped predictions from one post-processed SAM3 result.

    *processed* is one entry of ``post_process_instance_segmentation``:
    ``masks`` (N×H×W, binary), ``scores`` (N) and ``boxes`` (N×4, xyxy). A
    mask that yields no polygon - a stray pixel or an empty mask - is
    dropped, as a Roboflow response would not have listed it.
    """
    masks = processed.get("masks")
    scores = processed.get("scores")
    if masks is None or scores is None:
        return []
    masks = _to_numpy(masks)
    scores = _to_numpy(scores)
    boxes = processed.get("boxes")
    boxes = _to_numpy(boxes) if boxes is not None else None

    predictions = []
    for index in range(len(scores)):
        polygons = mask_to_polygons(masks[index])
        if not polygons:
            continue
        prediction = {
            "confidence": float(scores[index]),
            "polygons": polygons,
        }
        if boxes is not None and index < len(boxes):
            prediction["box"] = [float(v) for v in boxes[index][:4]]
        predictions.append(prediction)
    return predictions


def _to_numpy(value):
    """A numpy array from a tensor, an array or a nested list."""
    import numpy as np

    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if numpy is not None:
        try:
            return numpy()
        except TypeError:
            # bf16 has no numpy dtype; go through float32.
            return value.float().numpy()
    return np.asarray(value)
