# -*- coding: utf-8 -*-
"""Sample detector: DroneMegaDetector (ConservationDrones,
https://github.com/ConservationDronesAI/DroneMegaDetector, MIT licence).

NOT TESTED by the plugin authors against the real model - this file follows
the project's Hugging Face demo (HuggingFaceDemo.py) and is meant as a
starting point. Check the results on your own data.

What to know about DroneMegaDetector
------------------------------------
* An RF-DETR model that returns real bounding boxes for three classes:
  person, animal (in practice mammals) and vehicle.
* Trained on RGB drone imagery at 1280 x 1280, mostly from 50-120 m above
  ground. Larger frames are cut into overlapping tiles here ("tile": true)
  and the tiles' boxes merged.
* It was built for RGB imagery, so only "rgb" is declared as supported.

Setup (your responsibility - the plugin does not manage this)
------------------------------------------------------------
Install into QGIS' Python (the OSGeo4W Shell on Windows, or a terminal with
the Python QGIS uses on Linux / macOS) a transformers version that knows
RF-DETR:
    python -m pip install --upgrade transformers pillow
The weights (ConservationDrones/DroneMegaDetector) are downloaded from
Hugging Face on the first run.

IMPORTANT - installing packages can break other things: QGIS, the plugin
and all other plugins share one Python, and pip replaces packages they rely
on (torch, torchvision, transformers, numpy, opencv) if a new package asks
for other versions. Install only what is missing, avoid --upgrade unless an
error asks for it, check first with ``python -m pip install --dry-run ...``
and look at the plugin's Dependency Manager afterwards.
If the check fails while importing ``torchaudio`` ("Could not load this
library ... libtorchaudio"): transformers loads torchaudio when it is
installed, and one that does not match your torch breaks the import. It is
not needed here - ``python -m pip uninstall torchaudio``.

Options in the configuration (JSON), all optional:
    {"only_animals": true,     (drop persons and vehicles; default true)
     "tile": true,             (tile frames larger than 1280 px)
     "tile_size": 1280, "tile_overlap": 100,
     "revision": "<commit hash>"}   (pin the Hugging Face model version)
"""

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = ["torch", "transformers", "PIL", "cv2"]

REPOSITORY = "ConservationDrones/DroneMegaDetector"


class DroneMegaDetector:
    BAMBI_KIND = "detector"

    #: index 3 is an unused legacy slot of the published model
    NAMES = ["person", "animal", "vehicle"]

    def setup(self, context):
        import torch
        from transformers import (AutoImageProcessor,
                                  AutoModelForObjectDetection)

        options = context.options
        self.torch = torch
        self.device = context.device
        # "revision" pins the model to one commit of the Hugging Face
        # repository, so a later upload cannot change what runs here.
        context.log("Loading DroneMegaDetector from Hugging Face - the first "
                    "run downloads the model (about 150 MB)...")
        revision = options.get("revision") or None
        self.processor = AutoImageProcessor.from_pretrained(  # nosec B615
            REPOSITORY, revision=revision)
        self.model = AutoModelForObjectDetection.from_pretrained(  # nosec B615
            REPOSITORY, revision=revision).eval().to(self.device)
        self.threshold = max(0.05, float(context.min_confidence))
        self.only_animals = bool(options.get("only_animals", True))
        self.tile = bool(options.get("tile", True))
        self.tile_size = int(options.get("tile_size", 1280))
        self.overlap = int(options.get("tile_overlap", 100))
        context.log(f"DroneMegaDetector on {self.device}")
        return {"class_names": self.NAMES, "cameras": ["rgb"]}

    def class_names(self):
        return self.NAMES

    # -- one tile ---------------------------------------------------------

    def _predict(self, rgb):
        from PIL import Image

        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold,
            target_sizes=[(image.height, image.width)])[0]
        rows = []
        for score, label, box in zip(result["scores"], result["labels"],
                                     result["boxes"]):
            label = int(label)
            if label >= len(self.NAMES):
                continue
            if self.only_animals and self.NAMES[label] != "animal":
                continue
            x1, y1, x2, y2 = box.tolist()
            rows.append([x1, y1, x2, y2, float(score), label])
        return rows

    # -- whole frame ------------------------------------------------------

    def run(self, image, frame):
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        size = self.tile_size
        if not self.tile or (width <= size and height <= size):
            return [tuple(row) for row in self._predict(rgb)]

        step = max(1, size - self.overlap)
        rows = []
        for top in _starts(height, size, step):
            for left in _starts(width, size, step):
                for row in self._predict(rgb[top:top + size,
                                             left:left + size]):
                    row[0] += left
                    row[2] += left
                    row[1] += top
                    row[3] += top
                    rows.append(row)
        return [tuple(row) for row in _merge(rows)]

    def teardown(self):
        self.model = None


def _starts(length, size, step):
    """Tile offsets covering *length*, the last one flush with the edge."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return starts


def _iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _merge(rows, iou_threshold=0.5):
    """Non-maximum suppression across tiles: an animal in the overlap of
    two tiles is found twice."""
    kept = []
    for row in sorted(rows, key=lambda r: -r[4]):
        if all(row[5] != other[5] or _iou(row, other) < iou_threshold
               for other in kept):
            kept.append(row)
    return kept
