# -*- coding: utf-8 -*-
"""Sample detector: Ultralytics YOLO - what the plugin's built-in detection
does, as a custom implementation you can copy and change.

How a custom detector works
---------------------------
* One .py file with a class that has ``setup(context)`` and
  ``run(image, frame)``. Nothing is imported from the plugin.
* The file's folder belongs to it: other .py files next to it can be
  imported by name (``import my_helpers``), and weights or other data next
  to it are reached with
  ``os.path.join(os.path.dirname(__file__), "weights.pt")``.
* ``setup`` is called once per run. ``context`` carries:
    context.camera          "thermal" or "rgb"
    context.device          "cuda", "mps" or "cpu"
    context.model_dir       the plugin's shared model folder
    context.options         the JSON object from the configuration
    context.min_confidence  the "Min Confidence" of the Detection tab
    context.log(message)    writes to the plugin log
  It may return a dict with ``class_names`` (what class index i means) and
  ``cameras`` (which of "thermal" / "rgb" it supports; default both).
* ``run`` gets one frame as an H x W x 3 BGR uint8 array (what
  ``cv2.imread`` returns) plus ``frame``: ``frame.index``, ``frame.path``,
  ``frame.timestamp`` and ``frame.pose`` - where the camera was
  (``pose.position`` / ``pose.world``, ``pose.matrix``, ``pose.fovy``,
  correction factors applied). It returns a list of
  ``(x1, y1, x2, y2, confidence, class_index)`` in pixels of that image.
  A class name may be given instead of the index.
* ``teardown()`` is optional.

Options (JSON in the configuration), all optional:
    {"weights": "C:/models/my_model.pt", "imgsz": 1280, "iou": 0.5}
Without "weights" the plugin's default BAMBI model for the camera is used,
if it has been downloaded already.

Dependencies: ``ultralytics`` (installed by the Dependency Manager). The
packages of your own detectors are yours to install into QGIS' Python:
the OSGeo4W Shell on Windows, or a terminal with the Python QGIS uses on
Linux / macOS: ``python -m pip install <package>``. Take care: installing
or upgrading packages can change ones QGIS, the plugin or other plugins
rely on (torch, transformers, numpy, ...) and break steps that worked
before - install only what is missing and check the Dependency Manager
afterwards.
"""

import os

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = ["ultralytics"]


class UltralyticsYoloDetector:
    BAMBI_KIND = "detector"

    #: file names of the plugin's default models in ``context.model_dir``
    DEFAULT_WEIGHTS = {"thermal": "thermal_animal_detector.pt",
                       "rgb": "rgb_animal_detector.pt"}

    def setup(self, context):
        from ultralytics import YOLO

        options = context.options
        weights = options.get("weights") or os.path.join(
            context.model_dir, self.DEFAULT_WEIGHTS[context.camera])
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Model file not found: {weights} - set \"weights\" in the "
                "options")
        self.model = YOLO(weights, verbose=False)
        self.device = context.device
        self.min_confidence = context.min_confidence
        self.predict_args = {key: options[key] for key in ("imgsz", "iou")
                             if key in options}
        # ``names`` is {index: name}; the plugin wants a list in index order
        names = self.model.names
        self.names = [names[i] for i in sorted(names)]
        context.log(f"YOLO model {os.path.basename(weights)} on {self.device}")
        return {"class_names": self.names, "cameras": ["thermal", "rgb"]}

    def class_names(self):
        return self.names

    def run(self, image, frame):
        results = self.model.predict(
            image, conf=self.min_confidence, device=self.device,
            verbose=False, **self.predict_args)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((x1, y1, x2, y2, float(box.conf.item()),
                                   int(box.cls.item())))
        return detections

    def teardown(self):
        self.model = None
