# -*- coding: utf-8 -*-
"""Sample classifier on embeddings: a TorchScript head on the DINOv3
vectors of step C1 - what the plugin's built-in classifiers do (triplet-net
heads), as a custom implementation you can copy and change.

How a custom classifier works
-----------------------------
* One .py file with a class that has ``setup(context)`` and ``run`` (or
  ``run_track``). Nothing is imported from the plugin.
* ``setup(context)`` MUST return what the classifier reads and decides:
      "input":  "embeddings"  the DINOv3 vectors of step C1 - an (N, D)
                              float32 array; for the "matched" input the
                              RGB and thermal vectors concatenated
                "crops"       the cut-out boxes themselves (step C1 is not
                              needed) - see classifier_crops_torchscript.py
      "level":  "frame"       ``run(batch, items)`` scores every row; the
                              plugin chooses the frames (occlusion) and
                              takes the vote per animal            [default]
                "track"       ``run_track(batch, items)`` gets one animal's
                              frames and returns its class itself - see
                              classifier_track_level.py
      "class_names":  what class index i means                    [required]
      "cameras":      which inputs it supports, any of "thermal", "rgb",
                      "matched"                                [default: all]
  ``context`` carries ``task`` ("occlusion", "species", "sex",
  "life_stage"), ``camera`` (the input selected for the stage),
  ``device``, ``model_dir``, ``options`` and ``log(message)``.
* ``run(batch, items)`` returns one ``(class_index, probability)`` per row,
  in order - or an (N, classes) array of probabilities. ``items`` tells
  which row is what: ``detection_id``, ``track_id``, ``frame``, ``box``
  (pixels), ``world`` (the geo-referenced box corners, or None) and
  ``pose`` - where the camera was on that frame (``pose.position`` /
  ``pose.world``, ``pose.matrix``, ``pose.fovy``, correction factors
  applied), e.g. for geo-fencing.
* The class names are connected to the project's species / sex / age values
  with the "Labels" button of the Classification tab, as for any model.

Options (JSON in the configuration):
    {"weights": "C:/models/species_head.pt",       (required, TorchScript)
     "class_names": ["roe deer", "red deer"]}      (if the file has none)

Dependencies: ``torch`` (installed by the Dependency Manager's
classification group). Packages of your own classifiers are yours to
install into QGIS' Python: the OSGeo4W Shell on Windows, or a terminal
with the Python QGIS uses on Linux / macOS:
``python -m pip install <package>``.
"""

import os

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = ["torch"]


class EmbeddingHeadClassifier:
    BAMBI_KIND = "classifier"

    def setup(self, context):
        import torch

        options = context.options
        weights = options.get("weights", "")
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Classifier not found: {weights!r} - set \"weights\" in the "
                "options")
        self.torch = torch
        self.module = torch.jit.load(weights, map_location="cpu")
        # The published heads carry their class names; otherwise they have
        # to be given in the options.
        names = options.get("class_names")
        if not names:
            names = list(getattr(self.module, "classes", []) or [])
        if not names:
            raise ValueError(
                "The model file names no classes - add \"class_names\" to "
                "the options")
        self.names = [str(name) for name in names]
        context.log(f"{context.task}: {os.path.basename(weights)}, "
                    f"{len(self.names)} classes")
        return {"input": "embeddings", "level": "frame",
                "class_names": self.names,
                "cameras": ["thermal", "rgb", "matched"]}

    def class_names(self):
        return self.names

    def run(self, batch, items):
        torch = self.torch
        with torch.no_grad():
            # the heads return (embedding, probabilities)
            _embedding, probabilities = self.module(torch.from_numpy(batch))
        return probabilities.detach().cpu().numpy()
