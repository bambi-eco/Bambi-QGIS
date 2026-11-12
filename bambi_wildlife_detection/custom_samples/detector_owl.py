# -*- coding: utf-8 -*-
"""Sample detector: Microsoft's Overhead Wildlife Locator (OWL), published
as "MegaDetector-Overhead" (https://github.com/microsoft/MegaDetector-Overhead).

NOT TESTED by the plugin authors against the real model - this file follows
the repository's own inference code (animaloc/utils/owl_demo.py) and is
meant as a starting point. Check the results on your own data.

What to know about OWL
----------------------
* It locates animals as POINTS (peaks of a density heat map), not boxes.
  The plugin works with boxes, so a square of ``box_size`` pixels is put
  around each point. Choose it from the animal size in your imagery
  (animal length in metres / ground sampling distance in metres).
* One generic class, "animal". Scores are not calibrated probabilities.
* It was built for RGB imagery, so only "rgb" is declared as supported.
* Licence: code MIT, the published weights CC BY-NC-SA 4.0 (non-commercial).

Setup (your responsibility - the plugin does not manage this)
------------------------------------------------------------
1. Clone the repository and install its requirements into QGIS' Python
   (the OSGeo4W Shell on Windows, or a terminal with the Python QGIS uses
   on Linux / macOS):
       git clone https://github.com/microsoft/MegaDetector-Overhead
       python -m pip install -r <its requirements, see the repo's README>
   IMPORTANT - installing packages can break other things: QGIS, the plugin
   and all other plugins share one Python, and pip replaces packages they rely
   on (torch, torchvision, transformers, numpy, opencv) if a new package asks
   for other versions. Install only what is missing, avoid --upgrade unless an
   error asks for it, check first with ``python -m pip install --dry-run ...``
   and look at the plugin's Dependency Manager afterwards.
2. Download a checkpoint from https://zenodo.org/records/20802844
   (OWL-C.pth runs on a CPU; OWL-T.pth and OWL-D.pth are larger).
3. Options in the configuration (JSON):
       {"repo": "C:/code/MegaDetector-Overhead",
        "weights": "C:/models/OWL-C.pth",
        "model": "OWLC",          (OWLC | OWLT | OWLD_H, matching the file)
        "box_size": 32,           (pixels, side of the box around a point)
        "score_threshold": 0.2}
"""

import os
import sys

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = ["torch", "cv2", "numpy"]


class OwlDetector:
    BAMBI_KIND = "detector"

    def setup(self, context):
        import torch

        options = context.options
        repo = options.get("repo", "")
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)        # makes ``animaloc`` importable
        from animaloc.eval.lmds import HerdNet_Detection_Branch_LMDS
        from animaloc.eval.stitchers import HerdNet_Detection_Branch_Stitcher
        from animaloc.models import MODELS, LossWrapper

        weights = options.get("weights", "")
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"OWL checkpoint not found: {weights!r} - set \"weights\" in "
                "the options")
        variant = options.get("model", "OWLC")
        kwargs = {"pretrained": False, "head_conv": 64, "down_ratio": 2}
        if variant.startswith("OWLD"):
            kwargs.update(readout_type="ignore", freeze_backbone=True)

        self.torch = torch
        self.device = context.device if context.device != "mps" else "cpu"
        checkpoint = torch.load(weights, map_location="cpu",
                                weights_only=True)
        model = LossWrapper(MODELS[variant](**kwargs), [])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().requires_grad_(False).to(self.device)

        self.stitcher = HerdNet_Detection_Branch_Stitcher(
            model, size=(512, 512), overlap=160, batch_size=1, down_ratio=2,
            up=False, reduction="mean", device_name=self.device)
        self.lmds = HerdNet_Detection_Branch_LMDS(
            up=False, kernel_size=(3, 3), adapt_ts=0.3, neg_ts=0.1)
        self.half = float(options.get("box_size", 32)) / 2.0
        self.threshold = float(options.get(
            "score_threshold", max(0.2, context.min_confidence)))
        context.log(f"OWL {variant} on {self.device}; points become "
                    f"{2 * self.half:.0f} px boxes")
        return {"class_names": ["animal"], "cameras": ["rgb"]}

    def run(self, image, frame):
        import cv2
        import numpy as np

        torch = self.torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.float32([0.485, 0.456, 0.406])) \
            / np.float32([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        with torch.inference_mode():
            heatmap = self.stitcher(tensor).cpu()   # (1, 1, H // 2, W // 2)
            _counts, locations, _labels, scores = self.lmds(heatmap)

        detections = []
        for (y, x), score in zip(locations[0], scores[0]):
            if float(score) < self.threshold:
                continue
            x, y = float(x) * 2.0, float(y) * 2.0   # back to image pixels
            detections.append((x - self.half, y - self.half,
                               x + self.half, y + self.half,
                               min(1.0, float(score)), 0))
        return detections
