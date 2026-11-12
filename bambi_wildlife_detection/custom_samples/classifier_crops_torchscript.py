# -*- coding: utf-8 -*-
"""Sample classifier on image crops: a TorchScript image classifier that is
given the cut-out bounding boxes directly. Step C1 (embeddings) is not
needed for it.

See classifier_embeddings_head.py for the interface in general. What is
different for ``"input": "crops"``:

* ``batch`` is a list of N crops, each an RGB uint8 array of
  ``crop_size`` x ``crop_size`` x 3 (return "crop_size" from ``setup``;
  default 224). The crops are cut the way step C1 cuts them - with the
  padding and letterboxing of the Classification tab, from the imagery
  chosen as "Projection" there: the perspective camera frames, the
  orthorectified per-frame GeoTIFFs or the ALFS light-field integrals.
  ``context.projection`` says which ("perspective" / "orthographic" /
  "alfs"), so a classifier trained on one kind can refuse the others in
  ``setup``.
* For the "matched" input each element is a ``(thermal, rgb)`` pair of such
  crops; with the "unmatched" setting on "use RGB / thermal" the missing
  side is None. This sample supports single cameras only, so it declares
  ``"cameras": ["thermal", "rgb"]`` and the plugin refuses to run it on
  "matched".
* Frozen thermal frames and missing images give no crop; those rows are
  simply not passed.

Options (JSON in the configuration):
    {"weights": "C:/models/species_resnet.pt",     (required, TorchScript:
                                                    float (N, 3, H, W) in,
                                                    logits (N, classes) out)
     "class_names": ["roe deer", "red deer"],      (required)
     "crop_size": 224,
     "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
     "batch_size": 32}
"""

import os

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = ["torch", "numpy"]


class CropClassifier:
    BAMBI_KIND = "classifier"

    def setup(self, context):
        import numpy as np
        import torch

        options = context.options
        weights = options.get("weights", "")
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Classifier not found: {weights!r} - set \"weights\" in the "
                "options")
        self.names = [str(n) for n in options.get("class_names") or []]
        if not self.names:
            raise ValueError("Add \"class_names\" to the options")
        self.torch = torch
        self.device = context.device
        self.module = torch.jit.load(weights, map_location=self.device).eval()
        self.mean = np.float32(options.get("mean", [0.485, 0.456, 0.406]))
        self.std = np.float32(options.get("std", [0.229, 0.224, 0.225]))
        self.batch_size = int(options.get("batch_size", 32))
        return {"input": "crops", "level": "frame",
                "class_names": self.names,
                "crop_size": int(options.get("crop_size", 224)),
                "cameras": ["thermal", "rgb"]}

    def class_names(self):
        return self.names

    def _probabilities(self, crops):
        """(N, classes) probabilities of a list of RGB uint8 crops."""
        import numpy as np

        torch = self.torch
        out = []
        for start in range(0, len(crops), self.batch_size):
            chunk = np.stack(crops[start:start + self.batch_size])
            chunk = (chunk.astype(np.float32) / 255.0 - self.mean) / self.std
            tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                logits = self.module(tensor.to(self.device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out)

    def run(self, batch, items):
        return self._probabilities(batch)
