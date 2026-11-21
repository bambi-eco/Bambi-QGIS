# -*- coding: utf-8 -*-
"""Sample embedder: a colour histogram of each crop - the smallest thing
that is an embedding model, to copy and replace by a real one.

How a custom embedder works
---------------------------
* One .py file with a class that has ``setup(context)`` and
  ``run(crops, items)``. Nothing is imported from the plugin.
* Step C1 ("Compute Embeddings") cuts every tracked animal out of its
  frame - crop size, padding, letterbox and projection as the
  Classification tab says - and hands the crops to ``run`` in batches.
* ``run(crops, items)`` returns one vector per crop, in order: an (N, D)
  array (NumPy, or a torch tensor). All vectors have the same width D and
  no NaN. ``crops`` is a list of H x W x 3 RGB uint8 arrays; ``items``
  tells which crop is what: ``detection_id``, ``track_id``, ``frame`` and
  ``box`` (pixels).
* ``setup(context)`` may return
      "dim":         D, the width of the vectors (otherwise the plugin
                     finds out with one black crop)
      "batch_size":  how many crops ``run`` is handed at most (default:
                     the tab's batch size)
      "cameras":     any of "thermal", "rgb"                [default: both]
  ``context`` carries ``camera``, ``device``, ``model_dir``, ``options``
  and ``log(message)``.
* The vectors are stored where DINOv3's are and recorded under this file's
  name and hash: change the file and the next run starts afresh instead of
  mixing two models' vectors.
* The published classifiers read DINOv3 vectors only. With a custom
  embedder every classifier has to be one trained on its vectors - a
  TorchScript file, or a custom classifier with ``"input": "embeddings"``
  (see classifier_embeddings_head.py).

A real model looks the same. With a torchvision ResNet, for example::

    def setup(self, context):
        import torch, torchvision
        self.torch = torch
        self.device = context.device
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.prepare = weights.transforms()
        return {"dim": 2048}

    def run(self, crops, items):
        torch = self.torch
        batch = torch.stack([
            self.prepare(torch.from_numpy(crop).permute(2, 0, 1))
            for crop in crops]).to(self.device)
        with torch.no_grad():
            return self.model(batch).float().cpu().numpy()

Options (JSON in the configuration):
    {"bins": 8}        bins per colour channel (default 8, so D = 24)
"""

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = ["numpy"]


class ColourHistogramEmbedder:
    BAMBI_KIND = "embedder"

    def setup(self, context):
        self.bins = int(context.options.get("bins", 8))
        if self.bins < 2:
            raise ValueError("bins must be at least 2")
        context.log(f"Colour histogram with {self.bins} bins per channel")
        return {"dim": 3 * self.bins}

    def run(self, crops, items):
        import numpy as np

        vectors = np.zeros((len(crops), 3 * self.bins), dtype=np.float32)
        for row, crop in enumerate(crops):
            pixels = max(1, crop.shape[0] * crop.shape[1])
            for channel in range(3):
                counts, _edges = np.histogram(
                    crop[..., channel], bins=self.bins, range=(0, 256))
                start = channel * self.bins
                vectors[row, start:start + self.bins] = counts / pixels
        return vectors

    def teardown(self):
        pass
