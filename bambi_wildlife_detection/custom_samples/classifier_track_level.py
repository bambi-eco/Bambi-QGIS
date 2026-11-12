# -*- coding: utf-8 -*-
"""Sample track-level classifier: decides once per animal instead of once
per frame. It needs no package beyond numpy and no model file - it calls an
animal "large" or "small" from the area of its boxes - so it runs as it is
and shows the track-level interface in a few lines.

See classifier_embeddings_head.py for the interface in general. What is
different for ``"level": "track"``:

* The class needs ``run_track(batch, items)`` instead of ``run``.
* It is called once per animal (track) with the frames the plugin selected
  for it - the frames the occlusion step left as usable, or all of them.
  ``batch`` holds those frames' inputs (an (N, D) array for
  ``"input": "embeddings"``, a list of N crops for ``"input": "crops"``),
  ``items`` one dict per frame with ``detection_id``, ``track_id`` and
  ``frame``.
* It returns ``(class_index, probability)`` for the animal, or None for
  "no call" - the animal then stays undecided, exactly as when a vote
  fails.
* No per-frame results are stored and the plugin takes no vote: how the
  frames are combined is this class's business (average the probabilities,
  feed a sequence model, pick the sharpest frame, ...).
* Occlusion is a per-frame question, so a track-level classifier cannot be
  used for it.

Options (JSON in the configuration), all optional:
    {"min_frames": 3, "bright_fraction": 0.15}
"""

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = ["numpy"]


class TrackLevelExample:
    BAMBI_KIND = "classifier"

    def setup(self, context):
        options = context.options
        self.min_frames = int(options.get("min_frames", 3))
        self.fraction = float(options.get("bright_fraction", 0.15))
        return {"input": "crops", "level": "track",
                "class_names": ["small", "large"], "crop_size": 96,
                "cameras": ["thermal", "rgb"]}

    def class_names(self):
        return ["small", "large"]

    def run_track(self, batch, items):
        import numpy as np

        if len(batch) < self.min_frames:
            return None                      # too little evidence: no call
        # Share of each crop on the bright side of its own brightness range
        # - a stand-in for "how much of the crop the animal fills".
        shares = []
        for crop in batch:
            grey = crop.astype(np.float32).mean(axis=2)
            middle = (float(grey.min()) + float(grey.max())) / 2.0
            shares.append(float((grey > middle).mean()))
        share = float(np.median(shares))
        index = 1 if share >= self.fraction else 0
        # how consistently the frames agree with the call
        agree = float(np.mean([(s >= self.fraction) == bool(index)
                               for s in shares]))
        return index, agree
