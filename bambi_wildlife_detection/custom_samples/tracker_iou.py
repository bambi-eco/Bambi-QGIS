# -*- coding: utf-8 -*-
"""Sample tracker: a small greedy IoU tracker in image space. It needs no
package beyond Python itself, so it is the easiest one to start from.

How a custom tracker works
--------------------------
* One .py file with a class that has ``setup(context)`` and ``update``
  (or ``run``, see below). Nothing is imported from the plugin.
* ``setup`` may return:
      "space"        "geo" (default) - only geo-referenced detections are
                     passed, so "Geo-Reference Detections" has to have run -
                     or "pixel" - all of them
      "interpolate"  False when filling the gaps of a track with
                     interpolated boxes makes no sense for the tracker (it
                     fills them itself, or it re-identifies animals after
                     long absences). Where it applies (the default) the
                     "Interpolate gaps in the tracks" tick box decides.
      "images"       True to get the frame image as ``frame.image`` (BGR)
                     in ``update``, e.g. for appearance features
      "cameras"      which of "thermal" / "rgb" it supports (default both)

* **Frame by frame (this sample, recommended):**
  ``update(frame, detections)`` is called once per frame, in frame order,
  with that frame's detections - also for frames without any, so tracks
  can age. It returns ``{detection_id: track_id}`` for them. The plugin
  owns the loop: it moves the progress bar, can be cancelled between
  frames and loads the image if asked to.
  An optional ``finish()`` may return a final ``{detection_id: track_id}``
  that overrides earlier answers (for trackers that revise decisions).

* **Whole flight at once:** a class with ``run(detections, frames)``
  instead of ``update`` gets everything in one call and returns
  ``{detection_id: track_id}`` for all of it - for trackers that need to
  see the whole flight (global association). It should then report its own
  progress with ``context.progress(fraction)`` (0.0 - 1.0).

* A detection is a dict:
      detection_id   the id to give back
      frame          frame index
      x1 y1 x2 y2    box in image pixels
      confidence
      class          the detector's class name
      world          (x1, y1, z1, x2, y2, z2) in world coordinates, or None
  ``frame`` has ``.index``, ``.path``, ``.timestamp``, ``.duplicated`` and
  ``.pose`` - where the camera was (``pose.position`` / ``pose.world``,
  ``pose.matrix``, ``pose.fovy``, correction factors applied).
* A detection left out (or mapped to None / a negative id) stays untracked.

The plugin records the run and does everything downstream.

Options (JSON in the configuration), all optional:
    {"iou": 0.3, "max_age": 5, "min_length": 1}
"""

BAMBI_API = 1

#: Packages this file needs (import names). "Check" looks them up before
#: any of this code runs and names the missing ones.
REQUIRES = []


class IouTracker:
    BAMBI_KIND = "tracker"

    def setup(self, context):
        options = context.options
        self.min_iou = float(options.get("iou", 0.3))
        self.max_age = int(options.get("max_age", 5))
        self.min_length = int(options.get("min_length", 1))
        self.active = []         # [track_id, last box, last frame]
        self.assignment = {}     # detection_id -> track_id, all frames
        self.next_id = 1
        # short gaps (up to max_age frames) are bridged by this tracker, and
        # a straight line across them is a fair guess: interpolation applies
        return {"space": "pixel", "cameras": ["thermal", "rgb"],
                "interpolate": True}

    def update(self, frame, detections):
        index = frame.index
        self.active = [t for t in self.active
                       if index - t[2] <= self.max_age]
        candidates = []
        for track in self.active:
            for detection in detections:
                overlap = _iou(track[1], _box(detection))
                if overlap >= self.min_iou:
                    candidates.append((overlap, id(track), track, detection))

        result = {}
        taken_tracks = set()
        # best overlaps first; each track and detection used once
        for _overlap, key, track, detection in sorted(
                candidates, key=lambda c: -c[0]):
            if key in taken_tracks or detection["detection_id"] in result:
                continue
            taken_tracks.add(key)
            result[detection["detection_id"]] = track[0]
            track[1], track[2] = _box(detection), index
        for detection in detections:
            if detection["detection_id"] in result:
                continue
            self.active.append([self.next_id, _box(detection), index])
            result[detection["detection_id"]] = self.next_id
            self.next_id += 1

        self.assignment.update(result)
        return result

    def finish(self):
        """The final word: tracks shorter than ``min_length`` are dropped
        (``None`` = untracked), which is only known once all frames are in."""
        if self.min_length <= 1:
            return None
        sizes = {}
        for track_id in self.assignment.values():
            sizes[track_id] = sizes.get(track_id, 0) + 1
        return {detection_id: (track_id if sizes[track_id] >= self.min_length
                               else None)
                for detection_id, track_id in self.assignment.items()}


def _box(detection):
    return (detection["x1"], detection["y1"],
            detection["x2"], detection["y2"])


def _iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0
