# -*- coding: utf-8 -*-
"""Sample importer: detections or tracks from a CSV file - to copy and
change for whatever your other tool writes.

How a custom importer works
---------------------------
* One .py file with a class that has ``setup(context)`` and
  ``run(source, frames)``. Nothing is imported from the plugin.
* ``run(source, frames)`` reads the file or folder chosen as "Source" and
  returns one row per box, as a dict::

      {"frame": 12,                  index of the extracted frame
       "x1": 100, "y1": 80, "x2": 140, "y2": 130,     the box in pixels
       "confidence": 0.9,            optional, default 1
       "class": "roe deer",          optional, default "animal"
       "track_id": 3}                optional - see below

  or as a tuple ``(frame, x1, y1, x2, y2, confidence, class, track_id)``,
  the last three optional.
* **Tracks or detections.** Rows with a ``track_id`` are imported as
  tracks: they become the flight's tracks, as a tracker's would. Without
  any, only the boxes are imported, and "Track Animals" can run on them.
* **Frames.** ``frames`` lists the extracted frames (``index``, ``path``,
  ``timestamp``, ``pose``), for a tool that names its frames by file or by
  time rather than by number. A box on a frame that was not extracted is
  left out.
* **Pixels.** By default the boxes are in the pixels of the extracted
  frames. A tool that worked on the raw video returns
  ``{"space": "raw"}`` from ``setup`` and
  ``{"detections": rows, "raw_size": (width, height)}`` from ``run``; the
  plugin then undistorts the boxes with the camera's calibration.
* **Species.** A class named as a species of the project is that species;
  any other class is "animal" until it is mapped (Edit Class Mapping).
* An import replaces what an earlier import brought in. The detector's own
  boxes stay.

This sample reads a CSV file with a header line. The columns it knows:

    frame, x1, y1, x2, y2            required
    confidence, class, track_id      optional
    (or x, y, w, h instead of x1, y1, x2, y2: top left corner and size)

``frame`` may also be the file name of an extracted frame.

Options (JSON in the configuration):
    {"delimiter": ";",                (default: guessed from the file)
     "space": "raw",                  boxes are on the raw video ...
     "raw_size": [1920, 1080]}        ... which is this large
"""

import csv
import os

BAMBI_API = 1

#: Packages this file needs (import names) - none beyond Python itself.
REQUIRES = []


class CsvImporter:
    BAMBI_KIND = "importer"

    def setup(self, context):
        self.options = dict(context.options)
        self.log = context.log
        return {"space": self.options.get("space", "frames")}

    def run(self, source, frames):
        if not os.path.isfile(source):
            raise ValueError(f"Choose a CSV file as the source, not: {source}")
        by_name = {os.path.basename(frame.path): frame.index
                   for frame in frames if frame.path}

        with open(source, newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = self.options.get("delimiter") or (
                ";" if sample.count(";") > sample.count(",") else ",")
            rows = []
            for line in csv.DictReader(handle, delimiter=delimiter):
                line = {(key or "").strip().lower(): (value or "").strip()
                        for key, value in line.items()}
                frame = line.get("frame", "")
                frame = by_name[frame] if frame in by_name else int(
                    float(frame))
                if line.get("x1"):
                    box = [float(line[key]) for key in
                           ("x1", "y1", "x2", "y2")]
                else:
                    x, y = float(line["x"]), float(line["y"])
                    box = [x, y, x + float(line["w"]), y + float(line["h"])]
                rows.append({
                    "frame": frame,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "confidence": float(line.get("confidence") or 1.0),
                    "class": line.get("class") or "animal",
                    "track_id": (int(float(line["track_id"]))
                                 if line.get("track_id") else None)})
        self.log(f"{len(rows)} row(s) in {os.path.basename(source)}")

        if self.options.get("space") == "raw":
            return {"detections": rows,
                    "raw_size": tuple(self.options["raw_size"])}
        return rows

    def teardown(self):
        pass
