# -*- coding: utf-8 -*-
"""Sample exporter: detections, tracks and segmentation masks as plain
CSV / JSON files - to copy and change into the format you need.

How a custom exporter works
---------------------------
* One .py file with a class that has ``setup(context)`` and
  ``run(data, output)``. Nothing is imported from the plugin.
* Choose it under Processing - Export - Custom exporter and pick the
  format "Custom exporter..." - there for one flight, or in Survey
  Analytics - Export for several flights with train / validation / test
  splits (the exporter is then run once per flight, into
  ``<dataset>/<split>/<flight>``).
* ``output`` is a folder that exists and is yours to fill.
* ``data`` is a dict of plain Python - species and enum values are names,
  never ids:

      data["flight"]      {"name", "folder", "camera", "crs"}
      data["frames"]      [{"index", "file", "path", "timestamp",
                            "duplicated"}]
      data["frame_size"]  (width, height), or None when not known
      data["species"]     the species names that occur
      data["detections"]  [{"detection_id", "frame",
                            "box": (x1, y1, x2, y2),          pixels
                            "confidence", "species", "track_id",
                            "attributes": {...},
                            "world": (x1, y1, z1, x2, y2, z2) or None,
                            "lonlat": (lon, lat) or None}]
      data["tracks"]      [{"track_id", "species", "attributes": {...},
                            "detection_ids", "first_frame", "last_frame"}]
      data["segmentations"]  [{"segment_id", "detection_id", "frame",
                               "polygon": [ring, ...] or None    pixels
                               "world":   [ring, ...] or None    project CRS
                               "lonlat":  [ring, ...] or None    WGS84
                               "attributes": {...}}]
      data["options"]     {"include_images", "include_not_an_animal"}

  A ring is a list of ``[x, y]`` points. ``world`` and ``lonlat`` are
  there for what has been geo-referenced; ``lonlat`` needs the target CRS.
* What is in ``data`` follows the export's settings: detections labelled
  "not-an-animal" only when asked for, and in a survey that counts "only
  animals confirmed by the other camera" only those.
* ``data["options"]["include_images"]`` says whether the user wants the
  frames with the export; copying them (from ``frame["path"]``) is the
  exporter's own business.

This sample writes ``detections.csv``, ``tracks.csv`` and - when the
flight has masks - ``segmentations.json``.

Options (JSON in the configuration):
    {"delimiter": ";"}        (default ",")
"""

import csv
import json
import os
import shutil

BAMBI_API = 1

#: Packages this file needs (import names) - none beyond Python itself.
REQUIRES = []


class CsvExporter:
    BAMBI_KIND = "exporter"

    def setup(self, context):
        self.delimiter = context.options.get("delimiter", ",")
        self.log = context.log
        return {"name": "CSV"}

    def run(self, data, output):
        frames = {frame["index"]: frame for frame in data["frames"]}

        with open(os.path.join(output, "detections.csv"), "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=self.delimiter)
            writer.writerow(["detection_id", "frame", "image", "x1", "y1",
                             "x2", "y2", "confidence", "species",
                             "track_id", "lon", "lat"])
            for detection in data["detections"]:
                lon, lat = detection["lonlat"] or ("", "")
                frame = frames.get(detection["frame"], {})
                writer.writerow([
                    detection["detection_id"], detection["frame"],
                    frame.get("file", ""), *detection["box"],
                    detection["confidence"], detection["species"],
                    "" if detection["track_id"] is None
                    else detection["track_id"], lon, lat])

        with open(os.path.join(output, "tracks.csv"), "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=self.delimiter)
            writer.writerow(["track_id", "species", "sex", "age", "boxes",
                             "first_frame", "last_frame"])
            for track in data["tracks"]:
                writer.writerow([
                    track["track_id"], track["species"],
                    track["attributes"].get("sex", ""),
                    track["attributes"].get("age", ""),
                    len(track["detection_ids"]),
                    track["first_frame"], track["last_frame"]])

        if data["segmentations"]:
            with open(os.path.join(output, "segmentations.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"flight": data["flight"]["name"],
                           "crs": data["flight"]["crs"],
                           "segmentations": data["segmentations"]},
                          handle, indent=2)

        if data["options"]["include_images"]:
            images = os.path.join(output, "images")
            os.makedirs(images, exist_ok=True)
            used = {detection["frame"] for detection in data["detections"]}
            for index in sorted(used):
                path = frames.get(index, {}).get("path", "")
                if os.path.isfile(path):
                    shutil.copyfile(path, os.path.join(
                        images, os.path.basename(path)))

        self.log(f"{len(data['detections'])} detection(s), "
                 f"{len(data['tracks'])} track(s), "
                 f"{len(data['segmentations'])} segmentation(s)")

    def teardown(self):
        pass
