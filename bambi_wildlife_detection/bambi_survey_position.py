# -*- coding: utf-8 -*-
"""The Position Uncertainty step of the BAMBI processor (see
:mod:`core.position_uncertainty`).

Like the other survey analytics it works on a *survey* - the flights ticked
on the Survey Analytics tab, one camera, one output folder - and on what
the survey counts. Mixed into :class:`BambiProcessor`.
"""

import math
import os
from dataclasses import asdict
from typing import Any, Dict

from .core.survey import chosen_source

#: The terrain's slope is read once per square of this many metres.
TERRAIN_CELL_M = 5.0


class SurveyPositionMixin:
    """How far off the place of each detection may be."""

    def run_position_uncertainty(self, config: Dict[str, Any],
                                 progress_fn=None, log_fn=None,
                                 cancel_check=None):
        """Work out, for every detection the survey counts, how far its
        place on the map may be off - from how high the camera was, how
        far to the side it looked, the slope the animal stood on, and how
        well the camera's position and angle, the terrain model, the pixel
        and the timing are taken to be known.

        Writes into ``<survey>/analytics_{m}/``:

        * ``position_uncertainty_{source}.json`` - the accuracies used,
          the detections taken together (all, per flight) and - where
          animals are counted - one entry per animal;
        * ``position_uncertainty_{source}_boxes.csv`` - one row per
          detection;
        * ``position_uncertainty_{source}_tracks.csv`` - one row per
          animal (only where animals are counted).

        Settings (``config``): ``position_gps_horizontal_m``,
        ``position_gps_vertical_m``, ``position_camera_angle_deg``,
        ``position_terrain_m``, ``position_pixel_px``, ``position_sync_s``
        (:class:`core.position_uncertainty.Accuracy`).
        """
        from .bambi_processing import CancelledException
        from .core import (
            agl_profile, camera_pose, census, groups,
            position_uncertainty as uncertainty, poses, store as _store)
        from .core.exporters import common

        survey = self._survey(config)
        source = chosen_source(config, "position_source")
        try:
            accuracy = uncertainty.Accuracy.from_config(config)
        except uncertainty.UncertaintyError as exc:
            raise RuntimeError(str(exc)) from exc
        if log_fn:
            log_fn(f"Working out the position uncertainty "
                   f"({survey.camera_label}, {source}) of "
                   f"{self._flights_note(survey)}...")
        area = self._survey_area(survey, log_fn)
        if progress_fn:
            progress_fn(2)

        rows, animals = [], []
        left_out = 0
        level, without_pixel = [], []
        by_flight = {}
        for index, flight in enumerate(survey.flights):
            if cancel_check and cancel_check():
                raise CancelledException("Position uncertainty cancelled")
            folder, modality = flight.target_folder, survey.modality
            records = [record for record in self._census_records(
                survey, flight, source, area, log_fn)
                if census.counts_inside(record, area)]
            counted = {record["id"] for record in records}
            key = "track_id" if source == "tracks" else "detection_id"

            origin = agl_profile.dem_origin(
                (flight.dem_path or "").strip(), (flight.dem or "").strip(),
                folder)
            if origin is None:
                raise RuntimeError(
                    f"{flight.name}: the DEM origin is not known, so "
                    "nobody can say how high the camera was above an "
                    "animal. Set the DEM on the flight's Input tab.")
            cameras = {}
            images = poses.load_images(folder, modality)
            for frame, image in enumerate(images):
                location = (image or {}).get("location") or []
                if len(location) < 3:
                    continue
                try:
                    cameras[frame] = (float(location[0]) + origin[0],
                                      float(location[1]) + origin[1],
                                      float(location[2]) + origin[2])
                except (TypeError, ValueError):
                    continue

            boxes = []
            if os.path.isfile(_store.stage_path(folder, _store.DETECTIONS,
                                                modality)):
                for row in common.load_detections(
                        folder, modality, include_not_an_animal=True,
                        include_geo=True):
                    if row.get(key) not in counted:
                        continue
                    place = (None, None, None)
                    if row.get("gx1") is not None:
                        heights = (row.get("gz1") or 0.0,
                                   row.get("gz2") or 0.0)
                        place = ((row["gx1"] + row["gx2"]) / 2.0,
                                 (row["gy1"] + row["gy2"]) / 2.0,
                                 sum(heights) / 2.0)
                    boxes.append({
                        "detection_id": row["detection_id"],
                        "track_id": row.get("track_id"),
                        "frame": row["frame"],
                        "x": place[0], "y": place[1], "z": place[2],
                    })

            pixel = None
            size = poses.frame_size(folder, modality, images)
            if size is not None and images:
                pixel = uncertainty.pixel_angle(
                    camera_pose.pose_fovy(images[0]), size[1])
            if pixel is None:
                without_pixel.append(flight.name)
            gradient_of = self._terrain_gradients(
                flight, origin, [(box["x"], box["y"]) for box in boxes
                                 if box["x"] is not None], log_fn)
            if gradient_of is None:
                level.append(flight.name)
            of_flight, missing = uncertainty.box_rows(
                flight.name, boxes, cameras, accuracy, pixel, gradient_of,
                groups.frame_seconds(folder, modality))
            rows.extend(of_flight)
            left_out += missing
            by_flight[flight.name] = uncertainty.summarise(of_flight)
            if source == "tracks":
                animals.extend({
                    "flight": flight.name, "track_id": record["id"],
                    "species": record["species"], "x": record["x"],
                    "y": record["y"]} for record in records)
            if log_fn:
                note = f", {missing} without a place" if missing else ""
                log_fn(f"{flight.name}: {len(of_flight)} detection(s)"
                       f"{note}")
            if progress_fn:
                progress_fn(int(5 + 80 * (index + 1) / len(survey.flights)))

        if not rows:
            raise RuntimeError(
                "No detection of the survey has a place: run 'Geo-Reference "
                "Detections' (or 'Track Animals') first.")
        tracks = uncertainty.track_rows(rows, animals)
        document = {
            "source": source,
            "camera": survey.counted_label,
            "crs": f"EPSG:{survey.epsg}" if survey.epsg else None,
            "flights": [flight.name for flight in survey.flights],
            "survey_area": None if area is None else area.describe(),
            "species_filter": survey.species,
            "accuracy": asdict(accuracy),
            "summary": uncertainty.summarise(rows),
            "by_flight": by_flight if survey.several else {},
            "tracks": tracks,
            "left_out": left_out,
            "level_terrain": level,
            "notes": self._position_notes(level, without_pixel, left_out),
        }
        folder = survey.ensure_analytics_folder()
        stem = f"position_uncertainty_{source}"
        json_path = uncertainty.write_json(
            document, os.path.join(folder, f"{stem}.json"))
        uncertainty.write_table(
            rows, os.path.join(folder, f"{stem}_boxes.csv"))
        if source == "tracks":
            uncertainty.write_table(
                tracks, os.path.join(folder, f"{stem}_tracks.csv"),
                uncertainty.TRACK_COLUMNS)

        if log_fn:
            for line in uncertainty.describe(document):
                log_fn(line)
            log_fn(f"Position uncertainty written to {json_path}")
        if progress_fn:
            progress_fn(100)
        return document

    def _terrain_gradients(self, flight, origin, points, log_fn=None):
        """``gradient_of(x, y)`` - the terrain's rise per metre east and
        north around a place, read from the flight's DEM mesh once per
        square of :data:`TERRAIN_CELL_M`; ``None`` where the mesh cannot be
        read (the terrain then counts as level)."""
        from .core import agl_profile, inspection

        mesh_path = inspection.find_dem_mesh_path(
            (flight.dem_path or "").strip(),
            (flight.dem or "").strip() or None)
        if not mesh_path or not points:
            if log_fn and not mesh_path:
                log_fn(f"{flight.name}: no DEM mesh found - the terrain is "
                       "taken as level")
            return None
        cell = TERRAIN_CELL_M

        def square(x: float, y: float):
            return (int(math.floor((x - origin[0]) / cell)),
                    int(math.floor((y - origin[1]) / cell)))

        squares = sorted({square(x, y) for x, y in points})
        asked = []
        for column, row in squares:
            centre = ((column + 0.5) * cell, (row + 0.5) * cell)
            half = cell / 2.0
            asked += [(centre[0] + half, centre[1]),
                      (centre[0] - half, centre[1]),
                      (centre[0], centre[1] + half),
                      (centre[0], centre[1] - half)]
        errors = []
        heights = agl_profile.ground_heights(mesh_path, asked, errors)
        gradients = {}
        for index, key in enumerate(squares):
            east, west, north, south = heights[4 * index:4 * index + 4]
            if None in (east, west, north, south):
                continue
            gradients[key] = ((east - west) / cell, (north - south) / cell)
        if not gradients:
            if log_fn:
                why = f" ({errors[0]})" if errors else ""
                log_fn(f"{flight.name}: the DEM mesh gave no heights{why} - "
                       "the terrain is taken as level")
            return None

        def gradient_of(x: float, y: float):
            return gradients.get(square(x, y))

        return gradient_of

    @staticmethod
    def _position_notes(level, without_pixel, left_out: int) -> str:
        notes = (
            "A detection's place is where the line of sight through its "
            "pixel meets the terrain model. Its error is put together from "
            "what goes into that - the camera's position and height, the "
            "terrain model's height, the camera's angle, the pixel the box "
            "sits on, and the frame's timing against the camera's position "
            "- each with the accuracy set (one sigma), by the geometry of "
            "the one detection: how high the camera was above the animal, "
            "how far to the side it looked, and whether the ground rises or "
            "falls away from the camera there. 'Typically' is the "
            "root-mean-square distance between the true place and the "
            "mapped one - about two detections in three lie within it; "
            "twice that holds 95 to 98 in a hundred. An animal is given "
            "the median of its boxes: what they have in common - where the "
            "drone took itself to be, a terrain model that is off - does "
            "not average out. The accuracies are the ones set, not "
            "measured: the result says what follows from them, and which "
            "of them matters most. A constant offset that the Correction "
            "Wizard removes is not part of it. ")
        if level:
            named = ", ".join(level)
            notes += (f"The terrain was taken as level for {named} (no DEM "
                      "mesh to read the slope from): on slopes that fall "
                      "away from the camera the error is larger than "
                      "given. ")
        if without_pixel:
            named = ", ".join(without_pixel)
            notes += (f"The size of a pixel is not known for {named} (no "
                      "frame on disk): the pixel is left out there. ")
        if left_out:
            notes += (f"{left_out} detection(s) have no place, or one "
                      "nobody can judge (no camera position for the frame, "
                      "a camera below the animal): left out. ")
        return notes.strip()
