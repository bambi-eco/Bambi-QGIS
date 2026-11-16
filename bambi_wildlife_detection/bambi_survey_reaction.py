# -*- coding: utf-8 -*-
"""The Reaction to the Drone step of the BAMBI processor (see
:mod:`core.drone_reaction`).

Like the other survey analytics it works on a *survey* - the flights ticked
on the Survey Analytics tab, one camera, one output folder - and on its
tracked animals. Mixed into :class:`BambiProcessor`.
"""

import os
from typing import Any, Dict

from .core import quiet


class SurveyReactionMixin:
    """What the tracked animals did while the drone was over them."""

    def run_drone_reaction(self, config: Dict[str, Any], progress_fn=None,
                           log_fn=None, cancel_check=None):
        """Read from every track of the survey how far the animal went, how
        fast, and where to relative to the line the drone flew.

        Writes into ``<survey>/analytics_{m}/``:

        * ``drone_reaction.json`` - the settings, what the animals did
          taken together (all, per species, per flight), and every track;
        * ``drone_reaction_tracks.csv`` - one row per tracked animal.

        Settings (``config``): ``reaction_min_displacement_m`` - how far an
        animal has to have gone while in view to count as having moved
        (default 3 m); ``reaction_min_speed_ms`` - how fast it has to have
        gone at the least (default 0: at any speed).
        """
        from .bambi_processing import CancelledException
        from .core import census, drone_reaction, groups, store as _store
        from .core.exporters import common

        survey = self._survey(config)
        try:
            minimum = float(config.get(
                "reaction_min_displacement_m",
                drone_reaction.MIN_DISPLACEMENT_M))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "The distance an animal has to have gone is a number of "
                "metres.") from exc
        if minimum <= 0:
            raise RuntimeError(
                "The distance an animal has to have gone has to be more "
                "than nothing.")
        try:
            slowest = float(config.get("reaction_min_speed_ms",
                                       drone_reaction.MIN_SPEED_MS))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "The speed an animal has to have had is a number of metres "
                "per second.") from exc
        if slowest < 0:
            raise RuntimeError(
                "The speed an animal has to have had cannot be less than "
                "nothing.")
        if log_fn:
            log_fn(f"Reading the reaction to the drone ({survey.camera_label}"
                   f" tracks) of {self._flights_note(survey)}...")
        area = self._survey_area(survey, log_fn)
        if progress_fn:
            progress_fn(2)

        rows = []
        without_line = []
        for index, flight in enumerate(survey.flights):
            if cancel_check and cancel_check():
                raise CancelledException("Reaction to the drone cancelled")
            folder, modality = flight.target_folder, survey.modality
            records = self._census_records(survey, flight, "tracks", area,
                                           log_fn)
            animals = [
                {"track_id": record["id"], "species": record["species"],
                 "sex": record["sex"], "age": record["age"]}
                for record in records
                if census.counts_inside(record, area)]

            seconds = groups.frame_seconds(folder, modality)
            if animals and not seconds:
                raise RuntimeError(
                    f"{flight.name}: the frames carry no capture time, so "
                    "nobody can say how fast anything moved. Extract the "
                    "frames again with the flight log.")

            positions = {}
            if os.path.isfile(_store.stage_path(folder, _store.DETECTIONS,
                                                modality)):
                for row in common.load_detections(
                        folder, modality, include_not_an_animal=True,
                        include_geo=True):
                    needed = (row.get("track_id"), row.get("gx1"),
                              row.get("gx2"))
                    if None in needed:
                        continue
                    positions.setdefault(int(row["track_id"]), []).append((
                        int(row["frame"]),
                        (row["gx1"] + row["gx2"]) / 2.0,
                        (row["gy1"] + row["gy2"]) / 2.0))

            drone = self._drone_positions(flight, survey, log_fn)
            if drone is None:
                without_line.append(flight.name)
            of_flight = drone_reaction.flight_rows(
                flight.name, animals, positions, seconds, drone,
                min_displacement=minimum, min_speed=slowest)
            rows.extend(of_flight)
            if log_fn:
                moved = sum(1 for row in of_flight
                            if row["status"] == drone_reaction.MOVED)
                judged = sum(1 for row in of_flight
                             if row["status"] != drone_reaction.NOT_JUDGED)
                log_fn(f"{flight.name}: {len(of_flight)} animal(s), "
                       f"{judged} in view long enough to tell, {moved} "
                       "moved")
            if progress_fn:
                progress_fn(int(5 + 80 * (index + 1) / len(survey.flights)))

        if not rows:
            raise RuntimeError(
                "The survey has no tracked animals to read a reaction "
                "from.")
        summary = drone_reaction.summarise(rows, survey.species_names())
        by_flight = {}
        if survey.several:
            for flight in survey.flights:
                by_flight[flight.name] = drone_reaction.summarise(
                    [row for row in rows if row["flight"] == flight.name]
                )[drone_reaction.ALL]

        document = {
            "source": "tracks",
            "camera": survey.camera_label,
            "crs": f"EPSG:{survey.epsg}" if survey.epsg else None,
            "flights": [flight.name for flight in survey.flights],
            "survey_area": None if area is None else area.describe(),
            "species_filter": survey.species,
            "settings": {
                "min_displacement_m": minimum,
                "min_speed_ms": slowest,
                "min_duration_s": drone_reaction.MIN_DURATION_S,
                "min_positions": drone_reaction.MIN_POSITIONS,
                "sigmas": drone_reaction.SIGMAS,
            },
            "summary": summary,
            "by_flight": by_flight,
            "tracks": rows,
            "notes": self._reaction_notes(minimum, without_line, slowest),
        }
        folder = survey.ensure_analytics_folder()
        json_path = drone_reaction.write_json(
            document, os.path.join(folder, "drone_reaction.json"))
        drone_reaction.write_table(
            rows, os.path.join(folder, "drone_reaction_tracks.csv"))

        if log_fn:
            for line in drone_reaction.describe(document):
                log_fn(line)
            log_fn(f"Reaction to the drone written to {json_path}")
        if progress_fn:
            progress_fn(100)
        return document

    def _drone_positions(self, flight, survey, log_fn=None):
        """``frame -> (x, y)`` of the camera in the project's coordinates,
        or ``None`` where the flight's DEM - by whose origin the poses are
        shifted - is not known."""
        from .core.exporters import common

        try:
            x_offset, y_offset, _source = self._resolve_population_origin(
                flight)
        except FileNotFoundError as exc:
            quiet.problem(f"The DEM of {flight.name} is not known: its "
                          "animals are read without the line the drone flew")
            if log_fn:
                log_fn(f"{flight.name}: {exc} Without it the line the "
                       "drone flew is not known: how far and how fast the "
                       "animals moved is read, where to is not.")
            return None
        positions = {}
        for frame in common.load_frames(flight.target_folder,
                                        survey.modality):
            location = frame.get("location") or []
            if len(location) < 2:
                continue
            try:
                positions[int(frame["frame"])] = (
                    float(location[0]) + x_offset,
                    float(location[1]) + y_offset)
            except (TypeError, ValueError):
                continue
        return positions or None

    @staticmethod
    def _reaction_notes(minimum: float, without_line,
                        slowest: float = 0.0) -> str:
        from .core import drone_reaction

        pace = f", at {slowest:g} m/s or faster," if slowest > 0 else ""
        notes = (
            "A constant velocity is fitted through all positions of a "
            "track; what the positions lie off it by is their jitter, and "
            "says how well the velocity is known. An animal moved when "
            f"that velocity carried it at least {minimum:g} m{pace} during "
            f"its time in view and {drone_reaction.SIGMAS:g} times the standard "
            "error of that distance. Tracks of fewer than "
            f"{drone_reaction.MIN_POSITIONS} positions or less than "
            f"{drone_reaction.MIN_DURATION_S:g} s are not judged. The line "
            "is the direction the drone flew while the animal was in view; "
            "'away' is to the side of it the animal was on. The part of a "
            "movement across the line is the one to trust: a warm spot "
            "that is not at the height the terrain model has for it seems "
            "to slide along the line as the camera passes, never across. "
            "Without a reaction as many animals go towards the line as "
            "away from it (exact binomial test), and the mean speed away "
            "from the line is nought (its interval is Student's, over all "
            "animals judged). Animals that move away from the line make "
            "the perpendicular distances of distance sampling too large "
            "and its density too low. ")
        if without_line:
            notes += (
                "The DEM of " + ", ".join(without_line) + " is not known, "
                "so the line the drone flew there is not either: those "
                "animals count for how many moved, not for where to. ")
        return notes.strip()
