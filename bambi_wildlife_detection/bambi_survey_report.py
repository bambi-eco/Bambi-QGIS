# -*- coding: utf-8 -*-
"""The Survey Report step of the BAMBI processor (see
:mod:`core.survey_report`).

It gathers what a page about a survey needs that is not in a result file -
when each flight was flown and in what weather, where the drone went, where
the animals were, what made the data - reads the results the survey's
analytics wrote, and writes the one HTML file. Mixed into
:class:`BambiProcessor`.
"""

import base64
import json
import math
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List

from .core import quiet
from .core.survey import chosen_source

#: A route is drawn with at most this many points, and a map with at most
#: this many animals: a page is to open at once.
MAX_ROUTE_POINTS = 400
MAX_MAP_ANIMALS = 4000
#: A picture of the map larger than this is left out of the page.
MAX_CANVAS_BYTES = 6 * 1024 * 1024
#: Outlines are simplified to this many metres.
OUTLINE_TOLERANCE_M = 1.0


def _rings(geometry) -> List[List[List[float]]]:
    """The outer outlines of a (multi)polygon, as lists of ``[x, y]``."""
    if geometry is None or geometry.is_empty:
        return []
    try:
        geometry = geometry.simplify(OUTLINE_TOLERANCE_M)
    except Exception:  # noqa: BLE001 - drawn as it is, then
        quiet.detail("An outline could not be simplified: drawn as it is")
    parts = getattr(geometry, "geoms", [geometry])
    rings = []
    for part in parts:
        exterior = getattr(part, "exterior", None)
        if exterior is None:
            continue
        rings.append([[float(x), float(y)] for x, y in exterior.coords])
    return rings


def plugin_version() -> str:
    """The version of the plugin, from its ``metadata.txt``."""
    path = os.path.join(os.path.dirname(__file__), "metadata.txt")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        quiet.detail("The plugin's metadata could not be read")
    return ""


class SurveyReportMixin:
    """The survey in one HTML file."""

    def run_survey_report(self, config: Dict[str, Any], progress_fn=None,
                          log_fn=None, cancel_check=None):
        """Write ``<survey>/analytics_{m}/survey_report.html``: what was
        flown and in what weather, a map, the counts, every density the
        survey has, where the animals were, what changed between visits,
        how the animals took the drone, how good the places are, and the
        settings - one file that needs nothing else
        (:mod:`core.survey_report`).

        The report reads the results the other analytics wrote; only the
        census is built here - when it is missing, or was made from other
        flights or species than the survey has now - since the counts are
        what a report starts from. Every other result that is not of this
        survey is named, in the log and at the top of the page. Settings
        (``config``): ``report_title``, and
        ``report_canvas_png`` - a picture of the map as QGIS showed it, to
        go beside the drawn one.
        """
        from .bambi_processing import CancelledException
        from .core import census, survey_report, survey_trend

        survey = self._survey(config)
        source = chosen_source(config, "report_source")
        if log_fn:
            log_fn(f"Writing the survey report ({survey.camera_label}, "
                   f"{source}) of {self._flights_note(survey)}...")
        area = self._survey_area(survey, log_fn)
        folder = survey.ensure_analytics_folder()
        if progress_fn:
            progress_fn(3)

        stale = self._report_census_stale(
            os.path.join(folder, f"census_summary_{source}.json"), survey)
        if stale:
            if log_fn:
                log_fn(f"{stale} - it is built first, since the counts "
                       "are what a report starts from.")
            self.run_census_summary(
                dict(config, census_source=source), None, log_fn,
                cancel_check)
        if progress_fn:
            progress_fn(25)

        flights, routes, animals, seen = [], [], [], []
        unplaced: List[str] = []
        colours: Dict[str, str] = {}
        for index, flight in enumerate(survey.flights):
            if cancel_check and cancel_check():
                raise CancelledException("Survey report cancelled")
            records = [record for record in self._census_records(
                survey, flight, source, area)
                if census.counts_inside(record, area)]
            for record in records:
                if record.get("x") is not None and record.get("y") is not None:
                    animals.append({"x": float(record["x"]),
                                    "y": float(record["y"]),
                                    "species": record["species"]})
            span = survey_trend.flight_span(flight.target_folder,
                                            survey.modality)
            route, route_km, frames = self._report_route(flight, survey)
            if route:
                routes.append({"flight": flight.name, "points": route})
            elif frames:
                unplaced.append(flight.name)
                if log_fn:
                    log_fn(f"The route of {flight.name} is left off the "
                           "map: its camera positions are placed by its "
                           "elevation model, which is not set")
            geometry = self._report_seen(flight, survey, area)
            if geometry is not None:
                seen.append(geometry)
            date = time = None
            if span is not None:
                date = span[0].date().isoformat()
                time = f"{span[0]:%H:%M}-{span[1]:%H:%M}"
            flights.append({
                "name": flight.name,
                "date": date,
                "time": time,
                "frames": frames,
                "route_km": route_km,
                "animals": len(records),
                "weather": self._report_weather(flight),
                "settings": self._report_settings(flight, survey),
            })
            for name, colour in self._report_colours(flight).items():
                colours.setdefault(name, colour)
            if progress_fn:
                progress_fn(int(25 + 55 * (index + 1) / len(survey.flights)))

        names = []
        for animal in animals:
            if animal["species"] not in names:
                names.append(animal["species"])
        if len(animals) > MAX_MAP_ANIMALS:
            step = int(math.ceil(len(animals) / MAX_MAP_ANIMALS))
            if log_fn:
                log_fn(f"The map shows one in {step} of the "
                       f"{len(animals)} animals: all of them would make the "
                       "page slow to open")
            animals = animals[::step]

        seen_rings = []
        try:
            from .core import survey_rasters

            union = survey_rasters.looked_at_geometry(seen) if seen else None
            seen_rings = _rings(union)
        except Exception as exc:  # noqa: BLE001 - the map does without
            quiet.problem("The ground the flights saw could not be drawn "
                          "on the report's map")
            if log_fn:
                log_fn(f"The ground the flights saw is left off the map "
                       f"({exc})")

        results = survey_report.collect(folder, source)
        title = (config.get("report_title") or "").strip()
        if not title:
            name = os.path.basename(os.path.normpath(survey.folder))
            title = f"Survey report - {name}"
        area_rings, excluded_rings = [], []
        if area is not None:
            area_rings = _rings(area.geometry)
            excluded_rings = _rings(getattr(area, "excluded_geometry", None))
        report = {
            "title": title,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "plugin_version": plugin_version(),
            "camera": survey.camera_label,
            "source": source,
            "crs": f"EPSG:{survey.epsg}" if survey.epsg else None,
            "species_filter": survey.species,
            "survey_area": None if area is None else area.describe(),
            "flights": flights,
            "map": {
                "area": area_rings,
                "exclusions": excluded_rings,
                "seen": seen_rings,
                "routes": routes,
                "routes_missing": unplaced,
                "animals": animals,
                "colours": survey_report.species_colours(names, colours),
            },
            "canvas_png": self._report_canvas(config, log_fn),
            "results": results,
        }
        problems: List[tuple] = []
        path = survey_report.write(
            report, os.path.join(folder, survey_report.FILE_NAME), problems)
        for title, why in survey_report.out_of_step(report):
            if log_fn:
                log_fn(f"Note: {title} was {why} - not what this report "
                       "lists. Run it again, then write the report anew.")
        for part, why in problems:
            quiet.problem(f"The report's part '{part}' could not be made "
                          f"({why})")
            if log_fn:
                log_fn(f"Note: '{part}' could not be made from its result "
                       f"file ({why}). Run its analysis again, then write "
                       "the report anew.")
        if log_fn:
            included = ", ".join(
                survey_report.RESULTS[key][1] for key in results["found"])
            log_fn(f"In the report: {included or 'nothing but the flights'}")
            if results["missing"]:
                not_run = ", ".join(results["missing"])
                log_fn(f"Not run, and so not in it: {not_run}")
            log_fn(f"Survey report written to {path}")
        if progress_fn:
            progress_fn(100)
        return path

    @staticmethod
    def _report_census_stale(path: str, survey) -> str:
        """Why the census at *path* has to be built before the report -
        ``""`` when it is there and is of this survey."""
        from .core import survey_report

        if not os.path.isfile(path):
            return "The census has not been built for this survey yet"
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            return "The census that is there cannot be read"
        if not isinstance(document, dict):
            return "The census that is there cannot be read"
        why = survey_report.not_of_survey(
            document, [flight.name for flight in survey.flights],
            survey.species)
        return f"The census that is there was {why}" if why else ""

    # ------------------------------------------------------------------ #
    # What is not in a result file
    # ------------------------------------------------------------------ #

    def _report_route(self, flight, survey):
        """``(points, kilometres, frames)`` - where the camera went, in the
        project's coordinates and thinned out for drawing; the length
        flown; and how many frames there are. No points where the flight's
        DEM - by whose origin the poses are shifted - is not known."""
        from .core import poses
        from .core.transects import flight_positions

        images = poses.load_images(flight.target_folder, survey.modality)
        positions = [place for place in flight_positions(images)
                     if place is not None]
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(positions, positions[1:]))
        kilometres = length / 1000.0 if positions else None
        try:
            x_offset, y_offset, _source = self._resolve_population_origin(
                flight)
        except FileNotFoundError:
            return [], kilometres, len(images)
        step = max(1, int(math.ceil(len(positions) / MAX_ROUTE_POINTS)))
        thinned = positions[::step]
        if positions and thinned[-1] != positions[-1]:
            thinned.append(positions[-1])
        return ([[x + x_offset, y + y_offset] for x, y in thinned],
                kilometres, len(images))

    @staticmethod
    def _report_seen(flight, survey, area):
        """The ground the flight's camera saw, or ``None``."""
        from .core import habitat, store

        if not os.path.isfile(store.stage_path(
                flight.target_folder, store.FOV, survey.modality)):
            return None
        try:
            return habitat.monitored_geometry(
                flight.target_folder, survey.modality, area)
        except Exception:  # noqa: BLE001 - the map does without
            quiet.problem(f"The footprints of {flight.name} could not be "
                          "read for the report's map")
            return None

    @staticmethod
    def _report_weather(flight) -> dict:
        from .core import habitat

        try:
            weather = habitat.weather_values(flight.target_folder)
        except Exception:  # noqa: BLE001 - the weather is extra
            quiet.problem(f"The weather of {flight.name} could not be read "
                          "for the report")
            return {}
        return {key: value for key, value in weather.items()
                if value is not None}

    @staticmethod
    def _report_colours(flight) -> Dict[str, str]:
        from .core import map_symbols

        try:
            return map_symbols.species_colours(flight.target_folder)
        except Exception:  # noqa: BLE001 - the report has colours of its own
            quiet.detail(f"The species colours of {flight.name} could not "
                         "be read: the report uses its own")
            return {}

    @staticmethod
    def _report_settings(flight, survey) -> "OrderedDict[str, str]":
        """What made the flight's data, for the Settings of the report:
        the detector's model, the tracker, the classifiers, and the
        settings that decide where a detection lands."""
        from .core import (
            classification_store, flights as flights_core, store, track_store)

        settings: "OrderedDict[str, str]" = OrderedDict()
        folder, modality = flight.target_folder, survey.modality
        try:
            stored = flights_core.load_config(folder)
        except Exception:  # noqa: BLE001 - a flight without a stored form
            quiet.detail(f"{flight.name} has no stored configuration")
            stored = {}

        project_path = store.project_path(folder)
        det_path = store.stage_path(folder, store.DETECTIONS, modality)
        if os.path.isfile(project_path) and os.path.isfile(det_path):
            try:
                project = store.open_store(project_path, store.PROJECT)
                try:
                    sources = {int(row["source_id"]): (row["kind"],
                                                       row["model"] or "")
                               for row in project.execute(
                        "SELECT source_id, kind, model "
                        "FROM detection_sources")}
                finally:
                    project.close()
                conn = store.open_store(det_path, store.DETECTIONS, modality)
                try:
                    counts = {int(row["source_id"]): int(row["n"])
                              for row in conn.execute(
                        "SELECT source_id, COUNT(*) AS n FROM detections "
                        "GROUP BY source_id")}
                finally:
                    conn.close()
                for source_id, n in sorted(counts.items()):
                    kind, model = sources.get(source_id, ("unknown", ""))
                    name = os.path.basename(model) if model else "no model name"
                    settings[f"Boxes of the {kind}"] = f"{n} - {name}"
            except Exception:  # noqa: BLE001 - said as not known
                quiet.problem(f"What made the boxes of {flight.name} could "
                              "not be read for the report")
        confidence = stored.get("Detection/Confidence")
        if confidence not in (None, ""):
            settings["Detection confidence"] = str(confidence)

        try:
            run = track_store.active_run(folder, modality)
        except Exception:  # noqa: BLE001 - said as not known
            quiet.problem(f"The tracking run of {flight.name} could not be "
                          "read for the report")
            run = None
        if run is not None:
            settings["Tracker"] = str(run.get("tracker") or run.get("kind"))

        for task, title in (("species", "Species classifier"),
                            ("sex", "Sex classifier"),
                            (classification_store.LIFE_STAGE, "Age")):
            try:
                models = sorted({
                    os.path.basename(str(call.get("model") or ""))
                    for call in classification_store.track_predictions(
                        folder, modality, task)} - {""})
            except Exception:  # noqa: BLE001 - not classified, then
                quiet.detail(f"No {task} calls could be read for "
                             f"{flight.name}")
                models = []
            if models:
                settings[title] = ", ".join(models)

        dem = (flight.dem_path or flight.dem or "").strip()
        if dem:
            settings["Elevation model"] = os.path.basename(dem)
        correction = (stored.get("Input/CorrectionPath") or "").strip()
        if correction:
            settings["Position correction"] = os.path.basename(correction)
        return settings

    @staticmethod
    def _report_canvas(config, log_fn=None):
        """The picture of the map QGIS showed, ready to go into the page -
        or ``None`` where there is none, or it is too large."""
        path = (config.get("report_canvas_png") or "").strip()
        if not path or not os.path.isfile(path):
            return None
        size = os.path.getsize(path)
        if size > MAX_CANVAS_BYTES:
            if log_fn:
                log_fn(f"The picture of the map is {size / 1e6:.1f} MB: too "
                       "large for the page, which keeps the drawn map only")
            return None
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
