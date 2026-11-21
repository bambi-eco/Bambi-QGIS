# -*- coding: utf-8 -*-
"""The Grid Density step of the BAMBI processor (see
:mod:`core.grid_density`).

Like the other survey analytics it works on a *survey* - the flights ticked
on the Survey Analytics tab, one camera, one output folder. Mixed into
:class:`BambiProcessor`.
"""

import os
from typing import Any, Dict

from .core import quiet
from .core.survey import chosen_source


class SurveyGridMixin:
    """Animal density cell by cell over the survey's ground."""

    def run_grid_density(self, config: Dict[str, Any], progress_fn=None,
                         log_fn=None, cancel_check=None):
        """Cut the survey's ground into equal cells and count it cell by
        cell.

        Writes into ``<survey>/analytics_{m}/``:

        * ``grid_density_{source}.json`` - everything: the grid, every cell
          with its corners, the per-flight figures, the statistics;
        * ``grid_density_{source}_cells.csv`` - one row per cell;
        * ``grid_density_{source}_by_flight.csv`` - one row per cell and
          flight.

        Settings (``config``): ``grid_shape``, ``grid_size_mode``,
        ``grid_cell_area_ha``, ``grid_cell_edge_m``, ``grid_cell_width_m``,
        ``grid_cell_height_m`` (:class:`core.survey_grid.GridSpec`),
        ``grid_min_seen_pct`` - the share of a cell that has to have been
        seen for it to count - and ``grid_repeat``: how a cell several
        flights saw is read, ``"average"`` or ``"highest"``.
        """
        from .bambi_processing import CancelledException
        from .core import (
            env_landcover, grid_density, habitat, survey_grid,
            survey_rasters)

        survey = self._survey(config)
        source = chosen_source(config, "grid_source")
        try:
            spec = survey_grid.GridSpec.from_config(config)
        except survey_grid.GridError as exc:
            raise RuntimeError(str(exc)) from exc
        min_seen = float(config.get("grid_min_seen_pct", 50.0)) / 100.0
        repeat = str(config.get("grid_repeat") or grid_density.AVERAGE)
        if log_fn:
            log_fn(f"Grid density ({survey.camera_label}, {source}) of "
                   f"{self._flights_note(survey)}: {spec.describe()}...")
        area = self._survey_area(survey, log_fn)
        if progress_fn:
            progress_fn(2)

        flights = []
        for flight in survey.flights:
            if cancel_check and cancel_check():
                raise CancelledException("Grid density cancelled")
            found = self._flight_groups(survey, flight, source, log_fn)
            records = self._census_records(survey, flight, source, area,
                                           log_fn, found)
            seen = habitat.monitored_geometry(
                flight.target_folder, survey.modality, area,
                survey.confirmed)
            if seen is None:
                raise RuntimeError(
                    f"{flight.name}: no {survey.camera_label} field-of-view "
                    "footprints, so the ground this flight saw is not "
                    "known. Run 'Calculate Field of View' for this camera - "
                    "or draw a survey area, which then counts as seen.")
            if log_fn and not self._has_footprints(survey, flight):
                log_fn(f"{flight.name}: no field-of-view footprints - the "
                       "whole survey area counts as seen by this flight")
            flights.append({"name": flight.name, "records": records,
                            "seen": seen})

        if area is not None and area.geometry is not None:
            region, plot_m2 = area.geometry, area.adjusted_m2
        else:
            # no plot: the grid covers the ground the flights saw
            region = survey_rasters.looked_at_geometry(
                flight["seen"] for flight in flights)
            plot_m2 = None

        def scaled(value: int) -> None:
            if progress_fn:
                progress_fn(5 + int(value * 0.75))

        try:
            document = grid_density.analyse(
                spec, flights, region, min_seen=min_seen, repeat=repeat,
                plot_m2=plot_m2, species_order=survey.species_names(),
                progress=scaled, cancelled=cancel_check)
        except InterruptedError as exc:
            raise CancelledException(str(exc)) from exc
        except grid_density.GridDensityError as exc:
            raise RuntimeError(str(exc)) from exc

        # ---- what lies in each cell --------------------------------------- #
        layer_sets = []
        for flight in survey.flights:
            try:
                layers = habitat.load_layers(flight.target_folder)
            except Exception as exc:  # noqa: BLE001 - rasters are optional
                quiet.problem(f"The environment layers of {flight.name} "
                              "could not be read: left out of the grid")
                if log_fn:
                    log_fn(f"{flight.name}: the environment layers could "
                           f"not be read ({exc}) - left out of the grid")
                layers = {}
            if layers:
                layer_sets.append(layers)
        keys = grid_density.add_environment(
            document, layer_sets, class_name=env_landcover.class_name)
        if progress_fn:
            progress_fn(88)

        document.update({
            "source": source,
            "camera": survey.counted_label,
            "crs": f"EPSG:{survey.epsg}" if survey.epsg else None,
            "survey_area": None if area is None else area.describe(),
            "species_filter": survey.species,
            "notes": self._grid_notes(document, area is not None, keys),
        })
        folder = survey.ensure_analytics_folder()
        stem = f"grid_density_{source}"
        json_path = grid_density.write_json(
            document, os.path.join(folder, f"{stem}.json"))
        grid_density.write_table(
            grid_density.flat_cells(document),
            os.path.join(folder, f"{stem}_cells.csv"))
        grid_density.write_table(
            grid_density.flat_flights(document),
            os.path.join(folder, f"{stem}_by_flight.csv"))

        if log_fn:
            for line in grid_density.describe(document):
                log_fn(line)
            if keys:
                log_fn("Environment per cell: " + ", ".join(keys))
            log_fn(f"Grid density written to {json_path}")
        if progress_fn:
            progress_fn(100)
        return document

    @staticmethod
    def _has_footprints(survey, flight) -> bool:
        from .core import store

        return os.path.isfile(store.stage_path(
            flight.target_folder, store.FOV, survey.modality))

    @staticmethod
    def _grid_notes(document: dict, with_area: bool, layers) -> str:
        from .core import grid_density

        share = int(round(document["min_seen"] * 100))
        notes = (
            "A cell's density is its animals over the ground the cameras "
            "saw in it, not over the size of the cell. Cells of which less "
            f"than {share} % was seen are kept as 'too little seen' and "
            "left out of every statistic. ")
        if document["repeat"] == grid_density.HIGHEST:
            notes += (
                "Where several flights saw a cell, it is read from the "
                "flight with the highest density that saw enough of it on "
                "its own. ")
        else:
            notes += (
                "Where several flights saw a cell, all sightings are "
                "divided by all the ground looked at: the mean density over "
                "the visits. ")
        notes += (
            "The interval of the mean density comes from resampling the "
            "counted cells; it takes them as independent, so with clumped "
            "animals (Moran's I above its expectation) it is somewhat too "
            "narrow. The variance-to-mean ratio is 1 for animals scattered "
            "at random, above for clumped, below for evenly spread ones. A "
            "hot spot is a cell that holds more, together with its "
            "neighbours, than cells picked at random would (Getis-Ord Gi*, "
            "not corrected for the number of cells tested). The share of "
            "cells with animals is a minimum: an animal that was there and "
            "not found makes a cell look empty. ")
        if with_area:
            notes += ("The total is the mean density times the survey area "
                      "minus its exclusion zones. ")
        else:
            notes += ("Without a survey area the grid covers the ground "
                      "the flights saw, and no total is given. ")
        if layers:
            notes += ("Environment values are the mean of each layer over "
                      "the cell, and the most common land-cover class with "
                      "its share.")
        return notes.strip()
