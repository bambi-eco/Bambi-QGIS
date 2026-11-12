# -*- coding: utf-8 -*-
"""Environmental-context steps of the BAMBI processor.

Four per-flight sub-steps, one per data source, each fetching (or deriving)
what it needs for the ground the flight monitored and writing it into
``<flight>/environment/`` (see :mod:`core.environment`):

* ``env_sentinel2`` - NDVI, NDMI, NDWI and NDSI from the nearest clear
  Sentinel-2 scene;
* ``env_landcover`` - ESA WorldCover classes and tree cover;
* ``env_terrain`` - topographic wetness index from the flight's DEM;
* ``env_weather`` - hourly weather at flight time and the day before.

They are separate on purpose: a weather service that is down must not cost
the satellite layers, and the other way round. Within one sub-step there is
no partial result - a source that cannot be reached fails its step, and what
an earlier run left of that source is removed first, so old rasters never
sit beside a failed re-run looking current.

One survey-level step, ``habitat_summary``, joins the flights' environment
with their tracks (see :mod:`core.habitat`). Mixed into
:class:`BambiProcessor`.
"""

import os
from typing import Any, Dict

#: Worker step -> environment source.
ENVIRONMENT_STEPS = {
    "env_sentinel2": "sentinel2",
    "env_landcover": "landcover",
    "env_terrain": "terrain",
    "env_weather": "weather",
}


def _cancelled(message):
    from .bambi_processing import CancelledException
    return CancelledException(message)


class EnvironmentMixin:
    """Per-flight environment sources and the survey's habitat summary."""

    # ------------------------------------------------------------------ #
    # Shared
    # ------------------------------------------------------------------ #

    @staticmethod
    def _environment_origin(config: Dict[str, Any]):
        """The DEM mesh origin in the project CRS, or ``None``."""
        from .core import agl_profile

        return agl_profile.dem_origin(
            config.get("dem_path", "") or "",
            config.get("alfs_dem_metadata_path", "") or "",
            config.get("target_folder", "") or "")

    @staticmethod
    def _environment_area_bounds(config: Dict[str, Any], epsg, log_fn=None):
        """Bounds of the survey area to include in the extent, or ``None``.

        The survey area belongs to the survey, not to the flight; when one is
        switched on and drawn, the flight's environment is widened to cover
        it, so the layers show the whole plot and not just the strip flown.
        """
        data = config.get("survey")
        if not data or not config.get("environment_include_survey_area", True):
            return None
        from .core.survey import Survey

        try:
            survey = Survey.from_dict(data)
            if not survey.area_enabled or not survey.folder:
                return None
            from .core import survey_area
            area = survey_area.load(survey.folder, epsg)
        except Exception as exc:  # noqa: BLE001 - the area is an extra
            if log_fn:
                log_fn(f"Survey area not used for the extent: {exc}")
            return None
        if area is None or area.geometry is None or area.geometry.is_empty:
            return None
        return tuple(float(v) for v in area.geometry.bounds)

    def _environment_grid(self, config: Dict[str, Any], log_fn=None):
        """``(grid, extent description)`` of the flight's environment."""
        from .core import environment

        target_folder = config["target_folder"]
        epsg = config.get("target_epsg")
        grid, description = environment.flight_grid(
            target_folder, epsg,
            origin=self._environment_origin(config),
            buffer_m=float(config.get(
                "environment_buffer_m", environment.DEFAULT_BUFFER_M)),
            area_bounds=self._environment_area_bounds(config, epsg, log_fn))
        if log_fn:
            log_fn(f"Extent: {description} - {grid.width * grid.resolution / 1000:.2f} x "
                   f"{grid.height * grid.resolution / 1000:.2f} km on a "
                   f"{grid.resolution:g} m grid (EPSG:{grid.epsg})")
        return grid, description

    def _run_environment_source(self, source: str, config: Dict[str, Any],
                                produce, progress_fn=None, log_fn=None,
                                cancel_check=None):
        """Run one source: clear its old output, produce, write, record.

        *produce(grid, description)* returns ``(layers, metadata)``; the
        rasters are written first and the metadata last, so the metadata
        file's presence means the source finished.
        """
        from .core import environment

        target_folder = config.get("target_folder", "")
        if not target_folder or not os.path.isdir(target_folder):
            raise RuntimeError("The flight's target folder does not exist.")
        title = environment.SOURCE_TITLES[source]
        if log_fn:
            log_fn(f"{title}...")
        try:
            grid, description = self._environment_grid(config, log_fn)
            environment.clear(target_folder, source)
            layers, metadata = produce(grid, description)
        except environment.Cancelled as exc:
            raise _cancelled(f"{title} was cancelled") from exc
        except environment.EnvironmentFailure as exc:
            raise RuntimeError(f"{title}: {exc}") from exc
        if cancel_check and cancel_check():
            raise _cancelled(f"{title} was cancelled")

        files = []
        for key, array in layers.items():
            layer = environment.LAYERS_BY_KEY[key]
            colormap = metadata.pop(f"_colormap_{key}", None)
            environment.write_raster(
                environment.layer_path(target_folder, key), array, grid,
                description=layer.title,
                nodata=0 if layer.kind == environment.CATEGORICAL else None,
                colormap=colormap)
            files.append(layer.filename)
        metadata = dict(metadata)
        metadata.update({"files": files, "grid": grid.to_dict(),
                         "extent": description})
        path = environment.write_metadata(target_folder, source, metadata)
        if log_fn:
            if files:
                written = ", ".join(files)
                log_fn(f"Written to {environment.folder(target_folder)}: "
                       f"{written}")
            log_fn(f"{title} recorded in {path}")
        if progress_fn:
            progress_fn(100)
        return metadata

    # ------------------------------------------------------------------ #
    # The four sources
    # ------------------------------------------------------------------ #

    def run_env_sentinel2(self, config: Dict[str, Any], progress_fn=None,
                          log_fn=None, cancel_check=None):
        """NDVI, NDMI, NDWI and NDSI from the Sentinel-2 scene nearest to
        the flight that is clear over it (:mod:`core.env_sentinel2`).

        Config keys: ``environment_window_days`` (how far from the flight a
        scene may be), ``environment_min_clear`` (required usable share over
        the flight, 0-1), ``environment_buffer_m``.
        """
        from .core import env_sentinel2, environment

        def produce(grid, _description):
            start, end = environment.require_flight_time(
                config["target_folder"])
            middle = start + (end - start) / 2
            indices, metadata = env_sentinel2.fetch(
                grid, middle,
                window_days=float(config.get(
                    "environment_window_days",
                    env_sentinel2.DEFAULT_WINDOW_DAYS)),
                min_clear=float(config.get(
                    "environment_min_clear",
                    env_sentinel2.DEFAULT_MIN_CLEAR)),
                log=log_fn, cancelled=cancel_check, progress=progress_fn)
            metadata["means"] = env_sentinel2.means(indices)
            return indices, metadata

        return self._run_environment_source(
            "sentinel2", config, produce, progress_fn, log_fn, cancel_check)

    def run_env_landcover(self, config: Dict[str, Any], progress_fn=None,
                          log_fn=None, cancel_check=None):
        """ESA WorldCover classes and the tree cover derived from them
        (:mod:`core.env_landcover`)."""
        from .core import env_landcover

        def produce(grid, _description):
            layers, metadata = env_landcover.fetch(
                grid, log=log_fn, cancelled=cancel_check,
                progress=progress_fn)
            metadata["_colormap_landcover"] = env_landcover.colormap()
            return layers, metadata

        return self._run_environment_source(
            "landcover", config, produce, progress_fn, log_fn, cancel_check)

    def run_env_terrain(self, config: Dict[str, Any], progress_fn=None,
                        log_fn=None, cancel_check=None):
        """Topographic wetness index from the flight's DEM mesh
        (:mod:`core.env_terrain`). Needs no network."""
        from .core import env_terrain, inspection

        def produce(grid, _description):
            dem_path = config.get("dem_path", "") or ""
            dem_json = config.get("alfs_dem_metadata_path", "") or ""
            mesh_path = inspection.find_dem_mesh_path(dem_path, dem_json)
            if not mesh_path and dem_path and os.path.isfile(dem_path):
                mesh_path = dem_path
            return env_terrain.compute(
                grid, mesh_path or "", self._environment_origin(config),
                log=log_fn, cancelled=cancel_check, progress=progress_fn)

        return self._run_environment_source(
            "terrain", config, produce, progress_fn, log_fn, cancel_check)

    def run_env_weather(self, config: Dict[str, Any], progress_fn=None,
                        log_fn=None, cancel_check=None):
        """Hourly weather at flight time and over the 24 hours before it
        (:mod:`core.env_weather`). Writes ``weather.json`` and
        ``weather_hourly.csv``; there is no raster."""
        from .core import env_weather, environment

        def produce(grid, _description):
            target_folder = config["target_folder"]
            start, end = environment.require_flight_time(target_folder)
            lat, lon = environment.grid_centre_wgs84(grid)
            if progress_fn:
                progress_fn(20)
            document = env_weather.fetch(lat, lon, start, end, log=log_fn)
            env_weather.write_hourly_csv(document, os.path.join(
                environment.ensure_folder(target_folder),
                "weather_hourly.csv"))
            if log_fn:
                for line in env_weather.describe(document):
                    log_fn(line)
            return {}, document

        return self._run_environment_source(
            "weather", config, produce, progress_fn, log_fn, cancel_check)

    # ------------------------------------------------------------------ #
    # Survey analytics: habitat context
    # ------------------------------------------------------------------ #

    def run_habitat_summary(self, config: Dict[str, Any], progress_fn=None,
                            log_fn=None, cancel_check=None):
        """Environment values at the tracked animals of the survey, against
        the ground the flights monitored.

        Writes into ``<survey>/analytics_{m}``: ``habitat_tracks.csv`` /
        ``.json`` (one row per track), ``habitat_transects.csv`` (when a
        flight has transect areas from the population estimate) and
        ``habitat.json`` (the pooled distributions the charts are drawn
        from). See :mod:`core.habitat`.

        Config key: ``environment_radius_m`` - the neighbourhood a track's
        values are averaged over.
        """
        from .core import analytics_source, environment, habitat
        from .core import store as _store

        survey = self._survey(config)
        area = self._survey_area(survey, log_fn)
        radius = float(config.get("environment_radius_m",
                                  environment.DEFAULT_RADIUS_M))
        if log_fn:
            log_fn(f"Summarising the habitat of the {survey.camera_label} "
                   f"tracks of {self._flights_note(survey)}...")

        without = [flight.name for flight in survey.flights
                   if not any(environment.status(
                       flight.target_folder).values())]
        if without:
            raise RuntimeError(
                "No environment data for: " + ", ".join(without) + ". Run "
                "the Environmental Context steps on the Pre-Processing tab "
                "for each of these flights first.")

        results = []
        for index, flight in enumerate(survey.flights):
            if cancel_check and cancel_check():
                raise _cancelled("Habitat summary was cancelled")
            ids = survey.species_ids(flight)
            self._require_store(flight.target_folder, _store.TRACKS,
                                survey.modality, "Track Animals")
            try:
                result = habitat.summarise_flight(
                    flight.name, flight.target_folder, survey.modality,
                    radius, species_ids=ids,
                    include_manual=survey.include_manual, area=area,
                    epsg=survey.epsg)
            except analytics_source.AnalyticsError as exc:
                raise RuntimeError(f"{flight.name}: {exc}") from exc
            results.append(result)
            if log_fn:
                note = (f" (not run: {', '.join(result['missing'])})"
                        if result["missing"] else "")
                log_fn(f"{flight.name}: {len(result['tracks'])} track(s), "
                       f"{len(result['transects'])} transect(s){note}")
            if progress_fn:
                progress_fn(int(85 * (index + 1) / len(survey.flights)))

        summary = habitat.build_summary(
            results, radius, survey.modality, survey.species)
        summary["survey_area"] = None if area is None else area.describe()
        paths = habitat.write_outputs(
            survey.ensure_analytics_folder(), summary, results)
        if log_fn:
            for line in habitat.describe(summary):
                log_fn(line)
            for what, path in paths.items():
                log_fn(f"{what}: {path}")
        if progress_fn:
            progress_fn(100)
        return summary
