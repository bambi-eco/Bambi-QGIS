# -*- coding: utf-8 -*-
"""Survey Analytics steps of the BAMBI processor (see :mod:`core.survey`).

Every analytic here works on a *survey* - the processed flights ticked on the
Survey Analytics tab, one camera, one output folder - rather than on the
flight selected on the processing tabs. Mixed into :class:`BambiProcessor`.
"""

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _cancelled(message):
    from .bambi_processing import CancelledException
    return CancelledException(message)


class SurveyAnalyticsMixin:
    """Survey-level analytics: census, inventory, density, coverage,
    distance sampling and population estimation."""

    # ------------------------------------------------------------------ #
    # The survey
    # ------------------------------------------------------------------ #

    def _survey(self, config: Dict[str, Any]):
        """The validated :class:`core.survey.Survey` of *config*."""
        from .core.survey import Survey, SurveyError

        try:
            return Survey.from_config(config).validate()
        except SurveyError as exc:
            raise RuntimeError(str(exc)) from exc

    def _survey_area(self, survey, log_fn=None):
        """The survey plot / exclusion zones, when switched on and defined.

        ``None`` when the filter is off or no area is drawn - the analytic
        then covers the whole of every flight. See :mod:`core.survey_area`.
        """
        if not survey.area_enabled:
            return None
        from .core import survey_area

        try:
            area = survey.area()
        except survey_area.SurveyAreaError as exc:
            raise RuntimeError(str(exc)) from exc
        if area is None:
            if log_fn:
                log_fn(f"No survey area defined in {survey.folder} - the "
                       "whole flights are analysed.")
            return None
        if area.has_plot and not area.adjusted_m2:
            raise RuntimeError(
                "The exclusion zones cover the whole survey plot, so there "
                "is no ground left to analyse.")
        if log_fn:
            log_fn(f"Survey area: {area.summary()}")
        return area

    @staticmethod
    def _survey_area_mask(area, shape, transform):
        """Boolean raster, True on cells whose centre lies in *area*."""
        import numpy as np
        from rasterio.features import geometry_mask

        if area.geometry is not None:
            if area.geometry.is_empty:
                return np.zeros(shape, dtype=bool)
            return geometry_mask([area.geometry], out_shape=shape,
                                 transform=transform, invert=True)
        # Exclusions only: everything except them.
        return geometry_mask([area.excluded_geometry], out_shape=shape,
                             transform=transform, invert=False)

    def _survey_points(self, survey, source: str, names=None, area=None,
                       log_fn=None) -> Tuple[List[Tuple[float, float]], dict]:
        """World-coordinate points of every flight of the survey, pooled.

        For ``source == "detections"`` every geo-referenced detection centre;
        for ``"tracks"`` one point per track (the centroid of its box centres),
        so an animal followed across many frames counts once.

        Read from each flight's store, which applies the population filter of
        §8.2 - false positives excluded, one tracker run plus the manual one,
        superseded tracks dropped. *names* are species names (``None`` = all),
        translated into each flight's own ids. The provenance lists every
        flight's filter, so a pooled number can be traced back to its rows.
        """
        from .core import analytics_source, store as _store

        step = ("Track Animals" if source == "tracks"
                else "Geo-Reference Detections")
        points: List[Tuple[float, float]] = []
        per_flight = []
        for flight in survey.flights:
            ids = survey.species_ids(flight, names)
            if ids is not None and not ids:
                # The flight knows none of these species: nothing to count.
                per_flight.append({"flight": flight.name, "n_points": 0,
                                   "species_filter": []})
                continue
            self._require_store(flight.target_folder, _store.GEOREFERENCED,
                                survey.modality, step)
            try:
                found, provenance = analytics_source.load_points(
                    flight.target_folder, survey.modality, source,
                    species_ids=ids, include_manual=survey.include_manual,
                    area=area)
            except analytics_source.AnalyticsError as exc:
                raise RuntimeError(f"{flight.name}: {exc}") from exc
            provenance["flight"] = flight.name
            provenance["target_folder"] = flight.target_folder
            points.extend(found)
            per_flight.append(provenance)
            if log_fn:
                log_fn(f"{flight.name}: {len(found)} {source} - "
                       f"{analytics_source.describe_filter(provenance)}")

        provenance = {
            "source": source,
            "modality": survey.modality,
            "species": None if names is None else list(names),
            "n_points": len(points),
            "outside_survey_area": sum(
                p.get("outside_survey_area", 0) for p in per_flight),
            "survey_area": None if area is None else area.describe(),
            "flights": per_flight,
        }
        return points, provenance

    def _require_survey_points(self, survey, source, names, area, log_fn):
        """:meth:`_survey_points`, failing clearly when nothing is left."""
        points, provenance = self._survey_points(
            survey, source, names, area, log_fn)
        if not points:
            if provenance["outside_survey_area"]:
                raise RuntimeError(
                    f"All {provenance['outside_survey_area']} {source} lie "
                    "outside the survey area (or inside an exclusion zone). "
                    "Check the plot polygon, or untick 'Limit analytics to "
                    "the survey area'.")
            if names is not None:
                raise RuntimeError(
                    "No geo-referenced points match the selected species in "
                    "any selected flight. Choose 'All species', or tick a "
                    "species these flights recorded.")
            step = ("Track Animals" if source == "tracks"
                    else "Geo-Reference Detections")
            raise RuntimeError(
                f"No {source} points available in the selected flights. "
                f"Run '{step}' first.")
        return points, provenance

    def survey_strata(self, config: Dict[str, Any], source: str,
                      log_fn=None) -> List[Tuple[str, Optional[List[str]]]]:
        """The runs an analytic performs, as ``(label, species names)``.

        One pooled run - ``("", <the species filter>)`` - unless "separate
        result per species" is set: then one per species that has points in
        any selected flight. Species with nothing to count are reported and
        left out rather than producing an empty result apiece.
        """
        survey = self._survey(config)
        if not survey.per_species:
            return [("", survey.species)]
        area = self._survey_area(survey)

        def has_data(names):
            try:
                found, _provenance = self._survey_points(
                    survey, source, names, area)
            except (RuntimeError, OSError):
                return False
            return bool(found)

        strata, empty = survey.strata(has_data)
        if log_fn and empty:
            log_fn(f"No {source} for: {', '.join(empty)} - skipped.")
        if not strata:
            raise RuntimeError(
                "No species has any geo-referenced "
                f"{source} to analyse, so there is nothing to produce a "
                "result for.")
        if log_fn:
            log_fn(f"Running per species: "
                   f"{', '.join(name for name, _ in strata)}")
        return strata

    def _per_stratum(self, config, source, run_once, what, progress_fn,
                     log_fn, cancel_check, tolerate_failures=False):
        """Run *run_once(survey, label, names, progress)* for every stratum.

        With *tolerate_failures*, a stratum that cannot be estimated is
        reported and skipped: with several species it is normal for the rare
        ones to have too few observations, and that is a result about those
        species, not an error about the others.
        """
        survey = self._survey(config)
        strata = self.survey_strata(config, source, log_fn)
        failures = []
        for index, (label, names) in enumerate(strata):
            if cancel_check and cancel_check():
                raise _cancelled(f"{what} cancelled")
            if log_fn and label:
                log_fn(f"- {label} -")

            def scaled(value, index=index):
                """Progress across all strata, not within one."""
                if progress_fn:
                    span = 100.0 / len(strata)
                    progress_fn(int(index * span + value * span / 100.0))

            try:
                run_once(survey, label, names,
                         scaled if progress_fn else None)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                if not (tolerate_failures and label):
                    raise
                failures.append((label, str(exc)))
                if log_fn:
                    log_fn(f"{label}: no estimate - {exc}")
        if failures and len(failures) == len(strata):
            detail = "; ".join(f"{name} ({why})" for name, why in failures)
            raise RuntimeError(f"No species could be estimated: {detail}")
        if progress_fn:
            progress_fn(100)

    @staticmethod
    def _flights_note(survey) -> str:
        return ", ".join(f.name for f in survey.flights)

    # ------------------------------------------------------------------ #
    # Density heatmap
    # ------------------------------------------------------------------ #

    def run_density_heatmap(self, config: Dict[str, Any], progress_fn=None,
                            log_fn=None, cancel_check=None):
        """Generate a density raster - one, or one per species (§8.2)."""
        source = config.get("density_source", "detections")
        self._per_stratum(
            config, source,
            lambda survey, label, names, progress: self._run_density_heatmap_once(
                config, survey, label, names, progress, log_fn, cancel_check),
            "Density heatmap", progress_fn, log_fn, cancel_check)

    def _run_density_heatmap_once(self, config: Dict[str, Any], survey,
                                  label: str = "", names=None,
                                  progress_fn=None, log_fn=None,
                                  cancel_check=None):
        """Generate a kernel-density estimate raster of animal locations.

        Points come from either geo-referenced detections or tracks (one point
        per track) of every flight of the survey. A Gaussian kernel is applied
        and the result written as a single-band float32 GeoTIFF where each
        pixel is the estimated point density in animals per hectare. Cells
        with no signal are set to nodata.

        :param config: Configuration dictionary. Relevant keys:
            ``density_source`` ("detections"|"tracks"), ``density_cell_size`` (m),
            ``density_bandwidth`` (m).
        """
        import math
        import numpy as np

        source = config.get("density_source", "detections")
        cell_size = float(config.get("density_cell_size", 5.0))
        bandwidth = float(config.get("density_bandwidth", 25.0))
        target_epsg = survey.epsg or config.get("target_epsg", 32633)

        if cell_size <= 0:
            raise ValueError("Density cell size must be > 0.")
        if bandwidth <= 0:
            raise ValueError("Density bandwidth must be > 0.")

        if log_fn:
            log_fn(f"Generating {source} density heatmap "
                   f"(cell={cell_size} m, bandwidth={bandwidth} m) over "
                   f"{len(survey.flights)} flight(s)...")
        if progress_fn:
            progress_fn(5)

        area = self._survey_area(survey, log_fn=log_fn)
        points, provenance = self._require_survey_points(
            survey, source, names, area, log_fn)
        self._last_analytics_provenance = provenance
        suffix = survey.modality

        pts = np.asarray(points, dtype=np.float64)
        if progress_fn:
            progress_fn(20)

        # Grid bounds padded by 3 bandwidths so the kernel tails are captured.
        pad = 3.0 * bandwidth
        min_x = pts[:, 0].min() - pad
        max_x = pts[:, 0].max() + pad
        min_y = pts[:, 1].min() - pad
        max_y = pts[:, 1].max() + pad

        width = int(math.ceil((max_x - min_x) / cell_size))
        height = int(math.ceil((max_y - min_y) / cell_size))
        max_dim = 8192
        if width > max_dim or height > max_dim:
            scale = max((max_x - min_x) / max_dim, (max_y - min_y) / max_dim)
            cell_size = scale
            width = int(math.ceil((max_x - min_x) / cell_size))
            height = int(math.ceil((max_y - min_y) / cell_size))
            if log_fn:
                log_fn(f"Grid too large, increased cell size to {cell_size:.2f} m")
        width = max(1, width)
        height = max(1, height)

        if log_fn:
            log_fn(f"Density grid: {width} x {height} cells")
        if progress_fn:
            progress_fn(35)

        # Accumulate raw point counts into grid cells.
        counts = np.zeros((height, width), dtype=np.float64)
        for x, y in points:
            col = int((x - min_x) / cell_size)
            row = int((max_y - y) / cell_size)  # flip Y for raster orientation
            if 0 <= row < height and 0 <= col < width:
                counts[row, col] += 1.0

        if cancel_check and cancel_check():
            raise _cancelled("Density heatmap cancelled")
        if progress_fn:
            progress_fn(55)

        # Smooth with a Gaussian kernel (sigma in cells). gaussian_filter
        # conserves the total sum, so the smoothed value is counts per cell.
        sigma = bandwidth / cell_size
        try:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(counts, sigma=sigma, mode="constant", cval=0.0)
        except ImportError:
            smoothed = self._gaussian_blur_numpy(counts, sigma)

        if progress_fn:
            progress_fn(75)

        # Convert counts-per-cell to density per hectare.
        cell_area_m2 = cell_size * cell_size
        density = smoothed / cell_area_m2 * 10000.0

        # Everything below this threshold is treated as "no signal" (nodata) so
        # the empty surround renders transparent.
        nodata = -9999.0
        eps = (1.0 / cell_area_m2 * 10000.0) * 1e-4  # ~0 relative to one point
        signal = density > eps
        if area is not None:
            # Outside the plot or inside an exclusion zone is not surveyed
            # ground, so it shows no density at all rather than kernel tails.
            from rasterio.transform import from_bounds as _from_bounds
            signal &= self._survey_area_mask(
                area, density.shape,
                _from_bounds(min_x, min_y, max_x, max_y, width, height))
        out = np.where(signal, density, nodata).astype(np.float32)

        analytics_folder = survey.ensure_analytics_folder()
        stem = f"density_{source}"
        if label:
            stem += f"_{self.species_slug(label)}"
        out_file = os.path.join(analytics_folder, f"{stem}.tif")

        bounds = (min_x, min_y, max_x, max_y)
        self._save_single_band_raster(out, out_file, bounds, target_epsg, nodata, log_fn)

        valid = density[signal]
        stats = {
            "source": source,
            "camera_suffix": suffix,
            "crs": f"EPSG:{target_epsg}",
            "n_points": len(points),
            "cell_size_m": cell_size,
            "bandwidth_m": bandwidth,
            "unit": "points_per_hectare",
            "max_density": float(valid.max()) if valid.size else 0.0,
            "mean_density": float(valid.mean()) if valid.size else 0.0,
            "raster": out_file,
            "species": label or None,
            "flights": [f.name for f in survey.flights],
        }
        if area is not None:
            stats["survey_area"] = area.describe()
            if area.adjusted_m2:
                # The plain census figure: animals inside / adjusted area.
                stats["points_per_ha_in_survey_area"] = (
                    len(points) / (area.adjusted_m2 / 10000.0))
        # What went into the number, so it can be traced back (§8.2).
        if getattr(self, "_last_analytics_provenance", None):
            stats["population_filter"] = self._last_analytics_provenance
        with open(os.path.join(analytics_folder, f"{stem}.json"),
                  'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        if log_fn:
            log_fn(f"Density heatmap saved to: {out_file}")
            log_fn(f"Peak density: {stats['max_density']:.3f} points/ha "
                   f"from {len(points)} {source}")
            if "points_per_ha_in_survey_area" in stats:
                per_ha = stats["points_per_ha_in_survey_area"]
                log_fn(f"Survey area: {len(points)} {source} on "
                       f"{area.adjusted_m2 / 10000.0:.2f} ha = "
                       f"{per_ha:.3f} /ha ({per_ha * 100:.2f} /km²)")
        if progress_fn:
            progress_fn(100)
        return out_file

    @staticmethod
    def _gaussian_blur_numpy(grid, sigma):
        """Separable Gaussian blur fallback when SciPy is unavailable."""
        import math
        import numpy as np

        if sigma <= 0:
            return grid.astype(np.float64)
        radius = max(1, int(math.ceil(sigma * 3)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()

        out = grid.astype(np.float64)
        # Convolve rows then columns (separable).
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=out)
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=out)
        return out

    def _save_single_band_raster(self, array, output_file, bounds, crs_epsg, nodata, log_fn=None):
        """Write a single-band float32 GeoTIFF with the given world bounds."""
        import numpy as np

        height, width = array.shape
        min_x, min_y, max_x, max_y = bounds
        array = array.astype(np.float32)

        try:
            import rasterio
            from rasterio.transform import from_bounds

            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
            profile = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'width': width,
                'height': height,
                'count': 1,
                'transform': transform,
                'compress': 'lzw',
                'nodata': nodata,
            }
            try:
                from pyproj import CRS as PyprojCRS
                from rasterio.crs import CRS as RasterioCRS
                profile['crs'] = RasterioCRS.from_wkt(PyprojCRS.from_epsg(crs_epsg).to_wkt())
            except Exception:  # nosec B110
                try:
                    from rasterio.crs import CRS as RasterioCRS
                    profile['crs'] = RasterioCRS.from_epsg(crs_epsg)
                except Exception:  # nosec B110
                    pass
            if width > 256 and height > 256:
                profile.update({'tiled': True, 'blockxsize': 256, 'blockysize': 256})

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(array, 1)
                dst.set_band_description(1, "density_points_per_hectare")
            self._save_world_file(output_file, bounds, width, height)
            self._save_prj_file(output_file, crs_epsg, log_fn)
        except ImportError:
            raise RuntimeError(
                "rasterio is required to write the density heatmap GeoTIFF."
            )

    # ------------------------------------------------------------------ #
    # Track inventory
    # ------------------------------------------------------------------ #

    def _survey_inventory(self, survey, names=None, log_fn=None,
                          require_tracks=True) -> List[dict]:
        """Every tracked individual of every flight, with a ``flight`` column.

        Each flight also gets its own ``analytics_{m}/track_inventory.*`` -
        the inspector reads that one to show what is known about a track.
        """
        from .bambi_processing import _record_stage
        from .core import store as _store
        from .core import track_inventory

        epsg = survey.epsg
        everything = []
        for flight in survey.flights:
            if not os.path.isfile(_store.stage_path(
                    flight.target_folder, _store.TRACKS, survey.modality)):
                if require_tracks:
                    raise FileNotFoundError(
                        f"{flight.name}: no {survey.camera_label} tracks yet "
                        "- run 'Track Animals' for this camera first.")
                if log_fn:
                    log_fn(f"{flight.name}: no {survey.camera_label} tracks "
                           "- skipped.")
                continue
            ids = survey.species_ids(flight, names)
            if ids is not None and not ids:
                rows = []
            else:
                rows = track_inventory.build_inventory(
                    flight.target_folder, survey.modality,
                    epsg=int(epsg) if epsg else None, species_ids=ids)
            folder = os.path.join(flight.target_folder,
                                  f"analytics_{survey.modality}")
            os.makedirs(folder, exist_ok=True)
            track_inventory.write_csv(
                rows, os.path.join(folder, "track_inventory.csv"))
            track_inventory.write_json(
                rows, os.path.join(folder, "track_inventory.json"))
            _record_stage({"target_folder": flight.target_folder},
                          "track_inventory", survey.modality,
                          row_count=len(rows), log_fn=log_fn)
            if log_fn:
                log_fn(f"{flight.name}: {len(rows)} tracked individual(s)")
            for row in rows:
                tagged = {"flight": flight.name}
                tagged.update(row)
                everything.append(tagged)
        return everything

    def run_track_inventory(self, config: Dict[str, Any], progress_fn=None,
                            log_fn=None, cancel_check=None):
        """List every tracked individual of the survey's flights.

        Writes ``<survey>/analytics_{m}/track_inventory.csv`` and ``.json``
        with a ``flight`` column first: frames, capture times, start and end
        positions (project CRS and WGS84), species / sex / age with the votes
        behind them, occlusion counts, the cross-camera partner, the
        labelling tool's annotations and the flight-route distance. See
        :mod:`core.track_inventory`.
        """
        from .core import track_inventory

        survey = self._survey(config)
        if log_fn:
            log_fn(f"Listing the tracked {survey.camera_label} individuals "
                   f"of {self._flights_note(survey)}...")
        if progress_fn:
            progress_fn(10)

        inventory = self._survey_inventory(survey, survey.species, log_fn)
        if log_fn and survey.species is not None:
            log_fn(f"Species filter: {len(inventory)} track(s) of the "
                   f"{len(survey.species)} selected species listed")
        if progress_fn:
            progress_fn(70)

        analytics_folder = survey.ensure_analytics_folder()
        csv_path = os.path.join(analytics_folder, "track_inventory.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            columns = ["flight"] + list(track_inventory.COLUMNS)
            writer.writerow(columns)
            for row in inventory:
                writer.writerow([track_inventory.cell_text(row.get(c))
                                 for c in columns])
        track_inventory.write_json(
            inventory, os.path.join(analytics_folder, "track_inventory.json"))

        if log_fn:
            for line in track_inventory.summarise(inventory):
                log_fn(line)
            log_fn(f"Track inventory written to {csv_path}")
        if progress_fn:
            progress_fn(100)

    # ------------------------------------------------------------------ #
    # Census summary
    # ------------------------------------------------------------------ #

    def _census_records(self, survey, flight, source, area, log_fn=None):
        """``{species, sex, age, x, y, approved}`` per counted animal (tracks)
        or per box (detections) of one flight."""
        from .core import analytics_source, store as _store
        from .core import track_inventory
        from .core.exporters import common

        ids = survey.species_ids(flight)
        if ids is not None and not ids:
            return []
        if source == "tracks":
            if not os.path.isfile(_store.stage_path(
                    flight.target_folder, _store.TRACKS, survey.modality)):
                raise FileNotFoundError(
                    f"{flight.name}: no {survey.camera_label} tracks yet - "
                    "run 'Track Animals' for this camera first, or count "
                    "detections instead.")
            inventory = track_inventory.build_inventory(
                flight.target_folder, survey.modality, species_ids=ids)
            centroids = {}
            try:
                # Every track's position, whatever its boxes were labelled:
                # the species filter already chose the tracks (above).
                rows, _provenance = analytics_source.load_rows(
                    flight.target_folder, survey.modality,
                    include_manual=survey.include_manual)
                centroids, _untracked = analytics_source.track_centroids(rows)
            except analytics_source.AnalyticsError:
                pass  # nothing geo-referenced: every animal is unplaced
            records = []
            for row in inventory:
                x, y = centroids.get(row["track_id"], (None, None))
                records.append({
                    "species": row.get("species") or "",
                    "sex": row.get("sex"), "age": row.get("age"),
                    "x": x, "y": y, "approved": bool(row.get("approved")),
                })
            return records

        self._require_store(flight.target_folder, _store.DETECTIONS,
                            survey.modality, "Detect Animals")
        vocabulary = common.load_vocabulary(flight.target_folder)
        names = vocabulary["species"]
        enum_labels = vocabulary["enum_labels"]
        wanted = None if ids is None else set(ids)
        records = []
        for row in common.load_detections(flight.target_folder,
                                          survey.modality):
            if wanted is not None and row["species_id"] not in wanted:
                continue
            attributes = common.resolve_attributes(
                row.get("attributes"), enum_labels)
            placed = row.get("gx1") is not None
            records.append({
                "species": names.get(row["species_id"], ""),
                "sex": attributes.get("sex"), "age": attributes.get("age"),
                "x": (row["gx1"] + row["gx2"]) / 2.0 if placed else None,
                "y": (row["gy1"] + row["gy2"]) / 2.0 if placed else None,
                "approved": bool(attributes.get("approved")),
            })
        return records

    def _covered_m2(self, survey, flight, area, log_fn=None):
        """Ground the flight's frames saw inside *area* (``None`` without FoV)."""
        from .core import store as _store
        from .core.pipeline_outputs import load_fov_polygons_3d
        from .core.population import merged_fov_area

        if not os.path.isfile(_store.stage_path(
                flight.target_folder, _store.FOV, survey.modality)):
            if log_fn:
                log_fn(f"{flight.name}: no {survey.camera_label} FoV "
                       "footprints - covered area unknown.")
            return None
        fov_file = os.path.join(flight.target_folder,
                                f"fov_{survey.modality}", "fov_polygons.txt")
        polygons = load_fov_polygons_3d(fov_file, None)
        if not polygons:
            return None
        area_m2, geometry = merged_fov_area(polygons, sorted(polygons))
        if area is not None:
            geometry = area.clip(geometry)
            area_m2 = 0.0 if geometry is None else float(geometry.area)
        return area_m2

    def run_census_summary(self, config: Dict[str, Any], progress_fn=None,
                           log_fn=None, cancel_check=None):
        """Counts, densities and the sex / age breakdown per species and flight.

        Writes ``<survey>/analytics_{m}/census_summary.csv`` and ``.json``
        (see :mod:`core.census` for the two area bases). ``census_source``:
        ``"tracks"`` counts each tracked individual once; ``"detections"``
        counts every box, which is what a still-photo survey without
        tracking has.
        """
        from .core import census

        survey = self._survey(config)
        source = config.get("census_source", "tracks")
        if log_fn:
            log_fn(f"Building the {survey.camera_label} census summary "
                   f"({source}) of {self._flights_note(survey)}...")
        area = self._survey_area(survey, log_fn)

        flights = []
        for index, flight in enumerate(survey.flights):
            if cancel_check and cancel_check():
                raise _cancelled("Census summary cancelled")
            records = self._census_records(survey, flight, source, area,
                                           log_fn)
            covered = self._covered_m2(survey, flight, area, log_fn)
            flights.append({"name": flight.name, "records": records,
                            "covered_m2": covered})
            if log_fn:
                log_fn(f"{flight.name}: {len(records)} {source}")
            if progress_fn:
                progress_fn(int(10 + 80 * (index + 1) / len(survey.flights)))

        result = census.build(flights, area,
                              species_order=survey.species_names())
        result.update({
            "source": source,
            "camera": survey.camera_label,
            "crs": f"EPSG:{survey.epsg}" if survey.epsg else None,
            "survey_area": None if area is None else area.describe(),
            "species_filter": survey.species,
            "notes": (
                "Minimum counts: no detection probability is modelled. "
                "per_km2_plot divides by the survey plot minus its exclusion "
                "zones (assumes the flights cover it completely; across "
                "flights the counts are summed onto one plot). "
                "per_km2_covered divides by the ground the frames actually "
                "saw inside the plot, summed across flights."),
        })
        folder = survey.ensure_analytics_folder()
        stem = f"census_summary_{source}"
        csv_path = census.write_csv(result, os.path.join(folder, f"{stem}.csv"))
        census.write_json(result, os.path.join(folder, f"{stem}.json"))

        if log_fn:
            for row in result["rows"]:
                if row["species"] != census.ALL_SPECIES:
                    continue
                density = row["per_km2_plot"] or row["per_km2_covered"]
                note = f", {density:.2f} /km²" if density is not None else ""
                log_fn(f"{row['flight']}: {row['inside']} counted"
                       f"{' inside the survey area' if area else ''}{note}")
            log_fn(f"Census summary written to {csv_path}")
        if progress_fn:
            progress_fn(100)
        return result

    def run_coverage_map(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Build a survey coverage map from the exported frame GeoTIFFs.

        Combines the per-frame GeoTIFFs on the same grid as the orthomosaic,
        but instead of merging image content it counts, per output pixel, how
        many frames contain valid (non-nodata) data at that position. The
        result is a single-band raster where 1 means the ground was imaged
        once, N means it was seen in N overlapping frames, and nodata (0)
        means it was never covered.

        :param config: Configuration dictionary. Relevant keys:
            ``coverage_camera`` ("T"|"W"), ``coverage_cell_size`` (m, 0 =
            native GeoTIFF resolution), ``target_epsg``.
        """
        import numpy as np

        try:
            import rasterio
            from rasterio.merge import merge as rio_merge
        except ImportError as exc:
            raise ImportError(
                f"rasterio is required for coverage-map generation.\n"
                f"Original error: {exc}"
            )

        survey = self._survey(config)
        camera_suffix = survey.modality
        camera_name = survey.camera_label
        cell_size = float(config.get("coverage_cell_size", 1.0))
        nodata_src = config.get("ortho_nodata", 0)
        target_epsg = survey.epsg or config.get("target_epsg", 32633)

        if log_fn:
            log_fn(f"Building coverage map from the {camera_name} frame "
                   f"GeoTIFFs of {self._flights_note(survey)}...")

        # Every flight's frames on one grid: ground two flights saw counts
        # twice, which is exactly the survey effort.
        candidates = []
        for flight in survey.flights:
            geotiff_folder = os.path.join(flight.target_folder,
                                          f"geotiffs_{camera_suffix}")
            found = (self._collect_frame_geotiffs(geotiff_folder)
                     if os.path.isdir(geotiff_folder) else [])
            if not found:
                raise FileNotFoundError(
                    f"{flight.name}: no frame GeoTIFFs in {geotiff_folder}. "
                    f"Run 'Export Frames as GeoTIFF' for the {camera_name} "
                    "camera first.")
            candidates.extend(found)

        if log_fn:
            res_tex = " at native resolution..."
            if cell_size > 0:
                res_tex = f" at {cell_size} m/px..."
            log_fn(f"Counting overlap across {len(candidates)} GeoTIFF(s) {res_tex}")
        if progress_fn:
            progress_fn(10)

        # 0 (or negative) means: keep the native resolution of the exports.
        res = (cell_size, cell_size) if cell_size > 0 else None

        datasets = []
        source_crs = None
        try:
            for path in candidates:
                if cancel_check and cancel_check():
                    raise _cancelled("Coverage map cancelled")
                ds = rasterio.open(path)
                if source_crs is None and ds.crs is not None:
                    source_crs = ds.crs
                datasets.append(ds)

            if progress_fn:
                progress_fn(30)

            # First pass establishes the exact output grid (shape + transform);
            # the counting pass then reuses the same inputs/parameters, so
            # rasterio produces an identically aligned grid and the per-source
            # roff/coff offsets index our count array correctly (same approach
            # as _merge_orthomosaic_average).
            base, out_transform = rio_merge(
                datasets, method="first", nodata=nodata_src, res=res)
            out_height, out_width = base.shape[1], base.shape[2]
            del base

            if cancel_check and cancel_check():
                raise _cancelled("Coverage map cancelled")
            if progress_fn:
                progress_fn(50)

            count = np.zeros((out_height, out_width), dtype=np.float64)

            def _count_valid(merged_data, new_data, merged_mask, new_mask,
                             index=None, roff=0, coff=0, **kwargs):
                # Masks are True where data is *invalid* (nodata); a pixel is
                # covered by this frame when its first band is valid.
                valid = ~new_mask
                band0 = valid[0] if valid.ndim == 3 else valid
                h, w = band0.shape
                count[roff:roff + h, coff:coff + w] += band0
                if merged_mask.shape == valid.shape:
                    merged_mask[valid] = False

            rio_merge(datasets, method=_count_valid, nodata=nodata_src, res=res)
        finally:
            for d in datasets:
                d.close()

        if progress_fn:
            progress_fn(80)

        # Ground outside the survey plot (or inside an exclusion zone) is not
        # part of the survey, so it does not count as covered either.
        area = self._survey_area(survey, log_fn=log_fn)
        if area is not None:
            count[~self._survey_area_mask(area, count.shape, out_transform)] = 0

        out = count.astype(np.uint16)

        analytics_folder = survey.ensure_analytics_folder()
        output_file = os.path.join(analytics_folder, "coverage_map.tif")

        # Resolve the output CRS the same way the orthomosaic does:
        # authoritatively from target_epsg, falling back to the sources' CRS.
        out_crs = None
        try:
            from pyproj import CRS as PyprojCRS
            from rasterio.crs import CRS as RasterioCRS
            if target_epsg:
                out_crs = RasterioCRS.from_wkt(PyprojCRS.from_epsg(target_epsg).to_wkt())
        except Exception:
            out_crs = None
        if out_crs is None:
            out_crs = source_crs

        out_meta = {
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "count": 1,
            "dtype": "uint16",
            "crs": out_crs,
            "transform": out_transform,
            "compress": "LZW",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
            "nodata": 0,
        }
        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(out, 1)
            dst.set_band_description(1, "overlapping_frame_count")

        # Overview pyramids for fast GIS rendering; nearest keeps the counts
        # meaningful at reduced zoom (non-fatal on failure).
        try:
            from rasterio.enums import Resampling as RioResampling
            with rasterio.open(output_file, "r+") as dst:
                dst.build_overviews([2, 4, 8, 16], RioResampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
        except Exception as exc:
            if log_fn:
                log_fn(f"Note: could not build overviews ({exc})")

        # Summary statistics over the covered area.
        px_area_m2 = abs(out_transform.a * out_transform.e)
        covered = out[out > 0]
        stats = {
            "camera_suffix": camera_suffix,
            "crs": f"EPSG:{target_epsg}",
            "n_frames": len(candidates),
            "flights": [f.name for f in survey.flights],
            "cell_size_m": abs(out_transform.a),
            "unit": "overlapping_frame_count",
            "max_overlap": int(covered.max()) if covered.size else 0,
            "mean_overlap": float(covered.mean()) if covered.size else 0.0,
            "covered_area_ha": float(covered.size * px_area_m2 / 10000.0),
            "multi_covered_area_ha": float(
                int((covered >= 2).sum()) * px_area_m2 / 10000.0),
            "raster": output_file,
        }
        if area is not None:
            stats["survey_area"] = area.describe()
            if area.adjusted_m2:
                # Share of the plot the flight actually imaged - the gaps are
                # ground where an animal could not have been counted.
                stats["survey_area_coverage_fraction"] = min(
                    1.0, stats["covered_area_ha"] * 10000.0 / area.adjusted_m2)
        # No population filter is recorded here on purpose: the coverage map
        # counts frames rather than animals, so it has no species or tracking
        # run behind it. It used to attach whatever the last density run left
        # on the processor, which described a different product entirely.
        with open(os.path.join(analytics_folder, "coverage_map.json"),
                  'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        if log_fn:
            log_fn(f"Coverage map written to: {output_file} "
                   f"({out_width} x {out_height} px)")
            log_fn(f"Covered area: {stats['covered_area_ha']:.2f} ha, "
                   f"max overlap: {stats['max_overlap']} frame(s), "
                   f"mean overlap: {stats['mean_overlap']:.2f}")
            if "survey_area_coverage_fraction" in stats:
                log_fn(f"Survey area covered: "
                       f"{stats['survey_area_coverage_fraction'] * 100:.1f}% "
                       f"of {area.adjusted_m2 / 10000.0:.2f} ha")
        if progress_fn:
            progress_fn(100)

    # ------------------------------------------------------------------ #
    # Distance sampling
    # ------------------------------------------------------------------ #

    def run_distance_sampling(self, config: Dict[str, Any], progress_fn=None,
                              log_fn=None, cancel_check=None):
        """Fit a detection function - once, or once per species (§8.2).

        A stratum too thin to fit is reported and skipped rather than failing
        the whole run.
        """
        source = config.get("ds_source", "detections")
        self._per_stratum(
            config, source,
            lambda survey, label, names, progress:
                self._run_distance_sampling_once(
                    config, survey, label, names, progress, log_fn,
                    cancel_check),
            "Distance sampling", progress_fn, log_fn, cancel_check,
            tolerate_failures=True)

    def _run_distance_sampling_once(self, config: Dict[str, Any], survey,
                                    label: str = "", names=None,
                                    progress_fn=None, log_fn=None,
                                    cancel_check=None):
        """Estimate density/abundance via conventional line-transect distance sampling.

        Uses the perpendicular distances already computed by 'Calculate
        Perpendicular' (detections) or 'Calculate Track Perpendicular' (tracks),
        fits half-normal and hazard-rate detection functions by maximum
        likelihood, selects the best by AIC, and reports the effective strip
        width, detection probability, density and abundance with 95% CIs.

        :param config: Configuration dictionary. Relevant keys:
            ``ds_source`` ("detections"|"tracks"), ``ds_truncation`` (m or 0/None
            for automatic 95th-percentile truncation).

        The perpendicular distances of every flight of the survey are pooled
        and their flight-route lengths summed into the total effort L,
        assuming a shared detection function.
        """
        import math
        import numpy as np

        source = config.get("ds_source", "detections")
        target_epsg = survey.epsg or config.get("target_epsg", 32633)
        project_folders = survey.flights

        if log_fn:
            log_fn(f"Running distance-sampling estimation ({source}) over "
                   f"{len(project_folders)} flight(s)...")
        if progress_fn:
            progress_fn(5)

        # ---- Pool the perpendicular distances and effort L ---------------- #
        distance_arrays = []
        transect_length = 0.0
        projects_summary = []
        area = self._survey_area(survey, log_fn)
        for i, flight in enumerate(project_folders):
            d, length = self._load_distance_sampling_project(
                flight, survey, source, names, log_fn, area=area)
            distance_arrays.append(d)
            transect_length += length
            summary = {
                "name": flight.name,
                "target_folder": flight.target_folder,
                "n": int(d.size),
                "transect_length_m": length,
            }
            if area is not None:
                summary["survey_area"] = area.describe()
                summary["outside_survey_area"] = getattr(
                    self, "_last_ds_outside_area", 0)
            projects_summary.append(summary)
            if progress_fn:
                progress_fn(5 + int((i + 1) / len(project_folders) * 15))

        distances = (np.concatenate(distance_arrays) if distance_arrays
                     else np.asarray([], dtype=np.float64))
        if distances.size < 2:
            raise RuntimeError(
                "Not enough perpendicular distances for distance sampling.")
        if transect_length <= 0:
            raise RuntimeError("The pooled flight routes have zero length.")

        if progress_fn:
            progress_fn(20)

        # ---- Truncation -------------------------------------------------- #
        w_cfg = config.get("ds_truncation", 0) or 0
        if w_cfg and float(w_cfg) > 0:
            w = float(w_cfg)
        else:
            w = float(np.percentile(distances, 95))
        x = distances[distances <= w]
        n = int(x.size)
        if n < 2:
            raise RuntimeError("Truncation distance leaves too few observations.")

        if log_fn:
            log_fn(f"n={n} observations, transect length L={transect_length:.1f} m, "
                   f"truncation w={w:.2f} m")
        if progress_fn:
            progress_fn(40)

        # ---- Fit detection functions ------------------------------------- #
        models = []
        hn = self._fit_detection_function("half-normal", x, w, log_fn)
        if hn:
            models.append(hn)
        hr = self._fit_detection_function("hazard-rate", x, w, log_fn)
        if hr:
            models.append(hr)
        if not models:
            raise RuntimeError("Detection-function fitting failed for all models.")

        best = min(models, key=lambda m: m["aic"])
        if progress_fn:
            progress_fn(70)

        # ---- Density / abundance ----------------------------------------- #
        esw = best["esw"]          # effective strip half-width (m)
        p = esw / w                # average detection probability
        density_m2 = n / (2.0 * esw * transect_length)
        density_km2 = density_m2 * 1e6
        covered_area_m2 = 2.0 * w * transect_length
        abundance_covered = density_m2 * covered_area_m2  # == n / p

        # Combine encounter-rate (Poisson) and detection-function CVs.
        cv_n = 1.0 / math.sqrt(n)
        cv_esw = best["cv_esw"]
        cv_density = math.sqrt(cv_n * cv_n + cv_esw * cv_esw)
        ci_d = self._lognormal_ci(density_km2, cv_density)
        ci_n = self._lognormal_ci(abundance_covered, cv_density)

        if progress_fn:
            progress_fn(85)

        # ---- Detection-function curve + histogram for plotting ----------- #
        xs = np.linspace(0, w, 60)
        gx = best["g"](xs)
        hist_counts, hist_edges = np.histogram(x, bins=min(20, max(5, n // 5)), range=(0, w))

        # Abundance for the survey plot: density x adjusted area.
        survey_area_m2 = None if area is None else area.adjusted_m2

        notes = (
            "Encounter-rate variance uses a Poisson approximation (CV = 1/sqrt(n)); "
            "abundance is reported for the covered strip area 2*w*L. Multiply density "
            "by your study-area size for a study-area abundance estimate."
        )
        if area is not None:
            notes += (
                " A survey area was applied: observations outside the plot or "
                "inside an exclusion zone were dropped, and the effort L is the "
                "length of flight route inside the area.")
        if len(project_folders) > 1:
            notes += (
                " Several flights were pooled: their perpendicular distances "
                "were combined and their flight-route lengths summed into the "
                "total effort L, assuming a shared detection function."
            )

        result = {
            "source": source,
            "crs": f"EPSG:{target_epsg}",
            "n": n,
            "n_before_truncation": int(distances.size),
            "n_projects": len(project_folders),
            "flights": [f.name for f in project_folders],
            "projects": projects_summary,
            "transect_length_m": transect_length,
            "truncation_m": w,
            "best_model": best["name"],
            "effective_strip_width_m": esw,
            "detection_probability": p,
            "density_per_km2": density_km2,
            "density_ci95": ci_d,
            "cv_density": cv_density,
            "covered_area_km2": covered_area_m2 / 1e6,
            "abundance_in_covered_area": abundance_covered,
            "abundance_ci95": ci_n,
            "survey_area_km2": (None if survey_area_m2 is None
                                else survey_area_m2 / 1e6),
            "abundance_in_survey_area": (
                None if survey_area_m2 is None
                else density_m2 * survey_area_m2),
            "abundance_survey_area_ci95": (
                None if survey_area_m2 is None
                else self._lognormal_ci(density_m2 * survey_area_m2,
                                        cv_density)),
            "models": [
                {
                    "name": m["name"],
                    "params": m["params"],
                    "log_likelihood": m["log_likelihood"],
                    "aic": m["aic"],
                    "esw_m": m["esw"],
                }
                for m in models
            ],
            "detection_function_curve": {"x": xs.tolist(), "g": gx.tolist()},
            "distance_histogram": {
                "counts": hist_counts.tolist(),
                "edges": hist_edges.tolist(),
            },
            "notes": notes,
        }

        analytics_folder = survey.ensure_analytics_folder()
        stem = f"distance_sampling_{source}"
        if label:
            stem += f"_{self.species_slug(label)}"
            result["species"] = label
        out_file = os.path.join(analytics_folder, f"{stem}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        if log_fn:
            if len(project_folders) > 1:
                log_fn(f"Pooled {len(project_folders)} flights: "
                       f"{int(distances.size)} observations, "
                       f"total L={transect_length:.1f} m")
            log_fn(f"Best model: {best['name']} (AIC={best['aic']:.2f})")
            log_fn(f"ESW={esw:.2f} m, detection probability p={p:.3f}")
            log_fn(f"Density={density_km2:.3f} /km^2 "
                   f"(95% CI {ci_d[0]:.3f}-{ci_d[1]:.3f})")
            log_fn(f"Abundance in covered area ({covered_area_m2 / 1e6:.3f} km^2): "
                   f"{abundance_covered:.1f} (95% CI {ci_n[0]:.1f}-{ci_n[1]:.1f})")
            if survey_area_m2 is not None:
                ci_a = result["abundance_survey_area_ci95"]
                log_fn(f"Abundance in survey area ({survey_area_m2 / 1e6:.3f} "
                       f"km^2): {density_m2 * survey_area_m2:.1f} "
                       f"(95% CI {ci_a[0]:.1f}-{ci_a[1]:.1f})")
            log_fn(f"Distance-sampling results saved to: {out_file}")
        if progress_fn:
            progress_fn(100)

    def _load_distance_sampling_project(self, flight, survey, source,
                                        names=None, log_fn=None, area=None):
        """Perpendicular distances and flight-route length L for one flight.

        Returns ``(distances, transect_length_m)`` - a numpy array of the
        finite, non-negative perpendicular distances and the flight-route
        length in metres. Errors name the flight so a pooled run can say
        which one is at fault.

        With a survey *area*, observations whose position lies outside it are
        dropped and L is only the route length inside it - effort and
        observations must describe the same region.
        """
        import numpy as np
        from .core.survey import route_suffix

        target_folder = flight.target_folder
        m = survey.modality
        if source == "tracks":
            perp_name = f"perpendicular_tracks_{m}.json"
            list_key, dist_key = "tracks", "distance_m"
            prereq = "Calculate Track Perpendicular"
        else:
            perp_name = f"perpendicular_{m}.json"
            list_key, dist_key = "perpendiculas", "distance_m"
            prereq = "Calculate Perpendicular"
        fr_suffix = route_suffix(target_folder, m, perp_name) or m
        perp_file = os.path.join(
            target_folder, f"flight_route_{fr_suffix}", perp_name)

        if not os.path.exists(perp_file):
            raise FileNotFoundError(
                f"{flight.name}: {perp_name} not found. "
                f"Please run '{prereq}' for the {survey.camera_label} camera "
                "first.")
        with open(perp_file, 'r', encoding='utf-8') as f:
            perp_data = json.load(f)
        entries = [e for e in perp_data.get(list_key, []) if dist_key in e]
        species_ids = survey.species_ids(flight, names)
        if species_ids is not None:
            # class_id is the resolved species for anything the store wrote;
            # an entry without one predates that and cannot be attributed.
            wanted = set(species_ids)
            entries = [e for e in entries if e.get("class_id") in wanted]
        self._last_ds_outside_area = 0
        if area is not None:
            def _inside(entry):
                centre = entry.get("detection_center") or []
                if len(centre) < 2:
                    return False  # cannot be placed, so cannot be counted
                return area.contains(float(centre[0]), float(centre[1]))
            before = len(entries)
            entries = [e for e in entries if _inside(e)]
            self._last_ds_outside_area = before - len(entries)
        distances = np.asarray(
            [float(e[dist_key]) for e in entries], dtype=np.float64)
        distances = distances[np.isfinite(distances) & (distances >= 0)]

        route_folder = route_suffix(target_folder, m) or fr_suffix
        route_file = os.path.join(
            target_folder, f"flight_route_{route_folder}",
            "flight_route.geojson")
        if not os.path.exists(route_file):
            raise FileNotFoundError(
                f"{flight.name}: flight_route.geojson not found. "
                "Please run 'Generate Flight Route' first.")
        with open(route_file, 'r', encoding='utf-8') as f:
            route_geojson = json.load(f)
        route_coords = None
        for feature in route_geojson.get("features", []):
            if feature.get("geometry", {}).get("type") == "LineString":
                route_coords = feature["geometry"]["coordinates"]
                break
        if not route_coords or len(route_coords) < 2:
            raise RuntimeError(
                f"The flight route of {flight.name} does not contain a valid "
                "LineString.")
        rc = np.asarray(route_coords, dtype=np.float64)
        seg = np.diff(rc[:, :2], axis=0)
        transect_length = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        if transect_length <= 0:
            raise RuntimeError(
                f"The flight route of {flight.name} has zero length.")
        if area is not None:
            full_length = transect_length
            transect_length = area.clipped_length(rc[:, :2].tolist())
            if transect_length <= 0:
                raise RuntimeError(
                    f"The flight route of {flight.name} never enters the "
                    "survey area.")
            if log_fn:
                log_fn(f"  Survey area: {self._last_ds_outside_area} "
                       f"observation(s) outside dropped; route inside "
                       f"{transect_length:.1f} of {full_length:.1f} m")

        if log_fn:
            log_fn(f"  {flight.name}: {int(distances.size)} distance(s), "
                   f"L={transect_length:.1f} m")
        return distances, transect_length

    def _fit_detection_function(self, name, x, w, log_fn=None):
        """Fit a detection function by MLE; returns a dict of results or None.

        Supported: "half-normal" (1 param sigma) and "hazard-rate" (sigma, b).
        The observed-distance likelihood is f(x) = g(x) / mu, with
        mu = integral of g over [0, w] (the effective strip half-width, ESW).
        """
        import math
        import numpy as np

        n = x.size
        # NumPy 2.0 renamed ``trapz`` to ``trapezoid`` and 2.x removed the alias.
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        def esw_of(g):
            xs = np.linspace(0.0, w, 512)
            return float(_trapz(g(xs), xs))

        if name == "half-normal":
            def make_g(theta):
                sigma = math.exp(theta[0])
                two_var = 2.0 * sigma * sigma
                return lambda t: np.exp(-(np.asarray(t, dtype=np.float64) ** 2) / two_var)

            def nll(theta):
                sigma = math.exp(theta[0])
                g = make_g(theta)
                mu = esw_of(g)
                if mu <= 0:
                    return 1e12
                return float(np.sum(x * x) / (2.0 * sigma * sigma) + n * math.log(mu))

            theta0 = [math.log(max(np.std(x), 1.0))]
            k = 1
        elif name == "hazard-rate":
            def make_g(theta):
                sigma = math.exp(theta[0])
                b = 1.0 + math.exp(theta[1])

                def g(t):
                    t = np.asarray(t, dtype=np.float64)
                    out = np.ones_like(t)
                    nz = t > 0
                    out[nz] = 1.0 - np.exp(-((t[nz] / sigma) ** (-b)))
                    return out
                return g

            def nll(theta):
                g = make_g(theta)
                mu = esw_of(g)
                if mu <= 0:
                    return 1e12
                gx = g(x)
                gx = np.clip(gx, 1e-12, 1.0)
                return float(-np.sum(np.log(gx)) + n * math.log(mu))

            theta0 = [math.log(max(np.std(x), 1.0)), 0.0]
            k = 2
        else:
            return None

        try:
            from scipy.optimize import minimize
            res = minimize(nll, theta0, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
            theta = res.x
            best_nll = float(res.fun)
        except ImportError:
            if name != "half-normal":
                return None  # hazard-rate needs an optimizer
            # Closed-form-ish 1D search for the half-normal.
            grid = np.log(np.linspace(max(np.std(x), 0.5), w * 2 + 1, 400))
            vals = [nll([t]) for t in grid]
            theta = [grid[int(np.argmin(vals))]]
            best_nll = float(min(vals))

        g = make_g(theta)
        esw = esw_of(g)
        if not math.isfinite(best_nll) or esw <= 0:
            return None

        # CV of ESW via a numeric Hessian of the NLL (observed information).
        cv_esw = self._cv_esw_numeric(nll, esw_of, make_g, theta)
        aic = 2.0 * k + 2.0 * best_nll

        if name == "half-normal":
            params = {"sigma": math.exp(theta[0])}
        else:
            params = {"sigma": math.exp(theta[0]), "b": 1.0 + math.exp(theta[1])}

        if log_fn:
            log_fn(f"  {name}: ESW={esw:.2f} m, AIC={aic:.2f}, params={params}")

        return {
            "name": name, "params": params, "log_likelihood": -best_nll,
            "aic": aic, "esw": esw, "cv_esw": cv_esw, "g": g,
        }

    @staticmethod
    def _cv_esw_numeric(nll, esw_of, make_g, theta):
        """Delta-method CV of the ESW from a numeric Hessian of the NLL."""
        import math
        import numpy as np

        def perturb(base, idx, delta):
            out = base.copy()
            out[idx] += delta
            return out

        theta = np.asarray(theta, dtype=np.float64)
        m = theta.size
        h = 1e-4 * (np.abs(theta) + 1.0)

        # Hessian of the NLL (observed information matrix).
        hess = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                tpp = perturb(perturb(theta, i, h[i]), j, h[j])
                tpm = perturb(perturb(theta, i, h[i]), j, -h[j])
                tmp = perturb(perturb(theta, i, -h[i]), j, h[j])
                tmm = perturb(perturb(theta, i, -h[i]), j, -h[j])
                numer = nll(tpp.tolist()) - nll(tpm.tolist()) - nll(tmp.tolist()) + nll(tmm.tolist())
                hess[i, j] = hess[j, i] = numer / (4.0 * h[i] * h[j])
        try:
            cov = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return 0.0
        if not np.all(np.isfinite(cov)):
            return 0.0

        # Gradient of ESW w.r.t. theta (numeric).
        grad = np.zeros(m)
        esw0 = esw_of(make_g(theta.tolist()))
        for i in range(m):
            tp = perturb(theta, i, h[i])
            tm = perturb(theta, i, -h[i])
            d_esw = esw_of(make_g(tp.tolist())) - esw_of(make_g(tm.tolist()))
            grad[i] = d_esw / (2.0 * h[i])
        var_esw = float(grad @ cov @ grad)
        if var_esw <= 0 or not math.isfinite(var_esw) or esw0 <= 0:
            return 0.0
        return math.sqrt(var_esw) / esw0

    @staticmethod
    def _lognormal_ci(estimate, cv, z=1.96):
        """95% lognormal confidence interval for a positive estimate."""
        import math

        if estimate <= 0 or cv <= 0 or not math.isfinite(cv):
            return [estimate, estimate]
        c = math.exp(z * math.sqrt(math.log(1.0 + cv * cv)))
        return [estimate / c, estimate * c]

    # ------------------------------------------------------------------ #
    # Survey analytics: transect-based population estimation
    # ------------------------------------------------------------------ #

    def run_population_estimation(self, config: Dict[str, Any],
                                  progress_fn=None, log_fn=None,
                                  cancel_check=None):
        """Estimate a population - once, or once per species (§8.2)."""
        self._per_stratum(
            config, "tracks",
            lambda survey, label, names, progress:
                self._run_population_estimation_once(
                    config, survey, label, names, progress, log_fn,
                    cancel_check),
            "Population estimation", progress_fn, log_fn, cancel_check)

    def _run_population_estimation_once(self, config: Dict[str, Any],
                                        survey, label: str = "", names=None,
                                        progress_fn=None, log_fn=None,
                                        cancel_check=None):
        """Estimate population density from the transects of a split flight.

        Builds the per-transect count/area table the R analysis of Praschl et
        al. 2026 works on and runs the naive, bootstrap and ZINB estimators on
        it (see :mod:`core.population`):

        * **count** - every track is assigned to the transect whose centre
          line (the flight path between its start and end frame) is nearest in
          perpendicular distance, optionally truncated at ``pop_truncation``.
        * **area** - the union of the per-frame field-of-view footprints of
          the frames inside the transect's frame range, in hectares.

        The optional *study area* is what the densities are extrapolated to an
        abundance for, and it is a different quantity from the monitored area:
        the latter is the searched strip (the density's denominator), the
        former the region the density is assumed to hold over. Setting the
        study area to the monitored area therefore just returns the counted
        animals. ``pop_study_area_auto`` fills it with the flight's total FoV
        coverage - the union of *every* frame's footprint, which unlike the
        summed transect areas counts ground seen by two transects only once
        and also includes frames belonging to no transect.

        Prerequisites: the Transect Splitting Tool (``transects_{m}``), the
        FoV step (``fov_{m}/fov_polygons.txt``) and 'Calculate Track
        Perpendicular' (``flight_route_{fr}/perpendicular_tracks_{m}.json``).

        :param config: Configuration dictionary. Relevant keys:
            ``pop_truncation`` (m, 0 = keep all),
            ``pop_methods`` (list of "naive"/"bootstrap"/"zinb"),
            ``pop_n_boot``, ``pop_seed``, ``pop_study_area_auto`` (bool),
            ``pop_study_area_survey`` (bool), ``pop_study_area_ha``
            (manual, 0 = no abundance).

        Every flight's transects are pooled into one count/area table and the
        estimators run over all of them together; each flight is placed with
        its own DEM georeferencing.
        """
        from .core.population import estimate_population

        target_epsg = survey.epsg or config.get("target_epsg", 32633)
        suffix = survey.modality
        truncation = float(config.get("pop_truncation", 0.0) or 0.0)
        methods = list(config.get("pop_methods") or ["naive", "bootstrap", "zinb"])
        n_boot = int(config.get("pop_n_boot", 999) or 999)
        seed = int(config.get("pop_seed", 42) or 42)
        study_area_auto = bool(config.get("pop_study_area_auto", False))
        study_area_from_plot = bool(config.get("pop_study_area_survey", False))
        study_area_ha = float(config.get("pop_study_area_ha", 0.0) or 0.0)

        entries = survey.flights
        multi = survey.several
        camera_label = survey.camera_label

        if log_fn:
            log_fn(f"Running transect population estimation ({camera_label}) "
                   f"over {len(entries)} flight(s)...")
        area = self._survey_area(survey, log_fn)
        if progress_fn:
            progress_fn(3)

        combined_rows = []
        combined_usable = []
        total_flight_area_ha = 0.0
        agg = {"n_tracks": 0, "n_assigned": 0, "n_truncated": 0,
               "n_outside": 0, "n_without_area": 0, "mismatched": 0,
               "n_outside_survey_area": 0}
        projects_summary = []

        for pi, flight in enumerate(entries):
            if cancel_check and cancel_check():
                raise _cancelled("Population estimation cancelled")

            def _proj_progress(pct, _pi=pi):
                if progress_fn:
                    span = 68.0 / len(entries)
                    progress_fn(5 + int((_pi + pct / 100.0) * span))

            proj = self._population_project_table(
                flight, survey, names, area, config,
                log_fn, cancel_check, _proj_progress)

            proj_name = flight.name
            for r in proj["rows"]:
                row = dict(r)
                row["project"] = proj_name
                if multi:
                    row["name"] = f"{proj_name} / {r['name']}"
                combined_rows.append(row)
                if row["area_ha"] > 0:
                    combined_usable.append(row)

            total_flight_area_ha += proj["flight_area_ha"]
            for key in agg:
                agg[key] += proj[key]
            projects_summary.append({
                "target_folder": flight.target_folder,
                "name": proj_name,
                "n_transects": len(proj["usable"]),
                "n_transects_total": len(proj["rows"]),
                "total_count": proj["total_count"],
                "total_ha": proj["total_ha"],
                "flight_fov_area_ha": proj["flight_area_ha"],
                "dem_origin_source": proj["origin_source"],
                "n_tracks_outside_survey_area": proj["n_outside_survey_area"],
            })

        if not combined_usable:
            raise RuntimeError(
                "No transect has a monitored area in any flight. Check that "
                "the FoV step covered the transects' frame ranges.")

        # ---- Study area (region the density is extrapolated to) ------------ #
        study_area_source = "none"
        if study_area_from_plot:
            plot_m2 = None if area is None else area.adjusted_m2
            if not plot_m2:
                raise RuntimeError(
                    "'Use survey area' needs a survey plot. Draw one on the "
                    "Survey Analytics tab (and tick 'Limit analytics to the "
                    "survey area'), or choose another study area.")
            study_area_ha = plot_m2 / 10000.0
            study_area_source = "survey_area"
            if log_fn:
                log_fn(f"Study area from the survey plot(s) minus exclusion "
                       f"zones: {study_area_ha:.2f} ha")
        elif study_area_auto:
            study_area_ha = total_flight_area_ha
            study_area_source = "flight_fov"
            if total_flight_area_ha <= 0:
                raise RuntimeError(
                    "The flights have no FoV coverage, so no study area could "
                    "be derived. Untick 'Use flight FoV area' or check the FoV "
                    "step.")
            if log_fn:
                log_fn(f"Study area from the flights' total FoV coverage: "
                       f"{total_flight_area_ha:.2f} ha")
        elif study_area_ha > 0:
            study_area_source = "manual"

        # ---- Estimate over the pooled transects ---------------------------- #
        if progress_fn:
            progress_fn(78)
        result = estimate_population(
            combined_usable, methods=methods, n_boot=n_boot, seed=seed,
            study_area_ha=study_area_ha)

        notes = (
            "A track counts towards a transect only when it lies inside that "
            "transect's monitored area (the union of the field-of-view "
            "footprints of the frames in its range); where several transects "
            "cover it, the one whose flight path is nearest in perpendicular "
            "distance wins. "
            "Densities are per 100 ha (= per km²). The monitored area sums the "
            "transects (each is its own sample, so shared ground counts once "
            "per transect), while flight_fov_area_ha unions every frame and "
            "counts shared ground once - they differ where transects overlap "
            "or where frames belong to no transect."
        )
        if multi:
            notes += (
                " Several projects were pooled: every project's transects are "
                "one shared sample set for the estimators, the monitored areas "
                "and flight FoV coverages summed across flights. Each project's "
                "own DEM georeferencing was used to place its transects."
            )

        result.update({
            "camera": camera_label,
            "crs": f"EPSG:{target_epsg}",
            "truncation_m": truncation,
            "n_projects": len(entries),
            "flights": [f.name for f in entries],
            "survey_area": None if area is None else area.describe(),
            "projects": projects_summary,
            "n_tracks": agg["n_tracks"],
            "n_tracks_assigned": agg["n_assigned"],
            "n_tracks_truncated": agg["n_truncated"],
            "n_tracks_outside_fov": agg["n_outside"],
            "n_tracks_outside_survey_area": agg["n_outside_survey_area"],
            "n_transects_without_area": agg["n_without_area"],
            "flight_fov_area_ha": total_flight_area_ha,
            "study_area_source": study_area_source,
            "transects": combined_rows,
            "notes": notes,
        })
        # An estimate without a record of what it counted is unauditable
        # (§8.2). Built from this run's own filter rather than read off the
        # processor, which would describe whichever analytic ran last.
        result["population_filter"] = {
            "modality": suffix,
            "species_filter": None if names is None else sorted(names),
            "species": label or None,
            "manual_run_included": survey.include_manual,
        }

        # Pooled projects assign species ids independently, so a taxonomy
        # difference between them has to be visible: a species missing from one
        # survey looks exactly like one that was present and never seen.
        if multi:
            from .core import analytics_source

            folders = [f.target_folder for f in entries]
            by_folder = {f.target_folder: f.name for f in entries}
            comparison = analytics_source.compare_vocabularies(folders)
            result["species_vocabularies"] = comparison
            if comparison["only_in"] and log_fn:
                for folder, only in comparison["only_in"].items():
                    log_fn(f"Note: {by_folder.get(folder, folder)} has "
                           f"species the others do not: {', '.join(only)}")

        # ---- Write outputs -------------------------------------------------- #
        analytics_folder = survey.ensure_analytics_folder()

        stem = "population_estimate"
        if label:
            stem += f"_{self.species_slug(label)}"
            result["species"] = label
        out_file = os.path.join(analytics_folder, f"{stem}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        # A combined per-transect table (with a project column) alongside the
        # per-project CSVs each project already wrote into its own analytics
        # folder; only needed when more than one project was pooled.
        combined_csv = None
        if multi:
            combined_csv = os.path.join(
                analytics_folder, "population_transects_combined.csv")
            with open(combined_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["project", "id", "name", "start_frame",
                                 "end_frame", "length_m", "area_m2", "area_ha",
                                 "count", "n_frames", "n_frames_with_fov"])
                for r in combined_rows:
                    writer.writerow([
                        r.get("project", ""), r["id"], r["name"],
                        r["start_frame"], r["end_frame"], r["length_m"],
                        r["area_m2"], round(r["area_ha"], 4), r["count"],
                        r["n_frames"], r["n_frames_with_fov"],
                    ])

        if log_fn:
            total_ha = result["total_ha"]
            log_fn(f"Total: {int(result['total_count'])} animals on "
                   f"{total_ha:.2f} ha across {result['n_transects']} transect(s) "
                   f"({result['n_zero_transects']} with zero counts)")
            if agg["mismatched"]:
                log_fn(f"Note: {agg['mismatched']} track(s) were assigned to a "
                       "transect whose frame range does not contain their "
                       "sighting frame (transects running close together).")
            for name, est in result["estimates"].items():
                density = est.get("density_per_100ha")
                if density is None:
                    log_fn(f"  {name}: failed - {est.get('error')}")
                    continue
                ci = est.get("ci95")
                ci_txt = (f" (95% CI {ci[0]:.2f}–{ci[1]:.2f})" if ci else "")
                log_fn(f"  {name}: {density:.2f} animals/100 ha{ci_txt}")
                if est.get("error"):
                    log_fn(f"    warning: {est['error']}")
            log_fn(f"Population estimate saved to: {out_file}")
            if combined_csv:
                log_fn(f"Combined per-transect table saved to: {combined_csv}")

        if progress_fn:
            progress_fn(100)

    def _population_project_table(self, flight, survey, names, area, config,
                                  log_fn=None, cancel_check=None,
                                  progress_cb=None):
        """Build one project's transect count/area table and write its outputs.

        Runs the per-project half of :meth:`run_population_estimation` - loads
        the transects, poses, FoV footprints and track perpendicular distances,
        assigns the tracks and measures each transect's monitored area - then
        writes that project's own ``population_transects.csv``,
        ``population_tracks.csv``, ``transect_areas.geojson`` and
        ``transect_routes.geojson`` into its ``analytics_{m}`` folder. The
        estimation itself runs once on the pooled table in the caller.

        :param flight: a :class:`core.survey.SurveyFlight` - its DEM supplies
            the mesh->world origin of its poses.
        :param names: species names to count (``None`` = all).
        :param area: the survey area, or ``None``.
        :return: a dict with ``rows`` (all transects), ``usable`` (area > 0),
            ``flight_area_ha``, the track-assignment tallies (``n_tracks``,
            ``n_assigned``, ``n_truncated``, ``n_outside``, ``n_without_area``,
            ``mismatched``), ``total_count``/``total_ha`` over the usable
            transects and ``origin_source``.
        """
        from .core import store as _store_kinds
        from .core.pipeline_outputs import load_fov_polygons_3d
        from .core.population import (
            assign_tracks, geometry_to_rings, merged_fov_area,
            shapely_area_predicate, transect_centerline,
        )
        from .core.transects import TransectStore, cumulative_distances, \
            flight_positions, path_length

        from .core.survey import route_suffix

        target_folder = flight.target_folder
        target_epsg = survey.epsg or config.get("target_epsg", 32633)
        suffix = survey.modality
        fr_suffix = route_suffix(target_folder, suffix,
                                 f"perpendicular_tracks_{suffix}.json") or suffix
        truncation = float(config.get("pop_truncation", 0.0) or 0.0)

        if log_fn:
            log_fn(f"Flight {flight.name}:")

        # ---- Transects ---------------------------------------------------- #
        store = TransectStore(target_folder, suffix)
        if not store.load():
            raise FileNotFoundError(
                f"transects_{suffix}/transects.json not found in {target_folder}. "
                "Please define transects with the Transect Splitting Tool first."
            )
        transects = store.ordered()
        if not transects:
            raise RuntimeError(
                f"The transect definition in {target_folder} contains no transects.")

        # ---- Poses (transect centre lines) -------------------------------- #
        poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(
                f"poses_{suffix}.json not found in {target_folder}. Please run "
                "frame extraction first.")
        with open(poses_path, 'r', encoding='utf-8') as f:
            images = json.load(f).get("images", [])
        if not images:
            raise RuntimeError(f"poses_{suffix}.json in {target_folder} has no frames.")
        for transect in transects:
            transect.clamp(len(images))

        # The poses store camera positions mesh-locally; the tracks' geo-referenced
        # positions are in the world CRS, so shift the centre lines by the DEM origin.
        x_offset, y_offset, origin_source = self._resolve_population_origin(
            flight, log_fn)

        centerlines = {
            t.transect_id: transect_centerline(
                images, t.first_frame, t.last_frame, x_offset, y_offset)
            for t in transects
        }
        if progress_cb:
            progress_cb(20)

        # ---- Track perpendicular distances -------------------------------- #
        perp_file = os.path.join(
            target_folder, f"flight_route_{fr_suffix}",
            f"perpendicular_tracks_{suffix}.json")
        if not os.path.exists(perp_file):
            raise FileNotFoundError(
                f"{os.path.basename(perp_file)} not found in {target_folder}. "
                "Please run 'Calculate Track Perpendicular' first."
            )
        with open(perp_file, 'r', encoding='utf-8') as f:
            tracks = json.load(f).get("tracks", [])

        # class_id is the resolved species for anything the store wrote, so
        # the filter is applied here rather than where the file was written -
        # otherwise every species in a per-species run gets the same tracks
        # and the same numbers under a different name.
        species_ids = survey.species_ids(flight, names)
        if species_ids is not None:
            wanted = set(species_ids)
            tracks = [t for t in tracks if t.get("class_id") in wanted]

        # The survey plot: animals outside it (or inside an exclusion zone)
        # are not part of the census, and neither is their ground.
        survey_area = area
        n_outside_survey_area = 0
        if survey_area is not None:
            def _in_area(track):
                centre = track.get("detection_center") or []
                return len(centre) >= 2 and survey_area.contains(
                    float(centre[0]), float(centre[1]))
            before = len(tracks)
            tracks = [t for t in tracks if _in_area(t)]
            n_outside_survey_area = before - len(tracks)
            if log_fn:
                log_fn(f"{n_outside_survey_area} track(s) outside the survey "
                       "area are not counted")

        if not tracks:
            if survey_area is not None and n_outside_survey_area:
                raise RuntimeError(
                    f"Every track in {target_folder} lies outside its survey "
                    "area (or inside an exclusion zone).")
            if species_ids is not None:
                raise RuntimeError(
                    "No tracks with perpendicular distances match the "
                    f"selected species in {target_folder}. Re-run 'Calculate "
                    "Track Perpendicular' if the tracks changed since.")
            raise RuntimeError(
                f"No tracks with perpendicular distances found in {target_folder}.")

        # ---- Monitored area per transect (union of the frame FoVs) --------- #
        # Computed before the assignment: a track only counts towards a
        # transect whose footprint contains it, so the areas are needed first.
        # The path locates the project; load_fov_polygons_3d reads the store
        # beside it, never the text file (see core.pipeline_outputs).
        fov_file = os.path.join(target_folder, f"fov_{suffix}", "fov_polygons.txt")
        self._require_store(target_folder, _store_kinds.FOV, suffix,
                            "Calculate Field of View")
        fov_polygons = load_fov_polygons_3d(fov_file, log_fn)
        if not fov_polygons:
            raise RuntimeError(
                f"No field-of-view polygons stored for {target_folder}. "
                "Run 'Calculate Field of View' again.")

        cum = cumulative_distances(flight_positions(images))
        areas: Dict[int, object] = {}
        area_m2_by_id: Dict[int, float] = {}
        frames_by_id: Dict[int, Tuple[int, int]] = {}
        for i, transect in enumerate(transects):
            if cancel_check and cancel_check():
                raise _cancelled("Population estimation cancelled")

            frames = list(range(transect.first_frame, transect.last_frame + 1))
            area_m2, geometry = merged_fov_area(fov_polygons, frames)
            if survey_area is not None:
                geometry = survey_area.clip(geometry)
                area_m2 = 0.0 if geometry is None else float(geometry.area)
            areas[transect.transect_id] = geometry
            area_m2_by_id[transect.transect_id] = area_m2
            frames_by_id[transect.transect_id] = (
                len(frames), sum(1 for f in frames if f in fov_polygons))
            if area_m2 <= 0 and log_fn:
                log_fn(f"Warning: {transect.display_name} has no FoV footprint "
                       f"(frames {transect.first_frame}-{transect.last_frame}) - "
                       "it contributes a zero area.")
            if progress_cb:
                progress_cb(25 + int((i + 1) / len(transects) * 30))

        # ---- Assign the tracks --------------------------------------------- #
        assignments = assign_tracks(
            tracks, centerlines, truncation,
            contains=shapely_area_predicate(areas))
        counts: Dict[int, int] = {t.transect_id: 0 for t in transects}
        for a in assignments:
            if a["transect_id"] is not None:
                counts[a["transect_id"]] += 1
        if progress_cb:
            progress_cb(65)

        rows = []
        area_features = []
        for transect in transects:
            area_m2 = area_m2_by_id[transect.transect_id]
            n_frames, n_with_fov = frames_by_id[transect.transect_id]
            rows.append({
                "id": transect.transect_id,
                "name": transect.display_name,
                "start_frame": transect.start_frame,
                "end_frame": transect.end_frame,
                "length_m": round(path_length(
                    cum, transect.first_frame, transect.last_frame), 2),
                "area_m2": round(area_m2, 2),
                "area_ha": area_m2 / 10000.0,
                "count": counts[transect.transect_id],
                "n_frames": n_frames,
                "n_frames_with_fov": n_with_fov,
            })
            for ring in geometry_to_rings(areas[transect.transect_id]):
                area_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "transect_id": transect.transect_id,
                        "name": transect.display_name,
                        "area_ha": round(area_m2 / 10000.0, 4),
                        "count": counts[transect.transect_id],
                    },
                })

        usable = [r for r in rows if r["area_ha"] > 0]
        n_assigned = sum(1 for a in assignments if a["transect_id"] is not None)
        n_truncated = sum(1 for a in assignments if a["truncated"])
        n_outside = sum(1 for a in assignments if a["outside_fov"])
        if log_fn:
            log_fn(f"{len(transects)} transect(s), {len(assignments)} track(s): "
                   f"{n_assigned} assigned, {len(assignments) - n_assigned} "
                   f"not counted ({n_outside} outside every transect's field of "
                   f"view, {n_truncated} beyond the truncation distance)")
            if len(usable) < len(rows):
                log_fn(f"Warning: {len(rows) - len(usable)} transect(s) without a "
                       "monitored area were excluded from the estimation.")

        # ---- Flight FoV coverage (candidate study area) -------------------- #
        # The union over *every* frame, so ground seen by two transects counts
        # once - unlike the summed transect areas, which the density needs to
        # double-count because each transect is its own sample.
        if progress_cb:
            progress_cb(75)
        flight_area_m2, _flight_geom = merged_fov_area(
            fov_polygons, sorted(fov_polygons.keys()))
        if survey_area is not None:
            _flight_geom = survey_area.clip(_flight_geom)
            flight_area_m2 = (0.0 if _flight_geom is None
                              else float(_flight_geom.area))
        flight_area_ha = flight_area_m2 / 10000.0

        # ---- Write this project's outputs ---------------------------------- #
        analytics_folder = os.path.join(target_folder, f"analytics_{suffix}")
        os.makedirs(analytics_folder, exist_ok=True)

        transects_csv = os.path.join(analytics_folder, "population_transects.csv")
        with open(transects_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "start_frame", "end_frame", "length_m",
                             "area_m2", "area_ha", "count", "n_frames",
                             "n_frames_with_fov"])
            for r in rows:
                writer.writerow([
                    r["id"], r["name"], r["start_frame"], r["end_frame"],
                    r["length_m"], r["area_m2"], round(r["area_ha"], 4),
                    r["count"], r["n_frames"], r["n_frames_with_fov"],
                ])

        names = {t.transect_id: t.display_name for t in transects}
        ranges = {t.transect_id: (t.first_frame, t.last_frame) for t in transects}
        tracks_csv = os.path.join(analytics_folder, "population_tracks.csv")
        with open(tracks_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "last_frame", "x", "y", "class_id",
                             "transect_id", "transect_name", "distance_m",
                             "nearest_distance_m", "in_frame_range",
                             "truncated", "outside_fov"])
            for a in assignments:
                tid = a["transect_id"]
                rng = ranges.get(tid)
                in_range = ""
                if rng is not None and a["last_frame"] is not None:
                    in_range = int(rng[0] <= a["last_frame"] <= rng[1])
                nearest = a["nearest_distance_m"]
                writer.writerow([
                    a["track_id"], a["last_frame"],
                    round(a["x"], 3), round(a["y"], 3), a["class_id"],
                    "" if tid is None else tid,
                    "" if tid is None else names.get(tid, ""),
                    "" if a["distance_m"] is None else round(a["distance_m"], 3),
                    "" if nearest is None else round(nearest, 3),
                    in_range, int(a["truncated"]), int(a["outside_fov"]),
                ])

        areas_geojson = os.path.join(analytics_folder, "transect_areas.geojson")
        with open(areas_geojson, 'w', encoding='utf-8') as f:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name",
                        "properties": {"name": f"EPSG:{target_epsg}"}},
                "features": area_features,
            }, f, indent=2)

        # The sub-flight route of each transect (start frame -> end frame), so
        # the map layers can show the flown line inside its monitored area.
        by_id = {r["id"]: r for r in rows}
        route_features = []
        for transect in transects:
            line = centerlines.get(transect.transect_id) or []
            if len(line) < 2:
                continue
            row = by_id[transect.transect_id]
            route_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[x, y] for x, y in line],
                },
                "properties": {
                    "transect_id": transect.transect_id,
                    "name": transect.display_name,
                    "start_frame": transect.start_frame,
                    "end_frame": transect.end_frame,
                    "length_m": row["length_m"],
                    "count": row["count"],
                },
            })
        routes_geojson = os.path.join(analytics_folder, "transect_routes.geojson")
        with open(routes_geojson, 'w', encoding='utf-8') as f:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name",
                        "properties": {"name": f"EPSG:{target_epsg}"}},
                "features": route_features,
            }, f, indent=2)

        # Tracks assigned to a transect they were not filmed in usually mean two
        # transects run close together - worth surfacing, not worth failing on.
        def _filmed_elsewhere(a: dict) -> bool:
            tid, frame = a["transect_id"], a["last_frame"]
            if tid is None or frame is None:
                return False
            lo, hi = ranges[tid]
            return not lo <= frame <= hi

        mismatched = sum(1 for a in assignments if _filmed_elsewhere(a))
        if progress_cb:
            progress_cb(100)

        return {
            "target_folder": target_folder,
            "rows": rows,
            "usable": usable,
            "flight_area_ha": flight_area_ha,
            "n_tracks": len(assignments),
            "n_assigned": n_assigned,
            "n_truncated": n_truncated,
            "n_outside": n_outside,
            "n_without_area": len(rows) - len(usable),
            "mismatched": mismatched,
            "n_outside_survey_area": n_outside_survey_area,
            "total_count": sum(r["count"] for r in usable),
            "total_ha": sum(r["area_ha"] for r in usable),
            "origin_source": origin_source,
        }

    def _resolve_population_origin(self, flight, log_fn=None):
        """The mesh->world XY offset for one flight's transect centre lines.

        The poses store camera positions mesh-locally (world CRS minus the DEM
        origin), so the centre lines are shifted back by that origin, read
        from the flight's own DEM metadata. Returns ``(x, y, source)``.
        """
        from .core.pipeline_outputs import read_dem_origin_xy

        dem_path = (flight.dem_path or "").strip()
        dem_json = (flight.dem or "").strip()
        if not dem_path and not dem_json:
            raise FileNotFoundError(
                f"No DEM metadata (dem.json) known for flight {flight.name}. "
                "Set the DEM on the flight's Processing tab so its transects "
                "can be georeferenced.")
        if dem_json and not os.path.isfile(dem_json) and not dem_path:
            raise FileNotFoundError(
                f"DEM metadata of flight {flight.name} not found: {dem_json}")
        x_offset, y_offset = read_dem_origin_xy(dem_path, dem_json)
        if log_fn:
            log_fn(f"  DEM origin of {flight.name}: "
                   f"({x_offset:.1f}, {y_offset:.1f})")
        return x_offset, y_offset, "dem.json" if dem_json else "dem"
