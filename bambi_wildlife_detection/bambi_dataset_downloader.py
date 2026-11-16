# -*- coding: utf-8 -*-
"""Flights of the BAMBI dataset, made ready off the main thread (see
:mod:`core.bambi_dataset`).

:class:`DatasetFlightsWorker` takes the flights that were ticked, one after
the other: fetches a flight's recordings into a folder of its own - or,
without the videos, only its annotations - builds its elevation model from
the open worldwide models - the terrain model GEM-15 where it has the
flight, else the Copernicus DEM
(:mod:`copernicus_dem_downloader`) - and, as asked, takes its published
route and labels out of the labelled dataset. A flight that fails does not
cost the others: what went wrong is kept beside what is ready.
"""

import logging
import os
from typing import List, Sequence, Tuple

from qgis.PyQt.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

CANCELLED = "Cancelled"


class DatasetFlightsWorker(QObject):
    """Makes flights of the BAMBI dataset ready in a folder."""

    finished = pyqtSignal(bool, str)    # success, folder or message
    progress = pyqtSignal(int)          # of all the flights, in percent
    log = pyqtSignal(str)
    flight_started = pyqtSignal(int, int, int)   # flight, which one, of how many

    def __init__(self, flights: Sequence, folder: str,
                 route: bool = True, labels: bool = True,
                 correction: bool = True, videos: bool = True,
                 with_dem: bool = True, version: str = "processed"):
        super().__init__()
        #: The flights to fetch (``core.bambi_dataset.Flight``), as the
        #: dataset's repository lists them.
        self.flights = list(flights)
        self.videos = bool(videos)
        #: Which video: the labelled dataset's processed one, or "raw".
        self.version = version
        self.folder = folder
        self.route = bool(route)
        self.labels = bool(labels)
        self.correction = bool(correction)
        self.with_dem = bool(with_dem)
        #: What is ready (``core.bambi_dataset.prepare_flight``), in the
        #: order asked for - also after a cancel or a failure.
        self.prepared: List[dict] = []
        #: ``(flight, why)`` of the flights that could not be made ready.
        self.failures: List[Tuple[int, str]] = []
        self._cancelled = False

    def cancel(self):
        """Stop at the next megabyte; what arrived is kept."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _session(self):
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": "bambi-qgis-plugin"})
        return session

    def run(self):
        from .core import (bambi_dataset, checked_download, environment,
                           remote_zip)

        stopped = (checked_download.Cancelled, environment.Cancelled,
                   remote_zip.Cancelled)
        count = len(self.flights)
        # what Zenodo says a record holds: asked once for all its flights
        listings = {}
        try:
            session = self._session()
            for index, entry in enumerate(self.flights):
                flight_id = entry.id
                if self._cancelled:
                    raise checked_download.Cancelled("cancelled")
                self.flight_started.emit(flight_id, index + 1, count)
                self.progress.emit(int(100.0 * index / max(count, 1)))
                try:
                    self.log.emit(
                        f"--- {entry.name} ({index + 1} of {count}) ---")
                    self.prepared.append(bambi_dataset.prepare_flight(
                        entry, os.path.join(self.folder, entry.folder),
                        session, route=self.route, labels=self.labels,
                        correction=self.correction, videos=self.videos,
                        version=self.version, listings=listings,
                        elevation=self._elevation if self.with_dem else None,
                        make_route=self._route_files,
                        progress=self._progress_of(index, count),
                        cancelled=self.is_cancelled, log=self.log.emit))
                except stopped:
                    raise
                except bambi_dataset.DatasetError as exc:
                    self.failures.append((flight_id, str(exc)))
                    self.log.emit(f"Flight {flight_id}: {exc}")
                except Exception as exc:  # noqa: BLE001 - the next flight, then
                    import traceback

                    logger.error("Flight %s of the dataset: %s", flight_id,
                                 exc)
                    traceback.print_exc()
                    self.failures.append((flight_id, str(exc)))
                    self.log.emit(f"Flight {flight_id} could not be made "
                                  f"ready: {exc}")
        except stopped:
            self.finished.emit(False, CANCELLED)
            return
        except Exception as exc:  # noqa: BLE001 - told to the user, whole
            import traceback

            logger.error("Flights of the dataset: %s", exc)
            traceback.print_exc()
            self.finished.emit(
                False, f"The flights could not be made ready: {exc}")
            return
        self.progress.emit(100)
        if self.prepared:
            self.finished.emit(True, self.folder)
            return
        why = "\n".join(f"Flight {flight_id}: {reason}"
                        for flight_id, reason in self.failures)
        self.finished.emit(False, why or "No flight was asked for.")

    def _progress_of(self, index: int, count: int):
        """A flight's own percent, as percent of all the flights."""
        def report(value: int) -> None:
            done = index + min(max(value, 0), 100) / 100.0
            self.progress.emit(int(100.0 * done / max(count, 1)))
        return report

    def _elevation(self, bounds, folder: str, crs: str, log=None,
                   progress=None, cancelled=None) -> dict:
        """The flight's elevation model, beside its recordings."""
        from .copernicus_dem_downloader import build_dem

        return build_dem(bounds, folder, crs, log=log, progress=progress,
                         cancelled=cancelled)

    def _route_files(self, prepared: dict) -> None:
        """What 'Generate Flight Route' writes: the line the drone flew,
        from its flight log, and a point per published frame."""
        from .bambi_processing import BambiProcessor

        files = prepared["files"]
        BambiProcessor().run_flight_route({
            "flight_route_camera": "T",
            "target_folder": prepared["target_folder"],
            "target_epsg": prepared["epsg"],
            "airdata_path": files.get("airdata") or "",
            "alfs_dem_metadata_path": prepared["dem_metadata"],
            "dem_path": prepared["dem"],
            "filter_gps_origin": True,
        }, log_fn=self.log.emit)
