# -*- coding: utf-8 -*-
"""The demo flight, made ready off the main thread (see
:mod:`core.demo_data`).

:class:`DemoFlightWorker` fetches the published example flight into a
folder, unpacks it, and - so that nothing is left to do before the first
step - builds its elevation model from the open worldwide models: the
terrain model GEM-15 where it has the flight, else the Copernicus DEM
(:mod:`copernicus_dem_downloader`).
"""

import logging
import os

from qgis.PyQt.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class DemoFlightWorker(QObject):
    """Makes the demo flight ready in a folder."""

    finished = pyqtSignal(bool, str)    # success, folder or message
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, folder: str, with_dem: bool = True):
        super().__init__()
        self.folder = folder
        self.with_dem = with_dem
        #: Why the elevation model is missing although the flight is there.
        self.dem_problem = ""
        self._cancelled = False

    def cancel(self):
        """Stop at the next megabyte; what arrived is kept."""
        self._cancelled = True

    def _session(self):
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": "bambi-qgis-plugin"})
        return session

    def run(self):
        from .core import checked_download, demo_data, environment

        try:
            files = demo_data.prepare(
                self.folder, self._session(),
                progress=lambda value: self.progress.emit(int(value * 0.85)),
                cancelled=lambda: self._cancelled, log=self.log.emit)
            if self.with_dem:
                self._elevation(files)
        except (checked_download.Cancelled, environment.Cancelled):
            self.finished.emit(False, "Cancelled")
            return
        except demo_data.DemoError as exc:
            self.log.emit(str(exc))
            self.finished.emit(False, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - told to the user, whole
            import traceback

            logger.error("Demo flight: %s", exc)
            traceback.print_exc()
            self.finished.emit(
                False, f"The demo flight could not be made ready: {exc}")
            return
        self.progress.emit(100)
        self.finished.emit(True, self.folder)

    def _elevation(self, files) -> None:
        """The flight's elevation model, unless the folder has one. A
        model that cannot be fetched does not cost the flight: it is said,
        and can be fetched from the Input tab later."""
        from .copernicus_dem_downloader import STEMS, build_dem
        from .core import demo_data, environment
        from .core.crs_utils import utm_epsg_for_position

        if any(os.path.isfile(os.path.join(self.folder, f"{stem}.glb"))
               for stem in STEMS.values()):
            return
        bounds = demo_data.recording_bounds(files["airdata"])
        if bounds is None:
            self.dem_problem = "the flight log holds no position"
            return
        zone = utm_epsg_for_position((bounds[1] + bounds[3]) / 2.0,
                                     (bounds[0] + bounds[2]) / 2.0)
        try:
            build_dem(
                bounds, self.folder, f"EPSG:{zone}", log=self.log.emit,
                progress=lambda value: self.progress.emit(
                    85 + int(value * 0.14)),
                cancelled=lambda: self._cancelled)
        except environment.Cancelled:
            raise
        except environment.EnvironmentFailure as exc:
            self.dem_problem = str(exc)
            self.log.emit(
                f"The elevation model could not be fetched ({exc}). The "
                "flight is there all the same: fetch the model with "
                "'Download DEM (worldwide)' on the Input tab.")
