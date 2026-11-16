# -*- coding: utf-8 -*-
"""Input > Download Example Flights: flights of the BAMBI dataset, by their
number.

The row at the top of the Input tab; the window in which the flights are
ticked (``bambi_dataset_dialog``); the download
(``bambi_dataset_downloader``), which the same button stops; and what
happens to the flights that arrived: each becomes a flight of the project
with its recordings, elevation model, CRS and - where it was asked for -
its published correction filled in, so that the first step can be run at
once.
"""

import os
from typing import List

from qgis.PyQt.QtCore import QSettings, QThread
from qgis.PyQt.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
)

from ..core import bambi_dataset

#: Where the flights were put the last time - offered again.
FOLDER_SETTING = "bambi/example_flights/folder"
BUTTON_TEXT = "Download Example Flights…"
TITLE = "Example Flights"
TOOLTIP = (
    "Fetch flights of the BAMBI dataset from Zenodo (CC BY 4.0) - tick them "
    "by their number in a list; a short one of 122 MB is ticked to begin "
    "with - each with its elevation model, and add them to this project, to "
    "try every step without data of your own. A flight's published route, "
    "its hand-labelled animals and its camera correction come along if you "
    "ask for them.")
#: More routes than this are not put on the map by themselves.
ROUTES_SHOWN = 25


class ExampleFlightsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_example_flights_row(self) -> QHBoxLayout:
        """The row that offers the flights of the BAMBI dataset."""
        row = QHBoxLayout()
        hint = QLabel("New here?")
        hint.setStyleSheet("color: gray;")
        row.addWidget(hint)
        self.examples_btn = QPushButton(BUTTON_TEXT)
        self.examples_btn.setToolTip(TOOLTIP)
        self.examples_btn.clicked.connect(self.download_example_flights)
        row.addWidget(self.examples_btn)
        row.addStretch()
        return row

    def _examples_running(self) -> bool:
        return bool(getattr(self, "_examples_busy", False))

    def download_example_flights(self):
        """Ask which flights to fetch, and fetch them - or, while they are
        being fetched, stop."""
        if self._examples_running():
            self._cancel_example_flights()
            return
        if self._step_is_running():
            QMessageBox.information(
                self, TITLE, "A step is running. Wait for it to finish first.")
            return
        try:
            flights = bambi_dataset.flights()
        except bambi_dataset.DatasetError as exc:
            QMessageBox.warning(self, TITLE, str(exc))
            return

        from ..bambi_dataset_dialog import DatasetFlightsDialog

        dialog = DatasetFlightsDialog(
            flights, self,
            folder=QSettings().value(FOLDER_SETTING, "", type=str))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen, folder = dialog.selected_ids(), dialog.folder()
        if not chosen or not folder:
            return
        QSettings().setValue(FOLDER_SETTING, folder)
        self._start_example_flights(
            chosen, folder, dialog.import_route(), dialog.import_labels(),
            dialog.use_correction())

    def _start_example_flights(self, chosen: List[int], folder: str,
                               route: bool, labels: bool,
                               correction: bool) -> None:
        from ..bambi_dataset_downloader import DatasetFlightsWorker

        self._examples_busy = True
        self._examples_flight = ""
        self.examples_btn.setText("Cancel Download")
        self.examples_btn.setToolTip(
            "Stop fetching. What arrived is kept: flights that are ready "
            "are added to the project, and a download that broke off is "
            "continued the next time.")
        self.examples_thread = QThread()
        self.examples_worker = DatasetFlightsWorker(
            chosen, folder, route=route, labels=labels, correction=correction)
        self.examples_worker.moveToThread(self.examples_thread)
        self.examples_thread.started.connect(self.examples_worker.run)
        self.examples_worker.finished.connect(self._on_examples_finished)
        self.examples_worker.progress.connect(self._on_examples_progress)
        self.examples_worker.flight_started.connect(
            self._on_example_flight_started)
        self.examples_worker.log.connect(self.log)
        self.examples_worker.finished.connect(self.examples_thread.quit)
        self.examples_worker.finished.connect(self.examples_worker.deleteLater)
        self.examples_thread.finished.connect(self.examples_thread.deleteLater)
        what = [name for name, wanted in (
            ("route", route), ("labels", labels), ("correction", correction))
            if wanted]
        with_what = f" with their {', '.join(what)}" if what else ""
        self.log(f"Fetching {len(chosen)} flight(s) of the BAMBI dataset"
                 f"{with_what} into {folder}...")
        self.examples_thread.start()

    def _cancel_example_flights(self) -> None:
        self.examples_worker.cancel()
        self.examples_btn.setEnabled(False)
        self.examples_btn.setText("Stopping...")
        self.log("Stopping the download...")

    def _on_example_flight_started(self, flight_id: int, index: int,
                                   count: int):
        self._examples_flight = (f"flight {flight_id}" if count == 1 else
                                 f"flight {flight_id}, {index} of {count}")

    def _on_examples_progress(self, percent: int):
        if not self.examples_btn.isEnabled():
            return                       # stopping
        which = getattr(self, "_examples_flight", "")
        about = f"{which}: " if which else ""
        self.examples_btn.setText(f"Cancel Download ({about}{percent}%)")

    def _on_examples_finished(self, success: bool, message: str):
        """The flights are there - or some of them, or none."""
        from ..bambi_dataset_downloader import CANCELLED

        self._examples_busy = False
        self.examples_btn.setEnabled(True)
        self.examples_btn.setText(BUTTON_TEXT)
        self.examples_btn.setToolTip(TOOLTIP)
        worker = getattr(self, "examples_worker", None)
        prepared = list(getattr(worker, "prepared", None) or [])
        failures = list(getattr(worker, "failures", None) or [])

        added = self._add_example_flights(prepared) if prepared else []
        lines = []
        if added:
            names = ", ".join(f"'{name}'" for name in added[:8])
            if len(added) > 8:
                names += f" and {len(added) - 8} more"
            lines.append(
                f"{len(added)} flight(s) are ready and in the project: "
                f"{names}. The Input tab shows '{added[0]}'.")
            lines.append(self._examples_next_step(prepared))
        if message == CANCELLED:
            lines.append("The download was stopped. What arrived is kept: "
                         "asking for the same flights again continues from "
                         "there.")
        elif not success and not prepared:
            lines.append(f"No flight could be made ready:\n{message}")
        notes = [f"Flight {flight['flight']}: {note}"
                 for flight in prepared for note in flight["notes"]]
        if notes:
            lines.append("To know:\n" + "\n".join(notes[:12]))
        if failures and (success or message == CANCELLED):
            lines.append("Not ready:\n" + "\n".join(
                f"Flight {flight_id}: {why}"
                for flight_id, why in failures[:12]))
        text = "\n\n".join(line for line in lines if line)
        if not text:
            return
        if failures or (not success and message != CANCELLED):
            QMessageBox.warning(self, TITLE, text)
        else:
            QMessageBox.information(self, TITLE, text)

    @staticmethod
    def _examples_next_step(prepared: List[dict]) -> str:
        """What to do with the flights now, by what was taken for them."""
        routes = sum(1 for flight in prepared
                     if (flight.get("route") or {}).get("poses"))
        placed = [flight["labels"] for flight in prepared
                  if flight.get("labels")]
        waiting = sum(1 for flight in prepared
                      if flight.get("labels_wanted") and not flight.get("labels"))
        parts = []
        if routes:
            parts.append(
                f"The published route of {routes} flight(s) is the "
                "flight's own and on the map.")
        if placed:
            animals = sum(result["tracks"] for result in placed)
            boxes = sum(result["boxes"] for result in placed)
            parts.append(
                f"{animals} labelled animal(s) with {boxes} boxes are in "
                "the labelling tool and among the detections: go on with "
                "Processing > Geo-Reference Detections to see them on the "
                "map.")
        if waiting:
            parts.append(
                f"The labels of {waiting} flight(s) are put on the frames "
                "when they are extracted.")
        parts.append(
            "Go on with Pre-Processing > Extract Frames, then the steps of "
            "the Processing tab from top to bottom.")
        return " ".join(parts)

    # -- into the project ----------------------------------------------------

    def _add_example_flights(self, prepared: List[dict]) -> List[str]:
        """Make every prepared flight a flight of the project, with its
        settings stored in its folder, and show the first of them.

        A flight that is in the project already stays as it is set up; one
        whose folder holds settings from before keeps them - only a
        correction it does not have yet is added.

        :return: the names of the flights, as they are called in the
            project
        """
        from ..core import flights as flights_core
        from ..core.config_schema import save_config_entries

        # what the form shows stays with the flight it belongs to
        self.save_config_to_project()
        listed = self._flights()
        names, indices = [], []
        custom = self.thermal_calib_preset_combo.itemText(0)
        for flight in prepared:
            target = flight["target_folder"]
            stored = flights_core.load_config(target)
            if not stored:
                values = bambi_dataset.flight_configuration(
                    flight, self._pristine_config, custom)
                save_config_entries(
                    values, *flights_core.config_writers(stored))
                flights_core.save_config(target, stored)
            elif self._lacks_correction(flight, stored):
                settings = bambi_dataset.correction_settings(
                    flight["correction"])
                stored.update({key: str(value)
                               for key, value in settings.items()})
                flights_core.save_config(target, stored)
            index = flights_core.find_by_folder(listed, target)
            if index is None:
                name = self._free_flight_name(flight["name"], listed)
                listed = flights_core.add_flight(listed, name, target)
                index = len(listed) - 1
            names.append(listed[index]["name"])
            indices.append(index)

        self._flight_list = listed
        self._refresh_flight_combo()
        with_route = [index for index, flight in zip(indices, prepared)
                      if (flight.get("route") or {}).get("poses")]
        if len(with_route) > ROUTES_SHOWN:
            self.log(f"{len(with_route)} routes were imported; those of the "
                     f"first {ROUTES_SHOWN} flights are put on the map. For "
                     "another flight, switch to it and press 'Add Flight "
                     "Route to QGIS'.")
            with_route = with_route[:ROUTES_SHOWN]
        # the first flight last: it is the one that stays shown
        for index in reversed(with_route):
            if index != indices[0]:
                self._show_example_flight(index, route=True)
        self._show_example_flight(indices[0], route=indices[0] in with_route)
        self.save_config_to_project()
        for name in names:
            self.log(f"Added flight '{name}'")
        self.refresh_analytics_species()
        self._refresh_survey_area()
        return names

    @staticmethod
    def _lacks_correction(flight: dict, stored: dict) -> bool:
        """Whether a published correction came for a flight whose stored
        settings hold none of their own."""
        if not flight.get("correction"):
            return False
        return not bambi_dataset.has_correction(stored)

    @staticmethod
    def _free_flight_name(wanted: str, listed: List[dict]) -> str:
        """*wanted*, or - where another flight is called that - *wanted*
        with a number behind it."""
        taken = {str(flight.get("name", "")).strip().lower()
                 for flight in listed}
        name, count = wanted, 1
        while name.lower() in taken:
            count += 1
            name = f"{wanted} ({count})"
        return name

    def _show_example_flight(self, index: int, route: bool) -> None:
        """Switch the form to the flight at *index* and - where it has one
        - put its route on the map."""
        self._active_flight_index = index
        self._switching_flight = True
        try:
            self._refresh_flight_combo()
            self.load_config_from_project(restore_flights=False)
        finally:
            self._switching_flight = False
        folder = (self._active_flight() or {}).get("target_folder", "")
        self._check_existing_outputs(folder)
        if route and os.path.isdir(os.path.join(folder, "flight_route_t")):
            self.add_flight_route_to_qgis()
