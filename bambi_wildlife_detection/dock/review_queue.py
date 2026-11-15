# -*- coding: utf-8 -*-
"""Survey Analytics > Review Queue: going through the tracks in doubt.

Opens the review queue (``bambi_review_queue_dialog``) for the survey's
flights and gives it what it needs: the queue from the stores as they stand
(``core.review_queue``), the inspector to look at a track, and the functions
that write the reviewer's word (``core.track_review``).
"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QApplication, QMessageBox


class ReviewQueueMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def open_review_queue(self):
        """Open the queue of the tracks worth a look in the survey's
        flights."""
        from ..bambi_click_tool import open_track_in_viewer
        from ..bambi_review_queue_dialog import ReviewQueueDialog
        from ..core import (
            label_store, review, review_queue, track_review, track_store)

        survey = self._checked_survey()
        if survey is None:
            return None
        if not self._survey_prerequisites(survey, self._tracks_check(survey)):
            return None
        suffix, epsg = survey.modality, survey.epsg
        camera_label = survey.camera_label
        by_name = {flight.name: flight for flight in survey.flights}
        single = survey.flights[0]
        several = survey.several

        def flight_of(row):
            name = row.get("flight") if isinstance(row, dict) else row
            return by_name.get(name, single) if name else single

        def load(settings):
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                return review_queue.for_flights(
                    [(flight.name, flight.target_folder,
                      survey.species_ids(flight))
                     for flight in survey.flights],
                    suffix, epsg=epsg, settings=settings, several=several)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "Review Queue",
                                     f"Could not build the queue: {exc}")
                return [], 0
            finally:
                QApplication.restoreOverrideCursor()

        def vocabulary(flight_name):
            flight = flight_of(flight_name)
            known = label_store.vocabulary(flight.target_folder) or {}
            enums = known.get("enums") or {}
            return {
                "species": [(entry["species_id"], entry["name"])
                            for entry in known.get("species") or []],
                "sex": [(value["value_id"], value["label"])
                        for value in enums.get("sex") or []],
                "age": [(value["value_id"], value["label"])
                        for value in enums.get("age") or []]}

        def said(row, what):
            """A reviewer's word is on the track: the reports made before
            are out of date, and the log says what was said."""
            flight = flight_of(row)
            review.mark_reports_stale(flight.target_folder, suffix)
            self.log(f"Track {row['track_id']} ({flight.name}, "
                     f"{camera_label}): {what}")

        def refuse(exc):
            QMessageBox.warning(self, "Review Queue", str(exc))
            return False

        def open_track(row):
            flight = flight_of(row)
            return open_track_in_viewer(
                self.iface.mainWindow(), flight.target_folder, suffix,
                int(row["track_id"]), dem_path=flight.dem_path,
                correction_path=flight.correction_path,
                frame=row.get("start_frame"), epsg=epsg)

        def approve(row):
            flight = flight_of(row)
            track_store.set_track_attribute(
                flight.target_folder, suffix, int(row["track_id"]),
                "approved", True)
            said(row, "approved")
            return True

        def set_species(row, species_id):
            flight = flight_of(row)
            try:
                track_review.set_species(
                    flight.target_folder, suffix, int(row["track_id"]),
                    int(species_id))
            except track_review.TrackReviewError as exc:
                return refuse(exc)
            names = dict(vocabulary(row.get("flight"))["species"])
            said(row, f"species set to {names.get(species_id, species_id)}")
            return True

        def set_attribute(row, field, value):
            flight = flight_of(row)
            try:
                track_review.set_attribute(
                    flight.target_folder, suffix, int(row["track_id"]),
                    field, value)
            except track_review.TrackReviewError as exc:
                return refuse(exc)
            labels = dict(vocabulary(row.get("flight"))[field])
            said(row, f"{field} set to "
                      f"{labels.get(value, 'not known')}")
            return True

        dialog = ReviewQueueDialog(
            load, vocabulary, open_track, approve, set_species,
            set_attribute, camera_label=camera_label, several=several,
            parent=self)
        dialog.setModal(False)
        dialog.show()
        self._review_queue_dialog = dialog
        self._review_queue_context = (
            {flight.key: flight.name for flight in survey.flights}, suffix,
            several)
        # what is approved or deleted in the inspector leaves the queue
        self._connect_viewer_signals()
        return dialog

    def _review_queue_forget(self, target_folder: str, modality: str,
                             track_id: int) -> None:
        """A track was dealt with in the inspector - approved, or deleted:
        it leaves the open queue."""
        dialog = getattr(self, "_review_queue_dialog", None)
        context = getattr(self, "_review_queue_context", None)
        if dialog is None or context is None or not dialog.isVisible():
            return
        flights, suffix, several = context
        name = flights.get(os.path.normcase(os.path.abspath(target_folder)))
        if suffix != modality or name is None:
            return
        dialog.remove_track(track_id, name if several else None)
