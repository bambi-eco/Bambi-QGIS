# -*- coding: utf-8 -*-
"""The flights of a project, and moving between them.

A project holds any number of flights; each has a folder with its results
and its own settings. This part restores them when a project is opened,
fills the selector on the Input tab, and does what its buttons say: add a
flight (new, or a folder that has results already), rename it, remove it,
and switch to another - which stores the form for the flight that is left
and shows the settings of the one that is entered.
"""

import json
import os
from typing import Optional, List

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QInputDialog
from qgis.core import QgsProject

from ..gui_utils import PLUGIN_SCOPE


class FlightsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def browse_target_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Target Folder")
        if folder:
            self.target_folder_edit.setText(folder)
            self._on_target_folder_changed()

    def _on_target_folder_changed(self):
        """The target folder field was given a folder - typed, browsed for
        or filled in from the recordings' folder: the flight list follows,
        and everything that describes the folder is shown again."""
        if not getattr(self, "_switching_flight", False):
            self._adopt_target_folder(self.target_folder_edit.text().strip())
        # What the field holds now - a folder that was declined is gone.
        folder = self.target_folder_edit.text().strip()
        self._check_existing_outputs(folder)
        self._refresh_migrate_button()
        self.refresh_analytics_species()
        self._refresh_survey_area()

    def _refresh_migrate_button(self):
        """Show 'Migrate 5.x…' only where there is something to migrate."""
        from ..core import migration

        folder = self.target_folder_edit.text().strip()
        show = bool(folder) and os.path.isdir(folder) and \
            migration.has_legacy_outputs(folder) and \
            not migration.is_migrated(folder)
        self.migrate_btn.setVisible(show)

    def migrate_legacy_project(self):
        """Import this folder's 5.x outputs into the 6.0 store.

        Additive and read-only with respect to the legacy files, so it is safe
        to retry: delete the generated stores and run it again.
        """
        from qgis.PyQt.QtWidgets import QApplication
        from ..core import migration

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Migrate", "Select a target folder first.")
            return

        answer = QMessageBox.question(
            self, "Migrate to the 6.0 format",
            "Import this project's existing outputs into the new store?\n\n"
            "The current files are only read - nothing is modified or deleted, "
            "so the project keeps working exactly as it does now.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok)
        if answer != QMessageBox.StandardButton.Ok:
            return

        self.log("Migrating 5.x outputs into the 6.0 store...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = migration.migrate_project(folder, log_fn=self.log)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            QApplication.restoreOverrideCursor()
            self.log(f"Migration failed: {exc}")
            QMessageBox.critical(
                self, "Migration failed",
                f"The migration could not be completed:\n\n{exc}\n\n"
                "The existing outputs are unchanged.")
            return
        finally:
            QApplication.restoreOverrideCursor()

        summary = "\n".join(
            f"  {key}: {value}" for key, value in sorted(report.counts.items()))
        text = f"Migration finished.\n\n{summary or '  (nothing to import)'}"
        if report.warnings:
            text += "\n\nWarnings:\n" + "\n".join(
                f"  • {w}" for w in report.warnings)
        QMessageBox.information(self, "Migration finished", text)
        self._refresh_migrate_button()

    # ------------------------------------------------------------------
    # Flights
    # ------------------------------------------------------------------
    def _restore_flights(self, project) -> None:
        """Read the flight list from the QGIS project.

        A project written before 6.0 has none, so the target folder it stored
        becomes the first flight - that keeps a single-flight project working
        exactly as before, with the list simply describing what was already
        true.
        """
        from ..core import flights as flights_core

        raw, _ok = project.readEntry(PLUGIN_SCOPE, "Flights/List", "")
        try:
            restored = json.loads(raw) if raw else []
        except ValueError:
            restored = []
        # a flight added by name may not have its target folder yet
        restored = [f for f in restored
                    if isinstance(f, dict) and (f.get("target_folder") or f.get("name"))]

        if not restored:
            folder, _ok = project.readEntry(
                PLUGIN_SCOPE, "Input/TargetFolder", "")
            folder = (folder or "").strip()
            restored = [{"name": flights_core.default_name(folder),
                         "target_folder": folder}] if folder else []

        index_text, _ok = project.readEntry(PLUGIN_SCOPE, "Flights/Active", "0")
        try:
            index = int(index_text)
        except (TypeError, ValueError):
            index = 0

        self._flight_list = restored
        if restored:
            self._active_flight_index = max(0, min(index, len(restored) - 1))
        else:
            self._active_flight_index = 0
        self._refresh_flight_combo()

    def _flights(self) -> List[dict]:
        return list(getattr(self, "_flight_list", []) or [])

    def _active_flight(self) -> Optional[dict]:
        flights = self._flights()
        index = getattr(self, "_active_flight_index", 0)
        if 0 <= index < len(flights):
            return flights[index]
        return None

    def _refresh_flight_combo(self):
        from ..core import flights as flights_core

        blocked = self.flight_combo.blockSignals(True)
        self.flight_combo.clear()
        for flight in self._flights():
            self.flight_combo.addItem(flights_core.group_name(flight))
        index = getattr(self, "_active_flight_index", 0)
        if 0 <= index < self.flight_combo.count():
            self.flight_combo.setCurrentIndex(index)
        self.flight_combo.blockSignals(blocked)

        if hasattr(self, "flight_name_edit"):
            self._mark_flight_name()

        several = self.flight_combo.count() > 1
        self.flight_combo.setEnabled(several)
        self.flight_rename_btn.setEnabled(self.flight_combo.count() > 0)
        self.flight_remove_btn.setEnabled(self.flight_combo.count() > 0)
        # The survey's flight list is the same list, so it follows.
        self._refresh_survey()

    def _adopt_target_folder(self, folder: str):
        """Keep the flight list in step with a target folder chosen by hand.

        The folder is the flight's identity, so setting one on a project with
        no flights creates the first, and changing it moves the active flight
        rather than silently detaching it from its outputs.
        """

        folder = (folder or "").strip()
        if not folder or getattr(self, "_adopting_folder", False):
            return
        self._adopting_folder = True
        self._adoption_loaded_config = False
        try:
            self._adopt_target_folder_now(folder)
        finally:
            self._adopting_folder = False

    def _adopt_target_folder_now(self, folder: str):
        from ..core import flights as flights_core

        existing = self._flights()

        if not existing:
            # The first flight: it exists once it has a name the user gave
            # it - typed into the name field, or asked for here.
            name = self.flight_name_edit.text().strip() \
                or self._prompt_flight_name(folder)
            try:
                updated = flights_core.add_flight([], name, folder)
            except flights_core.FlightError as exc:
                self._mark_flight_name()
                if name:
                    QMessageBox.warning(self, "Flight", str(exc))
                else:
                    self.log("No flight yet: enter a flight name on the Input "
                             "tab.")
                return
            if self._settle_into_folder(updated, 0, folder):
                self.flight_name_edit.clear()
            return

        index = getattr(self, "_active_flight_index", 0)
        previous = (existing[index].get("target_folder") or "").strip() \
            if 0 <= index < len(existing) else ""
        if flights_core.normalise(previous) == flights_core.normalise(folder):
            return
        if self._step_is_running():
            QMessageBox.information(
                self, "Flight",
                "A step is running. Wait for it to finish before pointing "
                "the flight at another folder.")
            self._set_target_folder_text(previous)
            return
        try:
            # refuses a folder another flight already uses
            updated = flights_core.set_folder(existing, index, folder)
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            self._set_target_folder_text(previous)
            return
        self._settle_into_folder(updated, index, folder, previous)

    def _settle_into_folder(self, updated, index: int, folder: str,
                            previous: str = "") -> bool:
        """Make *updated* the flight list, whose flight at *index* now lives
        in *folder* - it lived in *previous*, or had no folder yet.

        **A folder that has been processed before is taken as it stands.**
        The user confirms, and the configuration stored in it is loaded into
        the form. That used to happen only for a flight just added by name:
        the first flight of a project, and a flight pointed at another
        folder, kept the form's values - and the next save, which saving the
        QGIS project is, replaced what the folder had stored with them.

        What the form shows when a flight leaves *previous* for a folder
        with a configuration of its own is the configuration of *previous*,
        and is stored there first, as when switching flights.

        Returns False - nothing changed, and the folder field is back at
        *previous* - when the user declines or the folder cannot be read.
        """
        from ..core import flights as flights_core

        try:
            found = flights_core.describe_existing(folder)
        except Exception as exc:  # noqa: BLE001 - a store that cannot be read
            QMessageBox.warning(
                self, "Flight",
                f"{folder}\nholds results that cannot be read:\n\n{exc}\n\n"
                "The flight was not pointed at this folder, and nothing in "
                "it was changed.")
            self._set_target_folder_text(previous)
            return False

        if found["is_flight"]:
            if not self._confirm_existing_flight(folder, found,
                                                 leaving=previous):
                self._set_target_folder_text(previous)
                return False
            if found["configured"] and previous and os.path.isdir(previous):
                self._store_form_config(previous)

        self._flight_list = updated
        self._active_flight_index = index
        if found["configured"]:
            self._switching_flight = True
            try:
                self.load_config_from_project(restore_flights=False)
            finally:
                self._switching_flight = False
            self._adoption_loaded_config = True
            self.log(f"Loaded the configuration stored in {folder}")
        self._set_target_folder_text(folder)
        self._refresh_flight_combo()
        if previous:
            self.log(f"Flight '{updated[index].get('name', '')}' now uses "
                     f"{folder}")
        # the configuration now has its home in the flight's folder
        self.save_config_to_project()
        return True

    def _flight_owns(self, folder: str) -> bool:
        """True when *folder* is the active flight's target folder."""
        from ..core import flights as flights_core

        flight = self._active_flight()
        if not flight or not (folder or "").strip():
            return False
        return flights_core.normalise(flight.get("target_folder") or "") == \
            flights_core.normalise(folder)

    def _store_form_config(self, folder: str) -> bool:
        """Store the form's values as the configuration of *folder*, in its
        ``project.gpkg``; False (and logged) when that could not be done."""
        from ..core import flights as flights_core
        from ..core.config_schema import save_config_entries

        values = self._current_config_values()
        values["Input/TargetFolder"] = folder
        stored = {}
        save_config_entries(values, *flights_core.config_writers(stored))
        try:
            flights_core.save_config(folder, stored)
        except Exception as exc:  # noqa: BLE001 - never lose a save
            self.log(f"Warning: could not save configuration to the "
                     f"store: {exc}")
            return False
        return True

    def _set_target_folder_text(self, folder: str):
        """Set the target folder field without it being taken for an edit."""
        was = getattr(self, "_switching_flight", False)
        self._switching_flight = True
        try:
            self.target_folder_edit.setText(folder)
        finally:
            self._switching_flight = was

    def _prompt_flight_name(self, folder: str) -> str:
        """Ask for the name of the flight in *folder*; ``""`` when the
        user gives none. Nothing is suggested: the name is the user's."""
        name, ok = QInputDialog.getText(
            self, "Flight name",
            "Every flight needs a name - it titles the flight's layer group "
            "in QGIS and identifies it in the survey tools.\n\nName of the "
            f"flight in\n{folder}:")
        return name.strip() if ok else ""

    def _mark_flight_name(self):
        """Show the name field as missing while a target folder is set
        but no named flight exists."""
        flight = self._active_flight()
        named = bool(flight and (flight.get("name") or "").strip())
        folder = self.target_folder_edit.text().strip() \
            if hasattr(self, "target_folder_edit") else ""
        missing = bool(folder) and not named
        self.flight_name_edit.setStyleSheet(
            "QLineEdit { border: 1px solid #c0392b; }" if missing else "")

    def _rename_active_flight(self, name: str) -> bool:
        """Rename the active flight and its layer group; False (with the
        reason shown) when the name is empty, reserved or taken."""
        from ..core import flights as flights_core

        flight = self._active_flight()
        if flight is None:
            return False
        previous = flights_core.group_name(flight)
        try:
            updated = flights_core.rename_flight(
                self._flights(), self._active_flight_index, name)
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            return False

        self._flight_list = updated
        self._refresh_flight_combo()

        # Settings the project keeps for this flight stay this flight's.
        project = QgsProject.instance()
        kept = self._kept_in_project(project).get(flight.get("name") or "")
        if kept is not None:
            self._keep_in_project(project, flight.get("name") or "", None)
            self._keep_in_project(
                project, updated[self._active_flight_index]["name"], kept)

        # Rename the group rather than leaving a stale one behind.
        group = QgsProject.instance().layerTreeRoot().findGroup(previous)
        if group is not None:
            group.setName(flights_core.group_name(updated[
                self._active_flight_index]))
        self.log(f"Renamed flight to '{updated[self._active_flight_index]['name']}'")
        return True

    def _on_flight_changed(self, index: int):
        """Switch flights: save the current configuration, load the new one."""
        if index < 0 or index >= len(self._flights()):
            return
        if getattr(self, "_switching_flight", False):
            return
        if self._step_is_running():
            QMessageBox.information(
                self, "Flight",
                "A step is running. Wait for it to finish before switching "
                "flights - otherwise the log and status would describe a "
                "flight you are no longer looking at.")
            self._refresh_flight_combo()
            return

        self._switching_flight = True
        try:
            self.save_config_to_project()
            self._active_flight_index = index
            self.load_config_from_project(restore_flights=False)
        finally:
            self._switching_flight = False

        flight = self._active_flight()
        if flight:
            self.log(f"Switched to flight '{flight['name']}'")
            self._check_existing_outputs(flight.get("target_folder", ""))
        self.refresh_analytics_species()
        self._refresh_survey_area()

    def add_flight(self):
        """Add a new flight with the name in the Name field and switch to
        it. Its target folder is set afterwards, in the target folder field."""
        from ..core import flights as flights_core
        from ..core.config_schema import WIDGET_BINDINGS

        if self._step_is_running():
            QMessageBox.information(
                self, "Flight",
                "A step is running. Wait for it to finish before adding a "
                "flight.")
            return

        existing = self._flights()
        name = self.flight_name_edit.text()
        if not name.strip():
            QMessageBox.information(
                self, "Flight",
                "Enter the name of the new flight in the Name field first.")
            self.flight_name_edit.setFocus()
            return

        if not existing:
            # The first flight: the form is this flight's already, so it
            # keeps what was entered, a target folder included.
            folder = self.target_folder_edit.text().strip()
            try:
                updated = (flights_core.add_flight([], name, folder) if folder
                           else flights_core.add_named_flight([], name))
            except flights_core.FlightError as exc:
                QMessageBox.warning(self, "Flight", str(exc))
                return
            if folder:
                # as when the folder is entered: one that has been
                # processed before brings its stored configuration
                self._adopting_folder = True
                try:
                    if not self._settle_into_folder(updated, 0, folder):
                        return
                finally:
                    self._adopting_folder = False
            else:
                self._flight_list = updated
                self._active_flight_index = 0
                self._refresh_flight_combo()
                self.save_config_to_project()
            self.flight_name_edit.clear()
            self.log(f"Added flight '{updated[0]['name']}'")
            return

        try:
            updated = flights_core.add_named_flight(existing, name)
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            return

        pending = flights_core.without_folder(existing)
        if pending is not None:
            QMessageBox.information(
                self, "Flight",
                f"'{existing[pending]['name']}' has no target folder yet. "
                "Set it before adding another flight - a flight keeps its "
                "configuration in its target folder.")
            return

        copy_configuration = self._ask_new_flight_configuration()
        if copy_configuration is None:
            return

        self.save_config_to_project()
        current = {key: self._config_widget_value(attr, role)
                   for key, (attr, role) in WIDGET_BINDINGS.items()}

        self._flight_list = updated
        self._active_flight_index = len(updated) - 1
        self.flight_name_edit.clear()
        self._refresh_flight_combo()

        self._switching_flight = True
        try:
            self._apply_flight_values(
                flights_core.new_flight_values(
                    current, copy_configuration,
                    defaults=self._pristine_config))
            self.target_folder_edit.setText("")
        finally:
            self._switching_flight = False

        how = "copied" if copy_configuration else "default"
        self.log(f"Added flight '{updated[-1]['name']}' ({how} configuration, "
                 "inputs cleared) - now set its target folder")
        self.save_config_to_project()
        self._check_existing_outputs("")   # nothing processed yet
        self._refresh_migrate_button()
        self.refresh_analytics_species()
        self._refresh_survey_area()
        self.target_folder_edit.setFocus()

    def browse_and_add_flight(self):
        """Add a flight by its target folder - meant for a folder that has
        been processed before - and switch to it."""
        from ..core import flights as flights_core

        if self._step_is_running():
            QMessageBox.information(
                self, "Flight",
                "A step is running. Wait for it to finish before adding a "
                "flight.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Target folder of the flight to add")
        if not folder:
            return
        try:
            flights_core.validate_folder(folder, self._flights())
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            return

        name = self.flight_name_edit.text().strip()
        if not name:
            name, ok = QInputDialog.getText(
                self, "Add flight", "Flight name (required):")
            if not ok:
                return

        try:
            updated = flights_core.add_flight(self._flights(), name, folder)
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            return

        # A folder that has already been processed describes itself, so it is
        # adopted as it stands rather than started from scratch.
        try:
            existing = flights_core.describe_existing(folder)
        except Exception as exc:  # noqa: BLE001 - a store that cannot be read
            QMessageBox.warning(
                self, "Flight",
                f"{folder}\nholds results that cannot be read:\n\n{exc}\n\n"
                "The flight was not added, and nothing in the folder was "
                "changed.")
            return
        if existing["is_flight"]:
            if not self._confirm_existing_flight(folder, existing):
                return
            self._adopt_existing_flight(updated, folder, existing["configured"])
            self.flight_name_edit.clear()
            return

        copy_configuration = self._ask_new_flight_configuration()
        if copy_configuration is None:
            return

        from ..core.config_schema import WIDGET_BINDINGS

        self.save_config_to_project()
        current = {key: self._config_widget_value(attr, role)
                   for key, (attr, role) in WIDGET_BINDINGS.items()}

        self._flight_list = updated
        self._active_flight_index = len(updated) - 1
        self.flight_name_edit.clear()
        self._refresh_flight_combo()

        self._switching_flight = True
        try:
            self._apply_flight_values(
                flights_core.new_flight_values(
                    current, copy_configuration,
                    defaults=self._pristine_config))
            self.target_folder_edit.setText(folder)
        finally:
            self._switching_flight = False

        how = "copied" if copy_configuration else "default"
        self.log(f"Added flight '{updated[-1]['name']}' → {folder} "
                 f"({how} configuration, inputs cleared)")
        self.save_config_to_project()
        self._check_existing_outputs(folder)

    def _confirm_existing_flight(self, folder: str, existing: dict,
                                 leaving: str = "") -> bool:
        """Confirm adopting a folder that has been processed before.

        *leaving* is the folder the flight is in now, when it is being
        pointed at another one rather than added.
        """
        if existing["results"]:
            items = "</li><li>".join(existing["results"])
            found = f"It already contains:<ul><li>{items}</li></ul>"
        else:
            found = "It already contains a saved configuration.<br><br>"

        if existing["configured"]:
            settings = ("Its stored configuration is loaded, so the inputs "
                        "and settings come back as they were.")
            if leaving:
                settings += (" The settings shown now stay with "
                             f"<b>{leaving}</b>.")
        else:
            settings = ("It has no stored configuration, so the settings stay "
                        "as they are - check the inputs before running "
                        "anything.")

        box = QMessageBox(self)
        box.setWindowTitle("Use existing folder" if leaving
                           else "Add existing flight")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"<b>{folder}</b> has been processed before.")
        box.setInformativeText(
            f"{found}{settings}<br><br>Nothing is recomputed and nothing is "
            "overwritten. Steps that already ran are shown as done.")
        add_btn = box.addButton("Use this folder" if leaving else "Add flight",
                                QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(add_btn)
        box.exec()
        return box.clickedButton() is not cancel

    def _adopt_existing_flight(self, updated, folder: str, configured: bool):
        """Add a previously processed folder and load what it already holds."""
        self.save_config_to_project()

        self._flight_list = updated
        self._active_flight_index = len(updated) - 1

        self._switching_flight = True
        try:
            self._refresh_flight_combo()
            if configured:
                # Reads the flight's own project.gpkg, which is the point: the
                # settings that produced those results come with the folder.
                self.load_config_from_project(restore_flights=False)
            # With no stored configuration there is nothing to load: falling
            # back to the QGIS project would pull in the *previous* flight's
            # settings, which is not "left as they are". Only the folder moves.
            self.target_folder_edit.setText(folder)
        finally:
            self._switching_flight = False

        self.log(f"Added existing flight '{updated[-1]['name']}' → {folder}")
        self.save_config_to_project()
        self._check_existing_outputs(folder)
        return True

    def _ask_new_flight_configuration(self):
        """Copy the current configuration into the new flight, or start clean?

        Returns True to copy, False for defaults, or None when cancelled. The
        recordings are cleared either way - the choice is only about the
        processing settings.
        """
        box = QMessageBox(self)
        box.setWindowTitle("New flight")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(
            "A new flight starts with <b>no input files</b> - the videos, "
            "photos, flight log, calibrations, DEM and correction are cleared, "
            "so the previous flight's recordings are not processed again into "
            "the new folder.<br><br>"
            "What should happen to the <b>processing settings</b> - detection "
            "confidence, tracker, extraction, ALFS and the rest?"
        )
        copy_btn = box.addButton("Ok (copy configurations)",
                                 QMessageBox.ButtonRole.AcceptRole)
        default_btn = box.addButton("Ok (default configurations)",
                                    QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(copy_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is copy_btn:
            return True
        if clicked is default_btn:
            return False
        return None

    def _apply_flight_values(self, values):
        """Push a full set of config values into the form widgets."""
        from ..core.config_schema import WIDGET_BINDINGS

        for key, (attr, role) in WIDGET_BINDINGS.items():
            if key in values:
                self._apply_config_value(attr, role, values[key])

    def rename_flight(self):
        """Rename the active flight, and its QGIS layer group with it."""

        flight = self._active_flight()
        if flight is None:
            QMessageBox.information(
                self, "Flight", "There is no flight to rename yet.")
            return

        name, ok = QInputDialog.getText(
            self, "Rename flight", "Flight name:", text=flight["name"])
        if not ok:
            return
        self._rename_active_flight(name)

    def remove_flight(self):
        """Remove the active flight from the project, once confirmed."""
        from ..core import flights as flights_core

        flight = self._active_flight()
        if flight is None:
            QMessageBox.information(
                self, "Flight", "There is no flight to remove.")
            return
        if self._step_is_running():
            QMessageBox.information(
                self, "Flight",
                "A step is running. Wait for it to finish before removing a "
                "flight.")
            return

        if not self._confirm_remove_flight(flight):
            return

        index = self._active_flight_index
        name = flights_core.group_name(flight)
        updated = flights_core.remove_flight(self._flights(), index)
        # what the project kept for it goes with it; the folder's own stay
        self._keep_in_project(
            QgsProject.instance(), flight.get("name") or "", None)
        active = flights_core.active_after_removal(
            len(self._flights()), index, index)

        # Drop the layer group; the layers only describe the flight that is
        # going away. The files they read stay where they are.
        root = QgsProject.instance().layerTreeRoot()
        group = root.findGroup(name)
        if group is not None:
            self._parent_group(group, root).removeChildNode(group)

        self._flight_list = updated
        self._active_flight_index = max(0, active)

        self._switching_flight = True
        try:
            self._refresh_flight_combo()
            if active < 0:
                # Nothing left to describe, so the form must stop pointing at
                # the folder that was just removed - otherwise the next save
                # would write straight back into it.
                self.target_folder_edit.setText("")
            else:
                self.load_config_from_project(restore_flights=False)
        finally:
            self._switching_flight = False

        self.log(f"Removed flight '{name}' from the project "
                 f"({flight.get('target_folder', '')} was not deleted)")
        self.save_config_to_project()
        return True

    def _confirm_remove_flight(self, flight) -> bool:
        """Ask before removing, and be exact about what is and is not deleted."""
        from ..core import flights as flights_core

        folder = flight.get("target_folder", "") or "(no folder)"
        box = QMessageBox(self)
        box.setWindowTitle("Remove flight")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            f"Remove the flight <b>{flights_core.group_name(flight)}</b> from "
            "this project?")
        box.setInformativeText(
            "Its QGIS layer group is removed and the project stops listing "
            "it.<br><br>"
            f"<b>Nothing on disk is deleted.</b> {folder} keeps its frames, "
            "detections, tracks and configuration, so adding the flight back "
            "later picks up exactly where it left off.")
        remove_btn = box.addButton("Remove flight",
                                   QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        return box.clickedButton() is remove_btn
