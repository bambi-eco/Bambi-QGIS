# -*- coding: utf-8 -*-
"""Survey selection and census table of the Survey Analytics tab.

The survey analytics are not tied to the flight selected on the processing
tabs. :class:`SurveyFlightsPanel` lets the user tick the processed flights
that make up the survey, pick the camera and the survey folder; it builds the
``config["survey"]`` description every survey analytic reads (see
:mod:`core.survey`). :class:`CensusTableDialog` shows the census summary.
"""

import json
import os

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)
from qgis.core import QgsProject

from .core import flights as flights_core
from .core import store
from .core.survey import Survey, SurveyFlight, flight_settings
from .gui_utils import PLUGIN_SCOPE

#: QGIS project entries of the survey selection.
ENTRY_FLIGHTS = "Survey/Flights"
ENTRY_CAMERA = "Survey/Camera"
ENTRY_FOLDER = "Survey/Folder"
DEFAULT_FOLDER_NAME = "bambi_survey"

_FOLDER_ROLE = Qt.ItemDataRole.UserRole
_ALIGN = Qt.AlignmentFlag
_RIGHT = _ALIGN.AlignRight | _ALIGN.AlignVCenter


class SurveyFlightsPanel(QGroupBox):
    """The "Survey" box: flights, camera and survey folder."""

    #: Emitted when the flights, the camera or the folder change.
    changed = pyqtSignal()

    def __init__(self, dock):
        super().__init__("Survey")
        self.dock = dock
        #: Normalised folders of the ticked flights; ``None`` = not chosen
        #: yet, in which case the active flight is ticked.
        self._selected = None

        layout = QVBoxLayout(self)
        desc = QLabel(
            "The flights analysed together. Every tool on this tab works on "
            "all ticked flights at once - independent of the flight selected "
            "on the processing tabs - and writes its results into the survey "
            "folder.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(desc)

        self.flights_list = QListWidget()
        self.flights_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self.flights_list.setMaximumHeight(110)
        self.flights_list.setToolTip(
            "Tick the flights of the survey. Their counts, areas and "
            "distances are pooled; the census summary also lists them one by "
            "one.")
        self.flights_list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.flights_list)

        row = QHBoxLayout()
        self.all_btn = QPushButton("All")
        self.all_btn.clicked.connect(lambda: self._tick_all(True))
        self.none_btn = QPushButton("None")
        self.none_btn.clicked.connect(lambda: self._tick_all(False))
        row.addWidget(self.all_btn)
        row.addWidget(self.none_btn)
        row.addSpacing(12)
        row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Thermal", "RGB"])
        self.camera_combo.setToolTip(
            "Whose detections, tracks, footprints and transects are analysed. "
            "Each camera has its own; the flights must have been processed "
            "for it.")
        self.camera_combo.currentIndexChanged.connect(
            lambda _i: self._emit_changed())
        row.addWidget(self.camera_combo)
        row.addStretch()
        layout.addLayout(row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Survey folder:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setToolTip(
            "Where the survey results (analytics_t / analytics_w) and the "
            "survey area are stored. Empty = 'bambi_survey' next to the QGIS "
            "project.")
        self.folder_edit.editingFinished.connect(self._emit_changed)
        folder_row.addWidget(self.folder_edit)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse)
        layout.addLayout(folder_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------ #
    # Flight list
    # ------------------------------------------------------------------ #

    @staticmethod
    def _available(folder: str) -> bool:
        """A flight can be ticked once its folder exists; what it lacks is
        reported per analytic when that runs."""
        return bool(folder) and os.path.isdir(folder)

    def _flights(self):
        """The project's flights - or, in a project that lists none yet, the
        target folder on the form as its one flight."""
        flights = self.dock._flights()
        if flights:
            return flights
        folder = self.dock.target_folder_edit.text().strip()
        if not folder:
            return []
        return [{"name": flights_core.default_name(folder),
                 "target_folder": folder}]

    def _active_folder(self) -> str:
        active = self.dock._active_flight()
        folder = (active or {}).get("target_folder") or \
            self.dock.target_folder_edit.text().strip()
        return flights_core.normalise(folder) if folder else ""

    def refresh(self):
        """Rebuild the list from the dock's flights, keeping the ticks."""
        flights = self._flights()
        # Until the user ticks something, the survey is the active flight.
        # Fixed once a real flight exists - switching flights afterwards does
        # not change the survey - but not before: the first refresh runs
        # before any flight is there.
        selected = self._selected
        if selected is None:
            active = self._active_folder()
            selected = {active} if active else set()
            if active and self.dock._flights():
                self._selected = set(selected)
        blocked = self.flights_list.blockSignals(True)
        self.flights_list.clear()
        for flight in flights:
            folder = flight.get("target_folder", "")
            key = flights_core.normalise(folder)
            item = QListWidgetItem(flights_core.group_name(flight))
            item.setData(_FOLDER_ROLE, folder)
            available = self._available(folder)
            flags = Qt.ItemFlag.ItemIsUserCheckable
            if available:
                flags |= Qt.ItemFlag.ItemIsEnabled
                processed = os.path.isfile(store.project_path(folder))
                item.setToolTip(folder if processed else
                                f"{folder}\nNot processed yet.")
            else:
                item.setToolTip(f"{folder}\nThe folder does not exist.")
            item.setFlags(flags)
            item.setCheckState(
                Qt.CheckState.Checked if available and key in selected
                else Qt.CheckState.Unchecked)
            self.flights_list.addItem(item)
        self.flights_list.blockSignals(blocked)
        self._update_status()

    def _on_item_changed(self, _item):
        self._selected = {flights_core.normalise(f["target_folder"])
                          for f in self.selected_flights()}
        self._emit_changed()

    def _tick_all(self, on: bool):
        blocked = self.flights_list.blockSignals(True)
        for index in range(self.flights_list.count()):
            item = self.flights_list.item(index)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if on
                                   else Qt.CheckState.Unchecked)
        self.flights_list.blockSignals(blocked)
        self._on_item_changed(None)

    def selected_flights(self):
        """The ticked flights, as the dock's flight dicts, in list order."""
        by_folder = {flights_core.normalise(f.get("target_folder", "")): f
                     for f in self._flights()}
        chosen = []
        for index in range(self.flights_list.count()):
            item = self.flights_list.item(index)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            flight = by_folder.get(
                flights_core.normalise(item.data(_FOLDER_ROLE)))
            if flight is not None:
                chosen.append(flight)
        return chosen

    def modality(self) -> str:
        return "t" if self.camera_combo.currentIndex() == 0 else "w"

    def _browse_folder(self):
        start = self.folder() or QgsProject.instance().homePath()
        folder = QFileDialog.getExistingDirectory(
            self, "Survey Folder", start)
        if folder:
            self.folder_edit.setText(folder)
            self._emit_changed()

    def default_folder(self) -> str:
        home = QgsProject.instance().homePath()
        if home:
            return os.path.join(home, DEFAULT_FOLDER_NAME)
        chosen = self.selected_flights()
        if chosen and chosen[0].get("target_folder"):
            parent = os.path.dirname(os.path.normpath(
                chosen[0]["target_folder"]))
            return os.path.join(parent, DEFAULT_FOLDER_NAME)
        return ""

    def folder(self) -> str:
        return self.folder_edit.text().strip() or self.default_folder()

    def _emit_changed(self):
        self._update_status()
        self.changed.emit()

    # ------------------------------------------------------------------ #
    # The survey description
    # ------------------------------------------------------------------ #

    def survey_flights(self):
        """The ticked flights as :class:`SurveyFlight` objects.

        The active flight takes its CRS and DEM from the live form (it may
        not be saved yet); every other flight from its stored configuration.
        """
        active_key = self._active_folder()
        out = []
        for flight in self.selected_flights():
            folder = flight["target_folder"]
            name = flight.get("name") or flights_core.default_name(folder)
            if flights_core.normalise(folder) == active_key:
                epsg = self.dock._parse_epsg_from_text(
                    self.dock.target_crs_edit.text().strip().upper())
                out.append(SurveyFlight(
                    name=name, target_folder=folder, epsg=epsg,
                    dem=self.dock.dem_metadata_path_edit.text().strip(),
                    dem_path=self.dock.dem_path_edit.text().strip(),
                    correction_path=self.dock.correction_path_edit.text().strip()))
                continue
            settings = flight_settings(folder)
            out.append(SurveyFlight(name=name, target_folder=folder,
                                    **settings))
        return out

    def survey(self, species=None, per_species=False, area_enabled=False,
               include_manual=True) -> Survey:
        # The list is rebuilt first: the flights or the form's target folder
        # may have changed without a signal reaching the panel. The ticks
        # survive it (see refresh).
        self.refresh()
        return Survey(folder=self.folder(), modality=self.modality(),
                      flights=self.survey_flights(), species=species,
                      per_species=per_species, include_manual=include_manual,
                      area_enabled=area_enabled)

    def epsg(self):
        """The ticked flights' common CRS, else the form's (for drawing)."""
        try:
            epsg = Survey(folder="", flights=self.survey_flights()).epsg
        except Exception:  # noqa: BLE001 - mixed CRSs are reported elsewhere
            epsg = None
        if epsg:
            return epsg
        return self.dock._parse_epsg_from_text(
            self.dock.target_crs_edit.text().strip().upper())

    def _update_status(self):
        chosen = self.selected_flights()
        if not chosen:
            self.status_label.setText(
                "No flight ticked - tick the processed flights to analyse.")
            return
        try:
            epsg = Survey(folder="", flights=self.survey_flights()).epsg
            crs = f", EPSG:{epsg}" if epsg else ""
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            self.status_label.setText(f"⚠ {exc}")
            return
        self.status_label.setText(
            f"{len(chosen)} flight(s){crs}; results in {self.folder()}")

    # ------------------------------------------------------------------ #
    # Persistence (QGIS project)
    # ------------------------------------------------------------------ #

    def save(self, project):
        folders = [f["target_folder"] for f in self.selected_flights()]
        project.writeEntry(PLUGIN_SCOPE, ENTRY_FLIGHTS, json.dumps(folders))
        project.writeEntry(PLUGIN_SCOPE, ENTRY_CAMERA, self.modality())
        project.writeEntry(PLUGIN_SCOPE, ENTRY_FOLDER,
                           self.folder_edit.text().strip())

    def load(self, project):
        raw, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_FLIGHTS, "")
        try:
            folders = json.loads(raw) if ok and raw else None
        except ValueError:
            folders = None
        self._selected = (None if folders is None else
                          {flights_core.normalise(f) for f in folders})
        camera, _ok = project.readEntry(PLUGIN_SCOPE, ENTRY_CAMERA, "t")
        folder, _ok = project.readEntry(PLUGIN_SCOPE, ENTRY_FOLDER, "")
        self._set_quietly(1 if camera == "w" else 0, folder or "")
        self.refresh()

    def reset(self):
        self._selected = None
        self._set_quietly(0, "")
        self.refresh()

    def _set_quietly(self, camera_index: int, folder: str):
        """Set camera and folder without the change signal - which would
        save them straight back into the project half-restored."""
        blocked = self.camera_combo.blockSignals(True)
        self.camera_combo.setCurrentIndex(camera_index)
        self.camera_combo.blockSignals(blocked)
        blocked = self.folder_edit.blockSignals(True)
        self.folder_edit.setText(folder)
        self.folder_edit.blockSignals(blocked)


class CensusTableDialog(QDialog):
    """The census summary as a table: one row per flight and species."""

    def __init__(self, census: dict, csv_path: str = "", parent=None):
        super().__init__(parent)
        from .core import census as census_core

        self.setWindowTitle(f"Census Summary ({census.get('camera', '')}, "
                            f"{census.get('source', '')})")
        self.resize(980, 460)
        self._csv_path = csv_path
        layout = QVBoxLayout(self)

        area = census.get("survey_area") or {}
        facts = [f"Flights: {', '.join(census.get('flights', []))}"]
        if census.get("plot_ha") is not None:
            facts.append(
                f"survey areas {area.get('plot_ha') or 0:.2f} ha - exclusions "
                f"{area.get('excluded_ha') or 0:.2f} ha = adjusted "
                f"{census['plot_ha']:.2f} ha")
        else:
            facts.append("no survey area - densities on the covered ground")
        header = QLabel(" · ".join(facts))
        header.setWordWrap(True)
        layout.addWidget(header)

        rows = census_core.flat_rows(census)
        columns = list(rows[0].keys()) if rows else list(census_core.COLUMNS)
        titles = {
            "flight": "Flight", "species": "Species", "counted": "Counted",
            "inside": "Inside", "outside": "Outside", "unplaced": "Unplaced",
            "approved": "Approved", "plot_ha": "Plot (ha)",
            "covered_ha": "Covered (ha)", "per_km2_plot": "/km² (plot)",
            "per_km2_covered": "/km² (covered)",
        }
        table = QTableWidget(len(rows), len(columns))
        table.setHorizontalHeaderLabels([
            titles.get(c, c.replace("sex_", "♂♀ ").replace("age_", "age "))
            for c in columns])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        bold = None
        for r, row in enumerate(rows):
            total = census_core.ALL_SPECIES == row["species"] or \
                census_core.ALL_FLIGHTS == row["flight"]
            for c, column in enumerate(columns):
                value = row[column]
                if value is None:
                    text = "–"
                elif isinstance(value, float):
                    text = f"{value:.2f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                if isinstance(value, (int, float)):
                    item.setTextAlignment(_RIGHT)
                if total:
                    if bold is None:
                        bold = item.font()
                        bold.setBold(True)
                    item.setFont(bold)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        self._table = table
        layout.addWidget(table)

        notes = QLabel(census.get("notes", ""))
        notes.setWordWrap(True)
        notes.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(notes)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_btn = buttons.addButton(
            "Copy", QDialogButtonBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(self.copy_table)
        if csv_path:
            path_label = QLabel(f"Saved to: {csv_path}")
            path_label.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(path_label)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def copy_table(self):
        """Copy the table as tab-separated text (pastes into a spreadsheet)."""
        table = self._table
        lines = ["\t".join(table.horizontalHeaderItem(c).text()
                           for c in range(table.columnCount()))]
        for r in range(table.rowCount()):
            lines.append("\t".join(table.item(r, c).text()
                                   for c in range(table.columnCount())))
        QApplication.clipboard().setText("\n".join(lines))
