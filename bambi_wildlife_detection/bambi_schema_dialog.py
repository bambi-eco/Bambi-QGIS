# -*- coding: utf-8 -*-
"""Project Schema dialog - species, enums and custom fields.

The single editor for the project vocabulary,
reached from three places:

* the Detection configuration tab, which hosts it;
* the labelling tool's gear button (custom fields);
* "Manage species…" next to any species combo.

All logic lives in :mod:`core.schema_editor`; this is a shell over it, so the
rules - protected base classes, append-only ids, no deleting what is still
referenced - are enforced whether or not the dialog is involved.

Edits are made on a private copy of the vocabulary and written into the
project when the dialog is accepted, so a half-finished taxonomy is never
persisted - and the project is not locked while the dialog is open, so a step
that finishes meanwhile can record its results.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QMessageBox, QInputDialog,
    QDialogButtonBox, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QColorDialog,
)
from qgis.PyQt.QtGui import QColor, QIcon, QPixmap

from .core import groups, store
from .core.schema_editor import SchemaEditor, SchemaError


class BambiSchemaDialog(QDialog):
    """Edit the vocabulary of the project rooted at *target_folder*."""

    def __init__(self, target_folder: str, parent=None, initial_tab: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Project Schema")
        self.setMinimumSize(560, 420)
        self.target_folder = target_folder
        self.editor = SchemaEditor(target_folder)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_species_tab(), "Species")
        self.tabs.addTab(self._build_enums_tab(), "Enums")
        self.tabs.addTab(self._build_fields_tab(), "Custom fields")
        self.tabs.setCurrentIndex(initial_tab)
        layout.addWidget(self.tabs)

        standard = QDialogButtonBox.StandardButton
        buttons = QDialogButtonBox(standard.Ok | standard.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._reload_species()
        self._reload_enums()
        self._reload_fields()

    # -- lifecycle -------------------------------------------------------

    def accept(self):
        # The edits were made on a private copy; this writes them into the
        # project. It can be refused - a step is writing just now, or the
        # vocabulary was changed elsewhere - and then the dialog stays open
        # with nothing lost.
        try:
            self.editor.commit()
        except SchemaError as exc:
            self._warn(exc)
            return
        self.editor.close()
        super().accept()

    def reject(self):
        # Nothing was written, so discarding the copy discards the edits.
        self.editor.close()
        super().reject()

    def _warn(self, exc: SchemaError) -> None:
        QMessageBox.warning(self, "Cannot apply that change", str(exc))

    # -- species ---------------------------------------------------------

    def _build_species_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(
            "Species available throughout the project. The three base classes "
            "are part of every project and cannot be changed. The colour is "
            "how a species is drawn on the map; the group distance is how "
            "far apart two animals of a species may be and still count as "
            "one group in the survey analytics."))

        self.species_list = QListWidget()
        self.species_list.itemSelectionChanged.connect(self._sync_species_buttons)
        layout.addWidget(self.species_list)

        self.species_add_btn = QPushButton("Add…")
        self.species_rename_btn = QPushButton("Rename…")
        self.species_taxonomy_btn = QPushButton("Taxonomy…")
        self.species_taxonomy_btn.setToolTip(
            "Scientific name and GBIF taxon key, used when publishing to "
            "GBIF or Camtrap DP. Optional otherwise.")
        self.species_delete_btn = QPushButton("Delete")
        self.species_colour_btn = QPushButton("Colour…")
        self.species_colour_btn.setToolTip(
            "The colour detections and tracks of this species are drawn in "
            "on the map. Any species, the base classes included.")
        self.species_colour_reset_btn = QPushButton("Default Colour")
        self.species_colour_reset_btn.setToolTip(
            "Go back to the colour the plugin gives this species.")
        self.species_colour_btn.clicked.connect(self._choose_species_colour)
        self.species_colour_reset_btn.clicked.connect(
            self._reset_species_colour)
        self.species_group_btn = QPushButton("Group Distance…")
        self.species_group_btn.setToolTip(
            "Two animals of this species no further apart than this, seen "
            "on the same pass of the drone, belong to one group - and so "
            "does everything that is joined to them in the same way. The "
            "census reports the groups, and distance sampling counts a "
            "group as one observation.\n\n"
            f"{groups.DEFAULT_DISTANCE_M:g} m unless set here; 0 for a "
            "species that is never grouped.")
        self.species_group_btn.clicked.connect(
            self._choose_species_group_distance)
        self.species_add_btn.clicked.connect(self._add_species)
        self.species_rename_btn.clicked.connect(self._rename_species)
        self.species_taxonomy_btn.clicked.connect(self._edit_taxonomy)
        self.species_delete_btn.clicked.connect(self._delete_species)

        row = QHBoxLayout()
        row.addWidget(self.species_add_btn)
        row.addWidget(self.species_rename_btn)
        row.addWidget(self.species_taxonomy_btn)
        row.addWidget(self.species_delete_btn)
        row.addStretch()
        row.addWidget(self.species_group_btn)
        row.addWidget(self.species_colour_btn)
        row.addWidget(self.species_colour_reset_btn)
        layout.addLayout(row)
        return tab

    @staticmethod
    def _group_note(metres: float) -> str:
        """What the species list says about a group distance the project
        set itself."""
        if metres <= 0:
            return "never grouped"
        return f"groups within {metres:g} m"

    @staticmethod
    def _swatch(colour: str) -> QIcon:
        pixmap = QPixmap(14, 14)
        pixmap.fill(QColor(colour))
        return QIcon(pixmap)

    def _reload_species(self) -> None:
        selected = self._selected_species_id()
        self.species_list.clear()
        colours = self.editor.species_colours()
        distances = self.editor.species_group_distances()
        for species in self.editor.species():
            label = f"{species['species_id']:>3}  {species['name']}"
            if species["protected"]:
                label += "   (base class)"
            elif species.get("scientific_name"):
                label += f"   - {species['scientific_name']}"
                if species.get("gbif_taxon_key"):
                    label += f" [GBIF {species['gbif_taxon_key']}]"
            if self.editor.has_own_group_distance(species["species_id"]):
                label += "   · " + self._group_note(
                    distances[species["species_id"]])
            item = QListWidgetItem(
                self._swatch(colours[species["species_id"]]), label)
            item.setData(Qt.ItemDataRole.UserRole, species["species_id"])
            if species["protected"]:
                item.setForeground(Qt.GlobalColor.gray)
            self.species_list.addItem(item)
            if species["species_id"] == selected:
                self.species_list.setCurrentItem(item)
        self._sync_species_buttons()

    def _selected_species_id(self):
        item = self.species_list.currentItem()
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _sync_species_buttons(self) -> None:
        species_id = self._selected_species_id()
        editable = species_id is not None and species_id > 0
        self.species_rename_btn.setEnabled(editable)
        self.species_taxonomy_btn.setEnabled(editable)
        self.species_delete_btn.setEnabled(editable)
        # A colour changes how a species looks, not what it means, so the
        # base classes can have one too.
        self.species_colour_btn.setEnabled(species_id is not None)
        own = species_id is not None and self.editor.has_own_colour(species_id)
        self.species_colour_reset_btn.setEnabled(own)
        # ... and so does the group distance: undetermined animals stand
        # together like any others.
        self.species_group_btn.setEnabled(species_id is not None)

    def _choose_species_group_distance(self) -> None:
        species_id = self._selected_species_id()
        if species_id is None:
            return
        name = next(row["name"] for row in self.editor.species()
                    if row["species_id"] == species_id)
        current = self.editor.species_group_distances()[species_id]
        metres, ok = QInputDialog.getDouble(
            self, "Group distance",
            f"How far apart may two animals of '{name}' be and still belong "
            "to one group?\n\nIn metres; 0 = never grouped. "
            f"The plugin's own value is {groups.DEFAULT_DISTANCE_M:g} m.",
            current, 0.0, groups.MAX_DISTANCE_M, 1)
        if not ok:
            return
        try:
            # the plugin's own value is not a choice of the project's
            self.editor.set_species_group_distance(
                species_id,
                None if metres == groups.DEFAULT_DISTANCE_M else metres)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_species()

    def _choose_species_colour(self) -> None:
        species_id = self._selected_species_id()
        if species_id is None:
            return
        current = QColor(self.editor.species_colours()[species_id])
        chosen = QColorDialog.getColor(current, self, "Species colour")
        if not chosen.isValid():
            return
        try:
            self.editor.set_species_colour(species_id, chosen.name())
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_species()

    def _reset_species_colour(self) -> None:
        species_id = self._selected_species_id()
        if species_id is None:
            return
        self.editor.set_species_colour(species_id, None)
        self._reload_species()

    def _add_species(self) -> None:
        name, ok = QInputDialog.getText(self, "Add species", "Name:")
        if not ok:
            return
        try:
            self.editor.add_species(name)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_species()
        self._reload_fields()

    def _rename_species(self) -> None:
        species_id = self._selected_species_id()
        if species_id is None:
            return
        current = [s for s in self.editor.species()
                   if s["species_id"] == species_id][0]["name"]
        name, ok = QInputDialog.getText(
            self, "Rename species", "Name:", text=current)
        if not ok:
            return
        try:
            self.editor.rename_species(species_id, name)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_species()

    def _edit_taxonomy(self) -> None:
        """Scientific name, rank and GBIF taxon key for the selected species."""
        species_id = self._selected_species_id()
        if species_id is None:
            return
        current = [s for s in self.editor.species()
                   if s["species_id"] == species_id][0]

        scientific_name, ok = QInputDialog.getText(
            self, "Taxonomy",
            f"Scientific name for '{current['name']}' - Darwin Core publishes "
            "this; the name above is the vernacular one:",
            text=current.get("scientific_name") or "")
        if not ok:
            return

        rank, ok = QInputDialog.getText(
            self, "Taxonomy", "Taxon rank (species, genus, class…):",
            text=current.get("taxon_rank") or "species")
        if not ok:
            return

        key_text, ok = QInputDialog.getText(
            self, "Taxonomy",
            "GBIF taxon key (optional - the number from the species' page on "
            "gbif.org):",
            text=str(current.get("gbif_taxon_key") or ""))
        if not ok:
            return

        try:
            self.editor.set_taxonomy(
                species_id, scientific_name, rank,
                int(key_text) if key_text.strip() else None)
        except (SchemaError, ValueError) as exc:
            self._warn(SchemaError(str(exc)))
            return
        self._reload_species()

    def _delete_species(self) -> None:
        species_id = self._selected_species_id()
        if species_id is None:
            return
        try:
            self.editor.delete_species(species_id)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_species()

    # -- enums -----------------------------------------------------------

    def _build_enums_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(
            "Reusable value sets. Renaming a value is safe - stored data holds "
            "the value's id, not its label."))

        picker = QHBoxLayout()
        self.enum_combo = QComboBox()
        self.enum_combo.currentIndexChanged.connect(self._reload_enum_values)
        picker.addWidget(self.enum_combo, 1)
        add_enum_btn = QPushButton("New enum…")
        add_enum_btn.clicked.connect(self._add_enum)
        picker.addWidget(add_enum_btn)
        layout.addLayout(picker)

        self.enum_values_list = QListWidget()
        self.enum_values_list.itemSelectionChanged.connect(
            self._sync_enum_buttons)
        layout.addWidget(self.enum_values_list)

        self.enum_add_btn = QPushButton("Add value…")
        self.enum_rename_btn = QPushButton("Rename…")
        self.enum_delete_btn = QPushButton("Delete")
        self.enum_up_btn = QPushButton("Move up")
        self.enum_down_btn = QPushButton("Move down")
        self.enum_add_btn.clicked.connect(self._add_enum_value)
        self.enum_rename_btn.clicked.connect(self._rename_enum_value)
        self.enum_delete_btn.clicked.connect(self._delete_enum_value)
        self.enum_up_btn.clicked.connect(lambda: self._move_enum_value(-1))
        self.enum_down_btn.clicked.connect(lambda: self._move_enum_value(1))

        row = QHBoxLayout()
        for button in (self.enum_add_btn, self.enum_rename_btn,
                       self.enum_delete_btn, self.enum_up_btn,
                       self.enum_down_btn):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        return tab

    def _reload_enums(self) -> None:
        self.enum_combo.blockSignals(True)
        self.enum_combo.clear()
        for enum in self.editor.enums():
            self.enum_combo.addItem(enum["name"], enum["enum_id"])
        self.enum_combo.blockSignals(False)
        self._reload_enum_values()

    def _current_enum_id(self):
        index = self.enum_combo.currentIndex()
        return None if index < 0 else self.enum_combo.itemData(index)

    def _reload_enum_values(self) -> None:
        self.enum_values_list.clear()
        enum_id = self._current_enum_id()
        if enum_id is None:
            self._sync_enum_buttons()
            return
        for value in self.editor.enum_values(enum_id):
            item = QListWidgetItem(f"{value['value_id']:>3}  {value['label']}")
            item.setData(Qt.ItemDataRole.UserRole, value["value_id"])
            self.enum_values_list.addItem(item)
        self._sync_enum_buttons()

    def _selected_value_id(self):
        item = self.enum_values_list.currentItem()
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _sync_enum_buttons(self) -> None:
        has_enum = self._current_enum_id() is not None
        has_value = self._selected_value_id() is not None
        self.enum_add_btn.setEnabled(has_enum)
        for button in (self.enum_rename_btn, self.enum_delete_btn,
                       self.enum_up_btn, self.enum_down_btn):
            button.setEnabled(has_value)

    def _add_enum(self) -> None:
        name, ok = QInputDialog.getText(self, "New enum", "Name:")
        if not ok:
            return
        try:
            self.editor.add_enum(name)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_enums()
        self.enum_combo.setCurrentIndex(self.enum_combo.count() - 1)

    def _add_enum_value(self) -> None:
        enum_id = self._current_enum_id()
        if enum_id is None:
            return
        label, ok = QInputDialog.getText(self, "Add value", "Label:")
        if not ok:
            return
        try:
            self.editor.add_enum_value(enum_id, label)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_enum_values()

    def _rename_enum_value(self) -> None:
        enum_id = self._current_enum_id()
        value_id = self._selected_value_id()
        if enum_id is None or value_id is None:
            return
        current = [v for v in self.editor.enum_values(enum_id)
                   if v["value_id"] == value_id][0]["label"]
        label, ok = QInputDialog.getText(
            self, "Rename value", "Label:", text=current)
        if not ok:
            return
        try:
            self.editor.rename_enum_value(enum_id, value_id, label)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_enum_values()

    def _delete_enum_value(self) -> None:
        enum_id = self._current_enum_id()
        value_id = self._selected_value_id()
        if enum_id is None or value_id is None:
            return
        try:
            self.editor.delete_enum_value(enum_id, value_id)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_enum_values()

    def _move_enum_value(self, delta: int) -> None:
        enum_id = self._current_enum_id()
        value_id = self._selected_value_id()
        if enum_id is None or value_id is None:
            return
        order = [v["value_id"] for v in self.editor.enum_values(enum_id)]
        index = order.index(value_id)
        target = index + delta
        if target < 0 or target >= len(order):
            return
        order[index], order[target] = order[target], order[index]
        try:
            self.editor.reorder_enum_values(enum_id, order)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_enum_values()
        self.enum_values_list.setCurrentRow(target)

    # -- custom fields ---------------------------------------------------

    def _build_fields_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(
            "Extra attributes carried through the whole pipeline. Enum fields "
            "avoid typos; string fields stay available for free-form notes."))

        self.fields_table = QTableWidget(0, 4)
        self.fields_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Scope", "Enum"])
        self.fields_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.fields_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.fields_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fields_table.itemSelectionChanged.connect(self._sync_field_buttons)
        layout.addWidget(self.fields_table)

        self.field_add_btn = QPushButton("Add…")
        self.field_rename_btn = QPushButton("Rename…")
        self.field_delete_btn = QPushButton("Delete")
        self.field_add_btn.clicked.connect(self._add_field)
        self.field_rename_btn.clicked.connect(self._rename_field)
        self.field_delete_btn.clicked.connect(self._delete_field)

        row = QHBoxLayout()
        row.addWidget(self.field_add_btn)
        row.addWidget(self.field_rename_btn)
        row.addWidget(self.field_delete_btn)
        row.addStretch()
        layout.addLayout(row)
        return tab

    def _reload_fields(self) -> None:
        fields = self.editor.fields()
        enum_names = {e["enum_id"]: e["name"] for e in self.editor.enums()}
        self.fields_table.setRowCount(len(fields))
        for row, field in enumerate(fields):
            protected = bool(field["protected"])
            values = [
                field["name"] + ("   (built-in)" if protected else ""),
                field["type"],
                field["scope"],
                enum_names.get(field["enum_id"], ""),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole,
                             (field["name"], field["scope"], protected))
                if protected:
                    item.setForeground(Qt.GlobalColor.gray)
                self.fields_table.setItem(row, column, item)
        self._sync_field_buttons()

    def _selected_field(self):
        row = self.fields_table.currentRow()
        if row < 0:
            return None
        item = self.fields_table.item(row, 0)
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _sync_field_buttons(self) -> None:
        selected = self._selected_field()
        editable = selected is not None and not selected[2]
        self.field_rename_btn.setEnabled(editable)
        self.field_delete_btn.setEnabled(editable)

    def _add_field(self) -> None:
        name, ok = QInputDialog.getText(self, "Add field", "Name:")
        if not ok:
            return
        types = list(store.FIELD_TYPES)
        type_name, ok = QInputDialog.getItem(
            self, "Add field", "Type:", types, types.index("string"), False)
        if not ok:
            return
        scopes = list(store.FIELD_SCOPES)
        scope, ok = QInputDialog.getItem(
            self, "Add field", "Scope:", scopes, scopes.index("track"), False)
        if not ok:
            return

        enum_id = None
        if type_name == "enum":
            enums = self.editor.enums()
            if not enums:
                QMessageBox.warning(
                    self, "No enums yet",
                    "Define an enum first - an enum field draws its values "
                    "from one.")
                return
            names = [e["name"] for e in enums]
            chosen, ok = QInputDialog.getItem(
                self, "Add field", "Values from:", names, 0, False)
            if not ok:
                return
            enum_id = enums[names.index(chosen)]["enum_id"]

        try:
            self.editor.add_field(name, type_name, scope, enum_id)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_fields()

    def _rename_field(self) -> None:
        selected = self._selected_field()
        if selected is None:
            return
        name, scope, _ = selected
        new_name, ok = QInputDialog.getText(
            self, "Rename field", "Name:", text=name)
        if not ok:
            return
        try:
            rewritten = self.editor.rename_field(name, scope, new_name)
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_fields()
        if rewritten:
            QMessageBox.information(
                self, "Field renamed",
                f"Renamed '{name}' to '{new_name}'. Its {rewritten} stored "
                "value(s) follow when you press OK.")

    def _delete_field(self) -> None:
        selected = self._selected_field()
        if selected is None:
            return
        name, scope, _ = selected
        usage = self.editor.field_usage(name, scope)
        if usage:
            total = sum(usage.values())
            standard = QMessageBox.StandardButton
            answer = QMessageBox.question(
                self, "Delete field",
                f"'{name}' still holds {total} value(s). Deleting the field "
                "discards them when you press OK. Continue?",
                standard.Ok | standard.Cancel, standard.Cancel)
            if answer != standard.Ok:
                return
        try:
            self.editor.delete_field(name, scope, force=bool(usage))
        except SchemaError as exc:
            self._warn(exc)
            return
        self._reload_fields()
