# -*- coding: utf-8 -*-
"""The Project Schema dialog.

The vocabulary rules are covered headlessly in ``tests/test_schema_editor.py``;
what needs a real QGIS is that the dialog presents them - that base classes are
not editable through the UI, that accepting persists and cancelling does not,
and that the Project tab can open it.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QDialog

from bambi_wildlife_detection.bambi_schema_dialog import BambiSchemaDialog
from bambi_wildlife_detection.core import store
from bambi_wildlife_detection.core.schema_editor import SchemaEditor


@pytest.fixture
def project_folder(tmp_path):
    root = str(tmp_path / "flight")
    os.makedirs(root, exist_ok=True)
    store.open_store(store.project_path(root), store.PROJECT).close()
    return root


@pytest.fixture
def dialog(project_folder):
    dlg = BambiSchemaDialog(project_folder)
    yield dlg
    dlg.editor.close()
    dlg.deleteLater()


def _species_names(folder):
    with SchemaEditor(folder) as editor:
        return {s["name"] for s in editor.species()}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_dialog_has_the_three_tabs(dialog):
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == \
        ["Species", "Enums", "Custom fields"]


def test_dialog_opens_on_the_requested_tab(project_folder):
    dlg = BambiSchemaDialog(project_folder, initial_tab=2)
    assert dlg.tabs.currentIndex() == 2
    dlg.editor.close()


def test_species_list_shows_the_base_classes(dialog):
    labels = [dialog.species_list.item(i).text()
              for i in range(dialog.species_list.count())]
    assert any("animal" in text and "base class" in text for text in labels)
    assert any("roe deer" in text for text in labels)


def test_enum_combo_lists_the_seeded_enums(dialog):
    names = [dialog.enum_combo.itemText(i)
             for i in range(dialog.enum_combo.count())]
    assert set(names) == {"sex", "age", "occlusion"}


def test_enum_values_are_listed_in_display_order(dialog):
    index = [dialog.enum_combo.itemText(i)
             for i in range(dialog.enum_combo.count())].index("sex")
    dialog.enum_combo.setCurrentIndex(index)
    labels = [dialog.enum_values_list.item(i).text()
              for i in range(dialog.enum_values_list.count())]
    assert [text.split()[-1] for text in labels] == ["unknown", "female", "male"]


def test_fields_table_lists_the_seeded_fields(dialog):
    names = [dialog.fields_table.item(row, 0).text()
             for row in range(dialog.fields_table.rowCount())]
    assert any("sex" in text and "built-in" in text for text in names)
    assert any("occlusion" in text for text in names)


# ---------------------------------------------------------------------------
# Protection is visible in the UI, not only in the model
# ---------------------------------------------------------------------------

def test_base_species_cannot_be_edited_from_the_ui(dialog):
    dialog.species_list.setCurrentRow(0)          # -2, a base class
    assert not dialog.species_rename_btn.isEnabled()
    assert not dialog.species_delete_btn.isEnabled()


def test_concrete_species_can_be_edited_from_the_ui(dialog):
    dialog.editor.add_species("wolf")
    dialog._reload_species()
    dialog.species_list.setCurrentRow(dialog.species_list.count() - 1)
    assert dialog.species_rename_btn.isEnabled()
    assert dialog.species_delete_btn.isEnabled()


def test_builtin_fields_cannot_be_edited_from_the_ui(dialog):
    for row in range(dialog.fields_table.rowCount()):
        if "built-in" in dialog.fields_table.item(row, 0).text():
            dialog.fields_table.setCurrentCell(row, 0)
            break
    assert not dialog.field_rename_btn.isEnabled()
    assert not dialog.field_delete_btn.isEnabled()


def test_enum_buttons_need_a_selected_value(dialog):
    dialog.enum_values_list.setCurrentRow(-1)
    dialog._sync_enum_buttons()
    assert not dialog.enum_rename_btn.isEnabled()
    assert dialog.enum_add_btn.isEnabled()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_accepting_persists_the_edits(project_folder):
    dlg = BambiSchemaDialog(project_folder)
    dlg.editor.add_species("wolf")
    dlg.accept()
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert "wolf" in _species_names(project_folder)


def test_cancelling_discards_the_edits(project_folder):
    dlg = BambiSchemaDialog(project_folder)
    dlg.editor.add_species("wolf")
    dlg.reject()
    assert "wolf" not in _species_names(project_folder)


def test_reordering_an_enum_persists(project_folder):
    dlg = BambiSchemaDialog(project_folder)
    index = [dlg.enum_combo.itemText(i)
             for i in range(dlg.enum_combo.count())].index("age")
    dlg.enum_combo.setCurrentIndex(index)
    dlg.enum_values_list.setCurrentRow(0)
    dlg._move_enum_value(1)
    dlg.accept()

    with SchemaEditor(project_folder) as editor:
        enum_id = editor.enum_by_name("age")["enum_id"]
        assert [v["label"] for v in editor.enum_values(enum_id)][0] == "adult"


def test_reordering_keeps_the_value_ids(project_folder):
    """Display order moves; ids do not, so stored rows are unaffected."""
    dlg = BambiSchemaDialog(project_folder)
    index = [dlg.enum_combo.itemText(i)
             for i in range(dlg.enum_combo.count())].index("age")
    dlg.enum_combo.setCurrentIndex(index)
    dlg.enum_values_list.setCurrentRow(0)
    dlg._move_enum_value(1)
    dlg.accept()

    with SchemaEditor(project_folder) as editor:
        enum_id = editor.enum_by_name("age")["enum_id"]
        values = {v["label"]: v["value_id"] for v in editor.enum_values(enum_id)}
        assert values == {"unknown": 0, "adult": 1, "juvenile": 2}


def test_moving_past_the_end_is_a_no_op(project_folder):
    dlg = BambiSchemaDialog(project_folder)
    dlg.enum_values_list.setCurrentRow(0)
    dlg._move_enum_value(-1)
    dlg.accept()


# ---------------------------------------------------------------------------
# Reached from the dock
# ---------------------------------------------------------------------------

def test_dock_refuses_to_open_the_schema_without_a_folder(dock, monkeypatch):
    """A modal warning would abort the headless run, so it is stubbed out."""
    from qgis.PyQt.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: shown.append(args))
    dock.target_folder_edit.setText("")
    assert dock.open_schema_dialog() is False
    assert shown, "the user should be told why nothing opened"


def test_dock_has_the_schema_button(dock):
    assert dock.schema_editor_btn is not None


# ---------------------------------------------------------------------------
# Where the editor is reached from
# ---------------------------------------------------------------------------

def _config_tabs(dock):
    from qgis.PyQt.QtWidgets import QTabWidget

    for widget in dock.findChildren(QTabWidget):
        names = [widget.tabText(i) for i in range(widget.count())]
        if "Extraction" in names:
            return widget
    raise AssertionError("configuration tab widget not found")


def test_the_project_tab_comes_first(dock):
    """The schema describes the survey, so it precedes the steps using it."""
    tabs = _config_tabs(dock)
    assert tabs.tabText(0) == "Project"


def test_the_schema_button_lives_on_the_project_tab(dock):
    tabs = _config_tabs(dock)
    project = tabs.widget(0)
    assert project.isAncestorOf(dock.schema_editor_btn)


def test_the_schema_button_left_the_detection_tab(dock):
    tabs = _config_tabs(dock)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    detection = tabs.widget(names.index("Detection"))
    assert not detection.isAncestorOf(dock.schema_editor_btn)


def test_the_detection_settings_are_untouched(dock):
    tabs = _config_tabs(dock)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    detection = tabs.widget(names.index("Detection"))
    assert detection.isAncestorOf(dock.confidence_spin)
    assert detection.isAncestorOf(dock.thermal_model_path_edit)


# ---------------------------------------------------------------------------
# The project is not locked while the dialog is open
# ---------------------------------------------------------------------------

def test_a_step_can_record_its_results_while_the_dialog_is_open(
        dialog, project_folder):
    """The dialog kept its edits in an open transaction until it was
    closed; a step that finished meanwhile got "database is locked"."""
    import sqlite3

    dialog.editor.add_species("lynx")                 # an edit is pending
    step = sqlite3.connect(store.project_path(project_folder), timeout=0.2)
    try:
        step.execute("INSERT OR REPLACE INTO bambi_meta (key, value) "
                     "VALUES ('a step', 'finished')")
        step.commit()
    finally:
        step.close()

    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert "lynx" in _species_names(project_folder)


def test_a_save_that_is_refused_keeps_the_dialog_open(dialog, project_folder,
                                                      monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    dialog.editor.add_species("lynx")
    with SchemaEditor(project_folder) as elsewhere:
        elsewhere.add_species("wolf")
        elsewhere.commit()

    dialog.accept()

    assert warned and "changed by something else" in str(warned[0])
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "lynx" not in _species_names(project_folder)
    assert dialog.editor.conn is not None             # still editable


# ---------------------------------------------------------------------------
# Group distance (core.groups)
# ---------------------------------------------------------------------------

def _select(dialog, name):
    for row in range(dialog.species_list.count()):
        if f"  {name}" in dialog.species_list.item(row).text():
            dialog.species_list.setCurrentRow(row)
            return dialog.species_list.item(row)
    raise AssertionError(f"{name} is not in the list")


def _row_of(dialog, name):
    return next(dialog.species_list.item(row).text()
                for row in range(dialog.species_list.count())
                if f"  {name}" in dialog.species_list.item(row).text())


def test_a_group_distance_can_be_set_for_any_species(dialog, monkeypatch):
    from qgis.PyQt.QtWidgets import QInputDialog

    assert not dialog.species_group_btn.isEnabled()       # nothing selected
    asked = []

    def answer(parent, title, text, value, low, high, decimals):
        asked.append((title, value, low, high))
        return 20.0, True

    monkeypatch.setattr(QInputDialog, "getDouble", answer)
    _select(dialog, "roe deer")
    assert dialog.species_group_btn.isEnabled()
    dialog.species_group_btn.click()

    assert asked == [("Group distance", 50.0, 0.0, 5000.0)]
    assert "groups within 20 m" in _row_of(dialog, "roe deer")
    assert "groups within" not in _row_of(dialog, "red deer")
    # the base classes too: undetermined animals stand together as well
    _select(dialog, "animal")
    assert dialog.species_group_btn.isEnabled()


def test_zero_reads_as_never_grouped(dialog, monkeypatch):
    from qgis.PyQt.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getDouble",
                        lambda *args: (0.0, True))
    _select(dialog, "fox")
    dialog.species_group_btn.click()

    assert "never grouped" in _row_of(dialog, "fox")


def test_the_plugins_own_value_is_no_choice_of_the_project(dialog,
                                                           monkeypatch):
    from qgis.PyQt.QtWidgets import QInputDialog

    answers = iter([(20.0, True), (50.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        lambda *args: next(answers))
    _select(dialog, "roe deer")
    dialog.species_group_btn.click()
    _select(dialog, "roe deer")
    dialog.species_group_btn.click()

    assert "groups within" not in _row_of(dialog, "roe deer")
    roe = dialog.editor.species_by_name("roe deer")["species_id"]
    assert not dialog.editor.has_own_group_distance(roe)


def test_a_cancelled_question_changes_nothing(dialog, monkeypatch):
    from qgis.PyQt.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getDouble",
                        lambda *args: (20.0, False))
    _select(dialog, "roe deer")
    dialog.species_group_btn.click()

    assert "groups within" not in _row_of(dialog, "roe deer")


def test_the_group_distance_is_saved_with_the_dialog(project_folder,
                                                     monkeypatch):
    from qgis.PyQt.QtWidgets import QInputDialog
    from bambi_wildlife_detection.core import groups

    monkeypatch.setattr(QInputDialog, "getDouble",
                        lambda *args: (35.5, True))
    dlg = BambiSchemaDialog(project_folder)
    _select(dlg, "wild boar")
    dlg.species_group_btn.click()
    assert groups.species_distances(project_folder)["wild boar"] == 50.0

    dlg.accept()

    assert groups.species_distances(project_folder)["wild boar"] == 35.5


# ---------------------------------------------------------------------------
# ... and the other flights of the project
# ---------------------------------------------------------------------------

@pytest.fixture
def two_flights(dock, tmp_path):
    from bambi_wildlife_detection.core import flights

    folders = []
    for name in ("meadow", "forest"):
        folder = str(tmp_path / name)
        os.makedirs(folder)
        store.open_store(store.project_path(folder), store.PROJECT).close()
        folders.append(folder)
    listed = flights.add_flight([], "Meadow", folders[0])
    listed = flights.add_flight(listed, "Forest", folders[1])
    dock._flight_list = listed
    dock._active_flight_index = 0
    dock.target_folder_edit.blockSignals(True)
    dock.target_folder_edit.setText(folders[0])
    dock.target_folder_edit.blockSignals(False)
    return folders


def _schema_dialog_that_sets(monkeypatch, name, metres):
    def set_and_accept(self):
        species = self.editor.species_by_name(name)["species_id"]
        self.editor.set_species_group_distance(species, metres)
        self.accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(BambiSchemaDialog, "exec", set_and_accept)


def test_a_changed_group_distance_is_offered_to_the_other_flights(
        dock, two_flights, monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox
    from bambi_wildlife_detection.core import groups

    meadow, forest = two_flights
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda parent, title, text, *rest: asked.append((title, text))
        or QMessageBox.StandardButton.Yes)
    _schema_dialog_that_sets(monkeypatch, "roe deer", 20)

    assert dock.open_schema_dialog() is True

    assert len(asked) == 1 and asked[0][0] == "Group distances"
    assert "other 1 flight(s)" in asked[0][1]
    assert groups.species_distances(meadow)["roe deer"] == 20.0
    assert groups.species_distances(forest)["roe deer"] == 20.0
    assert "carried over to 1 other flight(s)" in dock.log_text.toPlainText()


def test_saying_no_leaves_the_other_flights_as_they_are(
        dock, two_flights, monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox
    from bambi_wildlife_detection.core import groups

    meadow, forest = two_flights
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args: QMessageBox.StandardButton.No)
    _schema_dialog_that_sets(monkeypatch, "roe deer", 20)

    dock.open_schema_dialog()

    assert groups.species_distances(meadow)["roe deer"] == 20.0
    assert groups.species_distances(forest)["roe deer"] == 50.0


def test_nothing_is_asked_when_no_distance_changed(dock, two_flights,
                                                   monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox

    def unexpected(*args):
        raise AssertionError("asked although no group distance changed")

    monkeypatch.setattr(QMessageBox, "question", unexpected)

    def rename_and_accept(self):
        self.editor.add_species("wolf")
        self.accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(BambiSchemaDialog, "exec", rename_and_accept)

    assert dock.open_schema_dialog() is True


def test_nothing_is_asked_in_a_project_with_one_flight(dock, two_flights,
                                                       monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox

    dock._flight_list = dock._flight_list[:1]

    def unexpected(*args):
        raise AssertionError("asked although there is no other flight")

    monkeypatch.setattr(QMessageBox, "question", unexpected)
    _schema_dialog_that_sets(monkeypatch, "roe deer", 20)

    assert dock.open_schema_dialog() is True
