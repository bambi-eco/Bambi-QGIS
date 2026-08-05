# -*- coding: utf-8 -*-
"""The Project Schema dialog (EXCHANGE_FORMAT_PLAN.md §5.3, §10 Phase 2).

The vocabulary rules are covered headlessly in ``tests/test_schema_editor.py``;
what needs a real QGIS is that the dialog presents them — that base classes are
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
