# -*- coding: utf-8 -*-
"""The classifier mapping table and its two dialogs, in a real QGIS.

What matters here is that the mapping survives a project round trip and stays
keyed on class *index* — a model returns positions, so a mapping keyed on
names would silently re-point itself the moment a label was edited.
"""
import pytest

from qgis.PyQt.QtWidgets import QTableWidget

from bambi_wildlife_detection.bambi_label_mapping_dialog import (
    BambiLabelMappingDialog,
)
from bambi_wildlife_detection.bambi_sex_model_dialog import (
    DEFAULT_SPECIES, BambiSexModelDialog,
)
from bambi_wildlife_detection.core import hf_access, store


@pytest.fixture
def project_folder(tmp_path):
    """A flight folder with the seeded vocabulary."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    return root


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def test_the_table_has_a_row_per_task(dock):
    table = dock.classification_models_table
    assert table.rowCount() == 3
    assert [table.item(r, 0).text() for r in range(3)] == \
        ["Occlusion", "Species", "Sex"]


def test_the_tasks_are_in_the_order_they_run(dock):
    """Occlusion gates the frames; sex reuses the species frames."""
    table = dock.classification_models_table
    names = [table.item(r, 0).text().lower() for r in range(3)]
    assert names == list(hf_access.TASKS)


def test_species_cannot_be_set_to_a_default_that_does_not_exist(dock):
    """No species model is published yet, so offering 'Default' would only
    fail later."""
    row = dock._task_row("species")
    combo = dock.classification_models_table.cellWidget(row, 2)
    index = combo.findData("default")
    assert not combo.model().item(index).isEnabled()
    assert "not released" in combo.itemText(index)
    assert combo.currentData() == "off"


def test_released_tasks_default_to_their_published_model(dock):
    for task in ("occlusion", "sex"):
        combo = dock.classification_models_table.cellWidget(
            dock._task_row(task), 2)
        assert combo.currentData() == "default"


def test_matched_is_the_default_input(dock):
    """Fusion is where the two sensors complement each other most."""
    for task in hf_access.TASKS:
        combo = dock.classification_models_table.cellWidget(
            dock._task_row(task), 1)
        assert combo.currentData() == "matched"


def test_editing_the_table_updates_the_saved_mapping(dock):
    row = dock._task_row("occlusion")
    table = dock.classification_models_table
    table.cellWidget(row, 1).setCurrentIndex(1)      # RGB

    spec = dock._classification_spec("occlusion")
    assert spec["modality"] == "rgb"


def test_the_mapping_survives_a_project_round_trip(dock):
    row = dock._task_row("sex")
    table = dock.classification_models_table
    table.cellWidget(row, 1).setCurrentIndex(0)      # thermal
    table.item(row, 3).setText("/models/sex.pt")
    table.cellWidget(row, 2).setCurrentIndex(2)      # custom

    dock.target_folder_edit.setText("/tmp/bambi_mapping_roundtrip")
    dock.save_config_to_project()

    table.cellWidget(row, 1).setCurrentIndex(2)
    table.item(row, 3).setText("")
    dock.load_config_from_project()

    assert table.cellWidget(row, 1).currentData() == "thermal"
    assert table.cellWidget(row, 2).currentData() == "custom"
    assert table.item(row, 3).text() == "/models/sex.pt"


def test_the_config_reports_every_task(dock):
    models = dock.get_config()["classification_models"]
    assert set(models) == set(hf_access.TASKS)


# ---------------------------------------------------------------------------
# Label mapping dialog
# ---------------------------------------------------------------------------

def test_label_dialog_offers_the_project_vocabulary(project_folder):
    dialog = BambiLabelMappingDialog(
        "occlusion", {"modality": "rgb", "model": "default"}, project_folder)
    dialog._add_row("clear", None)
    combo = dialog.table.cellWidget(0, 2)
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert "clear" in labels and "occluded" in labels


def test_label_dialog_defaults_to_an_exact_name_match(project_folder):
    """With the 6.1 vocabulary the occlusion mapping fills itself in."""
    dialog = BambiLabelMappingDialog(
        "occlusion", {"modality": "rgb", "model": "default"}, project_folder)
    dialog._add_row("occluded", None)
    assert dialog.table.cellWidget(0, 2).currentText() == "occluded"


def test_label_dialog_maps_species_onto_the_species_list(project_folder):
    dialog = BambiLabelMappingDialog(
        "species", {"modality": "rgb", "model": "custom"}, project_folder)
    dialog._add_row("red deer", None)
    assert dialog.table.cellWidget(0, 2).currentText() == "red deer"


def test_label_mapping_is_keyed_on_index_not_name(project_folder):
    dialog = BambiLabelMappingDialog(
        "occlusion", {"modality": "rgb", "model": "default"}, project_folder)
    dialog._add_row("clear", None)
    dialog._add_row("occluded", None)

    # Rename the labels; the mapping must follow the positions, not the text.
    dialog.table.item(0, 1).setText("visible")
    dialog.table.item(1, 1).setText("hidden")
    spec = dialog.result_spec()

    assert spec["class_labels"] == ["visible", "hidden"]
    assert set(spec["labels"]) == {"0", "1"}


def test_a_manual_mapping_needs_no_model_at_all(project_folder):
    """The escape hatch for a model that is not downloaded yet."""
    dialog = BambiLabelMappingDialog(
        "sex", {"modality": "matched", "model": "custom", "path": ""},
        project_folder)
    dialog._add_row("female_juvenile", None)
    dialog._add_row("male", None)
    spec = dialog.result_spec()

    assert len(spec["class_labels"]) == 2
    assert spec["labels"]["1"] is not None


def test_unmapped_classes_are_simply_absent(project_folder):
    dialog = BambiLabelMappingDialog(
        "sex", {"modality": "matched", "model": "custom"}, project_folder)
    dialog._add_row("something unknown", None)
    assert dialog.result_spec()["labels"] == {}


def test_occlusion_records_which_class_means_usable(project_folder):
    """Frame selection needs this; guessing it from a name would break the
    moment someone used a model that says 'visible'."""
    dialog = BambiLabelMappingDialog(
        "occlusion", {"modality": "rgb", "model": "default"}, project_folder)
    dialog._add_row("visible", None)
    dialog._add_row("hidden", None)
    # Map "visible" onto the project's first occlusion value.
    dialog.table.cellWidget(0, 2).setCurrentIndex(1)
    spec = dialog.result_spec()

    assert spec["clear_labels"] == ["visible"]
    assert spec["clear_values"] == [0]


def test_a_saved_mapping_is_reloaded(project_folder):
    spec = {"modality": "rgb", "model": "default",
            "class_labels": ["clear", "occluded"], "labels": {"0": 0, "1": 1}}
    dialog = BambiLabelMappingDialog("occlusion", spec, project_folder)
    assert dialog.table.rowCount() == 2
    assert dialog.table.cellWidget(1, 2).currentData() == 1


# ---------------------------------------------------------------------------
# Per-species sex dialog
# ---------------------------------------------------------------------------

def test_sex_dialog_lists_the_projects_species(project_folder):
    dialog = BambiSexModelDialog({}, project_folder)
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    assert "red deer" in names and "wild boar" in names


def test_base_classes_are_not_offered(project_folder):
    """'animal' and 'unknown' are not species anyone has a model for."""
    dialog = BambiSexModelDialog({}, project_folder)
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    for base in ("animal", "unknown", "not-an-animal"):
        assert base not in names


def test_only_red_deer_defaults_to_the_published_model(project_folder):
    dialog = BambiSexModelDialog({}, project_folder)
    for row in range(dialog.table.rowCount()):
        species = dialog.table.item(row, 0).text()
        chosen = dialog.table.cellWidget(row, 1).currentData()
        assert chosen == ("default" if species == DEFAULT_SPECIES else "off")


def test_species_left_off_are_absent_from_the_result(project_folder):
    """Absent means 'not sexed', which is the honest answer for a species with
    no classifier."""
    dialog = BambiSexModelDialog({}, project_folder)
    per_species = dialog.result_spec()["species"]
    assert set(per_species) == {DEFAULT_SPECIES}


def test_a_custom_model_per_species_round_trips(project_folder):
    dialog = BambiSexModelDialog({}, project_folder)
    row = next(r for r in range(dialog.table.rowCount())
               if dialog.table.item(r, 0).text() == "wild boar")
    combo = dialog.table.cellWidget(row, 1)
    combo.setCurrentIndex(combo.findData("custom"))
    dialog.table.item(row, 2).setText("/models/boar_sex.pt")

    per_species = dialog.result_spec()["species"]
    assert per_species["wild boar"] == {"model": "custom",
                                        "path": "/models/boar_sex.pt"}


def test_a_saved_selection_is_reloaded(project_folder):
    spec = {"species": {"wild boar": {"model": "custom", "path": "/x.pt"}}}
    dialog = BambiSexModelDialog(spec, project_folder)
    row = next(r for r in range(dialog.table.rowCount())
               if dialog.table.item(r, 0).text() == "wild boar")
    assert dialog.table.cellWidget(row, 1).currentData() == "custom"
    assert dialog.table.item(row, 2).text() == "/x.pt"


def test_a_project_without_a_store_still_offers_red_deer(tmp_path):
    dialog = BambiSexModelDialog({}, str(tmp_path))
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    assert names == [DEFAULT_SPECIES]


def test_dialogs_are_tables_not_free_text(project_folder):
    """Typed species names were what made class ids unstable before 6.0."""
    assert isinstance(BambiSexModelDialog({}, project_folder).table,
                      QTableWidget)
