# -*- coding: utf-8 -*-
"""The classifier mapping table and its two dialogs, in a real QGIS.

What matters here is that the mapping survives a project round trip and stays
keyed on class *index* — a model returns positions, so a mapping keyed on
names would silently re-point itself the moment a label was edited.
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QTableWidget

from bambi_wildlife_detection.bambi_label_mapping_dialog import (
    BambiLabelMappingDialog,
)
from bambi_wildlife_detection.bambi_classification_model_dialog import (
    DEFAULT_SPECIES, BambiClassificationModelDialog,
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
    assert table.rowCount() == 4
    assert [table.item(r, 0).text() for r in range(4)] == \
        ["Occlusion", "Species", "Sex", "Life stage"]


def test_the_tasks_are_in_the_order_they_run(dock):
    """Occlusion gates the frames; sex and life stage reuse the species
    frames and pick their model from the species call."""
    table = dock.classification_models_table
    names = [table.item(r, 0).text().lower().replace(" ", "_")
             for r in range(table.rowCount())]
    assert names == list(hf_access.TASKS)


def test_both_demographic_tasks_are_configured_per_species(dock):
    """The cue is species-specific for each of them."""
    assert set(hf_access.PER_SPECIES_TASKS) == {"sex", "life_stage"}


def test_life_stage_defaults_to_the_size_estimate(dock):
    """No life-stage head is published and size needs none, so the size
    estimate *is* the default — offering a dead "Default" and starting on
    "Off" only left the per-species dialog's Size-based looking orphaned."""
    combo = dock.classification_models_table.cellWidget(
        dock._task_row("life_stage"), 1)
    assert combo.currentData() == "size"
    assert combo.currentText() == "Default (size-based)"
    assert combo.findData("default") == -1


def test_species_cannot_be_set_to_a_default_that_does_not_exist(dock):
    """No species model is published yet, so offering 'Default' would only
    fail later."""
    row = dock._task_row("species")
    combo = dock.classification_models_table.cellWidget(row, 1)
    index = combo.findData("default")
    assert not combo.model().item(index).isEnabled()
    assert "not released" in combo.itemText(index)
    assert combo.currentData() == "off"


def test_released_tasks_default_to_their_published_model(dock):
    for task in ("occlusion", "sex"):
        combo = dock.classification_models_table.cellWidget(
            dock._task_row(task), 1)
        assert combo.currentData() == "default"


# ---------------------------------------------------------------------------
# The per-step Input combos
# ---------------------------------------------------------------------------

def test_matched_is_the_default_input(dock):
    """Fusion is where the two sensors complement each other most."""
    for task in hf_access.TASKS:
        assert dock._classification_input_combo(task).currentData() == "matched"


def test_every_task_has_its_own_input_combo(dock):
    """The classifiers are separate models, and one may read a camera another
    cannot use — so the choice sits on each step rather than once."""
    combos = [dock._classification_input_combo(t) for t in hf_access.TASKS]
    assert len({id(c) for c in combos}) == len(hf_access.TASKS)


def test_the_input_combos_offer_both_cameras_and_matched(dock):
    combo = dock._classification_input_combo("species")
    assert [combo.itemData(i) for i in range(combo.count())] == \
        ["thermal", "rgb", "matched"]


def test_editing_the_input_updates_the_saved_mapping(dock):
    dock._classification_input_combo("occlusion").setCurrentIndex(1)   # RGB

    spec = dock._classification_spec("occlusion")
    assert spec["modality"] == "rgb"


def test_one_tasks_input_leaves_the_others_alone(dock):
    dock._classification_input_combo("species").setCurrentIndex(0)     # thermal

    assert dock._classification_spec("species")["modality"] == "thermal"
    assert dock._classification_spec("sex")["modality"] == "matched"


def test_the_mapping_survives_a_project_round_trip(dock):
    row = dock._task_row("sex")
    table = dock.classification_models_table
    modality = dock._classification_input_combo("sex")
    modality.setCurrentIndex(0)                      # thermal
    table.item(row, 2).setText("/models/sex.pt")
    table.cellWidget(row, 1).setCurrentIndex(2)      # custom

    dock.target_folder_edit.setText("/tmp/bambi_mapping_roundtrip")
    dock.save_config_to_project()

    modality.setCurrentIndex(2)
    table.item(row, 2).setText("")
    dock.load_config_from_project()

    assert modality.currentData() == "thermal"
    assert table.cellWidget(row, 1).currentData() == "custom"
    assert table.item(row, 2).text() == "/models/sex.pt"


def test_the_config_reports_every_task(dock):
    models = dock.get_config()["classification_models"]
    assert set(models) == set(hf_access.TASKS)


# ---------------------------------------------------------------------------
# Downloading the published models
# ---------------------------------------------------------------------------

def test_the_download_button_exists(dock):
    assert dock.download_models_btn is not None
    assert dock.download_models_status is not None


def test_it_downloads_every_task_set_to_default(dock, monkeypatch, tmp_path):
    """So the class mapping can be set up before anything is run — asking a
    user to run a classifier once before they can configure it is backwards."""
    calls = []
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_get_default_model_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_download_head",
        staticmethod(lambda repo, task, projection, modality, destination,
                     token, log_fn=None: calls.append((task, projection,
                                                       modality))))

    dock.download_classification_models()

    # Occlusion and sex have published models; species does not yet.
    assert sorted(task for task, _p, _m in calls) == ["occlusion", "sex"]
    assert all(projection == "non_geo" for _t, projection, _m in calls)


def test_the_download_follows_the_configured_input(dock, monkeypatch,
                                                   tmp_path):
    calls = []
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_get_default_model_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_download_head",
        staticmethod(lambda repo, task, projection, modality, destination,
                     token, log_fn=None: calls.append((task, modality))))

    dock._classification_input_combo("occlusion").setCurrentIndex(1)
    dock.classification_projection_combo.setCurrentIndex(2)
    dock.download_classification_models()

    assert ("occlusion", "rgb") in calls


def test_a_custom_model_is_not_downloaded(dock, monkeypatch, tmp_path):
    """A file you chose is used from where it is."""
    calls = []
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_get_default_model_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_download_head",
        staticmethod(lambda *a, **k: calls.append(a[1])))

    for task in ("occlusion", "sex"):
        combo = dock.classification_models_table.cellWidget(
            dock._task_row(task), 1)
        combo.setCurrentIndex(combo.findData("custom"))

    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.QMessageBox.information",
        lambda *a, **k: None)
    dock.download_classification_models()

    assert calls == []
    assert "Nothing to download" in dock.download_models_status.text()


def test_already_present_models_are_not_fetched_again(dock, monkeypatch,
                                                      tmp_path):
    calls = []
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_get_default_model_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_download_head",
        staticmethod(lambda *a, **k: calls.append(a[1])))

    for task in ("occlusion", "sex"):
        path = hf_access.head_local_path(
            str(tmp_path), task, "non_geo", "matched")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"weights")

    dock.download_classification_models()

    assert calls == []
    assert "already downloaded" in dock.download_models_status.text()


def test_a_failure_is_reported_and_the_button_recovers(dock, monkeypatch,
                                                       tmp_path):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("gated repo")

    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_get_default_model_dir", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.BambiProcessor."
        "_download_head", staticmethod(_boom))
    warned = {}
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.QMessageBox.warning",
        lambda _p, title, text: warned.update(title=title, text=text))

    dock.download_classification_models()

    assert "gated repo" in warned["text"]
    # One failure must not leave the button dead for the rest of the session.
    assert dock.download_models_btn.isEnabled()


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
    dialog = BambiClassificationModelDialog({}, project_folder)
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    assert "red deer" in names and "wild boar" in names


def test_base_classes_are_not_offered(project_folder):
    """'animal' and 'unknown' are not species anyone has a model for."""
    dialog = BambiClassificationModelDialog({}, project_folder)
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    for base in ("animal", "unknown", "not-an-animal"):
        assert base not in names


def test_only_red_deer_defaults_to_the_published_model(project_folder):
    dialog = BambiClassificationModelDialog({}, project_folder)
    for row in range(dialog.table.rowCount()):
        species = dialog.table.item(row, 0).text()
        chosen = dialog.table.cellWidget(row, 1).currentData()
        assert chosen == ("default" if species == DEFAULT_SPECIES else "off")


def test_species_left_off_are_absent_from_the_result(project_folder):
    """Absent means 'not sexed', which is the honest answer for a species with
    no classifier."""
    dialog = BambiClassificationModelDialog({}, project_folder)
    per_species = dialog.result_spec()["species"]
    assert set(per_species) == {DEFAULT_SPECIES}


def test_a_custom_model_per_species_round_trips(project_folder):
    dialog = BambiClassificationModelDialog({}, project_folder)
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
    dialog = BambiClassificationModelDialog(spec, project_folder)
    row = next(r for r in range(dialog.table.rowCount())
               if dialog.table.item(r, 0).text() == "wild boar")
    assert dialog.table.cellWidget(row, 1).currentData() == "custom"
    assert dialog.table.item(row, 2).text() == "/x.pt"


def test_life_stage_species_offer_size_as_the_default(project_folder):
    """The same wording as the main table, so the two views agree."""
    dialog = BambiClassificationModelDialog(
        {}, project_folder, task="life_stage")
    combo = dialog.table.cellWidget(0, 1)
    assert [combo.itemData(i) for i in range(combo.count())] == \
        ["off", "size", "custom"]
    assert combo.itemText(1) == "Default (size-based)"


def test_every_species_starts_on_the_size_estimate(project_folder):
    """It needs no model, so there is nothing to configure before it works."""
    dialog = BambiClassificationModelDialog(
        {}, project_folder, task="life_stage")
    for row in range(dialog.table.rowCount()):
        assert dialog.table.cellWidget(row, 1).currentData() == "size"


def test_a_project_without_a_store_still_offers_red_deer(tmp_path):
    dialog = BambiClassificationModelDialog({}, str(tmp_path))
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    assert names == [DEFAULT_SPECIES]


def test_dialogs_are_tables_not_free_text(project_folder):
    """Typed species names were what made class ids unstable before 6.0."""
    assert isinstance(BambiClassificationModelDialog({}, project_folder).table,
                      QTableWidget)
