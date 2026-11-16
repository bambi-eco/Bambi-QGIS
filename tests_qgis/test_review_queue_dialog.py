# -*- coding: utf-8 -*-
"""The review queue window and its button on the Survey Analytics tab
(bambi_review_queue_dialog.py, dock/review_queue.py).

Which tracks are in doubt is covered without QGIS in
tests/test_review_queue.py, a reviewer's word in tests/test_track_review.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.bambi_review_queue_dialog import (
    NOT_KNOWN, ReviewQueueDialog)
from bambi_wildlife_detection.core import review_queue

VOCABULARY = {
    "species": [(0, "animal"), (-2, "not-an-animal"), (3, "roe deer"),
                (4, "red deer")],
    "sex": [(1, "female"), (2, "male")],
    "age": [(1, "adult"), (2, "juvenile")],
}


def _row(track_id, **more):
    return dict({
        "flight": None, "track_id": track_id, "camera": "thermal",
        "approved": False, "label_track_ids": "", "by_hand": "",
        "species": "roe deer", "species_vote_fraction": 0.95,
        "detector_species": "", "sex": "", "sex_vote_fraction": None,
        "age": "", "mean_confidence": 0.85, "n_boxes": 30, "n_occluded": 0,
        "start_frame": 10 * track_id}, **more)


class Review:
    """Stands for the project: what is in doubt, and what was said."""

    def __init__(self, rows):
        self.all = rows
        self.said = []
        self.opened = []
        self.loaded = []
        self.refuse = False

    def load(self, settings):
        self.loaded.append(dict(settings))
        return review_queue.build(self.all, settings=settings), len(self.all)

    def open_track(self, row):
        self.opened.append(row["track_id"])

    def approve(self, row):
        self.said.append(("approve", row["track_id"]))
        return not self.refuse

    def set_species(self, row, species_id):
        self.said.append(("species", row["track_id"], species_id))
        return not self.refuse

    def set_attribute(self, row, field, value):
        self.said.append((field, row["track_id"], value))
        return not self.refuse

    def dialog(self, **more):
        return ReviewQueueDialog(
            self.load, lambda flight: VOCABULARY, self.open_track,
            self.approve, self.set_species, self.set_attribute,
            camera_label="Thermal", **more)


@pytest.fixture
def review():
    return Review([
        _row(1),                                             # plain
        _row(2, mean_confidence=0.2),                        # 3
        _row(3, mean_confidence=0.3, n_boxes=2),             # 6
        _row(4, species="animal", species_vote_fraction=None),   # 1.5
    ])


def _tracks(dialog):
    return [row["track_id"] for row in dialog.rows()]


def _cell(dialog, row, column):
    index = [dialog.table.horizontalHeaderItem(c).text()
             for c in range(dialog.table.columnCount())].index(column)
    return dialog.table.item(row, index).text()


# ---------------------------------------------------------------------------
# The list
# ---------------------------------------------------------------------------

def test_the_tracks_in_doubt_are_listed_the_most_doubtful_first(review):
    dialog = review.dialog()

    assert _tracks(dialog) == [3, 2, 4]
    assert dialog.table.rowCount() == 3
    assert _cell(dialog, 0, "Track") == "3"
    assert _cell(dialog, 0, "Why") == (
        "boxes of low confidence (mean 0.30); only 2 boxes")
    assert _cell(dialog, 0, "Confidence") == "0.30"
    assert dialog.summary_label.text().startswith(
        "3 of 4 track(s) are worth a look")
    assert dialog.progress_label.text() == "3 left"
    assert dialog.windowTitle() == "Review Queue (Thermal)"


def test_the_first_track_is_selected_and_shown_in_the_inspector(review):
    dialog = review.dialog()

    assert dialog.current()["track_id"] == 3
    assert review.opened == [3]


def test_the_inspector_can_be_left_alone(review):
    dialog = review.dialog()
    dialog.inspector_check.setChecked(False)

    dialog.select_next()

    assert dialog.current()["track_id"] == 2
    assert review.opened == [3]


def _track_item(dialog, row):
    index = [dialog.table.horizontalHeaderItem(c).text()
             for c in range(dialog.table.columnCount())].index("Track")
    return index, dialog.table.item(row, index)


def test_a_tracks_number_looks_like_the_link_it_is(review):
    """As the frames of the track inventory: underlined and blue."""
    dialog = review.dialog()

    _index, item = _track_item(dialog, 1)

    assert item.font().underline()
    assert item.foreground().color().name() == "#1e64c8"
    assert item.toolTip() == "Open track 2 in the inspector"
    # ... and nothing else in the row does
    why = dialog.table.item(1, _index + 4)
    assert not why.font().underline()


def test_clicking_the_number_opens_the_track(review):
    dialog = review.dialog()
    dialog.inspector_check.setChecked(False)     # selecting shows nothing
    index, _item = _track_item(dialog, 2)

    dialog.table.cellClicked.emit(2, index)

    assert review.opened == [3, 4]
    # another cell of the row is no link
    dialog.table.cellClicked.emit(1, index + 1)
    assert review.opened == [3, 4]


def test_the_click_that_selects_a_track_opens_it_once(review):
    dialog = review.dialog()                     # shows every selected track
    index, _item = _track_item(dialog, 1)

    dialog.table.selectRow(1)                    # the press: track 2 opens
    dialog.table.cellClicked.emit(1, index)      # the release, on its number

    assert review.opened == [3, 2]
    # clicking the number of the track that is selected opens it again
    dialog.table.cellClicked.emit(1, index)
    assert review.opened == [3, 2, 2]


def test_a_survey_of_several_flights_says_which_flight(review):
    for row in review.all:
        row["flight"] = "Meadow"
    dialog = review.dialog(several=True)

    assert dialog.table.horizontalHeaderItem(0).text() == "Flight"
    assert _cell(dialog, 0, "Flight") == "Meadow"


def test_the_selectors_offer_what_the_flight_knows_on_what_the_track_is(
        review):
    dialog = review.dialog()
    dialog.table.selectRow(1)                       # track 2, a roe deer

    assert [dialog.species_combo.itemText(i)
            for i in range(dialog.species_combo.count())] == [
        "animal", "not-an-animal", "roe deer", "red deer"]
    assert dialog.species_combo.currentText() == "roe deer"
    assert [dialog.sex_combo.itemText(i)
            for i in range(dialog.sex_combo.count())] == [
        NOT_KNOWN, "female", "male"]
    assert dialog.sex_combo.currentText() == NOT_KNOWN


# ---------------------------------------------------------------------------
# The reviewer's word
# ---------------------------------------------------------------------------

def test_approving_takes_the_track_out_and_goes_on(review):
    dialog = review.dialog()

    dialog.approve_btn.click()

    assert review.said == [("approve", 3)]
    assert _tracks(dialog) == [2, 4]
    assert dialog.current()["track_id"] == 2
    assert review.opened == [3, 2]
    assert dialog.progress_label.text() == "2 left, 1 done"


def test_ruling_out_makes_it_not_an_animal(review):
    dialog = review.dialog()

    dialog.reject_btn.click()

    assert review.said == [("species", 3, -2)]
    assert _tracks(dialog) == [2, 4]


def test_a_species_is_corrected_and_the_track_stays_to_be_approved(review):
    dialog = review.dialog()
    dialog.species_combo.setCurrentIndex(
        dialog.species_combo.findText("red deer"))
    dialog.species_combo.activated.emit(dialog.species_combo.currentIndex())

    assert review.said == [("species", 3, 4)]
    assert _tracks(dialog) == [3, 2, 4]
    assert _cell(dialog, 0, "Species") == "red deer"
    assert dialog.current()["species"] == "red deer"


def test_choosing_not_an_animal_in_the_list_rules_it_out_too(review):
    dialog = review.dialog()
    index = dialog.species_combo.findText("not-an-animal")
    dialog.species_combo.setCurrentIndex(index)
    dialog.species_combo.activated.emit(index)

    assert review.said == [("species", 3, -2)]
    assert _tracks(dialog) == [2, 4]


def test_choosing_what_it_already_is_says_nothing(review):
    dialog = review.dialog()

    dialog.species_combo.activated.emit(dialog.species_combo.currentIndex())

    assert review.said == []


def test_sex_and_age_are_set_also_to_not_known(review):
    review.all[2]["sex"] = "male"
    dialog = review.dialog()
    dialog.sex_combo.setCurrentIndex(dialog.sex_combo.findText("female"))
    dialog.sex_combo.activated.emit(dialog.sex_combo.currentIndex())
    dialog.age_combo.setCurrentIndex(dialog.age_combo.findText("juvenile"))
    dialog.age_combo.activated.emit(dialog.age_combo.currentIndex())
    dialog.sex_combo.setCurrentIndex(0)
    dialog.sex_combo.activated.emit(0)

    assert review.said == [("sex", 3, 1), ("age", 3, 2), ("sex", 3, None)]
    assert _cell(dialog, 0, "Sex") == "" and _cell(dialog, 0, "Age") == \
        "juvenile"


def test_skipping_leaves_the_track_and_wraps_around(review):
    dialog = review.dialog()

    dialog.skip_btn.click()
    dialog.skip_btn.click()
    assert dialog.current()["track_id"] == 4
    dialog.skip_btn.click()

    assert dialog.current()["track_id"] == 3
    assert _tracks(dialog) == [3, 2, 4] and review.said == []


def test_what_could_not_be_written_stays_in_the_queue(review):
    review.refuse = True
    dialog = review.dialog()

    dialog.approve_btn.click()
    dialog.reject_btn.click()
    dialog.species_combo.setCurrentIndex(
        dialog.species_combo.findText("red deer"))
    dialog.species_combo.activated.emit(dialog.species_combo.currentIndex())

    assert _tracks(dialog) == [3, 2, 4]
    assert _cell(dialog, 0, "Species") == "roe deer"
    assert dialog.species_combo.currentText() == "roe deer"


def test_the_last_track_leaves_an_empty_queue(review):
    dialog = review.dialog()

    for _ in range(3):
        dialog.approve_btn.click()

    assert dialog.rows() == [] and dialog.current() is None
    assert dialog.progress_label.text() == "0 left, 3 done"
    assert not dialog.approve_btn.isEnabled()
    assert not dialog.species_combo.isEnabled()
    dialog.approve_btn.click()                      # nothing to say it of
    dialog.select_next()
    assert len(review.said) == 3


def test_a_track_dealt_with_in_the_inspector_leaves_the_queue(review):
    dialog = review.dialog()

    assert dialog.remove_track(2) is True           # not the selected one
    assert _tracks(dialog) == [3, 4] and dialog.current()["track_id"] == 3
    assert dialog.remove_track(3) is True           # the selected one
    assert dialog.current()["track_id"] == 4
    assert dialog.remove_track(99) is False
    assert dialog.remove_track(4, flight="Forest") is False
    assert dialog.progress_label.text() == "1 left, 2 done"


# ---------------------------------------------------------------------------
# What is in doubt can be set
# ---------------------------------------------------------------------------

def test_the_limits_decide_what_is_in_doubt(review):
    dialog = review.dialog()
    dialog.confidence_spin.setValue(0.25)
    dialog.boxes_spin.setValue(1)
    dialog.vote_spin.setValue(50)

    dialog.refresh_btn.click()

    assert review.loaded[-1] == {"min_confidence": 0.25, "min_boxes": 1,
                                 "min_vote": 0.5}
    assert _tracks(dialog) == [2, 4]
    assert dialog.progress_label.text() == "2 left"


def test_nothing_in_doubt(review):
    review.all = [_row(1), _row(2)]
    dialog = review.dialog()

    assert dialog.table.rowCount() == 0
    assert dialog.summary_label.text() == (
        "None of the 2 track(s) is in doubt - or they are all approved.")
    assert not dialog.approve_btn.isEnabled()
    assert review.opened == []


def test_the_keys(review):
    from qgis.PyQt.QtWidgets import QShortcut

    dialog = review.dialog()
    keys = {shortcut.key().toString(): shortcut
            for shortcut in dialog.findChildren(QShortcut)}

    assert sorted(keys) == ["A", "N", "X"]
    keys["N"].activated.emit()
    assert dialog.current()["track_id"] == 2
    keys["A"].activated.emit()
    keys["X"].activated.emit()
    assert review.said == [("approve", 2), ("species", 4, -2)]


# ---------------------------------------------------------------------------
# The Survey Analytics tab
# ---------------------------------------------------------------------------

@pytest.fixture
def survey(dock, tmp_path, monkeypatch):
    """A flight with a plain track and two short ones, as the survey."""
    from bambi_wildlife_detection import bambi_click_tool
    from tests.survey_flights import make_flight

    folder = make_flight(str(tmp_path / "Meadow"), [
        {"id": 1, "species": "roe deer", "at": (0, 0), "frames": (1, 5)},
        {"id": 2, "species": "roe deer", "at": (60, 0), "frames": (2, 6)},
    ])
    dock.target_folder_edit.setText(folder)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    dock.survey_panel.folder_edit.setText(str(tmp_path / "survey"))
    opened = []
    monkeypatch.setattr(
        bambi_click_tool, "open_track_in_viewer",
        lambda window, target, modality, track, **more: opened.append(
            (track, more.get("frame"))) or True)

    def unexpected(_parent, title, text, *args, **kwargs):
        raise AssertionError(f"unexpected message: {title} - {text}")

    monkeypatch.setattr(QMessageBox, "warning", unexpected)
    monkeypatch.setattr(QMessageBox, "critical", unexpected)
    monkeypatch.setattr(QMessageBox, "information", unexpected)
    return folder, opened


def test_the_inventory_group_has_the_button(dock):
    assert dock.review_queue_btn.text() == "→ Review Queue…"


def test_the_button_opens_the_queue_of_the_survey(dock, survey):
    _folder, opened = survey

    dock.review_queue_btn.click()

    dialog = dock._review_queue_dialog
    assert dialog.isVisible()
    # both tracks have two boxes: very short
    assert _tracks(dialog) == [1, 2]
    assert _cell(dialog, 0, "Why") == "only 2 boxes"
    assert opened == [(1, 1)]                # track 1, at its first frame
    assert [dialog.species_combo.itemText(i)
            for i in range(dialog.species_combo.count())][:3] == [
        "animal", "unknown", "not-an-animal"]
    dialog.close()


def test_approving_ticks_the_track_in_the_project(dock, survey):
    from bambi_wildlife_detection.core import track_inventory

    folder, _opened = survey
    dialog = dock.open_review_queue()

    dialog.approve_btn.click()

    rows = {row["track_id"]: row
            for row in track_inventory.build_inventory(folder, "t")}
    assert rows[1]["approved"] is True and rows[2]["approved"] is False
    assert "Track 1 (Meadow, Thermal): approved" in dock.log_text.toPlainText()
    dialog.refresh_btn.click()
    assert _tracks(dialog) == [2]
    dialog.close()


def test_a_correction_is_written_and_stands(dock, survey):
    from bambi_wildlife_detection.core import track_inventory

    folder, _opened = survey
    dialog = dock.open_review_queue()
    index = dialog.species_combo.findText("red deer")
    dialog.species_combo.setCurrentIndex(index)
    dialog.species_combo.activated.emit(index)
    index = dialog.sex_combo.findText("female")
    dialog.sex_combo.setCurrentIndex(index)
    dialog.sex_combo.activated.emit(index)

    row = track_inventory.build_inventory(folder, "t", track_ids=[1])[0]
    assert (row["species"], row["sex"], row["by_hand"]) == (
        "red deer", "female", "species; sex")
    log = dock.log_text.toPlainText()
    assert "Track 1 (Meadow, Thermal): species set to red deer" in log
    assert "Track 1 (Meadow, Thermal): sex set to female" in log
    dialog.close()


def test_ruling_out_takes_the_animal_out_of_the_counts(dock, survey):
    from bambi_wildlife_detection.core import analytics_source

    folder, _opened = survey
    dialog = dock.open_review_queue()

    dialog.reject_btn.click()

    animals, facts = analytics_source.load_animals(folder, "t")
    assert [animal["track_id"] for animal in animals] == [2]
    assert facts["excluded_not_an_animal"] == 1
    assert "species set to not-an-animal" in dock.log_text.toPlainText()
    dialog.close()


def test_what_the_inspector_approves_or_deletes_leaves_the_queue(dock, survey):
    folder, _opened = survey
    dialog = dock.open_review_queue()

    dock._on_viewer_track_approved(folder, "t", 1, True)
    assert _tracks(dialog) == [2]
    dock._on_viewer_track_approved(folder, "w", 2, True)     # other camera
    dock._on_viewer_track_approved(folder, "t", 2, False)    # un-approved
    assert _tracks(dialog) == [2]
    dock._on_viewer_track_deleted(folder, "t", 2)
    assert _tracks(dialog) == []
    dialog.close()


def test_without_a_survey_no_queue(dock, monkeypatch):
    monkeypatch.setattr(dock, "_checked_survey", lambda: None)

    assert dock.open_review_queue() is None
