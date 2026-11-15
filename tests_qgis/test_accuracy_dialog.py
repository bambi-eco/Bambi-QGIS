# -*- coding: utf-8 -*-
"""The accuracy report window and its button on the Processing tab
(bambi_accuracy_dialog.py, dock/accuracy.py).

The comparison itself is covered without QGIS in tests/test_accuracy.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import csv
import json
import os

import pytest

from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.bambi_accuracy_dialog import AccuracyDialog
from tests.survey_flights import make_flight, make_labelled_flight


@pytest.fixture
def flight(tmp_path):
    return make_labelled_flight(str(tmp_path / "Meadow"))


@pytest.fixture
def dialog(flight):
    made = AccuracyDialog(flight, "Meadow", "t")
    yield made
    made.deleteLater()


def test_the_window_compares_and_shows_the_page(dialog, flight):
    assert dialog.compare() is True

    page = dialog.page.toPlainText()
    assert "Were the animals found?" in page
    assert "Was each animal one track?" in page
    assert "Was it named right?" in page
    assert "yolo_wildlife.pt" in page
    assert dialog.document["detection"]["found"] == 10
    assert dialog.windowTitle() == "Accuracy against hand labels - Meadow"


def test_every_comparison_is_written_into_the_flights_folder(dialog, flight):
    dialog.compare()

    folder = os.path.join(flight, "analytics_t")
    with open(os.path.join(folder, "accuracy.json")) as handle:
        stored = json.load(handle)
    assert stored["tracking"]["count_error"] == 2
    with open(os.path.join(folder, "accuracy_boxes.csv"), newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["kind"] for row in rows].count("missed") == 2
    with open(os.path.join(folder, "accuracy.html"),
              encoding="utf-8") as handle:
        page = handle.read()
    assert "<h2>Accuracy against hand labels - Meadow (Thermal)</h2>" in page
    assert "accuracy_boxes.csv" in dialog.saved_label.text()
    assert folder in dialog.saved_label.text()


def test_the_frames_that_were_used_are_shown(dialog):
    dialog.compare()

    assert dialog.auto_frames_check.isChecked()
    assert not dialog.first_spin.isEnabled()
    assert (dialog.first_spin.value(), dialog.last_spin.value()) == (0, 5)


def test_frames_can_be_named_and_the_overlap_set(dialog):
    dialog.auto_frames_check.setChecked(False)
    assert dialog.first_spin.isEnabled() and dialog.last_spin.isEnabled()
    dialog.first_spin.setValue(0)
    dialog.last_spin.setValue(2)
    dialog.iou_spin.setValue(0.9)

    dialog.compare()

    assert dialog.document["frames"] == {
        "first": 0, "last": 2, "given": True, "with_labels": 3,
        "with_boxes": 3}
    assert dialog.document["iou_threshold"] == pytest.approx(0.9)
    assert "0 to 2 (as given)" in dialog.page.toPlainText()


def test_the_other_camera_has_its_own_folder(dialog, flight):
    dialog.camera_combo.setCurrentIndex(1)

    assert dialog.modality() == "w"
    assert dialog.output_folder() == os.path.join(flight, "analytics_w")
    assert dialog.compare() is False              # nothing detected on RGB
    assert "Nothing to compare" in dialog.page.toPlainText()
    assert "Detect Animals" in dialog.page.toPlainText()
    assert dialog.document is None and dialog.saved_label.text() == ""


def test_a_flight_without_labels_says_so(tmp_path):
    folder = make_flight(str(tmp_path / "unlabelled"), [
        {"id": 1, "species": "roe deer", "at": (0, 0), "frames": (1, 5)}])
    dialog = AccuracyDialog(folder, "Unlabelled", "t")

    assert dialog.compare() is False
    assert "no hand labels" in dialog.page.toPlainText()
    assert not os.path.exists(os.path.join(folder, "analytics_t",
                                           "accuracy.json"))


def test_the_result_is_logged(flight):
    lines = []
    dialog = AccuracyDialog(flight, "Meadow", "t", log_fn=lines.append)

    dialog.compare()

    assert lines[0] == "Accuracy against hand labels (Meadow, Thermal):"
    assert lines[1].startswith("  Compared with 2 labelled animal(s)")
    assert any("precision 83 %, recall 83 %" in line for line in lines)


# ---------------------------------------------------------------------------
# The Processing tab
# ---------------------------------------------------------------------------

def test_the_processing_tab_has_the_button(dock):
    assert dock.accuracy_btn.text() == "→ Compare with Hand Labels…"


def test_the_button_opens_the_report_of_the_active_flight(dock, flight):
    dock.target_folder_edit.setText(flight)
    dock._on_target_folder_changed()

    dock.accuracy_btn.click()

    dialog = dock._accuracy_dialog
    assert dialog.isVisible()
    assert dialog.modality() == "t"
    assert dialog.document["detection"]["found"] == 10
    assert "Accuracy against hand labels" in dock.log_text.toPlainText()
    dialog.close()


def test_the_report_follows_the_tracking_camera(dock, flight):
    dock.target_folder_edit.setText(flight)
    dock._on_target_folder_changed()
    dock.tracking_camera_combo.setCurrentIndex(1)

    dialog = dock.open_accuracy_report()

    assert dialog.modality() == "w"
    dialog.close()


def test_without_a_flight_there_is_nothing_to_report_on(dock, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[1:]))
    dock.target_folder_edit.setText("")

    assert dock.open_accuracy_report() is None
    assert warned and "Select a flight first" in warned[0][1]
