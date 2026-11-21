# -*- coding: utf-8 -*-
"""Processing > Export > Custom exporter: the box, the format that uses it
and what both export panels do with it.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

from bambi_wildlife_detection import bambi_custom_impl_widget as widget
from bambi_wildlife_detection.core import custom_models, exporters


@pytest.fixture
def said(monkeypatch):
    shown = {"warning": [], "information": [], "critical": []}
    for kind, texts in shown.items():
        monkeypatch.setattr(
            QMessageBox, kind,
            lambda _parent, _title, text, *a, _t=texts, **k: _t.append(text))
    return shown


def exporter(dock, tmp_path):
    path = tmp_path / "mine.py"
    path.write_text(
        "import os\n\n\nclass E:\n    def setup(self, c):\n        pass\n"
        "    def run(self, data, output):\n"
        "        open(os.path.join(output, 'done.txt'), 'w').write(\n"
        "            str(len(data['detections'])))\n")
    dock.custom_export_box.set_spec({"path": str(path), "class": "E"})
    dock._on_custom_implementation_changed()
    return str(path)


def choose(combo, key):
    combo.setCurrentIndex(combo.findData(key))


def test_the_box_belongs_to_the_custom_format(dock):
    box = dock.custom_export_box

    assert box.kind == custom_models.EXPORTER
    assert dock._custom_boxes["exporter"] is box
    assert dock.export_format_combo.findData(exporters.CUSTOM) >= 0
    assert dock.dataset_format_combo.findData(exporters.CUSTOM) >= 0

    choose(dock.export_format_combo, "coco")
    assert box.isHidden()
    choose(dock.export_format_combo, exporters.CUSTOM)
    assert not box.isHidden()
    assert dock.export_images_check.isEnabled()


def test_the_dataset_panel_says_where_the_exporter_is_chosen(dock):
    choose(dock.dataset_format_combo, "yolo")
    assert dock.dataset_custom_label.isHidden()

    choose(dock.dataset_format_combo, exporters.CUSTOM)
    assert not dock.dataset_custom_label.isHidden()
    assert "Processing tab" in dock.dataset_custom_label.text()


def test_without_a_file_nothing_is_exported(dock, tmp_path, said):
    dock.target_folder_edit.setText(str(tmp_path))
    choose(dock.export_format_combo, exporters.CUSTOM)

    dock.run_export()

    assert "Choose the .py file with your exporter" in said["warning"][-1]


def test_one_flight_is_written_by_the_users_class(
        dock, tmp_path, said, monkeypatch):
    from tests.survey_flights import make_flight

    folder = make_flight(str(tmp_path / "Meadow"), [
        {"id": 1, "species": "roe deer", "at": (10, 10), "frames": (0, 2)}])
    output = tmp_path / "out"
    output.mkdir()
    dock.target_folder_edit.setText(folder)
    exporter(dock, tmp_path)
    choose(dock.export_format_combo, exporters.CUSTOM)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(output))
    asked = []
    monkeypatch.setattr(widget, "confirm_specs",
                        lambda _parent, specs: asked.append(specs) or True)

    dock.run_export()

    assert said["warning"] == [] and said["critical"] == []
    assert (output / "done.txt").read_text() == "2"
    # the user's own code: asked before it ran
    assert len(asked) == 1


def test_a_declined_exporter_does_not_run(dock, tmp_path, said, monkeypatch):
    dock.target_folder_edit.setText(str(tmp_path))
    exporter(dock, tmp_path)
    choose(dock.export_format_combo, exporters.CUSTOM)
    monkeypatch.setattr(widget, "confirm_specs", lambda _parent, specs: False)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **k: pytest.fail("nothing to choose a folder for"))

    dock.run_export()

    assert not os.path.exists(tmp_path / "done.txt")


def test_the_dataset_step_asks_before_the_file_runs(dock, tmp_path):
    path = exporter(dock, tmp_path)

    choose(dock.dataset_format_combo, exporters.CUSTOM)
    specs = dock._custom_specs_for_step("dataset_export", dock.get_config())
    assert [spec["path"] for spec in specs] == [path]

    choose(dock.dataset_format_combo, "coco")
    assert dock._custom_specs_for_step(
        "dataset_export", dock.get_config()) == []
