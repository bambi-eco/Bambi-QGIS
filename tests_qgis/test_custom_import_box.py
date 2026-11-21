# -*- coding: utf-8 -*-
"""Configuration > Tracking > Custom import: the box, its source and the
button that runs the import.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import pytest

from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.core import custom_models


@pytest.fixture
def warned(monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda _parent, _title, text, *a, **k: shown.append(text))
    return shown


@pytest.fixture
def started(dock, monkeypatch):
    steps = []
    monkeypatch.setattr(dock, "start_worker", steps.append)
    return steps


def importer(dock, tmp_path):
    path = tmp_path / "mine.py"
    path.write_text("class I:\n    def setup(self, c):\n        pass\n"
                    "    def run(self, source, frames):\n        return []\n")
    dock.custom_import_box.set_spec({"path": str(path), "class": "I"})
    dock._on_custom_implementation_changed()
    return str(path)


def test_the_tracking_tab_has_the_box(dock):
    box = dock.custom_import_box

    assert box.kind == custom_models.IMPORTER
    assert dock._custom_boxes["importer"] is box
    # nothing to switch on: it runs when its button is pressed
    assert not box.enable_check.isVisible()
    assert dock.custom_import_btn.text() == "Import now"


def test_the_source_reaches_the_step_and_is_saved(dock, tmp_path):
    source = tmp_path / "boxes.csv"
    source.write_text("frame,x1,y1,x2,y2\n")
    dock.custom_import_source_edit.setText(f"  {source}  ")

    assert dock.get_config()["custom_import_source"] == str(source)

    from bambi_wildlife_detection.core import config_schema
    assert config_schema.WIDGET_BINDINGS["Import/Source"][0] == \
        "custom_import_source_edit"


def test_without_an_importer_it_asks_for_one(dock, warned, started):
    dock.run_custom_import()

    assert started == []
    assert "importer" in warned[-1]


def test_a_source_that_is_not_there_is_named(dock, tmp_path, warned, started):
    importer(dock, tmp_path)
    dock.custom_import_source_edit.setText(str(tmp_path / "nowhere.csv"))

    dock.run_custom_import()

    assert started == []
    assert "nowhere.csv" in warned[-1]


def test_it_needs_the_frames(dock, tmp_path, warned, started):
    importer(dock, tmp_path)
    dock.target_folder_edit.setText(str(tmp_path))

    dock.run_custom_import()

    assert started == []
    assert "frame extraction" in warned[-1]


def test_the_step_asks_before_the_file_runs(dock, tmp_path):
    path = importer(dock, tmp_path)

    specs = dock._custom_specs_for_step("custom_import", dock.get_config())

    assert [spec["path"] for spec in specs] == [path]
    assert custom_models.is_custom(specs[0])
