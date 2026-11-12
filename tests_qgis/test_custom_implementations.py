# -*- coding: utf-8 -*-
"""Custom implementations in the dock: the boxes, the saved mapping, the
trust prompt and the routing of the steps (9.0.0)."""
import os
import textwrap

import pytest
from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection import bambi_custom_impl_widget as widget
from bambi_wildlife_detection.core import custom_models

DETECTOR = """
    class Det:
        def setup(self, context):
            return {"class_names": ["animal"]}
        def run(self, image, frame):
            return []
"""


@pytest.fixture
def impl(tmp_path):
    path = tmp_path / "det.py"
    path.write_text(textwrap.dedent(DETECTOR), encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def no_trust(monkeypatch):
    """Each test starts with nothing approved on 'this computer'."""
    store = {}
    monkeypatch.setattr(widget, "_trusted", lambda: set(store))
    monkeypatch.setattr(widget, "remember_trust",
                        lambda path: store.__setitem__(
                            custom_models.file_hash(path), True))
    return store


def test_the_boxes_exist_and_start_off(dock):
    assert set(dock._custom_boxes) == {"detector_t", "detector_w", "tracker"}
    config = dock.get_config()["custom_implementations"]
    assert not any(custom_models.is_custom(spec) for spec in config.values())
    assert dock.custom_samples_btn is not None


def test_a_box_is_mirrored_into_the_configuration(dock, impl):
    box = dock._custom_boxes["detector_t"]
    box.enable_check.setChecked(True)
    box.path_edit.setText(impl)
    box.options_edit.setText('{"a": 1}')
    box.changed.emit()

    spec = dock.get_config()["custom_implementations"]["detector_t"]
    assert spec == {"source": "implementation", "path": impl, "class": "",
                    "options": '{"a": 1}'}
    assert not custom_models.is_custom(
        dock.get_config()["custom_implementations"]["detector_w"])


def test_the_mapping_survives_a_round_trip(dock, impl):
    dock._custom_implementations = {"tracker": {
        "source": "implementation", "path": impl, "class": "Det",
        "options": ""}}
    dock._apply_custom_implementations()

    box = dock._custom_boxes["tracker"]
    assert box.enable_check.isChecked()
    assert box.path_edit.text() == impl
    assert box.class_edit.text() == "Det"
    assert not dock._custom_boxes["detector_t"].enable_check.isChecked()


def test_only_the_steps_own_implementations_are_asked_about(dock, impl):
    config = {"detection_camera": "W",
              "custom_implementations": {
                  "detector_t": {"source": "implementation", "path": "t.py"},
                  "detector_w": {"source": "implementation", "path": impl},
                  "tracker": {"source": "implementation", "path": "trk.py"}},
              "classification_models": {
                  "sex": {"model": "default", "species": {
                      "red deer": {"model": "implementation",
                                   "path": "sex.py"}}}}}
    paths = lambda step: [s.get("path") for s in
                          dock._custom_specs_for_step(step, config) if s]
    assert paths("detection") == [impl]
    assert paths("tracking") == ["trk.py"]
    assert "sex.py" in paths("classify_sex")
    assert paths("alfs") == []


def test_unapproved_code_is_asked_about_and_not_run_on_no(dock, impl,
                                                          monkeypatch):
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: asked.append(1) or QMessageBox.StandardButton.No)
    spec = {"source": "implementation", "path": impl}
    assert widget.confirm_specs(dock, [spec, None, {"source": "default"}]) \
        is False
    assert asked == [1]


def test_approved_code_is_not_asked_about_again_until_it_changes(
        dock, impl, monkeypatch):
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: asked.append(1) or QMessageBox.StandardButton.Yes)
    spec = {"source": "implementation", "path": impl}
    assert widget.confirm_specs(dock, [spec]) is True
    assert widget.confirm_specs(dock, [spec]) is True
    assert asked == [1]

    with open(impl, "a", encoding="utf-8") as fh:
        fh.write("\n# edited\n")
    assert widget.confirm_specs(dock, [spec]) is True
    assert asked == [1, 1]


def test_check_describes_the_implementation(dock, impl, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes)
    box = dock._custom_boxes["detector_t"]
    box.enable_check.setChecked(True)
    box.path_edit.setText(impl)
    box.check(blocking=True)
    log = box.log_text()
    assert "Required packages" in log            # listed step by step
    assert "Running Det.setup()" in log
    assert "OK - Det in det.py" in log
    assert "0 detection(s)" in log
    assert box.log_view.isReadOnly()             # a log: select and copy

    box.path_edit.setText(os.path.join(os.path.dirname(impl), "nope.py"))
    box.check(blocking=True)
    assert "FAILED" in box.log_text() and "not found" in box.log_text()


def test_check_runs_in_the_background(dock, impl, monkeypatch, qgis_app=None):
    """A setup() that loads a model must not freeze QGIS."""
    from qgis.PyQt.QtCore import QCoreApplication
    import time

    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes)
    box = dock._custom_boxes["detector_t"]
    box.enable_check.setChecked(True)
    box.path_edit.setText(impl)
    box.check()
    assert not box.check_btn.isEnabled()         # running, not blocking
    deadline = time.time() + 20
    while box._worker is not None and time.time() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)
    assert box.check_btn.isEnabled()
    assert "OK - Det in det.py" in box.log_text()


def test_a_missing_package_is_named_before_anything_runs(dock, tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes)
    path = tmp_path / "needs.py"
    path.write_text(
        'REQUIRES = ["a_package_nobody_has"]\n'
        'raise SystemExit("must not be executed")\n', encoding="utf-8")
    box = dock._custom_boxes["detector_t"]
    box.enable_check.setChecked(True)
    box.path_edit.setText(str(path))
    box.check(blocking=True)
    log = box.log_text()
    assert "a_package_nobody_has - MISSING" in log
    assert "pip install" in log


def test_the_classification_table_offers_a_custom_implementation(dock):
    combo = dock.classification_models_table.cellWidget(
        dock._task_row("species"), 1)
    assert combo.findData("implementation") >= 0


def test_a_custom_classifier_is_not_held_back_for_embeddings(
        dock, impl, tmp_path, monkeypatch):
    """Only its setup() knows whether it reads embeddings."""
    started = []
    monkeypatch.setattr(type(dock), "start_worker",
                        lambda self, step: started.append(step))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    dock.target_folder_edit.setText(str(tmp_path))
    dock._classification_models = {"species": {
        "modality": "thermal", "model": "implementation", "path": impl}}

    dock.run_classification("species")
    assert started == ["classify_species"]


def test_the_custom_tracker_has_an_interpolation_switch(dock, impl):
    assert dock.custom_tracker_interpolate_check.isChecked()
    dock._on_custom_implementation_changed()
    assert dock.get_config()["custom_implementations"]["tracker"][
        "interpolate"] is True

    dock.custom_tracker_interpolate_check.setChecked(False)
    assert dock.get_config()["custom_implementations"]["tracker"][
        "interpolate"] is False

    dock._custom_implementations = {"tracker": {
        "source": "implementation", "path": impl, "interpolate": True}}
    dock._apply_custom_implementations()
    assert dock.custom_tracker_interpolate_check.isChecked()


def test_detection_has_a_projection_setting(dock):
    assert dock.get_config()["detection_projection"] == "perspective"
    dock.detection_projection_combo.setCurrentIndex(1)
    assert dock.get_config()["detection_projection"] == "geotiff"


def test_geotiff_detection_needs_exported_geotiffs(dock, tmp_path,
                                                   monkeypatch):
    import json
    started, warned = [], []
    monkeypatch.setattr(type(dock), "start_worker",
                        lambda self, step: started.append(step))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    (tmp_path / "frames_t").mkdir()
    (tmp_path / "poses_t.json").write_text(
        json.dumps({"images": []}), encoding="utf-8")
    dock.target_folder_edit.setText(str(tmp_path))
    dock.detection_projection_combo.setCurrentIndex(1)

    dock.run_detection()
    assert started == []
    assert any("GeoTIFF" in text for text in warned)


def test_alfs_is_a_projection_of_detection_and_classification(dock):
    dock.detection_projection_combo.setCurrentIndex(2)
    assert dock.get_config()["detection_projection"] == "alfs"
    dock.classification_projection_combo.setCurrentIndex(3)
    assert dock.get_config()["classification_projection"] == "alfs"


def test_alfs_detection_needs_sampling_mode_integrals(dock, tmp_path,
                                                      monkeypatch):
    import json
    started, warned = [], []
    monkeypatch.setattr(type(dock), "start_worker",
                        lambda self, step: started.append(step))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    (tmp_path / "frames_t").mkdir()
    (tmp_path / "poses_t.json").write_text(
        json.dumps({"images": []}), encoding="utf-8")
    dock.target_folder_edit.setText(str(tmp_path))
    dock.detection_projection_combo.setCurrentIndex(2)

    dock.run_detection()
    assert started == []
    assert any("sampling mode" in text for text in warned)


def test_there_are_no_default_classifiers_for_alfs_to_download(
        dock, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown.append(a[2]))
    dock.classification_projection_combo.setCurrentIndex(3)
    dock.download_classification_models()
    assert any("ALFS" in text for text in shown)
