# -*- coding: utf-8 -*-
"""Camera Calibration Wizard: a stereo calibration that cannot be relied on
is shown as one.

The stereo fit finds nine numbers for the thermal camera. With a handful of
point pairs - four used to be enough to run it - it goes through every one of
them whatever the camera is, and the wizard showed that as a calibration with
a tiny error, in green. The checks are in ``core.camera_calibration`` and
tested there; this is what the wizard does with them.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import numpy as np
import pytest

from qgis.PyQt.QtCore import QObject, pyqtSignal

from bambi_wildlife_detection import bambi_camera_calibration as calib
from bambi_wildlife_detection.bambi_camera_calibration import (
    CameraCalibrationWizard)

CAMERA = {"ret": None, "name": "x", "dist": [0.0] * 5,
          "mtx": [[800.0, 0.0, 320.0], [0.0, 800.0, 256.0], [0.0, 0.0, 1.0]]}
SOUND = {"mode": "stereo", "homography_mse": 2.0, "optimized_mse": 0.64,
         "n_points": 20, "fit_rms_px": 0.8, "held_out_rms_px": 1.1,
         "warnings": [], "Thermal": dict(CAMERA), "Wide": dict(CAMERA)}
DOUBTFUL = dict(SOUND, n_points=6, held_out_rms_px=None,
                warnings=["Only 6 point pairs. Place 12 or more."])


@pytest.fixture
def wizard(monkeypatch):
    dialog = CameraCalibrationWizard()
    dialog._mode = "stereo"
    # the page of the point pairs is built when the wizard gets there
    dialog._pages.insertWidget(1, dialog._build_calib_page_stereo())
    # the preview needs the frames; it is not what is looked at here
    monkeypatch.setattr(dialog, "_populate_result_preview", lambda result: None)
    yield dialog
    dialog._worker = None
    dialog._worker_thread = None
    dialog.close()
    dialog.deleteLater()


def test_a_doubtful_result_is_shown_as_one(wizard):
    wizard._populate_results_page(DOUBTFUL)

    text = wizard._result_metrics_lbl.text()
    assert "Point pairs                : 6" in text
    assert "not checked - too few point pairs" in text
    assert "Not to be relied on as it is:" in text
    assert "Only 6 point pairs." in text
    assert "#ffcc66" in wizard._result_metrics_lbl.styleSheet()


def test_a_sound_result_is_green(wizard):
    wizard._populate_results_page(DOUBTFUL)
    wizard._populate_results_page(SOUND)       # the label is not left amber

    text = wizard._result_metrics_lbl.text()
    assert "Fit, points left out (RMS) : 1.10 px" in text
    assert "Not to be relied on" not in text
    assert "#88ff88" in wizard._result_metrics_lbl.styleSheet()


def test_a_single_camera_result_is_green_after_a_doubtful_stereo_one(wizard):
    wizard._populate_results_page(DOUBTFUL)
    wizard._populate_results_page({
        "mode": "single", "ret": 0.5, "n_images": 12, "n_points3d": 900,
        "mtx": CAMERA["mtx"], "dist": CAMERA["dist"], "camera_name": "Wide"})

    assert "#88ff88" in wizard._result_metrics_lbl.styleSheet()


@pytest.mark.parametrize("placed, expected", [
    (2, "Total: 2  (at least 6 needed, 12 or more recommended)"),
    (7, "Total: 7  (12 or more recommended)"),
    (12, "Total: 12"),
])
def test_the_point_counter_says_how_many_are_needed(wizard, placed, expected):
    points = [(10 * i, 10 * i) for i in range(placed)]
    wizard._pairs_annot = [{"rgb_pts": list(points), "th_pts": list(points)}]
    wizard._pair_idx = 0

    wizard._refresh_annot_state()

    assert wizard._pt_count_lbl.text().endswith(expected)


# -- what the calibration is told for judging its result ---------------------

class _Recorder(QObject):
    """Stands in for the worker: keeps what it was asked to do."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    asked = []

    def __init__(self, mode, params):
        super().__init__()
        type(self).asked.append(params)

    def moveToThread(self, thread):
        pass

    def run(self):
        pass

    def cancel(self):
        pass


class _NoThread(QObject):
    started = pyqtSignal()

    def start(self):
        pass

    def isRunning(self):
        return False


@pytest.fixture
def started(wizard, monkeypatch):
    """Start a stereo calibration of twelve pairs; returns what the worker
    was given."""
    _Recorder.asked = []
    monkeypatch.setattr(calib, "_CalibWorker", _Recorder)
    monkeypatch.setattr(calib, "QThread", _NoThread)
    frames = {"rgb.jpg": np.zeros((1080, 1920, 3), dtype=np.uint8),
              "thermal.jpg": np.zeros((512, 640, 3), dtype=np.uint8)}
    monkeypatch.setattr(calib, "_load_image_or_video_central", frames.get)
    points = [(40 * i, 30 * i) for i in range(12)]
    wizard._rgb_paths, wizard._th_paths = ["rgb.jpg"], ["thermal.jpg"]
    wizard._pairs_annot = [{"rgb_pts": list(points), "th_pts": list(points)}]

    def start():
        wizard._start_calibration()
        return _Recorder.asked[-1]
    return start


def test_the_calibration_is_told_the_size_of_the_thermal_frame(wizard, started):
    wizard._rgb_fov_spin.setValue(84.0)
    wizard._th_fov_spin.setValue(61.0)

    params = started()

    assert params["th_image_size"] == (640, 512)
    assert params["initial_focal_known"] is True


def test_a_focal_length_guessed_from_the_frame_is_said_to_be_one(
        wizard, started):
    wizard._rgb_fov_spin.setValue(84.0)
    wizard._th_fov_spin.setValue(0.0)           # no field of view given

    assert started()["initial_focal_known"] is False


def test_a_loaded_calibration_is_a_known_focal_length(wizard, started):
    wizard._rgb_calib_data = dict(CAMERA)
    wizard._th_calib_data = dict(CAMERA)
    wizard._rgb_fov_spin.setValue(0.0)
    wizard._th_fov_spin.setValue(0.0)

    params = started()

    assert params["initial_focal_known"] is True
    assert params["th_image_size"] == (640, 512)
    assert params["th_pre_undistorted"] is True


def test_five_pairs_do_not_start_a_calibration(wizard, started, monkeypatch):
    said = []
    monkeypatch.setattr(calib.QMessageBox, "warning",
                        lambda *args: said.append(args[2]))
    wizard._pairs_annot[0]["rgb_pts"] = wizard._pairs_annot[0]["rgb_pts"][:5]

    wizard._start_calibration()

    assert _Recorder.asked == []
    assert "at least 6 corresponding point pairs" in said[0].replace("\n", " ")
