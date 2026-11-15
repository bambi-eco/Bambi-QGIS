# -*- coding: utf-8 -*-
"""Unit tests for core.camera_calibration.

The image helpers and stereo optimisation run against the real cv2/scipy
from the unit image; pycolmap (not installed) is faked for the SfM
orchestration tests.
"""
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from bambi_wildlife_detection.core import camera_calibration as cc  # noqa: E402
from tests.fakes import make_module  # noqa: E402


class TestFovDiagToMtx:
    def test_ninety_degree_diagonal(self):
        # d = 5000 px, fov 90° -> f = 2500 / tan(45°) = 2500
        mtx = cc.fov_diag_to_mtx(90.0, 4000, 3000)
        assert mtx[0][0] == pytest.approx(2500.0)
        assert mtx[1][1] == pytest.approx(2500.0)
        assert (mtx[0][2], mtx[1][2]) == (2000.0, 1500.0)
        assert mtx[2] == [0.0, 0.0, 1.0]

    def test_narrower_fov_gives_longer_focal(self):
        f_wide = cc.fov_diag_to_mtx(90.0, 640, 512)[0][0]
        f_tele = cc.fov_diag_to_mtx(30.0, 640, 512)[0][0]
        assert f_tele > f_wide


class TestEstimateIntrinsics:
    def test_from_image_dimensions(self):
        calib = cc.estimate_intrinsics_from_image(np.zeros((512, 640, 3)))
        assert calib["mtx"][0][0] == pytest.approx(640 * 1.2)
        assert calib["mtx"][0][2] == 320.0
        assert calib["mtx"][1][2] == 256.0
        assert calib["dist"] == [0.0] * 5


class TestUndistortImg:
    def test_zero_distortion_keeps_image(self):
        img = np.random.default_rng(1).integers(
            0, 255, (32, 32, 3), dtype=np.uint8)
        calib = {"mtx": cc.fov_diag_to_mtx(60.0, 32, 32),
                 "dist": [0.0, 0.0, 0.0, 0.0, 0.0]}
        out = cc.undistort_img(img, calib)
        assert np.array_equal(out, img)

    def test_broken_calibration_returns_original(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert cc.undistort_img(img, {"mtx": None}) is img


class TestLoadImageOrVideoCentral:
    def test_reads_image_file(self, tmp_path):
        path = str(tmp_path / "img.png")
        cv2.imwrite(path, np.full((16, 16, 3), 99, dtype=np.uint8))
        img = cc.load_image_or_video_central(path)
        assert img is not None
        assert img.shape == (16, 16, 3)

    def test_missing_image_returns_none(self, tmp_path):
        assert cc.load_image_or_video_central(str(tmp_path / "x.png")) is None

    def test_unreadable_video_returns_none(self, tmp_path):
        path = tmp_path / "broken.mp4"
        path.write_bytes(b"not a video")
        assert cc.load_image_or_video_central(str(path)) is None


class TestExtractNFrames:
    def _write_video(self, tmp_path, n_frames=12):
        path = str(tmp_path / "clip.avi")
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
        if not writer.isOpened():
            pytest.skip("cv2 VideoWriter has no MJPG codec in this build")
        for i in range(n_frames):
            writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
        writer.release()
        return path

    def test_extracts_evenly_spaced_frames(self, tmp_path):
        video = self._write_video(tmp_path)
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        progress = []
        paths = cc.extract_n_frames_to_dir(
            video, str(out_dir), 4,
            progress_cb=lambda done, total: progress.append((done, total)))
        assert len(paths) == 4
        assert all(p.endswith(".jpg") for p in paths)
        assert progress[-1] == (4, 4)

    def test_unreadable_video_returns_empty(self, tmp_path):
        (tmp_path / "bad.avi").write_bytes(b"nope")
        assert cc.extract_n_frames_to_dir(
            str(tmp_path / "bad.avi"), str(tmp_path), 4) == []


# ---------------------------------------------------------------------------
# Single-camera SfM (fake pycolmap)
# ---------------------------------------------------------------------------

class _FakePoint:
    def __init__(self, error):
        self.error = error


class _FakeCamera:
    def __init__(self, params, model_name):
        self.params = params
        self.model_name = model_name


class _FakeReconstruction:
    def __init__(self, camera, n_images=5, errors=(1.0, 2.0)):
        self.cameras = {1: camera}
        self.images = {i: object() for i in range(n_images)}
        self.points3D = {i: _FakePoint(e) for i, e in enumerate(errors)}


def _install_fake_pycolmap(monkeypatch, reconstructions):
    class ImageReaderOptions:
        camera_model = None
        camera_params = None

    class CameraMode:
        SINGLE = "single"

    module = make_module(
        "pycolmap",
        ImageReaderOptions=ImageReaderOptions,
        CameraMode=CameraMode,
        extract_features=lambda **kw: None,
        match_exhaustive=lambda **kw: None,
        incremental_mapping=lambda **kw: reconstructions,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", module)
    return module


def _write_images(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = str(tmp_path / f"src_{i}.png")
        cv2.imwrite(p, np.full((16, 16, 3), i, dtype=np.uint8))
        paths.append(p)
    return paths


class TestRunSingleCalibration:
    def test_missing_pycolmap_raises(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "pycolmap", None)
        with pytest.raises(RuntimeError, match="pycolmap is required"):
            cc.run_single_calibration(
                {"image_paths": _write_images(tmp_path)})

    def test_too_few_images(self, monkeypatch, tmp_path):
        _install_fake_pycolmap(monkeypatch, {})
        with pytest.raises(RuntimeError, match="at least 3 images"):
            cc.run_single_calibration(
                {"image_paths": _write_images(tmp_path, 2)})

    def test_opencv_model_result(self, monkeypatch, tmp_path):
        camera = _FakeCamera(
            [1000.0, 1010.0, 320.0, 256.0, 0.1, -0.05, 0.001, 0.002, 0.01],
            "OPENCV")
        _install_fake_pycolmap(
            monkeypatch, {0: _FakeReconstruction(camera, errors=(1.0, 3.0))})
        result = cc.run_single_calibration(
            {"image_paths": _write_images(tmp_path), "camera_name": "M30T"})
        assert result["mode"] == "single"
        assert result["camera_name"] == "M30T"
        assert result["mtx"] == [[1000.0, 0.0, 320.0],
                                 [0.0, 1010.0, 256.0],
                                 [0.0, 0.0, 1.0]]
        assert result["dist"] == [0.1, -0.05, 0.001, 0.002, 0.01]
        assert result["ret"] == pytest.approx(2.0)   # mean point error
        assert result["n_images"] == 5

    def test_radial_model_result(self, monkeypatch, tmp_path):
        camera = _FakeCamera([800.0, 320.0, 256.0, 0.05], "SIMPLE_RADIAL")
        _install_fake_pycolmap(
            monkeypatch, {0: _FakeReconstruction(camera)})
        result = cc.run_single_calibration(
            {"image_paths": _write_images(tmp_path)})
        assert result["mtx"][0][0] == 800.0
        assert result["mtx"][1][1] == 800.0
        assert result["dist"] == [0.05, 0.0, 0.0, 0.0, 0.0]

    def test_fov_prior_configures_reader(self, monkeypatch, tmp_path):
        camera = _FakeCamera([1000.0, 1000.0, 8.0, 8.0], "PINHOLE")
        module = _install_fake_pycolmap(
            monkeypatch, {0: _FakeReconstruction(camera)})
        captured = {}

        def capture_extract(**kwargs):
            captured["reader"] = kwargs["reader_options"]

        module.extract_features = capture_extract
        logs = []
        cc.run_single_calibration(
            {"image_paths": _write_images(tmp_path), "fov_diag": 80.0},
            log_fn=logs.append)
        assert captured["reader"].camera_params is not None
        assert any("FoV prior" in m for m in logs)

    def test_empty_reconstruction_raises(self, monkeypatch, tmp_path):
        _install_fake_pycolmap(monkeypatch, {})
        with pytest.raises(RuntimeError, match="no valid reconstruction"):
            cc.run_single_calibration(
                {"image_paths": _write_images(tmp_path)})

    def test_cancel_returns_empty(self, monkeypatch, tmp_path):
        _install_fake_pycolmap(monkeypatch, {})
        result = cc.run_single_calibration(
            {"image_paths": _write_images(tmp_path)},
            cancel_check=lambda: True)
        assert result == {}


# ---------------------------------------------------------------------------
# Stereo calibration (real cv2 + scipy)
# ---------------------------------------------------------------------------

K_W = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 256.0], [0.0, 0.0, 1.0]])
K_T_TRUE = np.array([[800.0, 0.0, 300.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]])

W_POINTS = [
    [100.0, 80.0], [500.0, 80.0], [100.0, 400.0], [500.0, 400.0],
    [320.0, 256.0], [200.0, 300.0], [450.0, 150.0], [150.0, 200.0],
]


def _thermal_points():
    pts = []
    for wx, wy in W_POINTS:
        norm = np.linalg.inv(K_W) @ np.array([wx, wy, 1.0])
        t = K_T_TRUE @ norm
        pts.append([t[0] / t[2], t[1] / t[2]])
    return pts


def _initial_calib(perturb=1.05):
    return {
        "Wide": {"ret": 0.5, "mtx": K_W.tolist(), "dist": [0.0] * 5},
        "Thermal": {"ret": None,
                    "mtx": (K_T_TRUE * [[perturb, 1, perturb],
                                        [1, perturb, perturb],
                                        [1, 1, 1]]).tolist(),
                    "dist": [0.0] * 5},
    }


class TestRunStereoCalibration:
    def test_too_few_points(self):
        with pytest.raises(RuntimeError, match="at least 4 point"):
            cc.run_stereo_calibration({
                "T_points": [[0, 0]] * 3, "W_points": [[0, 0]] * 3,
                "initial_calibration": _initial_calib()})

    def test_missing_thermal_section(self):
        with pytest.raises(RuntimeError, match="must contain both"):
            cc.run_stereo_calibration({
                "T_points": _thermal_points(), "W_points": W_POINTS,
                "initial_calibration": {"Wide": {"mtx": K_W.tolist(),
                                                 "dist": [0.0] * 5}}})

    def test_recovers_thermal_intrinsics(self):
        logs = []
        result = cc.run_stereo_calibration({
            "T_points": _thermal_points(),
            "W_points": W_POINTS,
            "initial_calibration": _initial_calib(),
        }, log_fn=logs.append)

        assert result["mode"] == "stereo"
        # Points were generated with an exact linear relation, so both the
        # homography check and the optimisation must fit almost perfectly.
        assert result["homography_mse"] < 1e-6
        assert result["optimized_mse"] < 1.0
        mtx = np.array(result["Thermal"]["mtx"])
        assert mtx[0, 0] == pytest.approx(K_T_TRUE[0, 0], rel=0.1)
        assert mtx[1, 1] == pytest.approx(K_T_TRUE[1, 1], rel=0.1)
        # The Wide side passes through unchanged
        assert result["Wide"]["mtx"] == K_W.tolist()
        assert any("Optimisation complete" in m for m in logs)

    def test_cancel_returns_empty(self):
        result = cc.run_stereo_calibration({
            "T_points": _thermal_points(),
            "W_points": W_POINTS,
            "initial_calibration": _initial_calib(),
        }, cancel_check=lambda: True)
        assert result == {}


# ---------------------------------------------------------------------------
# Stereo calibration on top of a loaded thermal calibration
# ---------------------------------------------------------------------------

class TestStereoWithALoadedThermalCalibration:
    """With a thermal calibration loaded the points are placed on images
    undistorted with it. The fit then started from "no distortion" and
    saved only what was left over - as if that were the camera's whole
    distortion. What is saved is used on the camera's raw frames."""

    #: The thermal camera as it really is: clearly barrel-distorted.
    D_TRUE = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
    #: ... and the calibration that was loaded for it: close, not exact.
    K_LOADED = K_T_TRUE * [[1.03, 1, 1.02], [1, 1.03, 0.98], [1, 1, 1]]
    D_LOADED = np.array([-0.22, 0.05, 0.0, 0.0, 0.0])

    @classmethod
    def _points(cls):
        """A grid seen by both cameras: the RGB pixels (ideal), the thermal
        ones as the raw image has them, and as placed on the image
        undistorted with the loaded calibration."""
        import cv2

        wide = np.array([[x, y] for x in np.linspace(60, 580, 7)
                         for y in np.linspace(50, 460, 6)])
        rays = np.column_stack([(wide[:, 0] - K_W[0, 2]) / K_W[0, 0],
                                (wide[:, 1] - K_W[1, 2]) / K_W[1, 1],
                                np.ones(len(wide))])
        raw, _j = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3),
                                    np.zeros(3), K_T_TRUE,
                                    cls.D_TRUE.reshape(1, -1))
        placed = cv2.undistortPoints(
            raw.reshape(-1, 1, 2), cls.K_LOADED, cls.D_LOADED.reshape(1, -1),
            P=cls.K_LOADED)
        return wide, raw.reshape(-1, 2), placed.reshape(-1, 2)

    @staticmethod
    def _misfit(result, wide, raw):
        """How far the raw thermal points, undistorted with the saved
        calibration, are from the RGB points - in RGB pixels, RMS."""
        import cv2

        thermal = result["Thermal"]
        back = cv2.undistortPoints(
            raw.reshape(-1, 1, 2).astype(np.float64),
            np.array(thermal["mtx"], dtype=np.float64),
            np.array(thermal["dist"], dtype=np.float64).reshape(1, -1),
            P=K_W).reshape(-1, 2)
        return float(np.sqrt(((back - wide) ** 2).sum(axis=1).mean()))

    def _run(self, placed, wide):
        return cc.run_stereo_calibration({
            "T_points": placed.tolist(), "W_points": wide.tolist(),
            "initial_calibration": {
                "Wide": {"ret": 0.5, "mtx": K_W.tolist(), "dist": [0.0] * 5},
                "Thermal": {"ret": 1.0, "mtx": self.K_LOADED.tolist(),
                            "dist": self.D_LOADED.tolist()}},
            "rgb_pre_undistorted": True, "th_pre_undistorted": True})

    def test_the_saved_calibration_fits_the_raw_frames(self):
        wide, raw, placed = self._points()
        result = self._run(placed, wide)

        assert self._misfit(result, wide, raw) < 1.5
        # the camera's whole distortion, not the little that was left over
        assert result["Thermal"]["dist"][0] == pytest.approx(
            self.D_TRUE[0], abs=0.08)

    def test_only_the_leftover_would_not_fit_them(self):
        """What used to be saved, for comparison: a distortion near zero,
        which on the raw frames is off by many pixels."""
        import cv2

        wide, raw, placed = self._points()
        leftover = {"Thermal": {"mtx": self.K_LOADED.tolist(),
                                "dist": [0.0] * 5}}
        assert self._misfit(leftover, wide, raw) > 5.0
        # sanity: the points really were placed where the loaded
        # calibration puts them
        back = cc._raw_points(placed, self.K_LOADED, self.D_LOADED)
        assert np.allclose(back.reshape(-1, 2), raw, atol=1e-3)
        assert cv2 is not None

    def test_without_a_loaded_calibration_nothing_changes(self):
        """Points placed on the raw images, no distortion to start from."""
        result = cc.run_stereo_calibration({
            "T_points": _thermal_points(), "W_points": W_POINTS,
            "initial_calibration": _initial_calib()})
        assert result["optimized_mse"] < 1.0

    def test_a_loaded_calibration_without_distortion_is_left_as_it_is(self):
        """Nothing to take the points back through."""
        result = cc.run_stereo_calibration({
            "T_points": _thermal_points(), "W_points": W_POINTS,
            "initial_calibration": _initial_calib(),
            "th_pre_undistorted": True})
        assert result["optimized_mse"] < 1.0
