# -*- coding: utf-8 -*-
"""Unit tests for the bambi_thermal shim (DJI SDK / exiftool discovery).

The real parsing lives in the ``bambi.thermal`` package (bambi_detection),
which is not installed in the unit image — a fake module is injected before
importing the shim. Only the path-discovery logic is under test here.
"""
import os
import sys

import pytest

from tests.fakes import make_module


class _BaseThermal:
    """Records the constructor arguments the shim passes down."""

    def __init__(self, dtype=None, sdk_dir=None, exiftool_path=None):
        self.dtype = dtype
        self.sdk_dir = sdk_dir
        self.exiftool_path = exiftool_path


@pytest.fixture
def bambi_thermal(monkeypatch):
    """Import bambi_wildlife_detection.bambi_thermal against a fake bambi."""
    parser = make_module(
        "bambi.thermal.thermal_parser",
        Thermal=_BaseThermal,
        parse_dji_rjpeg=lambda *a, **k: ("raw", "meta"),
        apply_colormap=lambda *a, **k: "colored",
        _read_dji_xmp_params=lambda fp: {"emissivity": 0.95, "source": fp},
    )
    thermal_pkg = make_module("bambi.thermal")
    thermal_pkg.__path__ = []
    thermal_pkg.thermal_parser = parser
    bambi_pkg = make_module("bambi")
    bambi_pkg.__path__ = []
    bambi_pkg.thermal = thermal_pkg
    monkeypatch.setitem(sys.modules, "bambi", bambi_pkg)
    monkeypatch.setitem(sys.modules, "bambi.thermal", thermal_pkg)
    monkeypatch.setitem(sys.modules, "bambi.thermal.thermal_parser", parser)

    monkeypatch.delitem(
        sys.modules, "bambi_wildlife_detection.bambi_thermal", raising=False)
    import bambi_wildlife_detection.bambi_thermal as module
    yield module
    sys.modules.pop("bambi_wildlife_detection.bambi_thermal", None)


def _make_sdk(root, version_folder, system="windows", arch="x64", ext="dll",
              with_dll=True):
    sdk_dir = os.path.join(
        str(root), version_folder, "utility", "bin", system, f"release_{arch}")
    os.makedirs(sdk_dir, exist_ok=True)
    if with_dll:
        with open(os.path.join(sdk_dir, f"libdirp.{ext}"), "wb") as fh:
            fh.write(b"\x00")
    return sdk_dir


class TestFindSdkDir:
    @pytest.fixture(autouse=True)
    def windows_platform(self, monkeypatch):
        import platform
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "architecture", lambda: ("64bit", ""))

    def test_no_plugins_dir_returns_none(self, bambi_thermal, monkeypatch, tmp_path):
        monkeypatch.setattr(
            bambi_thermal, "_PLUGINS_DIR", str(tmp_path / "does_not_exist"))
        assert bambi_thermal._find_sdk_dir() is None

    def test_finds_sdk_dll_dir(self, bambi_thermal, monkeypatch, tmp_path):
        expected = _make_sdk(tmp_path, "dji_thermal_sdk_v1.4_20220929")
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        assert bambi_thermal._find_sdk_dir() == expected

    def test_newest_version_wins(self, bambi_thermal, monkeypatch, tmp_path):
        _make_sdk(tmp_path, "dji_thermal_sdk_v1.4_20220929")
        newer = _make_sdk(tmp_path, "dji_thermal_sdk_v1.5_20240507")
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        assert bambi_thermal._find_sdk_dir() == newer

    def test_version_without_dll_is_skipped(self, bambi_thermal, monkeypatch, tmp_path):
        older = _make_sdk(tmp_path, "dji_thermal_sdk_v1.4_20220929")
        _make_sdk(tmp_path, "dji_thermal_sdk_v1.6_20250101", with_dll=False)
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        assert bambi_thermal._find_sdk_dir() == older

    def test_linux_layout(self, bambi_thermal, monkeypatch, tmp_path):
        import platform
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        expected = _make_sdk(
            tmp_path, "dji_thermal_sdk_v1.4_20220929",
            system="linux", ext="so")
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        assert bambi_thermal._find_sdk_dir() == expected


class TestFindExiftool:
    def test_windows_exe_found(self, bambi_thermal, monkeypatch, tmp_path):
        import platform
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        exe = tmp_path / "exiftool-12.35.exe"
        exe.write_bytes(b"\x00")
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        assert bambi_thermal._find_exiftool() == str(exe)

    def test_linux_binary_name(self, bambi_thermal, monkeypatch, tmp_path):
        import platform
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        exe = tmp_path / "exiftool"
        exe.write_bytes(b"\x00")
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        assert bambi_thermal._find_exiftool() == str(exe)

    def test_missing_returns_none(self, bambi_thermal, monkeypatch, tmp_path):
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        assert bambi_thermal._find_exiftool() is None


class TestThermalShim:
    def test_read_thermal_meta_passthrough(self, bambi_thermal):
        meta = bambi_thermal.read_thermal_meta("photo.jpg")
        assert meta == {"emissivity": 0.95, "source": "photo.jpg"}

    def test_thermal_wires_discovered_paths(self, bambi_thermal, monkeypatch, tmp_path):
        import platform
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "architecture", lambda: ("64bit", ""))
        sdk = _make_sdk(tmp_path, "dji_thermal_sdk_v1.4_20220929")
        exe = tmp_path / "exiftool-12.35.exe"
        exe.write_bytes(b"\x00")
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))

        import numpy as np
        thermal = bambi_thermal.Thermal()
        assert thermal.sdk_dir == sdk
        assert thermal.exiftool_path == str(exe)
        assert thermal.dtype is np.float32

    def test_thermal_with_nothing_installed(self, bambi_thermal, monkeypatch, tmp_path):
        monkeypatch.setattr(bambi_thermal, "_PLUGINS_DIR", str(tmp_path))
        thermal = bambi_thermal.Thermal()
        assert thermal.sdk_dir is None
        assert thermal.exiftool_path is None
