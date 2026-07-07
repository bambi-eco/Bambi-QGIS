# -*- coding: utf-8 -*-
"""Unit tests for core.dependency_ops (pip/version helpers of the
Dependency Manager dialog). subprocess and network access are mocked."""
import io
import subprocess
import sys
import zipfile

import pytest

from bambi_wildlife_detection.core import dependency_ops as ops


class TestGitAvailable:
    def test_returns_version_string(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd == ['git', '--version']

            class R:
                stdout = "git version 2.44.0\n"
            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ops._git_available() == "git version 2.44.0"

    def test_returns_none_when_git_missing(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ops._git_available() is None


class TestDjiSdkVersion:
    def test_finds_version_from_folder_name(self, tmp_path):
        (tmp_path / "dji_thermal_sdk_v1.8_20250829").mkdir()
        assert ops._get_dji_sdk_version(str(tmp_path)) == "1.8"

    def test_ignores_files_and_foreign_folders(self, tmp_path):
        (tmp_path / "dji_thermal_sdk_v1.8_20250829.zip").write_bytes(b"")
        (tmp_path / "some_other_folder").mkdir()
        assert ops._get_dji_sdk_version(str(tmp_path)) is None

    def test_missing_dir(self, tmp_path):
        assert ops._get_dji_sdk_version(None) is None
        assert ops._get_dji_sdk_version(str(tmp_path / "nope")) is None


class TestGetVersionStatus:
    def test_not_installed(self):
        assert ops._get_version_status("definitely-not-a-package-xyz") == (None, 'not_found')
        assert ops._get_version_status("") == (None, 'not_found')

    def test_installed_without_range_is_ok(self):
        ver, status = ops._get_version_status("pytest")
        assert status == 'ok'
        assert ver

    def test_installed_within_range(self, monkeypatch):
        monkeypatch.setitem(ops._VERSION_RANGES, "pytest", ("0.1", "9999.0"))
        assert ops._get_version_status("pytest")[1] == 'ok'

    def test_installed_outside_range_is_untested(self, monkeypatch):
        monkeypatch.setitem(ops._VERSION_RANGES, "pytest", ("0.1", "0.2"))
        assert ops._get_version_status("pytest")[1] == 'untested'

    def test_dji_sdk_probed_via_folder(self, tmp_path):
        (tmp_path / "dji_thermal_sdk_v1.8_20250829").mkdir()
        ver, status = ops._get_version_status("dji-thermal-sdk", str(tmp_path))
        assert (ver, status) == ("1.8", 'ok')
        assert ops._get_version_status("dji-thermal-sdk", None) == (None, 'not_found')


class TestTorchBuildVariant:
    @pytest.mark.parametrize("ver,expected", [
        ("2.5.1+cu121", "cuda"),
        ("2.5.1+rocm5.7", "rocm"),
        ("2.5.1+cpu", "cpu"),
    ])
    def test_local_segment_classification(self, ver, expected):
        assert ops._torch_build_variant(ver) == expected

    def test_bare_version_on_windows_is_cpu(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert ops._torch_build_variant("2.5.1") == "cpu"

    def test_bare_version_elsewhere_is_unknown(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert ops._torch_build_variant("2.5.1") is None


class TestFindPython:
    def test_prefers_python_next_to_executable(self, monkeypatch, tmp_path):
        exe_dir = tmp_path / "bin"
        exe_dir.mkdir()
        (exe_dir / "python3.exe").write_bytes(b"")
        monkeypatch.setattr(sys, "executable", str(exe_dir / "qgis-ltr-bin.exe"))
        assert ops._find_python() == str(exe_dir / "python3.exe")

    def test_falls_back_to_sys_executable(self, monkeypatch, tmp_path):
        exe_dir = tmp_path / "empty"
        exe_dir.mkdir()
        fake_exe = str(exe_dir / "qgis-bin.exe")
        monkeypatch.setattr(sys, "executable", fake_exe)
        assert ops._find_python() == fake_exe


class _FakePopen:
    """Context-manager Popen whose stdout yields canned lines."""

    returncode = 0
    lines = ["Collecting pkg\n", "Successfully installed pkg\n"]
    last_cmd = None

    def __init__(self, cmd, **kwargs):
        type(self).last_cmd = cmd
        self.stdout = io.StringIO("".join(self.lines))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRunPip:
    def test_streams_output_and_succeeds(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 0)
        logs = []
        ops._run_pip(["install", "example"], logs.append)
        assert _FakePopen.last_cmd[-4:] == ['-m', 'pip', 'install', 'example']
        assert any("Successfully installed" in m for m in logs)
        assert any("pip finished successfully" in m for m in logs)

    def test_nonzero_exit_raises(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 1)
        with pytest.raises(RuntimeError, match="pip exited with code 1"):
            ops._run_pip(["install", "example"], lambda m: None)


class TestInstallGithubZip:
    def _make_archive_bytes(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("repo-main/setup.py", "# setup")
            zf.writestr("repo-main/pkg/__init__.py", "")
        return buffer.getvalue()

    def test_downloads_extracts_and_installs(self, monkeypatch, tmp_path):
        data = self._make_archive_bytes()

        class FakeResponse:
            status_code = 200
            headers = {"Content-Length": str(len(data))}

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=None):
                yield data

        import requests
        monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResponse())

        pip_calls = []
        monkeypatch.setattr(ops, "_run_pip", lambda args, log_fn: pip_calls.append(args))

        logs = []
        ops._install_github_zip(
            "https://github.com/x/repo/archive/main.zip", "repo",
            str(tmp_path), logs.append)

        assert (tmp_path / "repo-main" / "setup.py").exists()
        assert not (tmp_path / "repo_src.zip").exists()   # zip cleaned up
        assert pip_calls == [
            ['install', '--force-reinstall', str(tmp_path / "repo-main")]]

    def test_download_failure_raises(self, monkeypatch, tmp_path):
        import requests

        def failing_get(url, **kw):
            raise ConnectionError("offline")

        monkeypatch.setattr(requests, "get", failing_get)
        with pytest.raises(RuntimeError, match="Download failed"):
            ops._install_github_zip(
                "https://github.com/x/repo/archive/main.zip", "repo",
                str(tmp_path), lambda m: None)
