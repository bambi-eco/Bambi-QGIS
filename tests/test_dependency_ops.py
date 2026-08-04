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


_CONFLICT_OUTPUT = (
    "ERROR: Cannot install bambi-detection because these package versions "
    "have conflicting dependencies.\n"
    "    ultralytics 8.3.75 depends on numpy<=2.1.1 and >=1.23.0\n"
    "    The user requested (constraint) numpy==2.2.6\n"
    "ERROR: ResolutionImpossible\n"
)


class _FakePopen:
    """Context-manager Popen whose stdout yields canned lines."""

    returncode = 0
    lines = ["Collecting pkg\n", "Successfully installed pkg\n"]
    last_cmd = None
    cmds = []

    def __init__(self, cmd, **kwargs):
        type(self).last_cmd = cmd
        type(self).cmds = type(self).cmds + [cmd]
        self.stdout = io.StringIO("".join(self.lines))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _reset_fake_popen():
    _FakePopen.cmds = []
    _FakePopen.lines = ["Collecting pkg\n", "Successfully installed pkg\n"]
    yield
    _FakePopen.cmds = []


class TestRunPip:
    @pytest.fixture(autouse=True)
    def _no_shadow_repair(self, monkeypatch):
        """The post-install shadow check spawns its own probe – silence it
        here; it has its own tests below."""
        monkeypatch.setattr(ops, "_repair_user_site_shadows", lambda log_fn: None)

    def test_streams_output_and_succeeds(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 0)
        # Detection unavailable -> no constraint pinning injected.
        monkeypatch.setattr(ops, "_write_constraints_file", lambda log_fn: None)
        logs = []
        ops._run_pip(["install", "example"], logs.append)
        assert _FakePopen.last_cmd[-4:] == ['-m', 'pip', 'install', 'example']
        assert any("Successfully installed" in m for m in logs)
        assert any("pip finished successfully" in m for m in logs)

    def test_nonzero_exit_raises(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 1)
        monkeypatch.setattr(ops, "_write_constraints_file", lambda log_fn: None)
        with pytest.raises(RuntimeError, match="pip exited with code 1"):
            ops._run_pip(["install", "example"], lambda m: None)

    def test_install_injects_constraints(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 0)
        monkeypatch.setattr(ops, "_write_constraints_file",
                            lambda log_fn: "/tmp/c.txt")
        ops._run_pip(["install", "--force-reinstall", "example"], lambda m: None)
        assert _FakePopen.last_cmd[-2:] == ['-c', '/tmp/c.txt']

    def test_custom_index_url_is_not_pinned(self, monkeypatch):
        # A custom --index-url (e.g. the CUDA torch index) may not host the
        # pinned numpy/scipy, so pinning must be skipped there.
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 0)
        monkeypatch.setattr(ops, "_write_constraints_file",
                            lambda log_fn: pytest.fail("must not build constraints"))
        ops._run_pip(
            ["install", "torch", "--index-url", "https://x/whl/cu121"],
            lambda m: None)
        assert "-c" not in _FakePopen.last_cmd

    def test_uninstall_is_not_pinned(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 0)
        monkeypatch.setattr(ops, "_write_constraints_file",
                            lambda log_fn: pytest.fail("must not build constraints"))
        ops._run_pip(["uninstall", "torch", "-y"], lambda m: None)
        assert "-c" not in _FakePopen.last_cmd

    def test_unsatisfiable_pin_retries_unpinned(self, monkeypatch):
        # A dependency capping numpy below the QGIS build (ultralytics) makes
        # the pinned resolve impossible – the install must still go through.
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(ops, "_write_constraints_file",
                            lambda log_fn: "/tmp/c.txt")

        def rc_by_call(self):
            return 1 if len(_FakePopen.cmds) == 1 else 0

        monkeypatch.setattr(_FakePopen, "lines", [_CONFLICT_OUTPUT])
        monkeypatch.setattr(_FakePopen, "returncode", property(rc_by_call))

        repaired = []
        monkeypatch.setattr(ops, "_repair_user_site_shadows", repaired.append)

        ops._run_pip(["install", "example"], lambda m: None)

        assert len(_FakePopen.cmds) == 2
        assert _FakePopen.cmds[0][-2:] == ['-c', '/tmp/c.txt']
        assert '-c' not in _FakePopen.cmds[1]      # retry drops the pin
        assert repaired                            # shadows cleaned up after

    def test_unrelated_failure_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(_FakePopen, "returncode", 1)
        monkeypatch.setattr(_FakePopen, "lines", ["ERROR: Network unreachable\n"])
        monkeypatch.setattr(ops, "_write_constraints_file",
                            lambda log_fn: "/tmp/c.txt")
        with pytest.raises(RuntimeError, match="pip exited with code 1"):
            ops._run_pip(["install", "example"], lambda m: None)
        assert len(_FakePopen.cmds) == 1


class TestUserSiteShadows:
    def setup_method(self):
        ops._bundled_versions_cache = {"numpy": "2.2.6", "scipy": "1.13.0"}

    def teardown_method(self):
        ops._bundled_versions_cache = None

    def _probe_returning(self, stdout):
        def fake_run(cmd, **kwargs):
            assert "-s" not in cmd  # user site must stay visible here

            class R:
                pass
            R.stdout = stdout
            return R
        return fake_run

    def test_reports_only_diverging_user_site_copies(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            self._probe_returning("numpy==2.1.1\nscipy==1.13.0\n"))
        # scipy matches the bundled build -> harmless, must not be reported.
        assert ops._detect_user_site_shadows() == {"numpy": "2.1.1"}

    def test_no_shadows_when_user_site_is_clean(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", self._probe_returning(""))
        assert ops._detect_user_site_shadows() == {}

    def test_skipped_when_bundled_versions_unknown(self, monkeypatch):
        ops._bundled_versions_cache = {}
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: pytest.fail("must not probe without a baseline"))
        assert ops._detect_user_site_shadows() == {}

    def test_probe_failure_degrades_to_empty(self, monkeypatch):
        def boom(cmd, **kwargs):
            raise OSError("no python")

        monkeypatch.setattr(subprocess, "run", boom)
        assert ops._detect_user_site_shadows(log_fn=lambda m: None) == {}

    def test_repair_uninstalls_the_shadowing_copy(self, monkeypatch):
        detected = [{"numpy": "2.1.1"}, {}]
        monkeypatch.setattr(ops, "_detect_user_site_shadows",
                            lambda log_fn=None: detected.pop(0))
        calls = []
        monkeypatch.setattr(ops, "_run_pip",
                            lambda args, log_fn: calls.append(args))
        logs = []
        ops._repair_user_site_shadows(logs.append)
        assert calls == [["uninstall", "-y", "numpy"]]
        assert not any("still shadow" in m for m in logs)

    def test_repair_is_a_noop_without_shadows(self, monkeypatch):
        monkeypatch.setattr(ops, "_detect_user_site_shadows",
                            lambda log_fn=None: {})
        monkeypatch.setattr(ops, "_run_pip",
                            lambda *a, **k: pytest.fail("must not run pip"))
        ops._repair_user_site_shadows(lambda m: None)

    def test_repair_warns_when_removal_did_not_help(self, monkeypatch):
        monkeypatch.setattr(ops, "_detect_user_site_shadows",
                            lambda log_fn=None: {"numpy": "2.1.1"})
        monkeypatch.setattr(ops, "_run_pip", lambda args, log_fn: None)
        logs = []
        ops._repair_user_site_shadows(logs.append)
        assert any("still shadow" in m for m in logs)

    def test_repair_survives_a_failing_uninstall(self, monkeypatch):
        monkeypatch.setattr(ops, "_detect_user_site_shadows",
                            lambda log_fn=None: {"numpy": "2.1.1"})

        def boom(args, log_fn):
            raise RuntimeError("pip exited with code 1")

        monkeypatch.setattr(ops, "_run_pip", boom)
        logs = []
        ops._repair_user_site_shadows(logs.append)  # must not raise
        assert any("Could not remove" in m for m in logs)


class TestBundledVersionPinning:
    def setup_method(self):
        ops._bundled_versions_cache = None

    def teardown_method(self):
        ops._bundled_versions_cache = None

    def test_detect_parses_probe_output(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert "-s" in cmd  # user site must be disabled to see bundled build

            class R:
                stdout = "numpy==2.2.6\nscipy==1.13.0\n"
            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ops._detect_bundled_versions() == {
            "numpy": "2.2.6", "scipy": "1.13.0"}

    def test_detect_result_is_cached(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)

            class R:
                stdout = "numpy==2.2.6\n"
            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        ops._detect_bundled_versions()
        ops._detect_bundled_versions()
        assert len(calls) == 1  # second call served from cache

    def test_detect_failure_degrades_to_empty(self, monkeypatch):
        def boom(cmd, **kwargs):
            raise OSError("no python")

        monkeypatch.setattr(subprocess, "run", boom)
        assert ops._detect_bundled_versions(log_fn=lambda m: None) == {}

    def test_write_constraints_file_contents(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ops, "_detect_bundled_versions",
                            lambda log_fn=None: {"numpy": "2.2.6", "scipy": "1.13.0"})
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        path = ops._write_constraints_file(lambda m: None)
        content = open(path).read()
        assert "numpy==2.2.6" in content
        assert "scipy==1.13.0" in content

    def test_write_constraints_none_when_nothing_detected(self, monkeypatch):
        monkeypatch.setattr(ops, "_detect_bundled_versions",
                            lambda log_fn=None: {})
        assert ops._write_constraints_file(lambda m: None) is None


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
