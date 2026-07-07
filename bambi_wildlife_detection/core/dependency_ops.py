# -*- coding: utf-8 -*-
"""Headless dependency-management operations.

Moved verbatim from ``bambi_dependency_manager.py`` (which re-exports every
name here): version probing, pip invocation and GitHub-ZIP installation.
The dialog module keeps only the Qt widget code.
"""

import os
import sys
import subprocess  # nosec B404
import zipfile
import importlib.metadata


_DJI_SDK_URL = (
    'https://terra-1-g.djicdn.com/2640963bcd8e45c0a4f0cb9829739d6b/'
    'TSDK/v1.8(16.1)/dji_thermal_sdk_v1.8_20250829.zip'
)

# Tested version ranges per pip distribution name (or special key for non-pip packages).
# None means no bound (any version is accepted).
_VERSION_RANGES = {
    'bambi-detection': ("0.4.0", "0.4.0"),
    'AlfsPy': ("0.0.0", "0.0.0"),
    'pycolmap': ('4.0.3', '4.0.3'),
    'boxmot': ('17.0.0', '18.0.0'),
    'georef-tracker': ("0.1.0", "0.1.0"),
    'torch': ("2.5.1", "2.11.0"),
    'torchvision': ("0.20.1", "0.26.0"),
    'dji-thermal-sdk': ('1.7', '1.8'),
    'fiona': ('1.10.1', '1.10.1'),
    'simplekml': ('1.3.6', '1.3.6'),
}


def _git_available():
    """Return git version string if git is on PATH, otherwise None."""
    try:
        kwargs = dict(capture_output=True, check=True, timeout=5, text=True)
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(['git', '--version'], **kwargs)  # nosec B603 B607
        return result.stdout.strip()
    except Exception:
        return None


def _get_dji_sdk_version(plugins_dir):
    """Scan plugins_dir for an extracted DJI Thermal SDK folder and return its version string."""
    import re
    if not plugins_dir or not os.path.isdir(plugins_dir):
        return None
    pattern = re.compile(r'^dji_thermal_sdk_v([\d.]+)_\d+$')
    for entry in os.listdir(plugins_dir):
        if os.path.isdir(os.path.join(plugins_dir, entry)):
            m = pattern.match(entry)
            if m:
                return m.group(1)
    return None


def _get_version_status(dist_name, plugins_dir=None):
    """Return (version_str, status) for a distribution or special package.

    status is one of:
      'not_found' – package not installed
      'ok'        – installed and within the tested range defined in _VERSION_RANGES
      'untested'  – installed but outside the tested range
    """
    if not dist_name:
        return None, 'not_found'

    # DJI Thermal SDK is not a pip package – detect via extracted folder name.
    if dist_name == 'dji-thermal-sdk':
        ver_str = _get_dji_sdk_version(plugins_dir)
        if ver_str is None:
            return None, 'not_found'
    else:
        try:
            ver_str = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            return None, 'not_found'

    min_ver, max_ver = _VERSION_RANGES.get(dist_name, (None, None))
    if min_ver is None and max_ver is None:
        return ver_str, 'ok'

    try:
        from packaging.version import Version
        ver = Version(ver_str)
        if (min_ver and ver < Version(min_ver)) or (max_ver and ver > Version(max_ver)):
            return ver_str, 'untested'
    except Exception:  # nosec B110
        pass  # if packaging is unavailable or version is unparseable, assume ok
    return ver_str, 'ok'


def _torch_build_variant(ver_str):
    """Classify a torch/torchvision version string by compute platform.

    PyTorch encodes the platform in the local version segment
    (e.g. ``2.5.1+cu121``, ``2.5.1+cpu``). Wheels from PyPI carry no
    segment; on Windows those are always CPU-only. Returns ``'cuda'``,
    ``'rocm'``, ``'cpu'`` or ``None`` (unknown).
    """
    local = ver_str.split('+', 1)[1] if '+' in ver_str else ''
    if local.startswith('cu'):
        return 'cuda'
    if local.startswith('rocm'):
        return 'rocm'
    if local.startswith('cpu') or sys.platform == 'win32':
        return 'cpu'
    return None


def _find_python():
    """Return the Python interpreter suitable for running pip.

    On Windows, sys.executable inside QGIS is the QGIS application binary
    (e.g. qgis-ltr-bin.exe), not python.exe.  Running that binary with
    '-m pip' would relaunch the QGIS GUI and hang.  We look for python.exe /
    python3.exe in the same directory first.
    """
    exe_dir = os.path.dirname(sys.executable)
    for name in ('python3.exe', 'python.exe', 'python3', 'python'):
        candidate = os.path.join(exe_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return sys.executable  # fallback – may not work on Windows QGIS


def _run_pip(args, log_fn):
    python = _find_python()
    # -u: force unbuffered stdout/stderr so lines arrive in real time
    cmd = [python, '-u', '-m', 'pip'] + args
    log_fn(f'Python: {python}')
    log_fn('$ ' + ' '.join(cmd[1:]))  # omit python path for readability

    popen_kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout stream
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,                  # line-buffered on our side
    )
    if sys.platform == 'win32':
        popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

    with subprocess.Popen(cmd, **popen_kwargs) as proc:  # nosec B603
        for line in proc.stdout:
            log_fn(line.rstrip())

    if proc.returncode != 0:
        raise RuntimeError(f'pip exited with code {proc.returncode}')
    log_fn('pip finished successfully')


def _install_github_zip(zip_url, pkg_key, plugins_dir, log_fn):
    """Download a GitHub archive ZIP, extract it, and pip-install the result."""
    import requests as _requests
    zip_path = os.path.join(plugins_dir, f'{pkg_key}_src.zip')

    log_fn(f'Downloading {zip_url} …')
    try:
        response = _requests.get(zip_url, stream=True, timeout=60)
        log_fn(f'HTTP status: {response.status_code}')
        response.raise_for_status()
        total = int(response.headers.get('Content-Length', 0))
        if total:
            log_fn(f'Archive size: {total / 1024:.0f} KB')
        downloaded = 0
        last_pct = -25
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = (downloaded * 100) // total
                        if pct >= last_pct + 25:
                            log_fn(f'  {downloaded / 1024:.0f} / {total / 1024:.0f} KB ({pct}%)')
                            last_pct = pct
                    else:
                        log_fn(f'  {downloaded / 1024:.0f} KB …')
    except Exception as exc:
        raise RuntimeError(f'Download failed: {exc}') from exc
    log_fn(f'Download complete ({downloaded / 1024:.0f} KB)')

    log_fn('Extracting archive …')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        top_dirs = {entry.split('/')[0] for entry in zf.namelist() if entry}
        num_files = len(zf.namelist())
        zf.extractall(plugins_dir)
    os.remove(zip_path)
    log_fn(f'Extracted {num_files} files')

    # GitHub archives always produce exactly one top-level folder (repo-branch/)
    pkg_path = (
        os.path.join(plugins_dir, top_dirs.pop())
        if len(top_dirs) == 1
        else plugins_dir
    )
    log_fn(f'Source directory: {pkg_path}')
    _run_pip(['install', '--force-reinstall', pkg_path], log_fn)
