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

# ABI-sensitive packages that QGIS ships as compiled builds in its (read-only)
# application site-packages.  Our pip installs land in the *user* site, so a
# transitive dependency that pulls a newer numpy/scipy silently *shadows* the
# bundled build and breaks QGIS at import time (e.g. the numpy 2.1 vs 2.2
# ``_no_nep50_warning`` skew that killed GeoTIFF export, or a scipy ABI
# mismatch).  Every install is pinned to the versions QGIS already ships via a
# pip constraints file so no dependency can move them out from under QGIS.
_BUNDLED_PIN_PACKAGES = ('numpy', 'scipy')

# Cache for _detect_bundled_versions: None = not probed yet, dict = result.
_bundled_versions_cache = None

# Tested version ranges per pip distribution name (or special key for non-pip packages).
# None means no bound (any version is accepted).
_VERSION_RANGES = {
    'bambi-detection': ("0.5.0", "0.5.0"),
    'AlfsPy': ("0.0.0", "1.0.0"),
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


def _detect_bundled_versions(log_fn=None):
    """Return ``{pkg: version}`` for the QGIS-bundled builds of the packages in
    ``_BUNDLED_PIN_PACKAGES``.

    The versions are probed by running the pip interpreter with the *user*
    site-packages disabled (``-s``), so the numbers reflect the build QGIS
    ships even when a shadowing copy is already present in the user site.
    The result is cached for the life of the process; failures degrade to an
    empty dict (installs then proceed unpinned rather than breaking).
    """
    global _bundled_versions_cache
    if _bundled_versions_cache is not None:
        return _bundled_versions_cache

    python = _find_python()
    # importlib.metadata avoids importing (and paying the import cost of) the
    # packages themselves; -s drops the user site so we see the bundled dist.
    probe = (
        'import importlib.metadata as m\n'
        'for n in {names!r}:\n'
        '    try: print(n + "==" + m.version(n))\n'
        '    except Exception: pass\n'
    ).format(names=list(_BUNDLED_PIN_PACKAGES))

    versions = {}
    try:
        kwargs = dict(capture_output=True, text=True, timeout=30)
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run([python, '-s', '-c', probe], **kwargs)  # nosec B603
        for line in result.stdout.splitlines():
            line = line.strip()
            if '==' in line:
                name, ver = line.split('==', 1)
                if name in _BUNDLED_PIN_PACKAGES and ver:
                    versions[name] = ver
    except Exception as exc:  # nosec B110
        if log_fn:
            log_fn(f'Could not detect QGIS-bundled package versions ({exc}); '
                   f'installing without version pins.')

    _bundled_versions_cache = versions
    return versions


def _write_constraints_file(log_fn):
    """Write a pip constraints file pinning the QGIS-bundled ABI-sensitive
    packages to their current versions.

    Returns the file path, or ``None`` when nothing could be detected (in which
    case the caller installs unpinned).
    """
    versions = _detect_bundled_versions(log_fn)
    if not versions:
        return None

    import tempfile
    path = os.path.join(tempfile.gettempdir(), 'bambi_pip_constraints.txt')
    try:
        with open(path, 'w') as f:
            for name, ver in versions.items():
                f.write(f'{name}=={ver}\n')
    except Exception as exc:  # nosec B110
        if log_fn:
            log_fn(f'Could not write constraints file ({exc}); '
                   f'installing without version pins.')
        return None
    return path


def _run_pip(args, log_fn):
    python = _find_python()
    args = list(args)

    # Pin the ABI-sensitive bundled packages (numpy/scipy) on every install so a
    # transitive dependency cannot upgrade them into the user site and shadow
    # QGIS's own build.  Skipped for a custom --index-url (e.g. the CUDA torch
    # index), which may not host the pinned versions.
    if args and args[0] == 'install' and '--index-url' not in args:
        constraints = _write_constraints_file(log_fn)
        if constraints:
            args += ['-c', constraints]
            log_fn(f'Pinning bundled packages to QGIS versions via {constraints}')

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
