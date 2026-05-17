# -*- coding: utf-8 -*-
"""
QGIS-plugin shim for the BAMBI thermal parser.

All parsing logic lives in the ``bambi.thermal`` package (alfs_detection).
This module's sole responsibility is discovering the DJI SDK and exiftool
inside the ``bambi_deps/`` folder of the active QGIS profile and wiring
those paths into the framework's ``Thermal`` class.

The ``Thermal`` subclass exported here is a drop-in replacement for
direct instantiation — callers do not need to know about the path discovery.

Required layout inside the QGIS profile directory
(e.g. AppData/Roaming/QGIS/QGIS3/profiles/default/bambi_deps/):
  bambi_deps/
      exiftool-12.35.exe    (optional – enables camera-model detection)
      dji_thermal_sdk_vX.Y_YYYYMMDD/   (any version folder)
          utility/
              bin/
                  windows/
                      release_x64/
                          libdirp.dll
                          libv_dirp.dll
                          libv_iirp.dll
"""

import os
import platform
from typing import Optional

from bambi.thermal.thermal_parser import (  # noqa: F401 – re-exported
    Thermal as _BaseThermal,
    parse_dji_rjpeg,
    apply_colormap,
    _read_dji_xmp_params as _read_meta,
)

__all__ = ['Thermal', 'parse_dji_rjpeg', 'apply_colormap', 'read_thermal_meta']


def read_thermal_meta(filepath: str) -> dict:
    """Return DJI thermal measurement parameters for *filepath*.

    Keys present when available: ``reflected_apparent_temperature``,
    ``object_distance``, ``emissivity``, ``relative_humidity``,
    ``image_width``, ``image_height``.  Returns an empty dict on failure.
    """
    return _read_meta(filepath)


# ---------------------------------------------------------------------------
# Plugin-local path discovery
# ---------------------------------------------------------------------------

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Store downloaded/extracted dependencies outside the plugin installation
# directory so that locked DLLs (e.g. DJI SDK) never block plugin
# reinstallation.  The QGIS profile directory persists across plugin updates.


def _get_bambi_deps_dir() -> str:
    try:
        from qgis.core import QgsApplication
        return os.path.join(QgsApplication.qgisSettingsDirPath(), 'bambi_deps')
    except Exception:
        return os.path.join(_PLUGIN_DIR, 'plugins')


_PLUGINS_DIR = _get_bambi_deps_dir()


def _find_sdk_dir() -> Optional[str]:
    """Return the DLL/SO directory of the newest dji_thermal_sdk_* install."""
    system = platform.system()
    architecture = 'x64' if platform.architecture()[0] == '64bit' else 'x86'
    extension = 'so' if system == 'Linux' else 'dll'

    if not os.path.isdir(_PLUGINS_DIR):
        return None

    candidates = sorted(
        (
            entry for entry in os.listdir(_PLUGINS_DIR)
            if entry.startswith('dji_thermal_sdk_') and os.path.isdir(os.path.join(_PLUGINS_DIR, entry))
        ),
        reverse=True,
    )
    for sdk in candidates:
        sdk_dir = os.path.join(
            _PLUGINS_DIR, sdk, 'utility', 'bin', system.lower(), f'release_{architecture}'
        )
        if os.path.isfile(os.path.join(sdk_dir, f'libdirp.{extension}')):
            return sdk_dir
    return None


def _find_exiftool() -> Optional[str]:
    """Return the path to exiftool if present inside the plugins folder."""
    name = 'exiftool' if platform.system() == 'Linux' else 'exiftool-12.35.exe'
    path = os.path.join(_PLUGINS_DIR, name)
    return path if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Drop-in Thermal subclass with auto-discovered paths
# ---------------------------------------------------------------------------

class Thermal(_BaseThermal):
    """Thermal parser with paths auto-discovered from the QGIS plugin directory.

    Instantiate exactly like the base class but without supplying ``sdk_dir``
    or ``exiftool_path`` — both are found automatically from the ``plugins/``
    subfolder that ships alongside this file.
    """

    def __init__(self, dtype=None):
        import numpy as np
        if dtype is None:
            dtype = np.float32
        super().__init__(
            dtype=dtype,
            sdk_dir=_find_sdk_dir(),
            exiftool_path=_find_exiftool(),
        )
