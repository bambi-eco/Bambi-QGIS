# -*- coding: utf-8 -*-
"""
Thermal image parser for DJI and FLIR radiometric JPEGs.

Based on SanNianYiSi/thermal_parser (MIT licence).
Adapted for use as a QGIS plugin module:
  - sys.exit() replaced with proper exceptions so QGIS cannot be killed
  - assert replaced with RuntimeError / ValueError
  - DLL / exiftool paths resolved relative to this file's directory
    (place the 'plugins' subfolder alongside this file)

Required layout inside the plugin directory:
  bambi_wildlife_detection/
      bambi_thermal.py          ← this file
      plugins/
          exiftool-12.35.exe    (optional – enables camera-model detection)
          dji_thermal_sdk_vX.Y_YYYYMMDD/   (any version folder matching dji_thermal_sdk_*)
              utility/
                  bin/
                      windows/
                          release_x64/
                              libdirp.dll
                              libv_dirp.dll
                              libv_iirp.dll
"""

import os
import re
import platform
import subprocess
from ctypes import (
    CDLL, POINTER, Structure, cast, create_string_buffer, pointer, sizeof,
    c_float, c_int, c_int16, c_int32, c_uint8, c_uint32, c_void_p,
)
from io import BufferedIOBase, BytesIO
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

__all__ = ['Thermal', 'parse_dji_rjpeg']

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_PLUGINS_DIR = os.path.join(_PLUGIN_DIR, 'plugins')


def _find_sdk_dir() -> Optional[str]:
    """
    Return the DLL directory for the newest dji_thermal_sdk_* installation
    found under _PLUGINS_DIR, or None if none is present.

    Scans for any folder matching ``dji_thermal_sdk_*`` and picks the
    lexicographically largest name (newest version when DJI's naming
    convention ``dji_thermal_sdk_vX.Y_YYYYMMDD`` is followed).
    """
    system = platform.system()
    architecture = 'x64' if platform.architecture()[0] == '64bit' else 'x86'
    extension = 'so' if system == 'Linux' else 'dll'

    if not os.path.isdir(_PLUGINS_DIR):
        return None

    candidates = sorted(
        (
            entry for entry in os.listdir(_PLUGINS_DIR)
            if entry.startswith('dji_thermal_sdk_')
            and os.path.isdir(os.path.join(_PLUGINS_DIR, entry))
        ),
        reverse=True,  # newest version first by lexicographic order
    )

    for sdk in candidates:
        sdk_dir = os.path.join(
            _PLUGINS_DIR, sdk, 'utility', 'bin', system.lower(), f'release_{architecture}'
        )
        libdirp = os.path.join(sdk_dir, f'libdirp.{extension}')
        if os.path.isfile(libdirp):
            return sdk_dir

    return None


def _get_sdk_paths() -> Tuple[str, str, str, str]:
    """Return (libdirp, libv_dirp, libv_iirp, exiftool) absolute paths."""
    system = platform.system()
    architecture = 'x64' if platform.architecture()[0] == '64bit' else 'x86'
    extension = 'so' if system == 'Linux' else 'dll'
    exiftool_name = 'exiftool' if system == 'Linux' else 'exiftool-12.35.exe'

    if system not in ('Windows', 'Linux') or architecture not in ('x64', 'x86'):
        raise NotImplementedError(
            f'Platform not supported: {system} {architecture}'
        )

    sdk_dir = _find_sdk_dir() or os.path.join(
        _PLUGINS_DIR, 'dji_thermal_sdk', 'utility', 'bin',
        system.lower(), f'release_{architecture}',
    )
    return (
        os.path.join(sdk_dir, f'libdirp.{extension}'),
        os.path.join(sdk_dir, f'libv_dirp.{extension}'),
        os.path.join(sdk_dir, f'libv_iirp.{extension}'),
        os.path.join(_PLUGINS_DIR, exiftool_name),
    )


# ---------------------------------------------------------------------------
# DJI SDK ctypes structs
# ---------------------------------------------------------------------------

DIRP_HANDLE = c_void_p
DIRP_VERBOSE_LEVEL_NONE = 0


class dirp_rjpeg_version_t(Structure):
    _fields_ = [('rjpeg', c_uint32), ('header', c_uint32), ('curve', c_uint32)]


class dirp_resolotion_t(Structure):
    _fields_ = [('width', c_uint32), ('height', c_uint32)]


class dirp_measurement_params_t(Structure):
    _fields_ = [
        ('distance', c_float),
        ('humidity', c_float),
        ('emissivity', c_float),
        ('reflection', c_float),
    ]


# ---------------------------------------------------------------------------
# FLIR APP1 parsing (pure Python, no external tools needed for raw data)
# ---------------------------------------------------------------------------

SEGMENT_SEP = b'\xff'
APP1_MARKER = b'\xe1'
MAGIC_FLIR_DEF = b'FLIR\x00'
CHUNK_APP1_BYTES_COUNT = len(APP1_MARKER)
CHUNK_LENGTH_BYTES_COUNT = 2
CHUNK_MAGIC_BYTES_COUNT = len(MAGIC_FLIR_DEF)
CHUNK_SKIP_BYTES_COUNT = 1
CHUNK_NUM_BYTES_COUNT = 1
CHUNK_TOT_BYTES_COUNT = 1
CHUNK_METADATA_LENGTH = (
    CHUNK_APP1_BYTES_COUNT + CHUNK_LENGTH_BYTES_COUNT + CHUNK_MAGIC_BYTES_COUNT
    + CHUNK_SKIP_BYTES_COUNT + CHUNK_NUM_BYTES_COUNT + CHUNK_TOT_BYTES_COUNT
)


def _unpack_flir(path_or_stream: Union[str, BinaryIO]) -> np.ndarray:
    if isinstance(path_or_stream, str) and os.path.isfile(path_or_stream):
        with open(path_or_stream, 'rb') as f:
            return _unpack_flir(f)
    stream = path_or_stream
    flir_app1 = _extract_flir_app1(stream)
    records = _parse_flir_app1(flir_app1)
    return _parse_thermal(flir_app1, records)


def _extract_flir_app1(stream: BinaryIO) -> BinaryIO:
    stream.read(2)
    chunks_count: Optional[int] = None
    chunks: Dict[int, bytes] = {}
    while True:
        b = stream.read(1)
        if b == b'':
            break
        if b != SEGMENT_SEP:
            continue
        parsed = _parse_flir_chunk(stream, chunks_count)
        if not parsed:
            continue
        chunks_count, chunk_num, chunk = parsed
        if chunks.get(chunk_num) is not None:
            raise ValueError('Invalid FLIR: duplicate chunk number')
        chunks[chunk_num] = chunk
        if chunk_num == chunks_count:
            break
    if chunks_count is None:
        raise ValueError('Invalid FLIR: no metadata encountered')
    data = b''.join(chunks[i] for i in range(chunks_count + 1))
    s = BytesIO(data)
    s.seek(0)
    return s


def _parse_flir_chunk(
    stream: BinaryIO, chunks_count: Optional[int]
) -> Optional[Tuple[int, int, bytes]]:
    marker = stream.read(CHUNK_APP1_BYTES_COUNT)
    length_bytes = stream.read(CHUNK_LENGTH_BYTES_COUNT)
    length = int.from_bytes(length_bytes, 'big') - CHUNK_METADATA_LENGTH
    magic = stream.read(CHUNK_MAGIC_BYTES_COUNT)
    if not (marker == APP1_MARKER and magic == MAGIC_FLIR_DEF):
        stream.seek(-len(marker) - len(length_bytes) - len(magic), 1)
        return None
    stream.seek(1, 1)
    chunk_num = int.from_bytes(stream.read(CHUNK_NUM_BYTES_COUNT), 'big')
    chunks_tot = int.from_bytes(stream.read(CHUNK_TOT_BYTES_COUNT), 'big')
    if chunks_count is None:
        chunks_count = chunks_tot
    if chunk_num < 0 or chunk_num > chunks_tot or chunks_tot != chunks_count:
        raise ValueError(f'Invalid FLIR: inconsistent chunk count ({chunks_tot})')
    return chunks_tot, chunk_num, stream.read(length + 1)


def _parse_thermal(stream: BinaryIO, records: dict) -> np.ndarray:
    _, _, raw = _parse_raw_data(stream, records[1])
    return raw


def _parse_flir_app1(stream: BinaryIO) -> dict:
    stream.read(4)
    stream.seek(16, 1)
    stream.read(4)
    record_dir_offset = int.from_bytes(stream.read(4), 'big')
    record_dir_count = int.from_bytes(stream.read(4), 'big')
    stream.seek(28, 1)
    stream.read(4)
    stream.seek(record_dir_offset)
    stream.read(32 * record_dir_count)
    details: dict = {}
    for nr in range(record_dir_count):
        d = _parse_flir_record_metadata(stream, nr)
        if d:
            details[d[1]] = d
    return details


def _parse_flir_record_metadata(stream: BinaryIO, record_nr: int):
    stream.seek(32 * record_nr)
    rtype = int.from_bytes(stream.read(2), 'big')
    if rtype < 1:
        return None
    stream.read(2); stream.read(4); stream.read(4)
    offset = int.from_bytes(stream.read(4), 'big')
    length = int.from_bytes(stream.read(4), 'big')
    stream.read(4); stream.read(4); stream.read(4)
    return 32 * record_nr, rtype, offset, length


def _parse_raw_data(stream: BinaryIO, metadata: tuple):
    _, _, offset, length = metadata
    stream.seek(offset)
    stream.seek(2, 1)
    width = int.from_bytes(stream.read(2), 'little')
    height = int.from_bytes(stream.read(2), 'little')
    stream.seek(offset + 32)
    raw_bytes = stream.read(length)
    img = Image.open(BytesIO(raw_bytes))
    arr = np.array(img)
    if arr.shape != (height, width):
        raise ValueError(
            f'FLIR shape mismatch: metadata {(height, width)} vs data {arr.shape}'
        )
    fix = np.vectorize(lambda x: (x >> 8) + ((x & 0x00FF) << 8))
    return width, height, fix(arr)


# ---------------------------------------------------------------------------
# Pure-Python DJI R-JPEG extraction (no SDK required)
# ---------------------------------------------------------------------------

def _dji_block_score(arr: np.ndarray) -> float:
    """
    Score a uint16 array as potential DJI raw thermal data (Kelvin × 64).

    For −100 °C … 500 °C, raw values fall in ≈ [11 082, 49 482].
    Returns a value in [0.0, 1.0].  Penalises extreme outliers (values
    near 0 or near 65 535) which appear when garbage bytes are included
    at block boundaries.
    """
    in_range = float(np.mean((arr >= 11_000) & (arr <= 50_000)))
    if in_range < 0.90:
        return 0.0
    # Extreme outliers indicate reading across a boundary into non-thermal bytes
    extreme = float(np.mean((arr < 500) | (arr > 64_500)))
    return in_range * max(0.0, 1.0 - extreme * 200.0)


# Candidate trailing-byte counts to try, ordered by likelihood.
# The DJI H20T R-JPEG appends a 512-byte footer after the thermal block;
# other models may differ — we score every candidate and pick the best.
_DJI_TRAILING_CANDIDATES = [512, 0, 256, 128, 768, 64, 1024, 32, 2048]


def parse_dji_rjpeg(filepath: str, dtype=np.float32) -> np.ndarray:
    """
    Extract temperature data from a DJI R-JPEG without the DJI SDK.

    DJI stores raw uint16 sensor values (Kelvin × 64) as a contiguous block
    at the END of the R-JPEG file, immediately before an optional proprietary
    footer.  Conversion formula:
        T [°C] = raw_uint16 / 64.0 − 273.15

    The JPEG EOF byte sequence (\\xFF\\xD9) can appear coincidentally inside
    the raw thermal payload, so we never search for it to locate the boundary.
    Instead we try every entry in _DJI_TRAILING_CANDIDATES as the number of
    bytes that follow the thermal block, score each candidate block with
    _dji_block_score(), and return the best match.

    Key insight (DJI H20T): the file ends with a 512-byte footer.  Reading
    data[-expected_bytes:] therefore starts 512 bytes too late, causing a
    visible 256-pixel (60 %) horizontal rotation.  data[-expected_bytes-512:-512]
    gives the correct block.

    Supported cameras: M30T, H20T/ZH20T, M3T, M3TD, M4T, H30T, M2EA, H20N.
    For FLIR-based cameras use Thermal.parse_flir() instead.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    height, width = _jpeg_dimensions(data)
    n_pixels = height * width
    expected_bytes = n_pixels * 2

    if len(data) < expected_bytes:
        raise ValueError(
            f'File ({len(data)} B) is too small for a {width}×{height} '
            f'thermal payload ({expected_bytes} B).'
        )

    best_score = 0.0
    best_raw: Optional[np.ndarray] = None

    for trailing in _DJI_TRAILING_CANDIDATES:
        start = len(data) - expected_bytes - trailing
        if start < 0:
            continue
        block = np.frombuffer(data[start:start + expected_bytes], dtype=np.uint16)
        score = _dji_block_score(block)
        if score > best_score:
            best_score = score
            best_raw = block

    if best_raw is None or best_score == 0.0:
        raise ValueError(
            f"Could not locate the thermal payload in "
            f"'{os.path.basename(filepath)}'.\n"
            "\n"
            "The DJI H20T/ZH20T uses a proprietary R-JPEG format that "
            "requires the official DJI Thermal SDK for reliable decoding.\n"
            "\n"
            "To enable full SDK support:\n"
            "  1. Download 'DJI Thermal SDK v1.7' from the DJI developer portal.\n"
            "  2. Extract the Windows x64 DLLs to:\n"
            f"     {_PLUGINS_DIR}\\dji_thermal_sdk_vX.Y_YYYYMMDD\\utility\\bin\\windows\\release_x64\n"
            "  3. Reload the plugin — it detects the DLLs automatically."
        )

    temp = best_raw.astype(np.float32) / 64.0 - 273.15
    arr2d = temp.reshape(height, width)
    if dtype == np.int16:
        return (arr2d * 10).astype(np.int16)
    return arr2d.astype(dtype)


def _jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    """Return (height, width) by scanning JPEG SOF markers."""
    i = 0
    while i < len(data) - 8:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        # SOF0, SOF1, SOF2, SOF3, SOF5-SOF15 carry image dimensions
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = int.from_bytes(data[i + 5:i + 7], 'big')
            w = int.from_bytes(data[i + 7:i + 9], 'big')
            return h, w
        # Skip over other markers
        if marker == 0xD8:
            i += 2
        elif marker in (0xD9, 0xDA):
            break
        else:
            seg_len = int.from_bytes(data[i + 2:i + 4], 'big')
            i += 2 + seg_len
    raise ValueError('Could not determine JPEG image dimensions')


# ---------------------------------------------------------------------------
# ABSOLUTE_ZERO constant
# ---------------------------------------------------------------------------

ABSOLUTE_ZERO = 273.15


# ---------------------------------------------------------------------------
# Main Thermal class
# ---------------------------------------------------------------------------

class Thermal:
    """Parse DJI and FLIR radiometric JPEG files into temperature arrays."""

    DJI_XT2 = 'XT2'
    DJI_ZH20T = 'ZH20T'
    DJI_XTS = 'XT S'
    DJI_XTR = 'FLIR'
    FLIR_B60 = 'Flir b60'
    FLIR_E40 = 'FLIR E40'
    FLIR_T640 = 'FLIR T640'
    FLIR = 'FLIR'
    FLIR_DEFAULT = '*'
    FLIR_AX8 = 'FLIR AX8'
    DJI_M2EA = 'MAVIC2-ENTERPRISE-ADVANCED'
    DJI_H20N = 'ZH20N'
    DJI_M3T = 'M3T'
    DJI_M3TD = 'M3TD'
    DJI_M30T = 'M30T'
    DJI_H30T = 'H30T'
    DJI_M4T = 'M4T'

    # dirp return codes
    DIRP_SUCCESS = 0

    def __init__(self, dtype=np.float32):
        if dtype.__name__ not in {np.float32.__name__, np.int16.__name__}:
            raise ValueError(f'dtype must be np.float32 or np.int16, got {dtype}')
        self._dtype = dtype
        self._dll_dirp = None
        self._dll_dirp_sub = None
        self._dll_iirp = None
        self._sdk_loaded = False

        (
            self._filepath_dirp,
            self._filepath_dirp_sub,
            self._filepath_iirp,
            self._filepath_exiftool,
        ) = _get_sdk_paths()

        # Try to load DJI SDK DLLs — failure is non-fatal; DJI cameras will
        # use the pure-Python fallback (parse_dji_rjpeg) instead.
        if (os.path.isfile(self._filepath_dirp)
                and os.path.isfile(self._filepath_dirp_sub)
                and os.path.isfile(self._filepath_iirp)):
            try:
                sdk_dir = os.path.dirname(self._filepath_dirp)
                # Add the SDK directory so Windows can resolve transitive DLL
                # dependencies (libv_girp.dll, MicroIA_Release_x64.dll, …).
                _dll_dir_cookie = None
                if hasattr(os, 'add_dll_directory'):
                    _dll_dir_cookie = os.add_dll_directory(sdk_dir)
                self._dll_dirp = CDLL(self._filepath_dirp)
                self._dll_dirp_sub = CDLL(self._filepath_dirp_sub)
                self._dll_iirp = CDLL(self._filepath_iirp)
                if _dll_dir_cookie is not None:
                    _dll_dir_cookie.close()
                self._sdk_loaded = True
                self._setup_sdk_functions()
            except OSError:
                self._sdk_loaded = False
                self._dll_dirp = None
                if _dll_dir_cookie is not None:
                    _dll_dir_cookie.close()

    def close(self):
        """Explicitly unload the SDK DLLs so the files are no longer locked."""
        if not self._sdk_loaded:
            return
        try:
            if platform.system() == 'Windows':
                import ctypes as _ct
                _free = _ct.windll.kernel32.FreeLibrary
                for dll in (self._dll_iirp, self._dll_dirp_sub, self._dll_dirp):
                    if dll is not None:
                        _free(dll._handle)
            else:
                import ctypes as _ct
                _dl = _ct.CDLL('libdl.so')
                for dll in (self._dll_iirp, self._dll_dirp_sub, self._dll_dirp):
                    if dll is not None:
                        _dl.dlclose(dll._handle)
        except Exception:
            pass
        self._dll_dirp = None
        self._dll_dirp_sub = None
        self._dll_iirp = None
        self._sdk_loaded = False

    def _setup_sdk_functions(self):
        dll = self._dll_dirp
        dll.dirp_set_verbose_level.argtypes = [c_int]
        dll.dirp_set_verbose_level(DIRP_VERBOSE_LEVEL_NONE)

        dll.dirp_create_from_rjpeg.argtypes = [POINTER(c_uint8), c_int32, POINTER(DIRP_HANDLE)]
        dll.dirp_create_from_rjpeg.restype = c_int32

        dll.dirp_destroy.argtypes = [DIRP_HANDLE]
        dll.dirp_destroy.restype = c_int32

        dll.dirp_get_rjpeg_version.argtypes = [DIRP_HANDLE, POINTER(dirp_rjpeg_version_t)]
        dll.dirp_get_rjpeg_version.restype = c_int32

        dll.dirp_get_rjpeg_resolution.argtypes = [DIRP_HANDLE, POINTER(dirp_resolotion_t)]
        dll.dirp_get_rjpeg_resolution.restype = c_int32

        dll.dirp_get_measurement_params.argtypes = [DIRP_HANDLE, POINTER(dirp_measurement_params_t)]
        dll.dirp_get_measurement_params.restype = c_int32

        dll.dirp_set_measurement_params.argtypes = [DIRP_HANDLE, POINTER(dirp_measurement_params_t)]
        dll.dirp_set_measurement_params.restype = c_int32

        dll.dirp_measure.argtypes = [DIRP_HANDLE, POINTER(c_int16), c_int32]
        dll.dirp_measure.restype = c_int32

        dll.dirp_measure_ex.argtypes = [DIRP_HANDLE, POINTER(c_float), c_int32]
        dll.dirp_measure_ex.restype = c_int32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, filepath_image: str) -> np.ndarray:
        """
        Parse a radiometric JPEG and return a 2-D temperature array (°C).

        Detects camera model via exiftool if available, or falls back to the
        pure-Python DJI R-JPEG extractor for modern DJI cameras.
        """
        if not isinstance(filepath_image, str) or not os.path.exists(filepath_image):
            raise FileNotFoundError(f'File not found: {filepath_image}')

        # Try to get camera model from exiftool
        camera_model = self._get_camera_model(filepath_image)

        _flir_models = {
            self.FLIR, self.FLIR_DEFAULT, self.FLIR_T640,
            self.FLIR_E40, self.FLIR_B60, self.FLIR_AX8,
            self.DJI_XT2, self.DJI_XTR,
        }
        _dji_sdk_models = {
            self.DJI_ZH20T, self.DJI_XTS,
            self.DJI_M2EA, self.DJI_H20N,
            self.DJI_M3T, self.DJI_M3TD, self.DJI_M30T,
            self.DJI_H30T, self.DJI_M4T,
        }

        if camera_model and (camera_model in _flir_models or self.FLIR in camera_model):
            kwargs = self._extract_flir_params(filepath_image)
            return self.parse_flir(filepath_image, **kwargs)

        if camera_model and camera_model in _dji_sdk_models:
            if self._sdk_loaded:
                kwargs = self._extract_dji_params(filepath_image, camera_model)
                return self.parse_dirp2(filepath_image, **kwargs)
            else:
                return parse_dji_rjpeg(filepath_image, dtype=self._dtype)

        # Unknown / undetectable model (e.g. exiftool absent): try SDK first.
        # dirp_create_from_rjpeg validates the R-JPEG signature internally and
        # returns a non-zero code for non-DJI files, making it safe to call
        # speculatively.  DJI H20T/M3T/M3TD store thermal data in a proprietary
        # compressed format — the pure-Python fallback CANNOT decode them, so
        # the SDK is the only viable path for these cameras.
        if self._sdk_loaded:
            # Try m2ea_mode=True first (skips dirp_set_measurement_params, safer
            # for M3T/M3TD/M30T), then m2ea_mode=False (sets params, needed for
            # H20T/ZH20T when exiftool is absent).
            for _m2ea in (True, False):
                try:
                    return self.parse_dirp2(filepath_image, m2ea_mode=_m2ea)
                except Exception:
                    pass

        raise ValueError(
            f"Cannot decode '{os.path.basename(filepath_image)}'.\n\n"
            "DJI thermal cameras (H20T, M3T, M3TD, M30T, …) store temperature\n"
            "data in a proprietary compressed format that requires the DJI\n"
            "Thermal SDK.  The SDK DLLs were not found or failed to parse this\n"
            "file.\n\n"
            "Ensure a DJI Thermal SDK folder (dji_thermal_sdk_*) is installed at:\n"
            f"  {_PLUGINS_DIR}\n\n"
            "then restart QGIS."
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _run_exiftool(self, filepath: str) -> Optional[Dict[str, str]]:
        if not os.path.isfile(self._filepath_exiftool):
            return None
        try:
            result = subprocess.run(
                [self._filepath_exiftool, filepath],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15,
            )
            text = result.stdout.decode('utf-8', errors='replace').replace('\r', '')
            return {
                field.split(':', 1)[0].strip(): field.split(':', 1)[1].strip()
                for field in text.split('\n')
                if ':' in field
            }
        except Exception:
            return None

    def _get_camera_model(self, filepath: str) -> Optional[str]:
        meta = self._run_exiftool(filepath)
        if meta is None:
            return None
        return meta.get('Camera Model Name')

    def _extract_flir_params(self, filepath: str) -> dict:
        meta = self._run_exiftool(filepath) or {}
        kwargs = {}
        for name, key in [
            ('emissivity', 'Emissivity'),
            ('ir_window_transmission', 'IR Window Transmission'),
            ('planck_r1', 'Planck R1'),
            ('planck_b', 'Planck B'),
            ('planck_f', 'Planck F'),
            ('planck_o', 'Planck O'),
            ('planck_r2', 'Planck R2'),
            ('ata1', 'Atmospheric Trans Alpha 1'),
            ('ata2', 'Atmospheric Trans Alpha 2'),
            ('atb1', 'Atmospheric Trans Beta 1'),
            ('atb2', 'Atmospheric Trans Beta 2'),
            ('atx', 'Atmospheric Trans X'),
        ]:
            if key in meta:
                try:
                    kwargs[name] = float(meta[key])
                except ValueError:
                    pass
        for name, key in [
            ('object_distance', 'Object Distance'),
            ('atmospheric_temperature', 'Atmospheric Temperature'),
            ('reflected_apparent_temperature', 'Reflected Apparent Temperature'),
            ('ir_window_temperature', 'IR Window Temperature'),
            ('relative_humidity', 'Relative Humidity'),
        ]:
            if key in meta:
                try:
                    kwargs[name] = float(meta[key][:-2])
                except (ValueError, IndexError):
                    pass
        return kwargs

    def _extract_dji_params(self, filepath: str, camera_model: str) -> dict:
        meta = self._run_exiftool(filepath) or {}
        kwargs = {}
        for name, key in [
            ('object_distance', 'Object Distance'),
            ('relative_humidity', 'Relative Humidity'),
            ('emissivity', 'Emissivity'),
            ('reflected_apparent_temperature', 'Reflected Temperature'),
        ]:
            if key in meta:
                nums = re.findall(r'\d+\.\d+|\d+', meta[key])
                if nums:
                    kwargs[name] = float(nums[0])
        if camera_model != self.DJI_M30T:
            try:
                kwargs['image_height'] = int(meta.get('Image Height', 512))
                kwargs['image_width'] = int(meta.get('Image Width', 640))
            except (ValueError, TypeError):
                pass
        if 'emissivity' in kwargs:
            kwargs['emissivity'] /= 100
        if camera_model in {
            self.DJI_M2EA, self.DJI_H20N, self.DJI_M3T, self.DJI_M3TD,
            self.DJI_M30T, self.DJI_H30T, self.DJI_M4T,
        }:
            kwargs['m2ea_mode'] = True
        return kwargs

    # ------------------------------------------------------------------
    # FLIR Planck conversion
    # ------------------------------------------------------------------

    def parse_flir(
        self,
        filepath_image: str,
        emissivity: float = 1.0,
        object_distance: float = 1.0,
        atmospheric_temperature: float = 20.0,
        reflected_apparent_temperature: float = 20.0,
        ir_window_temperature: float = 20.0,
        ir_window_transmission: float = 1.0,
        relative_humidity: float = 50.0,
        planck_r1: float = 21106.77,
        planck_b: float = 1501.0,
        planck_f: float = 1.0,
        planck_o: float = -7340.0,
        planck_r2: float = 0.012545258,
        ata1: float = 0.006569,
        ata2: float = 0.01262,
        atb1: float = -0.002276,
        atb2: float = -0.00667,
        atx: float = 1.9,
    ) -> np.ndarray:
        thermal_bytes = subprocess.check_output([
            self._filepath_exiftool, '-RawThermalImage', '-b', filepath_image
        ])
        stream = BytesIO(thermal_bytes)
        img = Image.open(stream)
        if img.format == 'TIFF':
            raw = np.array(img)
        elif img.format == 'PNG':
            raw = _unpack_flir(filepath_image)
        else:
            raise ValueError(f'Unexpected raw thermal image format: {img.format}')

        emiss_wind = 1 - ir_window_transmission
        refl_wind = 0.0
        h2o = (relative_humidity / 100) * np.exp(
            1.5587 + 0.06939 * atmospheric_temperature
            - 0.00027816 * atmospheric_temperature ** 2
            + 0.00000068455 * atmospheric_temperature ** 3
        )
        tau1 = atx * np.exp(-np.sqrt(object_distance / 2) * (ata1 + atb1 * np.sqrt(h2o))) + (
            1 - atx
        ) * np.exp(-np.sqrt(object_distance / 2) * (ata2 + atb2 * np.sqrt(h2o)))
        tau2 = tau1

        def _rad(T):
            return planck_r1 / (planck_r2 * (np.exp(planck_b / (T + ABSOLUTE_ZERO)) - planck_f)) - planck_o

        raw_refl1_attn = (1 - emissivity) / emissivity * _rad(reflected_apparent_temperature)
        raw_atm1_attn = (1 - tau1) / emissivity / tau1 * _rad(atmospheric_temperature)
        raw_wind_attn = emiss_wind / emissivity / tau1 / ir_window_transmission * _rad(ir_window_temperature)
        raw_refl2_attn = refl_wind / emissivity / tau1 / ir_window_transmission * _rad(reflected_apparent_temperature)
        raw_atm2_attn = (1 - tau2) / emissivity / tau1 / ir_window_transmission / tau2 * _rad(atmospheric_temperature)

        raw_obj = (
            raw / emissivity / tau1 / ir_window_transmission / tau2
            - raw_atm1_attn - raw_atm2_attn - raw_wind_attn
            - raw_refl1_attn - raw_refl2_attn
        )
        val = planck_r1 / (planck_r2 * (raw_obj + planck_o)) + planck_f
        if np.any(val <= 0):
            raise ValueError(f'Planck formula produced non-positive values — file may be corrupt: {filepath_image}')
        temperature = planck_b / np.log(val) - ABSOLUTE_ZERO
        return temperature.astype(self._dtype)

    # ------------------------------------------------------------------
    # DJI SDK path (requires libdirp.dll etc.)
    # ------------------------------------------------------------------

    def parse_dirp2(
        self,
        filepath_image: str,
        image_height: int = 512,
        image_width: int = 640,
        object_distance: float = 5.0,
        relative_humidity: float = 70.0,
        emissivity: float = 1.0,
        reflected_apparent_temperature: float = 23.0,
        m2ea_mode: bool = False,
    ) -> np.ndarray:
        if not self._sdk_loaded:
            raise RuntimeError(
                f'DJI SDK DLLs are not loaded. '
                f'Place a dji_thermal_sdk_* folder inside: {_PLUGINS_DIR}'
            )
        with open(filepath_image, 'rb') as f:
            raw_bytes = f.read()
        # create_string_buffer produces a ctypes array; cast() requires a
        # ctypes object — a plain Python bytes would raise TypeError.
        raw_buf = create_string_buffer(raw_bytes, len(raw_bytes))
        raw_size = c_int32(len(raw_bytes))
        raw_ptr = cast(raw_buf, POINTER(c_uint8))
        handle = DIRP_HANDLE()
        dll = self._dll_dirp

        ret = dll.dirp_create_from_rjpeg(raw_ptr, raw_size, pointer(handle))
        if ret != self.DIRP_SUCCESS:
            raise RuntimeError(f'dirp_create_from_rjpeg failed: {ret}')

        ver = dirp_rjpeg_version_t()
        res = dirp_resolotion_t()
        if dll.dirp_get_rjpeg_version(handle, pointer(ver)) != self.DIRP_SUCCESS:
            raise RuntimeError('dirp_get_rjpeg_version failed')
        if dll.dirp_get_rjpeg_resolution(handle, pointer(res)) != self.DIRP_SUCCESS:
            raise RuntimeError('dirp_get_rjpeg_resolution failed')
        # Use the SDK-reported dimensions; caller-supplied values are kept as
        # a fallback in case the SDK returns zeros (should not happen).
        sdk_w = int(res.width) or image_width
        sdk_h = int(res.height) or image_height

        if not m2ea_mode:
            params = dirp_measurement_params_t()
            if dll.dirp_get_measurement_params(handle, pointer(params)) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_get_measurement_params failed')
            params.distance = float(object_distance)
            params.humidity = float(relative_humidity)
            params.emissivity = float(emissivity)
            params.reflection = float(reflected_apparent_temperature)
            if dll.dirp_set_measurement_params(handle, pointer(params)) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_set_measurement_params failed')

        n = sdk_w * sdk_h
        if self._dtype.__name__ == np.float32.__name__:
            data = np.zeros(n, dtype=np.float32)
            ptr = data.ctypes.data_as(POINTER(c_float))
            if dll.dirp_measure_ex(handle, ptr, c_int32(n * sizeof(c_float))) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_measure_ex failed')
            temp = data.reshape(sdk_h, sdk_w)
        else:
            data = np.zeros(n, dtype=np.int16)
            ptr = data.ctypes.data_as(POINTER(c_int16))
            if dll.dirp_measure(handle, ptr, c_int32(n * sizeof(c_int16))) != self.DIRP_SUCCESS:
                raise RuntimeError('dirp_measure failed')
            temp = data.reshape(sdk_h, sdk_w) / 10.0

        if dll.dirp_destroy(handle) != self.DIRP_SUCCESS:
            raise RuntimeError('dirp_destroy failed')

        return np.array(temp, dtype=self._dtype)
