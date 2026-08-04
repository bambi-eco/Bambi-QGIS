# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Processing Module
=============================================

This module contains the processing logic for all pipeline steps.
"""

import csv
import os
import json
from typing import Dict, Any, Optional, List, Tuple

from qgis.PyQt.QtCore import QObject, pyqtSignal

from contextlib import contextmanager
from typing import Callable

# Progress callback receives a single int: percentage 0-100
ProgressCallback = Callable[[int], None]


@contextmanager
def patch_frame_extraction_progress(
        progress_fn: Optional[ProgressCallback],
        total_frames: int,
        log_fn: Optional[Callable[[str], None]] = None,
        log_interval_percent: int = 10,
        cancel_check: Optional[Callable[[], bool]] = None,
):
    """
    Context manager that patches frame extraction to report progress.

    Usage:
        total = count_srt_frames(srt_paths)
        with patch_frame_extraction_progress(progress_fn, total, log_fn, cancel_check=is_cancelled):
            extractor.extract(...)

    :param progress_fn: Function receiving percentage (0-100 int)
    :param total_frames: Total number of frames to extract
    :param log_fn: Optional logging function
    :param log_interval_percent: Log every N percent (default 10)
    :param cancel_check: Optional function that returns True if cancelled
    """
    import bambi.webgl.timed_pose_extractor as tpe_module

    # Get reference to the callback class
    CallbackClass = tpe_module.TimedFrameExtractorCallback

    # Store original __call__
    original_call = CallbackClass.__call__

    # Progress tracking state
    state = {'processed': 0, 'last_logged_percent': -1}

    def patched_call(self, idx, img):
        # Check for cancellation
        if cancel_check is not None and cancel_check():
            raise CancelledException("Frame extraction cancelled")

        # Call original
        result = original_call(self, idx, img)

        # Track progress on success
        if result and total_frames > 0:
            state['processed'] += 1
            current = state['processed']
            percent = int((current / total_frames) * 100)

            # Report percentage via callback
            if progress_fn is not None:
                progress_fn(percent)

            # Log periodically
            if log_fn is not None and percent >= state['last_logged_percent'] + log_interval_percent:
                state['last_logged_percent'] = percent
                log_fn(f"Extracting frames: {current} / {total_frames} ({percent}%)")

        return result

    try:
        # Patch
        CallbackClass.__call__ = patched_call
        yield
    finally:
        # Restore
        CallbackClass.__call__ = original_call


def count_srt_frames(srt_paths: List[str]) -> int:
    """Count total frames in SRT files."""
    from bambi.srt.srt_parser import SrtParser
    parser = SrtParser()
    return sum(len(parser.parse(p)) for p in srt_paths)


def _make_curve_colorizer(parse_fn, colormap: str, curve_cfg: Dict[str, Any]):
    """Build a ThermalColorizer that tone-maps via a curve instead of lo/hi clipping.

    The curve (see ``core.thermal_curve.ThermalCurve``) normalizes the
    temperatures into its own domain and shapes them through the control-
    point curve; out-of-domain pixels clamp to the curve's endpoint values
    rather than being rendered black. Only ``_apply_colormap`` is replaced —
    parsing and BGR conversion stay in the framework class.
    """
    import numpy as np
    from bambi.thermal.thermal_colorizer import ThermalColorizer
    from .core.thermal_curve import ThermalCurve

    class _CurveThermalColorizer(ThermalColorizer):
        def __init__(self, parse_fn, colormap, curve):
            super().__init__(parse_fn=parse_fn, colormap=colormap)
            self.curve = curve

        def _apply_colormap(self, temp):
            norm = self.curve.apply(np.asarray(temp, dtype=np.float32))
            rgba = self._get_cmap()(norm)               # H×W×4 float64
            return (rgba[:, :, :3] * 255).astype(np.uint8)

    return _CurveThermalColorizer(
        parse_fn=parse_fn,
        colormap=colormap,
        curve=ThermalCurve.from_dict(curve_cfg),
    )


class ProcessingWorker(QObject):
    """Worker class for background processing."""

    finished = pyqtSignal(str, bool)  # step, success
    error = pyqtSignal(str, str)  # step, error_message
    progress = pyqtSignal(int)  # percentage
    log = pyqtSignal(str)  # log message

    def __init__(self, processor: 'BambiProcessor', step: str, config: Dict[str, Any]):
        super().__init__()
        self.processor = processor
        self.step = step
        self.config = config
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the current processing step."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def run(self):
        """Execute the processing step."""
        try:
            if self.step == "extract_thermal_frames":
                self.processor.extract_thermal_frames(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "extract_rgb_frames":
                self.processor.extract_rgb_frames(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "flight_route":
                self.processor.run_flight_route(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "detection":
                self.processor.run_detection(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "georeference":
                self.processor.run_georeference(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "calculate_fov":
                self.processor.run_calculate_fov(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "tracking":
                self.processor.run_tracking(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "alfs":
                self.processor.run_alfs(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "export_geotiffs":
                self.processor.run_export_geotiffs(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "orthomosaic":
                self.processor.run_orthomosaic(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "sam3_segmentation":
                self.processor.run_sam3_segmentation(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "sam3_georeference":
                self.processor.run_sam3_georeference(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "perpendicular":
                self.processor.run_perpendicular(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "track_perpendicular":
                self.processor.run_track_perpendicular(
                    self.config, self.progress.emit, self.log.emit, self.is_cancelled
                )
            elif self.step == "trex_import":
                self.processor.run_trex_import(self.config, self.progress.emit, self.log.emit, self.is_cancelled)
            elif self.step == "density_heatmap":
                self.processor.run_density_heatmap(
                    self.config, self.progress.emit, self.log.emit, self.is_cancelled
                )
            elif self.step == "coverage_map":
                self.processor.run_coverage_map(
                    self.config, self.progress.emit, self.log.emit, self.is_cancelled
                )
            elif self.step == "distance_sampling":
                self.processor.run_distance_sampling(
                    self.config, self.progress.emit, self.log.emit, self.is_cancelled
                )
            elif self.step == "population_estimation":
                self.processor.run_population_estimation(
                    self.config, self.progress.emit, self.log.emit, self.is_cancelled
                )
            else:
                raise ValueError(f"Unknown step: {self.step}")

            # Check if cancelled before signaling success
            if self._cancelled:
                self.log.emit(f"{self.step} was cancelled")
                self.finished.emit(self.step, False)
            else:
                self.finished.emit(self.step, True)

        except CancelledException:
            self.log.emit(f"{self.step} was cancelled")
            self.finished.emit(self.step, False)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(self.step, f"{str(e)}\n\n{tb}")
            self.finished.emit(self.step, False)


class CancelledException(Exception):
    """Exception raised when a processing step is cancelled."""
    pass


class BambiProcessor:
    """Main processing class for BAMBI wildlife detection pipeline."""

    # Default detection models per modality: HuggingFace repo, path of the
    # weights file inside the repo, and the local filename used in the shared
    # bambi_deps/models/ cache.
    DEFAULT_MODELS = {
        "T": {
            "repo": "cpraschl/bambi-thermal-detection",
            "remote_path": "thermal_animal_detector.pt",
            "filename": "thermal_animal_detector.pt",
        },
        "W": {
            "repo": "cpraschl/bambi-models",
            "remote_path": "rgb/weights/best.pt",
            "filename": "rgb_animal_detector.pt",
        },
    }

    @staticmethod
    def _get_default_model_dir() -> str:
        """Return the shared directory used to cache the default detection model.

        Uses the QGIS profile's ``bambi_deps/models/`` folder so the model is
        downloaded once and shared across all projects, rather than being
        re-downloaded into every project's output folder.
        """
        try:
            from qgis.core import QgsApplication
            return os.path.join(
                QgsApplication.qgisSettingsDirPath(), 'bambi_deps', 'models'
            )
        except Exception:
            import tempfile
            return os.path.join(tempfile.gettempdir(), 'bambi_deps', 'models')

    def __init__(self):
        """Initialize the processor."""
        pass

    @staticmethod
    def generate_flat_surface_mesh(
        lat: float,
        lon: float,
        flat_surface_msl: float,
        extent_m: float,
        output_glb_path: str,
        output_json_path: str,
        epsg: int = 0,
    ):
        """Generate a flat horizontal GLB mesh for use as a DEM.

        The mesh is centred at (lat, lon), spans ±extent_m in local X and Y, and sits
        at local z = 0 (the origin altitude equals flat_surface_msl, so cameras at GPS
        altitude H m MSL have local z = H − flat_surface_msl).

        epsg should match the plugin's target CRS (target_epsg in config) so the JSON
        origin is in the same projected CRS as the poses file.  When 0, the UTM zone is
        auto-detected from the centroid longitude.

        The companion JSON uses the same format as a regular DEM JSON so the file can
        be loaded without any special handling.

        Returns (output_glb_path, output_json_path).
        """
        import gltflib as gl
        from pyproj import Transformer
        import numpy as np

        # Use caller-supplied EPSG (must match target_epsg in plugin config so that
        # the JSON origin is in the same CRS as poses_*.json).  Fall back to
        # auto-detecting the UTM zone only when no EPSG is given.
        if not epsg:
            utm_zone = int((lon + 180) / 6) + 1
            epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        x_utm, y_utm = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        ).transform(lon, lat)

        # Flat quad at local z = 0, spanning ±extent_m in X and Y
        e = float(extent_m)
        vertices = np.array([
            [-e, -e, 0.0],
            [e, -e, 0.0],
            [e, e, 0.0],
            [-e, e, 0.0],
        ], dtype=np.float32)
        uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        vb = vertices.tobytes()
        ub = uvs.tobytes()
        ib = indices.flatten().tobytes()

        def _align4(n):
            return (4 - n % 4) % 4

        u_off = len(vb) + _align4(len(vb))
        i_off = u_off + len(ub) + _align4(len(ub))
        total = i_off + len(ib) + _align4(len(ib))
        buf = bytearray(total)
        buf[0:len(vb)] = vb
        buf[u_off:u_off + len(ub)] = ub
        buf[i_off:i_off + len(ib)] = ib

        model = gl.GLTFModel(
            asset=gl.Asset(version='2.0'), scene=0,
            scenes=[gl.Scene(nodes=[0])],
            nodes=[gl.Node(mesh=0)],
            meshes=[gl.Mesh(primitives=[gl.Primitive(
                attributes=gl.Attributes(POSITION=0, TEXCOORD_0=1), indices=2,
            )])],
            bufferViews=[
                gl.BufferView(buffer=0, byteOffset=0, byteLength=len(vb)),
                gl.BufferView(buffer=0, byteOffset=u_off, byteLength=len(ub)),
                gl.BufferView(buffer=0, byteOffset=i_off, byteLength=len(ib)),
            ],
            accessors=[
                gl.Accessor(bufferView=0,
                            componentType=gl.ComponentType.FLOAT.value, count=4,
                            type=gl.AccessorType.VEC3.value,
                            min=vertices.min(axis=0).tolist(),
                            max=vertices.max(axis=0).tolist()),
                gl.Accessor(bufferView=1,
                            componentType=gl.ComponentType.FLOAT.value, count=4,
                            type=gl.AccessorType.VEC2.value),
                gl.Accessor(bufferView=2,
                            componentType=gl.ComponentType.UNSIGNED_INT.value, count=6,
                            type=gl.AccessorType.SCALAR.value),
            ],
            buffers=[gl.Buffer(byteLength=total)],
        )
        gl.GLTF(model=model, resources=[gl.GLBResource(data=bytes(buf))]).export_glb(output_glb_path)

        dem_meta = {
            "origin": [x_utm, y_utm, float(flat_surface_msl)],
            "origin_wgs84": {
                "latitude": lat,
                "longitude": lon,
                "altitude": float(flat_surface_msl),
            },
            "crs": f"EPSG:{epsg}",
        }
        with open(output_json_path, 'w') as _f:
            json.dump(dem_meta, _f, indent=2)

        # GeoJSON polygon covering the mesh extent (for debugging in QGIS / any GIS tool)
        inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        corners_utm = [
            (x_utm - e, y_utm - e),
            (x_utm + e, y_utm - e),
            (x_utm + e, y_utm + e),
            (x_utm - e, y_utm + e),
        ]
        ring = [[round(lo, 8), round(la, 8)]
                for lo, la in (inv.transform(cx, cy) for cx, cy in corners_utm)]
        ring.append(ring[0])  # close the ring
        geojson_path = output_json_path.replace(".json", ".geojson")
        with open(geojson_path, 'w') as _f:
            json.dump({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "flat_surface_msl": float(flat_surface_msl),
                        "extent_m": float(extent_m),
                        "crs": f"EPSG:{epsg}",
                    },
                }],
            }, _f, indent=2)

        return output_glb_path, output_json_path

    @staticmethod
    def get_correction_for_frame(frame_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the appropriate correction factors for a given frame index.

        Single source: :func:`core.corrections.correction_for_frame_config`.
        """
        from .core.corrections import correction_for_frame_config
        return correction_for_frame_config(frame_idx, config)

    def download_default_model(self, log_fn=None, camera: str = "T") -> str:
        """Download the default detection model from HuggingFace to the shared
        ``bambi_deps/models/`` folder in the QGIS profile directory.

        :param log_fn: Optional logging function
        :param camera: Modality to fetch the model for ("T" thermal, "W" RGB)
        :return: Path to the downloaded model
        """
        import urllib.parse
        import requests

        model_spec = self.DEFAULT_MODELS.get(camera, self.DEFAULT_MODELS["T"])

        models_folder = self._get_default_model_dir()
        os.makedirs(models_folder, exist_ok=True)

        model_path = os.path.join(models_folder, model_spec["filename"])

        if os.path.exists(model_path):
            # Check if file is not empty (incomplete download)
            if os.path.getsize(model_path) > 1000000:  # > 1MB
                if log_fn:
                    log_fn(f"Model already exists at {model_path}")
                return model_path
            else:
                # Remove incomplete file
                os.remove(model_path)

        if log_fn:
            log_fn(f"Downloading model from HuggingFace: {model_spec['repo']}")
            log_fn("This may take a few minutes...")

        # Direct download URL from HuggingFace
        url = f"https://huggingface.co/{model_spec['repo']}/resolve/main/{model_spec['remote_path']}"

        try:
            # Validate URL scheme before downloading
            _scheme = urllib.parse.urlparse(url).scheme
            if _scheme not in ('https', 'http'):
                raise ValueError(f"Unexpected URL scheme: {_scheme!r}")

            def _write_stream(resp):
                total_size = int(resp.headers.get('content-length', 0))
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if log_fn and downloaded % 819200 == 0:
                                if total_size > 0:
                                    percent = min(100, (downloaded / total_size) * 100)
                                    log_fn(f"Download progress: {percent:.1f}% ({downloaded // 1024} KB)")

            # Try with SSL verification disabled first (for corporate networks)
            try:
                resp = requests.get(url, stream=True, verify=True, timeout=300)
                resp.raise_for_status()
                _write_stream(resp)
            except Exception as e1:
                if log_fn:
                    log_fn(f"Download with custom SSL failed: {e1}")
                    log_fn("Trying standard download...")
                resp = requests.get(url, stream=True, timeout=300)
                resp.raise_for_status()
                _write_stream(resp)

            # Verify download
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
                if log_fn:
                    log_fn(f"Model downloaded successfully to: {model_path}")
                return model_path
            else:
                raise RuntimeError("Downloaded file is too small or missing")

        except Exception as e:
            # Clean up partial download
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except Exception:  # nosec B110
                    pass

            error_msg = (
                f"Failed to download model automatically.\n\n"
                f"Please download manually:\n"
                f"1. Go to: https://huggingface.co/{model_spec['repo']}\n"
                f"2. Download: {model_spec['remote_path']}\n"
                f"   (rename it to: {model_spec['filename']})\n"
                f"3. Save to: {model_path}\n\n"
                f"Or specify a local model path in the Parameters tab.\n\n"
                f"Error: {str(e)}"
            )
            raise RuntimeError(error_msg)

    def _get_extraction_prerequisites(self, config: Dict[str, Any]):
        """Get common prerequisites for frame extraction.

        :param config: Configuration dictionary
        :return: Tuple of (target_folder, rel_transformer, ad_origin)
        """
        from pyproj import CRS, Transformer
        from bambi.airdata.air_data_frame import AirDataFrame

        target_folder = config["target_folder"]
        target_epsg = config.get("target_epsg", 32633)
        path_to_dem_json = config.get("alfs_dem_metadata_path")

        # Create target folder
        os.makedirs(target_folder, exist_ok=True)

        # Setup coordinate transformer
        input_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_epsg(target_epsg)

        rel_transformer = Transformer.from_crs(input_crs, target_crs)
        with open(path_to_dem_json, "r") as f:
            dem_json = json.load(f)

        origin_wgs84 = dem_json.get("origin_wgs84", {})
        origin_list = dem_json.get("origin", [0, 0, 0])

        origin_lat = float(origin_wgs84.get("latitude", 0.0))
        origin_lon = float(origin_wgs84.get("longitude", 0.0))
        origin_alt = float(origin_wgs84.get("altitude", origin_list[2] if len(origin_list) > 2 else 0.0))

        # Sanity check: if the stored origin_wgs84 latitude is essentially zero but the
        # DEM origin field contains a non-equatorial latitude, the origin_wgs84 was
        # likely computed incorrectly (e.g. WGS84 degree values were fed into a UTM→WGS84
        # transformer).  Fall back to interpreting the origin field as [longitude, latitude,
        # altitude] in WGS84, which is how the DEM downloader stores the raw bounds corner.
        #
        # The fallback must only fire when the origin field really *is* in degrees.  For
        # survey areas near the equator a correct origin_wgs84 latitude is legitimately
        # close to zero while origin holds projected UTM metres (northings up to 10⁷) —
        # feeding those into the WGS84→UTM transformer yields inf and every camera
        # position becomes -Infinity.  Require plausible lon/lat magnitudes first.
        _origin_looks_like_degrees = len(origin_list) >= 2 and all([
            abs(float(origin_list[0])) <= 180.0,
            abs(float(origin_list[1])) <= 90.0,
        ])
        if abs(origin_lat) < 0.01 and _origin_looks_like_degrees and abs(float(origin_list[1])) > 0.01:
            origin_lon = float(origin_list[0])
            origin_lat = float(origin_list[1])
            origin_alt = float(origin_list[2]) if len(origin_list) > 2 else 0.0

        if not (-90.0 <= origin_lat <= 90.0 and -180.0 <= origin_lon <= 180.0):
            raise RuntimeError(
                f"DEM origin resolves to an invalid WGS84 coordinate "
                f"(lat={origin_lat}, lon={origin_lon}).  Camera positions cannot be "
                f"computed relative to it.\n\n"
                f"Check the 'origin_wgs84' / 'origin' fields in "
                f"{os.path.basename(path_to_dem_json)} — regenerating the DEM mesh "
                f"(Load/Generate DEM) usually repairs them."
            )

        ad_origin = AirDataFrame()
        ad_origin.latitude = origin_lat
        ad_origin.longitude = origin_lon
        ad_origin.altitude = origin_alt

        return target_folder, rel_transformer, ad_origin

    def extract_thermal_frames(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Extract frames from thermal drone videos or photos.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        :param cancel_check: Optional function that returns True if cancelled
        """
        if log_fn:
            log_fn("=" * 50)
            log_fn("Extracting THERMAL frames...")
            log_fn("=" * 50)

        # Check for cancellation before starting
        if cancel_check and cancel_check():
            raise CancelledException("Cancelled before starting")

        target_folder, rel_transformer, ad_origin = self._get_extraction_prerequisites(config)
        airdata_path = config.get("airdata_path", "")

        frames_folder_t = os.path.join(target_folder, "frames_t")
        os.makedirs(frames_folder_t, exist_ok=True)
        target_poses_t = os.path.join(target_folder, "poses_t.json")

        if config.get("input_mode") == "photo":
            from bambi.webgl.photo_pose_extractor import UniqueMatchPhotoPoseExtractor
            from bambi.thermal.thermal_colorizer import ThermalColorizer

            calibration_res = config.get("thermal_photo_calibration_data")
            if calibration_res is None:
                calib_path = config.get("thermal_photo_calibration_path", "")
                if not calib_path:
                    raise ValueError(
                        "No thermal photo calibration provided. "
                        "Please select a preset or specify a calibration file path."
                    )
                with open(calib_path) as f:
                    calibration_res = json.load(f)
            if calibration_res is None:
                raise ValueError("Thermal photo calibration data is None. Check the calibration file or preset.")

            colormap = config.get("thermal_photo_colormap")
            lo_threshold = config.get("thermal_photo_lo_threshold")
            hi_threshold = config.get("thermal_photo_hi_threshold")
            curve_cfg = config.get("thermal_photo_curve")
            thermal_colorizer = None
            thermal_instance = None

            if any(v is not None for v in
                   (colormap, lo_threshold, hi_threshold, curve_cfg)):
                import numpy as np
                from .bambi_thermal import Thermal
                thermal_instance = Thermal(dtype=np.float32)
                if curve_cfg is not None:
                    thermal_colorizer = _make_curve_colorizer(
                        parse_fn=thermal_instance.parse,
                        colormap=colormap or "white-hotspot",
                        curve_cfg=curve_cfg,
                    )
                    if log_fn:
                        log_fn(
                            f"Thermal curve mapping "
                            f"({thermal_colorizer.curve.describe()}) with "
                            f"colormap '{colormap or 'white-hotspot'}' will "
                            f"be applied during frame extraction."
                        )
                else:
                    thermal_colorizer = ThermalColorizer(
                        parse_fn=thermal_instance.parse,
                        colormap=colormap or "white-hotspot",
                        lo_threshold=lo_threshold,
                        hi_threshold=hi_threshold,
                    )
                    if log_fn:
                        log_fn(
                            f"Thermal colormap '{colormap or 'white-hotspot'}' will be applied "
                            f"during frame extraction."
                        )

            # Build extension patterns once (shared by EXIF fallback + extractor)
            _ext_suffixes = (".JPG", ".jpg", ".jpeg", ".JPEG", ".tiff", ".TIFF", ".png", ".PNG")
            if config.get("thermal_photo_filter"):
                photo_extensions_t = tuple(
                    f"{p}{e}" for p in ("*_T_*", "*_T") for e in _ext_suffixes
                )
                if log_fn:
                    log_fn("Thermal filter active: only images with _T_ or _T in filename")
            else:
                photo_extensions_t = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG",
                                      "*.tiff", "*.TIFF", "*.png", "*.PNG")

            # EXIF fallback: reconstruct AirData CSV when no flight log is available
            _use_ordered_t = False
            if not airdata_path or not os.path.exists(airdata_path):
                if log_fn:
                    log_fn("No AirData file – reconstructing flight log from image EXIF…")
                import glob as _glob
                from bambi.airdata.air_data_from_exif_parser import write_airdata_csv_from_exif
                _img_paths = sorted(set(
                    p for ext in photo_extensions_t
                    for p in _glob.glob(os.path.join(config["thermal_photo_dir"], ext))
                ))
                _synthetic_csv = os.path.join(target_folder, "airdata_from_exif_t.csv")
                try:
                    _n = write_airdata_csv_from_exif(_img_paths, _synthetic_csv)
                except Exception as _exc:
                    raise RuntimeError(
                        f"No AirData file and EXIF fallback failed: {_exc}"
                    ) from _exc
                airdata_path = _synthetic_csv
                _use_ordered_t = True
                if log_fn:
                    log_fn(f"Reconstructed {_n} AirData row(s) from EXIF.")

            try:
                if _use_ordered_t:
                    from bambi.webgl.photo_pose_extractor import OrderedPhotoPoseExtractor
                    extractor = OrderedPhotoPoseExtractor(
                        rel_transformer=rel_transformer,
                        calibration_res=calibration_res,
                        thermal_colorizer=thermal_colorizer,
                        use_gimbal_heading=config.get("use_gimbal_heading", False),
                    )
                else:
                    extractor = UniqueMatchPhotoPoseExtractor(
                        rel_transformer=rel_transformer,
                        calibration_res=calibration_res,
                        thermal_colorizer=thermal_colorizer,
                        use_gimbal_heading=config.get("use_gimbal_heading", False),
                    )
                if log_fn:
                    log_fn(f"Photo mode: processing images in {config['thermal_photo_dir']}")
                    # Monkey-patch per-image progress logging onto this instance
                    # only — the class method is never touched.
                    _orig_undistort = extractor._undistort_and_save
                    _img_counter = [0]

                    def _logged_undistort(source_path, filename, output_dir):
                        _img_counter[0] += 1
                        log_fn(f"  [{_img_counter[0]}] {filename}")
                        return _orig_undistort(source_path, filename, output_dir)

                    extractor._undistort_and_save = _logged_undistort

                extractor.extract(
                    photo_dir=config["thermal_photo_dir"],
                    airdata_csv=airdata_path,
                    output_path=target_poses_t,
                    output_image_dir=frames_folder_t,
                    photo_timezone_offset_hours=config.get("timezone_offset_hours", 1.0),
                    origin=ad_origin,
                    skip=config.get("extract_skip", 0),
                    limit=config.get("extract_limit"),
                    extensions=photo_extensions_t,
                    no_images=config.get("no_images", False),
                )
            finally:
                if thermal_instance is not None:
                    thermal_instance.close()
        else:
            from bambi.video.calibrated_video_frame_accessor import CalibratedVideoFrameAccessor
            from bambi.webgl.timed_pose_extractor import TimedPoseExtractor
            from bambi.domain.camera import Camera
            from dateutil import tz as _dateutil_tz

            thermal_video_paths = config["thermal_video_paths"]
            thermal_srt_paths = config["thermal_srt_paths"]

            _tz_offset_hours = config.get("timezone_offset_hours", 1.0)
            _video_tz = _dateutil_tz.tzoffset(None, int(_tz_offset_hours * 3600))

            calibration_res = config.get("thermal_calibration_data")
            if calibration_res is None:
                with open(config["thermal_calibration_path"]) as f:
                    calibration_res = json.load(f)

            accessor = CalibratedVideoFrameAccessor(
                calibration_res,
                preserve_aspect_ratio=config.get("preserve_aspect_ratio", False))
            extractor = TimedPoseExtractor(
                accessor,
                rel_transformer=rel_transformer,
                camera_name=Camera.from_string("T"),
                use_gimbal_heading=config.get("use_gimbal_heading", False),
            )

            total_frames = count_srt_frames(thermal_srt_paths)
            if log_fn:
                log_fn(f"Processing {total_frames} frames...")

            with patch_frame_extraction_progress(progress_fn, total_frames, log_fn, cancel_check=cancel_check):
                extractor.extract(
                    frames_folder_t, airdata_path, thermal_video_paths, thermal_srt_paths,
                    origin=ad_origin, include_gps=True,
                    skip=config.get("extract_skip", 0),
                    limit=config.get("extract_limit"),
                    sampling_rate=config.get("extract_sampling_rate") or 0,
                    timezone=_video_tz,
                    no_images=config.get("no_images", False),
                )

        # Move poses.json written into frames_folder to target folder with suffix
        poses_in_frames = os.path.join(frames_folder_t, "poses.json")
        if os.path.exists(poses_in_frames):
            if os.path.exists(target_poses_t):
                os.remove(target_poses_t)
            os.rename(poses_in_frames, target_poses_t)

        if os.path.exists(os.path.join(frames_folder_t, "mask_T.png")):
            mask_path = os.path.join(frames_folder_t, "mask_T.png")
        elif os.path.exists(os.path.join(frames_folder_t, "mask.png")):
            mask_path = os.path.join(frames_folder_t, "mask.png")
        else:
            mask_path = None

        if mask_path is not None:
            target_mask_w = os.path.join(target_folder, "mask_T.png")
            if os.path.exists(target_mask_w):
                os.remove(target_mask_w)
            os.rename(mask_path, target_mask_w)

        # Log frame count from poses file
        if os.path.exists(target_poses_t):
            with open(target_poses_t, 'r') as f:
                poses_t = json.load(f)
            thermal_frames = len(poses_t.get("images", []))
            if log_fn:
                log_fn(f"Thermal extraction complete: {thermal_frames} frames")
                log_fn("Frames saved to: frames_t/")

    def extract_rgb_frames(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Extract frames from RGB drone videos or photos.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        :param cancel_check: Optional function that returns True if cancelled
        """
        if log_fn:
            log_fn("=" * 50)
            log_fn("Extracting RGB frames...")
            log_fn("=" * 50)

        # Check for cancellation before starting
        if cancel_check and cancel_check():
            raise CancelledException("Cancelled before starting")

        target_folder, rel_transformer, ad_origin = self._get_extraction_prerequisites(config)
        airdata_path = config.get("airdata_path", "")

        frames_folder_w = os.path.join(target_folder, "frames_w")
        os.makedirs(frames_folder_w, exist_ok=True)
        target_poses_w = os.path.join(target_folder, "poses_w.json")

        if config.get("input_mode") == "photo":
            from bambi.webgl.photo_pose_extractor import UniqueMatchPhotoPoseExtractor

            calibration_res = config.get("rgb_photo_calibration_data")
            if calibration_res is None:
                calib_path = config.get("rgb_photo_calibration_path", "")
                if not calib_path:
                    raise ValueError(
                        "No RGB photo calibration provided. "
                        "Please select a preset or specify a calibration file path."
                    )
                with open(calib_path) as f:
                    calibration_res = json.load(f)
            if calibration_res is None:
                raise ValueError("RGB photo calibration data is None. Check the calibration file or preset.")

            # Build extension patterns once (shared by EXIF fallback + extractor)
            _ext_suffixes = (".JPG", ".jpg", ".jpeg", ".JPEG", ".tiff", ".TIFF", ".png", ".PNG")
            if config.get("rgb_photo_filter"):
                photo_extensions_w = tuple(
                    f"{p}{e}" for p in ("*_W_*", "*_W", "*_V_*", "*_V") for e in _ext_suffixes
                )
                if log_fn:
                    log_fn("RGB filter active: only images with _W_, _W, _V_ or _V in filename")
            else:
                photo_extensions_w = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG",
                                      "*.tiff", "*.TIFF", "*.png", "*.PNG")

            # EXIF fallback: reconstruct AirData CSV when no flight log is available
            _use_ordered_w = False
            if not airdata_path or not os.path.exists(airdata_path):
                if log_fn:
                    log_fn("No AirData file – reconstructing flight log from image EXIF…")
                import glob as _glob
                from bambi.airdata.air_data_from_exif_parser import write_airdata_csv_from_exif
                _img_paths = sorted(set(
                    p for ext in photo_extensions_w
                    for p in _glob.glob(os.path.join(config["rgb_photo_dir"], ext))
                ))
                _synthetic_csv = os.path.join(target_folder, "airdata_from_exif_w.csv")
                try:
                    _n = write_airdata_csv_from_exif(_img_paths, _synthetic_csv)
                except Exception as _exc:
                    raise RuntimeError(
                        f"No AirData file and EXIF fallback failed: {_exc}"
                    ) from _exc
                airdata_path = _synthetic_csv
                _use_ordered_w = True
                if log_fn:
                    log_fn(f"Reconstructed {_n} AirData row(s) from EXIF.")

            if _use_ordered_w:
                from bambi.webgl.photo_pose_extractor import OrderedPhotoPoseExtractor
                extractor = OrderedPhotoPoseExtractor(
                    rel_transformer=rel_transformer,
                    calibration_res=calibration_res,
                    use_gimbal_heading=config.get("use_gimbal_heading", False),
                )
            else:
                extractor = UniqueMatchPhotoPoseExtractor(
                    rel_transformer=rel_transformer,
                    calibration_res=calibration_res,
                    use_gimbal_heading=config.get("use_gimbal_heading", False),
                )
            if log_fn:
                log_fn(f"Photo mode: processing images in {config['rgb_photo_dir']}")
                # Monkey-patch per-image progress logging onto this instance
                # only — the class method is never touched.
                _orig_undistort = extractor._undistort_and_save
                _img_counter = [0]

                def _logged_undistort(source_path, filename, output_dir):
                    _img_counter[0] += 1
                    log_fn(f"  [{_img_counter[0]}] {filename}")
                    return _orig_undistort(source_path, filename, output_dir)

                extractor._undistort_and_save = _logged_undistort

            extractor.extract(
                photo_dir=config["rgb_photo_dir"],
                airdata_csv=airdata_path,
                output_path=target_poses_w,
                output_image_dir=frames_folder_w,
                photo_timezone_offset_hours=config.get("timezone_offset_hours", 1.0),
                origin=ad_origin,
                skip=config.get("extract_skip", 0),
                limit=config.get("extract_limit"),
                extensions=photo_extensions_w,
                no_images=config.get("no_images", False),
            )
        else:
            from bambi.video.calibrated_video_frame_accessor import CalibratedVideoFrameAccessor
            from bambi.webgl.timed_pose_extractor import TimedPoseExtractor
            from bambi.domain.camera import Camera
            from dateutil import tz as _dateutil_tz

            rgb_video_paths = config["rgb_video_paths"]
            rgb_srt_paths = config["rgb_srt_paths"]

            _tz_offset_hours = config.get("timezone_offset_hours", 1.0)
            _video_tz = _dateutil_tz.tzoffset(None, int(_tz_offset_hours * 3600))

            calibration_res = config.get("rgb_calibration_data")
            if calibration_res is None:
                with open(config["rgb_calibration_path"]) as f:
                    calibration_res = json.load(f)

            accessor = CalibratedVideoFrameAccessor(
                calibration_res,
                preserve_aspect_ratio=config.get("preserve_aspect_ratio", False))
            extractor = TimedPoseExtractor(
                accessor,
                rel_transformer=rel_transformer,
                camera_name=Camera.from_string("W"),
                use_gimbal_heading=config.get("use_gimbal_heading", False),
            )

            total_frames = count_srt_frames(rgb_srt_paths)
            if log_fn:
                log_fn(f"Processing {total_frames} frames...")

            with patch_frame_extraction_progress(progress_fn, total_frames, log_fn, cancel_check=cancel_check):
                extractor.extract(
                    frames_folder_w, airdata_path, rgb_video_paths, rgb_srt_paths,
                    origin=ad_origin, include_gps=True,
                    skip=config.get("extract_skip", 0),
                    limit=config.get("extract_limit"),
                    sampling_rate=config.get("extract_sampling_rate") or 0,
                    timezone=_video_tz,
                    no_images=config.get("no_images", False),
                )

        # Move poses.json written into frames_folder to target folder with suffix
        poses_in_frames = os.path.join(frames_folder_w, "poses.json")
        if os.path.exists(poses_in_frames):
            if os.path.exists(target_poses_w):
                os.remove(target_poses_w)
            os.rename(poses_in_frames, target_poses_w)

        if os.path.exists(os.path.join(frames_folder_w, "mask_W.png")):
            mask_path = os.path.join(frames_folder_w, "mask_W.png")
        elif os.path.exists(os.path.join(frames_folder_w, "mask.png")):
            mask_path = os.path.join(frames_folder_w, "mask.png")
        else:
            mask_path = None

        if mask_path is not None:
            target_mask_w = os.path.join(target_folder, "mask_W.png")
            if os.path.exists(target_mask_w):
                os.remove(target_mask_w)
            os.rename(mask_path, target_mask_w)

        # Log frame count from poses file
        if os.path.exists(target_poses_w):
            with open(target_poses_w, 'r') as f:
                poses_w = json.load(f)
            rgb_frames = len(poses_w.get("images", []))
            if log_fn:
                log_fn(f"RGB extraction complete: {rgb_frames} frames")
                log_fn("Frames saved to: frames_w/")

    def run_flight_route(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Generate flight route layers from AirData GPS and camera frame positions.

        Creates two GeoJSON files:
        - flight_route.geojson: LineString from AirData GPS log
        - camera_positions.geojson: Point markers at each extracted frame position

        The poses file is optional: without it only the AirData route line is
        generated (camera positions are skipped with a warning). At least one of
        the two sources (poses, AirData) must be available.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        camera = config.get("flight_route_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        if log_fn:
            log_fn(f"Generating flight route from AirData + {camera_name} poses...")

        target_folder = config["target_folder"]
        target_epsg = config.get("target_epsg", 32633)
        airdata_path = config.get("airdata_path", "")

        if progress_fn:
            progress_fn(10)

        # Load poses file for camera position markers (optional — without it
        # only the AirData route line is generated)
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        images = []
        if os.path.exists(poses_file):
            with open(poses_file, 'r') as f:
                poses = json.load(f)

            images = poses.get("images", [])
            if len(images) < 1:
                raise RuntimeError("Need at least 1 frame to create a flight route")

            if log_fn:
                log_fn(f"Found {len(images)} frame positions")
        else:
            if not (airdata_path and os.path.exists(airdata_path)):
                raise FileNotFoundError(
                    f"poses_{camera_suffix}.json not found at {poses_file} and no "
                    "AirData file is available — nothing to generate. Run 'Extract "
                    "Frames' or select an AirData CSV first."
                )
            if log_fn:
                log_fn(
                    f"Warning: poses_{camera_suffix}.json not found — camera "
                    "positions will not be generated, only the AirData route line. "
                    "Run 'Extract Frames' and re-run this step to add them."
                )

        if progress_fn:
            progress_fn(20)

        # Get coordinate offset from DEM metadata
        coord_offset_x = 0.0
        coord_offset_y = 0.0

        dem_metadata_path = config.get("alfs_dem_metadata_path")
        dem_path = config.get("dem_path")

        if dem_metadata_path and os.path.exists(dem_metadata_path):
            try:
                with open(dem_metadata_path, 'r') as f:
                    dem_metadata = json.load(f)
                origin = dem_metadata.get("origin", [0, 0, 0])
                coord_offset_x = float(origin[0])
                coord_offset_y = float(origin[1])
            except Exception:  # nosec B110
                pass
        elif dem_path:
            auto_metadata_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(auto_metadata_path):
                try:
                    with open(auto_metadata_path, 'r') as f:
                        dem_metadata = json.load(f)
                    origin = dem_metadata.get("origin", [0, 0, 0])
                    coord_offset_x = float(origin[0])
                    coord_offset_y = float(origin[1])
                except Exception:  # nosec B110
                    pass

        if coord_offset_x != 0 or coord_offset_y != 0:
            if abs(coord_offset_x) <= 180 and abs(coord_offset_y) <= 90:
                # DEM origin looks like WGS84 degrees — applying it as a UTM offset
                # would place camera positions in the wrong location.  Reset to zero so
                # poses are shown as-is (they are already in the target projected CRS).
                if log_fn:
                    log_fn(
                        f"Warning: DEM origin ({coord_offset_x:.5f}, {coord_offset_y:.5f}) "
                        "appears to be in WGS84 geographic coordinates, not UTM metres. "
                        "Ignoring offset to avoid misplaced camera positions. "
                        "Re-generate the DEM mesh to fix this."
                    )
                coord_offset_x = 0.0
                coord_offset_y = 0.0
            elif log_fn:
                log_fn(f"Using coordinate offset: X={coord_offset_x:.2f}, Y={coord_offset_y:.2f}")

        if progress_fn:
            progress_fn(30)

        # Build camera position point features (no connecting line)
        point_features = []
        for i, img_info in enumerate(images):
            location = img_info.get("location", [0, 0, 0])
            x = location[0] + coord_offset_x
            y = location[1] + coord_offset_y
            z = location[2]
            point_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [x, y, z]
                },
                "properties": {
                    "frame_idx": i,
                    "imagefile": img_info.get("imagefile", ""),
                    "altitude": z,
                    "timestamp": img_info.get("timestamp", "")
                }
            })

        if progress_fn:
            progress_fn(50)

        # Build flight route line from AirData GPS log
        route_coordinates = []
        if airdata_path and os.path.exists(airdata_path):
            try:
                from bambi.airdata.air_data_parser import AirDataParser
                from pyproj import Transformer, CRS as PyprojCRS

                # Use PROJ4 strings to avoid PROJ database version conflicts
                # (QGIS bundles an older proj.db that pyproj may pick up first)
                wgs84_proj4 = "+proj=longlat +datum=WGS84 +no_defs"
                if 32601 <= target_epsg <= 32660:
                    zone = target_epsg - 32600
                    utm_proj4 = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
                elif 32701 <= target_epsg <= 32760:
                    zone = target_epsg - 32700
                    utm_proj4 = f"+proj=utm +zone={zone} +south +datum=WGS84 +units=m +no_defs"
                else:
                    from bambi_wildlife_detection.austria_dem_downloader import get_proj4_for_crs, WGS84_PROJ4
                    wgs84_proj4 = WGS84_PROJ4
                    utm_proj4 = get_proj4_for_crs(f"EPSG:{target_epsg}")

                transformer = Transformer.from_crs(
                    PyprojCRS.from_proj4(wgs84_proj4),
                    PyprojCRS.from_proj4(utm_proj4),
                    always_xy=True
                )

                filter_gps_origin = config.get("filter_gps_origin", True)
                parser = AirDataParser()
                for frame in parser.parse_yield(airdata_path):
                    if frame.latitude is None or frame.longitude is None:
                        continue
                    if filter_gps_origin and frame.latitude == 0.0 and frame.longitude == 0.0:
                        continue
                    alt = frame.altitude_above_seaLevel if frame.altitude_above_seaLevel is not None else (
                        frame.altitude if frame.altitude is not None else 0.0
                    )
                    x, y = transformer.transform(frame.longitude, frame.latitude)
                    route_coordinates.append([x, y, float(alt)])

                if log_fn:
                    log_fn(f"Loaded {len(route_coordinates)} GPS positions from AirData")
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Could not load AirData route: {e}")
        else:
            if log_fn:
                log_fn("No AirData file specified, skipping flight route line")

        if progress_fn:
            progress_fn(80)

        # Create output folder (camera-specific)
        route_folder = os.path.join(target_folder, f"flight_route_{camera_suffix}")
        os.makedirs(route_folder, exist_ok=True)

        crs_block = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{target_epsg}"}
        }

        # Save flight route line (from AirData)
        if route_coordinates:
            if len(route_coordinates) == 1:
                route_geometry = {"type": "Point", "coordinates": route_coordinates[0]}
            else:
                route_geometry = {"type": "LineString", "coordinates": route_coordinates}

            flight_line_geojson = {
                "type": "FeatureCollection",
                "name": "flight_route",
                "crs": crs_block,
                "features": [
                    {
                        "type": "Feature",
                        "geometry": route_geometry,
                        "properties": {
                            "name": "Flight Route",
                            "total_gps_points": len(route_coordinates)
                        }
                    }
                ]
            }
            route_line_file = os.path.join(route_folder, "flight_route.geojson")
            with open(route_line_file, 'w', encoding='utf-8') as f:
                json.dump(flight_line_geojson, f, indent=2)
            if log_fn:
                log_fn(f"Flight route line saved to: {route_line_file}")

        # Save camera frame positions (points only, no connecting line)
        if point_features:
            camera_points_geojson = {
                "type": "FeatureCollection",
                "name": "camera_positions",
                "crs": crs_block,
                "features": point_features
            }
            camera_points_file = os.path.join(route_folder, "camera_positions.geojson")
            with open(camera_points_file, 'w', encoding='utf-8') as f:
                json.dump(camera_points_geojson, f, indent=2)

            if log_fn:
                log_fn(f"Camera positions saved to: {camera_points_file}")

        if not route_coordinates and not point_features:
            raise RuntimeError(
                "No flight route data could be generated — the AirData file "
                "contained no usable GPS positions and no poses are available."
            )

        if log_fn:
            log_fn("Flight route generation complete")

        if progress_fn:
            progress_fn(100)

    @staticmethod
    def _nearest_on_linestring(route_coords: list, cx: float, cy: float):
        """Find the nearest point on a LineString to (cx, cy).

        :param route_coords: List of [x, y, ...] coordinate lists
        :param cx: Query point X
        :param cy: Query point Y
        :return: (foot_x, foot_y, distance)
        """
        import math
        best_dist = float('inf')
        best_fx, best_fy = route_coords[0][0], route_coords[0][1]

        for j in range(len(route_coords) - 1):
            ax, ay = route_coords[j][0], route_coords[j][1]
            bx, by = route_coords[j + 1][0], route_coords[j + 1][1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                fx, fy = ax, ay
            else:
                t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq))
                fx, fy = ax + t * dx, ay + t * dy
            dist = math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_fx, best_fy = fx, fy

        return best_fx, best_fy, best_dist

    @staticmethod
    def _compute_frame_fov_polygon(
            image_metadata: dict,
            x_offset: float,
            y_offset: float,
            aspect_ratio: float = 4.0 / 3.0
    ) -> list:
        """Approximate ground footprint of a camera frame as a 4-corner polygon.

        Uses a flat-earth projection: the camera altitude is taken from the
        DEM-relative Z coordinate of the poses location, and the FOV rectangle
        is rotated by the camera's yaw (rotation[2]).

        :param image_metadata: One entry from poses["images"]
        :param x_offset: DEM origin X in the target CRS
        :param y_offset: DEM origin Y in the target CRS
        :param aspect_ratio: Image width / height (default 4/3)
        :return: List of 4 (x, y) world-coordinate tuples (convex quadrilateral)
        """
        import math

        location = image_metadata.get("location", [0, 0, 0])
        cam_x = location[0] + x_offset
        cam_y = location[1] + y_offset
        altitude = max(float(location[2]), 1.0)

        fovy = image_metadata.get("fovy", [50])
        if isinstance(fovy, list):
            fovy = fovy[0]
        fovy = float(fovy)

        yaw_deg = float(image_metadata.get("rotation", [0, 0, 0])[2]) % 360.0

        half_h = altitude * math.tan(math.radians(fovy / 2.0))
        fov_x = 2.0 * math.atan(aspect_ratio * math.tan(math.radians(fovy / 2.0)))
        half_w = altitude * math.tan(fov_x / 2.0)

        corners_local = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

        yaw_rad = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        return [
            (cam_x + cos_y * lx - sin_y * ly,
             cam_y + sin_y * lx + cos_y * ly)
            for lx, ly in corners_local
        ]

    @staticmethod
    def _point_in_polygon(px: float, py: float, poly: list) -> bool:
        """Ray-casting point-in-polygon test for a simple polygon."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (
                    px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
        """Return True if segment AB properly crosses segment CD."""
        def cross(ox, oy, qx, qy, rx, ry):
            return (qx - ox) * (ry - oy) - (qy - oy) * (rx - ox)

        d1 = cross(cx, cy, dx, dy, ax, ay)
        d2 = cross(cx, cy, dx, dy, bx, by)
        d3 = cross(ax, ay, bx, by, cx, cy)
        d4 = cross(ax, ay, bx, by, dx, dy)
        return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))

    @staticmethod
    def _segment_intersects_polygon(ax, ay, bx, by, poly: list) -> bool:
        """Return True if segment AB has any part inside or crossing *poly*."""
        if (BambiProcessor._point_in_polygon(ax, ay, poly) or BambiProcessor._point_in_polygon(bx, by, poly)):
            return True
        n = len(poly)
        for i in range(n):
            cx, cy = poly[i]
            dx, dy = poly[(i + 1) % n]
            if BambiProcessor._segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
        return False

    @staticmethod
    def _nearest_on_fov_linestring(route_coords: list, fov_poly: list,
                                   cx: float, cy: float,
                                   cam_x: float = None, cam_y: float = None):
        """Perpendicular foot on the flight-route segment associated with this frame.

        When the camera position (cam_x, cam_y) is provided the segment is
        chosen by finding whichever route segment lies nearest to the camera —
        i.e. the segment the drone was actually flying on.  This prevents
        snapping to a parallel transect whose FOV overlap or shorter distance
        might otherwise win.

        Without camera position, falls back to filtering by FOV polygon
        intersection (legacy behaviour).

        The foot is the true perpendicular projection (unclamped t) so the
        returned line is always at 90° to the route.

        :param route_coords: List of [x, y, ...] route coordinates
        :param fov_poly: FOV footprint as list of (x, y) tuples
        :param cx, cy: Detection centre in world CRS
        :param cam_x, cam_y: Camera (drone) position in world CRS for this frame
        :return: (foot_x, foot_y, distance)
        """
        import math

        # Primary path: use camera position to identify the correct transect
        # segment, avoiding snapping to parallel routes.
        if cam_x is not None and cam_y is not None:
            best_cam_dist = float('inf')
            best_j = 0
            for j in range(len(route_coords) - 1):
                ax, ay = route_coords[j][0], route_coords[j][1]
                bx, by = route_coords[j + 1][0], route_coords[j + 1][1]
                dx, dy = bx - ax, by - ay
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-12:
                    fx, fy = ax, ay
                else:
                    t = max(0.0, min(1.0, ((cam_x - ax) * dx + (cam_y - ay) * dy) / seg_len_sq))
                    fx, fy = ax + t * dx, ay + t * dy
                dist = math.sqrt((cam_x - fx) ** 2 + (cam_y - fy) ** 2)
                if dist < best_cam_dist:
                    best_cam_dist = dist
                    best_j = j
            # Unclamped perpendicular projection of detection onto the transect line
            ax, ay = route_coords[best_j][0], route_coords[best_j][1]
            bx, by = route_coords[best_j + 1][0], route_coords[best_j + 1][1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                return ax, ay, math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2)
            t = ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq
            fx, fy = ax + t * dx, ay + t * dy
            return fx, fy, math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)

        # Legacy fallback: filter by FOV polygon intersection
        visible = [
            j for j in range(len(route_coords) - 1)
            if BambiProcessor._segment_intersects_polygon(
                route_coords[j][0], route_coords[j][1],
                route_coords[j + 1][0], route_coords[j + 1][1],
                fov_poly)
        ]

        if not visible:
            return BambiProcessor._nearest_on_linestring(route_coords, cx, cy)

        # Best interior foot (t ∈ [0, 1]) and best unclamped foot as fallback.
        best_interior_dist = float('inf')
        best_interior_fx = best_interior_fy = None
        best_any_dist = float('inf')
        best_any_fx = route_coords[visible[0]][0]
        best_any_fy = route_coords[visible[0]][1]

        for j in visible:
            ax, ay = route_coords[j][0], route_coords[j][1]
            bx, by = route_coords[j + 1][0], route_coords[j + 1][1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                t, fx, fy = 0.0, ax, ay
            else:
                # Unclamped projection → always perpendicular to the segment
                t = ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq
                fx, fy = ax + t * dx, ay + t * dy
            dist = math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
            if dist < best_any_dist:
                best_any_dist = dist
                best_any_fx, best_any_fy = fx, fy
            if 0.0 <= t <= 1.0 and dist < best_interior_dist:
                best_interior_dist = dist
                best_interior_fx, best_interior_fy = fx, fy

        if best_interior_fx is not None:
            return best_interior_fx, best_interior_fy, best_interior_dist
        return best_any_fx, best_any_fy, best_any_dist

    def run_perpendicular(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Calculate perpendicular distances from detections to the flight route.

        For each georeferenced detection center, finds the nearest point on the
        AirData flight route (LineString) and records the perpendicular distance.
        Results are saved to flight_route_{fr}/perpendicular_{det}.json, suffixed
        with the detection camera the distances were computed for.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        :param cancel_check: Optional function that returns True if cancelled
        """
        target_folder = config["target_folder"]
        target_epsg = config.get("target_epsg", 32633)
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        det_camera = config.get("detection_camera", "T")
        det_suffix = "t" if det_camera == "T" else "w"

        if log_fn:
            log_fn("Calculating perpendicular distances to flight route...")

        if progress_fn:
            progress_fn(5)

        # Load flight route LineString (camera-specific folder)
        route_line_file = os.path.join(target_folder, f"flight_route_{fr_suffix}", "flight_route.geojson")
        if not os.path.exists(route_line_file):
            raise FileNotFoundError(
                "flight_route.geojson not found. Please run 'Generate Flight Route' first."
            )

        with open(route_line_file, 'r') as f:
            route_geojson = json.load(f)

        # Extract LineString coordinates
        route_coords = None
        for feature in route_geojson.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "LineString":
                route_coords = geom["coordinates"]
                break

        if not route_coords or len(route_coords) < 2:
            raise RuntimeError(
                "Flight route does not contain a valid LineString. "
                "Need at least 2 GPS points in the AirData file."
            )

        if log_fn:
            log_fn(f"Flight route loaded: {len(route_coords)} GPS points")

        if progress_fn:
            progress_fn(15)

        # Load DEM coordinate offset so poses locations can be converted to world CRS
        x_offset, y_offset = 0.0, 0.0
        dem_path = config.get("dem_path", "")
        if dem_path:
            dem_meta_path = config.get("alfs_dem_metadata_path") or \
                dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(dem_meta_path):
                try:
                    with open(dem_meta_path, 'r') as f:
                        dem_meta = json.load(f)
                    origin = dem_meta.get("origin", [0, 0, 0])
                    x_offset, y_offset = float(origin[0]), float(origin[1])
                except Exception:  # nosec B110
                    pass

        # Load poses for frame→image mapping AND camera FOV computation
        poses_images = []
        frame_to_image: Dict[int, str] = {}
        for suffix in ("t", "w"):
            poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
            if os.path.exists(poses_path):
                try:
                    with open(poses_path, 'r') as f:
                        poses = json.load(f)
                    for idx, img_info in enumerate(poses.get("images", [])):
                        imagefile = img_info.get("imagefile", "")
                        if imagefile:
                            frame_to_image[idx] = imagefile
                    if frame_to_image:
                        poses_images = poses.get("images", [])
                        if log_fn:
                            log_fn(f"Mapped {len(frame_to_image)} frames to image filenames "
                                   f"(poses_{suffix}.json)")
                        break
                except Exception as e:
                    if log_fn:
                        log_fn(f"Warning: could not read poses_{suffix}.json: {e}")

        if progress_fn:
            progress_fn(20)

        # Load georeferenced detections (camera-specific folder)
        georef_file = os.path.join(target_folder, f"georeferenced_{det_suffix}", "georeferenced.txt")
        if not os.path.exists(georef_file):
            raise FileNotFoundError(
                "georeferenced.txt not found. Please run 'Geo-Reference Detections' first."
            )

        detections = []
        with open(georef_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        detections.append({
                            'idx': int(parts[0]),
                            'frame': int(parts[1]),
                            'x1': float(parts[2]), 'y1': float(parts[3]), 'z1': float(parts[4]),
                            'x2': float(parts[5]), 'y2': float(parts[6]), 'z2': float(parts[7]),
                            'confidence': float(parts[8]),
                            'class_id': int(parts[9])
                        })
                    except (ValueError, IndexError):
                        continue

        if not detections:
            raise RuntimeError("No valid georeferenced detections found.")

        if log_fn:
            log_fn(f"Loaded {len(detections)} georeferenced detections")

        if progress_fn:
            progress_fn(40)

        # FOV polygon cache: computed once per unique frame, reused for all
        # detections in the same view.
        fov_cache: Dict[int, list] = {}
        cam_pos_cache: Dict[int, tuple] = {}

        # For each detection compute perpendicular foot and distance
        results = []
        for i, det in enumerate(detections):
            if cancel_check and cancel_check():
                raise CancelledException("Perpendicular calculation cancelled")

            cx = (det['x1'] + det['x2']) / 2.0
            cy = (det['y1'] + det['y2']) / 2.0
            cz = (det['z1'] + det['z2']) / 2.0

            frame_idx = det['frame']
            if frame_idx not in fov_cache:
                if poses_images and frame_idx < len(poses_images):
                    img_meta = poses_images[frame_idx]
                    fov_cache[frame_idx] = self._compute_frame_fov_polygon(
                        img_meta, x_offset, y_offset)
                    location = img_meta.get("location", [0, 0, 0])
                    cam_pos_cache[frame_idx] = (
                        float(location[0]) + x_offset,
                        float(location[1]) + y_offset
                    )
                else:
                    fov_cache[frame_idx] = None

            fov_poly = fov_cache[frame_idx]
            cam_pos = cam_pos_cache.get(frame_idx)
            if fov_poly and cam_pos:
                best_fx, best_fy, best_dist = self._nearest_on_fov_linestring(
                    route_coords, fov_poly, cx, cy, cam_pos[0], cam_pos[1])
            elif fov_poly:
                best_fx, best_fy, best_dist = self._nearest_on_fov_linestring(
                    route_coords, fov_poly, cx, cy)
            else:
                best_fx, best_fy, best_dist = self._nearest_on_linestring(
                    route_coords, cx, cy)

            results.append({
                'det_idx': det['idx'],
                'frame': frame_idx,
                'confidence': det['confidence'],
                'class_id': det['class_id'],
                'detection_center': [cx, cy, cz],
                'foot_point': [best_fx, best_fy],
                'distance_m': round(best_dist, 4)
            })

            if progress_fn and i % max(1, len(detections) // 10) == 0:
                progress_fn(40 + int((i / len(detections)) * 50))

        if log_fn:
            log_fn(f"FOV polygons computed for {len(fov_cache)} unique frames")

        if progress_fn:
            progress_fn(95)

        route_folder = os.path.join(target_folder, f"flight_route_{fr_suffix}")
        os.makedirs(route_folder, exist_ok=True)

        # Save flat perpendicular_{det}.json (used by "Add Perpendicular Lines
        # to QGIS"), suffixed with the detection camera.
        output = {
            'crs': f"EPSG:{target_epsg}",
            'total_detections': len(results),
            'perpendiculas': results
        }
        output_file = os.path.join(route_folder, f"perpendicular_{det_suffix}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        # Save image-keyed perpendicular_by_image_{det}.json
        # Structure: { "imagefile": { "0": { center, perpendicular, distance }, ... } }
        by_image: Dict[str, Dict] = {}
        for r in results:
            imagefile = frame_to_image.get(r['frame'], f"frame_{r['frame']:06d}")
            if imagefile not in by_image:
                by_image[imagefile] = {}
            # Local detection index within this image (sequential order of appearance)
            local_idx = str(len(by_image[imagefile]))
            cz = r['detection_center'][2]
            by_image[imagefile][local_idx] = {
                'center': r['detection_center'],
                'perpendicular': [r['foot_point'][0], r['foot_point'][1], cz],
                'distance': r['distance_m']
            }

        by_image_file = os.path.join(route_folder, f"perpendicular_by_image_{det_suffix}.json")
        with open(by_image_file, 'w', encoding='utf-8') as f:
            json.dump(by_image, f, indent=2)

        if log_fn:
            log_fn(f"Perpendicular distances saved to: {output_file}")
            log_fn(f"Per-image summary saved to: {by_image_file}")
            distances = [r['distance_m'] for r in results]
            log_fn(f"Distance stats: min={min(distances):.1f}m, "
                   f"max={max(distances):.1f}m, "
                   f"mean={sum(distances) / len(distances):.1f}m")

        if progress_fn:
            progress_fn(100)

    def run_track_perpendicular(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Calculate perpendicular distances from the last bounding box of each track to the flight route.

        For each track, finds the last detection (highest frame number), computes its
        center, and finds the nearest point on the AirData flight route LineString.
        Results are saved to:
          flight_route_{fr}/perpendicular_tracks_{trk}.json          (flat list, used by QGIS layer)
          flight_route_{fr}/perpendicular_tracks_by_track_{trk}.json (keyed by track_id)
        Both are suffixed with the tracking camera the distances were computed for.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        :param cancel_check: Optional function that returns True if cancelled
        """
        target_folder = config["target_folder"]
        target_epsg = config.get("target_epsg", 32633)
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        trk_camera = config.get("tracking_camera", "T")
        trk_suffix = "t" if trk_camera == "T" else "w"

        if log_fn:
            log_fn("Calculating perpendicular distances for tracks to flight route...")

        if progress_fn:
            progress_fn(5)

        # Load flight route LineString (camera-specific folder)
        route_line_file = os.path.join(target_folder, f"flight_route_{fr_suffix}", "flight_route.geojson")
        if not os.path.exists(route_line_file):
            raise FileNotFoundError(
                "flight_route.geojson not found. Please run 'Generate Flight Route' first."
            )

        with open(route_line_file, 'r') as f:
            route_geojson = json.load(f)

        route_coords = None
        for feature in route_geojson.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "LineString":
                route_coords = geom["coordinates"]
                break

        if not route_coords or len(route_coords) < 2:
            raise RuntimeError(
                "Flight route does not contain a valid LineString. "
                "Need at least 2 GPS points in the AirData file."
            )

        if log_fn:
            log_fn(f"Flight route loaded: {len(route_coords)} GPS points")

        if progress_fn:
            progress_fn(15)

        # Load DEM coordinate offset so poses locations can be converted to world CRS
        x_offset, y_offset = 0.0, 0.0
        dem_path = config.get("dem_path", "")
        if dem_path:
            dem_meta_path = config.get("alfs_dem_metadata_path") or \
                dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(dem_meta_path):
                try:
                    with open(dem_meta_path, 'r') as f:
                        dem_meta = json.load(f)
                    origin = dem_meta.get("origin", [0, 0, 0])
                    x_offset, y_offset = float(origin[0]), float(origin[1])
                except Exception:  # nosec B110
                    pass

        # Load poses for frame→image mapping AND camera FOV computation
        poses_images = []
        frame_to_image: Dict[int, str] = {}
        for suffix in ("t", "w"):
            poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
            if os.path.exists(poses_path):
                try:
                    with open(poses_path, 'r') as f:
                        poses = json.load(f)
                    for idx, img_info in enumerate(poses.get("images", [])):
                        imagefile = img_info.get("imagefile", "")
                        if imagefile:
                            frame_to_image[idx] = imagefile
                    if frame_to_image:
                        poses_images = poses.get("images", [])
                        break
                except Exception:  # nosec B110
                    pass

        if progress_fn:
            progress_fn(20)

        # Load all georeferenced track CSV files from tracks folder (camera-specific)
        tracks_folder = os.path.join(target_folder, f"tracks_{trk_suffix}")
        if not os.path.exists(tracks_folder):
            raise FileNotFoundError(
                f"tracks_{trk_suffix}/ folder not found. Please run 'Track Animals' first."
            )

        # Collect all entries: {track_id: [rows...]}
        from collections import defaultdict
        all_tracks: Dict[int, list] = defaultdict(list)
        csv_files_found = 0

        for fname in os.listdir(tracks_folder):
            if not fname.endswith(".csv") or fname.endswith("_pixel.csv"):
                continue
            csv_path = os.path.join(tracks_folder, fname)
            csv_files_found += 1
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split(',')
                        if len(parts) >= 10:
                            try:
                                all_tracks[int(parts[1])].append({
                                    'frame': int(parts[0]),
                                    'x1': float(parts[2]), 'y1': float(parts[3]), 'z1': float(parts[4]),
                                    'x2': float(parts[5]), 'y2': float(parts[6]), 'z2': float(parts[7]),
                                    'conf': float(parts[8]),
                                    'cls': int(parts[9])
                                })
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: could not read {fname}: {e}")

        if not all_tracks:
            raise RuntimeError("No georeferenced track data found. Please run 'Track Animals' first.")

        if log_fn:
            log_fn(f"Loaded {len(all_tracks)} tracks from {csv_files_found} CSV file(s)")

        if progress_fn:
            progress_fn(40)

        # FOV polygon cache: computed once per unique frame, reused across tracks
        # that share the same last-detection frame.
        fov_cache: Dict[int, list] = {}
        cam_pos_cache: Dict[int, tuple] = {}

        # For each track, take the last detection and compute perpendicular
        results = []
        by_track: Dict[str, dict] = {}

        for i, (track_id, rows) in enumerate(sorted(all_tracks.items())):
            if cancel_check and cancel_check():
                raise CancelledException("Track perpendicular calculation cancelled")

            # Last detection = highest frame number
            last = max(rows, key=lambda r: r['frame'])
            cx = (last['x1'] + last['x2']) / 2.0
            cy = (last['y1'] + last['y2']) / 2.0
            cz = (last['z1'] + last['z2']) / 2.0

            frame_idx = last['frame']
            if frame_idx not in fov_cache:
                if poses_images and frame_idx < len(poses_images):
                    img_meta = poses_images[frame_idx]
                    fov_cache[frame_idx] = self._compute_frame_fov_polygon(
                        img_meta, x_offset, y_offset)
                    location = img_meta.get("location", [0, 0, 0])
                    cam_pos_cache[frame_idx] = (
                        float(location[0]) + x_offset,
                        float(location[1]) + y_offset
                    )
                else:
                    fov_cache[frame_idx] = None

            fov_poly = fov_cache[frame_idx]
            cam_pos = cam_pos_cache.get(frame_idx)
            if fov_poly and cam_pos:
                best_fx, best_fy, best_dist = self._nearest_on_fov_linestring(
                    route_coords, fov_poly, cx, cy, cam_pos[0], cam_pos[1])
            elif fov_poly:
                best_fx, best_fy, best_dist = self._nearest_on_fov_linestring(
                    route_coords, fov_poly, cx, cy)
            else:
                best_fx, best_fy, best_dist = self._nearest_on_linestring(
                    route_coords, cx, cy)

            last_image = frame_to_image.get(frame_idx, f"frame_{frame_idx:06d}")

            results.append({
                'track_id': track_id,
                'last_frame': frame_idx,
                'last_image': last_image,
                'confidence': last['conf'],
                'class_id': last['cls'],
                'detection_center': [cx, cy, cz],
                'foot_point': [best_fx, best_fy],
                'distance_m': round(best_dist, 4)
            })

            by_track[str(track_id)] = {
                'last_frame': frame_idx,
                'last_image': last_image,
                'center': [cx, cy, cz],
                'perpendicular': [best_fx, best_fy, cz],
                'distance': round(best_dist, 4)
            }

            if progress_fn:
                progress_fn(40 + int((i / len(all_tracks)) * 55))

        if log_fn:
            log_fn(f"FOV polygons computed for {len(fov_cache)} unique frames")

        if progress_fn:
            progress_fn(97)

        route_folder = os.path.join(target_folder, f"flight_route_{fr_suffix}")
        os.makedirs(route_folder, exist_ok=True)

        # Flat list for QGIS layer
        flat_output = {
            'crs': f"EPSG:{target_epsg}",
            'total_tracks': len(results),
            'tracks': results
        }
        flat_file = os.path.join(route_folder, f"perpendicular_tracks_{trk_suffix}.json")
        with open(flat_file, 'w', encoding='utf-8') as f:
            json.dump(flat_output, f, indent=2)

        # Per-track keyed output
        by_track_file = os.path.join(route_folder, f"perpendicular_tracks_by_track_{trk_suffix}.json")
        with open(by_track_file, 'w', encoding='utf-8') as f:
            json.dump(by_track, f, indent=2)

        if log_fn:
            log_fn(f"Track perpendicular saved to: {flat_file}")
            log_fn(f"Per-track summary saved to: {by_track_file}")
            distances = [r['distance_m'] for r in results]
            log_fn(f"Distance stats: min={min(distances):.1f}m, "
                   f"max={max(distances):.1f}m, "
                   f"mean={sum(distances) / len(distances):.1f}m")

        if progress_fn:
            progress_fn(100)

    # ------------------------------------------------------------------ #
    # Survey analytics: density heatmap + distance sampling
    # ------------------------------------------------------------------ #

    def _collect_analytics_points(self, config, source, log_fn=None):
        """Collect world-coordinate (UTM) point locations for analytics.

        For ``source == "detections"`` every geo-referenced detection centre is
        returned. For ``source == "tracks"`` one representative point per track
        (the centroid of that track's bounding-box centres) is returned, so a
        single animal followed across many frames counts once.

        :returns: (points, suffix) where points is a list of (x, y) tuples and
                  suffix is the camera folder suffix the data came from.
        """
        target_folder = config["target_folder"]
        points = []

        if source == "tracks":
            trk_camera = config.get("tracking_camera", "T")
            suffix = "t" if trk_camera == "T" else "w"
            tracks_folder = os.path.join(target_folder, f"tracks_{suffix}")
            if not os.path.isdir(tracks_folder):
                raise FileNotFoundError(
                    f"tracks_{suffix}/ folder not found. Please run 'Track Animals' first."
                )
            from collections import defaultdict
            centres = defaultdict(list)
            for fname in os.listdir(tracks_folder):
                if not fname.endswith(".csv") or fname.endswith("_pixel.csv"):
                    continue
                with open(os.path.join(tracks_folder, fname), 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split(',')
                        if len(parts) >= 8:
                            try:
                                tid = int(parts[1])
                                cx = (float(parts[2]) + float(parts[5])) / 2.0
                                cy = (float(parts[3]) + float(parts[6])) / 2.0
                            except (ValueError, IndexError):
                                continue
                            centres[tid].append((cx, cy))
            for tid, pts in centres.items():
                mx = sum(p[0] for p in pts) / len(pts)
                my = sum(p[1] for p in pts) / len(pts)
                points.append((mx, my))
        else:
            det_camera = config.get("detection_camera", "T")
            suffix = "t" if det_camera == "T" else "w"
            georef_file = os.path.join(
                target_folder, f"georeferenced_{suffix}", "georeferenced.txt")
            if not os.path.exists(georef_file):
                raise FileNotFoundError(
                    "georeferenced.txt not found. Please run 'Geo-Reference Detections' first."
                )
            with open(georef_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            cx = (float(parts[2]) + float(parts[5])) / 2.0
                            cy = (float(parts[3]) + float(parts[6])) / 2.0
                        except (ValueError, IndexError):
                            continue
                        points.append((cx, cy))

        if log_fn:
            log_fn(f"Collected {len(points)} {source} point(s) from {suffix} outputs")
        return points, suffix

    def run_density_heatmap(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Generate a kernel-density estimate raster of animal locations.

        Points come from either geo-referenced detections or tracks (one point
        per track). A Gaussian kernel is applied and the result written as a
        single-band float32 GeoTIFF where each pixel is the estimated point
        density in animals per hectare. Cells with no signal are set to nodata.

        :param config: Configuration dictionary. Relevant keys:
            ``density_source`` ("detections"|"tracks"), ``density_cell_size`` (m),
            ``density_bandwidth`` (m), ``target_epsg``.
        """
        import math
        import numpy as np

        source = config.get("density_source", "detections")
        cell_size = float(config.get("density_cell_size", 5.0))
        bandwidth = float(config.get("density_bandwidth", 25.0))
        target_epsg = config.get("target_epsg", 32633)
        target_folder = config["target_folder"]

        if cell_size <= 0:
            raise ValueError("Density cell size must be > 0.")
        if bandwidth <= 0:
            raise ValueError("Density bandwidth must be > 0.")

        if log_fn:
            log_fn(f"Generating {source} density heatmap "
                   f"(cell={cell_size} m, bandwidth={bandwidth} m)...")
        if progress_fn:
            progress_fn(5)

        points, suffix = self._collect_analytics_points(config, source, log_fn)
        if len(points) < 1:
            raise RuntimeError(f"No {source} points available to build a density heatmap.")

        pts = np.asarray(points, dtype=np.float64)
        if progress_fn:
            progress_fn(20)

        # Grid bounds padded by 3 bandwidths so the kernel tails are captured.
        pad = 3.0 * bandwidth
        min_x = pts[:, 0].min() - pad
        max_x = pts[:, 0].max() + pad
        min_y = pts[:, 1].min() - pad
        max_y = pts[:, 1].max() + pad

        width = int(math.ceil((max_x - min_x) / cell_size))
        height = int(math.ceil((max_y - min_y) / cell_size))
        max_dim = 8192
        if width > max_dim or height > max_dim:
            scale = max((max_x - min_x) / max_dim, (max_y - min_y) / max_dim)
            cell_size = scale
            width = int(math.ceil((max_x - min_x) / cell_size))
            height = int(math.ceil((max_y - min_y) / cell_size))
            if log_fn:
                log_fn(f"Grid too large, increased cell size to {cell_size:.2f} m")
        width = max(1, width)
        height = max(1, height)

        if log_fn:
            log_fn(f"Density grid: {width} x {height} cells")
        if progress_fn:
            progress_fn(35)

        # Accumulate raw point counts into grid cells.
        counts = np.zeros((height, width), dtype=np.float64)
        for x, y in points:
            col = int((x - min_x) / cell_size)
            row = int((max_y - y) / cell_size)  # flip Y for raster orientation
            if 0 <= row < height and 0 <= col < width:
                counts[row, col] += 1.0

        if cancel_check and cancel_check():
            raise CancelledException("Density heatmap cancelled")
        if progress_fn:
            progress_fn(55)

        # Smooth with a Gaussian kernel (sigma in cells). gaussian_filter
        # conserves the total sum, so the smoothed value is counts per cell.
        sigma = bandwidth / cell_size
        try:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(counts, sigma=sigma, mode="constant", cval=0.0)
        except ImportError:
            smoothed = self._gaussian_blur_numpy(counts, sigma)

        if progress_fn:
            progress_fn(75)

        # Convert counts-per-cell to density per hectare.
        cell_area_m2 = cell_size * cell_size
        density = smoothed / cell_area_m2 * 10000.0

        # Everything below this threshold is treated as "no signal" (nodata) so
        # the empty surround renders transparent.
        nodata = -9999.0
        eps = (1.0 / cell_area_m2 * 10000.0) * 1e-4  # ~0 relative to one point
        out = np.where(density > eps, density, nodata).astype(np.float32)

        analytics_folder = os.path.join(target_folder, f"analytics_{suffix}")
        os.makedirs(analytics_folder, exist_ok=True)
        out_file = os.path.join(analytics_folder, f"density_{source}.tif")

        bounds = (min_x, min_y, max_x, max_y)
        self._save_single_band_raster(out, out_file, bounds, target_epsg, nodata, log_fn)

        valid = density[density > eps]
        stats = {
            "source": source,
            "camera_suffix": suffix,
            "crs": f"EPSG:{target_epsg}",
            "n_points": len(points),
            "cell_size_m": cell_size,
            "bandwidth_m": bandwidth,
            "unit": "points_per_hectare",
            "max_density": float(valid.max()) if valid.size else 0.0,
            "mean_density": float(valid.mean()) if valid.size else 0.0,
            "raster": out_file,
        }
        with open(os.path.join(analytics_folder, f"density_{source}.json"),
                  'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        if log_fn:
            log_fn(f"Density heatmap saved to: {out_file}")
            log_fn(f"Peak density: {stats['max_density']:.3f} points/ha "
                   f"from {len(points)} {source}")
        if progress_fn:
            progress_fn(100)

    @staticmethod
    def _gaussian_blur_numpy(grid, sigma):
        """Separable Gaussian blur fallback when SciPy is unavailable."""
        import math
        import numpy as np

        if sigma <= 0:
            return grid.astype(np.float64)
        radius = max(1, int(math.ceil(sigma * 3)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()

        out = grid.astype(np.float64)
        # Convolve rows then columns (separable).
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=out)
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=out)
        return out

    def _save_single_band_raster(self, array, output_file, bounds, crs_epsg, nodata, log_fn=None):
        """Write a single-band float32 GeoTIFF with the given world bounds."""
        import numpy as np

        height, width = array.shape
        min_x, min_y, max_x, max_y = bounds
        array = array.astype(np.float32)

        try:
            import rasterio
            from rasterio.transform import from_bounds

            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
            profile = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'width': width,
                'height': height,
                'count': 1,
                'transform': transform,
                'compress': 'lzw',
                'nodata': nodata,
            }
            try:
                from pyproj import CRS as PyprojCRS
                from rasterio.crs import CRS as RasterioCRS
                profile['crs'] = RasterioCRS.from_wkt(PyprojCRS.from_epsg(crs_epsg).to_wkt())
            except Exception:  # nosec B110
                try:
                    from rasterio.crs import CRS as RasterioCRS
                    profile['crs'] = RasterioCRS.from_epsg(crs_epsg)
                except Exception:  # nosec B110
                    pass
            if width > 256 and height > 256:
                profile.update({'tiled': True, 'blockxsize': 256, 'blockysize': 256})

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(array, 1)
                dst.set_band_description(1, "density_points_per_hectare")
            self._save_world_file(output_file, bounds, width, height)
            self._save_prj_file(output_file, crs_epsg, log_fn)
        except ImportError:
            raise RuntimeError(
                "rasterio is required to write the density heatmap GeoTIFF."
            )

    def run_coverage_map(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Build a survey coverage map from the exported frame GeoTIFFs.

        Combines the per-frame GeoTIFFs on the same grid as the orthomosaic,
        but instead of merging image content it counts, per output pixel, how
        many frames contain valid (non-nodata) data at that position. The
        result is a single-band raster where 1 means the ground was imaged
        once, N means it was seen in N overlapping frames, and nodata (0)
        means it was never covered.

        :param config: Configuration dictionary. Relevant keys:
            ``coverage_camera`` ("T"|"W"), ``coverage_cell_size`` (m, 0 =
            native GeoTIFF resolution), ``target_epsg``.
        """
        import numpy as np

        try:
            import rasterio
            from rasterio.merge import merge as rio_merge
        except ImportError as exc:
            raise ImportError(
                f"rasterio is required for coverage-map generation.\n"
                f"Original error: {exc}"
            )

        camera_sel = config.get("coverage_camera", "T")
        camera_suffix = "t" if camera_sel == "T" else "w"
        camera_name = "Thermal" if camera_sel == "T" else "RGB"
        cell_size = float(config.get("coverage_cell_size", 1.0))
        nodata_src = config.get("ortho_nodata", 0)
        target_epsg = config.get("target_epsg", 32633)
        target_folder = config["target_folder"]
        geotiff_folder = os.path.join(target_folder, f"geotiffs_{camera_suffix}")

        if log_fn:
            log_fn(f"Building coverage map from {camera_name} frame GeoTIFFs...")

        if not os.path.isdir(geotiff_folder):
            raise FileNotFoundError(
                f"GeoTIFF folder not found: {geotiff_folder}\n"
                f"Run 'Export Frames as GeoTIFF' for the {camera_name} camera first."
            )

        candidates = self._collect_frame_geotiffs(geotiff_folder)
        if not candidates:
            raise RuntimeError(f"No frame GeoTIFFs found in {geotiff_folder}.")

        if log_fn:
            res_tex = " at native resolution..."
            if cell_size > 0:
                res_tex = f" at {cell_size} m/px..."
            log_fn(f"Counting overlap across {len(candidates)} GeoTIFF(s) {res_tex}")
        if progress_fn:
            progress_fn(10)

        # 0 (or negative) means: keep the native resolution of the exports.
        res = (cell_size, cell_size) if cell_size > 0 else None

        datasets = []
        source_crs = None
        try:
            for path in candidates:
                if cancel_check and cancel_check():
                    raise CancelledException("Coverage map cancelled")
                ds = rasterio.open(path)
                if source_crs is None and ds.crs is not None:
                    source_crs = ds.crs
                datasets.append(ds)

            if progress_fn:
                progress_fn(30)

            # First pass establishes the exact output grid (shape + transform);
            # the counting pass then reuses the same inputs/parameters, so
            # rasterio produces an identically aligned grid and the per-source
            # roff/coff offsets index our count array correctly (same approach
            # as _merge_orthomosaic_average).
            base, out_transform = rio_merge(
                datasets, method="first", nodata=nodata_src, res=res)
            out_height, out_width = base.shape[1], base.shape[2]
            del base

            if cancel_check and cancel_check():
                raise CancelledException("Coverage map cancelled")
            if progress_fn:
                progress_fn(50)

            count = np.zeros((out_height, out_width), dtype=np.float64)

            def _count_valid(merged_data, new_data, merged_mask, new_mask,
                             index=None, roff=0, coff=0, **kwargs):
                # Masks are True where data is *invalid* (nodata); a pixel is
                # covered by this frame when its first band is valid.
                valid = ~new_mask
                band0 = valid[0] if valid.ndim == 3 else valid
                h, w = band0.shape
                count[roff:roff + h, coff:coff + w] += band0
                if merged_mask.shape == valid.shape:
                    merged_mask[valid] = False

            rio_merge(datasets, method=_count_valid, nodata=nodata_src, res=res)
        finally:
            for d in datasets:
                d.close()

        if progress_fn:
            progress_fn(80)

        out = count.astype(np.uint16)

        analytics_folder = os.path.join(target_folder, f"analytics_{camera_suffix}")
        os.makedirs(analytics_folder, exist_ok=True)
        output_file = os.path.join(analytics_folder, "coverage_map.tif")

        # Resolve the output CRS the same way the orthomosaic does:
        # authoritatively from target_epsg, falling back to the sources' CRS.
        out_crs = None
        try:
            from pyproj import CRS as PyprojCRS
            from rasterio.crs import CRS as RasterioCRS
            if target_epsg:
                out_crs = RasterioCRS.from_wkt(PyprojCRS.from_epsg(target_epsg).to_wkt())
        except Exception:
            out_crs = None
        if out_crs is None:
            out_crs = source_crs

        out_meta = {
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "count": 1,
            "dtype": "uint16",
            "crs": out_crs,
            "transform": out_transform,
            "compress": "LZW",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
            "nodata": 0,
        }
        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(out, 1)
            dst.set_band_description(1, "overlapping_frame_count")

        # Overview pyramids for fast GIS rendering; nearest keeps the counts
        # meaningful at reduced zoom (non-fatal on failure).
        try:
            from rasterio.enums import Resampling as RioResampling
            with rasterio.open(output_file, "r+") as dst:
                dst.build_overviews([2, 4, 8, 16], RioResampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
        except Exception as exc:
            if log_fn:
                log_fn(f"Note: could not build overviews ({exc})")

        # Summary statistics over the covered area.
        px_area_m2 = abs(out_transform.a * out_transform.e)
        covered = out[out > 0]
        stats = {
            "camera_suffix": camera_suffix,
            "crs": f"EPSG:{target_epsg}",
            "n_frames": len(candidates),
            "cell_size_m": abs(out_transform.a),
            "unit": "overlapping_frame_count",
            "max_overlap": int(covered.max()) if covered.size else 0,
            "mean_overlap": float(covered.mean()) if covered.size else 0.0,
            "covered_area_ha": float(covered.size * px_area_m2 / 10000.0),
            "multi_covered_area_ha": float(
                int((covered >= 2).sum()) * px_area_m2 / 10000.0),
            "raster": output_file,
        }
        with open(os.path.join(analytics_folder, "coverage_map.json"),
                  'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        if log_fn:
            log_fn(f"Coverage map written to: {output_file} "
                   f"({out_width} x {out_height} px)")
            log_fn(f"Covered area: {stats['covered_area_ha']:.2f} ha, "
                   f"max overlap: {stats['max_overlap']} frame(s), "
                   f"mean overlap: {stats['mean_overlap']:.2f}")
        if progress_fn:
            progress_fn(100)

    def run_distance_sampling(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Estimate density/abundance via conventional line-transect distance sampling.

        Uses the perpendicular distances already computed by 'Calculate
        Perpendicular' (detections) or 'Calculate Track Perpendicular' (tracks),
        fits half-normal and hazard-rate detection functions by maximum
        likelihood, selects the best by AIC, and reports the effective strip
        width, detection probability, density and abundance with 95% CIs.

        :param config: Configuration dictionary. Relevant keys:
            ``ds_source`` ("detections"|"tracks"), ``ds_truncation`` (m or 0/None
            for automatic 95th-percentile truncation), ``flight_route_camera``,
            ``detection_camera``/``tracking_camera``, ``target_epsg``.
        """
        import math
        import numpy as np

        source = config.get("ds_source", "detections")
        target_folder = config["target_folder"]
        target_epsg = config.get("target_epsg", 32633)

        # One or more BAMBI target folders; the perpendicular distances are
        # pooled and the flight-route lengths summed into the total effort L.
        project_folders = list(config.get("ds_project_folders") or [target_folder])

        if log_fn:
            log_fn(f"Running distance-sampling estimation ({source}) over "
                   f"{len(project_folders)} project(s)...")
        if progress_fn:
            progress_fn(5)

        # ---- Pool the perpendicular distances and effort L ---------------- #
        distance_arrays = []
        transect_length = 0.0
        projects_summary = []
        for i, folder in enumerate(project_folders):
            d, length = self._load_distance_sampling_project(
                folder, source, config, log_fn)
            distance_arrays.append(d)
            transect_length += length
            projects_summary.append({
                "target_folder": folder,
                "n": int(d.size),
                "transect_length_m": length,
            })
            if progress_fn:
                progress_fn(5 + int((i + 1) / len(project_folders) * 15))

        distances = (np.concatenate(distance_arrays) if distance_arrays
                     else np.asarray([], dtype=np.float64))
        if distances.size < 2:
            raise RuntimeError(
                "Not enough perpendicular distances for distance sampling.")
        if transect_length <= 0:
            raise RuntimeError("The pooled flight routes have zero length.")

        if progress_fn:
            progress_fn(20)

        # ---- Truncation -------------------------------------------------- #
        w_cfg = config.get("ds_truncation", 0) or 0
        if w_cfg and float(w_cfg) > 0:
            w = float(w_cfg)
        else:
            w = float(np.percentile(distances, 95))
        x = distances[distances <= w]
        n = int(x.size)
        if n < 2:
            raise RuntimeError("Truncation distance leaves too few observations.")

        if log_fn:
            log_fn(f"n={n} observations, transect length L={transect_length:.1f} m, "
                   f"truncation w={w:.2f} m")
        if progress_fn:
            progress_fn(40)

        # ---- Fit detection functions ------------------------------------- #
        models = []
        hn = self._fit_detection_function("half-normal", x, w, log_fn)
        if hn:
            models.append(hn)
        hr = self._fit_detection_function("hazard-rate", x, w, log_fn)
        if hr:
            models.append(hr)
        if not models:
            raise RuntimeError("Detection-function fitting failed for all models.")

        best = min(models, key=lambda m: m["aic"])
        if progress_fn:
            progress_fn(70)

        # ---- Density / abundance ----------------------------------------- #
        esw = best["esw"]          # effective strip half-width (m)
        p = esw / w                # average detection probability
        density_m2 = n / (2.0 * esw * transect_length)
        density_km2 = density_m2 * 1e6
        covered_area_m2 = 2.0 * w * transect_length
        abundance_covered = density_m2 * covered_area_m2  # == n / p

        # Combine encounter-rate (Poisson) and detection-function CVs.
        cv_n = 1.0 / math.sqrt(n)
        cv_esw = best["cv_esw"]
        cv_density = math.sqrt(cv_n * cv_n + cv_esw * cv_esw)
        ci_d = self._lognormal_ci(density_km2, cv_density)
        ci_n = self._lognormal_ci(abundance_covered, cv_density)

        if progress_fn:
            progress_fn(85)

        # ---- Detection-function curve + histogram for plotting ----------- #
        xs = np.linspace(0, w, 60)
        gx = best["g"](xs)
        hist_counts, hist_edges = np.histogram(x, bins=min(20, max(5, n // 5)), range=(0, w))

        notes = (
            "Encounter-rate variance uses a Poisson approximation (CV = 1/sqrt(n)); "
            "abundance is reported for the covered strip area 2*w*L. Multiply density "
            "by your study-area size for a study-area abundance estimate."
        )
        if len(project_folders) > 1:
            notes += (
                " Several projects were pooled: their perpendicular distances "
                "were combined and their flight-route lengths summed into the "
                "total effort L, assuming a shared detection function."
            )

        result = {
            "source": source,
            "crs": f"EPSG:{target_epsg}",
            "n": n,
            "n_before_truncation": int(distances.size),
            "n_projects": len(project_folders),
            "projects": projects_summary,
            "transect_length_m": transect_length,
            "truncation_m": w,
            "best_model": best["name"],
            "effective_strip_width_m": esw,
            "detection_probability": p,
            "density_per_km2": density_km2,
            "density_ci95": ci_d,
            "cv_density": cv_density,
            "covered_area_km2": covered_area_m2 / 1e6,
            "abundance_in_covered_area": abundance_covered,
            "abundance_ci95": ci_n,
            "models": [
                {
                    "name": m["name"],
                    "params": m["params"],
                    "log_likelihood": m["log_likelihood"],
                    "aic": m["aic"],
                    "esw_m": m["esw"],
                }
                for m in models
            ],
            "detection_function_curve": {"x": xs.tolist(), "g": gx.tolist()},
            "distance_histogram": {
                "counts": hist_counts.tolist(),
                "edges": hist_edges.tolist(),
            },
            "notes": notes,
        }

        det_suffix_out = ("t" if config.get("tracking_camera", "T") == "T" else "w") \
            if source == "tracks" \
            else ("t" if config.get("detection_camera", "T") == "T" else "w")
        analytics_folder = os.path.join(target_folder, f"analytics_{det_suffix_out}")
        os.makedirs(analytics_folder, exist_ok=True)
        out_file = os.path.join(analytics_folder, f"distance_sampling_{source}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        if log_fn:
            if len(project_folders) > 1:
                log_fn(f"Pooled {len(project_folders)} projects: "
                       f"{int(distances.size)} observations, "
                       f"total L={transect_length:.1f} m")
            log_fn(f"Best model: {best['name']} (AIC={best['aic']:.2f})")
            log_fn(f"ESW={esw:.2f} m, detection probability p={p:.3f}")
            log_fn(f"Density={density_km2:.3f} /km^2 "
                   f"(95% CI {ci_d[0]:.3f}-{ci_d[1]:.3f})")
            log_fn(f"Abundance in covered area ({covered_area_m2 / 1e6:.3f} km^2): "
                   f"{abundance_covered:.1f} (95% CI {ci_n[0]:.1f}-{ci_n[1]:.1f})")
            log_fn(f"Distance-sampling results saved to: {out_file}")
        if progress_fn:
            progress_fn(100)

    def _load_distance_sampling_project(self, target_folder, source, config,
                                        log_fn=None):
        """Perpendicular distances and flight-route length L for one project.

        Returns ``(distances, transect_length_m)`` — a numpy array of the
        finite, non-negative perpendicular distances and the flight-route
        length in metres. Errors name the project folder so a multi-project run
        can say which flight is at fault.
        """
        import numpy as np

        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        if source == "tracks":
            trk_suffix = "t" if config.get("tracking_camera", "T") == "T" else "w"
            perp_file = os.path.join(
                target_folder, f"flight_route_{fr_suffix}",
                f"perpendicular_tracks_{trk_suffix}.json")
            list_key, dist_key = "tracks", "distance_m"
            prereq = "Calculate Track Perpendicular"
        else:
            det_suffix = "t" if config.get("detection_camera", "T") == "T" else "w"
            perp_file = os.path.join(
                target_folder, f"flight_route_{fr_suffix}",
                f"perpendicular_{det_suffix}.json")
            list_key, dist_key = "perpendiculas", "distance_m"
            prereq = "Calculate Perpendicular"

        if not os.path.exists(perp_file):
            raise FileNotFoundError(
                f"{os.path.basename(perp_file)} not found in {target_folder}. "
                f"Please run '{prereq}' first.")
        with open(perp_file, 'r', encoding='utf-8') as f:
            perp_data = json.load(f)
        distances = np.asarray(
            [float(e[dist_key]) for e in perp_data.get(list_key, []) if dist_key in e],
            dtype=np.float64)
        distances = distances[np.isfinite(distances) & (distances >= 0)]

        route_file = os.path.join(
            target_folder, f"flight_route_{fr_suffix}", "flight_route.geojson")
        if not os.path.exists(route_file):
            raise FileNotFoundError(
                f"flight_route.geojson not found in {target_folder}. "
                "Please run 'Generate Flight Route' first.")
        with open(route_file, 'r', encoding='utf-8') as f:
            route_geojson = json.load(f)
        route_coords = None
        for feature in route_geojson.get("features", []):
            if feature.get("geometry", {}).get("type") == "LineString":
                route_coords = feature["geometry"]["coordinates"]
                break
        if not route_coords or len(route_coords) < 2:
            raise RuntimeError(
                f"Flight route in {target_folder} does not contain a valid "
                "LineString.")
        rc = np.asarray(route_coords, dtype=np.float64)
        seg = np.diff(rc[:, :2], axis=0)
        transect_length = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        if transect_length <= 0:
            raise RuntimeError(
                f"Flight route in {target_folder} has zero length.")

        if log_fn:
            log_fn(f"  {target_folder}: {int(distances.size)} distance(s), "
                   f"L={transect_length:.1f} m")
        return distances, transect_length

    def _fit_detection_function(self, name, x, w, log_fn=None):
        """Fit a detection function by MLE; returns a dict of results or None.

        Supported: "half-normal" (1 param sigma) and "hazard-rate" (sigma, b).
        The observed-distance likelihood is f(x) = g(x) / mu, with
        mu = integral of g over [0, w] (the effective strip half-width, ESW).
        """
        import math
        import numpy as np

        n = x.size
        # NumPy 2.0 renamed ``trapz`` to ``trapezoid`` and 2.x removed the alias.
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        def esw_of(g):
            xs = np.linspace(0.0, w, 512)
            return float(_trapz(g(xs), xs))

        if name == "half-normal":
            def make_g(theta):
                sigma = math.exp(theta[0])
                two_var = 2.0 * sigma * sigma
                return lambda t: np.exp(-(np.asarray(t, dtype=np.float64) ** 2) / two_var)

            def nll(theta):
                sigma = math.exp(theta[0])
                g = make_g(theta)
                mu = esw_of(g)
                if mu <= 0:
                    return 1e12
                return float(np.sum(x * x) / (2.0 * sigma * sigma) + n * math.log(mu))

            theta0 = [math.log(max(np.std(x), 1.0))]
            k = 1
        elif name == "hazard-rate":
            def make_g(theta):
                sigma = math.exp(theta[0])
                b = 1.0 + math.exp(theta[1])

                def g(t):
                    t = np.asarray(t, dtype=np.float64)
                    out = np.ones_like(t)
                    nz = t > 0
                    out[nz] = 1.0 - np.exp(-((t[nz] / sigma) ** (-b)))
                    return out
                return g

            def nll(theta):
                g = make_g(theta)
                mu = esw_of(g)
                if mu <= 0:
                    return 1e12
                gx = g(x)
                gx = np.clip(gx, 1e-12, 1.0)
                return float(-np.sum(np.log(gx)) + n * math.log(mu))

            theta0 = [math.log(max(np.std(x), 1.0)), 0.0]
            k = 2
        else:
            return None

        try:
            from scipy.optimize import minimize
            res = minimize(nll, theta0, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
            theta = res.x
            best_nll = float(res.fun)
        except ImportError:
            if name != "half-normal":
                return None  # hazard-rate needs an optimizer
            # Closed-form-ish 1D search for the half-normal.
            grid = np.log(np.linspace(max(np.std(x), 0.5), w * 2 + 1, 400))
            vals = [nll([t]) for t in grid]
            theta = [grid[int(np.argmin(vals))]]
            best_nll = float(min(vals))

        g = make_g(theta)
        esw = esw_of(g)
        if not math.isfinite(best_nll) or esw <= 0:
            return None

        # CV of ESW via a numeric Hessian of the NLL (observed information).
        cv_esw = self._cv_esw_numeric(nll, esw_of, make_g, theta)
        aic = 2.0 * k + 2.0 * best_nll

        if name == "half-normal":
            params = {"sigma": math.exp(theta[0])}
        else:
            params = {"sigma": math.exp(theta[0]), "b": 1.0 + math.exp(theta[1])}

        if log_fn:
            log_fn(f"  {name}: ESW={esw:.2f} m, AIC={aic:.2f}, params={params}")

        return {
            "name": name, "params": params, "log_likelihood": -best_nll,
            "aic": aic, "esw": esw, "cv_esw": cv_esw, "g": g,
        }

    @staticmethod
    def _cv_esw_numeric(nll, esw_of, make_g, theta):
        """Delta-method CV of the ESW from a numeric Hessian of the NLL."""
        import math
        import numpy as np

        def perturb(base, idx, delta):
            out = base.copy()
            out[idx] += delta
            return out

        theta = np.asarray(theta, dtype=np.float64)
        m = theta.size
        h = 1e-4 * (np.abs(theta) + 1.0)

        # Hessian of the NLL (observed information matrix).
        hess = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                tpp = perturb(perturb(theta, i, h[i]), j, h[j])
                tpm = perturb(perturb(theta, i, h[i]), j, -h[j])
                tmp = perturb(perturb(theta, i, -h[i]), j, h[j])
                tmm = perturb(perturb(theta, i, -h[i]), j, -h[j])
                numer = nll(tpp.tolist()) - nll(tpm.tolist()) - nll(tmp.tolist()) + nll(tmm.tolist())
                hess[i, j] = hess[j, i] = numer / (4.0 * h[i] * h[j])
        try:
            cov = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return 0.0
        if not np.all(np.isfinite(cov)):
            return 0.0

        # Gradient of ESW w.r.t. theta (numeric).
        grad = np.zeros(m)
        esw0 = esw_of(make_g(theta.tolist()))
        for i in range(m):
            tp = perturb(theta, i, h[i])
            tm = perturb(theta, i, -h[i])
            d_esw = esw_of(make_g(tp.tolist())) - esw_of(make_g(tm.tolist()))
            grad[i] = d_esw / (2.0 * h[i])
        var_esw = float(grad @ cov @ grad)
        if var_esw <= 0 or not math.isfinite(var_esw) or esw0 <= 0:
            return 0.0
        return math.sqrt(var_esw) / esw0

    @staticmethod
    def _lognormal_ci(estimate, cv, z=1.96):
        """95% lognormal confidence interval for a positive estimate."""
        import math

        if estimate <= 0 or cv <= 0 or not math.isfinite(cv):
            return [estimate, estimate]
        c = math.exp(z * math.sqrt(math.log(1.0 + cv * cv)))
        return [estimate / c, estimate * c]

    # ------------------------------------------------------------------ #
    # Survey analytics: transect-based population estimation
    # ------------------------------------------------------------------ #

    def run_population_estimation(self, config: Dict[str, Any], progress_fn=None,
                                  log_fn=None, cancel_check=None):
        """Estimate population density from the transects of a split flight.

        Builds the per-transect count/area table the R analysis of Praschl et
        al. 2026 works on and runs the naive, bootstrap and ZINB estimators on
        it (see :mod:`core.population`):

        * **count** — every track is assigned to the transect whose centre
          line (the flight path between its start and end frame) is nearest in
          perpendicular distance, optionally truncated at ``pop_truncation``.
        * **area** — the union of the per-frame field-of-view footprints of
          the frames inside the transect's frame range, in hectares.

        The optional *study area* is what the densities are extrapolated to an
        abundance for, and it is a different quantity from the monitored area:
        the latter is the searched strip (the density's denominator), the
        former the region the density is assumed to hold over. Setting the
        study area to the monitored area therefore just returns the counted
        animals. ``pop_study_area_auto`` fills it with the flight's total FoV
        coverage — the union of *every* frame's footprint, which unlike the
        summed transect areas counts ground seen by two transects only once
        and also includes frames belonging to no transect.

        Prerequisites: the Transect Splitting Tool (``transects_{m}``), the
        FoV step (``fov_{m}/fov_polygons.txt``) and 'Calculate Track
        Perpendicular' (``flight_route_{fr}/perpendicular_tracks_{m}.json``).

        :param config: Configuration dictionary. Relevant keys:
            ``pop_camera`` ("T"|"W"), ``pop_truncation`` (m, 0 = keep all),
            ``pop_methods`` (list of "naive"/"bootstrap"/"zinb"),
            ``pop_n_boot``, ``pop_seed``, ``pop_study_area_auto`` (bool),
            ``pop_study_area_ha`` (manual, 0 = no abundance),
            ``flight_route_camera``, ``dem_path``, ``target_epsg``.
        """
        from .core.population import estimate_population

        target_folder = config["target_folder"]
        target_epsg = config.get("target_epsg", 32633)
        camera = config.get("pop_camera", "T")
        suffix = "t" if camera == "T" else "w"
        truncation = float(config.get("pop_truncation", 0.0) or 0.0)
        methods = list(config.get("pop_methods") or ["naive", "bootstrap", "zinb"])
        n_boot = int(config.get("pop_n_boot", 999) or 999)
        seed = int(config.get("pop_seed", 42) or 42)
        study_area_auto = bool(config.get("pop_study_area_auto", False))
        study_area_ha = float(config.get("pop_study_area_ha", 0.0) or 0.0)

        # One or more BAMBI projects: every project's transects are pooled into
        # a single count/area table and the estimators run over all of them
        # together. Each project keeps its own DEM georeferencing — the active
        # project from the config, added projects from their own dem.json.
        raw_entries = config.get("pop_project_folders") or [{"target": target_folder}]
        entries = []
        for e in raw_entries:
            if isinstance(e, dict):
                entries.append((e.get("target", ""), e.get("dem", "") or ""))
            else:                       # tolerate a bare folder string
                entries.append((e, ""))
        current_key = os.path.normcase(os.path.abspath(target_folder))
        multi = len(entries) > 1
        camera_label = "Thermal" if suffix == "t" else "RGB"

        if log_fn:
            log_fn(f"Running transect population estimation ({camera_label}) "
                   f"over {len(entries)} project(s)...")
        if progress_fn:
            progress_fn(3)

        combined_rows = []
        combined_usable = []
        total_flight_area_ha = 0.0
        agg = {"n_tracks": 0, "n_assigned": 0, "n_truncated": 0,
               "n_outside": 0, "n_without_area": 0, "mismatched": 0}
        projects_summary = []

        for pi, (folder, dem_override) in enumerate(entries):
            if cancel_check and cancel_check():
                raise CancelledException("Population estimation cancelled")
            is_current = os.path.normcase(os.path.abspath(folder)) == current_key

            def _proj_progress(pct, _pi=pi):
                if progress_fn:
                    span = 68.0 / len(entries)
                    progress_fn(5 + int((_pi + pct / 100.0) * span))

            proj = self._population_project_table(
                folder, config, is_current, dem_override,
                log_fn, cancel_check, _proj_progress)

            proj_name = os.path.basename(os.path.normpath(folder)) or folder
            for r in proj["rows"]:
                row = dict(r)
                row["project"] = proj_name
                if multi:
                    row["name"] = f"{proj_name} / {r['name']}"
                combined_rows.append(row)
                if row["area_ha"] > 0:
                    combined_usable.append(row)

            total_flight_area_ha += proj["flight_area_ha"]
            for key in agg:
                agg[key] += proj[key]
            projects_summary.append({
                "target_folder": folder,
                "name": proj_name,
                "n_transects": len(proj["usable"]),
                "n_transects_total": len(proj["rows"]),
                "total_count": proj["total_count"],
                "total_ha": proj["total_ha"],
                "flight_fov_area_ha": proj["flight_area_ha"],
                "dem_origin_source": proj["origin_source"],
            })

        if not combined_usable:
            raise RuntimeError(
                "No transect has a monitored area in any project. Check that "
                "the FoV step covered the transects' frame ranges.")

        # ---- Study area (region the density is extrapolated to) ------------ #
        study_area_source = "none"
        if study_area_auto:
            study_area_ha = total_flight_area_ha
            study_area_source = "flight_fov"
            if total_flight_area_ha <= 0:
                raise RuntimeError(
                    "The flights have no FoV coverage, so no study area could "
                    "be derived. Untick 'Use flight FoV area' or check the FoV "
                    "step.")
            if log_fn:
                log_fn(f"Study area from the flights' total FoV coverage: "
                       f"{total_flight_area_ha:.2f} ha")
        elif study_area_ha > 0:
            study_area_source = "manual"

        # ---- Estimate over the pooled transects ---------------------------- #
        if progress_fn:
            progress_fn(78)
        result = estimate_population(
            combined_usable, methods=methods, n_boot=n_boot, seed=seed,
            study_area_ha=study_area_ha)

        notes = (
            "A track counts towards a transect only when it lies inside that "
            "transect's monitored area (the union of the field-of-view "
            "footprints of the frames in its range); where several transects "
            "cover it, the one whose flight path is nearest in perpendicular "
            "distance wins. "
            "Densities are per 100 ha (= per km²). The monitored area sums the "
            "transects (each is its own sample, so shared ground counts once "
            "per transect), while flight_fov_area_ha unions every frame and "
            "counts shared ground once — they differ where transects overlap "
            "or where frames belong to no transect."
        )
        if multi:
            notes += (
                " Several projects were pooled: every project's transects are "
                "one shared sample set for the estimators, the monitored areas "
                "and flight FoV coverages summed across flights. Each project's "
                "own DEM georeferencing was used to place its transects."
            )

        result.update({
            "camera": camera_label,
            "crs": f"EPSG:{target_epsg}",
            "truncation_m": truncation,
            "n_projects": len(entries),
            "projects": projects_summary,
            "n_tracks": agg["n_tracks"],
            "n_tracks_assigned": agg["n_assigned"],
            "n_tracks_truncated": agg["n_truncated"],
            "n_tracks_outside_fov": agg["n_outside"],
            "n_transects_without_area": agg["n_without_area"],
            "flight_fov_area_ha": total_flight_area_ha,
            "study_area_source": study_area_source,
            "transects": combined_rows,
            "notes": notes,
        })

        # ---- Write outputs -------------------------------------------------- #
        analytics_folder = os.path.join(target_folder, f"analytics_{suffix}")
        os.makedirs(analytics_folder, exist_ok=True)

        out_file = os.path.join(analytics_folder, "population_estimate.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        # A combined per-transect table (with a project column) alongside the
        # per-project CSVs each project already wrote into its own analytics
        # folder; only needed when more than one project was pooled.
        combined_csv = None
        if multi:
            combined_csv = os.path.join(
                analytics_folder, "population_transects_combined.csv")
            with open(combined_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["project", "id", "name", "start_frame",
                                 "end_frame", "length_m", "area_m2", "area_ha",
                                 "count", "n_frames", "n_frames_with_fov"])
                for r in combined_rows:
                    writer.writerow([
                        r.get("project", ""), r["id"], r["name"],
                        r["start_frame"], r["end_frame"], r["length_m"],
                        r["area_m2"], round(r["area_ha"], 4), r["count"],
                        r["n_frames"], r["n_frames_with_fov"],
                    ])

        if log_fn:
            total_ha = result["total_ha"]
            log_fn(f"Total: {int(result['total_count'])} animals on "
                   f"{total_ha:.2f} ha across {result['n_transects']} transect(s) "
                   f"({result['n_zero_transects']} with zero counts)")
            if agg["mismatched"]:
                log_fn(f"Note: {agg['mismatched']} track(s) were assigned to a "
                       "transect whose frame range does not contain their "
                       "sighting frame (transects running close together).")
            for name, est in result["estimates"].items():
                density = est.get("density_per_100ha")
                if density is None:
                    log_fn(f"  {name}: failed — {est.get('error')}")
                    continue
                ci = est.get("ci95")
                ci_txt = (f" (95% CI {ci[0]:.2f}–{ci[1]:.2f})" if ci else "")
                log_fn(f"  {name}: {density:.2f} animals/100 ha{ci_txt}")
                if est.get("error"):
                    log_fn(f"    warning: {est['error']}")
            log_fn(f"Population estimate saved to: {out_file}")
            if combined_csv:
                log_fn(f"Combined per-transect table saved to: {combined_csv}")

        if progress_fn:
            progress_fn(100)

    def _population_project_table(self, target_folder, config, is_current,
                                  dem_override="", log_fn=None, cancel_check=None,
                                  progress_cb=None):
        """Build one project's transect count/area table and write its outputs.

        Runs the per-project half of :meth:`run_population_estimation` — loads
        the transects, poses, FoV footprints and track perpendicular distances,
        assigns the tracks and measures each transect's monitored area — then
        writes that project's own ``population_transects.csv``,
        ``population_tracks.csv``, ``transect_areas.geojson`` and
        ``transect_routes.geojson`` into its ``analytics_{m}`` folder. The
        estimation itself runs once on the pooled table in the caller.

        :param is_current: True when *target_folder* is the active project, so
            its DEM origin comes straight from the config.
        :param dem_override: DEM metadata JSON of an added (non-active) project,
            supplying its mesh->world origin.
        :return: a dict with ``rows`` (all transects), ``usable`` (area > 0),
            ``flight_area_ha``, the track-assignment tallies (``n_tracks``,
            ``n_assigned``, ``n_truncated``, ``n_outside``, ``n_without_area``,
            ``mismatched``), ``total_count``/``total_ha`` over the usable
            transects and ``origin_source``.
        """
        from .core.pipeline_outputs import load_fov_polygons_3d
        from .core.population import (
            assign_tracks, geometry_to_rings, merged_fov_area,
            shapely_area_predicate, transect_centerline,
        )
        from .core.transects import TransectStore, cumulative_distances, \
            flight_positions, path_length

        target_epsg = config.get("target_epsg", 32633)
        camera = config.get("pop_camera", "T")
        suffix = "t" if camera == "T" else "w"
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        truncation = float(config.get("pop_truncation", 0.0) or 0.0)

        if log_fn:
            log_fn(f"Project {target_folder}:")

        # ---- Transects ---------------------------------------------------- #
        store = TransectStore(target_folder, suffix)
        if not store.load():
            raise FileNotFoundError(
                f"transects_{suffix}/transects.json not found in {target_folder}. "
                "Please define transects with the Transect Splitting Tool first."
            )
        transects = store.ordered()
        if not transects:
            raise RuntimeError(
                f"The transect definition in {target_folder} contains no transects.")

        # ---- Poses (transect centre lines) -------------------------------- #
        poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(
                f"poses_{suffix}.json not found in {target_folder}. Please run "
                "frame extraction first.")
        with open(poses_path, 'r', encoding='utf-8') as f:
            images = json.load(f).get("images", [])
        if not images:
            raise RuntimeError(f"poses_{suffix}.json in {target_folder} has no frames.")
        for transect in transects:
            transect.clamp(len(images))

        # The poses store camera positions mesh-locally; the tracks' geo-referenced
        # positions are in the world CRS, so shift the centre lines by the DEM origin.
        x_offset, y_offset, origin_source = self._resolve_population_origin(
            target_folder, config, is_current, dem_override, log_fn)

        centerlines = {
            t.transect_id: transect_centerline(
                images, t.first_frame, t.last_frame, x_offset, y_offset)
            for t in transects
        }
        if progress_cb:
            progress_cb(20)

        # ---- Track perpendicular distances -------------------------------- #
        perp_file = os.path.join(
            target_folder, f"flight_route_{fr_suffix}",
            f"perpendicular_tracks_{suffix}.json")
        if not os.path.exists(perp_file):
            raise FileNotFoundError(
                f"{os.path.basename(perp_file)} not found in {target_folder}. "
                "Please run 'Calculate Track Perpendicular' first."
            )
        with open(perp_file, 'r', encoding='utf-8') as f:
            tracks = json.load(f).get("tracks", [])
        if not tracks:
            raise RuntimeError(
                f"No tracks with perpendicular distances found in {target_folder}.")

        # ---- Monitored area per transect (union of the frame FoVs) --------- #
        # Computed before the assignment: a track only counts towards a
        # transect whose footprint contains it, so the areas are needed first.
        fov_file = os.path.join(target_folder, f"fov_{suffix}", "fov_polygons.txt")
        if not os.path.exists(fov_file):
            raise FileNotFoundError(
                f"fov_{suffix}/fov_polygons.txt not found in {target_folder}. "
                "Please run 'Calculate Field of View' first."
            )
        fov_polygons = load_fov_polygons_3d(fov_file, log_fn)
        if not fov_polygons:
            raise RuntimeError(
                f"The FoV file in {target_folder} contains no polygons.")

        cum = cumulative_distances(flight_positions(images))
        areas: Dict[int, object] = {}
        area_m2_by_id: Dict[int, float] = {}
        frames_by_id: Dict[int, Tuple[int, int]] = {}
        for i, transect in enumerate(transects):
            if cancel_check and cancel_check():
                raise CancelledException("Population estimation cancelled")

            frames = list(range(transect.first_frame, transect.last_frame + 1))
            area_m2, geometry = merged_fov_area(fov_polygons, frames)
            areas[transect.transect_id] = geometry
            area_m2_by_id[transect.transect_id] = area_m2
            frames_by_id[transect.transect_id] = (
                len(frames), sum(1 for f in frames if f in fov_polygons))
            if area_m2 <= 0 and log_fn:
                log_fn(f"Warning: {transect.display_name} has no FoV footprint "
                       f"(frames {transect.first_frame}-{transect.last_frame}) — "
                       "it contributes a zero area.")
            if progress_cb:
                progress_cb(25 + int((i + 1) / len(transects) * 30))

        # ---- Assign the tracks --------------------------------------------- #
        assignments = assign_tracks(
            tracks, centerlines, truncation,
            contains=shapely_area_predicate(areas))
        counts: Dict[int, int] = {t.transect_id: 0 for t in transects}
        for a in assignments:
            if a["transect_id"] is not None:
                counts[a["transect_id"]] += 1
        if progress_cb:
            progress_cb(65)

        rows = []
        area_features = []
        for transect in transects:
            area_m2 = area_m2_by_id[transect.transect_id]
            n_frames, n_with_fov = frames_by_id[transect.transect_id]
            rows.append({
                "id": transect.transect_id,
                "name": transect.display_name,
                "start_frame": transect.start_frame,
                "end_frame": transect.end_frame,
                "length_m": round(path_length(
                    cum, transect.first_frame, transect.last_frame), 2),
                "area_m2": round(area_m2, 2),
                "area_ha": area_m2 / 10000.0,
                "count": counts[transect.transect_id],
                "n_frames": n_frames,
                "n_frames_with_fov": n_with_fov,
            })
            for ring in geometry_to_rings(areas[transect.transect_id]):
                area_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "transect_id": transect.transect_id,
                        "name": transect.display_name,
                        "area_ha": round(area_m2 / 10000.0, 4),
                        "count": counts[transect.transect_id],
                    },
                })

        usable = [r for r in rows if r["area_ha"] > 0]
        n_assigned = sum(1 for a in assignments if a["transect_id"] is not None)
        n_truncated = sum(1 for a in assignments if a["truncated"])
        n_outside = sum(1 for a in assignments if a["outside_fov"])
        if log_fn:
            log_fn(f"{len(transects)} transect(s), {len(assignments)} track(s): "
                   f"{n_assigned} assigned, {len(assignments) - n_assigned} "
                   f"not counted ({n_outside} outside every transect's field of "
                   f"view, {n_truncated} beyond the truncation distance)")
            if len(usable) < len(rows):
                log_fn(f"Warning: {len(rows) - len(usable)} transect(s) without a "
                       "monitored area were excluded from the estimation.")

        # ---- Flight FoV coverage (candidate study area) -------------------- #
        # The union over *every* frame, so ground seen by two transects counts
        # once — unlike the summed transect areas, which the density needs to
        # double-count because each transect is its own sample.
        if progress_cb:
            progress_cb(75)
        flight_area_m2, _flight_geom = merged_fov_area(
            fov_polygons, sorted(fov_polygons.keys()))
        flight_area_ha = flight_area_m2 / 10000.0

        # ---- Write this project's outputs ---------------------------------- #
        analytics_folder = os.path.join(target_folder, f"analytics_{suffix}")
        os.makedirs(analytics_folder, exist_ok=True)

        transects_csv = os.path.join(analytics_folder, "population_transects.csv")
        with open(transects_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "start_frame", "end_frame", "length_m",
                             "area_m2", "area_ha", "count", "n_frames",
                             "n_frames_with_fov"])
            for r in rows:
                writer.writerow([
                    r["id"], r["name"], r["start_frame"], r["end_frame"],
                    r["length_m"], r["area_m2"], round(r["area_ha"], 4),
                    r["count"], r["n_frames"], r["n_frames_with_fov"],
                ])

        names = {t.transect_id: t.display_name for t in transects}
        ranges = {t.transect_id: (t.first_frame, t.last_frame) for t in transects}
        tracks_csv = os.path.join(analytics_folder, "population_tracks.csv")
        with open(tracks_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "last_frame", "x", "y", "class_id",
                             "transect_id", "transect_name", "distance_m",
                             "nearest_distance_m", "in_frame_range",
                             "truncated", "outside_fov"])
            for a in assignments:
                tid = a["transect_id"]
                rng = ranges.get(tid)
                in_range = ""
                if rng is not None and a["last_frame"] is not None:
                    in_range = int(rng[0] <= a["last_frame"] <= rng[1])
                nearest = a["nearest_distance_m"]
                writer.writerow([
                    a["track_id"], a["last_frame"],
                    round(a["x"], 3), round(a["y"], 3), a["class_id"],
                    "" if tid is None else tid,
                    "" if tid is None else names.get(tid, ""),
                    "" if a["distance_m"] is None else round(a["distance_m"], 3),
                    "" if nearest is None else round(nearest, 3),
                    in_range, int(a["truncated"]), int(a["outside_fov"]),
                ])

        areas_geojson = os.path.join(analytics_folder, "transect_areas.geojson")
        with open(areas_geojson, 'w', encoding='utf-8') as f:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name",
                        "properties": {"name": f"EPSG:{target_epsg}"}},
                "features": area_features,
            }, f, indent=2)

        # The sub-flight route of each transect (start frame -> end frame), so
        # the map layers can show the flown line inside its monitored area.
        by_id = {r["id"]: r for r in rows}
        route_features = []
        for transect in transects:
            line = centerlines.get(transect.transect_id) or []
            if len(line) < 2:
                continue
            row = by_id[transect.transect_id]
            route_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[x, y] for x, y in line],
                },
                "properties": {
                    "transect_id": transect.transect_id,
                    "name": transect.display_name,
                    "start_frame": transect.start_frame,
                    "end_frame": transect.end_frame,
                    "length_m": row["length_m"],
                    "count": row["count"],
                },
            })
        routes_geojson = os.path.join(analytics_folder, "transect_routes.geojson")
        with open(routes_geojson, 'w', encoding='utf-8') as f:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name",
                        "properties": {"name": f"EPSG:{target_epsg}"}},
                "features": route_features,
            }, f, indent=2)

        # Tracks assigned to a transect they were not filmed in usually mean two
        # transects run close together — worth surfacing, not worth failing on.
        def _filmed_elsewhere(a: dict) -> bool:
            tid, frame = a["transect_id"], a["last_frame"]
            if tid is None or frame is None:
                return False
            lo, hi = ranges[tid]
            return not lo <= frame <= hi

        mismatched = sum(1 for a in assignments if _filmed_elsewhere(a))
        if progress_cb:
            progress_cb(100)

        return {
            "target_folder": target_folder,
            "rows": rows,
            "usable": usable,
            "flight_area_ha": flight_area_ha,
            "n_tracks": len(assignments),
            "n_assigned": n_assigned,
            "n_truncated": n_truncated,
            "n_outside": n_outside,
            "n_without_area": len(rows) - len(usable),
            "mismatched": mismatched,
            "total_count": sum(r["count"] for r in usable),
            "total_ha": sum(r["area_ha"] for r in usable),
            "origin_source": origin_source,
        }

    def _resolve_population_origin(self, target_folder, config, is_current,
                                   dem_override="", log_fn=None):
        """The mesh→world XY offset for one project's transect centre lines.

        The poses store camera positions mesh-locally (world CRS minus the DEM
        origin), so the centre lines are shifted back by that origin. The active
        project takes it from the config's DEM metadata (unchanged behaviour);
        an added project supplies its own ``dem.json`` via *dem_override*.
        Returns ``(x_offset, y_offset, source)``.
        """
        from .core.pipeline_outputs import read_dem_origin_xy

        # The active project takes its origin straight from the config's DEM
        # metadata — unchanged from the single-project behaviour, including the
        # (0, 0) fallback of read_dem_origin_xy when no metadata is configured.
        if is_current:
            x_offset, y_offset = read_dem_origin_xy(
                config.get("dem_path", "") or "",
                config.get("alfs_dem_metadata_path") or "")
            return x_offset, y_offset, "config"

        # An added project must carry its own DEM metadata JSON.
        if not dem_override or not os.path.isfile(dem_override):
            raise FileNotFoundError(
                f"No DEM metadata (dem.json) provided for project {target_folder}. "
                "Add the project's dem.json so its transects can be georeferenced."
            )
        x_offset, y_offset = read_dem_origin_xy("", dem_override)
        if log_fn:
            log_fn(f"  DEM origin from {os.path.basename(dem_override)}: "
                   f"({x_offset:.1f}, {y_offset:.1f})")
        return x_offset, y_offset, "provided"

    def run_detection(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Run animal detection on extracted frames.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import cv2

        from bambi.ai.models.ultralytics_yolo_detector import UltralyticsYoloDetector
        from bambi.ai.output.yolo_writer import YoloWriter

        camera = config.get("detection_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        target_folder = config["target_folder"]
        # Pick the model matching the selected modality; fall back to the
        # legacy single-model key for old configs.
        if camera == "T":
            model_path = config.get("thermal_model_path") or config.get("model_path")
        else:
            model_path = config.get("rgb_model_path") or config.get("model_path")
        min_confidence = config.get("min_confidence", 0.5)

        # Frame filter options (use start/end frame like alfs)
        use_all_frames = config.get("detect_use_all_frames", True)
        start_frame = config.get("detect_start_frame", 0)
        end_frame = config.get("detect_end_frame", 999999)
        sample_rate = config.get("detect_sample_rate", 1)

        # Set frames folder based on camera selection
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        if log_fn:
            log_fn(f"Running detection on {camera_name} frames...")

        # Download default model if not specified
        if not model_path:
            if log_fn:
                log_fn(f"No {camera_name} model specified, downloading default model...")
            model_path = self.download_default_model(log_fn, camera=camera)

        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if log_fn:
            log_fn(f"Using model: {model_path}")

        # Load poses to get frame list
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)

        images = poses.get("images", [])
        total_frames = len(images)

        if total_frames == 0:
            raise RuntimeError(f"No frames found in poses_{camera_suffix}.json")

        if log_fn:
            log_fn(f"Found {total_frames} extracted {camera_name} frames")

        # Apply frame filters (start/end frame, then sample rate)
        if use_all_frames:
            frame_indices = list(range(total_frames))
            if log_fn:
                log_fn("Frame range: All frames")
        else:
            # Clamp end_frame to valid range
            end_frame = min(end_frame, total_frames - 1)
            frame_indices = list(range(start_frame, end_frame + 1))
            if log_fn:
                log_fn(f"Frame range: {start_frame} to {end_frame}")

        # Apply sample rate
        if sample_rate > 1:
            frame_indices = frame_indices[::sample_rate]
            if log_fn:
                log_fn(f"Sampling every {sample_rate}th frame")

        if len(frame_indices) == 0:
            raise RuntimeError("No frames to process after applying filters")

        if log_fn:
            log_fn(f"Processing {len(frame_indices)} frames after filtering")

        # Create output folder (camera-specific)
        detections_folder = os.path.join(target_folder, f"detections_{camera_suffix}")
        os.makedirs(detections_folder, exist_ok=True)

        # Initialize detector
        detector = UltralyticsYoloDetector(
            model_path=model_path,
            labels=['animal'],
            detection_writer=YoloWriter(),
            min_confidence=min_confidence,
            verbose=False
        )

        # Process filtered frames
        detection_results = 0
        processed = 0

        # Write detection results
        output_file = os.path.join(detections_folder, "detections.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# frame x1 y1 x2 y2 confidence class_id\n")
            for idx in frame_indices:
                # Check for cancellation
                if cancel_check and cancel_check():
                    if log_fn:
                        log_fn("Detection cancelled by user")
                    raise CancelledException("Detection cancelled")

                image_info = images[idx]
                imagefile = image_info.get("imagefile")
                if not imagefile:
                    continue

                image_path = os.path.join(frames_folder, imagefile)
                if log_fn:
                    log_fn(f"Detecting frame {idx} / {total_frames}: {image_path}")

                if not os.path.exists(image_path):
                    if log_fn:
                        log_fn(f"Warning: Image not found: {image_path}")
                    continue

                # Load and detect
                frame = cv2.imread(image_path)
                if frame is None:
                    continue

                boxes = detector.detect_frame(idx, frame)

                for box in boxes:
                    detection_results += 1
                    f.write(f"{idx} {box.start_x:.2f} {box.start_y:.2f} "
                            f"{box.end_x:.2f} {box.end_y:.2f} {box.propability:.4f} "
                            f"{box.label}\n")

                processed += 1

                if progress_fn and processed % 10 == 0:
                    progress = int((processed / len(frame_indices)) * 100)
                    progress_fn(min(progress, 99))

        if log_fn:
            log_fn(f"Detection complete: {detection_results} detections in {processed} frames")

        if progress_fn:
            progress_fn(100)

    @staticmethod
    def _classify_projection_miss(camera, label_coords, input_resolution, mesh_bounds):
        """Explain why a detection's rays failed to intersect the DEM mesh.

        Re-creates the same rays ``pixel_to_world_coord`` casts (camera looks
        along -Z, NDC scaled by ``tan(fovy/2)``) and tests them against the
        mesh's mean-height plane.

        :param camera: alfspy ``Camera`` used for the failed projection
        :param label_coords: flat ``[x1, y1, x2, y1, ...]`` pixel corner list
        :param input_resolution: ``Resolution`` the pixel coords refer to
        :param mesh_bounds: ``(x0, x1, y0, y1, z_mean)`` of the mesh
        :return: Tuple of (reason key, ground range in metres or ``None``)
        """
        import numpy as np

        x0, x1, y0, y1, z_mean = mesh_bounds
        width, height = input_resolution.width, input_resolution.height

        rot = np.asarray(camera.transform.rotation.matrix33, dtype=float)
        pos = np.asarray(camera.transform.position, dtype=float)
        tan_fov_y = np.tan(np.deg2rad(camera.fovy) / 2.0)

        corners = list(zip(label_coords[0::2], label_coords[1::2]))
        saw_up = False
        best_range = None
        for px, py in corners:
            ndc_x = (2.0 * float(px) / width) - 1.0
            ndc_y = 1.0 - (2.0 * float(py) / height)
            direction = rot @ np.array([
                ndc_x * camera.aspect_ratio * tan_fov_y,
                ndc_y * tan_fov_y,
                -1.0,
            ])
            if direction[2] >= -1e-9:
                # Ray is level or rising — it can never reach the terrain.
                saw_up = True
                continue
            t = (z_mean - pos[2]) / direction[2]
            gx = pos[0] + t * direction[0]
            gy = pos[1] + t * direction[1]
            ground_range = float(np.hypot(gx - pos[0], gy - pos[1]))
            if x0 <= gx <= x1 and y0 <= gy <= y1:
                # Lands inside the footprint: grazing angle, occlusion or a
                # gap between the flat reference plane and the real surface.
                return "other", ground_range
            if best_range is None or ground_range < best_range:
                best_range = ground_range

        if best_range is not None:
            return "beyond_mesh", best_range
        return ("above_horizon", None) if saw_up else ("other", None)

    def run_georeference(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Geo-reference detections using DEM.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import numpy as np

        from pyrr import Vector3, Quaternion
        from trimesh import Trimesh

        from alfspy.core.rendering import Resolution
        from alfspy.render.render import read_gltf, process_render_data, release_all
        from bambi.util.projection_util import label_to_world_coordinates

        # Camera selection for dual-input mode
        camera = config.get("georeference_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        target_folder = config["target_folder"]
        dem_path = config["dem_path"]

        # Set frames folder based on camera selection
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        if log_fn:
            log_fn(f"Loading DEM and {camera_name} poses data...")

        # Load DEM metadata
        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        with open(dem_json_path, 'r') as f:
            dem_json = json.load(f)

        x_offset = dem_json["origin"][0]
        y_offset = dem_json["origin"][1]
        z_offset = dem_json["origin"][2]

        # Detect a common DJI DEM export issue: geographic (degree) coordinates mislabeled
        # as a UTM projected CRS.  When the mesh was built from such a file the origin is
        # in degrees (< 180) instead of UTM metres (> 1000), and mesh vertices are also in
        # degree units — ray casting from UTM camera positions will never intersect the mesh.
        if abs(x_offset) <= 180 and abs(y_offset) <= 90:
            raise RuntimeError(
                "DEM metadata 'origin' appears to be in WGS84 geographic coordinates "
                f"(origin=[{x_offset:.5f}, {y_offset:.5f}, {z_offset:.3f}]) rather than "
                "projected UTM metres.\n\n"
                "This typically means the DEM GeoTIFF was exported by DJI drone software "
                "with geographic (degree) coordinates but incorrectly tagged as a UTM CRS.\n\n"
                "To fix:\n"
                "  1. Delete the existing DEM mesh files (.glb / .json) in the project folder.\n"
                "  2. Set the correct target EPSG in the plugin Parameters tab.\n"
                "  3. Re-run 'Load/Generate DEM' — the plugin now auto-detects WGS84 source "
                "DEMs and reprojects them to UTM before building the mesh.\n"
                "  4. Re-run 'Extract RGB/Thermal Frames' so camera positions are re-computed "
                "relative to the corrected DEM origin.\n"
                "  5. Re-run 'Generate Flight Route' and 'Geo-Reference Detections'."
            )

        # Load poses for selected camera
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)

        # Get input resolution from first extracted frame
        input_resolution = None
        first_image = poses["images"][0]
        first_image_file = first_image.get("imagefile", "")
        if first_image_file:
            first_image_path = os.path.join(frames_folder, first_image_file)
            if os.path.exists(first_image_path):
                import cv2
                img = cv2.imread(first_image_path)
                if img is not None:
                    input_resolution = Resolution(img.shape[1], img.shape[0])
                    if log_fn:
                        log_fn(f"Input resolution from frame: {img.shape[1]}x{img.shape[0]}")

        # Fallback to config or default
        if input_resolution is None:
            res_width = config.get("input_resolution_width", 640)
            res_height = config.get("input_resolution_height", 512)
            input_resolution = Resolution(res_width, res_height)
            if log_fn:
                log_fn(f"Using configured resolution: {res_width}x{res_height}")

        # Load detections (camera-specific folder)
        detections_file = os.path.join(target_folder, f"detections_{camera_suffix}", "detections.txt")
        if not os.path.exists(detections_file):
            raise FileNotFoundError("Detections file not found")

        detections = []
        with open(detections_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    # Parse optional metadata headers embedded as comments
                    if line.startswith('# video_size'):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                det_w, det_h = int(parts[2]), int(parts[3])
                                input_resolution = Resolution(det_w, det_h)
                                if log_fn:
                                    log_fn(
                                        f"Detection frame size from detections.txt: {det_w}x{det_h}"
                                    )
                            except ValueError:
                                pass
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    detections.append({
                        "frame": int(parts[0]),
                        "x1": float(parts[1]),
                        "y1": float(parts[2]),
                        "x2": float(parts[3]),
                        "y2": float(parts[4]),
                        "confidence": float(parts[5]),
                        "class_id": int(parts[6]) if len(parts) > 6 else 0
                    })

        if log_fn:
            log_fn(f"Loaded {len(detections)} detections to geo-reference")

        if progress_fn:
            progress_fn(10)

        # Create output folder (camera-specific)
        georef_folder = os.path.join(target_folder, f"georeferenced_{camera_suffix}")
        os.makedirs(georef_folder, exist_ok=True)

        # Load DEM mesh
        if log_fn:
            log_fn("Loading DEM mesh...")

        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None

        try:
            mesh_data, texture_data = read_gltf(dem_path)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            n_faces = len(tri_mesh.faces)
            if n_faces > 20_000_000:
                if log_fn:
                    log_fn(
                        f"Warning: DEM mesh has {n_faces:,} triangles. "
                        f"Ray casting may fail with an out-of-memory error. "
                        f"Delete the .glb/.json and regenerate the DEM with a higher "
                        f"simplification factor (recommended: 4 or more) to reduce mesh size."
                    )
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)

            if progress_fn:
                progress_fn(30)

            # Process each detection
            georeferenced = []
            total_dets = len(detections)

            # Diagnostics for detections whose rays never reach the DEM.  Without
            # this the step just reports a smaller number with no explanation.
            miss_stats = {"above_horizon": 0, "beyond_mesh": 0, "other": 0,
                          "no_pose": 0}
            miss_ranges = []
            _mv = np.asarray(tri_mesh.vertices)
            _mesh_x0, _mesh_x1 = float(_mv[:, 0].min()), float(_mv[:, 0].max())
            _mesh_y0, _mesh_y1 = float(_mv[:, 1].min()), float(_mv[:, 1].max())
            _mesh_z = float(_mv[:, 2].mean())

            for idx, det in enumerate(detections):
                # Check for cancellation
                if cancel_check and cancel_check():
                    if log_fn:
                        log_fn("Geo-referencing cancelled by user")
                    raise CancelledException("Geo-referencing cancelled")

                frame_idx = det["frame"]

                if frame_idx >= len(poses["images"]):
                    miss_stats["no_pose"] += 1
                    continue

                # Get frame-specific correction factors
                correction = self.get_correction_for_frame(frame_idx, config)
                translation = correction["translation"]
                rotation = correction["rotation"]
                cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
                cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

                image_metadata = poses["images"][frame_idx]

                # Get camera for this frame
                fovy = image_metadata.get("fovy", [50])
                if isinstance(fovy, list):
                    fovy = fovy[0]
                position = Vector3(image_metadata["location"])
                rot = image_metadata["rotation"]
                rotation_eulers = (Vector3(
                    [np.deg2rad(val % 360.0) for val in rot]) - cor_rotation_eulers) * -1
                position += cor_translation
                rotation_quat = Quaternion.from_eulers(rotation_eulers)

                # Create camera-like object for projection
                from alfspy.core.rendering import Camera
                aspect_ratio = input_resolution.width / input_resolution.height
                camera = Camera(fovy=fovy, aspect_ratio=aspect_ratio, position=position, rotation=rotation_quat)

                # Project bounding box corners
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                label_coords = [x1, y1, x2, y1, x2, y2, x1, y2]

                try:
                    world_coords = label_to_world_coordinates(
                        label_coords, input_resolution, tri_mesh, camera
                    )

                    if len(world_coords) > 0:
                        xx = world_coords[:, 0] + x_offset
                        yy = world_coords[:, 1] + y_offset
                        zz = world_coords[:, 2] + z_offset

                        georeferenced.append({
                            "frame": frame_idx,
                            "x1": min(xx), "y1": min(yy), "z1": min(zz),
                            "x2": max(xx), "y2": max(yy), "z2": max(zz),
                            "confidence": det["confidence"],
                            "class_id": det["class_id"]
                        })
                    else:
                        reason, ground_range = self._classify_projection_miss(
                            camera, label_coords, input_resolution,
                            (_mesh_x0, _mesh_x1, _mesh_y0, _mesh_y1, _mesh_z),
                        )
                        miss_stats[reason] += 1
                        if ground_range is not None:
                            miss_ranges.append(ground_range)
                except Exception as e:
                    if log_fn:
                        log_fn(f"Warning: Could not geo-reference detection at frame {frame_idx}: {e}")
                    continue

                if progress_fn and idx % 50 == 0:
                    progress = 30 + int((idx / total_dets) * 60)
                    progress_fn(min(progress, 95))

        finally:
            if ctx:
                release_all(ctx)
            if mesh_data is not None:
                del mesh_data
            if texture_data is not None:
                del texture_data
            if tri_mesh is not None:
                del tri_mesh

        # Write georeferenced results
        output_file = os.path.join(georef_folder, "georeferenced.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# idx frame min_x min_y min_z max_x max_y max_z confidence class_id\n")
            for idx, det in enumerate(georeferenced):
                f.write(f"{idx} {det['frame']} {det['x1']:.6f} {det['y1']:.6f} {det['z1']:.6f} "
                        f"{det['x2']:.6f} {det['y2']:.6f} {det['z2']:.6f} "
                        f"{det['confidence']:.4f} {det['class_id']}\n")

        if log_fn:
            log_fn(f"Geo-referencing complete: {len(georeferenced)} detections")
            self._log_projection_miss_summary(
                miss_stats, miss_ranges, len(detections), log_fn)

        if progress_fn:
            progress_fn(100)

    @staticmethod
    def _log_projection_miss_summary(miss_stats, miss_ranges, total_dets, log_fn):
        """Report how many detections could not be placed on the DEM, and why."""
        import numpy as np

        dropped = sum(miss_stats.values())
        if not dropped or not total_dets:
            return

        log_fn(f"{dropped} of {total_dets} detections "
               f"({100.0 * dropped / total_dets:.1f}%) could not be placed on the DEM:")

        if miss_stats["above_horizon"]:
            log_fn(f"  - {miss_stats['above_horizon']} above the horizon — the camera "
                   f"was near-horizontal, so these pixels show sky and can never be "
                   f"geo-referenced.")
        if miss_stats["beyond_mesh"]:
            msg = (f"  - {miss_stats['beyond_mesh']} beyond the DEM edge — the view "
                   f"reached ground that the DEM does not cover.")
            if miss_ranges:
                msg += (f" Ground range: median "
                        f"{float(np.median(miss_ranges)):.0f} m, 90th pct "
                        f"{float(np.percentile(miss_ranges, 90)):.0f} m.")
            log_fn(msg)
            log_fn("    Re-download the DEM with a larger 'Padding' value "
                   "(Parameters tab) to cover the oblique field of view.")
        if miss_stats["other"]:
            log_fn(f"  - {miss_stats['other']} at grazing angles or hidden behind "
                   f"terrain.")
        if miss_stats["no_pose"]:
            log_fn(f"  - {miss_stats['no_pose']} reference frames that have no pose "
                   f"entry (detection ran on more frames than were extracted).")

    def run_trex_import(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Import TRex .npz tracklets and geo-reference them against the DEM.

        Reads TRex tracklet files, optionally undistorts bounding-box coordinates from
        raw video pixel space to undistorted square-frame space, then geo-references
        each detection against the DEM mesh using the per-frame poses.  Outputs are
        written into the same folder structure as the regular detection pipeline so that
        all downstream steps (Add Tracks to QGIS, perpendicular distance, etc.) work
        without modification.
        """
        import glob as _glob
        import cv2
        import numpy as np
        from pathlib import Path
        from pyrr import Vector3, Quaternion
        from trimesh import Trimesh
        from alfspy.core.rendering import Camera, Resolution
        from alfspy.render.render import read_gltf, process_render_data
        from bambi.util.projection_util import label_to_world_coordinates

        camera = config.get("tracking_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"
        already_undistorted = config.get("trex_already_undistorted", False)
        npz_dir = config.get("trex_npz_dir", "")
        target_folder = config["target_folder"]
        dem_path = config["dem_path"]

        # ---- Step 1: read .npz tracklets ----------------------------------------
        if log_fn:
            log_fn(f"Reading TRex tracklets from: {npz_dir}")

        npz_files = sorted(_glob.glob(os.path.join(npz_dir, "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in: {npz_dir}")

        detections = []
        raw_size = None

        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)

            if "video_size" in data and raw_size is None:
                vs = data["video_size"]
                raw_size = (int(vs[0]), int(vs[1]))

            if "id" in data and len(data["id"]):
                track_id = int(np.asarray(data["id"]).ravel()[0])
            else:
                stem = Path(npz_path).stem
                track_id = int(stem.split("id")[-1]) if "id" in stem else len(detections)

            frames = np.asarray(data["frame"]).astype(int)
            conf = np.nan_to_num(np.asarray(data["detection_p"], dtype=float), nan=0.0)
            cls = np.nan_to_num(np.asarray(data["detection_class"], dtype=float), nan=0.0).astype(int)

            pose_keys = []
            for key in data.keys():
                if key.startswith("poseX"):
                    suffix = key[len("poseX"):]
                    if suffix.isdigit() and f"poseY{suffix}" in data:
                        pose_keys.append(int(suffix))
            pose_keys = sorted(pose_keys)

            if not pose_keys:
                if log_fn:
                    log_fn(f"  WARNING: {Path(npz_path).name} has no pose points, skipped")
                continue

            pose_x = np.stack([np.asarray(data[f"poseX{i}"], dtype=float) for i in pose_keys], axis=1)
            pose_y = np.stack([np.asarray(data[f"poseY{i}"], dtype=float) for i in pose_keys], axis=1)

            kept = 0
            for i in range(len(frames)):
                xs = pose_x[i]
                ys = pose_y[i]
                mask = np.isfinite(xs) & np.isfinite(ys)
                if not mask.any():
                    continue
                xs, ys = xs[mask], ys[mask]
                detections.append({
                    "frame": int(frames[i]),
                    "track_id": track_id,
                    "x1": float(xs.min()), "y1": float(ys.min()),
                    "x2": float(xs.max()), "y2": float(ys.max()),
                    "confidence": float(conf[i]) if i < len(conf) else 1.0,
                    "class_id": int(cls[i]) if i < len(cls) else 0,
                })
                kept += 1
            if log_fn:
                log_fn(f"  {Path(npz_path).name}: track {track_id}, {kept} detections")

        detections.sort(key=lambda d: (d["frame"], d["track_id"]))
        if log_fn:
            log_fn(f"Total: {len(detections)} detections, raw video size: {raw_size}")

        if progress_fn:
            progress_fn(10)

        # ---- Step 2: write detections.txt ----------------------------------------
        det_folder = os.path.join(target_folder, f"detections_{camera_suffix}")
        os.makedirs(det_folder, exist_ok=True)
        det_file = os.path.join(det_folder, "detections.txt")
        with open(det_file, "w", encoding="utf-8") as f:
            f.write("# frame x1 y1 x2 y2 confidence class_id\n")
            for d in detections:
                f.write(f"{d['frame']} {d['x1']:.2f} {d['y1']:.2f} {d['x2']:.2f} {d['y2']:.2f} "
                        f"{d['confidence']:.4f} {d['class_id']}\n")
        if log_fn:
            log_fn(f"Wrote detections.txt ({len(detections)} rows)")

        # ---- Step 3: build undistorter (if labels are in raw pixel space) ---------
        # The TRex detections live in raw video pixel space, while the poses/cameras
        # were built for the *extracted* frames.  BAMBI's frame extraction either keeps
        # the full aspect ratio (preserve_aspect_ratio=True, e.g. 5120x2700) or forces a
        # square frame (min(w,h)).  We must map detections into that exact pixel space and
        # project with that exact resolution, otherwise the geo-referenced boxes are
        # offset/compressed relative to the GeoTIFFs and alfs.
        undistorter = None
        input_resolution = None

        # Determine the extracted-frame resolution that the poses correspond to.
        # Prefer an actual extracted frame; fall back to the mask (same dimensions as
        # the extracted frames); finally fall back to the preserve_aspect_ratio config.
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")
        extraction_size = None  # (width, height)
        if os.path.isdir(frames_folder):
            for fn in sorted(os.listdir(frames_folder)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    probe = cv2.imread(os.path.join(frames_folder, fn))
                    if probe is not None:
                        extraction_size = (probe.shape[1], probe.shape[0])
                        break
        if extraction_size is None:
            mask_candidates = []
            try:
                with open(os.path.join(target_folder, f"poses_{camera_suffix}.json"), "r", encoding="utf-8") as f:
                    mask_candidates.append(json.load(f).get("mask"))
            except Exception:  # nosec B110
                pass
            mask_candidates += [f"mask_{'T' if camera == 'T' else 'W'}.png", "mask.png"]
            for mc in mask_candidates:
                if not mc:
                    continue
                mpath = mc if os.path.isabs(mc) else os.path.join(target_folder, mc)
                if os.path.isfile(mpath):
                    mimg = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
                    if mimg is not None:
                        extraction_size = (mimg.shape[1], mimg.shape[0])
                        break
        if extraction_size is not None and log_fn:
            log_fn(f"Extracted-frame resolution: {extraction_size[0]}x{extraction_size[1]}")

        if not already_undistorted:
            if raw_size is None or raw_size == (0, 0):
                raise ValueError(
                    "Raw video size unknown — cannot undistort.  "
                    "Ensure the .npz files contain a 'video_size' field, or enable "
                    "'Labels already in undistorted frame space'."
                )

            calib_data = None
            if camera == "T":
                calib_data = config.get("thermal_calibration_data") or config.get("thermal_photo_calibration_data")
                calib_path = config.get("thermal_calibration_path") or config.get("thermal_photo_calibration_path", "")
            else:
                calib_data = config.get("rgb_calibration_data") or config.get("rgb_photo_calibration_data")
                calib_path = config.get("rgb_calibration_path") or config.get("rgb_photo_calibration_path", "")

            if calib_data is None:
                if not calib_path or not os.path.isfile(calib_path):
                    raise ValueError(
                        f"No {camera_name} calibration provided.  "
                        "Select a preset or specify a calibration file path in the Input tab."
                    )
                with open(calib_path, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)

            mtx = np.asarray(calib_data["mtx"], dtype=float)
            dist = np.asarray(calib_data["dist"], dtype=float)
            w, h = raw_size

            # Undistortion target size = extracted-frame size, replicating BAMBI's
            # CalibratedVideoFrameAccessor.prepare_undistort (alpha=0.5, centred principal
            # point, square FOV forced only when the frame is square).
            if extraction_size is not None:
                new_size = extraction_size
            elif config.get("preserve_aspect_ratio", False):
                new_size = (w, h)
            else:
                wh = min(w, h)
                new_size = (wh, wh)

            ncm, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.5, new_size,
                                                   centerPrincipalPoint=True)
            if new_size[0] == new_size[1]:
                fxy = max(ncm[0, 0], ncm[1, 1])
                ncm[0, 0] = ncm[1, 1] = fxy

            class _Undistorter:
                def __init__(self, _mtx, _dist, _ncm, _new_size):
                    self._mtx = _mtx
                    self._dist = _dist
                    self._ncm = _ncm
                    self.new_size = _new_size

                def points(self, pts_xy):
                    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
                    out = cv2.undistortPoints(pts, self._mtx, self._dist, P=self._ncm)
                    return out.reshape(-1, 2)

            undistorter = _Undistorter(mtx, dist, ncm, new_size)
            input_resolution = Resolution(new_size[0], new_size[1])
            if log_fn:
                log_fn(f"Undistorter ready: {raw_size} → {new_size}")
        else:
            # Labels already in extracted-frame space: project with that resolution directly.
            if extraction_size is not None:
                input_resolution = Resolution(extraction_size[0], extraction_size[1])
            if input_resolution is None:
                raise ValueError(
                    "Could not determine the extracted-frame resolution from frames or mask.  "
                    "Ensure frame extraction has been completed (frames_{suffix}/ or mask present)."
                )
            if log_fn:
                log_fn(f"Input resolution from extracted frames: {input_resolution.width}x{input_resolution.height}")

        if progress_fn:
            progress_fn(20)

        # ---- Step 4: write pixel-space tracks ------------------------------------
        tracks_pixel_folder = os.path.join(target_folder, f"tracks_pixel_{camera_suffix}")
        os.makedirs(tracks_pixel_folder, exist_ok=True)
        pixel_tracks_file = os.path.join(tracks_pixel_folder, "tracks_pixel.csv")
        with open(pixel_tracks_file, "w", encoding="utf-8") as f:
            for d in detections:
                f.write(f"{d['frame']:08d},{d['track_id']},{d['x1']:.6f},{d['y1']:.6f},"
                        f"{d['x2']:.6f},{d['y2']:.6f},{d['confidence']:.6f},{d['class_id']},0\n")
        if log_fn:
            log_fn(f"Wrote pixel-space tracks: {pixel_tracks_file}")

        # Always write tracks_pixel_undistorted.csv (extracted-frame pixel space).
        # When an undistorter is active the raw tracks_pixel.csv above is in raw
        # video space; here we map to the undistorted frame space that matches the
        # extracted frames. When the labels are already in extracted-frame space
        # (undistorter is None) the boxes are used verbatim, so downstream tools
        # (e.g. the Video Creator) can always rely on this file for overlays.
        pixel_und_file = os.path.join(tracks_pixel_folder, "tracks_pixel_undistorted.csv")
        with open(pixel_und_file, "w", encoding="utf-8") as f:
            for d in detections:
                if undistorter is not None:
                    corners = np.array(
                        [[d["x1"], d["y1"]], [d["x2"], d["y1"]],
                         [d["x2"], d["y2"]], [d["x1"], d["y2"]]], dtype=np.float32
                    )
                    und = undistorter.points(corners)
                    x1, y1 = float(und[:, 0].min()), float(und[:, 1].min())
                    x2, y2 = float(und[:, 0].max()), float(und[:, 1].max())
                else:
                    x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                f.write(f"{d['frame']:08d},{d['track_id']},"
                        f"{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f},"
                        f"{d['confidence']:.6f},{d['class_id']},0\n")
        if log_fn:
            log_fn(f"Wrote undistorted pixel-space tracks: {pixel_und_file}")

        # ---- Step 5: load DEM + poses -------------------------------------------
        if log_fn:
            log_fn("Loading DEM and poses...")

        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        with open(dem_json_path, "r", encoding="utf-8") as f:
            dem_meta = json.load(f)

        x_off = float(dem_meta["origin"][0])
        y_off = float(dem_meta["origin"][1])
        z_off = float(dem_meta["origin"][2])

        if abs(x_off) <= 180 and abs(y_off) <= 90:
            raise RuntimeError(
                "DEM metadata 'origin' appears to be in WGS84 geographic coordinates "
                f"(origin=[{x_off:.5f}, {y_off:.5f}, {z_off:.3f}]) rather than projected UTM metres.\n\n"
                "Delete the existing DEM .glb/.json files, set the correct EPSG, and re-run 'Load/Generate DEM'."
            )

        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        with open(poses_file, "r", encoding="utf-8") as f:
            poses = json.load(f)

        n_frames = len(poses["images"])
        if log_fn:
            log_fn(f"DEM origin: ({x_off:.2f}, {y_off:.2f}, {z_off:.2f})  poses: {n_frames} frames")

        if progress_fn:
            progress_fn(30)

        # ---- Step 6: geo-reference -----------------------------------------------
        georef_folder = os.path.join(target_folder, f"georeferenced_{camera_suffix}")
        tracks_folder = os.path.join(target_folder, f"tracks_{camera_suffix}")
        os.makedirs(georef_folder, exist_ok=True)
        os.makedirs(tracks_folder, exist_ok=True)
        georef_file = os.path.join(georef_folder, "georeferenced.txt")
        tracks_file = os.path.join(tracks_folder, "tracks.csv")

        mesh_data = None
        texture_data = None
        tri_mesh = None

        try:
            mesh_data, texture_data = read_gltf(dem_path)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)

            camera_cache: Dict[int, Camera] = {}
            n_ok = 0
            n_fail = 0
            total = len(detections)

            with open(georef_file, "w", encoding="utf-8") as gf, \
                    open(tracks_file, "w", encoding="utf-8") as tf:
                gf.write("# idx frame min_x min_y min_z max_x max_y max_z confidence class_id\n")

                for idx, d in enumerate(detections):
                    if cancel_check and cancel_check():
                        if log_fn:
                            log_fn("TRex import cancelled")
                        raise CancelledException("TRex import cancelled")

                    frame_idx = d["frame"]
                    if frame_idx >= n_frames:
                        gf.write(f"{idx} {frame_idx} -1 -1 -1 -1 -1 -1 "
                                 f"{d['confidence']:.4f} {d['class_id']}\n")
                        n_fail += 1
                        continue

                    corners = np.array(
                        [[d["x1"], d["y1"]], [d["x2"], d["y1"]],
                         [d["x2"], d["y2"]], [d["x1"], d["y2"]]], dtype=np.float32
                    )
                    if undistorter is not None:
                        corners = undistorter.points(corners)

                    if frame_idx not in camera_cache:
                        correction = self.get_correction_for_frame(frame_idx, config)
                        trans = correction["translation"]
                        rot = correction["rotation"]
                        cor_rot = Vector3([rot["x"], rot["y"], rot["z"]], dtype="f4")
                        cor_trans = Vector3([trans["x"], trans["y"], trans["z"]], dtype="f4")

                        img_data = poses["images"][frame_idx]
                        fovy = img_data.get("fovy", [50])
                        if isinstance(fovy, list):
                            fovy = fovy[0]
                        position = Vector3(img_data["location"]) + cor_trans
                        rotation_eulers = (
                            Vector3([np.deg2rad(v % 360.0) for v in img_data["rotation"]]) - cor_rot
                        ) * -1
                        rotation_q = Quaternion.from_eulers(rotation_eulers)
                        aspect = input_resolution.width / input_resolution.height
                        camera_cache[frame_idx] = Camera(
                            fovy=fovy, aspect_ratio=aspect,
                            position=position, rotation=rotation_q
                        )

                    cam = camera_cache[frame_idx]
                    poly = corners.reshape(-1).tolist()

                    try:
                        world = label_to_world_coordinates(poly, input_resolution, tri_mesh, cam)
                    except Exception:
                        world = None

                    if world is None or len(world) == 0:
                        gf.write(f"{idx} {frame_idx} -1 -1 -1 -1 -1 -1 "
                                 f"{d['confidence']:.4f} {d['class_id']}\n")
                        n_fail += 1
                        continue

                    xx = world[:, 0] + x_off
                    yy = world[:, 1] + y_off
                    zz = world[:, 2] + z_off
                    min_x, max_x = float(xx.min()), float(xx.max())
                    min_y, max_y = float(yy.min()), float(yy.max())
                    min_z, max_z = float(zz.min()), float(zz.max())

                    gf.write(f"{idx} {frame_idx} {min_x:.6f} {min_y:.6f} {min_z:.6f} "
                             f"{max_x:.6f} {max_y:.6f} {max_z:.6f} "
                             f"{d['confidence']:.4f} {d['class_id']}\n")
                    tf.write(f"{frame_idx:08d},{d['track_id']},"
                             f"{min_x:.6f},{min_y:.6f},{min_z:.6f},"
                             f"{max_x:.6f},{max_y:.6f},{max_z:.6f},"
                             f"{d['confidence']:.6f},{d['class_id']},0\n")
                    n_ok += 1

                    if progress_fn and idx % 100 == 0:
                        progress_fn(30 + int((idx / total) * 65))

        finally:
            if mesh_data is not None:
                del mesh_data
            if texture_data is not None:
                del texture_data
            if tri_mesh is not None:
                del tri_mesh

        if log_fn:
            log_fn(f"Geo-referenced {n_ok} detections ({n_fail} could not be projected).")
            log_fn(f"  → {georef_file}")
            log_fn(f"  → {tracks_file}")

        if progress_fn:
            progress_fn(100)

    def run_calculate_fov(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Calculate and save Field of View (FoV) polygons for each frame.

        This geo-references the image corners for each frame to create a polygon
        showing the camera's field of view on the ground.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import numpy as np
        import cv2

        from pyrr import Vector3, Quaternion
        from trimesh import Trimesh

        from alfspy.core.rendering import Resolution
        from alfspy.render.render import read_gltf, process_render_data, release_all

        camera = config.get("fov_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        target_folder = config["target_folder"]
        dem_path = config["dem_path"]

        # FoV mask options
        use_fov_mask = config.get("use_fov_mask", False)
        mask_path = config.get("fov_mask_path", "")
        mask_simplify_epsilon = config.get("mask_simplify_epsilon", 2.0)

        # Frame filter options (use start/end frame like alfs)
        use_all_frames = config.get("fov_use_all_frames", True)
        start_frame = config.get("fov_start_frame", 0)
        end_frame = config.get("fov_end_frame", 999999)
        sample_rate = config.get("fov_sample_rate", 1)

        # Set frames folder based on camera selection
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        if log_fn:
            log_fn(f"Starting FoV calculation for {camera_name} frames...")
            log_fn("Loading DEM and poses data...")

        # Load DEM metadata
        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        with open(dem_json_path, 'r') as f:
            dem_json = json.load(f)

        x_offset = dem_json["origin"][0]
        y_offset = dem_json["origin"][1]
        z_offset = dem_json["origin"][2]

        # Load poses for selected camera
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)

        # Get input resolution from first extracted frame
        input_resolution = None
        first_image_file = poses["images"][0].get("imagefile", "")
        if first_image_file:
            first_image_path = os.path.join(frames_folder, first_image_file)
            if os.path.exists(first_image_path):
                img = cv2.imread(first_image_path)
                if img is not None:
                    input_resolution = Resolution(img.shape[1], img.shape[0])

        # Fallback to config or default
        if input_resolution is None:
            res_width = config.get("input_resolution_width", 640)
            res_height = config.get("input_resolution_height", 512)
            input_resolution = Resolution(res_width, res_height)
        aspect_ratio = input_resolution.width / input_resolution.height

        if log_fn:
            log_fn(f"Input resolution: {input_resolution.width}x{input_resolution.height}")

        if progress_fn:
            progress_fn(10)

        # Create output folder (camera-specific)
        fov_folder = os.path.join(target_folder, f"fov_{camera_suffix}")
        os.makedirs(fov_folder, exist_ok=True)

        # Load or create FoV mask polygon
        mask_polygon = None
        if use_fov_mask and mask_path and os.path.exists(mask_path):
            if log_fn:
                log_fn(f"Loading FoV mask from {mask_path}...")
            mask_polygon = self._extract_mask_polygon(mask_path, mask_simplify_epsilon, log_fn)
            if mask_polygon:
                if log_fn:
                    log_fn(f"Loaded mask with {len(mask_polygon)} polygon points")

        # Fall back to the same mask the alfs / GeoTIFF export use so the
        # FoV footprint matches the rendered products.  Those pipelines always
        # constrain the frame to mask_<CAM>.png (or the poses "mask"); projecting
        # the full frame corners here instead made the FoV polygons overhang the
        # rendered content (the mask trims the invalid border of undistorted
        # frames).  Set use_fov_mask + fov_mask_path to override with a custom mask.
        if not mask_polygon:
            camera_specific_mask = os.path.join(target_folder, f"mask_{camera}.png")
            mask_filename = poses.get("mask")
            poses_mask = os.path.join(target_folder, mask_filename) if mask_filename else None
            default_mask = None
            if os.path.exists(camera_specific_mask):
                default_mask = camera_specific_mask
            elif poses_mask and os.path.exists(poses_mask):
                default_mask = poses_mask
            if default_mask:
                if log_fn:
                    log_fn(f"Using render mask for FoV footprint: {os.path.basename(default_mask)}")
                mask_polygon = self._extract_mask_polygon(default_mask, mask_simplify_epsilon, log_fn)

        # If still no mask, use image corners
        if not mask_polygon:
            if log_fn:
                log_fn("Using image corners as FoV polygon...")
            w, h = input_resolution.width, input_resolution.height
            # Define corners with more points for better ground projection
            mask_polygon = [
                (0, 0), (w / 2, 0), (w, 0),
                (w, h / 2), (w, h),
                (w / 2, h), (0, h),
                (0, h / 2)
            ]

        if log_fn:
            log_fn("Loading DEM mesh...")

        # Load DEM mesh
        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        try:
            mesh_data, texture_data = read_gltf(dem_path)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)

            if progress_fn:
                progress_fn(20)

            # Process each frame
            total_frames = len(poses["images"])

            # Apply frame filters (start/end frame, then sample rate)
            if use_all_frames:
                frame_indices = list(range(total_frames))
                if log_fn:
                    log_fn("Frame range: All frames")
            else:
                # Clamp end_frame to valid range
                end_frame_clamped = min(end_frame, total_frames - 1)
                frame_indices = list(range(start_frame, end_frame_clamped + 1))
                if log_fn:
                    log_fn(f"Frame range: {start_frame} to {end_frame_clamped}")

            # Apply sample rate
            if sample_rate > 1:
                frame_indices = frame_indices[::sample_rate]
                if log_fn:
                    log_fn(f"Sampling every {sample_rate}th frame")

            if log_fn:
                log_fn(f"Calculating FoV for {len(frame_indices)} frames...")

            # Output file for FoV polygons
            output_file = os.path.join(fov_folder, "fov_polygons.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# FoV polygon georeferenced data\n")
                f.write("# Format: frame_idx num_points x1 y1 z1 x2 y2 z2 ...\n")

                for i, frame_idx in enumerate(frame_indices):
                    # Check for cancellation
                    if cancel_check and cancel_check():
                        if log_fn:
                            log_fn("FoV calculation cancelled by user")
                        raise CancelledException("FoV calculation cancelled")

                    # Get frame-specific correction factors
                    correction = self.get_correction_for_frame(frame_idx, config)
                    translation = correction["translation"]
                    rotation = correction["rotation"]
                    cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
                    cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

                    image_metadata = poses["images"][frame_idx]
                    # Get camera for this frame
                    fovy = image_metadata.get("fovy", [50])
                    if isinstance(fovy, list):
                        fovy = fovy[0]
                    position = Vector3(image_metadata["location"])
                    rot = image_metadata["rotation"]
                    # Apply the rotation correction at 1× — the same amount alfspy's
                    # renderer applies via CtxShot.get_correction() — so the FoV
                    # footprint matches the rendered alfs and GeoTIFF content.
                    rotation_eulers = (Vector3(
                        [np.deg2rad(val % 360.0) for val in rot]) - cor_rotation_eulers) * -1
                    position += cor_translation
                    rotation_quat = Quaternion.from_eulers(rotation_eulers)

                    # Create camera for projection
                    from alfspy.core.rendering import Camera
                    camera = Camera(fovy=fovy, aspect_ratio=aspect_ratio, position=position, rotation=rotation_quat)

                    # Georeference the mask polygon points
                    georef_points = self._georeference_polygon(
                        mask_polygon, input_resolution, tri_mesh, camera,
                        x_offset, y_offset, z_offset
                    )

                    # Filter out None points
                    valid_points = [p for p in georef_points if p is not None]

                    if len(valid_points) > 0:
                        coords_str = " ".join(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in valid_points)
                        f.write(f"{frame_idx} {len(valid_points)} {coords_str}\n")
                    else:
                        f.write(f"{frame_idx} 0\n")

                    if progress_fn and i % 50 == 0:
                        progress = 20 + int((i / len(frame_indices)) * 75)
                        progress_fn(min(progress, 95))

                    if log_fn and (i + 1) % 100 == 0:
                        log_fn(f"Processed {i + 1}/{len(frame_indices)} frames...")

        finally:
            if ctx:
                release_all(ctx)
            if mesh_data is not None:
                del mesh_data
            if texture_data is not None:
                del texture_data
            if tri_mesh is not None:
                del tri_mesh

        if log_fn:
            log_fn(f"FoV calculation complete. Output: {output_file}")

        if progress_fn:
            progress_fn(100)

    def _extract_mask_polygon(self, mask_path: str, simplify_epsilon: float = 2.0, log_fn=None) -> Optional[List]:
        """Extract polygon from binary mask image.

        :param mask_path: Path to binary mask PNG image
        :param simplify_epsilon: Epsilon for polygon simplification
        :param log_fn: Logging callback function
        :return: List of (x, y) pixel coordinates or None
        """
        import cv2

        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                if log_fn:
                    log_fn(f"Warning: Could not load mask image: {mask_path}")
                return None

            # Threshold to ensure binary mask
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return None

            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Simplify the polygon
            if simplify_epsilon > 0:
                largest_contour = cv2.approxPolyDP(largest_contour, simplify_epsilon, True)

            # Convert to list of (x, y) tuples
            polygon_points = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]

            return polygon_points

        except Exception as e:
            if log_fn:
                log_fn(f"Warning: Error extracting mask polygon: {e}")
            return None

    def _georeference_polygon(
            self,
            polygon_points: List,
            input_resolution,
            tri_mesh,
            camera,
            x_offset: float,
            y_offset: float,
            z_offset: float
    ) -> List:
        """Georeference polygon points from pixel to world coordinates.

        :param polygon_points: List of (x, y) pixel coordinates
        :param input_resolution: Resolution object for input images
        :param tri_mesh: Trimesh object for ray casting
        :param camera: Camera object for the current frame
        :param x_offset, y_offset, z_offset: DEM origin offsets
        :return: List of (x, y, z) world coordinates or None for failed points
        """
        import numpy as np
        from bambi.util.projection_util import label_to_world_coordinates

        georeferenced_points = []

        for px, py in polygon_points:
            # Create a small bounding box around the point
            point_coords = [px, py, px + 1, py, px + 1, py + 1, px, py + 1]

            try:
                world_coordinates = label_to_world_coordinates(
                    point_coords, input_resolution, tri_mesh, camera
                )

                if len(world_coordinates) == 0:
                    georeferenced_points.append(None)
                    continue

                # Take the mean of projected points
                xx = world_coordinates[:, 0] + x_offset
                yy = world_coordinates[:, 1] + y_offset
                zz = world_coordinates[:, 2] + z_offset

                georeferenced_points.append((float(np.mean(xx)), float(np.mean(yy)), float(np.mean(zz))))

            except Exception:
                georeferenced_points.append(None)

        return georeferenced_points

    def run_tracking(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Run tracking on geo-referenced detections.

        Supports multiple tracking backends:
        - Built-in: Default Hungarian IoU tracker
        - BoxMOT: Standard BoxMOT trackers (DeepOCSORT, BoTSORT, etc.)
        - GeoRef: Geo-referenced trackers (GeoNative, GeoHybrid)

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        from .tracker_manager import get_tracker_manager

        tracker_id = config.get("tracker_id", "builtin")

        if log_fn:
            log_fn(f"Starting tracking with backend: {tracker_id}")

        # Use the shared global model dir so Re-ID weights are downloaded once
        # and reused across projects.
        tracker_manager = get_tracker_manager(self._get_default_model_dir())

        # Check which backend to use
        if tracker_id == "builtin":
            # Use existing built-in tracking implementation
            self._run_builtin_tracking(config, progress_fn, log_fn)
        else:
            # Use advanced tracker (BoxMOT or GeoRef)
            self._run_advanced_tracking(config, tracker_manager, progress_fn, log_fn)

        # Ensure a pixel-space tracks file exists regardless of backend, so
        # downstream tools (e.g. the Video Creator) have a consistent
        # tracks_pixel.csv like the advanced / TRex pipelines produce.
        camera = config.get("tracking_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        try:
            self._write_pixel_tracks_from_geo(config, camera_suffix, log_fn)
        except Exception as e:  # noqa: BLE001 - best-effort, never fail tracking
            if log_fn:
                log_fn(f"Warning: could not create pixel tracks file: {e}")

    def _write_pixel_tracks_from_geo(self, config: Dict[str, Any], camera_suffix: str,
                                     log_fn=None):
        """Create ``tracks_{cam}/tracks_pixel.csv`` from the geo-referenced tracks.

        The built-in tracker runs in geo space and only writes ``tracks.csv``.
        To keep the output consistent with the advanced / TRex pipelines (which
        emit a pixel-space tracks file), derive each track's pixel bounding box
        by matching its geo-referenced box back to the pixel detection: within a
        frame, ``detections.txt`` and ``georeferenced.txt`` share the same order,
        so the k-th pixel box maps to the k-th geo box, whose coordinates then
        look up the assigned track id in ``tracks.csv``. This is independent of
        the (frame, track_id) re-sorting applied when writing tracks.csv.

        No-op when a pixel tracks file already exists (advanced backend).
        """
        from .core.track_export import write_pixel_tracks_from_geo
        write_pixel_tracks_from_geo(
            config["target_folder"], camera_suffix, log_fn=log_fn)

    def _run_builtin_tracking(self, config: Dict[str, Any], progress_fn=None, log_fn=None):
        """Run the built-in Hungarian IoU tracker.

        This is the original tracking implementation.
        """
        from collections import defaultdict
        from dataclasses import dataclass
        from enum import Enum
        from typing import List, Tuple

        from scipy.optimize import linear_sum_assignment

        # Import tracking classes from bambi or define locally
        @dataclass
        class Detection:
            source_id: int
            frame: int
            x1: float
            y1: float
            z1: float
            x2: float
            y2: float
            z2: float
            conf: float
            cls: int
            interpolated: int = 0

        @dataclass
        class Track:
            tid: int
            cls: int
            x1: float
            y1: float
            z1: float
            x2: float
            y2: float
            z2: float
            last_frame: int
            age: int = 0
            hits: int = 0

        class TrackerMode(Enum):
            GREEDY = 1
            HUNGARIAN = 2
            CENTER = 3
            HUNGARIAN_CENTER = 4

        def iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0.0:
                return 0.0
            a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            denom = a_area + b_area - inter
            return inter / denom if denom > 0 else 0.0

        target_folder = config["target_folder"]
        iou_threshold = config.get("iou_threshold", 0.3)
        max_age = config.get("max_age", -1)
        tracker_mode_str = config.get("tracker_mode", "HUNGARIAN")
        class_aware = config.get("class_aware", True)
        interpolate = config.get("interpolate", True)
        camera = config.get("tracking_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"

        if log_fn:
            log_fn(f"Running built-in tracking with mode: {tracker_mode_str}")

        # Load georeferenced detections (camera-specific folder)
        georef_folder = os.path.join(target_folder, f"georeferenced_{camera_suffix}")
        georef_file = os.path.join(georef_folder, "georeferenced.txt")

        if not os.path.exists(georef_file):
            raise FileNotFoundError("Georeferenced detections not found")

        # Parse detections
        frames: Dict[int, List[Detection]] = defaultdict(list)

        with open(georef_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    idx = int(parts[0])
                    frame = int(parts[1])
                    det = Detection(
                        source_id=idx,
                        frame=frame,
                        x1=float(parts[2]),
                        y1=float(parts[3]),
                        z1=float(parts[4]),
                        x2=float(parts[5]),
                        y2=float(parts[6]),
                        z2=float(parts[7]),
                        conf=float(parts[8]),
                        cls=int(parts[9])
                    )
                    frames[frame].append(det)

        if log_fn:
            log_fn(f"Loaded {sum(len(v) for v in frames.values())} detections in {len(frames)} frames")

        if progress_fn:
            progress_fn(20)

        # Run tracking
        all_frames = sorted(frames.keys())
        active_tracks: List[Track] = []
        next_tid = 1
        results: List[Tuple[int, int, Detection]] = []

        for fidx, f in enumerate(all_frames):
            dets = frames[f]

            # Simple Hungarian matching on IoU
            nD, nT = len(dets), len(active_tracks)
            matches = []
            unmatched_dets = list(range(nD))
            unmatched_trks = list(range(nT))

            if nD > 0 and nT > 0:
                import numpy as np
                IoU = np.zeros((nD, nT), dtype=np.float32)

                for di, d in enumerate(dets):
                    for ti, t in enumerate(active_tracks):
                        if class_aware and t.cls is not None and d.cls != t.cls:
                            continue
                        IoU[di, ti] = iou(
                            (d.x1, d.y1, d.x2, d.y2),
                            (t.x1, t.y1, t.x2, t.y2)
                        )

                # Hungarian assignment
                cost = 1.0 - IoU
                LARGE = 1e6
                cost[IoU < iou_threshold] = LARGE

                row_ind, col_ind = linear_sum_assignment(cost)

                det_used = set()
                trk_used = set()

                for di, ti in zip(row_ind, col_ind):
                    if IoU[di, ti] >= iou_threshold and cost[di, ti] < LARGE:
                        matches.append((di, ti))
                        det_used.add(di)
                        trk_used.add(ti)

                unmatched_dets = [i for i in range(nD) if i not in det_used]
                unmatched_trks = [i for i in range(nT) if i not in trk_used]

            # Update matched tracks
            for di, ti in matches:
                d = dets[di]
                t = active_tracks[ti]
                t.x1, t.y1, t.z1 = d.x1, d.y1, d.z1
                t.x2, t.y2, t.z2 = d.x2, d.y2, d.z2
                t.last_frame = f
                t.hits += 1
                t.age = 0
                results.append((f, t.tid, d))

            # Create new tracks for unmatched detections
            for di in unmatched_dets:
                d = dets[di]
                t = Track(
                    tid=next_tid,
                    cls=d.cls if class_aware else None,
                    x1=d.x1, y1=d.y1, z1=d.z1,
                    x2=d.x2, y2=d.y2, z2=d.z2,
                    last_frame=f,
                    age=0, hits=1
                )
                active_tracks.append(t)
                results.append((f, t.tid, d))
                next_tid += 1

            # Age unmatched tracks
            survivors = []
            for ti in range(len(active_tracks)):
                if ti in unmatched_trks:
                    active_tracks[ti].age += 1
                if max_age < 0 or active_tracks[ti].age <= max_age:
                    survivors.append(active_tracks[ti])
            active_tracks = survivors

            if progress_fn and fidx % 50 == 0:
                progress = 20 + int((fidx / len(all_frames)) * 60)
                progress_fn(min(progress, 85))

        # Sort results
        results.sort(key=lambda r: (r[0], r[1]))

        # Interpolate missing frames if enabled
        if interpolate:
            if log_fn:
                log_fn("Interpolating missing frames...")
            results = self._interpolate_tracks(results, Detection)

        if progress_fn:
            progress_fn(90)

        # Create output folder and write results (camera-specific)
        tracks_folder = os.path.join(target_folder, f"tracks_{camera_suffix}")
        os.makedirs(tracks_folder, exist_ok=True)

        output_file = os.path.join(tracks_folder, "tracks.csv")

        with open(output_file, 'w', encoding='utf-8') as f:
            for frame, tid, d in results:
                f.write(f"{frame:08d},{tid},{d.x1:.6f},{d.y1:.6f},{d.z1:.6f},"
                        f"{d.x2:.6f},{d.y2:.6f},{d.z2:.6f},"
                        f"{d.conf:.6f},{d.cls},{d.interpolated}\n")

        # Count unique tracks
        unique_tracks = set(r[1] for r in results)

        if log_fn:
            log_fn(f"Tracking complete: {len(unique_tracks)} tracks, {len(results)} total detections")

        if progress_fn:
            progress_fn(100)

    def _run_advanced_tracking(self, config: Dict[str, Any], tracker_manager, progress_fn=None, log_fn=None):
        """Run advanced tracking with BoxMOT or GeoRef backends in pixel space.

        This method:
        1. Loads pixel-space detections from detections.txt
        2. Runs frame-by-frame tracking purely in pixel coordinates
        3. Outputs pixel-space tracks to tracks/tracks_pixel.csv
        4. Calls geo-referencing to convert tracks to world coordinates
        """
        from .tracker_manager import TrackerBackend, ReIDModel
        import numpy as np
        import cv2
        from collections import defaultdict
        from dataclasses import dataclass

        @dataclass
        class PixelTrack:
            frame: int
            track_id: int
            x1: float
            y1: float
            x2: float
            y2: float
            conf: float
            cls: int

        target_folder = config["target_folder"]
        tracker_id = config.get("tracker_id", "builtin")
        reid_model_str = config.get("reid_model", "osnet")
        custom_reid_path = config.get("custom_reid_path", "")
        tracker_params_json = config.get("tracker_params_json", "")
        interpolate = config.get("interpolate", True)

        # Map ReID model string to enum
        reid_model_map = {
            "osnet": ReIDModel.OSNET,
            "bambi": ReIDModel.BAMBI,
            "custom": ReIDModel.CUSTOM
        }
        reid_model = reid_model_map.get(reid_model_str, ReIDModel.OSNET)

        if log_fn:
            log_fn(f"Creating tracker: {tracker_id}")
            log_fn(f"ReID model: {reid_model_str}")

        # Determine device
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                if log_fn:
                    log_fn("Using CUDA for tracking")
        except ImportError:
            pass

        # Create the tracker
        tracker, backend = tracker_manager.create_tracker(
            tracker_id=tracker_id,
            reid_model=reid_model,
            custom_reid_path=custom_reid_path,
            json_params=tracker_params_json,
            device=device,
            log_fn=log_fn
        )

        if log_fn:
            log_fn(f"Tracker created with backend: {backend.value}")

        # Camera selection for dual-input mode
        camera = config.get("tracking_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        # Load poses for selected camera
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError(f"poses_{camera_suffix}.json not found - run frame extraction first")

        with open(poses_file, 'r') as f:
            poses_data = json.load(f)

        images = poses_data.get("images", [])

        # Build frame index to image path mapping
        frame_to_path = {}
        for idx, img in enumerate(images):
            frame_idx = img.get("frame_index", idx)
            imagefile = img.get("imagefile", "")
            if imagefile:
                frame_to_path[frame_idx] = os.path.join(frames_folder, imagefile)

        # Load pixel-space detections from detections file (camera-specific folder)
        detections_file = os.path.join(target_folder, f"detections_{camera_suffix}", "detections.txt")
        if not os.path.exists(detections_file):
            raise FileNotFoundError("Detections file not found - run detection first")

        frames_pixel: Dict[int, List] = defaultdict(list)

        with open(detections_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        frame = int(parts[0])
                        frames_pixel[frame].append({
                            'x1': float(parts[1]),
                            'y1': float(parts[2]),
                            'x2': float(parts[3]),
                            'y2': float(parts[4]),
                            'conf': float(parts[5]),
                            'cls': int(parts[6]) if len(parts) > 6 else 0
                        })
                    except (ValueError, IndexError):
                        continue

        if log_fn:
            log_fn(
                f"Loaded {sum(len(v) for v in frames_pixel.values())} pixel detections in {len(frames_pixel)} frames")

        # Check if this is a geo-referenced tracker that needs geodets
        is_geo_tracker = backend in [TrackerBackend.GEOREF_NATIVE, TrackerBackend.GEOREF_HYBRID]
        frames_geo: Dict[int, List] = defaultdict(list)

        if is_geo_tracker:
            # Load geo-referenced detections for GeoNative/GeoHybrid trackers (camera-specific folder)
            georef_folder = os.path.join(target_folder, f"georeferenced_{camera_suffix}")
            georef_file = os.path.join(georef_folder, "georeferenced.txt")

            if os.path.exists(georef_file):
                with open(georef_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) >= 10:
                            try:
                                idx = int(parts[0])
                                frame = int(parts[1])
                                frames_geo[frame].append({
                                    'source_id': idx,
                                    'frame': frame,
                                    'x1': float(parts[2]),
                                    'y1': float(parts[3]),
                                    'z1': float(parts[4]),
                                    'x2': float(parts[5]),
                                    'y2': float(parts[6]),
                                    'z2': float(parts[7]),
                                    'conf': float(parts[8]),
                                    'cls': int(parts[9])
                                })
                            except (ValueError, IndexError):
                                continue

                if log_fn:
                    log_fn(f"Loaded {sum(len(v) for v in frames_geo.values())} geo detections for geo-tracker")
            else:
                if log_fn:
                    log_fn("Warning: Geo-referenced detections not found - geo tracker may not work optimally")

        if progress_fn:
            progress_fn(15)

        # Run tracking frame by frame
        all_frames = sorted(frames_pixel.keys())
        if not all_frames:
            raise RuntimeError("No detections found to track")

        pixel_tracks = []  # List of PixelTrack

        for fidx, frame_num in enumerate(all_frames):
            # Load the frame image
            frame_path = frame_to_path.get(frame_num)
            img = None

            if frame_path and os.path.exists(frame_path):
                img = cv2.imread(frame_path)

            if img is None:
                # Create dummy image if we can't load the actual frame
                img = np.zeros((480, 640, 3), dtype=np.uint8)

            # Get pixel detections for this frame
            pixel_dets = frames_pixel.get(frame_num, [])

            if not pixel_dets:
                # No detections - still call tracker to age tracks
                dets = np.empty((0, 6))
            else:
                # Build detection array [x1, y1, x2, y2, conf, cls]
                dets = np.array([
                    [d['x1'], d['y1'], d['x2'], d['y2'], d['conf'], d['cls']]
                    for d in pixel_dets
                ])

            # Call the tracker's update method
            try:
                if is_geo_tracker:
                    # Build geodets array for GeoNative/GeoHybrid trackers
                    # Format: [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]
                    geo_dets = frames_geo.get(frame_num, [])
                    if geo_dets:
                        geodets = np.array([
                            [d['source_id'], d['frame'], d['x1'], d['y1'], d['z1'],
                             d['x2'], d['y2'], d['z2'], d['conf'], d['cls']]
                            for d in geo_dets
                        ])
                    else:
                        geodets = None

                    # Forward geodets via instance attribute (BoxMOT decorator workaround)
                    tracker._current_frame_index = frame_num
                    tracker._current_geodets = geodets
                    tracks = tracker.update(dets, img)
                else:
                    # Standard BoxMOT tracker
                    tracks = tracker.update(dets, img)
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Tracker update failed at frame {frame_num}: {e}")
                tracks = np.array([])

            # Process tracking results
            # BoxMOT returns: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
            if len(tracks) > 0:
                for track in tracks:
                    if len(track) >= 7:
                        pixel_tracks.append(PixelTrack(
                            frame=frame_num,
                            track_id=int(track[4]),
                            x1=float(track[0]),
                            y1=float(track[1]),
                            x2=float(track[2]),
                            y2=float(track[3]),
                            conf=float(track[5]),
                            cls=int(track[6])
                        ))

            if progress_fn and fidx % 50 == 0:
                progress = 15 + int((fidx / len(all_frames)) * 50)
                progress_fn(min(progress, 65))

        # Sort by frame, then track_id
        pixel_tracks.sort(key=lambda t: (t.frame, t.track_id))

        # Interpolate missing frames if enabled
        if interpolate and pixel_tracks:
            if log_fn:
                log_fn("Interpolating missing frames...")
            pixel_tracks = self._interpolate_pixel_tracks(pixel_tracks, PixelTrack)

        if progress_fn:
            progress_fn(70)

        # Create output folder and write pixel-space tracks (camera-specific)
        tracks_folder = os.path.join(target_folder, f"tracks_{camera_suffix}")
        os.makedirs(tracks_folder, exist_ok=True)

        pixel_output_file = os.path.join(tracks_folder, "tracks_pixel.csv")

        with open(pixel_output_file, 'w', encoding='utf-8') as f:
            f.write("# frame,track_id,x1,y1,x2,y2,conf,cls,interpolated\n")
            for t in pixel_tracks:
                interp = getattr(t, 'interpolated', 0)
                f.write(f"{t.frame},{t.track_id},{t.x1:.2f},{t.y1:.2f},"
                        f"{t.x2:.2f},{t.y2:.2f},{t.conf:.4f},{t.cls},{interp}\n")

        # Count unique tracks
        unique_tracks = set(t.track_id for t in pixel_tracks)

        if log_fn:
            log_fn(f"Pixel tracking complete: {len(unique_tracks)} tracks, {len(pixel_tracks)} detections")
            log_fn(f"Saved to: {pixel_output_file}")
            log_fn(f"Backend used: {backend.value}")

        if progress_fn:
            progress_fn(75)

        # Now geo-reference the tracks
        if log_fn:
            log_fn("Geo-referencing tracks...")

        self.run_georeference_tracks(config, progress_fn, log_fn, start_progress=75)

        if progress_fn:
            progress_fn(100)

    def _interpolate_pixel_tracks(self, tracks: List, track_class) -> List:
        """Interpolate missing frames within each pixel-space track."""
        from collections import defaultdict

        # Group by track_id
        track_dict: Dict[int, List] = defaultdict(list)
        for t in tracks:
            track_dict[t.track_id].append(t)

        # Sort each track by frame
        for tid in track_dict:
            track_dict[tid].sort(key=lambda x: x.frame)

        new_tracks = []

        for tid, seq in track_dict.items():
            if len(seq) < 2:
                new_tracks.extend(seq)
                continue

            for i in range(len(seq) - 1):
                new_tracks.append(seq[i])

                t1, t2 = seq[i], seq[i + 1]
                gap = t2.frame - t1.frame

                if gap > 1:
                    # Interpolate
                    for j in range(1, gap):
                        alpha = j / gap
                        interp_track = track_class(
                            frame=t1.frame + j,
                            track_id=tid,
                            x1=t1.x1 + alpha * (t2.x1 - t1.x1),
                            y1=t1.y1 + alpha * (t2.y1 - t1.y1),
                            x2=t1.x2 + alpha * (t2.x2 - t1.x2),
                            y2=t1.y2 + alpha * (t2.y2 - t1.y2),
                            conf=(t1.conf + t2.conf) / 2,
                            cls=t1.cls
                        )
                        interp_track.interpolated = 1
                        new_tracks.append(interp_track)

            # Add last track
            new_tracks.append(seq[-1])

        return new_tracks

    def run_georeference_tracks(self, config: Dict[str, Any], progress_fn=None, log_fn=None, start_progress: int = 0):
        """Geo-reference pixel-space tracks using DEM.

        This converts pixel bounding boxes in tracks to world coordinates using
        the same projection pipeline as detection geo-referencing.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        :param start_progress: Starting progress value (for integration with tracking)
        """
        import numpy as np

        from pyrr import Vector3, Quaternion
        from trimesh import Trimesh

        from alfspy.core.rendering import Resolution, Camera
        from alfspy.render.render import read_gltf, process_render_data, release_all
        from bambi.util.projection_util import label_to_world_coordinates

        # Camera selection for dual-input mode
        camera = config.get("tracking_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        target_folder = config["target_folder"]
        dem_path = config["dem_path"]

        # Set frames folder based on camera selection
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        if log_fn:
            log_fn(f"Loading DEM and {camera_name} poses for track geo-referencing...")

        # Load pixel tracks (camera-specific folder)
        tracks_folder = os.path.join(target_folder, f"tracks_{camera_suffix}")
        pixel_tracks_file = os.path.join(tracks_folder, "tracks_pixel.csv")

        if not os.path.exists(pixel_tracks_file):
            raise FileNotFoundError("Pixel tracks not found - run tracking first")

        pixel_tracks = []
        with open(pixel_tracks_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 8:
                    try:
                        pixel_tracks.append({
                            'frame': int(parts[0]),
                            'track_id': int(parts[1]),
                            'x1': float(parts[2]),
                            'y1': float(parts[3]),
                            'x2': float(parts[4]),
                            'y2': float(parts[5]),
                            'conf': float(parts[6]),
                            'cls': int(parts[7]),
                            'interpolated': int(parts[8]) if len(parts) > 8 else 0
                        })
                    except (ValueError, IndexError):
                        continue

        if not pixel_tracks:
            raise RuntimeError("No pixel tracks found to geo-reference")

        if log_fn:
            log_fn(f"Loaded {len(pixel_tracks)} pixel track entries")

        # Load DEM metadata
        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        with open(dem_json_path, 'r') as f:
            dem_json = json.load(f)

        x_offset = dem_json["origin"][0]
        y_offset = dem_json["origin"][1]
        z_offset = dem_json["origin"][2]

        # Load poses for selected camera
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)

        # Get input resolution from first extracted frame
        input_resolution = None
        first_image = poses["images"][0]
        first_image_file = first_image.get("imagefile", "")
        if first_image_file:
            first_image_path = os.path.join(frames_folder, first_image_file)
            if os.path.exists(first_image_path):
                import cv2
                img = cv2.imread(first_image_path)
                if img is not None:
                    input_resolution = Resolution(img.shape[1], img.shape[0])
                    if log_fn:
                        log_fn(f"Input resolution: {img.shape[1]}x{img.shape[0]}")

        if input_resolution is None:
            res_width = config.get("input_resolution_width", 640)
            res_height = config.get("input_resolution_height", 512)
            input_resolution = Resolution(res_width, res_height)
            if log_fn:
                log_fn(f"Using configured resolution: {res_width}x{res_height}")
        aspect_ratio = input_resolution.width / input_resolution.height

        if progress_fn:
            progress_fn(start_progress + 5)

        # Load DEM mesh
        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        georef_tracks = []

        try:
            mesh_data, texture_data = read_gltf(dem_path)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)

            total_tracks = len(pixel_tracks)
            failed_count = 0

            for idx, pt in enumerate(pixel_tracks):
                frame_idx = pt['frame']

                if frame_idx >= len(poses["images"]):
                    failed_count += 1
                    continue

                # Get frame-specific correction factors
                correction = self.get_correction_for_frame(frame_idx, config)
                translation = correction["translation"]
                rotation = correction["rotation"]
                cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
                cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

                image_metadata = poses["images"][frame_idx]

                # Get camera for this frame
                fovy = image_metadata.get("fovy", [50])
                if isinstance(fovy, list):
                    fovy = fovy[0]
                position = Vector3(image_metadata["location"])
                rot = image_metadata["rotation"]
                rotation_eulers = (Vector3(
                    [np.deg2rad(val % 360.0) for val in rot]) - cor_rotation_eulers) * -1
                position += cor_translation
                rotation_quat = Quaternion.from_eulers(rotation_eulers)

                camera = Camera(fovy=fovy, aspect_ratio=aspect_ratio, position=position, rotation=rotation_quat)

                # Project bounding box corners to world coordinates
                x1, y1, x2, y2 = pt['x1'], pt['y1'], pt['x2'], pt['y2']
                label_coords = [x1, y1, x2, y1, x2, y2, x1, y2]

                try:
                    world_coords = label_to_world_coordinates(
                        label_coords, input_resolution, tri_mesh, camera
                    )

                    if len(world_coords) > 0:
                        xx = world_coords[:, 0] + x_offset
                        yy = world_coords[:, 1] + y_offset
                        zz = world_coords[:, 2] + z_offset

                        georef_tracks.append({
                            'frame': pt['frame'],
                            'track_id': pt['track_id'],
                            'x1': min(xx), 'y1': min(yy), 'z1': min(zz),
                            'x2': max(xx), 'y2': max(yy), 'z2': max(zz),
                            'conf': pt['conf'],
                            'cls': pt['cls'],
                            'interpolated': pt['interpolated']
                        })
                    else:
                        failed_count += 1
                except Exception:
                    failed_count += 1
                    continue

                if progress_fn and idx % 100 == 0:
                    progress = start_progress + 5 + int((idx / total_tracks) * 18)
                    progress_fn(min(progress, start_progress + 23))

        finally:
            if ctx:
                release_all(ctx)
            if mesh_data is not None:
                del mesh_data
            if texture_data is not None:
                del texture_data
            if tri_mesh is not None:
                del tri_mesh

        # Write geo-referenced tracks
        output_file = os.path.join(tracks_folder, "tracks.csv")

        with open(output_file, 'w', encoding='utf-8') as f:
            for gt in georef_tracks:
                f.write(f"{gt['frame']:08d},{gt['track_id']},{gt['x1']:.6f},{gt['y1']:.6f},{gt['z1']:.6f},"
                        f"{gt['x2']:.6f},{gt['y2']:.6f},{gt['z2']:.6f},"
                        f"{gt['conf']:.6f},{gt['cls']},{gt['interpolated']}\n")

        unique_tracks = set(gt['track_id'] for gt in georef_tracks)

        if log_fn:
            log_fn(f"Geo-referenced {len(georef_tracks)} track entries ({len(unique_tracks)} unique tracks)")
            if failed_count > 0:
                log_fn(f"Warning: Failed to geo-reference {failed_count} entries")
            log_fn(f"Saved to: {output_file}")

    def _interpolate_tracks(self, results: List, detection_class) -> List:
        """Interpolate missing frames within each track."""
        from collections import defaultdict

        tracks: Dict[int, List] = defaultdict(list)
        for frame, tid, det in results:
            tracks[tid].append((frame, det))

        for tid in tracks:
            tracks[tid].sort(key=lambda x: x[0])

        new_results = []

        for tid, seq in tracks.items():
            if len(seq) < 2:
                for f, d in seq:
                    new_results.append((f, tid, d))
                continue

            # Add first detection
            prev_frame, prev_det = seq[0]
            new_results.append((prev_frame, tid, prev_det))

            for i in range(1, len(seq)):
                curr_frame, curr_det = seq[i]
                gap = curr_frame - prev_frame

                if gap > 1:
                    # Interpolate
                    for j in range(1, gap):
                        t = j / gap
                        interp_det = detection_class(
                            source_id=-1,
                            frame=prev_frame + j,
                            x1=prev_det.x1 + t * (curr_det.x1 - prev_det.x1),
                            y1=prev_det.y1 + t * (curr_det.y1 - prev_det.y1),
                            z1=prev_det.z1 + t * (curr_det.z1 - prev_det.z1),
                            x2=prev_det.x2 + t * (curr_det.x2 - prev_det.x2),
                            y2=prev_det.y2 + t * (curr_det.y2 - prev_det.y2),
                            z2=prev_det.z2 + t * (curr_det.z2 - prev_det.z2),
                            conf=(prev_det.conf + curr_det.conf) / 2,
                            cls=curr_det.cls,
                            interpolated=1
                        )
                        new_results.append((prev_frame + j, tid, interp_det))

                new_results.append((curr_frame, tid, curr_det))
                prev_frame, prev_det = curr_frame, curr_det

        new_results.sort(key=lambda r: (r[0], r[1]))
        return new_results

    def run_alfs(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Generate an alfs from extracted frames and DEM.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        camera = config.get("alfs_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        if log_fn:
            log_fn(f"Initializing alfs generation for {camera_name} frames...")

        # Check for required dependencies
        try:
            from moderngl import Context as MglContext  # noqa: F401
            from pyrr import Quaternion, Vector3  # noqa: F401
            from trimesh import Trimesh  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"Required dependency not found: {e}\n\n"
                "Please install the following packages:\n"
                "  pip install moderngl pyrr trimesh\n\n"
                "Note: ALFS generation requires OpenGL support."
            )

        # Check for alfspy (specialized rendering library)
        try:
            from alfspy.core.geo.aabb import AABB  # noqa: F401
            from alfspy.core.geo.transform import Transform  # noqa: F401
            from alfspy.core.rendering import Resolution, Camera, CtxShot, RenderResultMode, TextureData  # noqa: F401
            from alfspy.core.rendering.renderer import Renderer  # noqa: F401
            from alfspy.core.util.geo import get_aabb  # noqa: F401
            from alfspy.core.util.pyrrs import quaternion_from_eulers  # noqa: F401
            from alfspy.render.render import (  # noqa: F401
                make_mgl_context, read_gltf, process_render_data,
                make_shot_loader, release_all
            )
            HAS_ALFSPY = True
        except ImportError:
            HAS_ALFSPY = False
            if log_fn:
                log_fn("Warning: alfspy not available, using simplified alfs generation")

        # Get configuration parameters
        target_folder = config["target_folder"]
        dem_path = config["dem_path"]
        target_epsg = config.get("target_epsg", 32633)

        # Set frames folder based on camera selection
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        # ALFS specific settings
        ground_resolution = config.get("alfs_ground_resolution", 0.05)
        dem_metadata_path = config.get("alfs_dem_metadata_path")
        use_all_frames = config.get("alfs_use_all_frames", True)
        start_frame = config.get("alfs_start_frame")
        end_frame = config.get("alfs_end_frame")
        crop_to_content = config.get("alfs_crop_to_content", True)
        create_overviews = config.get("alfs_create_overviews", True)
        max_tile_size = config.get("alfs_max_tile_size", 8192)
        frame_step = config.get("alfs_frame_step", 1)

        if log_fn:
            log_fn(f"Ground resolution: {ground_resolution} m/px")
            if use_all_frames:
                log_fn("Frame range: All frames")
            else:
                log_fn(f"Frame range: {start_frame} to {end_frame}")
            if frame_step > 1:
                log_fn(f"Frame step: every {frame_step} frames")

        if progress_fn:
            progress_fn(5)

        # Load poses for selected camera
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError(f"poses_{camera_suffix}.json not found at {poses_file}")

        with open(poses_file, 'r') as f:
            poses = json.load(f)

        images = poses.get("images", [])
        total_images = len(images)

        if total_images == 0:
            raise RuntimeError("No images found in poses.json")

        if log_fn:
            log_fn(f"Found {total_images} images in poses.json")

        # Add original frame indices to each image info before filtering
        for i, img in enumerate(images):
            img["_original_frame_idx"] = i

        # Apply frame filter
        if not use_all_frames and start_frame is not None and end_frame is not None:
            images = [img for img in images if start_frame <= img["_original_frame_idx"] <= end_frame]
            if log_fn:
                log_fn(f"Filtered to {len(images)} images (frames {start_frame}-{end_frame})")

        # Apply frame step filter
        if frame_step > 1:
            images = images[::frame_step]
            if log_fn:
                log_fn(f"After frame step: {len(images)} images")

        if len(images) == 0:
            raise RuntimeError("No images remaining after filtering")

        if progress_fn:
            progress_fn(10)

        # Load DEM metadata if available
        dem_metadata = None
        coord_offset_x = 0.0
        coord_offset_y = 0.0

        if dem_metadata_path and os.path.exists(dem_metadata_path):
            if log_fn:
                log_fn(f"Loading DEM metadata from {dem_metadata_path}")
            try:
                with open(dem_metadata_path, 'r') as f:
                    dem_metadata = json.load(f)
                origin = dem_metadata.get("origin", [0, 0, 0])
                coord_offset_x = float(origin[0])
                coord_offset_y = float(origin[1])
                if log_fn:
                    log_fn(f"DEM origin offset: X={coord_offset_x:.2f}, Y={coord_offset_y:.2f}")
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Could not load DEM metadata: {e}")
        else:
            # Try to find metadata in same folder as DEM
            auto_metadata_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(auto_metadata_path):
                if log_fn:
                    log_fn(f"Found DEM metadata at {auto_metadata_path}")
                try:
                    with open(auto_metadata_path, 'r') as f:
                        dem_metadata = json.load(f)
                    origin = dem_metadata.get("origin", [0, 0, 0])
                    coord_offset_x = float(origin[0])
                    coord_offset_y = float(origin[1])
                except Exception as e:
                    if log_fn:
                        log_fn(f"Warning: Could not load auto-detected metadata: {e}")

        # Load mask if available.
        # During photo-mode extraction the mask is renamed to mask_T.png /
        # mask_W.png in target_folder, while poses.json still stores the
        # original basename "mask.png".  Check the camera-specific name first.
        camera_specific_mask = os.path.join(target_folder, f"mask_{camera}.png")
        mask_filename = poses.get("mask")
        poses_mask = os.path.join(target_folder, mask_filename) if mask_filename else None
        if os.path.exists(camera_specific_mask):
            mask_path = camera_specific_mask
        elif poses_mask and os.path.exists(poses_mask):
            mask_path = poses_mask
        else:
            mask_path = None

        if progress_fn:
            progress_fn(15)

        # Create output folder and determine output file
        # Check if a specific output file is requested (used by geotiff export)
        if config.get("alfs_output_file"):
            output_file = config["alfs_output_file"]
            alfs_folder = os.path.dirname(output_file)
            os.makedirs(alfs_folder, exist_ok=True)
        else:
            alfs_folder = os.path.join(target_folder, f"alfs_{camera_suffix}")
            os.makedirs(alfs_folder, exist_ok=True)
            output_file = os.path.join(alfs_folder, "alfs.tif")

        sampling_mode = config.get("alfs_sampling_mode", False)
        sampling_rate = config.get("alfs_sampling_rate", 10)
        sampling_range = config.get("alfs_sampling_range", 5)

        if HAS_ALFSPY and sampling_mode:
            self._run_alfs_sampling(
                config, images, dem_path, mask_path,
                alfs_folder, ground_resolution,
                coord_offset_x, coord_offset_y, target_epsg,
                crop_to_content, create_overviews, max_tile_size,
                frames_folder, sampling_rate, sampling_range,
                progress_fn, log_fn, cancel_check
            )
        elif HAS_ALFSPY:
            # Use full alfspy-based rendering pipeline
            self._run_alfs_alfspy(
                config, images, poses, dem_path, mask_path,
                output_file, ground_resolution,
                coord_offset_x, coord_offset_y, target_epsg,
                crop_to_content, create_overviews, max_tile_size,
                frames_folder, progress_fn, log_fn, cancel_check
            )
        else:
            # Use simplified alfs generation (projection without rendering)
            self._run_alfs_simple(
                config, images, poses, dem_path, mask_path,
                output_file, ground_resolution,
                coord_offset_x, coord_offset_y, target_epsg,
                crop_to_content, create_overviews,
                frames_folder, progress_fn, log_fn, cancel_check
            )

        if log_fn:
            log_fn(f"ALFS tiles saved to: {os.path.dirname(output_file)}")

        if progress_fn:
            progress_fn(100)

    def _create_tile_camera(self, global_camera, global_bounds, global_res,
                            tx, ty, tw, th, Vector3, Camera):
        """Create a sub-camera for tiling (core.ortho_tiling)."""
        from .core.ortho_tiling import create_tile_camera
        return create_tile_camera(
            global_camera, global_bounds, global_res,
            tx, ty, tw, th, Vector3, Camera)

    def _run_alfs_sampling(
            self, config, all_images, dem_path, mask_path,
            alfs_folder, ground_resolution,
            coord_offset_x, coord_offset_y, target_epsg,
            crop_to_content, create_overviews, max_tile_size,
            frames_folder, sampling_rate, sampling_range,
            progress_fn, log_fn, cancel_check=None
    ):
        """Sampling-mode alfs: render one small integral image per central frame.

        Central frames are picked every `sampling_rate` frames across the filtered image
        list.  Each central frame's alfs blends the frames within
        [central - sampling_range, central + sampling_range].  Results are saved as
        alfs/sample_XXXXXX/tile_RR_CC.tif.
        """
        import math
        import cv2
        import numpy as np
        from pyrr import Quaternion, Vector3

        from alfspy.core.geo.transform import Transform
        from alfspy.core.rendering import Resolution, Camera, CtxShot, TextureData
        from alfspy.core.rendering.renderer import Renderer
        from alfspy.core.util.geo import get_aabb
        from alfspy.core.util.pyrrs import quaternion_from_eulers
        from alfspy.render.render import (
            make_mgl_context, read_gltf, process_render_data
        )

        if log_fn:
            log_fn("Loading DEM mesh (sampling mode)...")

        mesh_data, texture_data = read_gltf(dem_path)
        mesh_data, texture_data = process_render_data(mesh_data, texture_data)
        mesh_aabb = get_aabb(mesh_data.vertices)

        ctx = make_mgl_context()

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                mask = TextureData(CtxShot._cvt_img(mask_img))

        # Derive aspect ratio from the first available frame
        frame_aspect_ratio = 1.0
        for img_probe in all_images:
            probe_path = os.path.join(frames_folder, img_probe.get("imagefile", ""))
            if os.path.exists(probe_path):
                probe_img = cv2.imread(probe_path)
                if probe_img is not None:
                    frame_aspect_ratio = probe_img.shape[1] / probe_img.shape[0]
                    break
        if log_fn:
            log_fn(f"Frame aspect ratio: {frame_aspect_ratio:.4f}")

        # Determine central frames from the available frame indices
        all_frame_indices = sorted(img["_original_frame_idx"] for img in all_images)
        if not all_frame_indices:
            if log_fn:
                log_fn("No frames available for sampling mode")
            return

        min_frame = all_frame_indices[0]
        max_frame = all_frame_indices[-1]
        central_frames = list(range(min_frame, max_frame + 1, sampling_rate))

        if log_fn:
            log_fn(
                f"Sampling mode: {len(central_frames)} central frame(s), "
                f"rate={sampling_rate}, range=±{sampling_range}"
            )

        total = len(central_frames)

        for sample_i, central_frame in enumerate(central_frames):
            if cancel_check and cancel_check():
                if log_fn:
                    log_fn("Sampling cancelled by user")
                break

            # Select images within the neighboring range
            sample_images = [
                img for img in all_images
                if central_frame - sampling_range <=  # noqa: W503, W504
                img["_original_frame_idx"] <=  # noqa: W503, W504
                central_frame + sampling_range
            ]

            if not sample_images:
                if log_fn:
                    log_fn(f"  Sample {sample_i + 1}/{total} (frame {central_frame}): no images in range, skipping")
                continue

            if log_fn:
                log_fn(
                    f"  Sample {sample_i + 1}/{total}: central frame {central_frame}, "
                    f"{len(sample_images)} frames"
                )

            # Load shots for this sample
            shots = []
            default_fovy = 50.0
            for img_info in sample_images:
                image_file = img_info.get("imagefile")
                image_path = os.path.join(frames_folder, image_file)
                if not os.path.exists(image_path):
                    continue

                frame_idx = img_info.get("_original_frame_idx", 0)
                correction_data = self.get_correction_for_frame(frame_idx, config)
                correction_translation = correction_data["translation"]
                correction_rotation = correction_data["rotation"]

                cor_translation = Vector3([
                    correction_translation.get('x', 0),
                    correction_translation.get('y', 0),
                    correction_translation.get('z', 0)
                ], dtype='f4')
                cor_rotation_eulers = Vector3([
                    correction_rotation.get('x', 0),
                    correction_rotation.get('y', 0),
                    correction_rotation.get('z', 0)
                ], dtype='f4')
                cor_quat = Quaternion.from_eulers(cor_rotation_eulers)
                correction = Transform(cor_translation, cor_quat)

                location = img_info.get("location", [0, 0, 0])
                rotation = img_info.get("rotation", [0, 0, 0])
                fovy = img_info.get("fovy", [default_fovy])
                fov_value = fovy[0] if isinstance(fovy, (list, tuple)) else fovy

                camera_position = Vector3(location, dtype='f4')
                if len(rotation) == 3:
                    # Canonical create_shot / CtxShot convention: no negation, 'zyx' order.
                    # A negated rotation renders the texture rotated ~2*heading off.
                    camera_rotation = quaternion_from_eulers(
                        [np.deg2rad(val % 360.0) for val in rotation], 'zyx'
                    )
                elif len(rotation) == 4:
                    camera_rotation = Quaternion(rotation)
                else:
                    continue

                try:
                    shot = CtxShot(
                        ctx, image_path, camera_position, camera_rotation,
                        fov_value, frame_aspect_ratio, correction, lazy=True
                    )
                    shots.append(shot)
                except Exception as e:
                    if log_fn:
                        log_fn(f"    Warning: failed to load shot {image_file}: {e}")

            if not shots:
                if log_fn:
                    log_fn(f"  Sample {sample_i + 1}/{total}: no valid shots, skipping")
                continue

            # Compute local bounds from shot positions only (not mesh AABB)
            shot_positions = np.array([shot.camera.transform.position for shot in shots])
            padding = 10.0
            min_x = shot_positions[:, 0].min() - padding
            min_y = shot_positions[:, 1].min() - padding
            max_x = shot_positions[:, 0].max() + padding
            max_y = shot_positions[:, 1].max() + padding
            global_bounds = (min_x, min_y, max_x, max_y)

            width_meters = max_x - min_x
            height_meters = max_y - min_y
            width_pixels = max(1, int(math.ceil(width_meters / ground_resolution)))
            height_pixels = max(1, int(math.ceil(height_meters / ground_resolution)))
            global_resolution = Resolution(width_pixels, height_pixels)

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            center_z = float(mesh_aabb.p_max.z) + 100.0
            global_camera = Camera(
                orthogonal=True,
                orthogonal_size=(max_x - min_x, -(max_y - min_y)),
                position=Vector3([center_x, center_y, center_z], dtype='f4'),
                rotation=Quaternion(),
                near=0.1,
                far=10000.0
            )

            # Tile and render
            pixel_size_x = (max_x - min_x) / width_pixels
            pixel_size_y = (max_y - min_y) / height_pixels
            n_cols = math.ceil(width_pixels / max_tile_size)
            n_rows = math.ceil(height_pixels / max_tile_size)

            sample_prefix = f"sample_{central_frame:06d}"

            tiles = []
            for row in range(n_rows):
                ty = row * max_tile_size
                th = min(max_tile_size, height_pixels - ty)
                for col in range(n_cols):
                    tx = col * max_tile_size
                    tw = min(max_tile_size, width_pixels - tx)
                    tiles.append((row, col, tx, ty, tw, th))

            for tile_i, (row, col, tx, ty, tw, th) in enumerate(tiles):
                tile_geo_bounds = (
                    min_x + tx * pixel_size_x,
                    max_y - (ty + th) * pixel_size_y,
                    min_x + (tx + tw) * pixel_size_x,
                    max_y - ty * pixel_size_y,
                )

                tile_camera = self._create_tile_camera(
                    global_camera, global_bounds, global_resolution,
                    tx, ty, tw, th, Vector3, Camera
                )
                tile_res = Resolution(tw, th)

                tile_shots = self._filter_shots_for_tile(shots, tile_geo_bounds, log_fn)

                renderer = Renderer(tile_res, ctx, tile_camera, mesh_data, texture_data)

                try:
                    tile_img = renderer.render_integral(
                        tile_shots, mask=mask, save=False, release_shots=False,
                        auto_contrast=True, alpha_threshold=0.5
                    )
                finally:
                    renderer.release()

                if crop_to_content:
                    tile_img, tile_geo_bounds = self._crop_to_content(tile_img, tile_geo_bounds)

                geo_bounds = (
                    tile_geo_bounds[0] + coord_offset_x,
                    tile_geo_bounds[1] + coord_offset_y,
                    tile_geo_bounds[2] + coord_offset_x,
                    tile_geo_bounds[3] + coord_offset_y,
                )

                has_content = (
                    tile_img[:, :, 3].max() > 0
                    if tile_img.ndim == 3 and tile_img.shape[2] == 4
                    else tile_img.max() > 0
                )
                if not has_content:
                    del tile_img
                    for shot in tile_shots:
                        if shot.tex is not None:
                            shot.tex.release()
                            shot.tex = None
                    continue

                tile_path = os.path.join(alfs_folder, f"{sample_prefix}_tile_{row:02d}_{col:02d}.tif")
                self._save_alfs(tile_img, tile_path, geo_bounds, target_epsg, create_overviews, log_fn)

                del tile_img

                # Free GPU textures for tile_shots only; unfiltered shots never
                # had their textures loaded, so there's nothing to release there.
                for shot in tile_shots:
                    if shot.tex is not None:
                        shot.tex.release()
                        shot.tex = None

            # Release shots for this sample (including CPU tex_data)
            for shot in shots:
                try:
                    shot.release()
                except Exception:  # nosec B110
                    pass

            if progress_fn:
                progress_fn(10 + int(((sample_i + 1) / total) * 85))

        try:
            ctx.release()
        except Exception:  # nosec B110
            pass

    def _filter_shots_for_tile(self, shots, tile_geo_bounds, log_fn=None, fov_default=50.0):
        """Cull shots to those overlapping a tile (core.ortho_tiling)."""
        from .core.ortho_tiling import filter_shots_for_tile
        return filter_shots_for_tile(
            shots, tile_geo_bounds, log_fn=log_fn, fov_default=fov_default)

    def _run_alfs_alfspy(
            self, config, images, poses, dem_path, mask_path,
            output_file, ground_resolution,
            coord_offset_x, coord_offset_y, target_epsg,
            crop_to_content, create_overviews, max_tile_size,
            frames_folder, progress_fn, log_fn, cancel_check=None
    ):
        """Run alfs generation using alfspy rendering pipeline with tiling support."""
        import math
        import cv2
        import numpy as np
        from pyrr import Quaternion, Vector3

        # Import alfspy components
        from alfspy.core.geo.transform import Transform
        from alfspy.core.rendering import Resolution, Camera, CtxShot, TextureData
        from alfspy.core.rendering.renderer import Renderer
        from alfspy.core.util.geo import get_aabb
        from alfspy.core.util.pyrrs import quaternion_from_eulers
        from alfspy.render.render import (
            make_mgl_context, read_gltf, process_render_data,
            release_all
        )

        if log_fn:
            log_fn("Loading DEM mesh...")

        # 1. Load Mesh
        mesh_data, texture_data = read_gltf(dem_path)
        mesh_data, texture_data = process_render_data(mesh_data, texture_data)
        mesh_aabb = get_aabb(mesh_data.vertices)

        # 2. Setup Context & Mask
        ctx = make_mgl_context()
        mask = None
        if mask_path and os.path.exists(mask_path):
            if log_fn:
                log_fn(f"Loading mask from {mask_path}")
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                mask = TextureData(CtxShot._cvt_img(mask_img))

        # Derive aspect ratio from the first available frame
        frame_aspect_ratio = 1.0
        for img_probe in images:
            probe_path = os.path.join(frames_folder, img_probe.get("imagefile", ""))
            if os.path.exists(probe_path):
                probe_img = cv2.imread(probe_path)
                if probe_img is not None:
                    frame_aspect_ratio = probe_img.shape[1] / probe_img.shape[0]
                    break
        if log_fn:
            log_fn(f"Frame aspect ratio: {frame_aspect_ratio:.4f}")

        # 3. Load Shots with per-frame corrections
        if log_fn:
            log_fn(f"Loading {len(images)} shots...")

        shots = []
        default_fovy = 50.0

        for i, img_info in enumerate(images):
            # Check for cancellation
            if cancel_check and cancel_check():
                if log_fn:
                    log_fn("ALFS generation cancelled by user")
                raise CancelledException("ALFS generation cancelled")

            image_file = img_info.get("imagefile")
            image_path = os.path.join(frames_folder, image_file)

            if not os.path.exists(image_path):
                continue

            # Get frame-specific correction factors using original frame index
            frame_idx = img_info.get("_original_frame_idx", i)
            correction_data = self.get_correction_for_frame(frame_idx, config)
            correction_translation = correction_data["translation"]
            correction_rotation = correction_data["rotation"]

            cor_translation = Vector3([
                correction_translation.get('x', 0),
                correction_translation.get('y', 0),
                correction_translation.get('z', 0)
            ], dtype='f4')

            cor_rotation_eulers = Vector3([
                correction_rotation.get('x', 0),
                correction_rotation.get('y', 0),
                correction_rotation.get('z', 0)
            ], dtype='f4')
            cor_quat = Quaternion.from_eulers(cor_rotation_eulers)
            correction = Transform(cor_translation, cor_quat)

            location = img_info.get("location", [0, 0, 0])
            rotation = img_info.get("rotation", [0, 0, 0])
            fovy = img_info.get("fovy", [default_fovy])

            camera_position = Vector3(location, dtype='f4')

            # Convert rotation
            if len(rotation) == 3:
                # Canonical create_shot / CtxShot convention: no negation, 'zyx' order.
                # A negated rotation renders the texture rotated ~2*heading off.
                camera_rotation = quaternion_from_eulers(
                    [np.deg2rad(val % 360.0) for val in rotation], 'zyx'
                )
            elif len(rotation) == 4:
                camera_rotation = Quaternion(rotation)
            else:
                continue

            fov_value = fovy[0] if isinstance(fovy, (list, tuple)) else fovy

            try:
                shot = CtxShot(
                    ctx, image_path, camera_position, camera_rotation,
                    fov_value, frame_aspect_ratio, correction, lazy=True
                )
                shots.append(shot)
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Failed to load shot {image_file}: {e}")

            if progress_fn and i % 50 == 0:
                progress_fn(10 + int((i / len(images)) * 10))

        if not shots:
            raise RuntimeError("No valid shots loaded")

        # 4. Compute Global Bounds (Merging Mesh AABB + Shot AABB)
        # Using the helper logic from orthomosaic.py
        shot_positions = np.array([shot.camera.transform.position for shot in shots])

        padding = 10.0
        min_x = min(shot_positions[:, 0].min(), float(mesh_aabb.p_min.x)) - padding
        min_y = min(shot_positions[:, 1].min(), float(mesh_aabb.p_min.y)) - padding
        max_x = max(shot_positions[:, 0].max(), float(mesh_aabb.p_max.x)) + padding
        max_y = max(shot_positions[:, 1].max(), float(mesh_aabb.p_max.y)) + padding

        global_bounds = (min_x, min_y, max_x, max_y)

        # 6. Compute Resolution & Camera
        width_meters = max_x - min_x
        height_meters = max_y - min_y
        width_pixels = int(math.ceil(width_meters / ground_resolution))
        height_pixels = int(math.ceil(height_meters / ground_resolution))

        global_resolution = Resolution(width_pixels, height_pixels)

        if log_fn:
            log_fn(f"Output resolution: {width_pixels} x {height_pixels}")
            log_fn(f"Global bounds: X[{min_x:.1f}, {max_x:.1f}] Y[{min_y:.1f}, {max_y:.1f}]")

        # Create global camera (using logic from orthomosaic.py)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = float(mesh_aabb.p_max.z) + 100.0  # Height above terrain

        ortho_width = max_x - min_x
        ortho_height = -(max_y - min_y)  # Negative for GeoTIFF Y-axis orientation

        global_camera = Camera(
            orthogonal=True,
            orthogonal_size=(ortho_width, ortho_height),
            position=Vector3([center_x, center_y, center_z], dtype='f4'),
            rotation=Quaternion(),  # Looking down
            near=0.1,
            far=10000.0
        )

        # 7. Render tiles and save each as an individual GeoTIFF
        output_dir = os.path.dirname(output_file)
        pixel_size_x = (max_x - min_x) / width_pixels
        pixel_size_y = (max_y - min_y) / height_pixels

        n_cols = math.ceil(width_pixels / max_tile_size)
        n_rows = math.ceil(height_pixels / max_tile_size)

        tiles = []
        for row in range(n_rows):
            ty = row * max_tile_size
            th = min(max_tile_size, height_pixels - ty)
            for col in range(n_cols):
                tx = col * max_tile_size
                tw = min(max_tile_size, width_pixels - tx)
                tiles.append((row, col, tx, ty, tw, th))

        if log_fn:
            log_fn(f"Processing {len(tiles)} tile(s) ({n_rows}x{n_cols} grid)...")

        # Bounded LRU of shots whose CPU-side image (tex_data, a float32 RGBA
        # array — ~33 MB at 1080p, ~130 MB at 4K) is kept resident.  Previously
        # tex_data was retained for every shot ever touched, so RAM grew without
        # bound as tiles were processed and eventually OOM'd on large flights.
        # We now keep only the most-recently-used shots cached; overlapping
        # neighbour tiles still hit the cache, while distant shots are evicted
        # and re-read from disk on demand.
        from collections import OrderedDict
        cpu_cache_size = int(config.get("alfs_cpu_cache_size", 32))
        cpu_cache = OrderedDict()  # id(shot) -> shot, ordered oldest-first
        if log_fn:
            log_fn(f"CPU image cache: up to {cpu_cache_size} shots resident (alfs_cpu_cache_size)")

        def _reclaim_cpu(used_shots):
            """Release GPU textures for the tile's shots, then trim the CPU cache.

            GPU textures are always freed (VRAM is scarce).  Each shot with
            resident CPU data is marked most-recently-used; shots beyond
            cpu_cache_size are evicted and their tex_data freed.
            """
            for shot in used_shots:
                if shot.tex is not None:
                    shot.tex.release()
                    shot.tex = None
                if shot.tex_data is not None:
                    key = id(shot)
                    cpu_cache.pop(key, None)
                    cpu_cache[key] = shot  # re-insert as newest
            while len(cpu_cache) > cpu_cache_size:
                _, old = cpu_cache.popitem(last=False)
                # shot.release() permanently disables the shot, so free the
                # CPU-side buffers directly instead.
                old.tex_data = None
                old._tex_gen_input = None

        try:
            for i, (row, col, tx, ty, tw, th) in enumerate(tiles):
                if cancel_check and cancel_check():
                    break

                # Tile geo bounds (before coord offset)
                tile_geo_bounds = (
                    min_x + tx * pixel_size_x,
                    max_y - (ty + th) * pixel_size_y,
                    min_x + (tx + tw) * pixel_size_x,
                    max_y - ty * pixel_size_y,
                )

                tile_camera = self._create_tile_camera(
                    global_camera, global_bounds, global_resolution,
                    tx, ty, tw, th, Vector3, Camera
                )

                tile_res = Resolution(tw, th)

                # Only upload GPU textures for shots whose footprint overlaps
                # this tile — reduces peak VRAM from ~2.7 GB (all shots) to the
                # few shots that actually cover the tile (~2-6 typically).
                tile_shots = self._filter_shots_for_tile(shots, tile_geo_bounds, log_fn)

                renderer = Renderer(tile_res, ctx, tile_camera, mesh_data, texture_data)

                try:
                    # Pass tile_shots list directly — re-iterable across tiles and
                    # CtxShot.tex_use() handles lazy loading, so AsyncShotLoader
                    # (which spawns 12 threads and is one-shot) is not needed.
                    tile_img = renderer.render_integral(
                        tile_shots, mask=mask, save=False, release_shots=False,
                        auto_contrast=True, alpha_threshold=0.5
                    )
                finally:
                    renderer.release()

                if crop_to_content:
                    tile_img, tile_geo_bounds = self._crop_to_content(tile_img, tile_geo_bounds)

                geo_bounds = (
                    tile_geo_bounds[0] + coord_offset_x,
                    tile_geo_bounds[1] + coord_offset_y,
                    tile_geo_bounds[2] + coord_offset_x,
                    tile_geo_bounds[3] + coord_offset_y,
                )

                has_content = (
                    tile_img[:, :, 3].max() > 0
                    if tile_img.ndim == 3 and tile_img.shape[2] == 4
                    else tile_img.max() > 0
                )
                if not has_content:
                    if log_fn:
                        log_fn(f"Skipping empty tile ({row}, {col})")
                    del tile_img
                    _reclaim_cpu(tile_shots)
                    continue

                tile_path = os.path.join(output_dir, f"tile_{row:02d}_{col:02d}.tif")
                if log_fn:
                    log_fn(f"Saving tile ({row}, {col}) to {os.path.basename(tile_path)}")

                self._save_alfs(tile_img, tile_path, geo_bounds, target_epsg, create_overviews, log_fn)

                if progress_fn:
                    progress_fn(20 + int(((i + 1) / len(tiles)) * 70))

                del tile_img

                # Free GPU textures for the shots used in this tile so VRAM is
                # released before the next Renderer (FBO + mesh re-upload) is
                # allocated, and trim the bounded CPU cache so RAM stays capped.
                _reclaim_cpu(tile_shots)
        finally:
            release_all(ctx, shots)

    def _run_alfs_simple(
            self, config, images, poses, dem_path, mask_path,
            output_file, ground_resolution,
            coord_offset_x, coord_offset_y, target_epsg,
            crop_to_content, create_overviews,
            frames_folder, progress_fn, log_fn, cancel_check=None
    ):
        """Simplified alfs generation without alfspy.

        This method creates a basic mosaic by projecting frames
        based on their GPS positions and orientations.
        """
        import cv2
        import numpy as np
        import math

        if log_fn:
            log_fn("Using simplified alfs generation...")

        # Collect all image positions (with corrections applied)
        positions = []
        valid_images = []

        for i, img_info in enumerate(images):
            location = list(img_info.get("location", [0, 0, 0]))

            # Apply frame-specific correction for bounds calculation
            frame_idx = img_info.get("_original_frame_idx", i)
            correction = self.get_correction_for_frame(frame_idx, config)
            cor_translation = correction["translation"]
            location[0] += cor_translation.get('x', 0)
            location[1] += cor_translation.get('y', 0)
            location[2] += cor_translation.get('z', 0)

            # Accept any location (even 0,0 might be valid for local coords)
            positions.append((location[0], location[1], location[2]))
            valid_images.append(img_info)

        if len(positions) == 0:
            raise RuntimeError("No valid image positions found")

        positions = np.array(positions)

        if log_fn:
            log_fn(f"Processing {len(valid_images)} images")
            log_fn(f"Position range: X=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
                   f"Y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}], "
                   f"Z=[{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")

        # Compute bounds with padding
        padding = 50.0  # meters
        min_x = positions[:, 0].min() - padding
        max_x = positions[:, 0].max() + padding
        min_y = positions[:, 1].min() - padding
        max_y = positions[:, 1].max() + padding

        if log_fn:
            log_fn(f"Canvas bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")

        # Compute output size
        width_meters = max_x - min_x
        height_meters = max_y - min_y
        width_pixels = int(math.ceil(width_meters / ground_resolution))
        height_pixels = int(math.ceil(height_meters / ground_resolution))

        # Limit size for memory
        max_dim = 16384
        if width_pixels > max_dim or height_pixels > max_dim:
            scale = min(max_dim / width_pixels, max_dim / height_pixels)
            width_pixels = int(width_pixels * scale)
            height_pixels = int(height_pixels * scale)
            # Adjust ground resolution accordingly
            ground_resolution = max(width_meters / width_pixels, height_meters / height_pixels)
            if log_fn:
                log_fn(f"Limiting output size, adjusted resolution: {ground_resolution:.4f} m/px")

        if log_fn:
            log_fn(f"Output size: {width_pixels} x {height_pixels} pixels")

        if progress_fn:
            progress_fn(30)

        # Create output canvas (use float for accumulation)
        canvas = np.zeros((height_pixels, width_pixels, 4), dtype=np.float32)
        count_map = np.zeros((height_pixels, width_pixels), dtype=np.float32)

        # Estimate footprint size based on altitude and FOV
        default_altitude = 50.0  # meters
        default_fov = 50.0  # degrees

        images_placed = 0

        for i, img_info in enumerate(valid_images):
            # Check for cancellation
            if cancel_check and cancel_check():
                if log_fn:
                    log_fn("ALFS generation cancelled by user")
                raise CancelledException("ALFS generation cancelled")

            image_file = img_info.get("imagefile")
            image_path = os.path.join(frames_folder, image_file)

            if not os.path.exists(image_path):
                if log_fn and i < 3:
                    log_fn(f"Warning: Image not found: {image_file}")
                continue

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                if log_fn and i < 3:
                    log_fn(f"Warning: Could not read: {image_file}")
                continue

            # Convert from BGR to RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img[:, :, 3] = 255

            # Get position and apply frame-specific correction
            frame_idx = img_info.get("_original_frame_idx", i)
            correction = self.get_correction_for_frame(frame_idx, config)
            cor_translation = correction["translation"]

            location = list(img_info.get("location", [0, 0, 0]))
            location[0] += cor_translation.get('x', 0)
            location[1] += cor_translation.get('y', 0)
            location[2] += cor_translation.get('z', 0)
            altitude = abs(location[2]) if location[2] != 0 else default_altitude

            # Get FOV
            fovy = img_info.get("fovy", [default_fov])
            fov_value = fovy[0] if isinstance(fovy, (list, tuple)) else fovy

            # Estimate footprint size
            footprint_y = 2 * altitude * math.tan(math.radians(fov_value / 2))
            aspect = img.shape[1] / img.shape[0]
            footprint_x = footprint_y * aspect

            # Calculate pixel position (center of image)
            px = int((location[0] - min_x) / ground_resolution)
            py = int((max_y - location[1]) / ground_resolution)  # Flip Y

            # Calculate output size for this frame
            out_w = int(footprint_x / ground_resolution)
            out_h = int(footprint_y / ground_resolution)

            # Clamp to reasonable sizes
            out_w = max(10, min(out_w, width_pixels // 2))
            out_h = max(10, min(out_h, height_pixels // 2))

            # Resize image
            resized = cv2.resize(img, (out_w, out_h)).astype(np.float32)

            # Calculate placement bounds (centered on camera position)
            x1 = max(0, px - out_w // 2)
            y1 = max(0, py - out_h // 2)
            x2 = min(width_pixels, x1 + out_w)
            y2 = min(height_pixels, y1 + out_h)

            # Calculate source region (if clipped)
            sx1 = max(0, (out_w // 2) - px) if px < out_w // 2 else 0
            sy1 = max(0, (out_h // 2) - py) if py < out_h // 2 else 0
            sx2 = sx1 + (x2 - x1)
            sy2 = sy1 + (y2 - y1)

            if x1 < x2 and y1 < y2 and sx1 < sx2 and sy1 < sy2 and sx2 <= out_w and sy2 <= out_h:
                try:
                    region = resized[sy1:sy2, sx1:sx2]

                    # Integral: accumulate for averaging across overlapping frames
                    current_count = count_map[y1:y2, x1:x2]
                    for c in range(4):
                        canvas[y1:y2, x1:x2, c] = (
                            canvas[y1:y2, x1:x2, c] * current_count + region[:, :, c]
                        ) / (current_count + 1)
                    count_map[y1:y2, x1:x2] += 1

                    images_placed += 1
                except Exception as e:
                    if log_fn and images_placed < 3:
                        log_fn(f"Warning: Error placing image {i}: {e}")

            if progress_fn and i % 20 == 0:
                progress = 30 + int((i / len(valid_images)) * 50)
                progress_fn(min(progress, 80))

        if log_fn:
            log_fn(f"Placed {images_placed} images on canvas")

        if progress_fn:
            progress_fn(80)

        # Convert to uint8 if needed
        result = canvas.astype(np.uint8)

        # Diagnostic: check result
        if log_fn:
            log_fn(f"Result shape: {result.shape}, min={result.min()}, max={result.max()}")
            non_zero = np.sum(result[:, :, 3] > 0)
            log_fn(f"Non-transparent pixels: {non_zero} / {result.shape[0] * result.shape[1]}")

        global_bounds = (min_x, min_y, max_x, max_y)

        # Crop to content if requested
        if crop_to_content:
            if log_fn:
                log_fn("Cropping to content area...")
            result, global_bounds = self._crop_to_content(result, global_bounds)

        # Apply coordinate offsets
        geo_bounds = (
            global_bounds[0] + coord_offset_x,
            global_bounds[1] + coord_offset_y,
            global_bounds[2] + coord_offset_x,
            global_bounds[3] + coord_offset_y
        )

        if progress_fn:
            progress_fn(85)

        # Save output
        self._save_alfs(
            result, output_file, geo_bounds, target_epsg,
            create_overviews, log_fn
        )

        if progress_fn:
            progress_fn(95)

    def _crop_to_content(self, image, bounds):
        """Crop image to the non-empty bounding box (core.ortho_tiling)."""
        from .core.ortho_tiling import crop_to_content
        return crop_to_content(image, bounds)

    def _save_alfs(
            self, image, output_file, bounds, crs_epsg,
            create_overviews, log_fn
    ):
        """Save alfs image with georeferencing."""
        import numpy as np

        height, width = image.shape[:2]
        min_x, min_y, max_x, max_y = bounds

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Check if image has content
        if log_fn:
            log_fn(f"Saving image: {width}x{height}, dtype={image.dtype}")
            if len(image.shape) == 3:
                for i in range(min(4, image.shape[2])):
                    band = image[:, :, i]
                    log_fn(f"  Band {i}: min={band.min()}, max={band.max()}, mean={band.mean():.1f}")

        # Try to use rasterio for GeoTIFF
        try:
            import rasterio
            from rasterio.transform import from_bounds

            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

            # Determine number of bands
            if len(image.shape) == 2:
                count = 1
                image = image[:, :, np.newaxis]
            else:
                count = image.shape[2]

            profile = {
                'driver': 'GTiff',
                'dtype': image.dtype,
                'width': width,
                'height': height,
                'count': count,
                'transform': transform,
                'compress': 'lzw',
            }

            # Try to set CRS using pyproj first (avoids PROJ database conflicts in QGIS)
            try:
                from pyproj import CRS as PyprojCRS
                pyproj_crs = PyprojCRS.from_epsg(crs_epsg)
                # Use WKT string which is more portable
                from rasterio.crs import CRS as RasterioCRS
                profile['crs'] = RasterioCRS.from_wkt(pyproj_crs.to_wkt())
                if log_fn:
                    log_fn(f"CRS set using pyproj: EPSG:{crs_epsg}")
            except Exception as e1:
                if log_fn:
                    log_fn(f"Warning: pyproj CRS failed: {e1}")
                # Try rasterio directly as fallback
                try:
                    from rasterio.crs import CRS as RasterioCRS
                    profile['crs'] = RasterioCRS.from_epsg(crs_epsg)
                except Exception as e2:
                    if log_fn:
                        log_fn(f"Warning: rasterio CRS also failed: {e2}")
                        log_fn("GeoTIFF will be saved without embedded CRS - use .prj file")

            if width > 256 and height > 256:
                profile['tiled'] = True
                profile['blockxsize'] = 256
                profile['blockysize'] = 256

            if log_fn:
                log_fn(f"Saving GeoTIFF to {output_file}")

            with rasterio.open(output_file, 'w', **profile) as dst:
                for i in range(count):
                    dst.write(image[:, :, i], i + 1)

                if count == 4:
                    dst.descriptions = ('Red', 'Green', 'Blue', 'Alpha')
                elif count == 3:
                    dst.descriptions = ('Red', 'Green', 'Blue')

                if create_overviews:
                    if log_fn:
                        log_fn("Creating overviews...")
                    try:
                        overview_levels = [2, 4, 8, 16]
                        dst.build_overviews(overview_levels, rasterio.enums.Resampling.average)
                    except Exception as e:
                        if log_fn:
                            log_fn(f"Warning: Could not create overviews: {e}")

            # Always save world file as backup
            self._save_world_file(output_file, bounds, width, height)

            # Always save PRJ file for CRS (as backup and for compatibility)
            self._save_prj_file(output_file, crs_epsg, log_fn)

            if log_fn:
                log_fn("GeoTIFF saved successfully")

        except ImportError:
            # Fall back to OpenCV + world file
            if log_fn:
                log_fn("rasterio not available, saving with world file")

            self._save_with_worldfile(image, output_file, bounds, crs_epsg, log_fn)
        except Exception as e:
            if log_fn:
                log_fn(f"Warning: GeoTIFF save failed ({e}), falling back to world file")
            self._save_with_worldfile(image, output_file, bounds, crs_epsg, log_fn)

    def _save_with_worldfile(self, image, output_file, bounds, crs_epsg, log_fn):
        """Save image with world file (fallback method)."""
        import cv2

        height, width = image.shape[:2]
        min_x, min_y, max_x, max_y = bounds

        # Convert to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 4:
            result_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            result_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            result_bgr = image

        cv2.imwrite(output_file, result_bgr)

        # Save world file
        self._save_world_file(output_file, bounds, width, height)

        # Save PRJ file
        self._save_prj_file(output_file, crs_epsg, log_fn)

        if log_fn:
            log_fn("Image saved with world file")

    def _save_world_file(self, output_file, bounds, width, height):
        """Save a world file (.tfw) for georeferencing."""
        min_x, min_y, max_x, max_y = bounds

        # Determine world file extension
        if output_file.lower().endswith('.tif') or output_file.lower().endswith('.tiff'):
            world_file = output_file.rsplit('.', 1)[0] + '.tfw'
        elif output_file.lower().endswith('.png'):
            world_file = output_file.rsplit('.', 1)[0] + '.pgw'
        elif output_file.lower().endswith('.jpg') or output_file.lower().endswith('.jpeg'):
            world_file = output_file.rsplit('.', 1)[0] + '.jgw'
        else:
            world_file = output_file + '.wld'

        pixel_size_x = (max_x - min_x) / width
        pixel_size_y = -(max_y - min_y) / height  # Negative for Y

        with open(world_file, 'w') as f:
            f.write(f"{pixel_size_x:.10f}\n")
            f.write("0.0\n")
            f.write("0.0\n")
            f.write(f"{pixel_size_y:.10f}\n")
            f.write(f"{min_x + pixel_size_x / 2:.10f}\n")
            f.write(f"{max_y + pixel_size_y / 2:.10f}\n")

    def _save_prj_file(self, output_file, crs_epsg, log_fn):
        """Save a .prj file with CRS definition."""
        prj_file = output_file.rsplit('.', 1)[0] + '.prj'

        try:
            from pyproj import CRS
            crs = CRS.from_epsg(crs_epsg)
            with open(prj_file, 'w') as f:
                f.write(crs.to_wkt())
            if log_fn:
                log_fn(f"PRJ file saved: {prj_file}")
        except Exception as e:
            if log_fn:
                log_fn(f"Warning: Could not save PRJ file: {e}")

    @staticmethod
    def _collect_frame_geotiffs(geotiff_folder, use_all_frames=True,
                                start_frame=None, end_frame=None):
        """Collect exported frame GeoTIFFs ({idx:08d}.tiff), filtered by range.

        Non-frame rasters (e.g. a previous orthomosaic) are skipped because
        their file names are not plain frame indices.
        """
        candidates = []
        for f in sorted(os.listdir(geotiff_folder)):
            if not f.lower().endswith((".tif", ".tiff")):
                continue
            try:
                frame_idx = int(os.path.splitext(f)[0])
            except ValueError:
                continue
            if not use_all_frames:
                if start_frame is not None and frame_idx < start_frame:
                    continue
                if end_frame is not None and frame_idx > end_frame:
                    continue
            candidates.append(os.path.join(geotiff_folder, f))
        return candidates

    def run_orthomosaic(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Merge exported per-frame GeoTIFFs into a single true orthomosaic.

        Unlike the ALFS product (which renders an integral light-field image),
        this mosaics the individually orthorectified frame GeoTIFFs produced by
        :meth:`run_export_geotiffs`.  Overlapping pixels are resolved with the
        configured merge method (first/last/min/max/average).  Frames are
        selected by camera (thermal/RGB) and optionally restricted to a
        frame-index range.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        :param cancel_check: Callable returning True if the run was cancelled
        """
        try:
            import rasterio
            from rasterio.merge import merge as rio_merge
        except ImportError as exc:
            raise ImportError(
                f"rasterio is required for orthomosaic generation.\n"
                f"Original error: {exc}"
            )

        camera_sel = config.get("ortho_camera", "T")
        camera_suffix = "t" if camera_sel == "T" else "w"
        camera_name = "Thermal" if camera_sel == "T" else "RGB"
        method = config.get("ortho_method", "first")
        use_all_frames = config.get("ortho_use_all_frames", True)
        start_frame = config.get("ortho_start_frame")
        end_frame = config.get("ortho_end_frame")
        nodata = config.get("ortho_nodata", 0)
        target_epsg = config.get("target_epsg", 32633)

        target_folder = config["target_folder"]
        geotiff_folder = os.path.join(target_folder, f"geotiffs_{camera_suffix}")

        if log_fn:
            log_fn(f"Building orthomosaic from {camera_name} frame GeoTIFFs...")
            log_fn(f"Merge mode: {method}")

        if not os.path.isdir(geotiff_folder):
            raise FileNotFoundError(
                f"GeoTIFF folder not found: {geotiff_folder}\n"
                f"Run 'Export Frames as GeoTIFF' for the {camera_name} camera first."
            )

        candidates = self._collect_frame_geotiffs(
            geotiff_folder, use_all_frames, start_frame, end_frame)

        if not candidates:
            raise RuntimeError(
                "No frame GeoTIFFs found for the selected range in "
                f"{geotiff_folder}."
            )

        if log_fn:
            log_fn(f"Merging {len(candidates)} GeoTIFF(s)...")
        if progress_fn:
            progress_fn(10)

        # Open all datasets. The exports already share the target CRS, so no
        # reprojection is needed — we only capture a source CRS as a fallback.
        datasets = []
        source_crs = None
        try:
            for path in candidates:
                if cancel_check and cancel_check():
                    raise CancelledException("Orthomosaic generation cancelled")
                ds = rasterio.open(path)
                if source_crs is None and ds.crs is not None:
                    source_crs = ds.crs
                datasets.append(ds)

            if progress_fn:
                progress_fn(40)

            if method == "average":
                mosaic, out_transform = self._merge_orthomosaic_average(
                    datasets, nodata, rio_merge
                )
            else:
                mosaic, out_transform = rio_merge(datasets, method=method, nodata=nodata)
        finally:
            for d in datasets:
                d.close()

        if progress_fn:
            progress_fn(80)

        out_folder = os.path.join(target_folder, f"orthomosaic_{camera_suffix}")
        os.makedirs(out_folder, exist_ok=True)
        output_file = os.path.join(out_folder, "orthomosaic.tif")

        # Resolve the output CRS the same way the frame GeoTIFFs were written
        # (authoritatively from target_epsg), so the orthomosaic is tagged with
        # the exact CRS the pipeline georeferenced the frames in. Fall back to a
        # source raster's embedded CRS if target_epsg is missing/invalid.
        out_crs = None
        try:
            from pyproj import CRS as PyprojCRS
            from rasterio.crs import CRS as RasterioCRS
            if target_epsg:
                out_crs = RasterioCRS.from_wkt(PyprojCRS.from_epsg(target_epsg).to_wkt())
        except Exception:
            out_crs = None
        if out_crs is None:
            out_crs = source_crs
        if log_fn:
            log_fn(f"Output CRS: {out_crs.to_string() if out_crs else 'unknown'}")

        out_meta = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": mosaic.shape[0],
            "dtype": mosaic.dtype,
            "crs": out_crs,
            "transform": out_transform,
            "compress": "LZW",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
            "nodata": nodata,
        }
        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(mosaic)

        # Build overview pyramids for fast GIS rendering (non-fatal on failure).
        try:
            from rasterio.enums import Resampling as RioResampling
            with rasterio.open(output_file, "r+") as dst:
                dst.build_overviews([2, 4, 8, 16], RioResampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
        except Exception as exc:
            if log_fn:
                log_fn(f"Note: could not build overviews ({exc})")

        if log_fn:
            log_fn(
                f"Orthomosaic written to: {output_file} "
                f"({mosaic.shape[2]} x {mosaic.shape[1]} px, {mosaic.shape[0]} band(s))"
            )
        if progress_fn:
            progress_fn(100)

    def _merge_orthomosaic_average(self, datasets, nodata, rio_merge):
        """Merge datasets by averaging overlapping pixels (core.ortho_tiling)."""
        from .core.ortho_tiling import merge_orthomosaic_average
        return merge_orthomosaic_average(datasets, nodata, rio_merge)

    def run_export_geotiffs(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Export each frame as an individual GeoTIFF.

        Uses the same alfspy projection pipeline as the alfs, with
        mask-polygon-derived bounds (like the FoV calculation) for accurate
        georeferencing.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import numpy as np
        import cv2

        from pyrr import Vector3, Quaternion

        try:
            from alfspy.core.geo.transform import Transform
            from alfspy.core.rendering import Resolution, Camera, CtxShot, TextureData
            from alfspy.core.rendering.renderer import Renderer
            from alfspy.core.util.geo import get_aabb
            from alfspy.core.util.pyrrs import quaternion_from_eulers
            from alfspy.render.render import read_gltf, process_render_data, make_mgl_context, release_all
            from trimesh import Trimesh
        except ImportError as exc:
            raise ImportError(
                f"alfspy and trimesh are required for GeoTIFF export.\n"
                f"Original error: {exc}"
            )

        camera_sel = config.get("geotiff_camera", "T")
        camera_suffix = "t" if camera_sel == "T" else "w"
        camera_name = "Thermal" if camera_sel == "T" else "RGB"

        if log_fn:
            log_fn(f"Initializing GeoTIFF export for {camera_name} frames...")

        target_folder = config["target_folder"]
        dem_path = config["dem_path"]
        target_epsg = config.get("target_epsg", 32633)
        use_all_frames = config.get("alfs_use_all_frames", True)
        start_frame = config.get("alfs_start_frame") or 0
        end_frame_cfg = config.get("alfs_end_frame") or 999999
        ground_resolution = config.get("alfs_ground_resolution", 0.1)
        dem_metadata_path = config.get("alfs_dem_metadata_path")
        frame_step = config.get("alfs_frame_step", 1)
        mask_simplify_epsilon = config.get("geotiff_mask_simplify_epsilon", 2.0)
        # Shrink the valid footprint by this many pixels before saving. The
        # rendered footprint edge is an antialiased/interpolated fringe where
        # alpha > 0 but RGB has bled towards black, producing the dark rim
        # visible where frames overlap in the orthomosaic. Eroding the mask
        # drops that ring to nodata. 0 disables the erosion.
        edge_erosion_px = int(config.get("geotiff_edge_erosion_px", 2))

        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        if log_fn:
            log_fn(f"Output resolution: {ground_resolution} m/px")

        if progress_fn:
            progress_fn(2)

        # Load poses
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError(f"poses_{camera_suffix}.json not found at {poses_file}")
        with open(poses_file, 'r') as f:
            poses = json.load(f)
        all_images = poses.get("images", [])
        if not all_images:
            raise RuntimeError("No images found in poses.json")
        if log_fn:
            log_fn(f"Found {len(all_images)} images in poses.json")

        # Build frame index list
        frame_indices = [
            i for i in range(len(all_images))
            if use_all_frames or (start_frame <= i <= end_frame_cfg)
        ]
        if frame_step > 1:
            frame_indices = frame_indices[::frame_step]
        if not frame_indices:
            raise RuntimeError("No frames to export after filtering")
        if log_fn:
            log_fn(f"Will export {len(frame_indices)} frames as GeoTIFF")

        if progress_fn:
            progress_fn(4)

        # Load DEM coordinate offset
        x_offset = y_offset = z_offset = 0.0
        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        if dem_metadata_path and os.path.exists(dem_metadata_path):
            dem_json_path = dem_metadata_path
        if os.path.exists(dem_json_path):
            with open(dem_json_path) as f:
                dem_json = json.load(f)
            origin = dem_json.get("origin", [0, 0, 0])
            x_offset, y_offset, z_offset = float(origin[0]), float(origin[1]), float(origin[2])
        if log_fn:
            log_fn(f"DEM origin: ({x_offset:.2f}, {y_offset:.2f}, {z_offset:.2f})")

        # Determine input resolution from first available frame
        input_resolution = None
        for img_info_probe in all_images:
            probe_path = os.path.join(frames_folder, img_info_probe.get("imagefile", ""))
            if os.path.exists(probe_path):
                probe_img = cv2.imread(probe_path)
                if probe_img is not None:
                    input_resolution = Resolution(probe_img.shape[1], probe_img.shape[0])
                    break
        if input_resolution is None:
            input_resolution = Resolution(1024, 1024)
        aspect_ratio = input_resolution.width / input_resolution.height
        if log_fn:
            log_fn(f"Input resolution: {input_resolution.width}x{input_resolution.height}")

        # Load mask polygon for bound calculation (same source as alfs mask)
        mask_filename = poses.get("mask")
        mask_file_path = os.path.join(target_folder, mask_filename) if mask_filename else None
        mask_polygon = None
        if mask_file_path and os.path.exists(mask_file_path):
            mask_polygon = self._extract_mask_polygon(mask_file_path, mask_simplify_epsilon, log_fn)
            if log_fn and mask_polygon:
                log_fn(f"Loaded mask polygon with {len(mask_polygon)} points")
        if not mask_polygon:
            w, h = input_resolution.width, input_resolution.height
            mask_polygon = [
                (0, 0), (w // 2, 0), (w, 0),
                (w, h // 2), (w, h),
                (w // 2, h), (0, h),
                (0, h // 2)
            ]
            if log_fn:
                log_fn("Using image corners as FoV polygon")

        # Create output folder (camera-specific)
        geotiff_folder = os.path.join(target_folder, f"geotiffs_{camera_suffix}")
        os.makedirs(geotiff_folder, exist_ok=True)

        if progress_fn:
            progress_fn(6)

        ctx = None
        try:
            # --- DEM mesh mode ---
            # Load DEM mesh
            if log_fn:
                log_fn(f"Loading DEM mesh from: {dem_path}")
            if not os.path.exists(dem_path):
                raise FileNotFoundError(f"DEM file not found: {dem_path}")

            mesh_data, texture_data = read_gltf(dem_path)
            # Build Trimesh for ray-casting before process_render_data modifies the data
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)
            mesh_aabb = get_aabb(mesh_data.vertices)

            if log_fn:
                log_fn(f"Mesh bounds: X[{mesh_aabb.p_min.x:.1f}, {mesh_aabb.p_max.x:.1f}] "
                       f"Y[{mesh_aabb.p_min.y:.1f}, {mesh_aabb.p_max.y:.1f}]")

            ctx = make_mgl_context()

            # Set up mask texture for rendering
            mask_texture = None
            if mask_file_path and os.path.exists(mask_file_path):
                mask_img_cv = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
                if mask_img_cv is not None:
                    mask_texture = TextureData(CtxShot._cvt_img(mask_img_cv))

            # Orthographic camera Z: well above the highest terrain point
            ortho_cam_z = float(mesh_aabb.p_max.z) + 100.0

            if progress_fn:
                progress_fn(15)

            exported_count = 0
            total_frames = len(frame_indices)
            if log_fn:
                log_fn("Starting frame export...")

            for i, frame_idx in enumerate(frame_indices):
                if cancel_check and cancel_check():
                    if log_fn:
                        log_fn("GeoTIFF export cancelled by user")
                    raise CancelledException("GeoTIFF export cancelled")

                try:
                    img_info = all_images[frame_idx]
                    image_file = img_info.get("imagefile")
                    image_path = os.path.join(frames_folder, image_file)
                    if not os.path.exists(image_path):
                        if log_fn and i < 5:
                            log_fn(f"Warning: Image not found: {image_file}")
                        continue

                    location = img_info.get("location", [0, 0, 0])
                    rotation = img_info.get("rotation", [0, 0, 0])
                    fovy = img_info.get("fovy", [45.0])
                    if isinstance(fovy, (list, tuple)):
                        fovy = fovy[0]

                    # Per-frame corrections
                    correction_data = self.get_correction_for_frame(frame_idx, config)
                    cor_t = correction_data["translation"]
                    cor_r = correction_data["rotation"]
                    cor_translation_v = Vector3(
                        [cor_t.get('x', 0), cor_t.get('y', 0), cor_t.get('z', 0)], dtype='f4'
                    )
                    cor_rotation_v = Vector3(
                        [cor_r.get('x', 0), cor_r.get('y', 0), cor_r.get('z', 0)], dtype='f4'
                    )

                    # Projection camera: 2× correction matches CtxShot's effective rotation.
                    # alfspy CtxShot doubles correction_transform.rotation in its shader;
                    # multiplying by 2 here keeps geographic bounds consistent with the
                    # rendered content and with TRexConnector (which uses the full correction
                    # value directly, i.e. 2× the QGIS calibration value).
                    proj_position = Vector3(location, dtype='f4') + cor_translation_v
                    proj_eulers = (
                        Vector3([np.deg2rad(val % 360.0) for val in rotation]) - cor_rotation_v * 2.0
                    ) * -1
                    proj_camera = Camera(
                        fovy=fovy, aspect_ratio=aspect_ratio,
                        position=proj_position,
                        rotation=Quaternion.from_eulers(proj_eulers)
                    )

                    # Project mask polygon → world coords → UTM bounds
                    georef_points = self._georeference_polygon(
                        mask_polygon, input_resolution, tri_mesh, proj_camera,
                        x_offset, y_offset, z_offset
                    )
                    valid_pts = [p for p in georef_points if p is not None]
                    if len(valid_pts) < 3:
                        if log_fn and i < 5:
                            log_fn(f"Warning: Insufficient mask hits for frame {frame_idx}")
                        continue

                    xs = [p[0] for p in valid_pts]
                    ys = [p[1] for p in valid_pts]
                    utm_min_x, utm_max_x = min(xs), max(xs)
                    utm_min_y, utm_max_y = min(ys), max(ys)
                    width_m = utm_max_x - utm_min_x
                    height_m = utm_max_y - utm_min_y
                    if width_m <= 0 or height_m <= 0:
                        continue

                    out_w = max(1, int(np.ceil(width_m / ground_resolution)))
                    out_h = max(1, int(np.ceil(height_m / ground_resolution)))
                    max_dim = 8000
                    if out_w > max_dim or out_h > max_dim:
                        scale = max_dim / max(out_w, out_h)
                        out_w = int(out_w * scale)
                        out_h = int(out_h * scale)

                    # Build the shot rotation exactly like bambi's canonical create_shot
                    # (alfspy CtxShot rendering convention): no negation, 'zyx' Euler order.
                    # A negated rotation here renders the texture rotated by ~2*heading
                    # relative to the (correct) georeferencing/footprint.
                    raw_rotation = quaternion_from_eulers(
                        [np.deg2rad(val % 360.0) for val in rotation], 'zyx'
                    )
                    cor_quat = Quaternion.from_eulers(cor_rotation_v)
                    correction_transform = Transform(cor_translation_v, cor_quat)
                    shot = CtxShot(
                        ctx, image_path,
                        Vector3(location, dtype='f4'), raw_rotation,
                        fovy, aspect_ratio, correction_transform, lazy=False
                    )

                    # Orthographic camera covering this frame's footprint (local DEM coords)
                    local_min_x = utm_min_x - x_offset
                    local_max_x = utm_max_x - x_offset
                    local_min_y = utm_min_y - y_offset
                    local_max_y = utm_max_y - y_offset
                    center_x = (local_min_x + local_max_x) / 2
                    center_y = (local_min_y + local_max_y) / 2
                    orthographic_camera = Camera(
                        orthogonal=True,
                        orthogonal_size=(local_max_x - local_min_x, local_max_y - local_min_y),
                        position=Vector3([center_x, center_y, ortho_cam_z], dtype='f4'),
                        rotation=Quaternion(),
                        near=0.1,
                        far=10000.0
                    )

                    # Render the frame as an orthophoto.
                    # Use ShotOnly mode so only pixels actually covered by this shot
                    # are returned — render_integral bleeds the DEM mesh texture into
                    # the background producing noise.
                    from alfspy.core.rendering import RenderResultMode
                    tile_res = Resolution(out_w, out_h)
                    renderer = Renderer(tile_res, ctx, orthographic_camera, mesh_data, texture_data)
                    proj_results = list(renderer.project_shots_iter(
                        shot, RenderResultMode.ShotOnly,
                        release_shots=True, mask=mask_texture
                    ))
                    renderer.release()

                    if proj_results:
                        rendered = np.array(proj_results[0])  # copy: GPU buffer is read-only
                    else:
                        rendered = np.zeros((out_h, out_w, 4), dtype=np.uint8)

                    output_image = rendered[:, :, :3]
                    output_valid = rendered[:, :, 3] > 0

                    # Erode the valid footprint to remove the dark antialiased
                    # rim at the frame border (see geotiff_edge_erosion_px).
                    if edge_erosion_px > 0:
                        from .core.ortho_tiling import erode_valid_mask
                        output_valid = erode_valid_mask(output_valid, edge_erosion_px)

                    # Save GeoTIFF
                    utm_bounds = (utm_min_x, utm_min_y, utm_max_x, utm_max_y)
                    output_file = os.path.join(geotiff_folder, f"{frame_idx:08d}.tiff")
                    self._save_frame_geotiff(output_image, output_valid, utm_bounds, output_file, target_epsg)

                    if os.path.exists(output_file):
                        exported_count += 1

                except CancelledException:
                    raise
                except Exception as e:
                    if log_fn and i < 10:
                        import traceback
                        log_fn(f"Warning: Failed to export frame {frame_idx}: {e}")
                        if i < 3:
                            log_fn(traceback.format_exc())

                if progress_fn:
                    progress_fn(15 + int((i + 1) / total_frames * 80))

                if log_fn and (i + 1) % 50 == 0:
                    log_fn(f"Exported {exported_count}/{total_frames} frames so far...")

        finally:
            if ctx:
                release_all(ctx)

        if log_fn:
            log_fn(f"Frame export complete. {exported_count}/{total_frames} frames saved to: {geotiff_folder}")

        if progress_fn:
            progress_fn(100)

    def _save_frame_geotiff(self, image: 'np.ndarray', valid_mask: 'np.ndarray',  # noqa: F821
                            bounds: tuple, output_path: str, crs_epsg: int):
        """Save a single frame as a georeferenced GeoTIFF.

        :param image: Image array (H, W) or (H, W, C)
        :param valid_mask: Boolean mask for valid pixels
        :param bounds: (min_x, min_y, max_x, max_y) in CRS coordinates
        :param output_path: Output file path
        :param crs_epsg: EPSG code for coordinate reference system
        """
        import numpy as np

        min_x, min_y, max_x, max_y = bounds
        height, width = image.shape[:2]

        # Determine number of bands
        if len(image.shape) == 2:
            count = 1
            data = image[np.newaxis, :, :]
        else:
            count = image.shape[2]
            data = np.moveaxis(image, -1, 0)

        # Set nodata for invalid pixels
        nodata = 0
        for band in range(count):
            data[band][~valid_mask] = nodata

        # Try to use rasterio
        try:
            import rasterio
            from rasterio.transform import from_bounds

            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

            profile = {
                'driver': 'GTiff',
                'dtype': data.dtype,
                'width': width,
                'height': height,
                'count': count,
                'transform': transform,
                'compress': 'lzw',
                'nodata': nodata
            }

            # Set CRS
            try:
                from pyproj import CRS as PyprojCRS
                from rasterio.crs import CRS as RasterioCRS
                pyproj_crs = PyprojCRS.from_epsg(crs_epsg)
                profile['crs'] = RasterioCRS.from_wkt(pyproj_crs.to_wkt())
            except Exception:
                try:
                    from rasterio.crs import CRS as RasterioCRS
                    profile['crs'] = RasterioCRS.from_epsg(crs_epsg)
                except Exception:  # nosec B110
                    pass  # Will save without CRS

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)

            # Save world file as backup
            self._save_world_file(output_path, bounds, width, height)

        except ImportError:
            # Fall back to OpenCV + world file
            import cv2

            if len(image.shape) == 3 and image.shape[2] == 3:
                result_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                result_bgr = image

            cv2.imwrite(output_path, result_bgr)
            self._save_world_file(output_path, bounds, width, height)
            self._save_prj_file(output_path, crs_epsg, None)

    # =========================================================================
    # SAM3 SEGMENTATION
    # =========================================================================

    def run_sam3_segmentation(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Run SAM3 segmentation on extracted frames using Roboflow Serverless API.

        Expects:
          config["target_folder"]
          config["sam3_api_key"]
          config["sam3_prompts"] (list[str])
          config["sam3_confidence"] (float, default 0.5)
          config["sam3_format"] (str: "polygon" | "rle" | "json", default "polygon")

          Optional frame filters:
            sam3_use_all_frames, sam3_start_frame, sam3_end_frame, sam3_step
        """
        import os
        import json
        import base64
        import requests

        camera = config.get("sam3_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        target_folder = config["target_folder"]
        api_key = config.get("sam3_api_key", "")
        prompts = config.get("sam3_prompts", [])
        confidence = float(config.get("sam3_confidence", 0.5))
        output_format = config.get("sam3_format", "polygon")  # FIX: was undefined in your code

        # Frame filter options (use start/end frame like alfs)
        use_all_frames = config.get("sam3_use_all_frames", True)
        start_frame = int(config.get("sam3_start_frame", 0))
        end_frame = int(config.get("sam3_end_frame", 999999))
        frame_step = int(config.get("sam3_step", 1))

        # Set frames folder based on camera selection
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        if not api_key:
            raise ValueError("Roboflow API key is required for SAM3 segmentation")
        if not prompts:
            raise ValueError("At least one text prompt is required for SAM3 segmentation")
        if frame_step < 1:
            raise ValueError("sam3_step must be >= 1")

        if log_fn:
            log_fn(
                f"Starting SAM3 segmentation on {camera_name} frames (serverless)"
                f" with {len(prompts)} prompts: {prompts}"
            )
            log_fn(f"Confidence threshold: {confidence}")
            log_fn(f"Output format: {output_format}")

        if progress_fn:
            progress_fn(5)

        # Load poses for selected camera
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        if not os.path.exists(poses_file):
            raise FileNotFoundError(f"poses_{camera_suffix}.json not found - run frame extraction first")

        with open(poses_file, "r", encoding="utf-8") as f:
            poses = json.load(f)

        images = poses.get("images", [])
        if not images:
            raise ValueError(f"No frames found in poses_{camera_suffix}.json")

        # Create output folder (camera-specific)
        segmentation_folder = os.path.join(target_folder, f"segmentation_{camera_suffix}")
        os.makedirs(segmentation_folder, exist_ok=True)

        # Hosted SAM3 concept segmentation endpoint (serverless)
        # Docs show: https://serverless.roboflow.com/sam3/concept_segment?api_key=...
        # :contentReference[oaicite:3]{index=3}
        endpoint = "https://serverless.roboflow.com/sam3/concept_segment"

        # Build list of frame indices to process (start/end frame, then step)
        total_frames = len(images)

        if use_all_frames:
            frame_indices = list(range(total_frames))
            if log_fn:
                log_fn("Frame range: All frames")
        else:
            # Clamp end_frame to valid range
            end_frame_clamped = min(end_frame, total_frames - 1)
            frame_indices = list(range(start_frame, end_frame_clamped + 1))
            if log_fn:
                log_fn(f"Frame range: {start_frame} to {end_frame_clamped}")

        if frame_step > 1:
            frame_indices = frame_indices[::frame_step]
            if log_fn:
                log_fn(f"Frame step: every {frame_step} frames")

        frames_to_process = frame_indices

        if log_fn:
            log_fn(f"Processing {len(frames_to_process)} {camera_name} frames after filtering...")

        if progress_fn:
            progress_fn(10)

        def _encode_image_b64(path: str) -> str:
            # Standard base64 (no data: prefix) is accepted in Roboflow Inference
            # request image schema. :contentReference[oaicite:4]{index=4}
            with open(path, "rb") as fimg:
                return base64.b64encode(fimg.read()).decode("utf-8")

        def _make_payload(image_b64: str) -> Dict[str, Any]:
            # Matches the SAM3 concept_segment JSON structure from Roboflow docs.
            # :contentReference[oaicite:5]{index=5}
            return {
                "format": output_format,
                "output_prob_thresh": confidence,
                # supported by the SAM3 request model :contentReference[oaicite:6]{index=6}
                "image": {
                    "type": "base64",
                    "value": image_b64,
                },
                "prompts": [{"type": "text", "text": p} for p in prompts],
            }

        all_results = []

        # Use a session for connection pooling
        session = requests.Session()
        headers = {"Content-Type": "application/json"}

        for idx, frame_idx in enumerate(frames_to_process):
            # Check for cancellation
            if cancel_check and cancel_check():
                if log_fn:
                    log_fn("SAM3 segmentation cancelled by user")
                raise CancelledException("SAM3 segmentation cancelled")

            if frame_idx >= len(images):
                continue

            image_info = images[frame_idx]
            imagefile = image_info.get("imagefile", "")
            if not imagefile:
                continue

            image_path = os.path.join(frames_folder, imagefile)
            if not os.path.exists(image_path):
                if log_fn:
                    log_fn(f"Warning: Frame not found: {image_path}")
                continue

            try:
                image_b64 = _encode_image_b64(image_path)
                payload = _make_payload(image_b64)

                # API key can be passed as query param (as shown in SAM3 serverless
                # example). :contentReference[oaicite:7]{index=7}
                resp = session.post(
                    endpoint,
                    params={"api_key": api_key},
                    headers=headers,
                    json=payload,
                    timeout=(10, 120),  # connect/read timeouts
                )
                resp.raise_for_status()
                resp_json = resp.json()

                # Normalize into your existing output structure
                frame_results = {
                    "frame_idx": frame_idx,
                    "imagefile": imagefile,
                    "prompts": [],
                }

                for prompt_result in resp_json.get("prompt_results", []):
                    echo = prompt_result.get("echo", {}) or {}
                    prompt_text = (echo.get("text") or prompt_result.get("prompt") or prompt_result.get("text") or "")

                    prompt_data = {
                        "prompt": prompt_text,
                        "predictions": [],
                    }

                    for prediction in prompt_result.get("predictions", []):
                        pred_data = {
                            "confidence": float(prediction.get("confidence", 0.0)),
                        }

                        masks = prediction.get("masks", None)

                        # For "polygon" format, masks is typically a list of polygons
                        # (list[list[points]]). :contentReference[oaicite:8]{index=8}
                        if output_format == "polygon" and masks is not None:
                            pred_data["polygons"] = masks
                        else:
                            # Keep raw masks for other formats (e.g., rle/json)
                            if masks is not None:
                                pred_data["masks"] = masks

                        prompt_data["predictions"].append(pred_data)

                    frame_results["prompts"].append(prompt_data)

                all_results.append(frame_results)

            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: Failed to process frame {frame_idx}: {str(e)}")
                continue

            if progress_fn:
                # 10..95 range
                progress = 10 + int(((idx + 1) / max(1, len(frames_to_process))) * 85)
                progress_fn(min(progress, 95))

            if log_fn and (idx + 1) % 10 == 0:
                log_fn(f"Processed {idx + 1}/{len(frames_to_process)} frames")

        # Save results
        output_file = os.path.join(segmentation_folder, "segmentation_pixel.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        total_predictions = sum(
            len(p.get("predictions", []))
            for r in all_results
            for p in r.get("prompts", [])
        )

        if log_fn:
            log_fn(f"SAM3 segmentation complete: {len(all_results)} frames, {total_predictions} predictions")
            log_fn(f"Results saved to: {output_file}")

        if progress_fn:
            progress_fn(100)

    def run_sam3_georeference(self, config: Dict[str, Any], progress_fn=None, log_fn=None, cancel_check=None):
        """Geo-reference SAM3 segmentation masks.

        Converts pixel polygon coordinates to world coordinates using the same
        projection pipeline as FoV calculation.

        :param config: Configuration dictionary
        :param progress_fn: Progress callback function
        :param log_fn: Logging callback function
        """
        import numpy as np

        from pyrr import Vector3, Quaternion
        from trimesh import Trimesh

        from alfspy.core.rendering import Resolution, Camera
        from alfspy.render.render import read_gltf, process_render_data, release_all
        from bambi.util.projection_util import label_to_world_coordinates

        # Camera selection for dual-input mode
        camera = config.get("sam3_camera", "T")
        camera_suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        target_folder = config["target_folder"]
        dem_path = config["dem_path"]

        # Set frames folder based on camera selection
        frames_folder = os.path.join(target_folder, f"frames_{camera_suffix}")

        if log_fn:
            log_fn(f"Starting SAM3 geo-referencing for {camera_name} frames...")

        # Load pixel segmentation results (camera-specific folder)
        segmentation_folder = os.path.join(target_folder, f"segmentation_{camera_suffix}")
        pixel_file = os.path.join(segmentation_folder, "segmentation_pixel.json")

        if not os.path.exists(pixel_file):
            raise FileNotFoundError("Pixel segmentation not found - run SAM3 segmentation first")

        with open(pixel_file, 'r', encoding='utf-8') as f:
            pixel_results = json.load(f)

        if not pixel_results:
            raise ValueError("No segmentation results found")

        if log_fn:
            log_fn(f"Loaded {len(pixel_results)} frame results")

        # Load DEM metadata
        dem_json_path = dem_path.replace(".gltf", ".json").replace(".glb", ".json")
        with open(dem_json_path, 'r') as f:
            dem_json = json.load(f)

        x_offset = dem_json["origin"][0]
        y_offset = dem_json["origin"][1]
        z_offset = dem_json["origin"][2]

        # Load poses for selected camera
        poses_file = os.path.join(target_folder, f"poses_{camera_suffix}.json")
        with open(poses_file, 'r') as f:
            poses = json.load(f)

        # Get input resolution from first extracted frame
        input_resolution = None
        first_image = poses["images"][0]
        first_image_file = first_image.get("imagefile", "")
        if first_image_file:
            first_image_path = os.path.join(frames_folder, first_image_file)
            if os.path.exists(first_image_path):
                import cv2
                img = cv2.imread(first_image_path)
                if img is not None:
                    input_resolution = Resolution(img.shape[1], img.shape[0])
                    if log_fn:
                        log_fn(f"Input resolution: {img.shape[1]}x{img.shape[0]}")

        if input_resolution is None:
            res_width = config.get("input_resolution_width", 640)
            res_height = config.get("input_resolution_height", 512)
            input_resolution = Resolution(res_width, res_height)
            if log_fn:
                log_fn(f"Using configured resolution: {res_width}x{res_height}")
        aspect_ratio = input_resolution.width / input_resolution.height

        if progress_fn:
            progress_fn(10)

        # Load DEM mesh
        ctx = None
        mesh_data = None
        texture_data = None
        tri_mesh = None
        georef_results = []

        try:
            mesh_data, texture_data = read_gltf(dem_path)
            tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            mesh_data, texture_data = process_render_data(mesh_data, texture_data)

            total_frames = len(pixel_results)
            failed_count = 0

            for idx, frame_result in enumerate(pixel_results):
                # Check for cancellation
                if cancel_check and cancel_check():
                    if log_fn:
                        log_fn("SAM3 geo-referencing cancelled by user")
                    raise CancelledException("SAM3 geo-referencing cancelled")

                frame_idx = frame_result['frame_idx']

                if frame_idx >= len(poses["images"]):
                    failed_count += 1
                    continue

                # Get frame-specific correction factors
                correction = self.get_correction_for_frame(frame_idx, config)
                translation = correction["translation"]
                rotation = correction["rotation"]
                cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
                cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

                image_metadata = poses["images"][frame_idx]

                # Get camera for this frame
                fovy = image_metadata.get("fovy", [50])
                if isinstance(fovy, list):
                    fovy = fovy[0]
                position = Vector3(image_metadata["location"])
                rot = image_metadata["rotation"]
                rotation_eulers = (Vector3(
                    [np.deg2rad(val % 360.0) for val in rot]) - cor_rotation_eulers) * -1
                position += cor_translation
                rotation_quat = Quaternion.from_eulers(rotation_eulers)

                camera = Camera(fovy=fovy, aspect_ratio=aspect_ratio, position=position, rotation=rotation_quat)

                # Process each prompt's predictions
                georef_frame = {
                    'frame_idx': frame_idx,
                    'imagefile': frame_result.get('imagefile', ''),
                    'prompts': []
                }

                for prompt_data in frame_result.get('prompts', []):
                    georef_prompt = {
                        'prompt': prompt_data.get('prompt', ''),
                        'predictions': []
                    }

                    for pred in prompt_data.get('predictions', []):
                        georef_pred = {
                            'confidence': pred.get('confidence', 0),
                            'world_polygons': []
                        }

                        # Process each polygon
                        for polygon in pred.get('polygons', []):
                            if not polygon:
                                continue

                            try:
                                # Flatten polygon points to label format [x1,y1,x2,y2,...]
                                if isinstance(polygon[0], (list, tuple)):
                                    # List of [x, y] points
                                    label_coords = []
                                    for pt in polygon:
                                        label_coords.extend([pt[0], pt[1]])
                                else:
                                    # Already flat
                                    label_coords = polygon

                                # Project to world coordinates
                                world_coords = label_to_world_coordinates(
                                    label_coords, input_resolution, tri_mesh, camera
                                )

                                if len(world_coords) > 0:
                                    # Apply offsets
                                    world_polygon = []
                                    for pt in world_coords:
                                        world_polygon.append([
                                            float(pt[0] + x_offset),
                                            float(pt[1] + y_offset),
                                            float(pt[2] + z_offset)
                                        ])
                                    georef_pred['world_polygons'].append(world_polygon)

                            except Exception:
                                failed_count += 1
                                continue

                        if georef_pred['world_polygons']:
                            georef_prompt['predictions'].append(georef_pred)

                    if georef_prompt['predictions']:
                        georef_frame['prompts'].append(georef_prompt)

                if georef_frame['prompts']:
                    georef_results.append(georef_frame)

                if progress_fn and idx % 10 == 0:
                    progress = 10 + int((idx / total_frames) * 85)
                    progress_fn(min(progress, 95))

        finally:
            if ctx:
                release_all(ctx)
            if mesh_data is not None:
                del mesh_data
            if texture_data is not None:
                del texture_data
            if tri_mesh is not None:
                del tri_mesh

        # Save geo-referenced results
        output_file = os.path.join(segmentation_folder, "segmentation_georef.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(georef_results, f, indent=2)

        # Count total predictions
        total_predictions = sum(
            len(p['predictions'])
            for r in georef_results
            for p in r['prompts']
        )

        if log_fn:
            log_fn(f"Geo-referenced {len(georef_results)} frames with {total_predictions} predictions")
            if failed_count > 0:
                log_fn(f"Warning: {failed_count} polygons failed to geo-reference")
            log_fn(f"Saved to: {output_file}")

        if progress_fn:
            progress_fn(100)
