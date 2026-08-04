# -*- coding: utf-8 -*-
"""Declarative schema for the plugin configuration saved in QGIS projects.

Replaces the three hand-maintained key lists in ``bambi_dock_widget``
(``save_config_to_project`` / ``load_config_from_project``): every project
variable is described exactly once here, and the dock widget only provides
a widget getter/setter per key. Save and load are driven by the same table,
so the two directions can no longer drift apart.

Storage kinds (matching the pre-refactor QgsProject entry usage):

``str``
    ``writeEntry`` / ``readEntry`` string; missing → ``""``.
``bool01``
    Stored as the string ``"1"`` / ``"0"``. ``truthy`` selects the historic
    load interpretation (``"eq1"``: value == "1"; ``"ne0"``: value != "0").
    ``default`` applies when the entry is missing/empty; ``None`` means
    "skip" (leave the widget untouched).
``int_str``
    Written as ``str(int)`` via ``writeEntry``, read back through the
    double reader and truncated to int (historic behaviour).
``double``
    ``writeEntryDouble`` / ``readDoubleEntry``.
``int_double``
    ``writeEntryDouble`` on save, read as double and truncated to int
    (used for spin boxes and combo indices).
``bool``
    ``writeEntryBool`` / ``readBoolEntry`` with a typed default.

``Correction/AdditionalCorrections`` (a JSON list) is intentionally not in
the table — it is bound to a list widget, not a value widget — and the SAM3
API key is intentionally never saved.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple


class ConfigEntry(NamedTuple):
    key: str
    kind: str
    default: Any = None
    truthy: str = "eq1"     # bool01 only: "eq1" or "ne0"


KINDS = frozenset({"str", "bool01", "int_str", "double", "int_double", "bool"})

# Widget roles (see ``WIDGET_BINDINGS``). Purely a get/set protocol name; the
# dock widget maps each to the concrete Qt call:
#   line               QLineEdit.text() / setText()
#   crs                like ``line`` but a blank value loads as EPSG:32633
#   text               QTextEdit.toPlainText() / setPlainText()
#   check              QCheckBox.isChecked() / setChecked()
#   spin               Q(Double)SpinBox.value() / setValue()
#   combo_text         QComboBox currentText(); load = findText, set if found
#   combo_text_editable  like combo_text but falls back to setCurrentText()
#   combo_index        QComboBox currentIndex(); load = bounds-checked setter
ROLES = frozenset({
    "line", "crs", "text", "check", "spin",
    "combo_text", "combo_text_editable", "combo_index",
})

# Entries in the order the pre-refactor loader applied them (widget setters
# can trigger Qt signals, so application order is part of the behaviour).
CONFIG_ENTRIES: List[ConfigEntry] = [
    # ===== Input Mode =====
    ConfigEntry("Input/VideoMode", "bool01", None, "ne0"),
    ConfigEntry("Input/EmbeddedSrt", "bool01", None, "ne0"),
    # ===== Video Input Paths =====
    ConfigEntry("Input/ThermalVideoPaths", "str"),
    ConfigEntry("Input/ThermalSrtPaths", "str"),
    ConfigEntry("Input/ThermalCalibrationPath", "str"),
    ConfigEntry("Input/ThermalCalibrationPreset", "str"),
    ConfigEntry("Input/RgbVideoPaths", "str"),
    ConfigEntry("Input/RgbSrtPaths", "str"),
    ConfigEntry("Input/RgbCalibrationPath", "str"),
    ConfigEntry("Input/RgbCalibrationPreset", "str"),
    # ===== Photo Input Paths =====
    ConfigEntry("Input/ThermalPhotoDir", "str"),
    ConfigEntry("Input/ThermalPhotoCalibrationPath", "str"),
    ConfigEntry("Input/ThermalPhotoCalibrationPreset", "str"),
    ConfigEntry("Input/RgbPhotoDir", "str"),
    ConfigEntry("Input/RgbPhotoCalibrationPath", "str"),
    ConfigEntry("Input/RgbPhotoCalibrationPreset", "str"),
    ConfigEntry("Input/ThermalPhotoFilter", "bool01", False, "eq1"),
    ConfigEntry("Input/RgbPhotoFilter", "bool01", False, "eq1"),
    ConfigEntry("Config/Timezone", "str"),
    ConfigEntry("Config/TimezoneAuto", "bool01", True, "ne0"),
    ConfigEntry("Input/ThermalVisColormap", "str"),
    ConfigEntry("Input/ThermalVisLoEnable", "bool01", False, "eq1"),
    ConfigEntry("Input/ThermalVisLoValue", "double", 0.0),
    ConfigEntry("Input/ThermalVisHiEnable", "bool01", False, "eq1"),
    ConfigEntry("Input/ThermalVisHiValue", "double", 100.0),
    # The curve itself (Input/ThermalVisCurve, a JSON dict) is intentionally
    # not in the table — it is bound to a dialog, not a value widget, and is
    # saved/loaded separately like Correction/AdditionalCorrections.
    ConfigEntry("Input/ThermalVisMode", "int_double", 0),
    # ===== Extraction =====
    ConfigEntry("Extraction/Skip", "int_str", 0),
    ConfigEntry("Extraction/LimitEnable", "bool01", False, "eq1"),
    ConfigEntry("Extraction/LimitValue", "int_str", 100),
    ConfigEntry("Extraction/SamplingRateEnable", "bool01", False, "eq1"),
    ConfigEntry("Extraction/SamplingRate", "int_str", 5),
    ConfigEntry("Extraction/PreserveAspectRatio", "bool01", False, "eq1"),
    ConfigEntry("Extraction/UseGimbalHeading", "bool01", False, "eq1"),
    # ===== Common Inputs =====
    ConfigEntry("Input/AirdataPath", "str"),
    ConfigEntry("Input/DemPath", "str"),
    ConfigEntry("Input/DemMetadataPath", "str"),
    ConfigEntry("Input/DemPadding", "int_double", 30),
    ConfigEntry("Input/GeotiffInputPath", "str"),
    ConfigEntry("Input/GeotiffSimplifyFactor", "int_double", 2),
    ConfigEntry("Input/CorrectionPath", "str"),
    ConfigEntry("Input/TargetFolder", "str"),
    ConfigEntry("Input/TargetCrs", "str"),
    # Legacy text outputs alongside the 6.0 store; see
    # EXCHANGE_FORMAT_PLAN.md §9. On by default for 6.x.
    ConfigEntry("Input/WriteLegacyTextOutputs", "bool", True),
    # ===== Detection Settings =====
    ConfigEntry("Detection/ThermalModelPath", "str"),
    ConfigEntry("Detection/RgbModelPath", "str"),
    ConfigEntry("Detection/Confidence", "double", 0.5),
    ConfigEntry("Detection/AllFrames", "bool", True),
    ConfigEntry("Detection/StartFrame", "int_double", 0),
    ConfigEntry("Detection/EndFrame", "int_double", 999999),
    ConfigEntry("Detection/SampleRate", "int_double", 1),
    # ===== TRex Settings =====
    ConfigEntry("TRex/NpzDir", "str"),
    ConfigEntry("TRex/AlreadyUndistorted", "bool", False),
    # ===== Correction Settings =====
    ConfigEntry("Correction/RotationUnit", "int_double", 0),
    ConfigEntry("Correction/TransX", "double", 0.0),
    ConfigEntry("Correction/TransY", "double", 0.0),
    ConfigEntry("Correction/TransZ", "double", 0.0),
    ConfigEntry("Correction/RotX", "double", 0.0),
    ConfigEntry("Correction/RotY", "double", 0.0),
    ConfigEntry("Correction/RotZ", "double", 0.0),
    # ===== Tracking Settings =====
    ConfigEntry("Tracking/TrackerType", "int_double", 0),
    ConfigEntry("Tracking/ReidModel", "int_double", 0),
    ConfigEntry("Tracking/CustomReidPath", "str"),
    ConfigEntry("Tracking/TrackerParams", "str"),
    ConfigEntry("Tracking/IouThreshold", "double", 0.3),
    ConfigEntry("Tracking/MaxAge", "int_double", -1),
    ConfigEntry("Tracking/MaxCenterDist", "double", 0.2),
    ConfigEntry("Tracking/TrackerMode", "int_double", 1),
    ConfigEntry("Tracking/ClassAware", "bool", True),
    ConfigEntry("Tracking/Interpolate", "bool", True),
    # ===== Field of View Settings =====
    ConfigEntry("FoV/UseMask", "bool", False),
    ConfigEntry("FoV/MaskPath", "str"),
    ConfigEntry("FoV/MaskSimplify", "double", 2.0),
    ConfigEntry("FoV/AllFrames", "bool", True),
    ConfigEntry("FoV/StartFrame", "int_double", 0),
    ConfigEntry("FoV/EndFrame", "int_double", 999999),
    ConfigEntry("FoV/SampleRate", "int_double", 1),
    # ===== ALFS Settings =====
    ConfigEntry("ALFS/Resolution", "double", 0.05),
    ConfigEntry("ALFS/AllFrames", "bool", True),
    ConfigEntry("ALFS/StartFrame", "int_double", 0),
    ConfigEntry("ALFS/EndFrame", "int_double", 999999),
    ConfigEntry("ALFS/Crop", "bool", True),
    ConfigEntry("ALFS/Overviews", "bool", True),
    ConfigEntry("ALFS/TileSize", "int_double", 8192),
    ConfigEntry("ALFS/FrameStep", "int_double", 1),
    ConfigEntry("ALFS/SamplingMode", "bool", False),
    ConfigEntry("ALFS/SamplingRate", "int_double", 10),
    ConfigEntry("ALFS/SamplingRange", "int_double", 5),
    # ===== GeoTIFF Export Settings =====
    ConfigEntry("GeoTIFF/EdgeErosionPx", "int_double", 2),
    # ===== Orthomosaic Settings =====
    ConfigEntry("Ortho/Method", "int_double", 0),
    ConfigEntry("Ortho/AllFrames", "bool", True),
    ConfigEntry("Ortho/StartFrame", "int_double", 0),
    ConfigEntry("Ortho/EndFrame", "int_double", 999999),
    # ===== SAM3 Settings (API key intentionally NOT saved) =====
    ConfigEntry("SAM3/Prompts", "str"),
    ConfigEntry("SAM3/Confidence", "double", 0.5),
    ConfigEntry("SAM3/AllFrames", "bool", True),
    ConfigEntry("SAM3/StartFrame", "int_double", 0),
    ConfigEntry("SAM3/EndFrame", "int_double", 999999),
    ConfigEntry("SAM3/FrameStep", "int_double", 1),
    # ===== Flight Route Settings =====
    ConfigEntry("FlightRoute/FrameMarkersEnabled", "bool", True),
    ConfigEntry("FlightRoute/FrameMarkerInterval", "int_double", 100),
    ConfigEntry("FlightRoute/IncludeFrameZero", "bool", False),
    ConfigEntry("FlightRoute/ImageLabelsEnabled", "bool", False),
    ConfigEntry("FlightRoute/ImageLabelInterval", "int_double", 100),
    ConfigEntry("FlightRoute/DistanceMarkersEnabled", "bool", False),
    ConfigEntry("FlightRoute/DistanceMarkerInterval", "int_double", 100),
    ConfigEntry("FlightRoute/IncludeDistanceStart", "bool", False),
    ConfigEntry("FlightRoute/TimeMarkersEnabled", "bool", False),
    ConfigEntry("FlightRoute/TimeMarkerInterval", "int_double", 10),
    ConfigEntry("FlightRoute/TimeMarkerTypeIndex", "int_double", 0),
    ConfigEntry("FlightRoute/IncludeTimeStart", "bool", False),
    ConfigEntry("FlightRoute/CameraComboIndex", "int_double", 0),
]

CONFIG_KEYS = [entry.key for entry in CONFIG_ENTRIES]


# Binding of each config key to a dock-widget attribute name and its get/set
# role. These are ``BambiDockWidget`` attribute names (plain strings — no Qt
# import here); the dock resolves them with ``getattr`` and applies the role.
# The key set must match ``CONFIG_KEYS`` exactly (enforced by check_schema and
# a unit test), so save and load can never reference a key the other omits.
WIDGET_BINDINGS: Dict[str, Tuple[str, str]] = {
    "Input/VideoMode": ("video_mode_check", "check"),
    "Input/EmbeddedSrt": ("embedded_srt_check", "check"),
    "Input/ThermalVideoPaths": ("thermal_video_paths_edit", "line"),
    "Input/ThermalSrtPaths": ("thermal_srt_paths_edit", "line"),
    "Input/ThermalCalibrationPath": ("thermal_calibration_path_edit", "line"),
    "Input/ThermalCalibrationPreset": ("thermal_calib_preset_combo", "combo_text"),
    "Input/RgbVideoPaths": ("rgb_video_paths_edit", "line"),
    "Input/RgbSrtPaths": ("rgb_srt_paths_edit", "line"),
    "Input/RgbCalibrationPath": ("rgb_calibration_path_edit", "line"),
    "Input/RgbCalibrationPreset": ("rgb_calib_preset_combo", "combo_text"),
    "Input/ThermalPhotoDir": ("thermal_photo_dir_edit", "line"),
    "Input/ThermalPhotoCalibrationPath": ("thermal_photo_calibration_path_edit", "line"),
    "Input/ThermalPhotoCalibrationPreset": ("thermal_photo_calib_preset_combo", "combo_text"),
    "Input/RgbPhotoDir": ("rgb_photo_dir_edit", "line"),
    "Input/RgbPhotoCalibrationPath": ("rgb_photo_calibration_path_edit", "line"),
    "Input/RgbPhotoCalibrationPreset": ("rgb_photo_calib_preset_combo", "combo_text"),
    "Input/ThermalPhotoFilter": ("thermal_photo_filter_check", "check"),
    "Input/RgbPhotoFilter": ("rgb_photo_filter_check", "check"),
    "Config/Timezone": ("timezone_combo", "combo_text_editable"),
    "Config/TimezoneAuto": ("tz_auto_check", "check"),
    "Input/ThermalVisColormap": ("thermal_vis_cmap_combo", "combo_text"),
    "Input/ThermalVisLoEnable": ("thermal_vis_lo_check", "check"),
    "Input/ThermalVisLoValue": ("thermal_vis_lo_spin", "spin"),
    "Input/ThermalVisHiEnable": ("thermal_vis_hi_check", "check"),
    "Input/ThermalVisHiValue": ("thermal_vis_hi_spin", "spin"),
    "Input/ThermalVisMode": ("thermal_vis_mode_combo", "combo_index"),
    "Extraction/Skip": ("extract_skip_spin", "spin"),
    "Extraction/LimitEnable": ("extract_limit_check", "check"),
    "Extraction/LimitValue": ("extract_limit_spin", "spin"),
    "Extraction/SamplingRateEnable": ("extract_sampling_rate_check", "check"),
    "Extraction/SamplingRate": ("extract_sampling_rate_spin", "spin"),
    "Extraction/PreserveAspectRatio": ("preserve_aspect_ratio_check", "check"),
    "Extraction/UseGimbalHeading": ("use_gimbal_heading_check", "check"),
    "Input/AirdataPath": ("airdata_path_edit", "line"),
    "Input/DemPath": ("dem_path_edit", "line"),
    "Input/DemMetadataPath": ("dem_metadata_path_edit", "line"),
    "Input/DemPadding": ("dem_padding_spin", "spin"),
    "Input/GeotiffInputPath": ("geotiff_input_path_edit", "line"),
    "Input/GeotiffSimplifyFactor": ("geotiff_simplify_spin", "spin"),
    "Input/CorrectionPath": ("correction_path_edit", "line"),
    "Input/TargetFolder": ("target_folder_edit", "line"),
    "Input/TargetCrs": ("target_crs_edit", "crs"),
    "Input/WriteLegacyTextOutputs": ("legacy_text_outputs_check", "check"),
    "Detection/ThermalModelPath": ("thermal_model_path_edit", "line"),
    "Detection/RgbModelPath": ("rgb_model_path_edit", "line"),
    "Detection/Confidence": ("confidence_spin", "spin"),
    "Detection/AllFrames": ("detect_all_frames_check", "check"),
    "Detection/StartFrame": ("detect_start_frame_spin", "spin"),
    "Detection/EndFrame": ("detect_end_frame_spin", "spin"),
    "Detection/SampleRate": ("detect_sample_rate_spin", "spin"),
    "TRex/NpzDir": ("trex_npz_dir_edit", "line"),
    "TRex/AlreadyUndistorted": ("trex_undistorted_check", "check"),
    "Correction/RotationUnit": ("rotation_unit_combo", "combo_index"),
    "Correction/TransX": ("trans_x_spin", "spin"),
    "Correction/TransY": ("trans_y_spin", "spin"),
    "Correction/TransZ": ("trans_z_spin", "spin"),
    "Correction/RotX": ("rot_x_spin", "spin"),
    "Correction/RotY": ("rot_y_spin", "spin"),
    "Correction/RotZ": ("rot_z_spin", "spin"),
    "Tracking/TrackerType": ("tracker_backend_combo", "combo_index"),
    "Tracking/ReidModel": ("reid_model_combo", "combo_index"),
    "Tracking/CustomReidPath": ("custom_reid_path_edit", "line"),
    "Tracking/TrackerParams": ("tracker_params_edit", "text"),
    "Tracking/IouThreshold": ("iou_threshold_spin", "spin"),
    "Tracking/MaxAge": ("max_age_spin", "spin"),
    "Tracking/MaxCenterDist": ("max_center_dist_spin", "spin"),
    "Tracking/TrackerMode": ("tracker_mode_combo", "combo_index"),
    "Tracking/ClassAware": ("class_aware_check", "check"),
    "Tracking/Interpolate": ("interpolate_check", "check"),
    "FoV/UseMask": ("use_fov_mask_check", "check"),
    "FoV/MaskPath": ("fov_mask_path_edit", "line"),
    "FoV/MaskSimplify": ("mask_simplify_spin", "spin"),
    "FoV/AllFrames": ("fov_all_frames_check", "check"),
    "FoV/StartFrame": ("fov_start_frame_spin", "spin"),
    "FoV/EndFrame": ("fov_end_frame_spin", "spin"),
    "FoV/SampleRate": ("fov_sample_rate_spin", "spin"),
    "ALFS/Resolution": ("alfs_resolution_spin", "spin"),
    "ALFS/AllFrames": ("alfs_all_frames_check", "check"),
    "ALFS/StartFrame": ("alfs_start_frame_spin", "spin"),
    "ALFS/EndFrame": ("alfs_end_frame_spin", "spin"),
    "ALFS/Crop": ("alfs_crop_check", "check"),
    "ALFS/Overviews": ("alfs_overviews_check", "check"),
    "ALFS/TileSize": ("alfs_tile_size_spin", "spin"),
    "ALFS/FrameStep": ("alfs_frame_step_spin", "spin"),
    "ALFS/SamplingMode": ("alfs_sampling_check", "check"),
    "ALFS/SamplingRate": ("alfs_sampling_rate_spin", "spin"),
    "ALFS/SamplingRange": ("alfs_sampling_range_spin", "spin"),
    "GeoTIFF/EdgeErosionPx": ("geotiff_edge_erosion_spin", "spin"),
    "Ortho/Method": ("ortho_method_combo", "combo_index"),
    "Ortho/AllFrames": ("ortho_all_frames_check", "check"),
    "Ortho/StartFrame": ("ortho_start_frame_spin", "spin"),
    "Ortho/EndFrame": ("ortho_end_frame_spin", "spin"),
    "SAM3/Prompts": ("sam3_prompts_edit", "text"),
    "SAM3/Confidence": ("sam3_confidence_spin", "spin"),
    "SAM3/AllFrames": ("sam3_all_frames_check", "check"),
    "SAM3/StartFrame": ("sam3_start_frame_spin", "spin"),
    "SAM3/EndFrame": ("sam3_end_frame_spin", "spin"),
    "SAM3/FrameStep": ("sam3_step_spin", "spin"),
    "FlightRoute/FrameMarkersEnabled": ("frame_markers_enabled_check", "check"),
    "FlightRoute/FrameMarkerInterval": ("frame_marker_interval_spin", "spin"),
    "FlightRoute/IncludeFrameZero": ("frame_marker_include_zero_check", "check"),
    "FlightRoute/ImageLabelsEnabled": ("image_labels_enabled_check", "check"),
    "FlightRoute/ImageLabelInterval": ("image_label_interval_spin", "spin"),
    "FlightRoute/DistanceMarkersEnabled": ("distance_markers_enabled_check", "check"),
    "FlightRoute/DistanceMarkerInterval": ("distance_marker_interval_spin", "spin"),
    "FlightRoute/IncludeDistanceStart": ("distance_marker_include_start_check", "check"),
    "FlightRoute/TimeMarkersEnabled": ("time_markers_enabled_check", "check"),
    "FlightRoute/TimeMarkerInterval": ("time_marker_interval_spin", "spin"),
    "FlightRoute/TimeMarkerTypeIndex": ("time_marker_type_combo", "combo_index"),
    "FlightRoute/IncludeTimeStart": ("time_marker_include_start_check", "check"),
    "FlightRoute/CameraComboIndex": ("flight_route_camera_combo", "combo_index"),
}


def save_config_entries(
    values: Dict[str, Any],
    write_str: Callable[[str, str], Any],
    write_double: Callable[[str, float], Any],
    write_bool: Callable[[str, bool], Any],
) -> None:
    """Write every schema entry from *values* through the given writers.

    *values* must contain a value for every schema key (a missing key is a
    programming error in the widget bindings and raises ``KeyError``).
    """
    for entry in CONFIG_ENTRIES:
        value = values[entry.key]
        if entry.kind == "str":
            write_str(entry.key, str(value))
        elif entry.kind == "bool01":
            write_str(entry.key, "1" if value else "0")
        elif entry.kind == "int_str":
            write_str(entry.key, str(int(value)))
        elif entry.kind in ("double", "int_double"):
            write_double(entry.key, float(value))
        elif entry.kind == "bool":
            write_bool(entry.key, bool(value))
        else:  # pragma: no cover — guarded by the schema sanity test
            raise ValueError(f"Unknown kind {entry.kind!r} for {entry.key}")


def load_config_entries(
    read_str: Callable[[str, str], str],
    read_double: Callable[[str, float], float],
    read_bool: Callable[[str, bool], bool],
) -> List[Tuple[str, Any]]:
    """Read every schema entry through the given readers.

    Returns ``(key, typed_value)`` pairs in schema order. Entries whose
    stored value is missing yield their default; ``bool01`` entries with a
    ``None`` default are omitted entirely (historic "leave the widget
    untouched" behaviour).
    """
    result: List[Tuple[str, Any]] = []
    for entry in CONFIG_ENTRIES:
        value: Any
        if entry.kind == "str":
            value = read_str(entry.key, "")
        elif entry.kind == "bool01":
            raw = read_str(entry.key, "")
            if raw == "":
                if entry.default is None:
                    continue
                value = entry.default
            elif entry.truthy == "eq1":
                value = raw == "1"
            else:
                value = raw != "0"
        elif entry.kind in ("int_str", "int_double"):
            value = int(read_double(entry.key, float(entry.default)))
        elif entry.kind == "double":
            value = read_double(entry.key, float(entry.default))
        elif entry.kind == "bool":
            value = read_bool(entry.key, entry.default)
        else:  # pragma: no cover — guarded by the schema sanity test
            raise ValueError(f"Unknown kind {entry.kind!r} for {entry.key}")
        result.append((entry.key, value))
    return result


def check_schema() -> Optional[str]:
    """Return an error description when the schema is inconsistent, else None."""
    seen = set()
    for entry in CONFIG_ENTRIES:
        if entry.key in seen:
            return f"duplicate key {entry.key}"
        seen.add(entry.key)
        if entry.kind not in KINDS:
            return f"unknown kind {entry.kind!r} for {entry.key}"
        if "/" not in entry.key:
            return f"key without scope: {entry.key}"
        if entry.kind == "bool01":
            if entry.default not in (None, True, False):
                return f"bool01 default must be None/bool: {entry.key}"
            if entry.truthy not in ("eq1", "ne0"):
                return f"bool01 truthy must be eq1/ne0: {entry.key}"
        elif entry.kind == "bool" and not isinstance(entry.default, bool):
            return f"bool default must be bool: {entry.key}"
        elif entry.kind in ("int_str", "int_double") and not isinstance(entry.default, int):
            return f"int default must be int: {entry.key}"
        elif entry.kind == "double" and not isinstance(entry.default, float):
            return f"double default must be float: {entry.key}"

    entry_keys = {entry.key for entry in CONFIG_ENTRIES}
    binding_keys = set(WIDGET_BINDINGS)
    if entry_keys != binding_keys:
        missing = entry_keys - binding_keys
        extra = binding_keys - entry_keys
        return f"WIDGET_BINDINGS mismatch (missing {missing}, extra {extra})"
    for key, (attr, role) in WIDGET_BINDINGS.items():
        if role not in ROLES:
            return f"unknown role {role!r} for {key}"
        if not attr:
            return f"empty widget attr for {key}"
    return None
