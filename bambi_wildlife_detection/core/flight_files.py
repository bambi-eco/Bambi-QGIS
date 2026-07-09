# -*- coding: utf-8 -*-
"""DJI flight-folder conventions: input auto-detection and SRT extraction.

Moved from ``bambi_dock_widget.py`` (whose methods delegate here).
"""

import os
import subprocess  # nosec B404
from typing import Callable, Dict, List, Optional

LogFn = Optional[Callable[[str], None]]


def detect_common_files(video_folder: str, log_fn: LogFn = None) -> Dict[str, str]:
    """Detect the common input files inside a flight's video folder.

    Returns a dict with any of these keys (only when found):

    * ``airdata`` — first ``.csv`` in the folder
    * ``dem`` — first ``.gltf`` / ``.glb``
    * ``dem_metadata`` — matching DEM JSON (same stem, or common
      ``_mesh``/``_dem``/``_metadata`` suffixes)
    * ``thermal_calibration`` / ``rgb_calibration`` — ``T_calib.json`` /
      ``W_calib.json``
    * ``correction`` — ``correction.json``
    * ``target_folder`` — the conventional ``qgis/`` output subfolder
      (always present; may not exist on disk yet)

    Purely detective — nothing is created or modified.
    """
    found: Dict[str, str] = {}
    try:
        folder_files = os.listdir(video_folder)
    except Exception as e:
        if log_fn:
            log_fn(f"Warning: Could not list folder contents: {e}")
        folder_files = []

    csv_files = [f for f in folder_files if f.lower().endswith('.csv')]
    if csv_files:
        found["airdata"] = os.path.join(video_folder, csv_files[0])

    dem_files = [f for f in folder_files if f.lower().endswith(('.gltf', '.glb'))]
    if dem_files:
        found["dem"] = os.path.join(video_folder, dem_files[0])
        dem_base = os.path.splitext(dem_files[0])[0]
        for suffix in (".json", "_mesh.json", "_dem.json", "_metadata.json"):
            candidate = os.path.join(video_folder, dem_base + suffix)
            if os.path.exists(candidate):
                found["dem_metadata"] = candidate
                break

    for key, name in (("thermal_calibration", "T_calib.json"),
                      ("rgb_calibration", "W_calib.json"),
                      ("correction", "correction.json")):
        path = os.path.join(video_folder, name)
        if os.path.exists(path):
            found[key] = path

    found["target_folder"] = os.path.join(video_folder, "qgis")
    return found


def extract_embedded_srts(video_paths: List[str], cache_dir: str,
                          log_fn: LogFn = None) -> List[str]:
    """Extract embedded SRT subtitle streams from video files using ffmpeg.

    *cache_dir* acts as a persistent cache — if a .srt file for a video
    already exists there it is reused without re-running ffmpeg.
    Returns a list of extracted .srt paths (one per video that succeeded).
    """
    def log(message):
        if log_fn:
            log_fn(message)

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        log("ERROR: imageio-ffmpeg not available — cannot extract embedded SRT.")
        return []

    extracted = []
    for vpath in video_paths:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        out_srt = os.path.join(cache_dir, stem + ".srt")
        if os.path.exists(out_srt) and os.path.getsize(out_srt) > 0:
            extracted.append(out_srt)
            continue
        try:
            result = subprocess.run(  # nosec B603
                [ffmpeg_exe, "-i", vpath, "-map", "0:s:0", "-y", out_srt],
                capture_output=True, text=True, timeout=60
            )
            if os.path.exists(out_srt) and os.path.getsize(out_srt) > 0:
                extracted.append(out_srt)
                log(f"Extracted embedded SRT from {os.path.basename(vpath)}")
            else:
                log(
                    f"WARNING: No subtitle stream found in {os.path.basename(vpath)}. "
                    f"ffmpeg stderr: {result.stderr.strip()[-200:]}"
                )
        except subprocess.TimeoutExpired:
            log(f"ERROR: SRT extraction timed out for {os.path.basename(vpath)}")
        except Exception as exc:
            log(f"ERROR: SRT extraction failed for {os.path.basename(vpath)}: {exc}")

    return extracted
