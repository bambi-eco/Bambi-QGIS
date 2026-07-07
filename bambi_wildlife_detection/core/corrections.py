# -*- coding: utf-8 -*-
"""Single source for reading and resolving drone-pose corrections.

``correction.json`` stores a global translation/rotation plus optional
frame-range overrides (``additional``). Two call conventions existed in the
plugin and are both kept here:

* :func:`read_correction` / :func:`correction_for_frame` — the raw-file
  convention of the box projector, click tool and labelling tool
  (``additional`` key, returns the (translation, rotation) tuple).
* :func:`correction_for_frame_config` — the pipeline-config convention of
  ``BambiProcessor`` and the dock widget (``additional_corrections`` key,
  returns a ``{"translation":…, "rotation":…}`` dict).

Consolidated from ``bambi_box_projector.py``, ``bambi_processing.py`` and
``bambi_dock_widget.py`` (which re-export / delegate for compatibility).
"""

import os
import json
from typing import Any, Dict, Tuple


def read_correction(target_folder: str, correction_path: str) -> dict:
    """Load correction.json, return dict with translation/rotation/additional."""
    path = ""
    if correction_path and os.path.isfile(correction_path):
        path = correction_path
    else:
        fallback = os.path.join(target_folder, "correction.json")
        if os.path.isfile(fallback):
            path = fallback

    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # nosec B110
            pass
    return {}


def correction_for_frame(frame_idx: int, corr: dict) -> Tuple[dict, dict]:
    """Return (translation_dict, rotation_dict) for *frame_idx*."""
    default_t = corr.get("translation", {"x": 0.0, "y": 0.0, "z": 0.0})
    default_r = corr.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
    for entry in corr.get("additional", []):
        s = entry.get("start", 0)
        e = entry.get("end", float("inf"))
        if s <= frame_idx <= e:
            return (
                entry.get("translation", default_t),
                entry.get("rotation", default_r),
            )
    return default_t, default_r


def correction_for_frame_config(frame_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get the appropriate correction factors for a given frame index.

    This method checks if there's a frame-range specific correction that
    applies to the given frame index. If found, returns that correction;
    otherwise returns the default correction.

    :param frame_idx: The frame index to get correction for
    :param config: The configuration dictionary containing correction factors
    :return: Dictionary with 'translation' and 'rotation' keys
    """
    # Get default correction
    default_correction = {
        "translation": config.get("translation", {"x": 0, "y": 0, "z": 0}),
        "rotation": config.get("rotation", {"x": 0, "y": 0, "z": 0})
    }

    # Check additional corrections
    additional = config.get("additional_corrections", [])
    for add_corr in additional:
        start = add_corr.get("start", 0)
        end = add_corr.get("end", float('inf'))
        if start <= frame_idx <= end:
            return {
                "translation": add_corr.get("translation", {"x": 0, "y": 0, "z": 0}),
                "rotation": add_corr.get("rotation", {"x": 0, "y": 0, "z": 0})
            }

    return default_correction
