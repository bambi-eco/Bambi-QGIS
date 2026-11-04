# -*- coding: utf-8 -*-
"""Heading source of the extracted poses (bambi_detection >= 1.1.0).

The camera yaw of every pose comes from one of four places. Which one is right
depends on the flight: on fast, agile flights the drone body yaws while the
gimbal holds its heading, so the compass heading rotates every frame with the
body. The flight-log (AirData) and SRT gimbal values can also differ by a
constant offset, so the choice is the user's. ``poses.json`` records the
source used and a ``heading_check`` that flags the wobble.
"""
from typing import Any, Dict, Optional

#: Source key (as bambi_detection's ``heading_source``) -> combo box label.
#: The first entry is the default and keeps the poses of earlier versions.
HEADING_SOURCE_LABELS: Dict[str, str] = {
    "airdata_compass": "Compass heading (flight log) - drone body",
    "airdata_gimbal": "Gimbal heading (flight log)",
    "srt_gimbal": "Gimbal heading (SRT, per frame)",
    "srt_drone": "Drone heading (SRT, only some models)",
}
DEFAULT_HEADING_SOURCE = "airdata_compass"


def resolve_heading_source(config: Dict[str, Any]) -> str:
    """Heading source of *config*; missing or unknown means the default."""
    source = config.get("heading_source")
    return source if source in HEADING_SOURCE_LABELS else DEFAULT_HEADING_SOURCE


#: Sources that read the gimbal rather than the drone body.
GIMBAL_SOURCES = frozenset({"airdata_gimbal", "srt_gimbal"})


def photo_heading_source(source: str) -> str:
    """Heading source for bambi_detection's photo extractors.

    Photos have no SRT, so the photo extractors only support the two
    flight-log sources (``PHOTO_HEADING_SOURCES``); an SRT choice falls back
    to the flight-log heading of the same part (gimbal or drone body).
    """
    return "airdata_gimbal" if source in GIMBAL_SOURCES else DEFAULT_HEADING_SOURCE


def legacy_project_heading_source(use_gimbal_heading_raw: str) -> str:
    """Heading source for a project saved before 7.1, which stored a checkbox
    (``Extraction/UseGimbalHeading``) instead; read once on load."""
    return "airdata_gimbal" if use_gimbal_heading_raw == "1" else DEFAULT_HEADING_SOURCE


def heading_check_message(poses: Dict[str, Any], camera_label: str) -> Optional[str]:
    """Log line for the heading check in *poses*, or None when there is nothing to say.

    Only speaks up when the check recommends another source than the one used,
    i.e. when the compass wobbled while the gimbal held still.
    """
    check = poses.get("heading_check")
    if not isinstance(check, dict) or not check.get("prefer_gimbal_heading"):
        return None
    used = poses.get("heading_source", DEFAULT_HEADING_SOURCE)
    if used in GIMBAL_SOURCES:
        return None
    recommended = check.get("recommended_heading_source") or "airdata_gimbal"
    return (
        f"Warning ({camera_label}): the compass heading varies "
        f"{check.get('compass_heading_std_deg')}° (std) over the video, the gimbal heading only "
        f"{check.get('gimbal_heading_std_deg')}°. The drone yawed while the camera held its "
        f"heading, so the frames are rotated by the body's yaw. Set 'Heading source' to "
        f"'{HEADING_SOURCE_LABELS.get(recommended, recommended)}' and extract again.")


def make_timed_pose_extractor(extractor_cls, accessor, config: Dict[str, Any],
                              padding_ms: Optional[int] = None, **kwargs):
    """Build bambi_detection's TimedPoseExtractor with the heading source and
    flight log margin of *config*.

    *padding_ms* overrides the fixed margin of *config* (the automatic margin,
    measured per camera by :func:`core.log_margin.resolve_margin_ms`).

    The class is passed in so this module stays importable without
    bambi_detection. An install older than 1.1.0 does not know the two
    arguments; say so instead of failing with a bare TypeError.
    """
    source = resolve_heading_source(config)
    if padding_ms is None:
        padding_ms = int(config.get("airdata_video_padding_ms") or 0)
    try:
        return extractor_cls(accessor, heading_source=source,
                             airdata_video_padding_ms=padding_ms, **kwargs)
    except TypeError as exc:
        text = str(exc)
        if "heading_source" not in text and "airdata_video_padding_ms" not in text:
            raise
        raise RuntimeError(
            "The installed BAMBI Detection Framework is older than 1.1.0 and does not "
            "support the heading source / flight log margin settings. Update it in the "
            "Dependency Manager.") from exc
