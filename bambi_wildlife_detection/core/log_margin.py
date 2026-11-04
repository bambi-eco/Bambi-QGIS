# -*- coding: utf-8 -*-
"""Automatic flight log margin around a video recording.

bambi_detection's ``TimedPoseExtractor`` keeps only the AirData rows flagged as
recording (``isVideo``), plus ``airdata_video_padding_ms`` on both sides. DJI
can set the flag about 1 s after the video starts (1.1 s on the bat flight), so
without a margin the first frames lie before the kept log and their positions
are extrapolated. The margin only has to cover that overhang: each frame is
interpolated between the two log rows around it, so a larger margin gives the
same poses.

"Auto" measures the overhang here, before extraction, with bambi_detection's
own SRT and AirData readers (same clock and timezone handling as the
extraction), and passes the result as a plain ``airdata_video_padding_ms``.
"""
import math
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

#: Added to the measured overhang: the SRT/AirData clock offset is only fitted
#: during extraction (0.18 s on the bat flight), and extra margin is harmless.
AUTO_MARGIN_BUFFER_MS = 1000
#: Upper bound, the same as the margin spin box. More than that means the
#: video and the flagged rows hardly belong together (timezone? wrong log?).
AUTO_MARGIN_MAX_MS = 30000


def auto_margin_ms(first_flagged: datetime, last_flagged: datetime,
                   video_start: datetime, video_end: datetime) -> Tuple[int, int]:
    """Margin (ms) that covers the video time outside the flagged log rows.

    :return: ``(margin_ms, overhang_ms)``. The overhang is the larger of
        *flag after video start* and *video end after last flag* (0 when the
        flags cover the video); the margin adds AUTO_MARGIN_BUFFER_MS and is
        capped at AUTO_MARGIN_MAX_MS.
    """
    lead_ms = (first_flagged - video_start).total_seconds() * 1000
    tail_ms = (video_end - last_flagged).total_seconds() * 1000
    overhang = int(math.ceil(max(0.0, lead_ms, tail_ms)))
    return min(overhang + AUTO_MARGIN_BUFFER_MS, AUTO_MARGIN_MAX_MS), overhang


def measure_auto_margin_ms(extractor_cls, air_data_path: str, srt_paths: Sequence[str],
                           timezone) -> Tuple[int, int]:
    """Read the SRT and the flight log the way *extractor_cls*
    (``TimedPoseExtractor``) does and return :func:`auto_margin_ms`."""
    srt_frames, _num_videos, _frame_to_video = extractor_cls.get_srt_frames(
        list(srt_paths), timezone=timezone)
    if not srt_frames:
        raise ValueError("The SRT files contain no frames")
    _start, flagged = extractor_cls.get_air_data_frames(
        air_data_path, srt_frames[0].timestamp, timezone=timezone, padding_ms=0)
    return auto_margin_ms(flagged[0].datetime, flagged[-1].datetime,
                          srt_frames[0].timestamp, srt_frames[-1].timestamp)


def resolve_margin_ms(config: Dict[str, Any], extractor_cls, air_data_path: str,
                      srt_paths: Sequence[str], timezone, camera_label: str,
                      log_fn: Optional[Callable[[str], None]] = None) -> int:
    """Flight log margin (ms) to pass to the extractor for one camera.

    The fixed value of *config* unless ``airdata_video_padding_auto`` is set,
    in which case it is measured from this camera's SRT and logged.
    """
    if not config.get("airdata_video_padding_auto"):
        return int(config.get("airdata_video_padding_ms") or 0)
    margin, overhang = measure_auto_margin_ms(extractor_cls, air_data_path, srt_paths, timezone)
    if log_fn:
        if overhang + AUTO_MARGIN_BUFFER_MS > AUTO_MARGIN_MAX_MS:
            log_fn(f"Warning ({camera_label}): the video extends {overhang / 1000:.1f} s beyond the "
                   f"flight log rows flagged as recording; the margin is capped at "
                   f"{AUTO_MARGIN_MAX_MS / 1000:.0f} s. Check the timezone and that the flight "
                   f"log belongs to this video.")
        else:
            log_fn(f"Flight log margin ({camera_label}, auto): {margin / 1000:.1f} s "
                   f"(video extends {overhang / 1000:.1f} s beyond the recording flag).")
    return margin
