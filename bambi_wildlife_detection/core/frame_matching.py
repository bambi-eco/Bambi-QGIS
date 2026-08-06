# -*- coding: utf-8 -*-
"""Cross-modality frame correspondence, by capture timestamp.

Thermal and RGB are recorded by two cameras on one aircraft at different frame
rates, so "frame 100" means nothing across them — but both stamp every frame
with the same real-world clock, taken from the SRT capture time. Matching by
that clock is therefore the only correspondence that holds, and it is the
foundation of everything cross-modal in the plugin:

* the labelling tool's "Copy labels from RGB/Thermal", which reprojects key
  frames between the two;
* cross-modal track matching (:mod:`core.track_matching`), which has to know
  which two frames to compare detections in before it can compare them.

Lifted verbatim out of :mod:`core.labelling`, where it started life as a
private helper, so both callers share one implementation rather than the second
one growing a near-copy.
"""

import bisect
from datetime import datetime
from typing import List, Optional, Tuple


def pose_epochs(images: List[dict]) -> List[Optional[float]]:
    """Return the capture time (epoch seconds) of every pose image.

    Poses store an ISO-8601 ``timestamp`` (with timezone) taken from the SRT
    capture time, so thermal and RGB frames share the same real-world clock
    and can be matched across modalities. Entries without a parseable
    timestamp yield ``None``.
    """
    epochs: List[Optional[float]] = []
    for img in images:
        ts = img.get("timestamp", "")
        if not ts:
            epochs.append(None)
            continue
        try:
            epochs.append(datetime.fromisoformat(ts).timestamp())
        except Exception:
            epochs.append(None)
    return epochs


class FrameMatcher:
    """Maps a source-modality frame index to the nearest target-modality one.

    Correspondence is by capture time: for a source frame's timestamp the
    target frame with the closest timestamp is returned, together with the
    absolute time difference so callers can reject poor matches.
    """

    def __init__(self, src_images: List[dict], dst_images: List[dict]):
        self._src_epochs = pose_epochs(src_images)
        dst_epochs = pose_epochs(dst_images)
        # Sorted (epoch, dst_index) of target frames that have a timestamp.
        self._pairs = sorted(
            (e, i) for i, e in enumerate(dst_epochs) if e is not None)
        self._keys = [e for e, _ in self._pairs]
        self._bisect = bisect.bisect_left

    @property
    def usable(self) -> bool:
        """True when both sides carry enough timestamps to match by time."""
        return bool(self._pairs) and any(e is not None for e in self._src_epochs)

    def match(self, src_frame: int) -> Optional[Tuple[int, float]]:
        """Return ``(dst_frame, dt_seconds)`` nearest in time, or ``None``."""
        if not (0 <= src_frame < len(self._src_epochs)):
            return None
        target = self._src_epochs[src_frame]
        if target is None or not self._keys:
            return None
        pos = self._bisect(self._keys, target)
        best = None
        for j in (pos - 1, pos):
            if 0 <= j < len(self._pairs):
                epoch, idx = self._pairs[j]
                dt = abs(epoch - target)
                if best is None or dt < best[1]:
                    best = (idx, dt)
        return best

    def matches_within(self, src_frame: int,
                       max_dt: float) -> Optional[int]:
        """The corresponding frame, or ``None`` when nothing is close enough.

        A tolerance is what separates "the other camera saw this moment" from
        "the other camera was not recording" — at the ends of a flight the
        nearest frame in time can be many seconds away, and pairing detections
        across such a gap would invent correspondences.
        """
        found = self.match(src_frame)
        if found is None:
            return None
        dst_frame, dt = found
        return dst_frame if dt <= max_dt else None
