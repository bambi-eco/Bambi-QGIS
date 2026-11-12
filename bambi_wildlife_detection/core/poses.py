# -*- coding: utf-8 -*-
"""Reading ``poses_{t|w}.json`` and its "duplicated" frames (9.0.0).

A thermal camera closes its shutter every so often to recalibrate; while it
is closed the video repeats the previous image. Since bambi_detection 1.2.0
the extraction can keep these frozen frames in the poses file, marked
``"duplicated": true``, instead of dropping them - which keeps the index of
``images`` identical to the video's frame number.

The rule for everything that reads poses: **the list is never shortened.**
A frame is identified by its index everywhere (detections, tracks, labels,
corrections, ``{idx:08d}.tiff``, embeddings), so duplicated frames are
skipped or marked at the point of use. The *pose* of a duplicated frame
(position, rotation, timestamp) is valid; only its image is not.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: The key a frozen frame carries in ``poses["images"][i]``.
DUPLICATED_KEY = "duplicated"

#: What the tools call such a frame.
FROZEN_LABEL = "frozen frame (thermal calibration)"

_CACHE: Dict[Tuple[str, str], Tuple[float, List[dict]]] = {}


def poses_path(target_folder: str, modality: str) -> str:
    return os.path.join(target_folder, f"poses_{modality}.json")


def load_images(target_folder: str, modality: str) -> List[dict]:
    """The ``images`` list of ``poses_{modality}.json``, or ``[]``.

    Cached per file and re-read when it changes; callers must not modify
    the returned list or its entries.
    """
    path = poses_path(target_folder, modality)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return []
    key = (os.path.normcase(os.path.abspath(target_folder)), modality)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            images = json.load(fh).get("images", []) or []
    except Exception:  # an unreadable file has no frames
        images = []
    _CACHE[key] = (stamp, images)
    return images


def is_duplicated(entry) -> bool:
    """True for a pose entry marked as a frozen (repeated) image."""
    return isinstance(entry, dict) and bool(entry.get(DUPLICATED_KEY))


def frame_is_duplicated(images: Sequence[dict], index: int) -> bool:
    return 0 <= index < len(images) and is_duplicated(images[index])


def duplicated_frames(images: Optional[Iterable[dict]]) -> Set[int]:
    """Indices of the duplicated frames of an ``images`` list."""
    return {i for i, entry in enumerate(images or ()) if is_duplicated(entry)}


def without_duplicated(indices: Iterable[int],
                       images: Sequence[dict]) -> List[int]:
    """*indices* without the duplicated frames, order kept."""
    return [i for i in indices if not frame_is_duplicated(images, i)]


def nearest_usable(images: Sequence[dict], index: int) -> Optional[int]:
    """*index*, or the closest frame that is not duplicated (the earlier one
    on a tie); None when there is none."""
    count = len(images)
    if count == 0:
        return None
    index = max(0, min(int(index), count - 1))
    for offset in range(count):
        for candidate in (index - offset, index + offset):
            if 0 <= candidate < count and not is_duplicated(images[candidate]):
                return candidate
    return None


def flagged(target_folder: str, modality: str) -> Set[int]:
    """Duplicated frame indices of a flight's camera (cached read)."""
    return duplicated_frames(load_images(target_folder, modality))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

#: Top-level key the plugin writes into ``poses_t.json``: whether the
#: extraction kept (and flagged) the duplicated frames. Switching it
#: renumbers every frame, so a re-extraction compares against it.
FLAGGED_KEY = "duplicates_flagged"


def extract_with_flag(extract, *args, flag_duplicates: bool = False, **kwargs):
    """Call ``TimedPoseExtractor.extract`` with ``flag_duplicates``.

    :raises RuntimeError: when the installed bambi_detection is older than
        1.2.0 and does not know the argument
    """
    if not flag_duplicates:
        return extract(*args, **kwargs)
    try:
        return extract(*args, flag_duplicates=True, **kwargs)
    except TypeError as exc:
        if "flag_duplicates" not in str(exc):
            raise
        raise RuntimeError(
            "Flagging duplicated (frozen thermal) frames needs bambi_detection "
            "1.2.0 or newer, but an older version is installed. Update it in "
            "the Dependency Manager, or untick 'Keep duplicated thermal "
            "frames' on the Extraction tab.") from exc


def record_flag_setting(path: str, flagged_on: bool) -> int:
    """Note in the poses file whether duplicates were flagged; returns the
    number of duplicated frames in it."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data[FLAGGED_KEY] = bool(flagged_on)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return len(duplicated_frames(data.get("images")))


def flag_setting(target_folder: str, modality: str) -> Optional[bool]:
    """How the existing poses file was extracted: True / False, or None
    when there is no poses file. A file from before 9.0.0 reads False -
    its duplicated frames were dropped."""
    path = poses_path(target_folder, modality)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not data.get("images"):
        return None
    return bool(data.get(FLAGGED_KEY, False))
