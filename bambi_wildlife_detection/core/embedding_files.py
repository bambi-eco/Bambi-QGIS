# -*- coding: utf-8 -*-
"""Embedding vectors on disk, one ``.npz`` per frame.

The vectors are hours of GPU time and are useful well beyond this plugin — a
notebook, the authors' own analysis scripts, a future re-identification step —
so they are written as ordinary files mirroring the frames folder rather than
buried in a GeoPackage::

    frames_t/       frame_000123.jpg    frame_000124.jpg
    embeddings_t/
        non_geo/    frame_000123.npz    frame_000124.npz
        geo_2k/     …

**There is no sidecar metadata and no index file.** Everything describing a run
lives in ``embedding_runs``; where a vector lives is pure convention, resolved
from facts the caller already has:

=========  =================================================================
folder     ``embedding_runs.folder``
file       the frame's ``imagefile`` from the poses, extension swapped
array      ``det_<detection_id>``
=========  =================================================================

so the only thing the store records is *whether* a detection has been embedded.
A second copy of derivable data is a second thing that can disagree.

One file per frame rather than one per detection: tens of thousands of ~5 KB
files are slow to create and slower to scan on Windows, while one per frame
gives exactly the file count of ``frames_{m}/`` and mirrors it one-to-one.
"""

import os
import re
import zipfile
from typing import Dict, Iterable, List, Optional

#: Prefix of the array holding one detection's vector inside a frame's archive.
KEY_PREFIX = "det_"

#: Frame file name used when a pose carries no ``imagefile``. Matches the
#: fallback ``BambiProcessor`` already applies when writing outputs, so both
#: sides agree on what an unnamed frame is called.
_UNNAMED = "frame_%06d"


def array_key(detection_id: int) -> str:
    """Name of a detection's array inside its frame archive."""
    return f"{KEY_PREFIX}{int(detection_id)}"


def detection_of_key(key: str) -> Optional[int]:
    """The detection id an array name refers to, or ``None`` if it is not one."""
    match = re.fullmatch(rf"{KEY_PREFIX}(\d+)", key or "")
    return int(match.group(1)) if match else None


def frame_file_name(frame: int, imagefile: str = "") -> str:
    """The ``.npz`` a frame's vectors go in.

    Named after the image it came from so the correspondence is obvious in a
    file manager — ``DJI_0042.JPG`` embeds to ``DJI_0042.npz``, and an
    extracted ``frame_000123.jpg`` to ``frame_000123.npz``. Frames with no
    recorded image name fall back to the frame number, which is the same
    convention the rest of the pipeline uses.

    This function is the *only* place the naming lives, so the writer and the
    reader cannot drift apart.
    """
    base = os.path.basename((imagefile or "").strip())
    if not base:
        base = _UNNAMED % int(frame)
    stem, _extension = os.path.splitext(base)
    return f"{stem}.npz"


def run_folder(target_folder: str, modality: str, projection: str) -> str:
    """Folder holding one run's vectors, relative paths resolved."""
    return os.path.join(target_folder, f"embeddings_{modality}", projection)


def relative_run_folder(modality: str, projection: str) -> str:
    """What ``embedding_runs.folder`` records — relative, so the flight moves."""
    return f"embeddings_{modality}/{projection}"


def frame_path(target_folder: str, modality: str, projection: str,
               frame: int, imagefile: str = "") -> str:
    """Full path of a frame's archive."""
    return os.path.join(run_folder(target_folder, modality, projection),
                        frame_file_name(frame, imagefile))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_frame(path: str, vectors: Dict[int, "object"]) -> int:
    """Write one frame's vectors, replacing whatever was there.

    *vectors* maps ``detection_id`` onto a 1-D float32 array. Written whole
    rather than appended to: a frame is embedded in one pass, and a partial
    archive that looked complete would be worse than one that is absent.
    """
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {array_key(detection_id): np.asarray(vector, dtype=np.float32)
               for detection_id, vector in vectors.items()}
    np.savez(path, **payload)
    return len(payload)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_frame(path: str) -> Dict[int, "object"]:
    """Every vector in a frame's archive, keyed by detection id.

    A missing or unreadable file reads as empty rather than raising: the store
    records membership, the files are the truth, and the honest response to a
    vector that is gone is to re-embed it (see
    ``classification_store.forget_embedded``).
    """
    import numpy as np

    if not os.path.isfile(path):
        return {}
    try:
        with np.load(path) as archive:
            result = {}
            for key in archive.files:
                detection_id = detection_of_key(key)
                if detection_id is not None:
                    result[detection_id] = archive[key]
            return result
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        # A truncated archive — an interrupted write, a full disk — is
        # indistinguishable from an absent one as far as the caller is
        # concerned: both mean "these vectors need computing again".
        # BadZipFile derives from Exception rather than OSError, so it has to
        # be named: an .npz is a zip, and a half-written one lands here.
        return {}


def read_vectors(target_folder: str, modality: str, projection: str,
                 wanted: Iterable[dict]) -> Dict[int, "object"]:
    """Vectors for the given detections, opening each frame archive once.

    *wanted* rows carry ``detection_id``, ``frame`` and optionally
    ``imagefile``. Grouping by frame is what keeps this to one ``np.load`` per
    frame rather than one per detection.
    """
    by_file: Dict[str, List[dict]] = {}
    for row in wanted:
        name = frame_file_name(row["frame"], row.get("imagefile", ""))
        by_file.setdefault(name, []).append(row)

    folder = run_folder(target_folder, modality, projection)
    vectors: Dict[int, "object"] = {}
    for name, rows in by_file.items():
        archive = read_frame(os.path.join(folder, name))
        if not archive:
            continue
        for row in rows:
            vector = archive.get(int(row["detection_id"]))
            if vector is not None:
                vectors[int(row["detection_id"])] = vector
    return vectors


def present_ids(target_folder: str, modality: str, projection: str,
                wanted: Iterable[dict]) -> List[int]:
    """Which of *wanted* actually have a vector on disk.

    Reconciles the store's membership against the files, so a ``.npz`` deleted
    by hand is noticed rather than believed.
    """
    return sorted(read_vectors(target_folder, modality, projection, wanted))
