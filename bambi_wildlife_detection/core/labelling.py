# -*- coding: utf-8 -*-
"""Headless data model and geometry logic of the labelling tool.

Moved verbatim from ``bambi_labelling_tool.py`` (which re-exports every
name here for backwards compatibility); see that module's docstring for
the labelling workflow. Contains:

* the class taxonomies and track colour palette (as RGB tuples — the GUI
  wraps them in ``QColor``),
* :class:`LabelTrack` / :class:`LabelStore` — key-frame storage,
  interpolation, stop frames, JSON/CSV persistence and the
  ``detections.txt`` export,
* read-only loaders for pipeline outputs (detections, pixel tracks),
* :class:`_FrameMatcher` — cross-modality frame matching by capture time,
* :class:`_GeoPropagator` — DEM-based box propagation between frames and
  modalities (lazy alfspy/trimesh/bambi imports).
"""

import os
import json
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Class taxonomies
# ---------------------------------------------------------------------------

SPECIES_CLASSES = [
    "unknown", "roe deer", "red deer", "fallow deer", "wild boar",
    "chamois", "fox", "hare", "bird", "other",
]
SEX_CLASSES = ["unknown", "female", "male"]
AGE_CLASSES = ["unknown", "adult", "juvenile"]
OCCLUSION_LEVELS = ["none", "partially", "fully"]

# Distinct colours cycled per label track id (RGB tuples; the GUI layer
# converts them to QColor).
TRACK_COLORS_RGB = [
    (0, 220, 0), (255, 160, 0), (255, 80, 200),
    (0, 200, 255), (255, 230, 0), (170, 110, 255),
    (255, 90, 90), (120, 255, 170),
]


def track_color_rgb(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS_RGB[track_id % len(TRACK_COLORS_RGB)]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class LabelTrack:
    """A single annotated track made of key frames.

    Key frames map ``frame -> {"x1","y1","x2","y2","occlusion"[,"stop"]}``;
    boxes on frames between two key frames are linearly interpolated —
    except after a key frame flagged ``stop`` (the animal disappeared):
    frames between a stop frame and the next key frame have no box.
    """

    def __init__(self, track_id: int, species: str = "unknown",
                 sex: str = "unknown", age: str = "unknown"):
        self.track_id = track_id
        self.species = species
        self.sex = sex
        self.age = age
        self.keyframes: Dict[int, dict] = {}

    # -- key frame access ------------------------------------------------

    def frames(self) -> List[int]:
        return sorted(self.keyframes.keys())

    def frame_range(self) -> Optional[Tuple[int, int]]:
        if not self.keyframes:
            return None
        fs = self.frames()
        return fs[0], fs[-1]

    def set_keyframe(self, frame: int, box: Tuple[float, float, float, float],
                     occlusion: Optional[str] = None,
                     stop: Optional[bool] = None) -> None:
        prev = self.keyframes.get(frame)
        if occlusion is None:
            if prev is not None:
                occlusion = prev.get("occlusion", "none")
            else:
                # inherit from the interpolation state at this frame
                interp = self.box_at(frame)
                occlusion = interp[2] if interp else "none"
        if stop is None:
            # preserve an existing stop flag when only the box is updated
            stop = bool(prev.get("stop", False)) if prev else False
        entry = {
            "x1": float(box[0]), "y1": float(box[1]),
            "x2": float(box[2]), "y2": float(box[3]),
            "occlusion": occlusion,
        }
        if stop:
            entry["stop"] = True
        self.keyframes[frame] = entry

    def is_stop(self, frame: int) -> bool:
        return bool(self.keyframes.get(frame, {}).get("stop", False))

    def visible_segments(self) -> List[Tuple[int, int]]:
        """Return the frame ranges in which the track has boxes.

        Without stop frames this is one ``[(first, last)]`` segment; every
        stop frame ends its segment and the next key frame starts a new one.
        """
        fs = self.frames()
        if not fs:
            return []
        segments: List[Tuple[int, int]] = []
        start = fs[0]
        for i, f in enumerate(fs[:-1]):
            if self.keyframes[f].get("stop"):
                segments.append((start, f))
                start = fs[i + 1]
        segments.append((start, fs[-1]))
        return segments

    def remove_keyframe(self, frame: int) -> bool:
        return self.keyframes.pop(frame, None) is not None

    def box_at(self, frame: int) -> Optional[Tuple[tuple, bool, str]]:
        """Return ``((x1,y1,x2,y2), is_keyframe, occlusion)`` or ``None``.

        Frames outside the [first, last] key-frame range return ``None``.
        """
        if not self.keyframes:
            return None
        if frame in self.keyframes:
            kf = self.keyframes[frame]
            return ((kf["x1"], kf["y1"], kf["x2"], kf["y2"]), True,
                    kf.get("occlusion", "none"))

        fs = self.frames()
        if frame < fs[0] or frame > fs[-1]:
            return None

        f_prev = max(f for f in fs if f < frame)
        f_next = min(f for f in fs if f > frame)
        a = self.keyframes[f_prev]
        if a.get("stop"):
            return None  # gap: the animal disappeared after f_prev
        b = self.keyframes[f_next]
        alpha = (frame - f_prev) / (f_next - f_prev)
        box = tuple(
            a[k] + alpha * (b[k] - a[k]) for k in ("x1", "y1", "x2", "y2")
        )
        return box, False, a.get("occlusion", "none")

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "species": self.species,
            "sex": self.sex,
            "age": self.age,
            "keyframes": {str(f): kf for f, kf in self.keyframes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelTrack":
        track = cls(
            int(d.get("track_id", 0)),
            d.get("species", "unknown"),
            d.get("sex", "unknown"),
            d.get("age", "unknown"),
        )
        for f, kf in d.get("keyframes", {}).items():
            entry = {
                "x1": float(kf["x1"]), "y1": float(kf["y1"]),
                "x2": float(kf["x2"]), "y2": float(kf["y2"]),
                "occlusion": kf.get("occlusion", "none"),
            }
            if kf.get("stop"):
                entry["stop"] = True
            track.keyframes[int(f)] = entry
        return track


class LabelStore:
    """Loads / saves the label tracks of one modality."""

    def __init__(self, target_folder: str, modality: str):
        self.target_folder = target_folder
        self.modality = modality
        self.tracks: Dict[int, LabelTrack] = {}

    @property
    def folder(self) -> str:
        return os.path.join(self.target_folder, f"labels_{self.modality}")

    @property
    def json_path(self) -> str:
        return os.path.join(self.folder, "labels.json")

    @property
    def csv_path(self) -> str:
        return os.path.join(self.folder, "labels.csv")

    def next_track_id(self) -> int:
        return max(self.tracks.keys(), default=0) + 1

    def load(self) -> None:
        self.tracks = {}
        if not os.path.isfile(self.json_path):
            return
        with open(self.json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for td in data.get("tracks", []):
            track = LabelTrack.from_dict(td)
            self.tracks[track.track_id] = track

    def save(self) -> None:
        os.makedirs(self.folder, exist_ok=True)
        data = {
            "modality": self.modality,
            "tracks": [t.to_dict() for t in sorted(
                self.tracks.values(), key=lambda t: t.track_id)],
        }
        with open(self.json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self._export_csv()

    # Marker line that delimits the label-tool block in detections.txt so a
    # re-export replaces the previous block instead of duplicating it.
    DETECTIONS_MARKER = "# --- labelled detections (BAMBI labelling tool) ---"

    def species_class_ids(self) -> Dict[str, int]:
        """Return a ``species -> class_id`` mapping for the pipeline export.

        The pipeline only knows integer class ids, so species from the
        default taxonomy keep their list index and custom species are
        appended after it in alphabetical order.
        """
        mapping = {s: i for i, s in enumerate(SPECIES_CLASSES)}
        custom = sorted({
            (t.species or "unknown").strip().lower()
            for t in self.tracks.values()
        } - set(SPECIES_CLASSES))
        for i, s in enumerate(custom):
            mapping[s] = len(SPECIES_CLASSES) + i
        return mapping

    def export_to_detections(self) -> Tuple[str, int]:
        """Merge the interpolated label boxes into ``detections.txt``.

        Uses the exact line format of the "Detect Animals" stage
        (``frame x1 y1 x2 y2 confidence class_id``, confidence 1.0 for
        manual labels) so downstream steps (geo-referencing, tracking)
        consume the labels like regular detections.  Detector output is
        preserved; a previous label export block is replaced.

        Returns ``(detections_path, number_of_exported_boxes)``.
        """
        det_folder = os.path.join(
            self.target_folder, f"detections_{self.modality}")
        os.makedirs(det_folder, exist_ok=True)
        det_file = os.path.join(det_folder, "detections.txt")

        # Keep everything up to (excluding) a previous export block.
        existing: List[str] = []
        if os.path.isfile(det_file):
            with open(det_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.rstrip("\n") == self.DETECTIONS_MARKER:
                        break
                    existing.append(line)
        if not existing:
            existing = ["# frame x1 y1 x2 y2 confidence class_id\n"]
        if existing and not existing[-1].endswith("\n"):
            existing[-1] += "\n"

        mapping = self.species_class_ids()
        rows: List[tuple] = []  # (frame, track_id, x1, y1, x2, y2, class_id)
        for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
            rng = track.frame_range()
            if rng is None:
                continue
            species = (track.species or "unknown").strip().lower()
            class_id = mapping.get(species, 0)
            for frame in range(rng[0], rng[1] + 1):
                res = track.box_at(frame)
                if res is None:
                    continue
                (x1, y1, x2, y2), _is_kf, _occ = res
                rows.append((frame, track.track_id, x1, y1, x2, y2, class_id))
        rows.sort(key=lambda r: (r[0], r[1]))

        used_ids = sorted({r[6] for r in rows})
        id_to_species = {i: s for s, i in mapping.items()}

        with open(det_file, "w", encoding="utf-8") as fh:
            fh.writelines(existing)
            fh.write(self.DETECTIONS_MARKER + "\n")
            if used_ids:
                pairs = ", ".join(f"{i}={id_to_species[i]}" for i in used_ids)
                fh.write(f"# class_id mapping: {pairs}\n")
            for frame, _tid, x1, y1, x2, y2, class_id in rows:
                fh.write(f"{frame} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                         f"1.0000 {class_id}\n")
        return det_file, len(rows)

    def _export_csv(self) -> None:
        """Write the per-frame interpolated CSV export."""
        with open(self.csv_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# frame,track_id,x1,y1,x2,y2,species,sex,age,occlusion,keyframe\n"
            )
            for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
                rng = track.frame_range()
                if rng is None:
                    continue
                for frame in range(rng[0], rng[1] + 1):
                    res = track.box_at(frame)
                    if res is None:
                        continue
                    (x1, y1, x2, y2), is_kf, occ = res
                    fh.write(
                        f"{frame},{track.track_id},"
                        f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},"
                        f"{track.species},{track.sex},{track.age},"
                        f"{occ},{1 if is_kf else 0}\n"
                    )


# ---------------------------------------------------------------------------
# Overlay data loaders (pipeline outputs, read-only)
# ---------------------------------------------------------------------------

def _load_detections_by_frame(det_file: str) -> Dict[int, List[tuple]]:
    """Parse ``detections_{m}/detections.txt`` into ``frame -> [boxes]``.

    Format (space-separated, ``#`` comments): ``frame x1 y1 x2 y2 conf cls``
    """
    result: Dict[int, List[tuple]] = {}
    if not os.path.isfile(det_file):
        return result
    try:
        with open(det_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    frame = int(parts[0])
                    result.setdefault(frame, []).append((
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]),
                        float(parts[5]),
                        int(parts[6]) if len(parts) > 6 else 0,
                    ))
    except Exception:  # nosec B110
        pass
    return result


def _load_pixel_tracks(tracks_file: str) -> Dict[int, List[dict]]:
    """Parse ``tracks_{m}/tracks_pixel.csv`` into ``track_id -> [entries]``.

    Format (comma-separated, ``#`` comments):
    ``frame,track_id,x1,y1,x2,y2,conf,cls[,interpolated]``
    """
    result: Dict[int, List[dict]] = {}
    if not os.path.isfile(tracks_file):
        return result
    try:
        with open(tracks_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 8:
                    tid = int(parts[1])
                    result.setdefault(tid, []).append({
                        "frame": int(parts[0]),
                        "x1": float(parts[2]), "y1": float(parts[3]),
                        "x2": float(parts[4]), "y2": float(parts[5]),
                        "conf": float(parts[6]), "cls": int(parts[7]),
                    })
    except Exception:  # nosec B110
        pass
    for entries in result.values():
        entries.sort(key=lambda d: d["frame"])
    return result


# ---------------------------------------------------------------------------
# Cross-modality frame correspondence (by capture timestamp)
# ---------------------------------------------------------------------------

def _pose_epochs(images: List[dict]) -> List[Optional[float]]:
    """Return the capture time (epoch seconds) of every pose image.

    Poses store an ISO-8601 ``timestamp`` (with timezone) taken from the SRT
    capture time, so thermal and RGB frames share the same real-world clock
    and can be matched across modalities.  Entries without a parseable
    timestamp yield ``None``.
    """
    from datetime import datetime

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


class _FrameMatcher:
    """Maps a source-modality frame index to the nearest target-modality one.

    Correspondence is by capture time: for a source frame's timestamp the
    target frame with the closest timestamp is returned, together with the
    absolute time difference so callers can reject poor matches.
    """

    def __init__(self, src_images: List[dict], dst_images: List[dict]):
        import bisect

        self._src_epochs = _pose_epochs(src_images)
        dst_epochs = _pose_epochs(dst_images)
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


# ---------------------------------------------------------------------------
# Geo-referenced box propagation
# ---------------------------------------------------------------------------

#: Pixel-space bounding box ``(x1, y1, x2, y2)``.
Box = Tuple[float, float, float, float]


def propagation_frames(src_frame: int, dst_frame: int,
                       step: int = 0) -> List[int]:
    """Frames that get a key frame when propagating *src_frame* -> *dst_frame*.

    With ``step <= 0`` only the target frame is returned.  With a positive
    *step* additional intermediate frames are sampled every *step* frames
    between source and target, so the linear interpolation between the key
    frames never has to bridge more than *step* frames of drone ego-motion.
    The target frame is always the last entry; the source frame is never
    included.
    """
    if dst_frame == src_frame:
        return []
    sign = 1 if dst_frame > src_frame else -1
    frames: List[int] = []
    if step > 0:
        frame = src_frame + sign * step
        while (dst_frame - frame) * sign > 0:
            frames.append(frame)
            frame += sign * step
    frames.append(dst_frame)
    return frames


class _GeoPropagator:
    """Projects a pixel-space box from one frame to another via the DEM.

    Forward step: the four box corners are ray-cast onto the DEM mesh with
    the source frame's camera (``label_to_world_coordinates``).  Reverse
    step: the resulting local world points are projected with the target
    frame's camera (``_world_to_pixel`` from :mod:`bambi_box_projector`).
    Both cameras are built exactly like in ``BambiProcessor.run_georeference``
    (1× rotation correction), so no DEM origin offset is needed — everything
    stays in the mesh-local coordinate space of the poses file.
    """

    def __init__(self, target_folder: str, dem_path: str,
                 correction_path: str, modality: str):
        self.target_folder = target_folder
        self.dem_path = dem_path
        self.correction_path = correction_path
        self.modality = modality
        self._tri_mesh = None
        self._correction = None

    @property
    def is_loaded(self) -> bool:
        return self._tri_mesh is not None

    def load(self) -> None:
        """Load the DEM mesh (may take a while for large meshes)."""
        from alfspy.render.render import read_gltf
        from trimesh import Trimesh
        from .corrections import read_correction

        if not self.dem_path or not os.path.isfile(self.dem_path):
            raise RuntimeError(
                "DEM mesh file not found. Please set the DEM path (GLB/GLTF) "
                "in the plugin's Input tab or via the field in this dialog."
            )
        mesh_data, _ = read_gltf(self.dem_path)
        tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
        try:
            _ = tri_mesh.triangles_tree  # pre-build BVH
        except Exception:  # nosec B110
            pass
        self._tri_mesh = tri_mesh
        self._correction = read_correction(self.target_folder, self.correction_path)

    def _build_camera(self, images: List[dict], frame_idx: int,
                      img_width: int, img_height: int):
        """Camera construction identical to ``run_georeference``
        (single source: :func:`core.camera_pose.frame_camera`)."""
        from .camera_pose import frame_camera

        return frame_camera(
            images, frame_idx, self._correction,
            aspect_ratio=img_width / img_height)

    def propagate(self, box: Tuple[float, float, float, float],
                  src_frame: int, dst_frame: int, images: List[dict],
                  img_width: int, img_height: int,
                  ) -> Tuple[float, float, float, float]:
        """Return *box* re-projected from *src_frame* to *dst_frame*.

        Both frames belong to the same modality (same ``images`` list and
        resolution).  Raises ``RuntimeError`` with a human-readable message
        on failure.
        """
        if not (0 <= src_frame < len(images)) or not (0 <= dst_frame < len(images)):
            raise RuntimeError("Frame index outside the poses file range.")
        return self.propagate_between(
            box, src_frame, images, img_width, img_height,
            dst_frame, images, img_width, img_height)

    def propagate_series(self, box: Tuple[float, float, float, float],
                         src_frame: int, dst_frame: int, images: List[dict],
                         img_width: int, img_height: int, step: int = 0,
                         ) -> Tuple[List[Tuple[int, Box]],
                                    List[Tuple[int, str]]]:
        """Propagate *box* to the target frame and to intermediate samples.

        The source box is ray-cast onto the DEM once and back-projected into
        every frame from :func:`propagation_frames`, so projection errors do
        not accumulate along the series.  Returns ``(boxes, failures)`` where
        *boxes* is a list of ``(frame, box)`` in propagation order and
        *failures* a list of ``(frame, message)`` for frames the box could not
        be projected into (outside the frame, frame outside the poses range).

        Raises ``RuntimeError`` if the source box itself cannot be ray-cast
        onto the DEM — that failure applies to the whole series.
        """
        frames = propagation_frames(src_frame, dst_frame, step)
        world = self._box_to_world(
            box, src_frame, images, img_width, img_height)

        boxes: List[Tuple[int, Box]] = []
        failures: List[Tuple[int, str]] = []
        for frame in frames:
            try:
                boxes.append((frame, self._world_to_box(
                    world, frame, images, img_width, img_height)))
            except RuntimeError as exc:
                failures.append((frame, str(exc)))
        return boxes, failures

    def propagate_between(self, box: Tuple[float, float, float, float],
                          src_frame: int, src_images: List[dict],
                          src_width: int, src_height: int,
                          dst_frame: int, dst_images: List[dict],
                          dst_width: int, dst_height: int,
                          ) -> Tuple[float, float, float, float]:
        """Re-project *box* from a source frame/camera onto a target one.

        The source and target may come from different modalities (different
        ``images`` lists and resolutions): the box corners are ray-cast onto
        the shared DEM with the source camera and back-projected with the
        target camera.  Both cameras apply the same drone-pose correction
        (``correction.json`` is per-flight, not per-camera), each keyed by its
        own frame index.

        Raises ``RuntimeError`` with a human-readable message on failure.
        """
        world = self._box_to_world(
            box, src_frame, src_images, src_width, src_height)
        return self._world_to_box(
            world, dst_frame, dst_images, dst_width, dst_height)

    def _box_to_world(self, box: Tuple[float, float, float, float],
                      src_frame: int, src_images: List[dict],
                      src_width: int, src_height: int):
        """Ray-cast the four box corners onto the DEM with the source camera."""
        try:
            from alfspy.core.rendering import Resolution
            from bambi.util.projection_util import label_to_world_coordinates
        except ImportError as exc:
            raise RuntimeError(
                "alfspy / bambi packages are not available — cannot "
                f"geo-reference bounding boxes.\n({exc})"
            )

        if not self.is_loaded:
            self.load()
        if not (0 <= src_frame < len(src_images)):
            raise RuntimeError("Source frame index outside the poses range.")

        x1, y1, x2, y2 = box
        label_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
        cam_src = self._build_camera(src_images, src_frame, src_width, src_height)
        world = label_to_world_coordinates(
            label_coords, Resolution(src_width, src_height),
            self._tri_mesh, cam_src)
        if world is None or len(world) == 0:
            raise RuntimeError(
                "The bounding box could not be ray-cast onto the DEM "
                "(no mesh intersection)."
            )
        return world

    def _world_to_box(self, world, dst_frame: int, dst_images: List[dict],
                      dst_width: int, dst_height: int,
                      ) -> Tuple[float, float, float, float]:
        """Project local world points into the target frame's image plane."""
        import numpy as np
        from .camera_pose import world_to_pixel

        if not (0 <= dst_frame < len(dst_images)):
            raise RuntimeError("Target frame index outside the poses range.")

        cam_dst = self._build_camera(dst_images, dst_frame, dst_width, dst_height)
        pxs, pys = world_to_pixel(
            np.asarray(world, dtype=np.float64), dst_width, dst_height, cam_dst)

        nx1, ny1 = float(np.min(pxs)), float(np.min(pys))
        nx2, ny2 = float(np.max(pxs)), float(np.max(pys))
        if nx2 < 0 or ny2 < 0 or nx1 > dst_width or ny1 > dst_height:
            raise RuntimeError(
                f"The projected box lies completely outside frame {dst_frame}."
            )
        return (
            max(0.0, nx1), max(0.0, ny1),
            min(float(dst_width - 1), nx2), min(float(dst_height - 1), ny2),
        )
