# -*- coding: utf-8 -*-
"""
BAMBI Labelling Tool
====================

Non-modal dialog for reviewing and creating track annotations on the
extracted frames of a processed flight.

Features
--------
* Visualises existing pipeline detections (``detections_{m}/detections.txt``)
  and tracks (``tracks_{m}/tracks_pixel.csv``) as read-only overlays.
* Modality selector: label either on thermal or RGB frames — never mixed.
  Label tracks are stored per modality in ``labels_{m}/labels.json``.
* Key-frame based annotation: bounding boxes are only stored on key frames
  (e.g. every 10th frame); all frames in between are linearly interpolated.
* New tracks are drawn as bounding boxes and carry a species class, sex
  class, age class (per track) and an occlusion level (per key frame).
* Existing label boxes can be moved/resized (which writes a key frame at the
  current frame) and their classes edited at any time.
* Existing pipeline tracks can be imported as editable label tracks.
* Geo-referenced propagation: the current bounding box is ray-cast onto the
  DEM (pixel → world) with the camera pose of the current frame and
  back-projected (world → pixel) with the pose of an offset frame — the
  result usually only needs small size adaptions.

Files written (relative to *target_folder*, ``{m}`` = ``t`` / ``w``)
--------------------------------------------------------------------
``labels_{m}/labels.json``  — key-frame source of truth
``labels_{m}/labels.csv``   — per-frame interpolated export
    format: ``frame,track_id,x1,y1,x2,y2,species,sex,age,occlusion,keyframe``

"Add detections to project" additionally merges the interpolated label boxes
into ``detections_{m}/detections.txt`` in the exact format of the "Detect
Animals" stage (``frame x1 y1 x2 y2 confidence class_id``) so the rest of the
pipeline (geo-referencing, tracking, …) can consume them.  The exported block
is delimited by a marker comment and replaced on re-export.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import Qt, QRectF, QPointF, QSettings, QTimer, pyqtSignal
from qgis.PyQt.QtGui import (
    QColor, QPen, QFont, QPixmap, QPainter, QCursor, QPainterPath,
)
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QSpinBox, QCheckBox, QSlider, QWidget, QGroupBox,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QApplication,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,
    QGraphicsSimpleTextItem, QSizePolicy,
)


# ---------------------------------------------------------------------------
# Class taxonomies
# ---------------------------------------------------------------------------

SPECIES_CLASSES = [
    "unknown", "roe deer", "red deer", "fallow deer", "wild boar",
    "chamois", "fox", "hare", "bird", "other",
]
SEX_CLASSES = ["unknown", "female", "male"]
AGE_CLASSES = ["unknown", "adult", "juvenile"]
OCCLUSION_LEVELS = ["none", "partial", "heavy", "full"]

# Distinct colours cycled per label track id
_TRACK_COLORS = [
    QColor(0, 220, 0), QColor(255, 160, 0), QColor(255, 80, 200),
    QColor(0, 200, 255), QColor(255, 230, 0), QColor(170, 110, 255),
    QColor(255, 90, 90), QColor(120, 255, 170),
]

_DETECTION_COLOR = QColor(80, 140, 255)   # read-only detection overlay
_TRACK_OVERLAY_COLOR = QColor(0, 190, 190)  # read-only track overlay


def _track_color(track_id: int) -> QColor:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class LabelTrack:
    """A single annotated track made of key frames.

    Key frames map ``frame -> {"x1","y1","x2","y2","occlusion"}``; boxes on
    frames between two key frames are linearly interpolated.
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
                     occlusion: Optional[str] = None) -> None:
        prev = self.keyframes.get(frame)
        if occlusion is None:
            if prev is not None:
                occlusion = prev.get("occlusion", "none")
            else:
                # inherit from the interpolation state at this frame
                interp = self.box_at(frame)
                occlusion = interp[2] if interp else "none"
        self.keyframes[frame] = {
            "x1": float(box[0]), "y1": float(box[1]),
            "x2": float(box[2]), "y2": float(box[3]),
            "occlusion": occlusion,
        }

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
            track.keyframes[int(f)] = {
                "x1": float(kf["x1"]), "y1": float(kf["y1"]),
                "x2": float(kf["x2"]), "y2": float(kf["y2"]),
                "occlusion": kf.get("occlusion", "none"),
            }
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
# Geo-referenced box propagation
# ---------------------------------------------------------------------------

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
        from .bambi_box_projector import _read_correction

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
        self._correction = _read_correction(self.target_folder, self.correction_path)

    def _build_camera(self, images: List[dict], frame_idx: int,
                      img_width: int, img_height: int):
        """Camera construction identical to ``run_georeference``."""
        import numpy as np
        from pyrr import Vector3, Quaternion
        from alfspy.core.rendering import Camera
        from .bambi_box_projector import _correction_for_frame

        meta = images[frame_idx]
        fovy = meta.get("fovy", [50])
        if isinstance(fovy, list):
            fovy = fovy[0]

        t_corr, r_corr = _correction_for_frame(frame_idx, self._correction or {})
        cor_t = Vector3(
            [t_corr.get("x", 0), t_corr.get("y", 0), t_corr.get("z", 0)],
            dtype="f4")
        cor_r = Vector3(
            [r_corr.get("x", 0), r_corr.get("y", 0), r_corr.get("z", 0)],
            dtype="f4")

        position = Vector3(meta["location"]) + cor_t
        rotation_eulers = (
            Vector3([np.deg2rad(v % 360.0) for v in meta["rotation"]]) - cor_r
        ) * -1
        rotation_quat = Quaternion.from_eulers(rotation_eulers)

        return Camera(
            fovy=fovy,
            aspect_ratio=img_width / img_height,
            position=position,
            rotation=rotation_quat,
        )

    def propagate(self, box: Tuple[float, float, float, float],
                  src_frame: int, dst_frame: int, images: List[dict],
                  img_width: int, img_height: int,
                  ) -> Tuple[float, float, float, float]:
        """Return *box* re-projected from *src_frame* to *dst_frame*.

        Raises ``RuntimeError`` with a human-readable message on failure.
        """
        import numpy as np
        try:
            from alfspy.core.rendering import Resolution
            from bambi.util.projection_util import label_to_world_coordinates
        except ImportError as exc:
            raise RuntimeError(
                "alfspy / bambi packages are not available — cannot "
                f"geo-reference bounding boxes.\n({exc})"
            )
        from .bambi_box_projector import _world_to_pixel

        if not self.is_loaded:
            self.load()

        if not (0 <= src_frame < len(images)) or not (0 <= dst_frame < len(images)):
            raise RuntimeError("Frame index outside the poses file range.")

        resolution = Resolution(img_width, img_height)
        x1, y1, x2, y2 = box
        label_coords = [x1, y1, x2, y1, x2, y2, x1, y2]

        # ---- forward: pixel -> local world (DEM ray-cast) -----------------
        cam_src = self._build_camera(images, src_frame, img_width, img_height)
        world = label_to_world_coordinates(
            label_coords, resolution, self._tri_mesh, cam_src)
        if world is None or len(world) == 0:
            raise RuntimeError(
                "The bounding box could not be ray-cast onto the DEM "
                "(no mesh intersection)."
            )

        # ---- reverse: local world -> pixel of the target frame -----------
        cam_dst = self._build_camera(images, dst_frame, img_width, img_height)
        pxs, pys = _world_to_pixel(
            np.asarray(world, dtype=np.float64), img_width, img_height, cam_dst)

        nx1, ny1 = float(np.min(pxs)), float(np.min(pys))
        nx2, ny2 = float(np.max(pxs)), float(np.max(pys))
        if nx2 < 0 or ny2 < 0 or nx1 > img_width or ny1 > img_height:
            raise RuntimeError(
                f"The projected box lies completely outside frame {dst_frame}."
            )
        return (
            max(0.0, nx1), max(0.0, ny1),
            min(float(img_width - 1), nx2), min(float(img_height - 1), ny2),
        )


# ---------------------------------------------------------------------------
# Canvas items
# ---------------------------------------------------------------------------

class _BoxItem(QGraphicsRectItem):
    """A bounding-box rectangle in image pixel coordinates.

    Editable items support moving (drag inside) and resizing (drag edges /
    corners); the final geometry is committed through *on_committed*.
    Non-editable items only report clicks through *on_clicked*.
    """

    def __init__(self, track_id: int, rect: QRectF, color: QColor,
                 label: str, font_px: int, editable: bool = False,
                 dashed: bool = False, emphasized: bool = False,
                 on_committed=None, on_clicked=None):
        super().__init__(rect)
        self._track_id = track_id
        self._editable = editable
        self._on_committed = on_committed
        self._on_clicked = on_clicked
        self._drag = None  # (zone, press_pos, press_rect)

        pen = QPen(color, 3 if emphasized else 2,
                   Qt.DashLine if dashed else Qt.SolidLine)
        pen.setCosmetic(True)
        self.setPen(pen)

        if label:
            text = QGraphicsSimpleTextItem(label, self)
            text.setBrush(color)
            font = QFont("Arial")
            font.setPixelSize(font_px)
            text.setFont(font)
            text.setPos(rect.left() + 2,
                        max(0.0, rect.top() - font_px - 4))

        self.setAcceptHoverEvents(editable)
        if editable or on_clicked is not None:
            self.setAcceptedMouseButtons(Qt.LeftButton)
        else:
            self.setAcceptedMouseButtons(Qt.NoButton)

    # -- geometry helpers --------------------------------------------------

    def _margin(self) -> float:
        """Hit margin for edge handles, in scene units (zoom independent)."""
        scale = 1.0
        if self.scene() and self.scene().views():
            scale = max(1e-6, self.scene().views()[0].transform().m11())
        return 8.0 / scale

    def _hit_zone(self, pos: QPointF) -> Optional[str]:
        r = self.rect()
        m = self._margin()
        left = abs(pos.x() - r.left()) <= m
        right = abs(pos.x() - r.right()) <= m
        top = abs(pos.y() - r.top()) <= m
        bottom = abs(pos.y() - r.bottom()) <= m
        in_x = r.left() - m <= pos.x() <= r.right() + m
        in_y = r.top() - m <= pos.y() <= r.bottom() + m
        if not (in_x and in_y):
            return None
        zone = ""
        if top:
            zone += "t"
        elif bottom:
            zone += "b"
        if left:
            zone += "l"
        elif right:
            zone += "r"
        return zone or "move"

    def boundingRect(self) -> QRectF:
        m = self._margin()
        return self.rect().adjusted(-m, -m, m, m)

    def shape(self):
        m = self._margin()
        path = QPainterPath()
        path.addRect(self.rect().adjusted(-m, -m, m, m))
        if not self._editable:
            # non-editable boxes are only clickable on their outline so they
            # don't swallow clicks meant for boxes drawn below them
            inner = QPainterPath()
            inner.addRect(self.rect().adjusted(m, m, -m, -m))
            path = path.subtracted(inner)
        return path

    # -- interaction --------------------------------------------------------

    _CURSORS = {
        "move": Qt.SizeAllCursor,
        "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
        "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
        "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
        "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
    }

    def hoverMoveEvent(self, event):
        zone = self._hit_zone(event.pos())
        self.setCursor(QCursor(self._CURSORS.get(zone, Qt.ArrowCursor)))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if self._on_clicked is not None:
            self._on_clicked(self._track_id)
        if self._editable:
            zone = self._hit_zone(event.pos())
            if zone:
                self._drag = (zone, event.pos(), QRectF(self.rect()))
                event.accept()
                return
        event.ignore()

    def mouseMoveEvent(self, event):
        if not self._drag:
            return
        zone, press_pos, press_rect = self._drag
        d = event.pos() - press_pos
        r = QRectF(press_rect)
        if zone == "move":
            r.translate(d)
        else:
            if "t" in zone:
                r.setTop(press_rect.top() + d.y())
            if "b" in zone:
                r.setBottom(press_rect.bottom() + d.y())
            if "l" in zone:
                r.setLeft(press_rect.left() + d.x())
            if "r" in zone:
                r.setRight(press_rect.right() + d.x())
            r = r.normalized()
        if r.width() >= 2 and r.height() >= 2:
            self.prepareGeometryChange()
            self.setRect(r)

    def mouseReleaseEvent(self, event):
        if self._drag:
            self._drag = None
            if self._on_committed is not None:
                self._on_committed(self._track_id, QRectF(self.rect()))
        event.accept()


class _LabelCanvas(QGraphicsView):
    """Zoomable / pannable graphics view holding the frame + box items."""

    boxDrawn = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setZValue(-1)
        self.scene().addItem(self._pix_item)

        self._box_items: List[_BoxItem] = []
        self._draw_mode = False
        self._rubber_item = None
        self._rubber_start = None
        self._pan_last = None
        self._fitted = False

    # -- content -------------------------------------------------------------

    def set_frame(self, pixmap: QPixmap) -> None:
        first = self._pix_item.pixmap().isNull() or (
            self._pix_item.pixmap().size() != pixmap.size())
        self._pix_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        if first or not self._fitted:
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)
            self._fitted = True

    def clear_boxes(self) -> None:
        for item in self._box_items:
            self.scene().removeItem(item)
        self._box_items = []

    def add_box(self, item: _BoxItem) -> None:
        self.scene().addItem(item)
        self._box_items.append(item)

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self.viewport().setCursor(
            Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def fit(self) -> None:
        if not self._pix_item.pixmap().isNull():
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)

    # -- interaction -----------------------------------------------------------

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_last = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self._draw_mode and event.button() == Qt.LeftButton:
            self._rubber_start = self.mapToScene(event.pos())
            self._rubber_item = QGraphicsRectItem(
                QRectF(self._rubber_start, self._rubber_start))
            pen = QPen(QColor(255, 255, 255), 1, Qt.DashLine)
            pen.setCosmetic(True)
            self._rubber_item.setPen(pen)
            self.scene().addItem(self._rubber_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._rubber_item is not None:
            rect = QRectF(self._rubber_start,
                          self.mapToScene(event.pos())).normalized()
            self._rubber_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self.viewport().setCursor(
                Qt.CrossCursor if self._draw_mode else Qt.ArrowCursor)
            event.accept()
            return
        if self._rubber_item is not None and event.button() == Qt.LeftButton:
            rect = self._rubber_item.rect().normalized()
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
            self._rubber_start = None
            if rect.width() >= 3 and rect.height() >= 3:
                self.boxDrawn.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _TimelineWidget(QWidget):
    """Slim timeline bar showing the selected track's range and key frames."""

    frameClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self._total = 1
        self._current = 0
        self._keyframes: List[int] = []
        self._range: Optional[Tuple[int, int]] = None

    def set_data(self, total: int, current: int,
                 keyframes: List[int],
                 track_range: Optional[Tuple[int, int]]) -> None:
        self._total = max(1, total)
        self._current = current
        self._keyframes = keyframes
        self._range = track_range
        self.update()

    def _frame_to_x(self, frame: int) -> float:
        return frame / max(1, self._total - 1) * (self.width() - 1)

    def mousePressEvent(self, event):
        frac = event.pos().x() / max(1, self.width() - 1)
        self.frameClicked.emit(
            int(round(frac * (self._total - 1))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(45, 45, 45))

        # Selected track range
        if self._range is not None:
            x0 = self._frame_to_x(self._range[0])
            x1 = self._frame_to_x(self._range[1])
            painter.fillRect(
                int(x0), 4, max(2, int(x1 - x0)), self.height() - 8,
                QColor(70, 110, 70))

        # Key frames
        painter.setPen(QPen(QColor(0, 220, 0), 2))
        for kf in self._keyframes:
            x = int(self._frame_to_x(kf))
            painter.drawLine(x, 2, x, self.height() - 2)

        # Current frame cursor
        painter.setPen(QPen(QColor(255, 220, 0), 2))
        x = int(self._frame_to_x(self._current))
        painter.drawLine(x, 0, x, self.height())
        painter.end()


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class LabellingToolDialog(QDialog):
    """Non-modal key-frame labelling tool for extracted frames."""

    def __init__(self, iface, dock_widget=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._dock_widget = dock_widget

        self.setWindowTitle("BAMBI Labelling Tool")
        flags = Qt.Window | Qt.WindowCloseButtonHint
        flags |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
        self.setWindowFlags(flags)
        self.resize(1200, 800)

        # Project state
        self._target_folder = ""
        self._dem_path = ""
        self._correction_path = ""
        self._modality = "t"
        self._images: List[dict] = []          # poses images of the modality
        self._detections: Dict[int, List[tuple]] = {}
        self._pixel_tracks: Dict[int, List[dict]] = {}
        self._store: Optional[LabelStore] = None
        self._propagator: Optional[_GeoPropagator] = None
        self._img_size: Optional[Tuple[int, int]] = None

        # UI state
        self._current_frame = 0
        self._selected_track: Optional[int] = None
        self._dirty = False
        self._updating_ui = False

        # Autosave debounce: rapid edits (dragging, typing a species name)
        # collapse into one save shortly after the last change.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1000)
        self._autosave_timer.timeout.connect(self._autosave_now)

        self._setup_ui()
        self.apply_dock_defaults()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- top row: project inputs --------------------------------------
        top = QHBoxLayout()
        top.addWidget(QLabel("Target folder:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Plugin output folder (contains frames_t / frames_w)")
        top.addWidget(self.folder_edit, 1)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._on_browse_folder)
        top.addWidget(browse_btn)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load_clicked)
        top.addWidget(load_btn)

        top.addSpacing(12)
        top.addWidget(QLabel("Label modality:"))
        self.modality_combo = QComboBox()
        self.modality_combo.addItem("Thermal", "t")
        self.modality_combo.addItem("RGB", "w")
        self.modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        top.addWidget(self.modality_combo)

        top.addStretch()
        self.autosave_check = QCheckBox("Autosave")
        self.autosave_check.setToolTip(
            "Automatically save the labels (labels.json / labels.csv) "
            "shortly after every change.")
        self.autosave_check.setChecked(
            QSettings().value("bambi/labelling_tool/autosave", False, type=bool))
        self.autosave_check.toggled.connect(self._on_autosave_toggled)
        top.addWidget(self.autosave_check)
        self.save_btn = QPushButton("Save Labels")
        self.save_btn.clicked.connect(self._on_save)
        top.addWidget(self.save_btn)
        layout.addLayout(top)

        # ---- center: canvas + side panel ----------------------------------
        center = QHBoxLayout()

        self.canvas = _LabelCanvas()
        # NoFocus so arrow keys reach the dialog's keyPressEvent for frame
        # navigation instead of scrolling the graphics view.
        self.canvas.setFocusPolicy(Qt.NoFocus)
        self.canvas.boxDrawn.connect(self._on_box_drawn)
        center.addWidget(self.canvas, 1)

        center.addWidget(self._build_side_panel())
        layout.addLayout(center, 1)

        # ---- bottom: navigation --------------------------------------------
        nav = QHBoxLayout()
        self.nav_step_spin = QSpinBox()
        self.nav_step_spin.setRange(1, 500)
        self.nav_step_spin.setValue(10)
        self.nav_step_spin.setToolTip(
            "Frame stride of the step navigation buttons (and PageUp / "
            "PageDown).")
        self.prev_step_btn = QPushButton("<< Step")
        self.prev_step_btn.clicked.connect(lambda: self._go_relative(-self.nav_step_spin.value()))
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(lambda: self._go_relative(-1))
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(lambda: self._go_relative(1))
        self.next_step_btn = QPushButton("Step >>")
        self.next_step_btn.clicked.connect(lambda: self._go_relative(self.nav_step_spin.value()))

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self._on_frame_spin)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)

        nav.addWidget(self.prev_step_btn)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.frame_spin)
        nav.addWidget(self.frame_slider, 1)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.next_step_btn)
        nav.addWidget(QLabel("Step:"))
        nav.addWidget(self.nav_step_spin)
        layout.addLayout(nav)

        self.timeline = _TimelineWidget()
        self.timeline.frameClicked.connect(self._goto_frame)
        layout.addWidget(self.timeline)

        self.status_label = QLabel("No project loaded.")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        # Overlays
        overlay_group = QGroupBox("Overlays (read-only)")
        og = QVBoxLayout(overlay_group)
        self.show_det_check = QCheckBox("Show detections")
        self.show_det_check.setChecked(True)
        self.show_det_check.stateChanged.connect(lambda _: self._render_frame())
        self.show_tracks_check = QCheckBox("Show existing tracks")
        self.show_tracks_check.setChecked(True)
        self.show_tracks_check.stateChanged.connect(lambda _: self._render_frame())
        og.addWidget(self.show_det_check)
        og.addWidget(self.show_tracks_check)

        import_row = QHBoxLayout()
        self.import_track_combo = QComboBox()
        self.import_track_combo.setToolTip("Existing pipeline track id")
        import_btn = QPushButton("Import as label track")
        import_btn.setToolTip(
            "Convert the selected pipeline track into an editable label "
            "track.")
        import_btn.clicked.connect(self._on_import_track)
        import_row.addWidget(self.import_track_combo)
        import_row.addWidget(import_btn)
        og.addLayout(import_row)

        resample_row = QHBoxLayout()
        resample_row.addWidget(QLabel("Resample:"))
        self.import_resample_spin = QSpinBox()
        self.import_resample_spin.setRange(1, 500)
        self.import_resample_spin.setValue(1)
        self.import_resample_spin.setToolTip(
            "Keep a key frame only at every N-th frame of the imported "
            "track (first and last frame are always kept). 1 = keep every "
            "frame as a key frame (no simplification).")
        resample_row.addWidget(self.import_resample_spin)
        resample_row.addStretch()
        og.addLayout(resample_row)
        vbox.addWidget(overlay_group)

        # Label tracks
        tracks_group = QGroupBox("Label tracks")
        tg = QVBoxLayout(tracks_group)
        self.track_list = QListWidget()
        self.track_list.currentItemChanged.connect(self._on_track_list_selection)
        tg.addWidget(self.track_list)

        btn_row = QHBoxLayout()
        self.new_track_btn = QPushButton("New Track (N)")
        self.new_track_btn.setCheckable(True)
        self.new_track_btn.setToolTip(
            "Draw a bounding box on the canvas to start a new track at the "
            "current frame.")
        self.new_track_btn.toggled.connect(self._on_new_track_toggled)
        del_track_btn = QPushButton("Delete Track")
        del_track_btn.clicked.connect(self._on_delete_track)
        btn_row.addWidget(self.new_track_btn)
        btn_row.addWidget(del_track_btn)
        tg.addLayout(btn_row)
        vbox.addWidget(tracks_group)

        # Attributes
        attr_group = QGroupBox("Classes")
        ag = QVBoxLayout(attr_group)

        def combo_row(label, combo):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(combo, 1)
            ag.addLayout(row)

        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)
        self.species_combo.addItems(SPECIES_CLASSES)
        # editable combo: react on both selection and manual edit
        self.species_combo.currentTextChanged.connect(self._on_attributes_changed)
        combo_row("Species:", self.species_combo)

        self.sex_combo = QComboBox()
        self.sex_combo.addItems(SEX_CLASSES)
        self.sex_combo.currentTextChanged.connect(self._on_attributes_changed)
        combo_row("Sex:", self.sex_combo)

        self.age_combo = QComboBox()
        self.age_combo.addItems(AGE_CLASSES)
        self.age_combo.currentTextChanged.connect(self._on_attributes_changed)
        combo_row("Age:", self.age_combo)

        self.occlusion_combo = QComboBox()
        self.occlusion_combo.addItems(OCCLUSION_LEVELS)
        self.occlusion_combo.setToolTip(
            "Occlusion level of the current key frame (stored per key frame).")
        self.occlusion_combo.currentTextChanged.connect(self._on_occlusion_changed)
        combo_row("Occlusion:", self.occlusion_combo)
        vbox.addWidget(attr_group)

        # Key frames
        kf_group = QGroupBox("Key frames")
        kg = QVBoxLayout(kf_group)
        self.kf_info_label = QLabel("–")
        kg.addWidget(self.kf_info_label)
        kf_row = QHBoxLayout()
        add_kf_btn = QPushButton("Set key frame (K)")
        add_kf_btn.setToolTip(
            "Store the currently shown (interpolated) box as a key frame "
            "on this frame.")
        add_kf_btn.clicked.connect(self._on_add_keyframe)
        del_kf_btn = QPushButton("Delete key frame")
        del_kf_btn.clicked.connect(self._on_delete_keyframe)
        kf_row.addWidget(add_kf_btn)
        kf_row.addWidget(del_kf_btn)
        kg.addLayout(kf_row)
        vbox.addWidget(kf_group)

        # Geo propagation
        prop_group = QGroupBox("Geo-referenced propagation")
        pg = QVBoxLayout(prop_group)
        prop_row = QHBoxLayout()
        prop_row.addWidget(QLabel("Frame offset:"))
        self.prop_offset_spin = QSpinBox()
        self.prop_offset_spin.setRange(-5000, 5000)
        self.prop_offset_spin.setValue(10)
        prop_row.addWidget(self.prop_offset_spin)
        pg.addLayout(prop_row)
        self.propagate_btn = QPushButton("Propagate box (geo)")
        self.propagate_btn.setToolTip(
            "Ray-cast the current box onto the DEM and back-project it into "
            "the offset frame; a key frame is created there. Usually only "
            "small size adaptions are needed afterwards.")
        self.propagate_btn.clicked.connect(self._on_propagate)
        pg.addWidget(self.propagate_btn)
        prop_hint = QLabel(
            "Projection quality depends on calibration and correction "
            "accuracy.")
        prop_hint.setWordWrap(True)
        prop_hint.setStyleSheet("color: #888; font-style: italic;")
        pg.addWidget(prop_hint)
        vbox.addWidget(prop_group)

        # Pipeline export
        export_group = QGroupBox("Pipeline export")
        eg = QVBoxLayout(export_group)
        self.export_det_btn = QPushButton("Add detections to project")
        self.export_det_btn.setToolTip(
            "Save the labels and merge the interpolated label boxes into "
            "detections_{t,w}/detections.txt in the same format as the "
            "'Detect Animals' stage, so geo-referencing and tracking can "
            "use them. Re-exporting replaces the previously added block.")
        self.export_det_btn.clicked.connect(self._on_add_detections_to_project)
        eg.addWidget(self.export_det_btn)
        vbox.addWidget(export_group)

        vbox.addStretch()
        fit_btn = QPushButton("Fit view")
        fit_btn.clicked.connect(self.canvas.fit)
        vbox.addWidget(fit_btn)
        return panel

    def keyPressEvent(self, event):
        """Frame navigation / labelling shortcuts.

        Handled at the dialog level so widgets that need the keys themselves
        (line edits, spin boxes, the editable species combo) keep working —
        keys only arrive here when the focused widget did not consume them.
        The canvas has ``Qt.NoFocus`` so arrow keys navigate frames instead
        of scrolling the view.
        """
        key = event.key()
        if key == Qt.Key_Right:
            self._go_relative(1)
        elif key == Qt.Key_Left:
            self._go_relative(-1)
        elif key == Qt.Key_PageUp:
            self._go_relative(self.nav_step_spin.value())
        elif key == Qt.Key_PageDown:
            self._go_relative(-self.nav_step_spin.value())
        elif key == Qt.Key_N:
            self.new_track_btn.toggle()
        elif key == Qt.Key_K:
            self._on_add_keyframe()
        elif key == Qt.Key_Delete:
            self._on_delete_keyframe()
        elif key == Qt.Key_Escape:
            # Do not close the dialog on Escape — just cancel draw mode.
            if self.new_track_btn.isChecked():
                self.new_track_btn.setChecked(False)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def apply_dock_defaults(self):
        """Seed folder / DEM / correction from the plugin dock widget."""
        if self._dock_widget is None:
            return
        try:
            config = self._dock_widget.get_config()
        except Exception:  # nosec B110
            return
        folder = config.get("target_folder", "")
        if not self.folder_edit.text() and folder:
            self.folder_edit.setText(folder)
            self._dem_path = config.get("dem_path", "")
            self._correction_path = config.get("correction_path", "")
            # Only auto-load quietly when extracted frames actually exist;
            # otherwise the user picks / fixes the folder manually.
            if os.path.isfile(os.path.join(folder, "poses_t.json")) or \
                    os.path.isfile(os.path.join(folder, "poses_w.json")):
                self._on_load_clicked()

    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select plugin output folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self._on_load_clicked()

    def _on_load_clicked(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "BAMBI Labelling Tool",
                                "Please select a valid target folder.")
            return
        if not self._maybe_save_dirty():
            return

        self._target_folder = folder
        if not self._dem_path or not self._correction_path:
            self._resolve_paths_from_layers()

        # Determine available modalities from the poses files
        has_t = os.path.isfile(os.path.join(folder, "poses_t.json"))
        has_w = os.path.isfile(os.path.join(folder, "poses_w.json"))
        if not has_t and not has_w:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                "No poses_t.json / poses_w.json found in the target folder.\n\n"
                "Please run the frame extraction step first.")
            return

        self._updating_ui = True
        self.modality_combo.model().item(0).setEnabled(has_t)
        self.modality_combo.model().item(1).setEnabled(has_w)
        if self.modality_combo.currentData() == "t" and not has_t:
            self.modality_combo.setCurrentIndex(1)
        elif self.modality_combo.currentData() == "w" and not has_w:
            self.modality_combo.setCurrentIndex(0)
        self._updating_ui = False

        self._load_modality()

    def _resolve_paths_from_layers(self):
        """Borrow DEM / correction paths from BAMBI layers of this folder."""
        try:
            from qgis.core import QgsProject, QgsVectorLayer
            for layer in QgsProject.instance().mapLayers().values():
                if not isinstance(layer, QgsVectorLayer):
                    continue
                if layer.customProperty("bambi_target_folder", "") != self._target_folder:
                    continue
                if not self._dem_path:
                    self._dem_path = layer.customProperty("bambi_dem_path", "")
                if not self._correction_path:
                    self._correction_path = layer.customProperty(
                        "bambi_correction_path", "")
                if self._dem_path and self._correction_path:
                    break
        except Exception:  # nosec B110
            pass
        # Common fallbacks inside the folder
        if not self._correction_path:
            candidate = os.path.join(self._target_folder, "correction.json")
            if os.path.isfile(candidate):
                self._correction_path = candidate

    def _on_modality_changed(self):
        if self._updating_ui or not self._target_folder:
            return
        if not self._maybe_save_dirty():
            return
        self._load_modality()

    def _load_modality(self):
        """(Re)load poses, overlays and label store for the current modality."""
        self._modality = self.modality_combo.currentData() or "t"
        m = self._modality

        poses_path = os.path.join(self._target_folder, f"poses_{m}.json")
        self._images = []
        if os.path.isfile(poses_path):
            try:
                with open(poses_path, "r", encoding="utf-8") as fh:
                    self._images = json.load(fh).get("images", [])
            except Exception:  # nosec B110
                pass
        if not self._images:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not read frames from {poses_path}.")
            return

        self._detections = _load_detections_by_frame(
            os.path.join(self._target_folder, f"detections_{m}", "detections.txt"))
        self._pixel_tracks = _load_pixel_tracks(
            os.path.join(self._target_folder, f"tracks_{m}", "tracks_pixel.csv"))

        self._store = LabelStore(self._target_folder, m)
        try:
            self._store.load()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not load existing labels:\n{exc}")
        self._dirty = False
        self._propagator = None  # modality-specific, rebuild lazily
        self._img_size = None
        self._selected_track = None

        # Navigation ranges
        total = len(self._images)
        self._updating_ui = True
        self.frame_spin.setRange(0, total - 1)
        self.frame_slider.setRange(0, total - 1)
        self.import_track_combo.clear()
        for tid in sorted(self._pixel_tracks.keys()):
            self.import_track_combo.addItem(f"Track {tid}", tid)
        self._updating_ui = False

        self._current_frame = min(self._current_frame, total - 1)
        self._refresh_track_list()
        self._goto_frame(self._current_frame, force=True)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_relative(self, delta: int):
        self._goto_frame(self._current_frame + delta)

    def _goto_frame(self, frame: int, force: bool = False):
        if not self._images:
            return
        frame = max(0, min(frame, len(self._images) - 1))
        if frame == self._current_frame and not force:
            return
        self._current_frame = frame
        self._updating_ui = True
        self.frame_spin.setValue(frame)
        self.frame_slider.setValue(frame)
        self._updating_ui = False
        self._render_frame()

    def _on_frame_spin(self, value: int):
        if not self._updating_ui:
            self._goto_frame(value)

    def _on_frame_slider(self, value: int):
        if not self._updating_ui:
            self._goto_frame(value)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _frame_image_path(self, frame: int) -> str:
        if not (0 <= frame < len(self._images)):
            return ""
        imagefile = self._images[frame].get("imagefile", "")
        if not imagefile:
            return ""
        path = os.path.join(
            self._target_folder, f"frames_{self._modality}", imagefile)
        return path if os.path.isfile(path) else ""

    def _render_frame(self):
        if not self._images or self._store is None:
            return
        frame = self._current_frame

        path = self._frame_image_path(frame)
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            pixmap = QPixmap(640, 512)
            pixmap.fill(QColor(40, 40, 40))
        if self._img_size is None and path:
            self._img_size = (pixmap.width(), pixmap.height())
        self.canvas.set_frame(pixmap)
        font_px = max(10, pixmap.width() // 80)

        self.canvas.clear_boxes()

        # ---- read-only overlays -------------------------------------------
        if self.show_det_check.isChecked():
            for det in self._detections.get(frame, []):
                item = _BoxItem(
                    -1, QRectF(det[0], det[1], det[2] - det[0], det[3] - det[1]),
                    _DETECTION_COLOR, f"det {det[4]:.2f}", font_px)
                item.setZValue(1)
                self.canvas.add_box(item)

        if self.show_tracks_check.isChecked():
            for tid, entries in self._pixel_tracks.items():
                for d in entries:
                    if d["frame"] == frame:
                        item = _BoxItem(
                            -1, QRectF(d["x1"], d["y1"],
                                       d["x2"] - d["x1"], d["y2"] - d["y1"]),
                            _TRACK_OVERLAY_COLOR, f"T{tid}", font_px)
                        item.setZValue(1)
                        self.canvas.add_box(item)

        # ---- label tracks ---------------------------------------------------
        for track in self._store.tracks.values():
            res = track.box_at(frame)
            if res is None:
                continue
            (x1, y1, x2, y2), is_kf, _occ = res
            selected = track.track_id == self._selected_track
            item = _BoxItem(
                track.track_id,
                QRectF(x1, y1, x2 - x1, y2 - y1),
                _track_color(track.track_id),
                f"L{track.track_id} {track.species}",
                font_px,
                editable=selected,
                dashed=not is_kf,
                emphasized=selected,
                on_committed=self._on_box_committed,
                on_clicked=self._select_track,
            )
            item.setZValue(3 if selected else 2)
            self.canvas.add_box(item)

        self._update_side_panel()
        self._update_timeline()
        self._update_status()

    def _update_timeline(self):
        keyframes: List[int] = []
        track_range = None
        if self._selected_track is not None and self._store:
            track = self._store.tracks.get(self._selected_track)
            if track:
                keyframes = track.frames()
                track_range = track.frame_range()
        self.timeline.set_data(
            len(self._images), self._current_frame, keyframes, track_range)

    def _update_status(self):
        parts = [f"Frame {self._current_frame} / {len(self._images) - 1}"]
        if self._selected_track is not None and self._store:
            track = self._store.tracks.get(self._selected_track)
            if track:
                res = track.box_at(self._current_frame)
                if res is None:
                    parts.append(
                        f"track L{track.track_id}: outside key-frame range")
                elif res[1]:
                    parts.append(f"track L{track.track_id}: KEY FRAME")
                else:
                    parts.append(f"track L{track.track_id}: interpolated")
        if self._dirty:
            parts.append("unsaved changes")
        self.status_label.setText("   |   ".join(parts))

    def _update_side_panel(self):
        """Sync attribute editors with the selected track / current frame."""
        self._updating_ui = True
        track = None
        if self._selected_track is not None and self._store:
            track = self._store.tracks.get(self._selected_track)

        enabled = track is not None
        for w in (self.species_combo, self.sex_combo, self.age_combo,
                  self.occlusion_combo):
            w.setEnabled(enabled)

        if track is not None:
            self.species_combo.setCurrentText(track.species)
            self.sex_combo.setCurrentText(track.sex)
            self.age_combo.setCurrentText(track.age)
            res = track.box_at(self._current_frame)
            if res is not None:
                self.occlusion_combo.setCurrentText(res[2])
            kfs = track.frames()
            kf_list = ", ".join(str(f) for f in kfs[:12])
            ellipsis = "…" if len(kfs) > 12 else ""
            self.kf_info_label.setText(
                f"{len(kfs)} key frame(s): {kf_list}{ellipsis}")
        else:
            self.kf_info_label.setText("–")
        self._updating_ui = False

    def _refresh_track_list(self):
        self._updating_ui = True
        self.track_list.clear()
        if self._store:
            for track in sorted(self._store.tracks.values(),
                                key=lambda t: t.track_id):
                rng = track.frame_range()
                rng_str = f"{rng[0]}–{rng[1]}" if rng else "empty"
                item = QListWidgetItem(
                    f"L{track.track_id}  {track.species}  [{rng_str}]")
                item.setData(Qt.UserRole, track.track_id)
                item.setForeground(_track_color(track.track_id))
                self.track_list.addItem(item)
                if track.track_id == self._selected_track:
                    self.track_list.setCurrentItem(item)
        self._updating_ui = False

    # ------------------------------------------------------------------
    # Track selection / creation / deletion
    # ------------------------------------------------------------------

    def _select_track(self, track_id: Optional[int]):
        if track_id == -1:  # read-only overlay clicked
            return
        if track_id == self._selected_track:
            return
        self._selected_track = track_id
        self._refresh_track_list()
        # Deferred: this is reached from a _BoxItem mouse event — re-rendering
        # immediately would delete the item that is still handling the event.
        QTimer.singleShot(0, self._render_frame)

    def _on_track_list_selection(self, current, _previous):
        if self._updating_ui or current is None:
            return
        tid = current.data(Qt.UserRole)
        if tid != self._selected_track:
            self._selected_track = tid
            # Jump into the track's range so the user sees the box
            track = self._store.tracks.get(tid) if self._store else None
            if track:
                rng = track.frame_range()
                if rng and not (rng[0] <= self._current_frame <= rng[1]):
                    self._goto_frame(rng[0], force=True)
                    return
            self._render_frame()

    def _on_new_track_toggled(self, checked: bool):
        self.canvas.set_draw_mode(checked)

    def _clamp_rect(self, rect: QRectF) -> QRectF:
        """Clamp *rect* to the frame image area (scene rect)."""
        scene_rect = self.canvas.scene().sceneRect()
        clamped = rect.intersected(scene_rect)
        return clamped if not clamped.isEmpty() else rect

    def _on_box_drawn(self, rect: QRectF):
        """A new box was drawn on the canvas (New Track mode)."""
        if self._store is None:
            return
        rect = self._clamp_rect(rect)
        self.new_track_btn.setChecked(False)
        track = LabelTrack(self._store.next_track_id())
        track.species = self.species_combo.currentText() or "unknown"
        track.set_keyframe(
            self._current_frame,
            (rect.left(), rect.top(), rect.right(), rect.bottom()),
            occlusion="none")
        self._store.tracks[track.track_id] = track
        self._selected_track = track.track_id
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_delete_track(self):
        track = self._current_track()
        if track is None:
            return
        reply = QMessageBox.question(
            self, "Delete Track",
            f"Delete label track L{track.track_id} "
            f"({len(track.keyframes)} key frame(s))?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        del self._store.tracks[track.track_id]
        self._selected_track = None
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_import_track(self):
        """Convert the selected pipeline track into an editable label track."""
        if self._store is None or self.import_track_combo.currentIndex() < 0:
            return
        src_tid = self.import_track_combo.currentData()
        entries = self._pixel_tracks.get(src_tid, [])
        if not entries:
            return

        step = self.import_resample_spin.value()
        track = LabelTrack(self._store.next_track_id())
        frames = [d["frame"] for d in entries]
        first, last = frames[0], frames[-1]
        for d in entries:
            f = d["frame"]
            if f == first or f == last or (f - first) % step == 0:
                track.set_keyframe(
                    f, (d["x1"], d["y1"], d["x2"], d["y2"]), occlusion="none")

        self._store.tracks[track.track_id] = track
        self._selected_track = track.track_id
        self._mark_dirty()
        self._refresh_track_list()
        self._goto_frame(first, force=True)

    def _current_track(self) -> Optional[LabelTrack]:
        if self._store is None or self._selected_track is None:
            return None
        return self._store.tracks.get(self._selected_track)

    # ------------------------------------------------------------------
    # Key frame editing
    # ------------------------------------------------------------------

    def _on_box_committed(self, track_id: int, rect: QRectF):
        """Selected box was moved/resized — write a key frame here."""
        track = self._store.tracks.get(track_id) if self._store else None
        if track is None:
            return
        rect = self._clamp_rect(rect)
        track.set_keyframe(
            self._current_frame,
            (rect.left(), rect.top(), rect.right(), rect.bottom()))
        self._mark_dirty()
        self._refresh_track_list()
        # Deferred: reached from the _BoxItem's mouseReleaseEvent — see
        # _select_track.
        QTimer.singleShot(0, self._render_frame)

    def _on_add_keyframe(self):
        """Freeze the interpolated box at the current frame as a key frame."""
        track = self._current_track()
        if track is None:
            return
        res = track.box_at(self._current_frame)
        if res is None:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "The current frame is outside the track's key-frame range.\n"
                "Use 'Propagate box (geo)' or move/draw the box directly to "
                "extend the track.")
            return
        if res[1]:
            return  # already a key frame
        track.set_keyframe(self._current_frame, res[0], occlusion=res[2])
        self._mark_dirty()
        self._render_frame()

    def _on_delete_keyframe(self):
        track = self._current_track()
        if track is None:
            return
        if not track.remove_keyframe(self._current_frame):
            return
        if not track.keyframes:
            del self._store.tracks[track.track_id]
            self._selected_track = None
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_attributes_changed(self):
        if self._updating_ui:
            return
        track = self._current_track()
        if track is None:
            return
        track.species = self.species_combo.currentText()
        track.sex = self.sex_combo.currentText()
        track.age = self.age_combo.currentText()
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_occlusion_changed(self):
        if self._updating_ui:
            return
        track = self._current_track()
        if track is None:
            return
        res = track.box_at(self._current_frame)
        if res is None:
            return
        # Occlusion is stored per key frame; setting it on an interpolated
        # frame promotes that frame to a key frame.
        track.set_keyframe(
            self._current_frame, res[0],
            occlusion=self.occlusion_combo.currentText())
        self._mark_dirty()
        self._render_frame()

    # ------------------------------------------------------------------
    # Geo-referenced propagation
    # ------------------------------------------------------------------

    def _on_propagate(self):
        track = self._current_track()
        if track is None:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "Please select a label track first.")
            return
        res = track.box_at(self._current_frame)
        if res is None:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "The selected track has no box on the current frame.")
            return
        offset = self.prop_offset_spin.value()
        if offset == 0:
            return
        dst_frame = self._current_frame + offset
        if not (0 <= dst_frame < len(self._images)):
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Target frame {dst_frame} is outside the frame range.")
            return

        if self._propagator is None:
            if not self._dem_path:
                self._resolve_paths_from_layers()
            self._propagator = _GeoPropagator(
                self._target_folder, self._dem_path,
                self._correction_path, self._modality)

        if not self._propagator.is_loaded:
            reply = QMessageBox.question(
                self, "Load Digital Elevation Model",
                "Geo-referenced propagation needs to load the DEM mesh.\n\n"
                "This may take some time on the first use. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return

        if self._img_size is None:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                "Frame image size unknown — cannot project.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            new_box = self._propagator.propagate(
                res[0], self._current_frame, dst_frame, self._images,
                self._img_size[0], self._img_size[1])
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Propagation Failed", str(exc))
            return
        QApplication.restoreOverrideCursor()

        track.set_keyframe(dst_frame, new_box, occlusion=res[2])
        self._mark_dirty()
        self._refresh_track_list()
        self._goto_frame(dst_frame, force=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _mark_dirty(self):
        self._dirty = True
        if self.autosave_check.isChecked():
            self._autosave_timer.start()  # (re)start the debounce interval

    def _on_autosave_toggled(self, checked: bool):
        QSettings().setValue("bambi/labelling_tool/autosave", checked)
        if checked and self._dirty:
            self._autosave_now()
        elif not checked:
            self._autosave_timer.stop()

    def _autosave_now(self):
        """Debounced autosave — silent except for the status line."""
        if not self._dirty or self._store is None:
            return
        try:
            self._store.save()
        except Exception as exc:
            # Do not nag once per edit: disable autosave and tell the user.
            self.autosave_check.setChecked(False)
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Autosave failed — autosave has been disabled:\n{exc}")
            return
        self._dirty = False
        self._update_status()
        self.status_label.setText(
            self.status_label.text() + "   |   autosaved")

    def _on_save(self):
        if self._store is None:
            return
        self._autosave_timer.stop()  # manual save supersedes a pending one
        try:
            self._store.save()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool", f"Could not save labels:\n{exc}")
            return
        self._dirty = False
        self._update_status()
        self.status_label.setText(
            f"Saved {len(self._store.tracks)} track(s) to {self._store.json_path}")

    def _on_add_detections_to_project(self):
        """Merge the label boxes into the pipeline's detections.txt."""
        if self._store is None:
            return
        if not any(t.keyframes for t in self._store.tracks.values()):
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "There are no label tracks with key frames to export.")
            return
        try:
            # Keep labels.json/csv in sync with what is exported.
            self._store.save()
            self._dirty = False
            det_file, n_boxes = self._store.export_to_detections()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not export detections:\n{exc}")
            return

        # Refresh the read-only overlay so the exported boxes show up.
        self._detections = _load_detections_by_frame(det_file)
        self._render_frame()

        QMessageBox.information(
            self, "BAMBI Labelling Tool",
            f"Added {n_boxes} bounding box(es) from "
            f"{len(self._store.tracks)} label track(s) to:\n{det_file}\n\n"
            "Re-run 'Geo-Reference Detections' (and tracking, if needed) in "
            "the plugin panel to update the QGIS layers.")

    def _maybe_save_dirty(self) -> bool:
        """Ask the user about unsaved changes. Returns False on cancel."""
        if self._dirty and self._store is not None \
                and self.autosave_check.isChecked():
            # Autosave mode: flush pending changes without prompting.
            self._autosave_timer.stop()
            self._autosave_now()
        if not self._dirty or self._store is None:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Labels",
            "There are unsaved label changes. Save them now?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._on_save()
        else:
            # Discard: restore the on-disk state so the in-memory store does
            # not silently keep (and later save) the discarded changes.
            self._dirty = False
            self._selected_track = None
            try:
                self._store.load()
            except Exception:  # nosec B110
                self._store.tracks = {}
            self._refresh_track_list()
            self._render_frame()
        return True

    def closeEvent(self, event):
        if not self._maybe_save_dirty():
            event.ignore()
            return
        super().closeEvent(event)
