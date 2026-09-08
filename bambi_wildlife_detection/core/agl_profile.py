# -*- coding: utf-8 -*-
"""The flight's altitude above the ground it flew over, as a profile.

Per extracted frame: where the aircraft was (from the poses), how high the
DEM is straight below it, and the difference - the height above ground
level (AGL) that decides pixel size, detection range and what a thermal
camera can resolve. Along the x axis the frames are laid out either by the
distance flown or by the time elapsed since the first frame.

Everything is absolute: the poses store positions relative to the DEM mesh
origin (``dem_mesh.json`` ``origin``), so that origin is added back before
anything is reported. No Qt in here; the tool window only draws what this
returns.
"""
import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Origin = Tuple[float, float, float]


def dem_origin(dem_path: str = "", dem_json_path: str = "",
               target_folder: str = "") -> Optional[Origin]:
    """The DEM mesh origin ``(x, y, z)`` in the project CRS, or ``None``.

    Looked for, in order, in the explicit metadata JSON, in the JSON beside
    the mesh, and in any ``*.json`` carrying an ``origin`` in the target
    folder or its parent - the same places the rest of the plugin looks.
    """
    candidates: List[str] = []
    if dem_json_path:
        candidates.append(dem_json_path)
    if dem_path:
        base = os.path.splitext(dem_path)[0]
        candidates.append(base + ".json")
    for folder in (target_folder, os.path.dirname(target_folder or "")):
        if folder and os.path.isdir(folder):
            try:
                candidates += [os.path.join(folder, name)
                               for name in sorted(os.listdir(folder))
                               if name.lower().endswith(".json")]
            except OSError:
                pass
    seen = set()
    for path in candidates:
        if not path or path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        origin = data.get("origin") if isinstance(data, dict) else None
        if origin and len(origin) >= 3:
            try:
                return (float(origin[0]), float(origin[1]), float(origin[2]))
            except (TypeError, ValueError):
                continue
    return None


def _seconds(value) -> Optional[float]:
    """Epoch seconds of a stored capture time (ISO or epoch), or ``None``."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:  # nosec B110
        return None


_MESH_CACHE: Dict[str, object] = {}


def ground_heights(mesh_path: str,
                   xy: Sequence[Tuple[float, float]]) -> List[Optional[float]]:
    """DEM height (mesh-local z) straight below each ``(x, y)``, or ``None``.

    One batched vertical ray cast per call: a flight is a thousand frames
    and casting them one by one through the Python loop was what made the
    inspector's per-click cast feel slow when repeated.
    """
    if not mesh_path or not xy:
        return [None for _ in xy]
    try:
        import numpy as np
        from alfspy.render.render import read_gltf
        from trimesh import Trimesh

        mesh = _MESH_CACHE.get(mesh_path)
        if mesh is None:
            mesh_data, _ = read_gltf(mesh_path)
            mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
            _MESH_CACHE[mesh_path] = mesh
        origins = np.array([[x, y, 10_000.0] for x, y in xy], dtype=float)
        directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(xy), 1))
        locations, index_ray, _ = mesh.ray.intersects_location(
            origins, directions, multiple_hits=False)
        result: List[Optional[float]] = [None for _ in xy]
        for location, ray in zip(locations, index_ray):
            z = float(location[2])
            current = result[int(ray)]
            result[int(ray)] = z if current is None else max(current, z)
        return result
    except Exception:  # nosec B110 - no mesh stack, or a mesh that fails
        return [None for _ in xy]


def build_profile(target_folder: str, modality: str, dem_path: str = "",
                  dem_json_path: str = "",
                  ground_fn: Optional[Callable[[Sequence[Tuple[float, float]]],
                                               List[Optional[float]]]] = None
                  ) -> dict:
    """The AGL profile of one camera's flight.

    :param modality: ``"t"`` or ``"w"``
    :param dem_path: the DEM mesh (``.glb``/``.gltf``) or anything whose
        sibling ``.json`` names the origin
    :param dem_json_path: the DEM metadata JSON, when known
    :param ground_fn: what samples the ground under ``[(x, y), …]`` in
        mesh-local coordinates; defaults to the DEM mesh ray cast
    :return: ``{"points": [...], "origin": (x, y, z) | None,
        "mesh_path": str, "modality": str, "n_ground": int}`` where each
        point holds ``frame, x, y, east, north, altitude, ground, agl,
        distance_m, time_s, timestamp`` - ``ground``/``agl`` ``None`` where
        the DEM has no height there.
    """
    from . import inspection

    images = inspection._pose_images(target_folder, modality)
    origin = dem_origin(dem_path, dem_json_path, target_folder)
    ox, oy, oz = origin if origin else (0.0, 0.0, 0.0)

    mesh_path = inspection.find_dem_mesh_path(dem_path, dem_json_path) or ""
    if not mesh_path and dem_path and os.path.isfile(dem_path):
        mesh_path = dem_path

    located = []
    for frame, image in enumerate(images):
        location = image.get("location")
        if not location or len(location) < 3:
            continue
        try:
            x, y, z = (float(location[0]), float(location[1]),
                       float(location[2]))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        located.append((frame, x, y, z, image.get("timestamp")
                        or image.get("epoch")))

    xy = [(x, y) for _f, x, y, _z, _t in located]
    if ground_fn is not None:
        grounds = list(ground_fn(xy))
    else:
        grounds = ground_heights(mesh_path, xy)

    points = []
    distance = 0.0
    start_time = None
    previous = None
    for (frame, x, y, z, stamp), ground in zip(located, grounds):
        if previous is not None:
            distance += math.hypot(x - previous[0], y - previous[1])
        previous = (x, y)
        seconds = _seconds(stamp)
        if seconds is not None and start_time is None:
            start_time = seconds
        ground_abs = (ground + oz) if ground is not None else None
        points.append({
            "frame": frame,
            "x": x, "y": y,
            "east": x + ox, "north": y + oy,
            "altitude": z + oz,
            "ground": ground_abs,
            "agl": (z - ground) if ground is not None else None,
            "distance_m": distance,
            "time_s": (seconds - start_time) if seconds is not None
            and start_time is not None else None,
            "timestamp": stamp if stamp is not None else "",
        })
    return {
        "points": points,
        "origin": origin,
        "mesh_path": mesh_path,
        "modality": modality,
        "n_ground": sum(1 for p in points if p["ground"] is not None),
    }


def summarise(profile: dict) -> List[str]:
    """Log/status lines: frames, ground coverage and the AGL range."""
    points = profile.get("points", [])
    if not points:
        return ["No poses with a position."]
    lines = [f"{len(points)} frame(s), {points[-1]['distance_m']:,.0f} m flown"]
    times = [p["time_s"] for p in points if p["time_s"] is not None]
    if times:
        lines[0] += f" in {max(times):,.0f} s"
    agl = [p["agl"] for p in points if p["agl"] is not None]
    if agl:
        lines.append(f"AGL {min(agl):.1f}-{max(agl):.1f} m, mean "
                     f"{sum(agl) / len(agl):.1f} m over {len(agl)} frame(s) "
                     "with DEM ground")
    else:
        lines.append("No DEM ground under the route - check the DEM mesh path")
    if profile.get("origin") is None:
        lines.append("Warning: no DEM origin found; altitudes are mesh-local, "
                     "not above sea level")
    return lines
