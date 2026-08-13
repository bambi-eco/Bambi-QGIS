# -*- coding: utf-8 -*-
"""Single source for pose → camera construction and world → pixel projection.

The camera build applies the drone-pose correction exactly like
``BambiProcessor.run_georeference`` / alfspy's ``get_camera_for_frame``:

* the **1× rotation-correction rule**: corrections are subtracted once from
  the pose eulers (alfspy applies them exactly once when rendering, so all
  reprojection code must match - see the GeoTIFF rotation-sign incident);
* the corrected pose goes through alfspy's ``quaternion_from_drone_pose``,
  which applies the heading about **world up** after the tilt (the older
  ``quaternion_from_eulers(e, 'zyx')`` spelling applied it about the camera's
  own optical axis - correct at nadir, up to 128° wrong at the horizon);
* the result is conjugated when the *installed* alfspy still builds rays with
  the transposed rotation - see :func:`ray_convention`.

Before this module the same construction existed in five places
(box projector, click tool, labelling tool, correction wizard dialog and
its probe worker) that had to be kept in sync by hand.

The rotation convention
-----------------------

``alfspy.core.convert.pixel_to_world_coord`` has shipped in two incompatible
forms.  Older releases built rays with ``dirs @ R33.T`` - the *inverse* camera
rotation - and callers compensated by negating every Euler angle.  alfs_py
commit ``86e0d92`` (2026-08-04) fixed it to ``dirs @ R33``, at which point the
negation stops cancelling and starts corrupting: measured ray errors of 29° to
132° depending on pose, silently.

Rather than pin a version, this module probes the installed alfspy once and
builds the matching camera.  Both branches produce the *same world rays*, equal
to the renderer's own ``quaternion_from_eulers(eulers, 'zyx')``:

* legacy - ``Quaternion.from_eulers(-e)``, since
  ``from_eulers(-e).T == quaternion_from_eulers(e, 'zyx')`` exactly, and the
  legacy ray path transposes;
* fixed - ``quaternion_from_eulers(e, 'zyx')`` directly, which is literally the
  construction ``bambi.util.projection_util.create_shot`` uses for rendering.

Raises ``RuntimeError`` when alfspy/pyrr are not installed.
"""

from typing import Optional, Tuple

#: Cached result of :func:`ray_convention` - ``"legacy"``, ``"fixed"`` or ``None``.
_RAY_CONVENTION = None


def ray_convention(force_probe: bool = False) -> str:
    """Detect how the installed alfspy rotates camera-space rays into the world.

    Casts one ray through a sphere centred on the camera - every direction hits
    it exactly once - and compares the hit against both candidate rotations.

    :param force_probe: re-run the probe instead of using the cached answer
    :return: ``"legacy"`` when rays are built with ``R33.T``, ``"fixed"`` when
        built with ``R33``
    :raises RuntimeError: when alfspy/pyrr/trimesh are unavailable, or when the
        probe cannot tell the two conventions apart
    """
    global _RAY_CONVENTION
    if _RAY_CONVENTION is not None and not force_probe:
        return _RAY_CONVENTION

    import numpy as np
    try:
        import trimesh
        from pyrr import Quaternion, Vector3
        from alfspy.core.convert.convert import pixel_to_world_coord
        from alfspy.core.rendering import Camera
    except ImportError as exc:
        raise RuntimeError(f"alfspy / pyrr / trimesh not available: {exc}") from exc

    # Deliberately asymmetric so R33 and R33.T give clearly different rays.
    eulers = np.deg2rad([35.0, 12.0, 70.0])
    camera = Camera(fovy=50.0, aspect_ratio=1.0,
                    position=Vector3([0.0, 0.0, 0.0]),
                    rotation=Quaternion.from_eulers(Vector3(eulers)))
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=100.0)

    hits = pixel_to_world_coord([256], [128], 512, 512, sphere, camera,
                                include_misses=False)
    hits = np.reshape(np.asarray(hits, dtype=float), (-1, 3))
    if len(hits) != 1 or not np.all(np.isfinite(hits)):
        raise RuntimeError(
            "Could not probe the alfspy ray convention: the test ray did not "
            "hit the probe sphere.")
    hit = hits[0] / np.linalg.norm(hits[0])

    rot = np.asarray(camera.transform.rotation.matrix33, dtype=np.float64)
    tan_fov = np.tan(np.deg2rad(50.0) / 2.0)
    local = np.array([0.0 * tan_fov, 0.5 * tan_fov, -1.0])
    candidates = {
        "fixed": local @ rot,
        "legacy": local @ rot.T,
    }
    scores = {name: float(np.dot(hit, vec / np.linalg.norm(vec)))
              for name, vec in candidates.items()}
    best = max(scores, key=scores.get)
    if scores[best] < 0.999 or sorted(scores.values())[-2] > 0.99:
        raise RuntimeError(
            "Could not probe the alfspy ray convention unambiguously "
            f"(scores: {scores}). Refusing to guess the camera rotation.")

    _RAY_CONVENTION = best
    return best


def build_camera(meta: dict, t_corr: dict, r_corr: dict,
                 aspect_ratio: float = 1.0):
    """Build an ``alfspy.core.rendering.Camera`` from a poses-file entry.

    :param meta: one entry of the poses file's ``images`` list
        (keys ``location``, ``rotation`` in degrees, optional ``fovy``)
    :param t_corr: translation correction dict (``x``/``y``/``z``, metres)
    :param r_corr: rotation correction dict (``x``/``y``/``z``, radians)
    :param aspect_ratio: camera aspect ratio (width / height); the
        geo-referencing and box-projection paths use 1.0
    """
    import numpy as np
    try:
        from pyrr import Vector3
        from alfspy.core.rendering import Camera
        from alfspy.core.util.pyrrs import quaternion_from_drone_pose
    except ImportError as exc:
        raise RuntimeError(f"alfspy / pyrr not available: {exc}") from exc

    fovy = meta.get("fovy", 50.0)
    if isinstance(fovy, (list, tuple)):
        fovy = fovy[0] if fovy else 50.0
    fovy = float(fovy)

    cor_t = Vector3(
        [t_corr.get("x", 0), t_corr.get("y", 0), t_corr.get("z", 0)],
        dtype="f4")
    cor_r = Vector3(
        [r_corr.get("x", 0), r_corr.get("y", 0), r_corr.get("z", 0)],
        dtype="f4")

    position = Vector3(meta.get("location", [0.0, 0.0, 0.0])) + cor_t
    # Corrections are stored in radians; the pose itself is in degrees.
    rotation_eulers = (
        Vector3([np.deg2rad(v % 360.0)
                 for v in meta.get("rotation", [0.0, 0.0, 0.0])[:3]]) - cor_r
    )
    rotation_quat = quaternion_from_drone_pose(np.degrees(rotation_eulers))
    if ray_convention() == "legacy":
        # The legacy ray path multiplies by R33.T, so hand it the inverse
        # rotation; both branches then cast the same world ray.
        rotation_quat = rotation_quat.conjugate

    return Camera(
        fovy=fovy,
        aspect_ratio=aspect_ratio,
        position=position,
        rotation=rotation_quat,
    )


def world_to_pixel(
    corners,
    img_width: int,
    img_height: int,
    camera,
) -> "Tuple[object, object]":
    """Project world-space coordinates to pixel space.

    Reimplements the math of ``alfspy.core.convert.world_to_pixel_coord``
    directly to avoid a broadcasting bug in that function when more than one
    point is passed at a time (``ndc[:, 3]`` produces shape ``(N,)`` which
    cannot broadcast against ``ndc`` of shape ``(N, 4)``).

    Returns ``(pixel_xs, pixel_ys)`` arrays of the same length as *corners*.
    """
    import numpy as np

    coords = np.reshape(corners, (-1, 3)).astype(np.float64)
    n = len(coords)

    # Homogeneous coordinates (N, 4)
    homo = np.ones((n, 4), dtype=np.float64)
    homo[:, :3] = coords

    # View and projection (use float64 throughout)
    view = np.array(camera.get_view(), dtype=np.float64)
    proj = np.array(camera.get_proj(), dtype=np.float64)

    cam_coords = homo @ view  # (N, 4)
    ndc = cam_coords @ proj  # (N, 4)

    # Perspective divide - use column slice [:,3:4] so shape stays (N,1)
    w = ndc[:, 3:4]
    ndc_norm = ndc / w  # (N, 4)

    pixel_xs = (ndc_norm[:, 0] + 1.0) * img_width / 2.0
    pixel_ys = img_height - (ndc_norm[:, 1] + 1.0) * img_height / 2.0

    return pixel_xs, pixel_ys


def frame_camera(images: list, frame_idx: int, correction: Optional[dict],
                 aspect_ratio: float = 1.0):
    """Camera for ``images[frame_idx]`` with the per-frame correction applied.

    Convenience wrapper combining :func:`core.corrections.correction_for_frame`
    with :func:`build_camera`.
    """
    from .corrections import correction_for_frame

    t_corr, r_corr = correction_for_frame(frame_idx, correction or {})
    return build_camera(images[frame_idx], t_corr, r_corr, aspect_ratio)
