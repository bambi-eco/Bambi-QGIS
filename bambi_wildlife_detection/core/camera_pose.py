# -*- coding: utf-8 -*-
"""Single source for pose → camera construction and world → pixel projection.

The camera build applies the drone-pose correction exactly like
``BambiProcessor.run_georeference`` / alfspy's ``get_camera_for_frame``:

* the **1× rotation-correction rule**: corrections are subtracted once from
  the pose eulers (alfspy applies them exactly once when rendering, so all
  reprojection code must match — see the GeoTIFF rotation-sign incident);
* eulers are negated (``* -1``) to convert the compass-style pose rotation
  into the camera rotation.

Before this module the same construction existed in five places
(box projector, click tool, labelling tool, correction wizard dialog and
its probe worker) that had to be kept in sync by hand.

Raises ``RuntimeError`` when alfspy/pyrr are not installed.
"""

from typing import Optional, Tuple


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
        from pyrr import Vector3, Quaternion
        from alfspy.core.rendering import Camera
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
    rotation_eulers = (
        Vector3([np.deg2rad(v % 360.0)
                 for v in meta.get("rotation", [0.0, 0.0, 0.0])[:3]]) - cor_r
    ) * -1
    rotation_quat = Quaternion.from_eulers(rotation_eulers)

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

    # Perspective divide — use column slice [:,3:4] so shape stays (N,1)
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
