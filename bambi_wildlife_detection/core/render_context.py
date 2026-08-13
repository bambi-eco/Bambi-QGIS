# -*- coding: utf-8 -*-
"""Backend-neutral construction of the alfspy render context.

alfspy ships in two implementations that install under the same package name:

* ``alfs_py`` rasterises through ModernGL and offers ``make_mgl_context()``;
* ``alfs_pytorch`` rasterises through PyTorch tensors and offers
  ``make_torch_context()``, keeping ``make_mgl_context()`` as a deprecated
  alias so ModernGL-era callers keep working.

The plugin therefore never names a backend.  It asks for a context here, and
this module prefers the torch factory when it exists — which also picks up CUDA
automatically — and falls back to the ModernGL one otherwise.  Swapping the
installed alfspy needs no change anywhere else in the plugin, and nothing
outside alfspy imports ``moderngl``.

This module is free of QGIS/Qt imports; alfspy is imported lazily so the plugin
still loads when the dependency is missing.
"""
from typing import Optional


def render_backend() -> str:
    """Name the rasteriser the installed alfspy uses.

    :return: ``"torch"``, ``"moderngl"``, or ``"unavailable"`` when alfspy
        cannot be imported at all
    """
    try:
        from alfspy.render import render as _render
    except ImportError:
        return "unavailable"
    return "torch" if hasattr(_render, "make_torch_context") else "moderngl"


def make_render_context(device: Optional[str] = None):
    """Create an alfspy render context using whichever backend is installed.

    :param device: torch device to render on (e.g. ``"cuda"``, ``"cpu"``).
        Only meaningful for the torch backend, which otherwise selects CUDA
        when it is available; ignored by the ModernGL backend.
    :return: the backend's context object, ready to pass to ``Renderer`` and
        ``CtxShot``
    :raises RuntimeError: when alfspy is not installed
    """
    try:
        from alfspy.render import render as _render
    except ImportError as exc:
        raise RuntimeError(f"alfspy is not available: {exc}") from exc

    factory = getattr(_render, "make_torch_context", None)
    if factory is not None:
        return factory(device=device) if device else factory()
    return _render.make_mgl_context()


def require_render_stack() -> None:
    """Raise a helpful ImportError when the rendering dependencies are missing.

    Checks only what the plugin itself needs — alfspy plus the array/geometry
    libraries it hands data to.  The rasteriser's own dependency (ModernGL or
    PyTorch) is alfspy's business and is reported through it.
    """
    missing = []
    for label, module in (("pyrr", "pyrr"), ("trimesh", "trimesh")):
        try:
            __import__(module)
        except ImportError:
            missing.append(label)
    if missing:
        raise ImportError(
            "Required dependency not found: " + ", ".join(missing) + "\n\n"
            "Please install the following packages:\n"
            f"  pip install {' '.join(missing)}"
        )

    backend = render_backend()
    if backend == "unavailable":
        raise ImportError(
            "alfspy is not installed.\n\n"
            "Install it from the Dependency Manager (ALFS-PY Framework), "
            "which also pulls in the rasteriser it needs."
        )
