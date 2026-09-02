# -*- coding: utf-8 -*-
"""Construction of the alfspy render context.

alfspy 3.0 rasterises through one of three interchangeable engines - ModernGL,
PyTorch or Vulkan - selected by ``$ALFS_ENGINE`` and created through a single
``make_context()``. Before 3.0 the engines were two separate distributions that
each installed a package called ``alfspy`` and offered their own factory
(``make_mgl_context`` / ``make_torch_context``); this module used to sniff which
one was installed. There is nothing left to sniff: one package now provides all
three, and which is *usable* depends on the pip extras that were installed with
it.

So the plugin still never names an engine at a call site. It exports the
project's choice into the environment once per step (see
``core.alfs_runtime.apply_selection``) and asks for a context here, and alfspy
resolves the rest.

This module is free of QGIS/Qt imports; alfspy is imported lazily so the plugin
still loads when the dependency is missing.
"""
from typing import List, Optional


def available_engines() -> List[str]:
    """The engines that can actually create a context on this machine.

    alfspy probes rather than merely importing, because ModernGL imports
    perfectly well on a machine with no usable GL driver and only fails when a
    context is created - which is exactly the failure this is here to predict.

    :return: engine names, or ``[]`` when alfspy is not installed at all
    """
    try:
        from alfspy.core.backends import available_engines as _available
    except ImportError:
        return []
    try:
        return list(_available())
    except Exception:      # noqa: BLE001 - a probe must not raise
        return []


def render_backend() -> str:
    """Name the engine a render would use right now.

    :return: the resolved engine name (``"moderngl"``, ``"torch"``,
        ``"vulkan"``, or whatever ``$ALFS_ENGINE`` names), or ``"unavailable"``
        when alfspy cannot be imported at all
    """
    try:
        from alfspy.core.backends import resolve_engine
    except ImportError:
        return "unavailable"
    return str(resolve_engine())


def make_render_context(device: Optional[str] = None,
                        engine: Optional[str] = None):
    """Create an alfspy render context.

    :param device: which device to render on (``"cuda"``, ``"cpu"``). PyTorch
        uses it directly, Vulkan maps ``"cpu"`` onto its software adapter, and
        ModernGL ignores it - OpenGL offers no device selection. Left ``None``,
        alfspy resolves ``$ALFS_DEVICE`` and then the backend's own choice, so
        nothing here has to decide.
    :param engine: force an engine, overriding ``$ALFS_ENGINE``. Left ``None``
        by every caller in the plugin: the choice belongs to the project, and
        naming one here would put it back in the code.
    :return: the engine's context object, ready to pass to ``Renderer`` and
        ``CtxShot``
    :raises RuntimeError: when alfspy is missing, or the selected engine cannot
        create a context here
    """
    try:
        from alfspy.core.backends import make_context
    except ImportError as exc:
        raise RuntimeError(f"alfspy is not available: {exc}") from exc

    try:
        return make_context(engine, device=device)
    except Exception as exc:
        # The common case by far is an engine that was selected but never
        # installed - its extra is a separate download - so say which ones do
        # work here rather than letting an ImportError from three frames down
        # be the whole explanation.
        wanted = engine or render_backend()
        usable = available_engines()
        working = (f"Working engines on this machine: {', '.join(usable)}."
                   if usable else "No render engine works on this machine.")
        raise RuntimeError(
            f"The '{wanted}' render engine could not start: {exc}\n\n"
            f"{working}\n"
            "Choose one of them in the Dependency Manager, or press Install "
            "there to add what the current choice needs."
        ) from exc


def make_ray_caster(mesh):
    """Build a reusable ray caster for *mesh*, or return *mesh* unchanged.

    alfspy 3.0 selects the caster - Embree or Warp - through
    ``$ALFS_RAYCASTER``, and ``pixel_to_world_coord`` accepts either a mesh or
    a built caster. Handing it a mesh rebuilds the acceleration structure on
    **every** call, which is invisible for a one-off projection and ruinous
    inside the correction wizard's solver loop. So anything that projects more
    than once builds the caster here and passes that instead.

    Falls back to the mesh itself when alfspy is too old or the caster cannot
    be built: the result is slower, not wrong, and a projection that works is
    worth more than one that refuses to start.
    """
    try:
        from alfspy.core.raycast import create_raycaster
    except ImportError:
        return mesh
    try:
        return create_raycaster(mesh)
    except Exception:      # noqa: BLE001 - a slower path is not a failure
        return mesh


def require_render_stack() -> None:
    """Raise a helpful ImportError when the rendering dependencies are missing.

    Checks only what the plugin itself needs - alfspy plus the array/geometry
    libraries it hands data to. Which engine alfspy rasterises with is its own
    business, and is reported through :func:`make_render_context` when a
    context is actually asked for.
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

    if render_backend() == "unavailable":
        raise ImportError(
            "alfspy is not installed.\n\n"
            "Install it from the Dependency Manager (ALFS-PY Framework), "
            "where the render engine and ray caster are also chosen - each is "
            "an optional extra of the same release."
        )

    # The first alfspy 3.0.0 build imported its PyTorch rasteriser eagerly from
    # ``alfspy/render/render.py``, so the whole ``alfspy.render`` package needed
    # torch whichever engine was selected. That is fixed, but the fix shipped
    # under the same 3.0.0 version, so an install made before it cannot be told
    # apart by version - and every step that renders imports from there. The
    # bare "No module named 'torch'" three frames in explains nothing, so say
    # what it means. On a current install the check simply never fires.
    try:
        __import__("alfspy.render.render")
    except ImportError as exc:
        if "torch" not in str(exc):
            raise
        raise ImportError(
            "This alfspy build needs PyTorch before it can render at all, "
            "whichever engine is selected - it loads its PyTorch rasteriser "
            "with the render module. That was fixed in alfspy after 3.0.0 was "
            "first published.\n\n"
            "Press Install in the Dependency Manager to fetch the current "
            "release, which does not need it.\n\n"
            f"Original error: {exc}"
        ) from exc
