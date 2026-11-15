# -*- coding: utf-8 -*-
"""A real OpenGL context for the tests that need one, and a small flight to
render.

The render steps had no test that rendered: everything around them was
tested with stand-ins. Mesa's software renderer needs no display and no
graphics card, so where it is installed the real steps can run - and where
it is not, these tests are skipped.

* :func:`software_gl` - a ModernGL context without a display, or a skip;
* :class:`Alive` - counts the OpenGL objects alive on the current context;
* :func:`small_flight` - frames, poses and a flat DEM, with every render
  context the plugin asks for made on software OpenGL.
"""
import ctypes
import json
import os

import numpy as np
import pytest

N_FRAMES = 6
HEIGHT_AGL = 60.0
FOVY = 50.0
FRAME_SIZE = (160, 128)


def software_gl():
    """``(context, libGL)`` - a ModernGL context that needs no display and
    the library to ask what is alive on it - or a skip."""
    pytest.importorskip("moderngl")
    pytest.importorskip("alfspy")
    from alfspy.core.backends import make_context

    try:
        ctx = make_context("moderngl", backend="egl")
        library = ctypes.CDLL("libGL.so.1")
    except Exception as exc:      # noqa: BLE001 - no GL here: nothing to test
        pytest.skip(f"no OpenGL context without a display here: {exc}")
    return ctx, library


class Alive:
    """Counts the OpenGL objects alive on the current context."""

    KINDS = {"textures": "glIsTexture", "renderbuffers": "glIsRenderbuffer",
             "framebuffers": "glIsFramebuffer", "buffers": "glIsBuffer",
             "vertex arrays": "glIsVertexArray"}

    def __init__(self, library):
        self._is = {}
        for kind, name in self.KINDS.items():
            function = getattr(library, name)
            function.argtypes = [ctypes.c_uint]
            function.restype = ctypes.c_ubyte
            self._is[kind] = function

    def __call__(self):
        return {kind: sum(1 for name in range(1, 800) if function(name))
                for kind, function in self._is.items()}

    def none(self):
        return dict.fromkeys(self.KINDS, 0)


def ended(ctx) -> bool:
    """Whether a ModernGL context has been released."""
    import moderngl

    return isinstance(ctx.mglo, moderngl.InvalidObject)


def small_flight(tmp_path, monkeypatch, dem_size=500.0, rotation=(0.0, 0.0, 0.0)):
    """Six thermal frames from 60 m above flat ground, a metre apart.

    Every render context the plugin makes is made on software OpenGL and
    kept in ``contexts``; ``textures`` gets the number of textures alive
    each time a renderer is made, ``read`` the frame images as they are
    read from disk.
    """
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("trimesh")
    pytest.importorskip("bambi")
    pytest.importorskip("gltflib")
    pytest.importorskip("rasterio")
    ctx, library = software_gl()
    ctx.release()
    from alfspy.core.backends import make_context
    from alfspy.core.backends.moderngl_.renderer import Renderer
    from alfspy.core.backends.moderngl_.shot import CtxShot

    from bambi_wildlife_detection.bambi_processing import BambiProcessor
    from bambi_wildlife_detection.core import render_context

    root = str(tmp_path / "flight")
    frames = os.path.join(root, "frames_t")
    os.makedirs(frames)
    width, height = FRAME_SIZE
    picture = np.random.default_rng(0).integers(
        40, 255, (height, width, 3), dtype=np.uint8)
    for index in range(N_FRAMES):
        cv2.imwrite(os.path.join(frames, f"frame_{index:06d}.png"), picture)
    glb = os.path.join(root, "dem.glb")
    BambiProcessor.generate_flat_surface_mesh(
        47.0, 14.0, dem_size, dem_size, glb, os.path.join(root, "dem.json"),
        epsg=32633)
    with open(os.path.join(root, "poses_t.json"), "w") as handle:
        json.dump({"images": [
            {"imagefile": f"frame_{index:06d}.png",
             "location": [float(index), 0.0, HEIGHT_AGL],
             "rotation": list(rotation), "fovy": [FOVY]}
            for index in range(N_FRAMES)]}, handle)

    contexts = []

    def software_context(device=None, engine=None):
        contexts.append(make_context("moderngl", backend="egl"))
        return contexts[-1]

    monkeypatch.setattr(render_context, "make_render_context", software_context)
    monkeypatch.setattr(render_context, "require_render_stack", lambda: None)

    alive, textures = Alive(library), []
    make = Renderer.__init__

    def counting(self, *args, **kwargs):
        textures.append(alive()["textures"])
        make(self, *args, **kwargs)

    monkeypatch.setattr(Renderer, "__init__", counting)

    read = []
    use = CtxShot.tex_use

    def noting(self, *args, **kwargs):
        if self.tex is None:
            read.append(self)
        return use(self, *args, **kwargs)

    monkeypatch.setattr(CtxShot, "tex_use", noting)
    return {
        "root": root, "contexts": contexts, "textures": textures,
        "read": read, "processor": BambiProcessor(),
        "config": {"target_folder": root, "dem_path": glb,
                   "target_epsg": 32633, "alfs_camera": "T",
                   "geotiff_camera": "T", "alfs_ground_resolution": 0.5,
                   "alfs_create_overviews": False},
    }
