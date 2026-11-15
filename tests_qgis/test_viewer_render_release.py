# -*- coding: utf-8 -*-
"""The viewers give back what their renders hold on the graphics card.

Nothing on an OpenGL context is freed when the Python object goes away. The
3D Flight Viewer's light-field tab kept a renderer per image size and only
ever dropped them; its search for correction factors, and the Correction
Wizard's preview, made a context per run and left it - with all that was on
it - when the run failed, or always.

``core.render_resources`` is tested on a real context in the unit tests;
these are the workers that use it, with stand-ins for the engine.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import sys
import types

import numpy as np
import pytest


class Thing:
    def __init__(self, *args):
        self.args = args
        self.released = 0

    def release(self):
        self.released += 1


@pytest.fixture
def engine(monkeypatch):
    """A stand-in alfspy: renderers, shots and contexts that count their
    releases."""
    made = types.SimpleNamespace(renderers=[], contexts=[])

    def renderer(resolution, ctx, camera, mesh, texture=None):
        made.renderers.append(Thing(resolution, mesh))
        return made.renderers[-1]

    rendering = types.ModuleType("alfspy.core.rendering")
    rendering.Renderer = renderer
    rendering.Resolution = lambda width, height: (width, height)
    rendering.MeshData = lambda vertices, faces: ("mesh", float(vertices[0, 2]))
    rendering.CtxShot = Thing
    rendering.TextureData = Thing
    for name, module in {
            "alfspy": types.ModuleType("alfspy"),
            "alfspy.core": types.ModuleType("alfspy.core"),
            "alfspy.core.rendering": rendering}.items():
        monkeypatch.setitem(sys.modules, name, module)

    from bambi_wildlife_detection.core import render_context

    def context(device=None, engine=None):
        made.contexts.append(Thing())
        return made.contexts[-1]

    monkeypatch.setattr(render_context, "make_render_context", context)
    return made


DEM = types.SimpleNamespace(
    vertices=np.zeros((4, 3), dtype=np.float32),
    faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))


# ---------------------------------------------------------------------------
# 3D Flight Viewer: light-field tab
# ---------------------------------------------------------------------------

@pytest.fixture
def worker(engine):
    from bambi_wildlife_detection.viewer3d.alfs_panel import AlfsWorker

    worker = AlfsWorker()
    worker.ctx = Thing()
    return worker


def test_a_renderer_that_fits_is_used_again(worker, engine):
    first = worker._renderer((640, 512), "camera a", DEM, 0.0)
    again = worker._renderer((640, 512), "camera b", DEM, 0.0)

    assert again is first and len(engine.renderers) == 1
    assert again.camera == "camera b"


def test_stepping_through_a_flight_does_not_keep_a_renderer_per_frame(
        worker, engine):
    """Over the terrain the integral has another size for nearly every
    frame. Every one of those renderers used to stay."""
    for height in range(700, 760):
        worker._renderer((1024, height), "camera", DEM, 0.0)

    alive = [r for r in engine.renderers if not r.released]
    assert len(engine.renderers) == 60
    assert len(alive) == 2


def test_moving_the_focus_releases_the_renderers_of_the_old_surface(
        worker, engine):
    worker._renderer((640, 512), "camera", DEM, 0.0)
    worker._renderer((800, 600), "camera", DEM, 0.0)

    moved = worker._renderer((640, 512), "camera", DEM, 1.5)

    assert [r.released for r in engine.renderers] == [1, 1, 0]
    assert moved.args[1] == ("mesh", 1.5)        # made for the raised surface


def test_closing_gives_everything_back(worker, engine):
    ctx = worker.ctx
    shot = Thing()
    worker._shots[("frame", 0, 1024)] = shot
    worker._renderer((640, 512), "camera", DEM, 0.0)

    worker._release_all()

    assert shot.released == 1
    assert [r.released for r in engine.renderers] == [1]
    assert ctx.released == 1 and worker.ctx is None


def test_the_thread_ends_with_nothing_left(engine):
    """Whatever ends the loop - here: being told to stop."""
    from bambi_wildlife_detection.viewer3d.alfs_panel import AlfsWorker

    worker = AlfsWorker()

    def render(request):
        worker._renderer(request["size"], "camera", DEM, 0.0)
        worker._queue.put(None)     # ... and then told to stop
        return dict(request)

    worker._render = render
    worker.submit({"size": (640, 512)})
    worker.run()                    # one request, rendered ...
    assert engine.contexts[0].released == 1

    worker = AlfsWorker()
    worker._queue.put(None)
    worker.run()
    assert [ctx.released for ctx in engine.contexts] == [1, 1]
    assert all(r.released == 1 for r in engine.renderers)


# ---------------------------------------------------------------------------
# 3D Flight Viewer: search for correction factors
# ---------------------------------------------------------------------------

def test_a_search_gives_its_context_and_renderers_back(engine):
    from bambi_wildlife_detection.viewer3d.corrections_panel import (
        RefineWorker)

    worker = RefineWorker({})

    def search(ctx, renderers):
        renderers.extend([Thing(), Thing()])
        worker.kept = list(renderers)
        return {"score": 1.0}

    worker._search_on = search
    assert worker._search() == {"score": 1.0}

    assert [r.released for r in worker.kept] == [1, 1]
    assert [ctx.released for ctx in engine.contexts] == [1]


def test_a_search_that_fails_gives_them_back_too(engine):
    from bambi_wildlife_detection.viewer3d.corrections_panel import (
        RefineWorker)

    worker = RefineWorker({})

    def search(ctx, renderers):
        renderers.append(Thing())
        worker.kept = list(renderers)
        raise RuntimeError("a frame could not be read")

    worker._search_on = search
    with pytest.raises(RuntimeError):
        worker._search()

    assert worker.kept[0].released == 1
    assert [ctx.released for ctx in engine.contexts] == [1]


# ---------------------------------------------------------------------------
# Correction Wizard: light-field preview
# ---------------------------------------------------------------------------

@pytest.fixture
def preview(engine, monkeypatch):
    render = types.ModuleType("alfspy.render.render")
    render.read_gltf = lambda path: ("mesh", "texture")
    render.process_render_data = lambda mesh, texture: (mesh, texture)
    geo = types.ModuleType("alfspy.core.util.geo")
    geo.get_aabb = lambda vertices: "bounds"
    for name, module in {
            "alfspy.render": types.ModuleType("alfspy.render"),
            "alfspy.render.render": render,
            "alfspy.core.util": types.ModuleType("alfspy.core.util"),
            "alfspy.core.util.geo": geo}.items():
        monkeypatch.setitem(sys.modules, name, module)

    from bambi_wildlife_detection.bambi_correction_wizard import (
        _LightFieldRenderWorker)

    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    return _LightFieldRenderWorker({
        "correction": {"translation": dict(zero), "rotation": dict(zero)},
        "mesh_data": types.SimpleNamespace(vertices=[]), "texture_data": None,
        "frames": []})


def test_a_preview_gives_back_what_it_made(preview, engine):
    kept = {}

    def render(ctx, shots, made, *rest):
        shots.extend([Thing(), Thing()])
        made["renderer"] = Thing()
        kept.update(shots=list(shots), renderer=made["renderer"])
        return {"image": "picture"}

    preview._render_on = render
    assert preview._render() == {"image": "picture"}

    assert kept["renderer"].released == 1
    assert [shot.released for shot in kept["shots"]] == [1, 1]
    assert [ctx.released for ctx in engine.contexts] == [1]


def test_a_preview_that_fails_gives_it_back_too(preview, engine):
    """No frame found, a driver error: once per try, it all used to stay."""
    kept = []

    def render(ctx, shots, made, *rest):
        shots.append(Thing())
        kept.extend(shots)
        raise RuntimeError("No valid shots could be loaded.")

    preview._render_on = render
    with pytest.raises(RuntimeError):
        preview._render()

    assert kept[0].released == 1
    assert [ctx.released for ctx in engine.contexts] == [1]
