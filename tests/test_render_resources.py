# -*- coding: utf-8 -*-
"""What a light-field render holds on the graphics card is given back.

OpenGL objects live until they are released, or their context is: nothing is
freed when the Python object goes away. Two things went wrong with that.

* The 3D Flight Viewer kept every renderer it had made, by image size, and
  dropped them - never released them. The light-field over the terrain has
  another size for nearly every frame.
* ``Renderer.release()`` of alfspy leaves the framebuffer's textures and its
  depth buffer behind, the bulk of a renderer's memory. The processing steps
  make a renderer per tile, per sample or per frame.

The first half of this file needs nothing but the module. The second runs
the real thing on a software OpenGL context and counts what is alive on it;
it is skipped where no such context can be made.
"""
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import render_resources as rr

from .software_gl import N_FRAMES, Alive, ended, small_flight, software_gl


# ---------------------------------------------------------------------------
# With stand-ins
# ---------------------------------------------------------------------------

class Thing:
    """Anything with a ``release()``; counts how often it was called."""

    def __init__(self, fails=False):
        self.released = 0
        self.fails = fails

    def release(self):
        self.released += 1
        if self.fails:
            raise RuntimeError("driver said no")


class FakeRenderer(Thing):
    """Holds what alfspy's ModernGL renderer holds - and, like it, releases
    only a part of that itself."""

    def __init__(self, size=(0, 0), fails=False):
        super().__init__(fails)
        self.size = size
        self._color_tex, self._coverage_tex = Thing(), Thing()
        self._depth_buf = Thing()
        self._shot = type("RenderObject", (), {"vao": Thing()})()

    def left_behind(self):
        return [self._color_tex, self._coverage_tex, self._depth_buf,
                self._shot.vao]


class TestReleaseRenderer:
    def test_it_releases_what_the_renderer_leaves_behind(self):
        renderer = FakeRenderer()
        rr.release_renderer(renderer)

        assert renderer.released == 1
        assert [thing.released for thing in renderer.left_behind()] == [1] * 4

    def test_a_renderer_that_fails_to_release_still_gives_the_rest_back(self):
        renderer, said = FakeRenderer(fails=True), []
        rr.release_renderer(renderer, said.append)

        assert [thing.released for thing in renderer.left_behind()] == [1] * 4
        assert said == ["Could not release the renderer: driver said no"]

    def test_one_part_that_fails_does_not_keep_the_others(self):
        renderer = FakeRenderer()
        renderer._color_tex.fails = True
        rr.release_renderer(renderer)                # does not raise

        assert renderer._coverage_tex.released == 1
        assert renderer._depth_buf.released == 1

    def test_an_engine_that_holds_nothing_of_the_kind_is_left_alone(self):
        """PyTorch and Vulkan renderers have no such attachments."""
        plain = Thing()
        rr.release_renderer(plain)
        assert plain.released == 1

    def test_nothing_to_release(self):
        rr.release_renderer(None)
        rr.release_shot(None)
        rr.release_context(None)
        rr.release_renderer(object())                # no release() at all

    def test_shots_and_contexts(self):
        shot, ctx, said = Thing(), Thing(fails=True), []
        rr.release_shot(shot)
        rr.release_context(ctx, said.append)

        assert shot.released == 1 and ctx.released == 1
        assert said == ["Could not release the render context: driver said no"]


class TestRendererPool:
    @pytest.fixture
    def made(self):
        return []

    @pytest.fixture
    def pool(self, made):
        def make(size):
            made.append(FakeRenderer(size))
            return made[-1]
        return rr.RendererPool(make, keep=2)

    def test_a_renderer_that_fits_is_used_again(self, pool, made):
        first = pool.get((640, 512))
        assert pool.get((640, 512)) is first
        assert pool.get([640.0, 512.0]) is first      # whatever names the size
        assert len(made) == 1 and (640, 512) in pool

    def test_only_a_few_are_kept_and_the_others_released(self, pool, made):
        """What the viewer did: one per size, all of them kept."""
        for width in range(600, 640):
            pool.get((width, 512))

        assert len(pool) == 2
        assert [r.released for r in made] == [1] * 38 + [0, 0]
        assert all(thing.released == 1
                   for r in made[:38] for thing in r.left_behind())

    def test_the_one_not_used_for_longest_goes_first(self, pool, made):
        a = pool.get((1, 1))
        b = pool.get((2, 2))
        pool.get((1, 1))                # a was used after b
        pool.get((3, 3))

        assert b.released == 1 and a.released == 0
        assert (1, 1) in pool and (2, 2) not in pool

    def test_room_is_made_before_the_new_one(self, made):
        """The new framebuffer does not have to fit next to the old one."""
        order = []

        class Noted(FakeRenderer):
            def release(self):
                order.append(("released", self.size))

        def make(size):
            order.append(("made", size))
            return Noted(size)

        pool = rr.RendererPool(make, keep=1)
        pool.get((1, 1))
        pool.get((2, 2))
        assert order == [("made", (1, 1)), ("released", (1, 1)),
                         ("made", (2, 2))]

    def test_clear_releases_all_of_them(self, pool, made):
        pool.get((1, 1))
        pool.get((2, 2))
        pool.clear()

        assert len(pool) == 0
        assert [r.released for r in made] == [1, 1]
        pool.clear()                                   # and again: nothing
        assert [r.released for r in made] == [1, 1]

    def test_a_renderer_that_could_not_be_made_takes_no_place(self, made):
        def make(size):
            if size == (2, 2):
                raise RuntimeError("out of memory")
            made.append(FakeRenderer(size))
            return made[-1]

        pool = rr.RendererPool(make, keep=2)
        pool.get((1, 1))
        with pytest.raises(RuntimeError):
            pool.get((2, 2))
        assert len(pool) == 1 and (2, 2) not in pool


# ---------------------------------------------------------------------------
# On a real OpenGL context
# ---------------------------------------------------------------------------

@pytest.fixture
def gl():
    ctx, library = software_gl()
    yield ctx, Alive(library)
    rr.release_context(ctx)


def _ground():
    from alfspy.core.rendering import MeshData

    return MeshData(
        np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], np.float32),
        np.array([[0, 1, 2], [0, 2, 3]], np.uint32))


class TestOnOpenGL:
    def test_the_renderers_own_release_leaves_its_framebuffer_behind(self, gl):
        """Why :func:`release_renderer` exists. When this fails, alfspy has
        learned to do it itself - and the helper costs nothing."""
        from alfspy.core.rendering import Camera, Renderer, Resolution

        ctx, alive = gl
        renderer = Renderer(Resolution(256, 256), ctx, Camera(), _ground())
        renderer.release()

        left = alive()
        assert left["textures"] == 2 and left["renderbuffers"] == 1
        rr.release_renderer(renderer)                 # tidy up after the test
        assert alive() == alive.none()

    def test_release_renderer_leaves_nothing(self, gl):
        from alfspy.core.rendering import Camera, Renderer, Resolution

        ctx, alive = gl
        for _ in range(5):
            renderer = Renderer(Resolution(256, 256), ctx, Camera(), _ground())
            renderer.render_background()
            rr.release_renderer(renderer)
            rr.release_renderer(renderer)             # twice does no harm

        assert alive() == alive.none()

    def test_a_pool_holds_what_it_keeps_and_nothing_else(self, gl):
        from alfspy.core.rendering import Camera, Renderer, Resolution

        ctx, alive = gl
        mesh = _ground()
        pool = rr.RendererPool(
            lambda size: Renderer(Resolution(*size), ctx, Camera(), mesh),
            keep=2)
        one = None
        for width in range(200, 230):
            image = pool.get((width, 128)).render_background()
            assert image.shape == (128, width, 4)     # they still render
            one = one or alive()

        assert len(pool) == 2
        assert alive() == {kind: 2 * count for kind, count in one.items()}
        pool.clear()
        assert alive() == alive.none()


# ---------------------------------------------------------------------------
# The processing steps, on a small flight over flat ground
# ---------------------------------------------------------------------------

@pytest.fixture
def flight(tmp_path, monkeypatch):
    return small_flight(tmp_path, monkeypatch)


class TestTheStepsGiveBackWhatTheyTake:
    def test_light_field_per_sample(self, flight):
        """One renderer per sample: its textures do not stay for the next."""
        flight["processor"].run_alfs(dict(
            flight["config"], alfs_sampling_mode=True, alfs_sampling_rate=1,
            alfs_sampling_range=1))

        seen = flight["textures"]
        assert len(seen) == N_FRAMES
        assert seen == [0] * N_FRAMES, "textures pile up from sample to sample"
        assert [ended(ctx) for ctx in flight["contexts"]] == [True]
        written = os.listdir(os.path.join(flight["root"], "alfs_t"))
        assert len([name for name in written if name.endswith(".tif")]) \
            == N_FRAMES

    def test_light_field_in_tiles(self, flight):
        flight["processor"].run_alfs(dict(
            flight["config"], alfs_max_tile_size=64,
            alfs_ground_resolution=0.25))

        seen = flight["textures"]
        assert len(seen) > 4
        assert seen == [0] * len(seen), "textures pile up from tile to tile"
        assert [ended(ctx) for ctx in flight["contexts"]] == [True]

    def test_a_geotiff_per_frame(self, flight):
        """One renderer per frame, next to the frame's own image."""
        flight["processor"].run_export_geotiffs(dict(flight["config"]))

        seen = flight["textures"]
        assert len(seen) == N_FRAMES
        assert seen == [1] * N_FRAMES, "textures pile up from frame to frame"
        assert [ended(ctx) for ctx in flight["contexts"]] == [True]
        written = os.listdir(os.path.join(flight["root"], "geotiffs_t"))
        assert len([name for name in written if name.endswith(".tiff")]) \
            == N_FRAMES

    def test_a_light_field_that_fails_gives_its_context_back(
            self, flight, monkeypatch):
        """It used to be released on the way out of a run that got to its
        end only."""
        from alfspy.core.backends.moderngl_.renderer import Renderer

        counting = Renderer.__init__

        def failing_on_the_third(self, *args, **kwargs):
            if len(flight["textures"]) == 2:
                raise RuntimeError("the driver gave up")
            counting(self, *args, **kwargs)

        monkeypatch.setattr(Renderer, "__init__", failing_on_the_third)
        with pytest.raises(RuntimeError, match="the driver gave up"):
            flight["processor"].run_alfs(dict(
                flight["config"], alfs_sampling_mode=True,
                alfs_sampling_rate=1, alfs_sampling_range=1))

        assert [ended(ctx) for ctx in flight["contexts"]] == [True]

    def test_a_cancelled_light_field_gives_its_context_back(self, flight):
        calls = {"n": 0}

        def cancel_after_two():
            calls["n"] += 1
            return calls["n"] > 2

        flight["processor"].run_alfs(dict(
            flight["config"], alfs_sampling_mode=True, alfs_sampling_rate=1,
            alfs_sampling_range=1), cancel_check=cancel_after_two)

        assert [ended(ctx) for ctx in flight["contexts"]] == [True]
