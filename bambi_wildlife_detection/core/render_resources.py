# -*- coding: utf-8 -*-
"""Giving back what a render holds on the graphics card.

A light-field render needs a *renderer*: a framebuffer of the size of the
image - a colour and a coverage texture and a depth buffer - plus the
terrain mesh's buffers. Nothing of that is freed when the Python object goes
away: OpenGL objects live until they are released, or until their context
is. So every place that makes a renderer has to release it, and two things
went wrong with that:

* the 3D Flight Viewer kept every renderer it had made, by image size, and
  only ever dropped them - on a change of the focus height, on closing - without
  releasing them. A light-field over the terrain has a different size for
  nearly every frame, so stepping through a flight made a new renderer per
  frame and kept them all;
* ``Renderer.release()`` of alfspy (up to 3.1 at least) frees the
  framebuffer and leaves what was attached to it: both textures and the
  depth buffer, the bulk of the memory - 24 bytes per pixel, 400 MB for a
  4096 px tile. The processing steps make a renderer per tile or per frame
  and released each of them, and still filled the card over a run.

:func:`release_renderer` frees all of it, :class:`RendererPool` keeps a few
renderers and releases the rest, :func:`release_shot` and
:func:`release_context` do the same for a frame's image and for a context.

Free of QGIS and Qt, and of alfspy: what is released is found by name on
the object, so an engine that holds nothing of the kind (PyTorch, Vulkan)
is simply left alone, and a later alfspy that frees these itself costs a
second, harmless ``release()``.
"""

from collections import OrderedDict
from typing import Callable, Tuple

#: What ``Renderer.release()`` leaves behind on the ModernGL engine: the
#: framebuffer's two colour attachments and its depth buffer.
_ATTACHMENTS = ("_color_tex", "_coverage_tex", "_depth_buf")


def _release(thing, log_fn=None, what: str = "") -> bool:
    """Call ``thing.release()`` if it has one. A failure is reported and
    does not stop whatever else is being released."""
    release = getattr(thing, "release", None)
    if not callable(release):
        return False
    try:
        release()
    except Exception as exc:  # noqa: BLE001 - one failure must not leak the rest
        if log_fn:
            log_fn(f"Could not release {what or type(thing).__name__}: {exc}")
        return False
    return True


def release_renderer(renderer, log_fn=None) -> None:
    """Free everything *renderer* holds on the graphics card.

    ``renderer.release()`` first - the mesh, the framebuffer, the shaders -
    then what it leaves: the framebuffer's attachments and the vertex array
    the shots are drawn with. Safe to call twice, and on ``None``.
    """
    if renderer is None:
        return
    _release(renderer, log_fn, "the renderer")
    shot_object = getattr(renderer, "_shot", None)
    # only its vertex array: the buffers are the mesh's, released above
    _release(getattr(shot_object, "vao", None), log_fn, "the shots' vertex array")
    for name in _ATTACHMENTS:
        _release(getattr(renderer, name, None), log_fn, name.strip("_"))


def release_shot(shot, log_fn=None) -> None:
    """Free a shot's image on the graphics card. Harmless on a shot that
    was released already, and on ``None``."""
    if shot is not None:
        _release(shot, log_fn, "a shot")


def release_context(ctx, log_fn=None) -> None:
    """End a render context; everything still alive on it goes with it."""
    if ctx is not None:
        _release(ctx, log_fn, "the render context")


class RendererPool:
    """The renderers of one render context, by image size.

    Making a renderer uploads the terrain mesh, so one that fits is used
    again. Only the *keep* most recently used are held - the others are
    released, not just forgotten.

    :param make: ``make(size) -> renderer`` for a ``(width, height)``
    :param keep: how many renderers to hold at most
    """

    def __init__(self, make: Callable[[Tuple[int, int]], object],
                 keep: int = 2, log_fn=None):
        self._make = make
        self._keep = max(1, int(keep))
        self._log = log_fn
        self._renderers: "OrderedDict[Tuple[int, int], object]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._renderers)

    def __contains__(self, size) -> bool:
        return self._key(size) in self._renderers

    @staticmethod
    def _key(size) -> Tuple[int, int]:
        return int(size[0]), int(size[1])

    def get(self, size):
        """The renderer for images of *size*, made if there is none."""
        key = self._key(size)
        renderer = self._renderers.pop(key, None)
        if renderer is None:
            # room first: the new framebuffer should not have to fit next to
            # the one it replaces
            while len(self._renderers) >= self._keep:
                _size, oldest = self._renderers.popitem(last=False)
                release_renderer(oldest, self._log)
            renderer = self._make(key)
        self._renderers[key] = renderer          # most recently used
        return renderer

    def clear(self) -> None:
        """Release every renderer - the mesh they were made for changed, or
        the work is over."""
        while self._renderers:
            _size, renderer = self._renderers.popitem()
            release_renderer(renderer, self._log)
