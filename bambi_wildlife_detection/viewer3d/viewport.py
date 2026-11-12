# -*- coding: utf-8 -*-
"""The 3D Flight Viewer's OpenGL widget.

Holds the scene description (DEM, draped layers, lines, points, surfaces) as
plain data and uploads it to the GPU the next time it paints, so callers never
need a current GL context. Navigation: left drag orbits, right / middle drag
(or Shift + left) pans, the wheel zooms towards the point under the cursor,
double click picks a DEM point. In *follow* mode the view is the drone
camera's own; dragging leaves follow mode where the drone camera was.
"""

import math
import time
from typing import Callable, Dict, Optional

import numpy as np

from ..core.flight_viewer import FrameCamera
from ..core.view_camera import OrbitCamera
from .qt import QOpenGLWidget, QtCore, QtGui, Signal

_Qt = QtCore.Qt


def _btn(name):
    return getattr(_Qt.MouseButton, name, None) or getattr(_Qt, name)


LEFT, RIGHT, MIDDLE = _btn("LeftButton"), _btn("RightButton"), _btn("MiddleButton")
SHIFT = getattr(getattr(_Qt, "KeyboardModifier", _Qt), "ShiftModifier")


def surface_format():
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    return fmt


def _pos(event):
    p = event.position() if hasattr(event, "position") else event.pos()
    return float(p.x()), float(p.y())


class FlightViewport(QOpenGLWidget):
    #: the user moved the view (used to leave follow mode)
    navigated = Signal()
    #: world point under the cursor (np.ndarray) or None
    hovered = Signal(object)
    #: double-clicked world point
    picked = Signal(object)
    #: GL could not start; the message says why
    failed = Signal(str)
    #: left click without dragging, widget pixels (x, y)
    clicked = Signal(float, float)
    #: mouse moved without a button, widget pixels (throttled)
    moved = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(surface_format())
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(_Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self.orbit = OrbitCamera()
        self.follow: Optional[FrameCamera] = None
        self.scene_radius = 1000.0
        self._gl = None
        self._gl_failed = ""
        self._pending: Dict[str, object] = {}
        self._pending_textures: Dict[object, np.ndarray] = {}
        self._dropped_textures = set()
        self._ready_textures = set()
        self._canvas_spec = None         # (width, height, matrix) to create
        self._canvas_clear = False
        self._canvas_paint = []          # [(texture_key, shot_matrix)]
        self.canvas_painted = 0
        self.layers = []
        self.dem_mode = 0
        self.dem_opacity = 1.0
        self.dem_visible = True
        self.wireframe = False
        self.base_texture = None
        self.base_matrix = None
        self.hidden = set()
        self._names = set()
        self._drag = None
        self._pick: Optional[Callable] = None
        self._last_hover = 0.0
        self.gl_info = ""

    # -- scene API (any thread-free caller, no GL context needed) ---------

    def set_pick_function(self, fn: Optional[Callable]):
        """``fn(origin, direction) -> point | None`` used for hover/picking."""
        self._pick = fn

    def set_dem(self, dem):
        self._pending["dem"] = dem
        lo, hi = dem.bounds
        self.scene_radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1000.0
        self.orbit.max_distance = self.scene_radius * 20.0
        self.update()

    def set_texture(self, key, image: np.ndarray):
        self._pending_textures[key] = image
        self._dropped_textures.discard(key)
        self.update()

    def has_texture(self, key) -> bool:
        return key in self._ready_textures or key in self._pending_textures

    def drop_texture(self, key):
        self._pending_textures.pop(key, None)
        self._dropped_textures.add(key)
        self._ready_textures.discard(key)

    #: texture key of the trail canvas, for :class:`Layer`
    CANVAS_KEY = "__trail_canvas__"

    def set_canvas(self, width: int, height: int, matrix):
        """(Re)create the trail canvas covering the extent *matrix* maps."""
        self._canvas_spec = (int(width), int(height), matrix)
        self._canvas_paint = []
        self.canvas_painted = 0
        self.update()

    def clear_canvas(self):
        self._canvas_clear = True
        self._canvas_paint = []
        self.canvas_painted = 0
        self.update()

    def paint_canvas(self, texture_key, shot_matrix):
        """Paint a (ready or pending) frame texture into the trail canvas
        at the next repaint, in the order of the calls."""
        self._canvas_paint.append((texture_key, shot_matrix))
        self.update()

    def set_layers(self, layers):
        self.layers = list(layers)
        self.update()

    def set_lines(self, name, spec):
        self._pending[f"line:{name}"] = spec
        self._names.add(name)
        self.update()

    def set_points(self, name, spec):
        self._pending[f"point:{name}"] = spec
        self._names.add(name)
        self.update()

    def set_surface(self, name, spec):
        self._pending[f"surface:{name}"] = spec
        self._names.add(name)
        self.update()

    def set_visible(self, name: str, visible: bool):
        (self.hidden.discard if visible else self.hidden.add)(name)
        self.update()

    def set_follow(self, camera: Optional[FrameCamera]):
        self.follow = camera
        self.update()

    def leave_follow(self, distance: Optional[float] = None):
        """Switch to orbiting, keeping the current drone view."""
        if self.follow is not None:
            self.orbit.adopt(self.follow, distance or self._follow_distance())
            self.follow = None
            self.update()

    def _follow_distance(self) -> float:
        if self._pick is not None and self.follow is not None:
            hit = self._pick(self.follow.position, self.follow.forward)
            if hit is not None:
                return float(np.linalg.norm(np.asarray(hit) - self.follow.position))
        return 100.0

    def fit(self, lower, upper):
        self.orbit.fit(lower, upper)
        self.update()

    # -- matrices ------------------------------------------------------------

    def aspect(self) -> float:
        return self.width() / max(self.height(), 1)

    def view_projection(self):
        """(view, proj) of what is on screen right now."""
        if self.follow is not None:
            cam = self.follow.with_aspect(self.aspect(), near=0.5,
                                          far=self.scene_radius * 10 + 5000)
            return cam.view, cam.proj
        return self.orbit.view(), self.orbit.proj(self.aspect(), self.scene_radius)

    def screen_camera(self) -> FrameCamera:
        """The on-screen view as a camera (for ALFS *View* mode)."""
        view, proj = self.view_projection()
        fovy = self.follow.fovy if self.follow is not None else self.orbit.fovy
        return FrameCamera(view, proj, fovy, self.aspect())

    def ray_at(self, x: float, y: float):
        if self.follow is not None:
            from ..core.flight_viewer import pixel_rays
            cam = self.follow.with_aspect(self.aspect())
            return cam.position, pixel_rays(cam, self.width(), self.height(),
                                            [[x, y]])[0]
        return self.orbit.ray(x, y, self.width(), self.height())

    def pick(self, x: float, y: float):
        if self._pick is None:
            return None
        origin, direction = self.ray_at(x, y)
        return self._pick(origin, direction)

    # -- GL ------------------------------------------------------------------

    def initializeGL(self):
        try:
            from OpenGL import GL
            from .gl_scene import GLScene
            fmt = self.context().format()
            version = (fmt.majorVersion(), fmt.minorVersion())
            self.gl_info = "{} - {}".format(
                GL.glGetString(GL.GL_VERSION).decode(errors="replace"),
                GL.glGetString(GL.GL_RENDERER).decode(errors="replace"))
            if version < (3, 3):
                raise RuntimeError(
                    f"OpenGL {version[0]}.{version[1]} - the 3D view needs 3.3.")
            self._gl = GLScene()
        except Exception as exc:   # noqa: BLE001 - reported to the dialog
            self._gl = None
            self._gl_failed = str(exc)
            QtCore.QTimer.singleShot(0, lambda: self.failed.emit(self._gl_failed))

    def _flush(self):
        gl = self._gl
        for key in list(self._dropped_textures):
            gl.drop_texture(key)
        self._dropped_textures.clear()
        for key, image in list(self._pending_textures.items()):
            gl.set_texture(key, image)
            self._ready_textures.add(key)
        self._pending_textures.clear()
        # The GL scene evicts old textures; mirror what it still holds.
        self._ready_textures &= set(gl.textures)
        if self._canvas_spec is not None:
            w, h, m = self._canvas_spec
            self._canvas_spec = None
            gl.set_canvas(w, h, m)
        elif self._canvas_clear and gl.canvas is not None:
            gl.canvas.clear()
        self._canvas_clear = False
        # Creating or clearing the trail canvas binds its framebuffer; the
        # scene must be drawn into the widget's own one again, or the next
        # frame renders the whole view into the trail texture.
        from OpenGL import GL
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        for key, value in list(self._pending.items()):
            kind, _, name = key.partition(":")
            if kind == "dem":
                gl.set_dem(value.vertices, value.faces, value.normals, value.uvs)
                if value.texture is not None:
                    gl.set_texture("__dem_texture__", value.texture)
            elif kind == "line":
                gl.set_lines(name, value)
            elif kind == "point":
                gl.set_points(name, value)
            elif kind == "surface":
                gl.set_surface(name, value)
        self._pending.clear()

    def paintGL(self):
        if self._gl is None:
            return
        self._flush()
        if self._canvas_paint and self._gl.canvas is not None:
            fbo = self.defaultFramebufferObject()
            for key, matrix in self._canvas_paint:
                if self._gl.paint_into_canvas(key, matrix, fbo):
                    self.canvas_painted += 1
            self._canvas_paint = []
        view, proj = self.view_projection()
        ratio = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        size = (self.width() * ratio, self.height() * ratio)
        base = self.base_texture
        if self.dem_mode == 1:
            base = "__dem_texture__"
        visible = self._names - self.hidden
        drawable = self._ready_textures | {self.CANVAS_KEY}
        self._gl.draw(proj @ view, size, dem_mode=self.dem_mode,
                      dem_opacity=self.dem_opacity, base_texture=base,
                      base_matrix=self.base_matrix,
                      dem_visible=self.dem_visible and "dem" not in self.hidden,
                      wireframe=self.wireframe,
                      layers=[layer for layer in self.layers
                              if layer.texture_key in drawable],
                      visible=visible, scale=ratio)

    def dispose(self):
        if self._gl is not None:
            self.makeCurrent()
            try:
                self._gl.delete()
            finally:
                self._gl = None
                self.doneCurrent()

    # -- interaction ----------------------------------------------------------

    def mousePressEvent(self, event):
        self._drag = (_pos(event), event.button(), event.modifiers())
        self._press = _pos(event)

    def mouseReleaseEvent(self, event):
        self._drag = None
        press = getattr(self, "_press", None)
        self._press = None
        if press is not None and event.button() == LEFT:
            x, y = _pos(event)
            if abs(x - press[0]) + abs(y - press[1]) < 4:
                self.clicked.emit(x, y)

    def mouseMoveEvent(self, event):
        x, y = _pos(event)
        if self._drag is None:
            now = time.monotonic()
            if now - self._last_hover > 0.05:
                self._last_hover = now
                self.hovered.emit(self.pick(x, y))
                self.moved.emit(x, y)
            return
        (x0, y0), button, mods = self._drag
        dx, dy = x - x0, y - y0
        if abs(dx) + abs(dy) < 1:
            return
        press = getattr(self, "_press", None)
        if press is not None and abs(x - press[0]) + abs(y - press[1]) < 4:
            return                       # still a click, not a drag yet
        if self.follow is not None:
            self.leave_follow()
            self.navigated.emit()
        if button == LEFT and not (mods & SHIFT):
            self.orbit.orbit(-dx * 0.3, dy * 0.3)
        else:
            self.orbit.pan(dx, dy, self.height())
        self._drag = ((x, y), button, mods)
        self.navigated.emit()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        if not delta:
            return
        if self.follow is not None:
            self.leave_follow()
        x, y = _pos(event)
        towards = self.pick(x, y)
        self.orbit.zoom(math.pow(0.85, delta), towards)
        self.navigated.emit()
        self.update()

    def mouseDoubleClickEvent(self, event):
        x, y = _pos(event)
        hit = self.pick(x, y)
        if hit is not None:
            self.picked.emit(np.asarray(hit))
