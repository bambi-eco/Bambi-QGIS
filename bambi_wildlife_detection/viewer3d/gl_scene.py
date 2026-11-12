# -*- coding: utf-8 -*-
"""GL resources and drawing of the 3D Flight Viewer (OpenGL 3.3 core).

Everything here must run with the viewport's GL context current. Matrices
are column-vector NumPy arrays (see :mod:`core.flight_viewer`).

Projective texturing
--------------------
A frame is draped onto the DEM by drawing the DEM again and sampling the
frame's texture at ``shot_matrix @ world`` - the same mapping alfspy's shot
program uses for ALFS, so a draped frame lands where ALFS puts it. An ALFS
integral or a raster base layer is draped the same way with an orthographic
matrix. Textures are uploaded bottom row first, so NDC ``y = -1`` samples the
image's last row, as in alfspy (``TextureData`` flips on upload).
"""

import ctypes
from typing import Dict, Optional

import numpy as np
from OpenGL import GL

_F32 = ctypes.sizeof(ctypes.c_float)

# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

DEM_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
uniform mat4 uMVP;
uniform float uZOffset;
out vec3 vWorld;
out vec3 vNormal;
out vec2 vUv;
void main() {
    vec3 p = aPos + vec3(0.0, 0.0, uZOffset);
    vWorld = p;
    vNormal = aNormal;
    vUv = aUv;
    gl_Position = uMVP * vec4(p, 1.0);
}
"""

DEM_BASE_FS = """
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec2 vUv;
uniform int uMode;          // 0 shaded relief, 1 mesh texture, 2 projected raster
uniform sampler2D uTex;
uniform mat4 uTexMatrix;
uniform vec3 uLight;
uniform vec2 uZRange;
uniform float uOpacity;
uniform int uWire;
out vec4 frag;
vec3 relief(float z) {
    float t = clamp((z - uZRange.x) / max(uZRange.y - uZRange.x, 1e-3), 0.0, 1.0);
    vec3 low = vec3(0.36, 0.50, 0.30);
    vec3 mid = vec3(0.62, 0.60, 0.42);
    vec3 high = vec3(0.86, 0.84, 0.80);
    return t < 0.5 ? mix(low, mid, t * 2.0) : mix(mid, high, t * 2.0 - 1.0);
}
void main() {
    if (uWire == 1) { frag = vec4(0.1, 0.1, 0.1, 0.6); return; }
    vec3 n = normalize(vNormal);
    float light = 0.35 + 0.65 * max(dot(n, normalize(uLight)), 0.0);
    vec3 base = relief(vWorld.z);
    if (uMode == 1) {
        base = texture(uTex, vUv).rgb;
        light = 0.7 + 0.3 * light;
    } else if (uMode == 2) {
        vec4 c = uTexMatrix * vec4(vWorld, 1.0);
        vec2 t = c.xy / c.w * 0.5 + 0.5;
        if (t.x >= 0.0 && t.x <= 1.0 && t.y >= 0.0 && t.y <= 1.0) {
            base = texture(uTex, t).rgb;
            light = 0.7 + 0.3 * light;
        }
    }
    frag = vec4(base * light, uOpacity);
}
"""

DEM_PROJECT_FS = """
#version 330 core
in vec3 vWorld;
uniform sampler2D uTex;
uniform mat4 uShot;
uniform float uAlpha;
uniform float uMinAlpha;
out vec4 frag;
void main() {
    vec4 c = uShot * vec4(vWorld, 1.0);
    if (c.w <= 0.0) discard;
    vec2 ndc = c.xy / c.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) discard;
    vec4 t = texture(uTex, ndc * 0.5 + 0.5);
    if (t.a <= uMinAlpha) discard;
    frag = vec4(t.rgb, t.a * uAlpha);
}
"""

LINE_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uMVP;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

# Lines wider than one pixel are not available in a core profile, so each
# segment is expanded into a screen-space quad. Segments are clipped at the
# near plane first, because a frustum's apex sits on the camera.
LINE_GS = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2 uViewport;
uniform float uWidth;
in vec4 vColor[];
out vec4 gColor;
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    float eps = 1e-4;
    if (p0.w < eps && p1.w < eps) return;
    vec4 c0 = vColor[0], c1 = vColor[1];
    if (p0.w < eps) { float t = (eps - p0.w) / (p1.w - p0.w); p0 = mix(p0, p1, t); c0 = mix(c0, c1, t); }
    if (p1.w < eps) { float t = (eps - p1.w) / (p0.w - p1.w); p1 = mix(p1, p0, t); c1 = mix(c1, c0, t); }
    vec2 s0 = p0.xy / p0.w * uViewport * 0.5;
    vec2 s1 = p1.xy / p1.w * uViewport * 0.5;
    vec2 d = s1 - s0;
    float len = length(d);
    vec2 n = len > 1e-6 ? vec2(-d.y, d.x) / len : vec2(0.0, 1.0);
    vec2 off = n * uWidth * 0.5 / (uViewport * 0.5);
    float bias = 2e-4;
    gColor = c0; gl_Position = vec4(p0.xy + off * p0.w, p0.z - bias * p0.w, p0.w); EmitVertex();
    gColor = c0; gl_Position = vec4(p0.xy - off * p0.w, p0.z - bias * p0.w, p0.w); EmitVertex();
    gColor = c1; gl_Position = vec4(p1.xy + off * p1.w, p1.z - bias * p1.w, p1.w); EmitVertex();
    gColor = c1; gl_Position = vec4(p1.xy - off * p1.w, p1.z - bias * p1.w, p1.w); EmitVertex();
    EndPrimitive();
}
"""

LINE_FS = """
#version 330 core
in vec4 gColor;
out vec4 frag;
void main() { frag = gColor; }
"""

POINT_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uMVP;
uniform float uSize;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_Position.z -= 2e-4 * gl_Position.w;
    gl_PointSize = uSize;
}
"""

POINT_FS = """
#version 330 core
in vec4 vColor;
uniform int uRound;
out vec4 frag;
void main() {
    if (uRound == 1 && length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
    frag = vColor;
}
"""

# Coloured / textured triangle meshes (the reconstruction surface).
SURF_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aColor;
uniform mat4 uMVP;
out vec3 vWorld;
out vec3 vNormal;
out vec4 vColor;
void main() {
    vWorld = aPos;
    vNormal = aNormal;
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

SURF_FS = """
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec4 vColor;
uniform vec3 uLight;
uniform float uShade;
uniform int uTextured;
uniform sampler2D uTex;
uniform mat4 uTexMatrix;
out vec4 frag;
void main() {
    float light = 0.4 + 0.6 * max(dot(normalize(vNormal), normalize(uLight)), 0.0);
    vec4 color = vColor;
    if (uTextured == 1) {
        vec4 c = uTexMatrix * vec4(vWorld, 1.0);
        color = vec4(texture(uTex, c.xy / c.w * 0.5 + 0.5).rgb, 1.0);
    }
    frag = vec4(color.rgb * mix(1.0, light, uShade), color.a);
}
"""


class GLError(RuntimeError):
    pass


def _compile(source: str, kind) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        raise GLError(f"shader compile failed: {log!r}")
    return shader


class Program:
    def __init__(self, vs: str, fs: str, gs: Optional[str] = None):
        shaders = [_compile(vs, GL.GL_VERTEX_SHADER),
                   _compile(fs, GL.GL_FRAGMENT_SHADER)]
        if gs:
            shaders.append(_compile(gs, GL.GL_GEOMETRY_SHADER))
        self.id = GL.glCreateProgram()
        for s in shaders:
            GL.glAttachShader(self.id, s)
        GL.glLinkProgram(self.id)
        if not GL.glGetProgramiv(self.id, GL.GL_LINK_STATUS):
            raise GLError(f"program link failed: {GL.glGetProgramInfoLog(self.id)!r}")
        for s in shaders:
            GL.glDeleteShader(s)
        self._loc: Dict[str, int] = {}

    def use(self):
        GL.glUseProgram(self.id)

    def loc(self, name: str) -> int:
        if name not in self._loc:
            self._loc[name] = GL.glGetUniformLocation(self.id, name)
        return self._loc[name]

    def mat4(self, name: str, m):
        # NumPy is row-major, GL expects column-major: transpose on upload.
        GL.glUniformMatrix4fv(self.loc(name), 1, GL.GL_TRUE,
                              np.ascontiguousarray(m, dtype=np.float32))

    def f(self, name: str, *v):
        fn = (GL.glUniform1f, GL.glUniform2f, GL.glUniform3f, GL.glUniform4f)[len(v) - 1]
        fn(self.loc(name), *[float(x) for x in v])

    def i(self, name: str, v: int):
        GL.glUniform1i(self.loc(name), int(v))

    def delete(self):
        GL.glDeleteProgram(self.id)


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------

class _VertexArray:
    """A VAO with float attributes and optional element indices."""

    def __init__(self, attributes, indices=None):
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self.buffers = []
        self.count = 0
        for location, data in enumerate(attributes):
            if data is None:
                continue
            data = np.ascontiguousarray(data, dtype=np.float32)
            vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data,
                            GL.GL_STATIC_DRAW)
            width = data.shape[1] if data.ndim > 1 else 1
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(location, width, GL.GL_FLOAT, GL.GL_FALSE,
                                     width * _F32, ctypes.c_void_p(0))
            self.buffers.append(vbo)
            if location == 0:
                self.count = len(data)
        self.index_count = 0
        if indices is not None:
            idx = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
            ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx,
                            GL.GL_STATIC_DRAW)
            self.buffers.append(ebo)
            self.index_count = len(idx)
        GL.glBindVertexArray(0)

    def draw(self, mode):
        GL.glBindVertexArray(self.vao)
        if self.index_count:
            GL.glDrawElements(mode, self.index_count, GL.GL_UNSIGNED_INT,
                              ctypes.c_void_p(0))
        elif self.count:
            GL.glDrawArrays(mode, 0, self.count)
        GL.glBindVertexArray(0)

    def delete(self):
        if self.buffers:
            GL.glDeleteBuffers(len(self.buffers), self.buffers)
        GL.glDeleteVertexArrays(1, [self.vao])
        self.buffers = []


class Texture:
    def __init__(self, image: np.ndarray):
        self.id = GL.glGenTextures(1)
        self.width = self.height = 0
        self.update(image)

    def update(self, image: np.ndarray):
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.shape[2] == 3:
            img = np.concatenate(
                [img, np.full(img.shape[:2] + (1,), 255, dtype=img.dtype)], axis=2)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img[::-1])          # bottom row first
        self.height, self.width = img.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width,
                        self.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        for wrap in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, wrap, GL.GL_CLAMP_TO_EDGE)

    def bind(self, unit: int = 0):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)

    def delete(self):
        GL.glDeleteTextures([self.id])


from .specs import Layer, LineSpec, PointSpec, SurfaceSpec  # noqa: E402,F401


class Canvas:
    """A persistent top-down texture the replay paints frames into - the
    "scanner" trail. Each frame is drawn into it once (projective texture
    of the DEM, seen orthographically from above); showing the whole trail
    then costs a single draped layer instead of one DEM pass per frame."""

    def __init__(self, width: int, height: int, matrix):
        self.width, self.height = int(width), int(height)
        self.matrix = np.asarray(matrix, dtype=np.float64)   # world -> clip
        self.id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width,
                        self.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        for wrap in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, wrap, GL.GL_CLAMP_TO_EDGE)
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self.id, 0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise GLError(f"trail canvas framebuffer incomplete ({status})")
        self.clear()

    def clear(self):
        """Clear the canvas and restore the framebuffer that was bound."""
        previous = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(previous))

    def bind(self, unit: int = 0):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)

    def delete(self):
        GL.glDeleteFramebuffers(1, [self.fbo])
        GL.glDeleteTextures([self.id])


class GLScene:
    """Owns all GL objects of one viewport. Create inside ``initializeGL``."""

    MAX_TEXTURES = 160

    def __init__(self):
        self.dem_program = Program(DEM_VS, DEM_BASE_FS)
        self.project_program = Program(DEM_VS, DEM_PROJECT_FS)
        self.line_program = Program(LINE_VS, LINE_FS, LINE_GS)
        self.point_program = Program(POINT_VS, POINT_FS)
        self.surface_program = Program(SURF_VS, SURF_FS)
        self.dem_vao: Optional[_VertexArray] = None
        self.dem_z_range = (0.0, 1.0)
        self.textures: Dict[object, Texture] = {}
        self.canvas: Optional[Canvas] = None
        self.surface_textures: Dict[str, Texture] = {}
        self._texture_order = []
        self.lines: Dict[str, tuple] = {}
        self.points: Dict[str, tuple] = {}
        self.surfaces: Dict[str, tuple] = {}

    # -- uploads -----------------------------------------------------------

    def set_dem(self, vertices, faces, normals, uvs=None):
        if self.dem_vao is not None:
            self.dem_vao.delete()
        uv = uvs if uvs is not None else np.zeros((len(vertices), 2), np.float32)
        self.dem_vao = _VertexArray([vertices, normals, uv], faces)
        z = np.asarray(vertices)[:, 2]
        self.dem_z_range = (float(z.min()), float(z.max())) if len(z) else (0, 1)

    def set_texture(self, key, image):
        tex = self.textures.get(key)
        if tex is None:
            self.textures[key] = Texture(image)
        else:
            tex.update(image)
            self._texture_order.remove(key)
        self._texture_order.append(key)
        # Only replay frames are evicted; the orthomosaic, ALFS integrals
        # and other one-off textures stay until they are dropped.
        frames = [k for k in self._texture_order
                  if isinstance(k, tuple) and k and k[0] == "frame"]
        while len(frames) > self.MAX_TEXTURES:
            old = frames.pop(0)
            self._texture_order.remove(old)
            self.textures.pop(old).delete()

    CANVAS_KEY = "__trail_canvas__"

    def set_canvas(self, width, height, matrix):
        if self.canvas is not None:
            self.canvas.delete()
        self.canvas = Canvas(width, height, matrix)

    def paint_into_canvas(self, texture_key, shot_matrix, default_fbo: int):
        """Drape one frame into the trail canvas (newest on top)."""
        tex = self.textures.get(texture_key)
        if self.canvas is None or tex is None or self.dem_vao is None:
            return False
        c = self.canvas
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, c.fbo)
        GL.glViewport(0, 0, c.width, c.height)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        # Colour: over-blend; alpha: keep the canvas opaque where painted.
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                               GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        p = self.project_program
        p.use()
        p.mat4("uMVP", c.matrix)
        p.mat4("uShot", shot_matrix)
        p.f("uAlpha", 1.0)
        p.f("uMinAlpha", 0.004)
        p.f("uZOffset", 0.0)
        p.i("uTex", 0)
        tex.bind(0)
        self.dem_vao.draw(GL.GL_TRIANGLES)
        GL.glUseProgram(0)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(default_fbo))
        return True

    def has_texture(self, key) -> bool:
        return key in self.textures

    def drop_texture(self, key):
        tex = self.textures.pop(key, None)
        if tex is not None:
            tex.delete()
            self._texture_order.remove(key)

    def set_lines(self, name: str, spec: Optional[LineSpec]):
        old = self.lines.pop(name, None)
        if old:
            old[0].delete()
        if spec is None or not len(spec.segments):
            return
        pos = np.asarray(spec.segments, np.float32).reshape(-1, 3)
        col = np.asarray(spec.colors, np.float32).reshape(-1, 4)
        self.lines[name] = (_VertexArray([pos, col]), spec)

    def set_points(self, name: str, spec: Optional[PointSpec]):
        old = self.points.pop(name, None)
        if old:
            old[0].delete()
        if spec is None or not len(spec.points):
            return
        self.points[name] = (_VertexArray([spec.points, spec.colors]), spec)

    def set_surface(self, name: str, spec: Optional[SurfaceSpec]):
        old = self.surfaces.pop(name, None)
        if old:
            old[0].delete()
        old_tex = self.surface_textures.pop(name, None)
        if old_tex is not None:
            old_tex.delete()
        if spec is None or not len(spec.faces):
            return
        vao = _VertexArray([spec.vertices, spec.normals, spec.colors], spec.faces)
        self.surfaces[name] = (vao, spec)
        if spec.texture is not None:
            # Kept apart from the frame textures, which are evicted as the
            # replay runs.
            self.surface_textures[name] = Texture(spec.texture)

    # -- drawing -----------------------------------------------------------

    def draw(self, mvp, viewport_size, *, dem_mode=0, dem_opacity=1.0,
             base_texture=None, base_matrix=None, dem_visible=True,
             wireframe=False, layers=(), visible=None,
             light=(0.4, -0.5, 0.8), background=(0.13, 0.15, 0.18),
             scale=1.0):
        """Draw one frame. *visible* filters line/point/surface names."""
        w, h = viewport_size
        GL.glViewport(0, 0, int(w), int(h))
        GL.glClearColor(*background, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_CULL_FACE)
        mvp = np.asarray(mvp, dtype=np.float64)
        show = (lambda name: True) if visible is None else (lambda name: name in visible)

        for name, (vao, spec) in self.surfaces.items():
            if not show(name):
                continue
            p = self.surface_program
            p.use()
            p.mat4("uMVP", mvp)
            p.f("uLight", *light)
            p.f("uShade", spec.shade)
            tex = self.surface_textures.get(name)
            if tex is not None and spec.texture_matrix is not None:
                tex.bind(0)
                p.i("uTex", 0)
                p.mat4("uTexMatrix", spec.texture_matrix)
                p.i("uTextured", 1)
            else:
                p.i("uTextured", 0)
            vao.draw(GL.GL_TRIANGLES)

        if self.dem_vao is not None and dem_visible:
            p = self.dem_program
            p.use()
            p.mat4("uMVP", mvp)
            p.f("uZOffset", 0.0)
            mode = dem_mode
            if mode in (1, 2):
                tex = self.textures.get(base_texture)
                if tex is None:
                    mode = 0
                else:
                    tex.bind(0)
                    p.i("uTex", 0)
                    p.mat4("uTexMatrix", base_matrix if base_matrix is not None
                           else np.eye(4))
            p.i("uMode", mode)
            p.f("uLight", *light)
            p.f("uZRange", *self.dem_z_range)
            p.f("uOpacity", dem_opacity)
            p.i("uWire", 0)
            self.dem_vao.draw(GL.GL_TRIANGLES)
            if wireframe:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
                p.i("uWire", 1)
                self.dem_vao.draw(GL.GL_TRIANGLES)
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

            # Draped layers: same geometry, so LEQUAL lets each one through;
            # later layers paint over earlier ones.
            pp = self.project_program
            pp.use()
            pp.mat4("uMVP", mvp)
            pp.i("uTex", 0)
            GL.glDepthMask(GL.GL_FALSE)
            for layer in layers:
                tex = (self.canvas if layer.texture_key == self.CANVAS_KEY
                       else self.textures.get(layer.texture_key))
                if tex is None:
                    continue
                tex.bind(0)
                pp.mat4("uShot", layer.matrix)
                pp.f("uAlpha", layer.alpha)
                pp.f("uMinAlpha", layer.min_alpha)
                pp.f("uZOffset", layer.z_offset)
                self.dem_vao.draw(GL.GL_TRIANGLES)
            GL.glDepthMask(GL.GL_TRUE)

        lp = self.line_program
        lp.use()
        lp.mat4("uMVP", mvp)
        lp.f("uViewport", float(w), float(h))
        for name, (vao, spec) in self.lines.items():
            if not show(name):
                continue
            if spec.on_top:
                GL.glDisable(GL.GL_DEPTH_TEST)
            lp.f("uWidth", spec.width * scale)
            vao.draw(GL.GL_LINES)
            if spec.on_top:
                GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        pt = self.point_program
        pt.use()
        pt.mat4("uMVP", mvp)
        for name, (vao, spec) in self.points.items():
            if not show(name):
                continue
            pt.f("uSize", spec.size * scale)
            pt.i("uRound", 1 if spec.round else 0)
            vao.draw(GL.GL_POINTS)
        GL.glUseProgram(0)

    def delete(self):
        for tex in self.textures.values():
            tex.delete()
        self.textures.clear()
        if self.canvas is not None:
            self.canvas.delete()
            self.canvas = None
        for tex in self.surface_textures.values():
            tex.delete()
        self.surface_textures.clear()
        for store in (self.lines, self.points, self.surfaces):
            for vao, _ in store.values():
                vao.delete()
            store.clear()
        if self.dem_vao is not None:
            self.dem_vao.delete()
        for p in (self.dem_program, self.project_program, self.line_program,
                  self.point_program, self.surface_program):
            p.delete()
