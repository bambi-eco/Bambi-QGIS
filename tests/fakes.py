# -*- coding: utf-8 -*-
"""Shared fakes and helpers for the unit test suite.

Consolidates the stand-ins that individual test modules would otherwise
duplicate:

* :class:`SignalRecorder` — replaces a stubbed ``pyqtSignal`` on a worker
  instance so emissions can be asserted.
* :func:`make_module` — build an injectable fake module (with ``__spec__``
  set, so ``importlib.util.find_spec`` works on it).
* :class:`FakeResponse` / :class:`FakeSession` — ``requests`` stand-ins for
  download code.
* :func:`install_fake_render_stack` — fake ``pyrr`` + ``alfspy`` modules
  with an identity-matrix camera, for testing world→pixel projection code
  with hand-computable expectations.
* :class:`FakeRasterioDataset` / :func:`install_fake_rasterio` — a minimal
  in-memory ``rasterio`` for code paths that only read one band + metadata.
* :func:`parse_glb` — parse and structurally validate a GLB container.
"""
import importlib.machinery
import json
import struct
import sys
import types

import numpy as np


class SignalRecorder:
    """Stands in for a stubbed pyqtSignal so emissions can be asserted.

    Assign to a worker *instance* attribute (``worker.finished =
    SignalRecorder()``); the stubbed class-level signal is shadowed and all
    ``emit`` calls land in ``self.calls``.
    """

    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)

    def connect(self, *args, **kwargs):
        pass

    def disconnect(self, *args, **kwargs):
        pass


def make_module(name, **attrs):
    """Create a fake module with ``__spec__`` set.

    The spec matters: ``importlib.util.find_spec`` raises ``ValueError`` for
    a module found in ``sys.modules`` whose ``__spec__`` is None, which
    breaks installed-package probes (e.g. ``TrackerManager``).
    """
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


# ---------------------------------------------------------------------------
# requests stand-ins
# ---------------------------------------------------------------------------

class FakeResponse:
    """Streaming ``requests`` response with queued chunks."""

    def __init__(self, status_code=200, chunks=None, content_length=None):
        self.status_code = status_code
        self._chunks = chunks or []
        total = content_length
        if total is None:
            total = sum(len(c) for c in self._chunks)
        self.headers = {"content-length": str(total)} if total else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=None):
        for chunk in self._chunks:
            yield chunk


class FakeSession:
    """Returns queued responses (or raises queued exceptions) per get() call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requested_urls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.requested_urls.append(url)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ---------------------------------------------------------------------------
# pyrr / alfspy render stack
# ---------------------------------------------------------------------------

class IdentityCamera:
    """``alfspy.core.rendering.Camera`` stand-in with identity view/projection.

    With identity matrices the NDC mapping reduces to
    ``px = (x + 1) * w / 2`` and ``py = h - (y + 1) * h / 2``, so expected
    pixel coordinates are hand-computable from world coordinates.
    """

    def __init__(self, fovy=50, aspect_ratio=1.0, position=None, rotation=None):
        self.fovy = fovy
        self.aspect_ratio = aspect_ratio
        self.position = position
        self.rotation = rotation

    def get_view(self):
        return np.eye(4)

    def get_proj(self):
        return np.eye(4)


def install_fake_render_stack(monkeypatch, camera_cls=IdentityCamera):
    """Install fake ``pyrr`` and ``alfspy`` modules into sys.modules."""
    pyrr = make_module("pyrr")
    pyrr.Vector3 = lambda values, dtype=None: np.asarray(values, dtype=np.float64)

    class Quaternion:
        @staticmethod
        def from_eulers(eulers):
            return tuple(eulers)

    pyrr.Quaternion = Quaternion

    alfspy = make_module("alfspy")
    alfspy.__path__ = []
    core = make_module("alfspy.core")
    core.__path__ = []
    rendering = make_module("alfspy.core.rendering")
    rendering.Camera = camera_cls
    alfspy.core = core
    core.rendering = rendering

    monkeypatch.setitem(sys.modules, "pyrr", pyrr)
    monkeypatch.setitem(sys.modules, "alfspy", alfspy)
    monkeypatch.setitem(sys.modules, "alfspy.core", core)
    monkeypatch.setitem(sys.modules, "alfspy.core.rendering", rendering)
    return alfspy


# ---------------------------------------------------------------------------
# rasterio stand-ins (for read-only single-band access)
# ---------------------------------------------------------------------------

class FakeAffine:
    def __init__(self, a, b, c, d, e, f):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f


class FakeBounds:
    def __init__(self, left, bottom, right, top):
        self.left, self.bottom, self.right, self.top = left, bottom, right, top


class FakeCrs:
    def __init__(self, epsg):
        self._epsg = epsg

    def to_epsg(self):
        return self._epsg

    def __str__(self):
        return f"EPSG:{self._epsg}"


class FakeRasterioDataset:
    def __init__(self, elevation, transform, bounds, epsg, nodata=None):
        self._elevation = elevation
        self.transform = transform
        self.bounds = bounds
        self.crs = FakeCrs(epsg)
        self.nodata = nodata
        self.height, self.width = elevation.shape

    def read(self, band):
        return self._elevation

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install_fake_rasterio(monkeypatch, dataset):
    fake = make_module("rasterio")
    fake.open = lambda path: dataset
    monkeypatch.setitem(sys.modules, "rasterio", fake)
    return fake


# ---------------------------------------------------------------------------
# GLB container parsing
# ---------------------------------------------------------------------------

def parse_glb(path):
    """Parse a GLB container, returning (gltf_json, binary_chunk).

    Asserts the structural invariants of the GLB 2.0 format on the way.
    """
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    version, total_length = struct.unpack_from("<II", data, 4)
    assert version == 2
    assert total_length == len(data)
    json_length = struct.unpack_from("<I", data, 12)[0]
    assert data[16:20] == b"JSON"
    gltf = json.loads(data[20:20 + json_length].decode("utf-8"))
    bin_offset = 20 + json_length
    bin_length = struct.unpack_from("<I", data, bin_offset)[0]
    assert data[bin_offset + 4:bin_offset + 8] == b"BIN\x00"
    binary = data[bin_offset + 8:bin_offset + 8 + bin_length]
    return gltf, binary
