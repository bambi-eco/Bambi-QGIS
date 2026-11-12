# -*- coding: utf-8 -*-
"""Background loading of frame images for the replay.

Frames are read with ``QImage`` (safe to use off the GUI thread), shrunk to
``max_edge`` and handed back as RGBA ``uint8`` arrays through the ``loaded``
signal, which Qt delivers on the GUI thread. A small LRU keeps the arrays of
recently used frames so stepping back and forth does not re-read the disk.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from .qt import QtCore, QtGui, Signal


def read_rgba(path: str, max_edge: int = 1024) -> Optional[np.ndarray]:
    """Read an image as an ``(H, W, 4)`` RGBA array, shrunk to *max_edge*."""
    reader = QtGui.QImageReader(path)
    reader.setAutoTransform(False)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_edge:
        scaled = size.scaled(max_edge, max_edge,
                             QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(scaled)
    image = reader.read()
    if image.isNull():
        return None
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h = image.width(), image.height()
    ptr = image.constBits()
    if hasattr(ptr, "setsize"):        # PyQt5 sip.voidptr
        ptr.setsize(image.sizeInBytes() if hasattr(image, "sizeInBytes")
                    else image.byteCount())
    stride = image.bytesPerLine()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, stride)[:, :w * 4]
    return arr.reshape(h, w, 4).copy()


def image_size(path: str):
    """``(width, height)`` of an image file without decoding it, or None."""
    size = QtGui.QImageReader(path).size()
    return (size.width(), size.height()) if size.isValid() else None


class FrameCache(QtCore.QObject):
    #: (key, rgba ndarray or None when unreadable)
    loaded = Signal(object, object)

    def __init__(self, max_edge: int = 1024, capacity: int = 96,
                 workers: int = 3, parent=None):
        super().__init__(parent)
        self.max_edge = int(max_edge)
        self.capacity = int(capacity)
        self._items: "OrderedDict[object, np.ndarray]" = OrderedDict()
        self._pending = set()
        self._generation = 0
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="bambi-frames")

    def get(self, key) -> Optional[np.ndarray]:
        arr = self._items.get(key)
        if arr is not None:
            self._items.move_to_end(key)
        return arr

    def contains(self, key) -> bool:
        return key in self._items

    def request(self, key, path: str) -> bool:
        """Start loading *path* unless cached or already loading.

        :return: True when the image is already available
        """
        if key in self._items:
            self._items.move_to_end(key)
            return True
        if key in self._pending or not path:
            return False
        self._pending.add(key)
        generation = self._generation
        max_edge = self.max_edge
        future = self._pool.submit(read_rgba, path, max_edge)
        future.add_done_callback(
            lambda f, k=key, g=generation: self._done(k, g, f))
        return False

    def _done(self, key, generation, future):
        try:
            arr = future.result()
        except Exception:  # noqa: BLE001 - an unreadable frame is just missing
            arr = None
        # Called on a worker thread; hop to the GUI thread.
        QtCore.QMetaObject.invokeMethod(
            self, "_deliver", QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, (key, generation, arr)))

    @QtCore.pyqtSlot(object) if hasattr(QtCore, "pyqtSlot") else QtCore.Slot(object)
    def _deliver(self, payload):
        key, generation, arr = payload
        self._pending.discard(key)
        if generation != self._generation:
            return
        if arr is not None:
            self._items[key] = arr
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        self.loaded.emit(key, arr)

    def clear(self, max_edge: Optional[int] = None):
        """Forget everything (e.g. after changing the texture size)."""
        self._generation += 1
        self._items.clear()
        self._pending.clear()
        if max_edge is not None:
            self.max_edge = int(max_edge)

    def shutdown(self):
        self._generation += 1
        self._pool.shutdown(wait=False, cancel_futures=True)
