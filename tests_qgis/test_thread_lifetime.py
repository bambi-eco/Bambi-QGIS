# -*- coding: utf-8 -*-
"""A thread is never destroyed while it runs (gui_utils.let_thread_finish).

Qt aborts the whole process when a QThread object is destroyed while its
thread is still running. The tools used to wait a few seconds and then let
go of the thread: cancelling a calibration, closing a window or unloading
the plugin during work that cannot be interrupted took QGIS down.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import gc
import threading
import time

from qgis.PyQt.QtCore import QCoreApplication, QObject, QThread, pyqtSignal

from bambi_wildlife_detection import gui_utils
from bambi_wildlife_detection.gui_utils import let_thread_finish


class Busy(QThread):
    """Work that cannot be interrupted, until *release* is set."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.release = threading.Event()
        self.started_work = threading.Event()

    def run(self):
        self.started_work.set()
        self.release.wait(10)


def _spin(seconds=0.05):
    end = time.time() + seconds
    while time.time() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def _kept():
    keeper = gui_utils._thread_keeper
    return 0 if keeper is None else keeper.held()


def test_a_thread_that_has_ended_needs_no_keeping():
    thread = Busy()
    thread.release.set()
    thread.start()
    thread.wait()
    assert let_thread_finish(thread) is True
    assert let_thread_finish(None) is True


def test_a_thread_that_ends_within_the_wait_is_not_kept():
    before = _kept()
    thread = Busy()
    thread.start()
    thread.started_work.wait(5)
    threading.Timer(0.05, thread.release.set).start()
    assert let_thread_finish(thread, wait_ms=5000) is True
    assert _kept() == before


def test_a_busy_thread_survives_the_last_reference_to_it():
    """This is the crash: the reference is dropped, the garbage collector
    destroys the thread object, the thread is still running."""
    before = _kept()
    thread = Busy()
    thread.start()
    thread.started_work.wait(5)
    release = thread.release

    assert let_thread_finish(thread, wait_ms=20) is False
    del thread
    gc.collect()
    _spin()                              # ... and QGIS is still here
    assert _kept() == before + 1

    release.set()
    for _ in range(100):
        _spin()
        if _kept() == before:
            break
    assert _kept() == before             # let go once it has ended


def test_a_busy_thread_survives_its_parent():
    """A worker created with the window as parent goes down with it."""
    before = _kept()
    parent = QObject()
    thread = Busy(parent)
    thread.start()
    thread.started_work.wait(5)
    release = thread.release

    assert let_thread_finish(thread, wait_ms=20) is False
    assert thread.parent() is None
    del thread
    parent.deleteLater()
    del parent
    gc.collect()
    _spin()

    release.set()
    for _ in range(100):
        _spin()
        if _kept() == before:
            break
    assert _kept() == before


class _Worker(QObject):
    """Stands in for the workers of the dock and the calibration wizard:
    their signals, ``cancel()``, and a ``run()`` that takes its time."""

    finished = pyqtSignal()
    error = pyqtSignal()
    progress = pyqtSignal()
    log = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.running = threading.Event()
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        self.running.set()
        self.release.wait(10)
        self.finished.emit()


def test_a_worker_moved_onto_the_thread_lives_as_long(monkeypatch):
    before = _kept()
    thread = QThread()
    worker = _Worker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()
    worker.running.wait(5)
    release = worker.release

    assert let_thread_finish(thread, worker, wait_ms=20) is False
    del thread, worker
    gc.collect()
    _spin()
    release.set()
    for _ in range(100):
        _spin()
        if _kept() == before:
            break
    assert _kept() == before


class EndsWhenAsked(QThread):
    """A thread that only waits for work - and has ended by the time
    ``quit()`` returns, as an idle thread on a fast machine may have."""

    def quit(self):
        super().quit()
        self.wait()


def test_a_thread_that_ends_while_it_is_handed_over_is_let_go():
    """The keeper listened for "finished" only after it had asked the
    thread to stop. A thread that was merely waiting had ended by then: its
    "finished" was over, nothing ever said so again, and the thread - with
    the worker kept alongside it - was held until some other thread ended,
    or for good."""
    before = _kept()
    thread = EndsWhenAsked()
    thread.start()
    for _ in range(200):
        if thread.isRunning():
            break
        time.sleep(0.005)

    let_thread_finish(thread, object())
    _spin(0.2)

    assert _kept() == before


def test_unloading_during_a_step_lets_the_step_end(dock):
    """The dock was deleted with the step still running."""
    before = _kept()
    thread = QThread()
    worker = _Worker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    dock.worker, dock.worker_thread = worker, thread
    thread.start()
    worker.running.wait(5)
    release = worker.release

    dock.shut_down()

    assert worker.cancelled
    assert dock.worker is None and dock.worker_thread is None
    assert _kept() == before + 1
    del thread, worker
    gc.collect()
    _spin()
    release.set()
    for _ in range(100):
        _spin()
        if _kept() == before:
            break
    assert _kept() == before


def test_shutting_down_an_idle_dock_is_nothing(dock):
    dock.shut_down()
    assert dock.worker is None


def test_cancelling_a_calibration_lets_it_end(monkeypatch):
    """Structure from motion cannot be interrupted in the middle of a
    step; the wizard waited three seconds and dropped the thread."""
    from bambi_wildlife_detection.bambi_camera_calibration import (
        CameraCalibrationWizard)

    before = _kept()
    wizard = CameraCalibrationWizard.__new__(CameraCalibrationWizard)
    thread = QThread()
    worker = _Worker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    wizard._worker, wizard._worker_thread = worker, thread
    thread.start()
    worker.running.wait(5)
    release = worker.release

    CameraCalibrationWizard._stop_worker(wizard)

    assert worker.cancelled
    assert wizard._worker is None and wizard._worker_thread is None
    assert _kept() == before + 1
    del thread, worker
    gc.collect()
    _spin()
    release.set()
    for _ in range(100):
        _spin()
        if _kept() == before:
            break
    assert _kept() == before
