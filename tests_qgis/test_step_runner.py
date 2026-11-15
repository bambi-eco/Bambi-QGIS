# -*- coding: utf-8 -*-
"""The step runner: one step at a time, on a thread of its own
(dock/step_runner.py).

Every step of the panel is started through ``dock.start_worker``, which hands
it to ``dock.steps``. The runner was part of the dock class; these tests pin
down what it does now that it is an object of its own - with real threads,
because a thread that is let go of while it runs takes the process down
rather than failing a test.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import gc
import threading
import time

import pytest

from qgis.PyQt.QtCore import QCoreApplication

from bambi_wildlife_detection import gui_utils
from bambi_wildlife_detection.dock import step_runner
from bambi_wildlife_detection.dock.step_runner import StepRunner


def _spin_until(condition, seconds=10.0):
    end = time.time() + seconds
    while time.time() < end:
        QCoreApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _thread_has_ended(thread) -> bool:
    try:
        return not thread.isRunning()
    except RuntimeError:        # deleted by Qt once it had finished
        return True


class Steps:
    """A processor with a step that does what the test says."""

    def __init__(self):
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.ran_on = None

    def run_flight_route(self, config, progress, log, cancelled):
        self.ran_on = threading.current_thread()
        self.started.set()
        log(f"route for {config['name']}")
        progress(40)
        self.release.wait(10)
        if cancelled():
            return
        if config.get("fails"):
            raise ValueError("no flight log")
        progress(100)


class Listener:
    def __init__(self, runner):
        self.runner = runner
        self.said = []

    def finished(self, step, success):
        self.said.append(("finished", step, success))
        self.runner.ended()

    def error(self, step, message):
        self.said.append(("error", step, message.splitlines()[0]))
        self.runner.ended()

    def progress(self, value):
        self.said.append(("progress", value))

    def log(self, text):
        self.said.append(("log", text))

    def start(self, processor, config=None, starting=None):
        self.runner.start(
            processor, "flight_route", config or {"name": "A"},
            finished=self.finished, error=self.error,
            progress=self.progress, log=self.log, starting=starting)

    def ended(self):
        return any(entry[0] == "finished" for entry in self.said)


def _kept() -> int:
    """Threads the plugin holds on to until they have ended."""
    keeper = gui_utils._thread_keeper
    return 0 if keeper is None else keeper.held()


@pytest.fixture
def runner():
    """A runner - and, afterwards, no thread of it left anywhere: a thread
    that outlives its test is counted by the next one that counts."""
    before = _kept()
    made = StepRunner()
    yield made
    made.shut_down()
    assert _spin_until(lambda: _kept() <= before)


def test_nothing_runs_at_first(runner):
    assert not runner.is_running()
    assert runner.worker is None and runner.thread is None
    assert runner.cancel() is False


def test_a_step_runs_on_its_own_thread_and_says_how_it_went(runner):
    steps, listener = Steps(), Listener(runner)
    steps.release.clear()

    listener.start(steps)

    assert runner.is_running()
    assert steps.started.wait(5)
    assert steps.ran_on is not threading.current_thread()
    thread = runner.thread
    steps.release.set()
    assert _spin_until(listener.ended)

    assert listener.said == [
        ("log", "route for A"), ("progress", 40), ("progress", 100),
        ("log", "flight_route completed successfully"),
        ("finished", "flight_route", True)] or listener.said == [
        ("log", "route for A"), ("progress", 40), ("progress", 100),
        ("finished", "flight_route", True)]
    assert not runner.is_running()
    assert _spin_until(lambda: _thread_has_ended(thread))


def test_the_panel_is_told_before_the_thread_starts(runner):
    """What shows that a step runs is on screen before the step can say
    anything."""
    steps, listener = Steps(), Listener(runner)
    order = []

    def starting():
        order.append(("starting", steps.started.is_set(),
                      runner.is_running()))

    listener.start(steps, starting=starting)
    assert _spin_until(listener.ended)

    assert order == [("starting", False, True)]


def test_a_step_that_fails_says_why_and_then_ends(runner):
    steps, listener = Steps(), Listener(runner)

    listener.start(steps, {"name": "A", "fails": True})
    assert _spin_until(listener.ended)

    kinds = [entry[0] for entry in listener.said]
    assert kinds.index("error") < kinds.index("finished")
    assert ("error", "flight_route", "no flight log") in listener.said
    assert ("finished", "flight_route", False) in listener.said
    assert not runner.is_running()


def test_cancel_stops_the_step_at_its_next_check(runner):
    steps, listener = Steps(), Listener(runner)
    steps.release.clear()

    listener.start(steps)
    assert steps.started.wait(5)
    assert runner.cancel() is True
    steps.release.set()
    assert _spin_until(listener.ended)

    assert ("finished", "flight_route", False) in listener.said
    assert ("log", "flight_route was cancelled") in listener.said
    assert ("progress", 100) not in listener.said


def test_a_second_step_is_refused_while_one_runs(runner):
    steps, listener = Steps(), Listener(runner)
    steps.release.clear()
    listener.start(steps)
    first = runner.worker

    with pytest.raises(RuntimeError, match="still running"):
        listener.start(Steps())

    assert runner.worker is first
    steps.release.set()
    assert _spin_until(listener.ended)


def test_the_next_step_can_be_started_from_the_end_of_the_last(runner):
    """"finished" arrives before the thread was told to quit. Starting the
    next step right there replaced the thread object while its thread was
    still running - which Qt answers by ending the process. The environment
    panel had to learn that the hard way; the runner now keeps the old
    thread until it has ended.

    Nothing here holds on to a thread: only the runner does, as in the dock.
    """
    steps = Steps()
    said = []
    started = []

    def finished(step, success):
        said.append((step, success, len(started)))
        runner.ended()
        if len(started) < 3:
            start()

    def start():
        runner.start(steps, "flight_route", {"name": str(len(started))},
                     finished=finished, error=lambda *a: said.append(a),
                     progress=lambda value: None, log=lambda text: None)
        started.append(runner.thread.isRunning())

    start()
    assert _spin_until(lambda: len(said) == 3)
    gc.collect()
    _spin_until(lambda: False, seconds=0.2)      # ... and QGIS is still here

    assert said == [("flight_route", True, 1), ("flight_route", True, 2),
                    ("flight_route", True, 3)]
    assert started == [True, True, True]
    assert _spin_until(lambda: _thread_has_ended(runner.thread))


def test_shutting_down_lets_a_running_step_end(runner):
    steps, listener = Steps(), Listener(runner)
    steps.release.clear()
    listener.start(steps)
    assert steps.started.wait(5)
    thread = runner.thread

    runner.shut_down()

    assert not runner.is_running() and runner.thread is None
    steps.release.set()
    assert _spin_until(lambda: _thread_has_ended(thread))
    # nobody is listening any more
    assert not listener.ended()


# --- the dock and its runner --------------------------------------------------

def test_the_dock_has_one_runner(dock):
    assert isinstance(dock.steps, StepRunner)
    assert dock.worker is None and dock.worker_thread is None
    assert not dock._step_is_running()


def test_the_docks_worker_is_the_runners(dock):
    """``dock.worker`` and ``dock.worker_thread`` are the names the rest of
    the panel - and the environment panel - have always used."""
    worker, thread = object(), object()

    dock.worker, dock.worker_thread = worker, thread
    try:
        assert dock.steps.worker is worker and dock.steps.thread is thread
        assert dock._step_is_running()
    finally:
        dock.worker = dock.worker_thread = None

    dock.steps.worker = worker
    try:
        assert dock.worker is worker
    finally:
        dock.steps.worker = None


class _Recorder:
    """Stands for a worker and for a thread, and writes down what is done
    with it."""

    calls = []

    def __init__(self, *args):
        self.kind = "worker" if args else "thread"
        self.args = args
        _Recorder.calls.append((self.kind, "made"))
        for name in ("finished", "error", "progress", "log", "started"):
            setattr(self, name, _Signal(self.kind, name))

    def moveToThread(self, thread):
        _Recorder.calls.append((self.kind, "moved"))

    def start(self):
        _Recorder.calls.append((self.kind, "started"))

    def isRunning(self):
        return False

    def run(self):
        pass

    def quit(self):
        pass

    def deleteLater(self):
        pass

    def cancel(self):
        _Recorder.calls.append((self.kind, "cancelled"))


class _Signal:
    def __init__(self, kind, name):
        self.kind, self.name = kind, name

    def connect(self, slot):
        _Recorder.calls.append(
            (self.kind, self.name, getattr(slot, "__name__", "?")))


@pytest.fixture
def recorded(monkeypatch):
    _Recorder.calls = []
    monkeypatch.setattr(step_runner, "ProcessingWorker", _Recorder)
    monkeypatch.setattr(step_runner, "QThread", _Recorder)
    return _Recorder.calls


def test_start_worker_hands_the_step_to_the_runner(dock, recorded,
                                                   monkeypatch):
    """The same connections in the same order as when the dock made them
    itself, and the panel is switched over before the thread starts."""
    shown = []
    monkeypatch.setattr(dock, "set_buttons_enabled",
                        lambda enabled: shown.append(("buttons", enabled)))
    try:
        dock.start_worker("flight_route")

        assert dock._step_is_running()
        assert dock.steps.worker.args[0] is dock.processor
        assert dock.steps.worker.args[1] == "flight_route"
        assert dock.steps.worker.args[2]["target_folder"] == \
            dock.get_config()["target_folder"]
        assert recorded == [
            ("thread", "made"), ("worker", "made"), ("worker", "moved"),
            ("thread", "started", "run"),
            ("worker", "finished", "on_worker_finished"),
            ("worker", "error", "on_worker_error"),
            ("worker", "progress", "on_worker_progress"),
            ("worker", "log", "log"),
            ("worker", "finished", "quit"),
            ("worker", "finished", "deleteLater"),
            ("thread", "finished", "deleteLater"),
            ("thread", "started"),
        ]
        assert shown == [("buttons", False)]
        assert dock.abort_btn.isEnabled()
        assert dock.progress_bar.value() == 0
        assert "Starting flight_route..." in dock.log_text.toPlainText()
    finally:
        dock.worker = dock.worker_thread = None


def test_the_end_of_a_step_frees_the_runner(dock, recorded):
    try:
        dock.start_worker("flight_route")
        dock.on_worker_finished("flight_route", True)

        assert not dock.steps.is_running()
        assert not dock.abort_btn.isEnabled()
        assert dock.progress_bar.value() == 100
    finally:
        dock.worker = dock.worker_thread = None


def test_abort_goes_to_the_runner(dock, recorded):
    try:
        dock.start_worker("flight_route")
        dock._abort_current_process()

        assert ("worker", "cancelled") in recorded
        assert not dock.abort_btn.isEnabled()
        assert "Aborting current process..." in dock.log_text.toPlainText()
    finally:
        dock.worker = dock.worker_thread = None


def test_a_second_start_is_answered_with_a_message(dock, recorded,
                                                   monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warned.append(args[1:]))
    try:
        dock.start_worker("flight_route")
        made = len(recorded)
        dock.start_worker("detection")

        assert warned == [("Processing", "Another process is already running.")]
        assert len(recorded) == made
    finally:
        dock.worker = dock.worker_thread = None
