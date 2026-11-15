# -*- coding: utf-8 -*-
"""What the plugin survives quietly can be read in QGIS.

``core.quiet`` writes to a Python logger, which in QGIS goes nowhere. The
plugin shows it in the *Log Messages* panel, tab "BAMBI", and problems in
its own log as well.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import threading

import pytest

from qgis.core import Qgis, QgsApplication
from qgis.PyQt.QtWidgets import QApplication

from bambi_wildlife_detection import gui_utils
from bambi_wildlife_detection.core import quiet


@pytest.fixture
def log_panel():
    """What arrives in QGIS' message log while the notes are shown."""
    quiet.forget()
    received, in_the_panel = [], []

    def note(message, tag, level):
        received.append((tag, level, message))

    QgsApplication.messageLog().messageReceived.connect(note)
    gui_utils.show_notes_in_qgis(also=in_the_panel.append)
    yield received, in_the_panel
    gui_utils.hide_notes_from_qgis()
    QgsApplication.messageLog().messageReceived.disconnect(note)
    quiet.forget()


def test_a_problem_is_a_warning_in_the_log_messages_panel(log_panel):
    received, in_the_panel = log_panel
    try:
        raise ValueError("Expecting value: line 1 column 1")
    except Exception:
        quiet.problem("The correction file c.json could not be read; "
                      "no correction is applied")
    QApplication.processEvents()

    text = ("The correction file c.json could not be read; no correction is "
            "applied (ValueError: Expecting value: line 1 column 1).")
    assert received == [("BAMBI", Qgis.MessageLevel.Warning, text)]
    assert in_the_panel == [text]


def test_a_detail_is_information_and_stays_out_of_the_panel(log_panel):
    received, in_the_panel = log_panel
    quiet.detail("The map canvas could not be refreshed")
    QApplication.processEvents()

    assert received == [("BAMBI", Qgis.MessageLevel.Info,
                         "The map canvas could not be refreshed.")]
    assert in_the_panel == []


def test_a_note_from_a_step_running_in_its_own_thread_arrives(log_panel):
    received, in_the_panel = log_panel

    def step():
        try:
            raise OSError("read-only file system")
        except Exception:
            quiet.problem("The surface mesh could not be written")

    worker = threading.Thread(target=step)
    worker.start()
    worker.join()
    for _ in range(10):
        QApplication.processEvents()

    assert [message for _tag, _level, message in received] == [
        "The surface mesh could not be written (OSError: read-only file "
        "system)."]
    assert len(in_the_panel) == 1


def test_nothing_arrives_once_the_plugin_is_unloaded(log_panel):
    received, _in_the_panel = log_panel
    gui_utils.hide_notes_from_qgis()
    quiet.problem("Said to nobody")
    QApplication.processEvents()
    assert received == []


def test_showing_twice_does_not_say_everything_twice(log_panel):
    received, _in_the_panel = log_panel
    gui_utils.show_notes_in_qgis()
    quiet.problem("Once")
    QApplication.processEvents()
    assert len(received) == 1
