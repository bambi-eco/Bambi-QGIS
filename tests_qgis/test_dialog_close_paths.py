# -*- coding: utf-8 -*-
"""A tool tidies up however it is closed.

A dialog goes away in several ways and only one of them is a close event:
the window's close button. Escape rejects it, its own Close or OK button
rejects or accepts it. The tools did their tidying up in ``closeEvent`` - so
a viewer closed with Escape was hidden with its worker threads running, its
files open and everything it had loaded still in memory; and as the plugin
lets go of a tool when it has finished, opening the tool again made a second
one beside it.

``gui_utils.ClosesOnce`` gives a dialog one place to ask before it closes
and one place to tidy up afterwards.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtTest import QTest
from qgis.PyQt.QtWidgets import QDialog, QMessageBox

from bambi_wildlife_detection import gui_utils
from bambi_wildlife_detection.gui_utils import ClosesOnce


def _press_escape(dialog):
    QTest.keyClick(dialog, Qt.Key.Key_Escape)


#: Every way a dialog is closed.
WAYS = {
    "the window's close button": lambda dialog: dialog.close(),
    "Escape": _press_escape,
    "reject": lambda dialog: dialog.reject(),
    "accept": lambda dialog: dialog.accept(),
    "done": lambda dialog: dialog.done(3),
}
#: ... and those that ask first.
ASKING = ("the window's close button", "Escape", "reject")


@pytest.fixture(params=sorted(WAYS))
def close(request):
    return WAYS[request.param]


@pytest.fixture(params=ASKING)
def ask_to_close(request):
    return WAYS[request.param]


@pytest.fixture
def opened(request):
    """Show a dialog for the test and dispose of it afterwards."""
    dialogs = []

    def show(dialog):
        dialogs.append(dialog)
        dialog.show()
        return dialog

    yield show
    for dialog in dialogs:
        dialog.hide()
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------

class Tool(ClosesOnce, QDialog):
    def __init__(self):
        super().__init__()
        self.may_close = True
        self.asked = self.tidied = 0
        self.fails = False

    def _before_closing(self):
        self.asked += 1
        return self.may_close

    def _when_closed(self):
        self.tidied += 1
        if self.fails:
            raise RuntimeError("could not save the settings")


class TestClosesOnce:
    def test_it_tidies_up_whichever_way_it_is_closed(self, opened, close):
        tool = opened(Tool())
        close(tool)
        assert not tool.isVisible()
        assert tool.tidied == 1

    def test_it_asks_before_closing(self, opened, ask_to_close):
        tool = opened(Tool())
        tool.may_close = False

        ask_to_close(tool)
        assert tool.isVisible() and tool.asked == 1 and tool.tidied == 0

        tool.may_close = True
        ask_to_close(tool)
        assert not tool.isVisible() and tool.tidied == 1

    def test_it_tidies_up_once_per_closing(self, opened):
        tool = opened(Tool())
        tool.reject()
        tool.done(0)                    # closed already
        assert tool.tidied == 1

        tool.show()                     # and open again
        tool.accept()
        assert tool.tidied == 2

    def test_a_failure_while_tidying_up_does_not_keep_it_open(
            self, opened, capsys):
        tool = opened(Tool())
        tool.fails = True
        tool.reject()
        assert not tool.isVisible()
        assert "could not save the settings" in capsys.readouterr().err

    def test_a_dialog_that_was_never_shown_tidies_up_when_closed(self):
        """What it started before it was shown has to end too - and its
        close event does not reject a dialog that is not on screen."""
        tool = Tool()
        tool.close()
        assert tool.tidied == 1
        tool.close()
        assert tool.tidied == 1
        tool.deleteLater()

    def test_a_dialog_that_was_never_shown_is_asked_too(self):
        from qgis.PyQt.QtGui import QCloseEvent

        tool = Tool()
        tool.may_close = False
        event = QCloseEvent()
        tool.closeEvent(event)
        assert not event.isAccepted() and tool.tidied == 0

        tool.may_close = True
        event = QCloseEvent()
        tool.closeEvent(event)
        assert event.isAccepted() and tool.tidied == 1
        tool.deleteLater()

    def test_a_hidden_dialog_is_not_asked_or_tidied_up_twice(self, opened):
        tool = opened(Tool())
        tool.reject()
        tool.close()
        assert tool.tidied == 1 and tool.asked == 1

    def test_a_modal_run_ends_the_same_way(self, opened):
        from qgis.PyQt.QtCore import QTimer

        tool = Tool()
        QTimer.singleShot(0, tool.accept)
        assert tool.exec() == QDialog.DialogCode.Accepted
        assert tool.tidied == 1
        tool.deleteLater()


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

def _tools():
    from bambi_wildlife_detection.bambi_agl_tool import AglToolDialog
    from bambi_wildlife_detection.bambi_camera_calibration import (
        CameraCalibrationWizard)
    from bambi_wildlife_detection.bambi_correction_wizard import (
        BambiCorrectionWizard)
    from bambi_wildlife_detection.bambi_feature_viewer import (
        FeatureViewerDialog)
    from bambi_wildlife_detection.bambi_flight_planner import (
        FlightPlannerDialog)
    from bambi_wildlife_detection.bambi_flight_viewer import FlightViewerDialog
    from bambi_wildlife_detection.bambi_labelling_tool import (
        LabellingToolDialog)
    from bambi_wildlife_detection.bambi_segmentation_tool import (
        SegmentationToolDialog)
    from bambi_wildlife_detection.bambi_thermal_viewer import (
        ThermalViewerDialog)
    from bambi_wildlife_detection.bambi_transect_tool import TransectToolDialog
    from bambi_wildlife_detection.bambi_video_creator import VideoCreatorDialog

    return [AglToolDialog, CameraCalibrationWizard, BambiCorrectionWizard,
            FeatureViewerDialog, FlightPlannerDialog, FlightViewerDialog,
            LabellingToolDialog, SegmentationToolDialog, ThermalViewerDialog,
            TransectToolDialog, VideoCreatorDialog]


def test_no_tool_tidies_up_in_its_close_event_only():
    """Whatever a tool has to do when it closes is said in the two places
    every way of closing gets to."""
    for tool in _tools():
        assert issubclass(tool, ClosesOnce), tool.__name__
        assert tool.__mro__.index(ClosesOnce) < tool.__mro__.index(QDialog)
        own = set(vars(tool))
        assert not own & {"closeEvent", "reject", "accept", "done"}, \
            tool.__name__
        assert own & {"_before_closing", "_when_closed"}, tool.__name__


class Handle:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_the_thermal_viewer_closes_its_file(opened, close):
    from bambi_wildlife_detection.bambi_thermal_viewer import ThermalViewerDialog

    viewer = opened(ThermalViewerDialog())
    handle = viewer._thermal = Handle()

    close(viewer)

    assert handle.closed == 1 and viewer._thermal is None


def test_the_feature_viewer_stops_its_projection(opened, close, monkeypatch):
    from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog

    stopped = []
    monkeypatch.setattr(FeatureViewerDialog, "_stop_projection_worker",
                        lambda self: stopped.append(self))
    viewer = opened(FeatureViewerDialog())

    close(viewer)

    assert stopped == [viewer]


def test_the_video_creator_stops_rendering(opened, close, iface, dock):
    from bambi_wildlife_detection.bambi_video_creator import VideoCreatorDialog

    creator = opened(VideoCreatorDialog(iface, dock))
    creator._rendering = True
    creator._cancel = False

    close(creator)

    assert creator._cancel is True


def test_the_flight_planner_removes_its_temporary_files(
        opened, close, iface, tmp_path, monkeypatch):
    from bambi_wildlife_detection.bambi_flight_planner import FlightPlannerDialog

    saved = []
    monkeypatch.setattr(FlightPlannerDialog, "_save_settings",
                        lambda self: saved.append(self))
    planner = opened(FlightPlannerDialog(iface))
    temporary = tmp_path / "plan.gpkg"
    temporary.write_bytes(b"x")
    planner._temp_files.append(str(temporary))

    close(planner)

    assert not os.path.exists(temporary) and planner._temp_files == []
    assert saved == [planner]


def test_the_calibration_wizard_stops_its_worker(opened, close, monkeypatch):
    """Accepting it - after saving - did not."""
    from bambi_wildlife_detection.bambi_camera_calibration import (
        CameraCalibrationWizard)

    stopped = []
    monkeypatch.setattr(CameraCalibrationWizard, "_stop_worker",
                        lambda self: stopped.append(self))
    wizard = opened(CameraCalibrationWizard())

    close(wizard)

    assert stopped == [wizard]


def test_the_agl_tool_lets_its_worker_finish(opened, close, iface, dock,
                                             monkeypatch):
    from bambi_wildlife_detection.bambi_agl_tool import AglToolDialog

    let_go = []
    monkeypatch.setattr(gui_utils, "let_thread_finish",
                        lambda thread, *rest, **kwargs: let_go.append(thread))
    tool = opened(AglToolDialog(iface, dock))
    tool._worker = worker = object()
    tool._pending = "another flight"

    close(tool)

    assert let_go == [worker] and tool._pending is None


def test_the_flight_viewer_gives_everything_back(opened, close, iface, dock,
                                                 monkeypatch):
    """Its own worker threads, the panels' (the light-field renderer with
    its render context among them), the frame cache, the 3D view."""
    from bambi_wildlife_detection import bambi_flight_viewer
    from bambi_wildlife_detection.bambi_flight_viewer import FlightViewerDialog

    monkeypatch.setattr(bambi_flight_viewer, "let_thread_finish",
                        lambda *args, **kwargs: True)
    saved = []
    monkeypatch.setattr(FlightViewerDialog, "_save_settings",
                        lambda self: saved.append(self))
    viewer = opened(FlightViewerDialog(iface, dock))

    class Part:
        def __init__(self):
            self.shut = 0

        def shutdown(self):
            self.shut += 1

        dispose = shutdown

    parts = {name: Part() for name in ("alfs_panel", "corrections_panel",
                                       "recon_panel", "cache", "viewport")}
    for name, part in parts.items():
        setattr(viewer, name, part)

    close(viewer)

    assert {name: part.shut for name, part in parts.items()} \
        == dict.fromkeys(parts, 1)
    assert saved == [viewer]


EDITORS = [("bambi_labelling_tool", "LabellingToolDialog"),
           ("bambi_transect_tool", "TransectToolDialog")]
#: Tools in which Escape ends a mode, and does not close the window.
KEEP_OPEN_ON_ESCAPE = EDITORS + [
    ("bambi_segmentation_tool", "SegmentationToolDialog")]


def _editor(module, name):
    import importlib

    return getattr(importlib.import_module(
        f"bambi_wildlife_detection.{module}"), name)


@pytest.mark.parametrize("module, name", EDITORS)
@pytest.mark.parametrize("way", ["the window's close button", "reject"])
def test_unsaved_work_is_asked_about(opened, iface, dock, monkeypatch,
                                     module, name, way):
    tool_class = _editor(module, name)
    answers = [False, True]             # "Cancel", then "Save"
    monkeypatch.setattr(tool_class, "_maybe_save_dirty",
                        lambda self: answers.pop(0))
    tool = opened(tool_class(iface, dock))

    WAYS[way](tool)
    assert tool.isVisible()             # the answer was: do not close

    WAYS[way](tool)
    assert not tool.isVisible() and answers == []


@pytest.mark.parametrize("module, name", KEEP_OPEN_ON_ESCAPE)
def test_escape_does_not_close_an_editor(opened, iface, dock, monkeypatch,
                                         module, name):
    """It ends a drawing mode there; closing is not what it is for."""
    tool_class = _editor(module, name)
    asked = []
    monkeypatch.setattr(tool_class, "_before_closing",
                        lambda self: asked.append(self) or True)
    tool = opened(tool_class(iface, dock))

    _press_escape(tool)

    assert tool.isVisible() and asked == []


@pytest.mark.parametrize("way", ["the window's close button", "reject"])
def test_a_running_segmentation_is_asked_about(opened, iface, dock,
                                               monkeypatch, way):
    from bambi_wildlife_detection import bambi_segmentation_tool as tool_module
    from bambi_wildlife_detection.bambi_segmentation_tool import (
        SegmentationToolDialog)

    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(tool_module.QMessageBox, "question",
                        lambda *args, **kwargs: answers.pop(0))
    monkeypatch.setattr(gui_utils, "let_thread_finish",
                        lambda *args, **kwargs: True)
    done = []
    monkeypatch.setattr(SegmentationToolDialog, "_cancel",
                        lambda self: done.append("cancelled"))
    monkeypatch.setattr(SegmentationToolDialog, "_save_settings",
                        lambda self: done.append("settings saved"))
    tool = opened(SegmentationToolDialog(iface, dock))
    tool._worker = object()             # a run in progress

    WAYS[way](tool)
    assert tool.isVisible() and done == []

    WAYS[way](tool)
    assert not tool.isVisible()
    assert done == ["cancelled", "settings saved"]


@pytest.mark.parametrize("way", sorted(set(WAYS) - {"Escape"}))
def test_an_idle_segmentation_tool_is_not_asked_about(opened, iface, dock,
                                                      monkeypatch, way):
    from bambi_wildlife_detection import bambi_segmentation_tool as tool_module
    from bambi_wildlife_detection.bambi_segmentation_tool import (
        SegmentationToolDialog)

    asked = []
    monkeypatch.setattr(tool_module.QMessageBox, "question",
                        lambda *args, **kwargs: asked.append(args))
    saved = []
    monkeypatch.setattr(SegmentationToolDialog, "_save_settings",
                        lambda self: saved.append(self))
    tool = opened(SegmentationToolDialog(iface, dock))

    WAYS[way](tool)

    assert asked == [] and saved == [tool]


def test_a_wizard_that_could_not_open_closes_without_a_word(
        opened, close, iface, monkeypatch, capsys):
    """It has built nothing; there is nothing to tidy up - and the message
    it was about to show is not shown into a dialog that is gone."""
    from bambi_wildlife_detection import bambi_correction_wizard as module
    from bambi_wildlife_detection.bambi_correction_wizard import (
        BambiCorrectionWizard)
    from qgis.PyQt.QtWidgets import QApplication

    shown = []
    monkeypatch.setattr(module.QMessageBox, "critical",
                        lambda *args: shown.append(args[1:]))
    wizard = BambiCorrectionWizard(iface, {})

    close(wizard)
    QApplication.processEvents()        # the deferred refusal, if any

    assert capsys.readouterr().err == ""
    assert shown == []
    wizard.deleteLater()


def test_a_wizard_that_cannot_open_says_why(iface, monkeypatch):
    from bambi_wildlife_detection import bambi_correction_wizard as module
    from bambi_wildlife_detection.bambi_correction_wizard import (
        BambiCorrectionWizard)

    shown = []
    monkeypatch.setattr(module.QMessageBox, "critical",
                        lambda *args: shown.append(args[1]))
    wizard = BambiCorrectionWizard(iface, {})

    assert wizard.exec() == QDialog.DialogCode.Rejected
    assert shown == ["Cannot open wizard"]
    wizard.deleteLater()


# ---------------------------------------------------------------------------
# A closed tool goes away
# ---------------------------------------------------------------------------

def _deleted(dialog) -> bool:
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return sip.isdeleted(dialog)


class TestForgetWhenFinished:
    """The plugin emptied its attribute when a tool had finished - which let
    go of the Python object, not of the dialog: a child of the QGIS window
    lives as long as that window. Every tool opened and closed stayed,
    hidden, with all it had loaded."""

    class Owner:
        tool = None

    @pytest.fixture
    def window(self):
        from qgis.PyQt.QtWidgets import QWidget

        window = QWidget()
        yield window
        window.deleteLater()

    def test_a_finished_tool_is_forgotten_and_deleted(self, window, close):
        owner = self.Owner()
        owner.tool = tool = Tool()
        tool.setParent(window, tool.windowFlags())
        gui_utils.forget_when_finished(owner, "tool")
        tool.show()
        assert not _deleted(tool)

        close(tool)

        assert owner.tool is None
        assert _deleted(tool)
        assert window.findChildren(QDialog) == []

    def test_a_tool_that_stays_open_is_kept(self, window):
        owner = self.Owner()
        owner.tool = tool = Tool()
        tool.setParent(window, tool.windowFlags())
        gui_utils.forget_when_finished(owner, "tool")
        tool.show()
        tool.may_close = False

        tool.close()                    # "there is unsaved work": stays

        assert owner.tool is tool and not _deleted(tool)

    def test_a_newer_tool_is_not_forgotten_for_an_older_one(self, window):
        owner = self.Owner()
        owner.tool = old = Tool()
        gui_utils.forget_when_finished(owner, "tool")
        old.show()
        owner.tool = new = Tool()       # opened again in the meantime

        old.reject()

        assert owner.tool is new
        assert _deleted(old) and not _deleted(new)
        new.deleteLater()

    def test_the_plugin_lets_go_of_its_tools_this_way(self):
        import inspect

        from bambi_wildlife_detection import bambi_wildlife_detection as plugin

        source = inspect.getsource(plugin)
        assert source.count("forget_when_finished(self, ") == 7
        assert "setattr(self, '_" not in source
