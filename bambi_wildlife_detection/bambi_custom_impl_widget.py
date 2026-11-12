# -*- coding: utf-8 -*-
"""Widgets for choosing a custom implementation of a stage (9.0.0).

The interface itself is :mod:`core.custom_models`; this is the Qt side:
the box on the configuration tab (file, class, options, "Check"), the
dialog version of it used by the classification table, the trust prompt
that precedes running somebody's ``.py`` file, and the sample download.
"""

import json
import os
import time
import traceback

from qgis.PyQt.QtCore import (
    QEventLoop, QSettings, QThread, QTimer, pyqtSignal,
)
from qgis.PyQt.QtGui import QFontDatabase
from qgis.PyQt.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QProgressDialog, QPushButton, QVBoxLayout, QWidget,
)

from .core import custom_models

#: QSettings key: sha256 of every implementation file the user approved on
#: this machine. Deliberately not in the project - a flight's configuration
#: travels with its folder, an approval must not.
_TRUST_KEY = "BAMBI/custom_code_trusted"

DEPENDENCY_NOTE = (
    "Custom code runs inside QGIS with your user rights, and the plugin "
    "does not manage its dependencies: install the packages it imports "
    "into QGIS' Python yourself - in the OSGeo4W Shell on Windows, or a "
    "terminal with the Python QGIS uses on Linux / macOS: "
    "<code>python -m pip install &lt;package&gt;</code>. "
    "<b>Take care:</b> installing or upgrading a package can change one "
    "the plugin, QGIS or another plugin relies on (torch, transformers, "
    "numpy, opencv ...) and break steps that worked before. Install only "
    "what is missing, avoid <code>--upgrade</code> where you can, and look "
    "at the Dependency Manager afterwards - it shows what is out of range "
    "and can put the plugin's own packages back."
)


#: What may sit next to the selected file.
FOLDER_NOTE = (
    "The file does not have to stand alone: everything in its folder is "
    "available to it. Other .py files there can be imported by name "
    "(<code>import my_helpers</code>), and model weights, label lists or "
    "other data next to it are reached with "
    "<code>os.path.join(os.path.dirname(__file__), \"weights.pt\")</code>."
)


# ---------------------------------------------------------------------------
# Trust
# ---------------------------------------------------------------------------

def _trusted() -> set:
    raw = QSettings().value(_TRUST_KEY, "[]")
    try:
        return set(json.loads(raw or "[]"))
    except (TypeError, ValueError):
        return set()


def is_trusted(path: str) -> bool:
    """True when this exact file content was approved on this machine."""
    try:
        return custom_models.file_hash(path) in _trusted()
    except OSError:
        return False


def remember_trust(path: str) -> None:
    trusted = _trusted()
    trusted.add(custom_models.file_hash(path))
    QSettings().setValue(_TRUST_KEY, json.dumps(sorted(trusted)))


def confirm_trust(parent, path: str) -> bool:
    """Ask before running an implementation file for the first time (or
    the first time since it changed). True when it may run."""
    path = (path or "").strip()
    if not path or not os.path.isfile(path):
        return True          # the run reports the missing file itself
    if is_trusted(path):
        return True
    answer = QMessageBox.question(
        parent, "Run custom code?",
        f"{path}\n\n"
        "is a custom implementation that has not been run on this computer "
        "before, or has changed since. It is ordinary Python code and runs "
        "inside QGIS with your user rights - it can read and change "
        "anything you can.\n\n"
        "Only continue if you wrote this file or trust where it came from "
        "(a flight folder you received may point at a file you have never "
        "looked at).\n\nRun it?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No)
    if answer != QMessageBox.StandardButton.Yes:
        return False
    remember_trust(path)
    return True


def confirm_specs(parent, specs) -> bool:
    """Trust prompt for every custom implementation among *specs*."""
    seen = set()
    for spec in specs:
        if not custom_models.is_custom(spec):
            continue
        path = (spec.get("path") or "").strip()
        if path in seen:
            continue
        seen.add(path)
        if not confirm_trust(parent, path):
            return False
    return True


# ---------------------------------------------------------------------------
# Checking an implementation
# ---------------------------------------------------------------------------

def describe(spec: dict, kind: str, camera: str = "", task: str = "",
             model_dir: str = "", status=None) -> str:
    """Check an implementation step by step and say what it declared:
    required packages, import, ``setup``, and for a detector one run on an
    empty image.

    :param status: ``callable(text)`` told each step as it starts - the
        import and ``setup`` of a model can take minutes
    :raises custom_models.CustomModelError: with what is wrong
    """
    status = status or (lambda text: None)
    path = (spec.get("path") or "").strip()

    if os.path.isfile(path):
        present, missing = custom_models.check_requirements(path)
        if present or missing:
            status("Required packages: " + ", ".join(
                [f"{name} - found" for name in present]
                + [f"{name} - MISSING" for name in missing]))
        else:
            status("Required packages: none declared in the file "
                   "(REQUIRES = [...]), so they show up on import.")
        if missing:
            raise custom_models.CustomModelError(
                f"{os.path.basename(path)} needs "
                f"{', '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not installed in "
                f"QGIS' Python. {custom_models.INSTALL_HINT}")

    context = custom_models.Context(
        kind=kind, camera=camera, task=task, model_dir=model_dir,
        log=lambda message: status(f"  {message}"))
    impl = custom_models.open_implementation(spec, kind, context, status)
    try:
        lines = [f"{impl.class_name} in {os.path.basename(impl.path)} "
                 "loaded and set up.",
                 f"Supports: {', '.join(impl.cameras)}"]
        if impl.class_names:
            lines.append(f"Classes: {', '.join(impl.class_names)}")
        if kind == custom_models.CLASSIFIER:
            lines.append(f"Reads {impl.input}, decides per {impl.level}"
                         + (f" (crops of {impl.crop_size} px)"
                            if impl.input == custom_models.INPUT_CROPS
                            else ""))
        elif kind == custom_models.TRACKER:
            lines.append("Called frame by frame (update)" if impl.online
                         else "Called once with the whole flight (run)")
            lines.append(f"Works in {impl.space} space; interpolation of "
                         "gaps " + ("applicable" if impl.interpolate
                                    else "not applicable"))
        elif kind == custom_models.DETECTOR:
            import numpy as np
            status("Test run on an empty image...")
            found = impl.detect(np.zeros((256, 256, 3), np.uint8),
                                custom_models.Frame(0, camera=camera))
            lines.append(f"Test run on an empty image: {len(found)} "
                         "detection(s)")
        return "\n".join(lines)
    finally:
        impl.close()


# ---------------------------------------------------------------------------
# Running user code off the GUI thread
# ---------------------------------------------------------------------------

#: Workers still running after the user stopped waiting for them. A thread
#: inside somebody's setup() cannot be killed safely, so it is left to
#: finish - and kept here so it is not destroyed while it runs.
_ABANDONED = []


class _Worker(QThread):
    """Runs ``fn(status)`` in the background.

    Importing torch or downloading a model inside ``setup()`` takes
    seconds to minutes; on the GUI thread that froze QGIS (2026-11-12).
    """

    status = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)          # message, traceback

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn(self.status.emit)
        except Exception as exc:  # noqa: BLE001 - anything from user code
            self.failed.emit(*_failure(exc))
            return
        self.succeeded.emit(result)


def _failure(exc: Exception):
    """``(message, traceback)`` of a failed check. The traceback is left
    out for the plugin's own findings (a missing package, a file without a
    fitting class) - there it would only bury the message - and kept
    whenever the user's code raised, for searching the error."""
    if isinstance(exc, custom_models.CustomModelError):
        trace = traceback.format_exc() if exc.__cause__ is not None else ""
        return str(exc), trace
    return f"{type(exc).__name__}: {exc}", traceback.format_exc()


def _abandon(worker):
    for signal in (worker.status, worker.succeeded, worker.failed):
        try:
            signal.disconnect()
        except (TypeError, RuntimeError):
            pass
    if worker.isRunning():
        _ABANDONED.append(worker)
        worker.finished.connect(lambda w=worker: w in _ABANDONED
                                and _ABANDONED.remove(w))


def run_with_feedback(parent, title: str, fn):
    """Run ``fn(status)`` in the background behind a small progress window
    that shows what it is doing; the rest of QGIS stays responsive.

    :return: ``(True, result)``, ``(False, message)`` on an error, or
        ``(None, None)`` when the user stopped waiting
    """
    dialog = QProgressDialog(title, "Stop waiting", 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumDuration(400)
    dialog.setMinimumWidth(420)
    outcome = {}
    loop = QEventLoop()
    worker = _Worker(fn)
    worker.status.connect(dialog.setLabelText)
    worker.succeeded.connect(
        lambda result: (outcome.update(ok=True, value=result), loop.quit()))
    worker.failed.connect(
        lambda message, _trace: (outcome.update(ok=False, value=message),
                                 loop.quit()))
    dialog.canceled.connect(loop.quit)
    worker.start()
    loop.exec()
    dialog.close()
    if not outcome:
        _abandon(worker)
        return None, None
    worker.wait(2000)
    return outcome["ok"], outcome["value"]


# ---------------------------------------------------------------------------
# The box
# ---------------------------------------------------------------------------

class CustomImplementationBox(QGroupBox):
    """"Use a custom implementation" with file, class, options and Check."""

    changed = pyqtSignal()

    def __init__(self, title: str, kind: str, camera: str = "",
                 task: str = "", checkable: bool = True, parent=None):
        super().__init__(title, parent)
        self.kind = kind
        self.camera = camera
        self.task = task
        self._model_dir = ""
        self._checkable = bool(checkable)

        layout = QVBoxLayout(self)
        self.enable_check = QCheckBox("Use a custom implementation")
        self.enable_check.setToolTip(
            "Replace the built-in method of this stage by a class from "
            "your own .py file (see \"Download sample implementations\").")
        self.enable_check.setVisible(checkable)
        self.enable_check.setChecked(not checkable)
        layout.addWidget(self.enable_check)

        self.body = QWidget()
        form = QFormLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("A .py file with the class")
        self.path_edit.setToolTip(
            "The .py file that contains the class. Other .py files in the "
            "same folder can be imported by it, and files next to it "
            "(weights, label lists) are reached through "
            "os.path.dirname(__file__).")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        form.addRow("File:", row)

        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText(
            "Empty = the only fitting class in the file")
        form.addRow("Class:", self.class_edit)

        self.options_edit = QLineEdit()
        self.options_edit.setPlaceholderText(
            'JSON passed to setup(), e.g. {"weights": "C:/models/x.pt"}')
        self.options_edit.setToolTip(
            "Free-form options for the implementation: a JSON object that "
            "arrives as context.options in its setup().")
        form.addRow("Options:", self.options_edit)

        check_row = QHBoxLayout()
        self.check_btn = QPushButton("Check")
        self.check_btn.setToolTip(
            "Check the implementation step by step - required packages, "
            "import, setup(), and for a detector a test run - and show "
            "what it declares, before a long run fails on it. Runs in the "
            "background: loading or downloading a model can take a while.")
        self.check_btn.clicked.connect(lambda: self.check())
        check_row.addWidget(self.check_btn)
        self.stop_btn = QPushButton("Stop waiting")
        self.stop_btn.setToolTip(
            "Stop waiting for the check. Code that is already running "
            "(e.g. a download inside setup()) cannot be interrupted and "
            "finishes in the background.")
        self.stop_btn.clicked.connect(self._stop_waiting)
        self.stop_btn.setVisible(False)
        check_row.addWidget(self.stop_btn)
        self.copy_btn = QPushButton("Copy log")
        self.copy_btn.setToolTip("Copy the whole check log to the clipboard")
        self.copy_btn.clicked.connect(self._copy_log)
        self.copy_btn.setVisible(False)
        check_row.addWidget(self.copy_btn)
        check_row.addStretch()
        form.addRow("", check_row)

        # A log rather than a label: the text can be selected and copied
        # (an error message is often something to search for), and a long
        # traceback scrolls instead of stretching the tab.
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log_view.setStyleSheet("font-size: 10px;")
        self.log_view.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log_view.setFixedHeight(140)
        self.log_view.setPlaceholderText(
            "Press Check to test the implementation; what happens is "
            "listed here.")
        self.log_view.setVisible(False)
        form.addRow(self.log_view)
        layout.addWidget(self.body)

        self._worker = None
        self._started = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self.enable_check.toggled.connect(self._on_toggled)
        for edit in (self.path_edit, self.class_edit, self.options_edit):
            edit.editingFinished.connect(self.changed.emit)
        self._on_toggled(self.enable_check.isChecked())

    def set_model_dir(self, folder: str):
        self._model_dir = folder or ""

    def _on_toggled(self, on: bool):
        self.body.setEnabled(bool(on))
        self.changed.emit()

    def _browse(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Choose a custom implementation", self.path_edit.text(),
            "Python file (*.py)")
        if not path:
            return
        self.path_edit.setText(path)
        self._clear_log()
        self.changed.emit()

    # -- value --------------------------------------------------------------

    def spec(self) -> dict:
        """``{"source", "path", "class", "options"}`` as configured."""
        return {
            "source": (custom_models.SOURCE
                       if self.enable_check.isChecked() else "default"),
            "path": self.path_edit.text().strip(),
            "class": self.class_edit.text().strip(),
            "options": self.options_edit.text().strip(),
        }

    def set_spec(self, spec) -> None:
        spec = spec or {}
        blocked = self.blockSignals(True)
        self.enable_check.setChecked(
            custom_models.is_custom(spec) or not self._checkable)
        self.path_edit.setText(spec.get("path", "") or "")
        self.class_edit.setText(spec.get("class", "") or "")
        options = spec.get("options", "") or ""
        if isinstance(options, dict):
            options = json.dumps(options)
        self.options_edit.setText(options)
        self._clear_log()
        self.body.setEnabled(self.enable_check.isChecked())
        self.blockSignals(blocked)

    # -- check --------------------------------------------------------------

    # -- log --------------------------------------------------------------

    def log_text(self) -> str:
        """Everything the last check wrote."""
        return self.log_view.toPlainText()

    def _clear_log(self):
        self.log_view.clear()
        self.log_view.setVisible(False)
        self.copy_btn.setVisible(False)

    def _log(self, text: str):
        self.log_view.setVisible(True)
        self.copy_btn.setVisible(True)
        self.log_view.appendPlainText(text)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _copy_log(self):
        QApplication.clipboard().setText(self.log_text())

    # -- check ------------------------------------------------------------

    def check(self, blocking: bool = False):
        """Check the implementation and list what happens in the log.

        Runs in the background, so QGIS stays usable while a model loads;
        *blocking* runs it in place (tests, scripts).
        """
        if self._worker is not None:
            return
        spec = dict(self.spec(), source=custom_models.SOURCE)
        self.log_view.clear()
        if not spec["path"]:
            self._log("Choose a .py file first.")
            return
        if not confirm_trust(self, spec["path"]):
            self._log("Not run: the file was not approved.")
            return
        self._log(f"Checking {spec['path']}")

        def work(status):
            return describe(spec, self.kind, self.camera, self.task,
                            self._model_dir, status)

        if blocking:
            try:
                self._on_succeeded(work(self._log))
            except Exception as exc:  # noqa: BLE001 - anything from user code
                self._on_failed(*_failure(exc))
            return

        self._worker = _Worker(work, self)
        self._worker.status.connect(self._log)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._started = time.time()
        self.check_btn.setEnabled(False)
        self.check_btn.setText("Checking...")
        self.stop_btn.setVisible(True)
        self._timer.start()
        self._worker.start()

    def _tick(self):
        self.check_btn.setText(
            f"Checking... {int(time.time() - self._started)} s")

    def _finish(self):
        self._timer.stop()
        self.check_btn.setEnabled(True)
        self.check_btn.setText("Check")
        self.stop_btn.setVisible(False)
        worker, self._worker = self._worker, None
        if worker is not None and not worker.isRunning():
            worker.deleteLater()

    def _stop_waiting(self):
        worker = self._worker
        if worker is None:
            return
        _abandon(worker)
        worker.setParent(None)
        self._worker = None
        self._finish()
        self._log("Stopped waiting. What was already running finishes in "
                  "the background; nothing was changed.")

    def _on_succeeded(self, text):
        self._log("")
        self._log("OK - " + str(text))
        self._finish()

    def _on_failed(self, message: str, trace: str = ""):
        self._log("")
        self._log("FAILED - " + message)
        if trace:
            # the full traceback, for searching the error or reporting it
            self._log("")
            self._log(trace.rstrip())
        self._finish()


class CustomImplementationDialog(QDialog):
    """The box as a dialog - for the classification table, where the
    implementation is one of several model sources of a row."""

    def __init__(self, spec: dict, kind: str, camera: str = "",
                 task: str = "", model_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom implementation")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)

        self.box = CustomImplementationBox(
            f"Custom {task or kind} implementation", kind, camera, task,
            checkable=False)
        self.box.set_model_dir(model_dir)
        self.box.set_spec(dict(spec or {}, source=custom_models.SOURCE))
        self.box.enable_check.setChecked(True)
        self.box.body.setEnabled(True)
        layout.addWidget(self.box)

        note = QLabel(FOLDER_NOTE + " " + DEPENDENCY_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)

        standard = QDialogButtonBox.StandardButton
        buttons = QDialogButtonBox(standard.Ok | standard.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result(self) -> dict:
        """``path``, ``class`` and ``options`` as entered."""
        spec = self.box.spec()
        return {key: spec[key] for key in ("path", "class", "options")}


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def download_samples(parent) -> list:
    """Copy the sample implementations into a folder the user picks."""
    folder = QFileDialog.getExistingDirectory(
        parent, "Folder for the sample implementations")
    if not folder:
        return []
    written = custom_models.copy_samples(folder)
    names = custom_models.sample_files()
    skipped = len(names) - len(written)
    QMessageBox.information(
        parent, "Sample implementations",
        f"{len(written)} sample file(s) written to\n{folder}"
        + (f"\n\n{skipped} already existed there and were left as they are."
           if skipped else "")
        + "\n\nEach file explains the interface at its top. Copy one, "
          "adapt it and select it with \"Use a custom implementation\".")
    return written
